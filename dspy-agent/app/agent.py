"""
Agent 1 — Case Manager + Decision & Safety Logic

ลำดับต่อข้อความ
    Consent → Raw Message → Privacy Filter → Extraction (Gemini) → ground() → Merge Draft
    → Context completeness → Decision: ช่องทางฉุกเฉิน / คำแนะนำ KB / ให้กำลังใจ / ถาม Slot / สรุปให้ยืนยัน

โค้ดเป็นผู้ตัดสินทุกอย่างที่มีผลต่อการช่วยเหลือ: สร้าง/สลับเคส, Slot ถัดไป และ Version
Gemini ใช้ 2 จุด: Extractor (สกัด + จำแนกเจตนา) และ Response Generator (เรียบเรียงคำแนะนำ/ให้กำลังใจ)
Agent 1 ไม่ปิดเคสเอง ไม่ยืนยันแทนผู้ใช้ และไม่ส่ง Draft ให้ Pipeline
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Callable

from .case import (
    Case,
    all_slots_satisfied,
    append_raw,
    diff_snapshots,
    mark_asked,
    mask_phone,
    missing_notice,
    missing_slots,
    next_slot,
    refresh_slots,
    summary_text,
    version_snapshot,
)
from .extraction import (Extractor, ground, is_medical_or_distress, proposed_people, replaced_person,
                         explicit_count_list, is_people_addition)
from .knowledge import KBDoc, KnowledgeBase
from .priority import evaluate_priority, merge_facts
from .privacy import FilterResult, filter_text, normalize_text
from .response import ResponseGenerator, FORBIDDEN, ADVICE_WORDS
from .schemas import (
    AFFECTED_GROUPS,
    AFFECTED_LABELS_TH,
    AffectedPeople,
    MAX_SLOT_ASKS,
    PriorityLevel,
    SLOT_LABELS_TH,
    CaseIntent,
    CaseStatus,
    ConversationTurn,
    ContextAnswer,
    Extraction,
    IncidentData,
    IntakeStatus,
    Location,
    MessageIntent,
    QuickReply,
    SlotName,
    TurnOutput,
    now_bkk,
    requested_help_from_text,
    requested_help_phrase,
)
from .store import Store, parse_time

APP_DIR = Path(__file__).resolve().parent
DRAFT_REMINDER_AFTER = timedelta(minutes=30)
SUMMARY_MESSAGES = 3
# คำถามเติมบริบทของปัญหาถามได้ไม่เกินนี้ต่อเคส ครบแล้วพักเรื่องรายละเอียด (ไม่ถือว่าครบ) ไปเก็บข้อมูลอื่นต่อ
MAX_CONTEXT_ASKS = 3
# ตอบรับข้อความรอ ("โอเค" "เดี๋ยวนะ") ได้ไม่เกินนี้ต่อช่อง หลังจากนั้นถือว่าไม่ได้ตอบ จึงไม่วนตอบซ้ำ
FILLER_ACK_LIMIT = 3
NAME_SLOTS = (SlotName.FIRST_NAME, SlotName.LAST_NAME)
CONTACT_SLOTS = (*NAME_SLOTS, SlotName.PHONE_NUMBER, SlotName.LOCATION)
_NAME_PREFIX = re.compile(r"^(?:ผม|ดิฉัน|ฉัน|หนู|เรา)?\s*(?:ชื่อ(?:จริง)?|นามสกุล)?\s*(?:ว่า)?\s*(?:นาย|นางสาว|นาง|น\.ส\.)?\s*")
_POLITE = re.compile(r"\s*(?:ครับ|ค่ะ|คะ|ค่า|จ้า|นะ)+\s*$")
_NOT_A_NAME = ("น้ำ", "ท่วม", "ช่วย", "ไม่", "ทำไม", "อะไร", "ยังไง", "ไหม", "ได้", "แล้ว",
               "โอเค", "ตกลง", "เดี๋ยว", "แป๊บ", "แปป", "สักครู่", "ขอบคุณ", "สวัสดี", "ขอโทษ", "สะดวก", "รับทราบ")
# ข้อความรับทราบ/ขอเวลา ที่ไม่ใช่คำตอบของ Slot เช่น "โอเค" "เดี๋ยวนะ" "รอแป๊บ" "ขอเวลาแป๊บนึง"
_FILLER = re.compile(r"^(?:(?:โอเค|โอเช|ตกลง|ได้เลย|ได้|รอ|เดี๋ยว|แป๊บ|แปป|นึง|สักครู่|ขอเวลา|ขอบคุณ|ขอบใจ|สวัสดี|อืม|เออ"
                     r"|จ้า|จ๊ะ|ค่ะ|คะ|ครับ|คับ|ใช่|รับทราบ|เข้าใจ|ก่อน|นะ|น่า|แล้ว|เลย|ok|okay)\s*)+$", re.IGNORECASE)
_CONTEXT_ADMIN = re.compile(r"ผู้ประสบภัย|นามสกุล|ชื่อจริง|เบอร์โทร|หมุด|พิกัด|ทั้งหมดกี่คน|แบ่ง.*กลุ่ม|ระดับน้ำ|น้ำ.{0,30}(?:สูง|ลึก|ท่วมถึง|ถึงระดับ|แค่ไหน|เท่าไร|เท่าไหร่|กี่เมตร|กี่เซน|ถึงไหน)")

ExtractFn = Callable[[str, IncidentData, "str | None", str], Extraction]


def is_filler(text: str) -> bool:
    """ข้อความรับทราบ/ขอเวลา ไม่มีสาระให้บันทึก"""
    return bool(_FILLER.fullmatch(normalize_text(text).strip()))


def _question_key(question: str) -> str:
    return re.sub(r"ไม่ทราบว่า|ช่วยบอก|ช่วยเล่า|ตอนนี้|คุณพ่อ|คุณแม่|เพิ่มเติม|หน่อย|ได้ไหม|ครับ|ค่ะ|คะ|นะ|[\s?？]", "", question)


@dataclass
class Reply:
    parts: list[str] = field(default_factory=list)
    quick_replies: list[QuickReply] = field(default_factory=list)
    kb_doc_ids: list[str] = field(default_factory=list)

    def add(self, text: str | None):
        if text:
            self.parts.append(text)


class Agent1:
    def __init__(self, store: Store, knowledge: KnowledgeBase, *, extractor: ExtractFn | None = None,
                 generator: ResponseGenerator | None = None, templates_path: Path = APP_DIR / "templates_th.json"):
        self.store = store
        self.kb = knowledge
        self.extract = extractor or Extractor()
        self.generate = generator or ResponseGenerator()
        self.t = json.loads(Path(templates_path).read_text(encoding="utf-8"))
        self.labels = self.t["labels"]

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def handle(self, user_ref: str, *, text: str | None = None, location: tuple[float, float] | None = None,
               postback: str | None = None) -> TurnOutput:
        user = self.store.user(user_ref)
        if user["consent"] != "granted":
            return self._consent(user, text, postback)
        if text is not None and text.strip() in {self.labels["new_case"], "เหตุใหม่", "แจ้งใหม่", "ขอแจ้งเหตุใหม่"}:
            postback = "new_case"
        if postback is not None:
            return self._postback(user, postback)
        if location is not None:
            return self._location(user, Location(latitude=location[0], longitude=location[1]))
        return self._text(user, text or "")

    # ------------------------------------------------------------------
    # Consent (หัวข้อ 7)
    # ------------------------------------------------------------------

    def _consent(self, user, text, postback) -> TurnOutput:
        reply = Reply()
        if postback == "consent=yes":
            user["consent"], user["consent_at"] = "granted", now_bkk().isoformat()
            self.store.save_user(user)
            reply.add(self.t["consent_granted"])
        elif postback == "consent=no":
            user["consent"], user["consent_at"] = "declined", now_bkk().isoformat()
            self.store.save_user(user)
            channels = [d for n in ("1669", "1784") if (d := self.kb.hotline(n))]
            reply.add(self.t["consent_declined"] if channels else self.t["consent_declined_no_kb"])
            self._add_channels(reply, channels)
        else:
            reply.add(self.t["privacy_notice"])
            reply.quick_replies = [QuickReply(label=self.labels["consent_yes"], data="consent=yes"),
                                   QuickReply(label=self.labels["consent_no"], data="consent=no")]
        return self._output(reply)

    # ------------------------------------------------------------------
    # Free text
    # ------------------------------------------------------------------

    def _text(self, user, text: str, *, message_id: str | None = None, reply: Reply | None = None,
              selected: bool = False) -> TurnOutput:
        reply = reply or Reply()
        case = self._current_case(user)
        f = filter_text(text)
        if message_id is None:
            message_id = self.store.add_raw(user["user_ref"], case.case_id if case else None, "in", "text", text)
        user["pending_question"] = None

        if case and case.pending_question == "reuse_contact":
            answer = re.sub(r"\s+", "", f.text)
            if re.fullmatch(r"(?:ใช่|ใช้เดิม|เหมือนเดิม|ข้อมูลเดิม)(?:ครับ|ค่ะ|คะ)?", answer):
                self._reuse_contact(case, True, reply)
                return self._finish(user, case, reply)
            if re.fullmatch(r"(?:ไม่ใช่|ไม่ใช้|ไม่|ข้อมูลใหม่)(?:ครับ|ค่ะ|คะ)?", answer):
                self._reuse_contact(case, False, reply)
                return self._finish(user, case, reply)

        if case and case.pending_question == SlotName.PHONE_NUMBER.value:
            attempt = normalize_text(text).strip()
            if not f.phones and re.fullmatch(r"(?:เบอร์(?:โทรศัพท์|โทร)?|โทรศัพท์|โทร)?\s*[\d +().-]+\s*(?:ครับ|ค่ะ|คะ)?", attempt):
                case.last_user_at = now_bkk()
                reply.add(self.t["invalid_phone"])
                return self._finish(user, case, reply)

        # "โอเค" "เดี๋ยวนะ" ระหว่างถามข้อมูลติดต่อ: ไม่ใช่คำตอบ ไม่ส่ง LLM ไม่นับเป็นการถามซ้ำ และไม่เก็บเป็นชื่อ
        pending_slot = next((s for s in CONTACT_SLOTS if case and case.pending_question == s.value), None)
        if (pending_slot is not None and is_filler(text) and not f.phones
                and case.slots[pending_slot].filler_acks < FILLER_ACK_LIMIT):
            case.last_user_at = now_bkk()
            case.slots[pending_slot].filler_acks += 1
            reply.add(self.t["filler_ack"].format(label=SLOT_LABELS_TH[pending_slot]))
            if pending_slot == SlotName.LOCATION:
                reply.quick_replies = [QuickReply(type="location", label=self.labels["share_location"])]
            return self._finish(user, case, reply)

        subject_reply = re.fullmatch(r"(?:คุณ)?(พ่อ|แม่|ตา|ยาย|ปู่|ย่า|ลูก)(?:ครับ|ค่ะ|คะ)?", f.text.strip())
        if (case and case.context_subject and subject_reply
                and case.pending_question not in {"affected_people", "affected_people_confirmation"}):
            old, new = case.context_subject, subject_reply[1]
            case.context_subject = new
            if case.problem_display:
                case.problem_display = case.problem_display.replace(old, new)
            if case.context_question:
                case.context_question = re.sub(r"(?:คุณ)?(?:พ่อ|แม่|ตา|ยาย|ปู่|ย่า|ลูก)(?=มีอาการ|ได้รับ|เจ็บ|ล้ม|ปวด|เป็น)",
                                               "คุณ" + new, case.context_question)
            case.last_user_at = now_bkk()
            if old != new:
                case.incident.problem = append_raw(case.incident.problem, f.text)
                self._data_changed(case)
            reply.add(self.t["subject_received"].format(person=new))
            self._continue(case, reply)
            return self._finish(user, case, reply)

        if case and case.pending_people_confirmation and case.pending_question == "affected_people_confirmation":
            answer = re.sub(r"\s+", "", f.text)
            if re.fullmatch(r"(?:ใช่|ถูกต้อง|ถูก|ยืนยัน)(?:ครับ|ค่ะ|คะ)?", answer):
                self._resolve_people(case, True, reply)
                return self._finish(user, case, reply)
            if re.fullmatch(r"(?:ไม่ใช่|ไม่ถูก|ไม่)(?:ครับ|ค่ะ|คะ)?", answer):
                self._resolve_people(case, False, reply)
                return self._finish(user, case, reply)

        if case is not None and not is_medical_or_distress(f.text) and self._direct_answer(case, f, text, reply):
            return self._finish(user, case, reply)

        if explicit_count_list(f.text) is not None or proposed_people(f.text) is not None:
            ex = ground(Extraction(intents=[MessageIntent.INCIDENT]), f.text)
        else:
            ex = ground(self.extract(f.text, case.display_incident() if case else IncidentData(),
                                     case.pending_question if case else None,
                                     self._conversation_context(case) if case else ""), f.text,
                        risk_question=case.context_question if case and (case.pending_question == "problem" or (case.pending_question or "").startswith("assessment:")) else None)

        # Multi-Case: เจตนาและการยืนยันเป็นตัวแยกเคส ไม่ใช้ระยะทาง
        actives = self.store.active_cases(user["user_ref"])
        has_incident = bool(ex.intents or ex.problem or ex.affected_people.has_anyone())
        if ex.case_intent == CaseIntent.UNCLEAR and len(actives) > 1 and not selected:
            return self._ask_case_selection(user, actives, message_id, reply)
        if case is None and len(actives) > 1 and has_incident and not selected:
            return self._ask_case_selection(user, actives, message_id, reply)
        if case is None and len(actives) == 1:
            case = actives[0]
        if ex.case_intent == CaseIntent.NEW_CASE and case is not None and not selected and (case.incident.problem or case.current_version):
            case = self._open_case(user)
            ex = ground(self.extract(f.text, case.incident, None, ""), f.text)
        elif case is None and has_incident:
            case = self._open_case(user)
        if case is None:
            # ถามหรือระบายความรู้สึกโดยยังไม่แจ้งเหตุ: ตอบได้โดยไม่สร้างเคส
            self._support(reply, None, ex, f)
            reply.add(self.t["greeting"] if not reply.parts else None)
            return self._finish(user, None, reply)

        self.store.attach_raw(message_id, case.case_id)
        if ex.case_intent == CaseIntent.RELOCATE:
            case.expect_relocation = True
        self._merge(case, ex, f)
        if re.search(r"(?:อยาก|ขอ|แค่).*ปรึกษา|ปรึกษาก่อน", f.text):
            case.consultation_active = True
        if MessageIntent.QUESTION in ex.intents and re.search(r"แผล|ไข้|ป่วย|เจ็บ|ปวด|บวม|ยา", f.text):
            case.consultation_active = True
        if re.search(r"ขอแจ้งเหตุ|ต้องการแจ้งเหตุ|กรอกข้อมูลส่งต่อ|บันทึกเคส|ขอเรือ|ต้องการเรือ", f.text):
            case.consultation_active = False
        acknowledgement = ex.correction_acknowledgement if re.search(r"แก้|เปลี่ยนจาก|ไม่ใช่", f.text) else None
        if (acknowledgement and filter_text(acknowledgement).text == acknowledgement
                and not FORBIDDEN.search(acknowledgement) and not ADVICE_WORDS.search(acknowledgement)):
            reply.add(acknowledgement)
        self._notify_emergency(case, reply)
        fallback_only = self._support(reply, case, ex, f)
        if not self._is_critical(case) and case.help_status != "missing" and ((fallback_only and case.consultation_active)
                or (not case.context_complete and not case.context_question)):
            self._clarify_context(case, f.text)
        if ex.case_intent == CaseIntent.RELOCATE and not ex.problem:
            reply.add(self.t["relocation_ask_pin"])
            reply.quick_replies = [QuickReply(type="location", label=self.labels["share_location"])]
            return self._finish(user, case, reply)
        interjected = bool({MessageIntent.QUESTION, MessageIntent.EMOTIONAL} & set(ex.intents)) and bool(reply.parts)
        if MessageIntent.QUESTION in ex.intents and case.context_complete and reply.parts and not self._is_critical(case):
            # Answer the consultation first; a question is not a request to
            # interrupt with contact intake or consume another slot attempt.
            if case.consultation_active:
                self._continue(case, reply)
            else:
                case.pending_question = "consultation"
        else:
            self._continue(case, reply, resumed=interjected, count_unclear=ex.count_unclear)
        return self._finish(user, case, reply)

    def _direct_answer(self, case: Case, f: FilterResult, text: str, reply: Reply) -> bool:
        """คำตอบของคำถาม PII บันทึกตรงลง Personal Data โดยไม่ผ่าน LLM"""
        pending = case.pending_question
        if case.intake_status == IntakeStatus.AWAITING_USER_INPUT and pending is None:
            missing = [s for s in missing_slots(case) if s in NAME_SLOTS]
            # The user may supply a full name after several fields exhausted
            # their ask budget. Other missing fields must not block this answer.
            parts = self._name_parts(text)
            if missing and parts and (len(parts) == 2 or len(missing) == 1):
                pending = missing[0].value
        if pending == SlotName.PHONE_NUMBER.value and f.phones:
            case.personal.phone_number = f.phones[0]
        elif pending in (SlotName.FIRST_NAME.value, SlotName.LAST_NAME.value) and (parts := self._name_parts(text)):
            if pending == SlotName.FIRST_NAME.value:
                case.personal.first_name = parts[0]
                if len(parts) == 2:
                    case.personal.last_name = parts[1]
            else:
                if len(parts) == 2:
                    case.personal.first_name = parts[0]
                case.personal.last_name = parts[-1]
        else:
            return False
        self._data_changed(case)
        case.last_user_at = now_bkk()
        reply.add(self.t["contact_received"])
        self._continue(case, reply)
        return True

    @staticmethod
    def _name_parts(text: str) -> list[str] | None:
        if is_filler(text):
            return None
        t = _POLITE.sub("", _NAME_PREFIX.sub("", normalize_text(text).strip()))
        parts = t.split()
        if is_medical_or_distress(t) or is_filler(t):
            return None
        if not 1 <= len(parts) <= 2 or "?" in t or any(w in t for w in _NOT_A_NAME):
            return None
        if not all(re.fullmatch(r"[ก-๙A-Za-z.\-]{2,30}", p) for p in parts):
            return None
        return parts

    # ------------------------------------------------------------------
    # Merge Draft
    # ------------------------------------------------------------------

    def _merge(self, case: Case, ex: Extraction, f: FilterResult) -> None:
        risk_turn = (case.pending_question or "").startswith("assessment:")
        risk_question = case.context_question if case.pending_question == "problem" or risk_turn else None
        previous_complete = case.context_complete
        prior_problem = case.problem_text
        mentioned = re.search(r"(?:คุณ)?(พ่อ|แม่|ตา|ยาย|ปู่|ย่า|ลูก)(?=ล้ม|มีแผล|มีไข้|เจ็บ|ป่วย|ไม่สบาย|บวม)", f.text)
        count_only = proposed_people(f.text) is not None or explicit_count_list(f.text) is not None
        if (case.pending_question == "problem" and case.context_question and not count_only
                and not replaced_person(f.text) and MessageIntent.QUESTION not in ex.intents
                and f.text.strip() not in {"ต่อ", "แล้วไง", "แล้ว"}):
            case.conversation_context.answers.append(ContextAnswer(question=case.context_question[:240], answer=f.text[:600],
                subject=mentioned[1] if mentioned else case.context_subject))
            case.conversation_context.answers = case.conversation_context.answers[-12:]
        before = (case.incident.model_dump(), case.personal.model_dump(), case.context_complete,
                  case.pending_people_confirmation, case.context_subject, case.help_status,
                  case.priority_facts.model_dump(), dict(case.risk_fact_evidence))
        p = case.personal
        if f.phones and (not p.phone_number or case.pending_question == SlotName.PHONE_NUMBER.value):
            p.phone_number = f.phones[0]
        if f.first_names and not p.first_name:
            p.first_name = f.first_names[0]
        if f.last_names and not p.last_name:
            p.last_name = f.last_names[0]
        case.address_masked = case.address_masked or f.address_masked

        # DB เก็บข้อความต้นฉบับ (evidence) ส่วนฉบับเรียบเรียงจาก LLM ใช้แสดงผล
        if ex.problem:
            case.incident.problem = append_raw(case.incident.problem, ex.problem_evidence)
            case.problem_display = ex.problem
        answering_help = case.pending_question == "requested_help"
        help_status = requested_help_from_text(f.text, answering=answering_help)
        help_evidence = f.text
        if ex.help_evidence:
            evidenced_status = requested_help_from_text(ex.help_evidence, answering=answering_help)
            if evidenced_status != "missing" and evidenced_status == ex.help_status:
                help_status, help_evidence = evidenced_status, ex.help_evidence
        if help_status != "missing":
            case.help_status = help_status
            case.help_evidence = requested_help_phrase(help_evidence) if help_status == "known" else help_evidence
            case.incident.problem = append_raw(case.incident.problem, f.text)
            if not ex.problem:
                case.problem_display = append_raw(prior_problem, "ความช่วยเหลือที่ต้องการ: " + f.text)
        if ex.problem or ex.context_question or ex.context_missing:
            case.context_complete = bool(ex.context_complete and case.incident.problem
                                         and not ex.context_question and not ex.context_missing
                                         and case.help_status != "missing")
            question = ex.context_question or ""
            # If the model emits two complete questions, keep only the first.
            boundary = re.search(r"(คะ|ครับ|ค่ะ|[?？])\s*(?:และ|แล้ว|อีกทั้ง)", question)
            if boundary:
                question = question[:boundary.start() + len(boundary[1])]
            # Clarification is a request for information, never uncited treatment advice.
            filtered_question = filter_text(question).text
            if (question and filtered_question == question and not FORBIDDEN.search(question)
                    and not ADVICE_WORDS.search(question) and not _CONTEXT_ADMIN.search(question)):
                case.context_question = question
            else:
                case.context_question = None
            if case.context_complete:
                case.context_question = None
        if mentioned:
            case.context_subject = mentioned[1]
        replacement = replaced_person(f.text)
        if replacement:
            old, new = replacement
            case.context_subject = new
            for answer in case.conversation_context.answers:
                if answer.subject == old:
                    answer.subject = new
                    answer.question = answer.question.replace(old, new)
            # Correct the current rendering, while raw messages/evidence retain
            # the user's original words and explicit correction for audit.
            if case.problem_display and old not in case.problem_display and new in case.problem_display:
                pass  # The extractor already produced the corrected current rendering.
            elif prior_problem:
                case.problem_display = prior_problem.replace(old, new)
            case.context_complete = False
            if not ex.context_question:
                case.context_question = None
            case.pending_people_confirmation = None
            case.pending_people_label = None
        proposal = proposed_people(f.text)
        if proposal:
            if is_people_addition(f.text):
                proposal = {group: (getattr(case.incident.affected_people, group) or 0) + proposal.get(group, 0)
                            for group in AFFECTED_GROUPS
                            if getattr(case.incident.affected_people, group) is not None or group in proposal}
            case.pending_people_confirmation = AffectedPeople(**proposal)
            case.pending_people_label = f.text[:80]
        elif ex.affected_people.has_anyone():
            case.pending_people_confirmation = None
            case.pending_people_label = None
        for group in AFFECTED_GROUPS:
            if (value := getattr(ex.affected_people, group)) is not None:
                previous = getattr(case.incident.affected_people, group) or 0
                setattr(case.incident.affected_people, group, previous + value if is_people_addition(f.text) else value)
        if case.context_question and self._question_answered(case, case.context_question):
            case.context_question = None
            case.context_complete = False
        case.last_user_at = now_bkk()
        case.last_intents = [i.value for i in ex.intents]
        recent = [*filter(None, case.conversation_summary.split("\n")), f.text[:200]]
        case.conversation_summary = "\n".join(recent[-SUMMARY_MESSAGES:])

        # Priority รายเคส (ข้อ 7.1): สะสมข้อเท็จจริงแล้วประเมินด้วยโค้ดทุกครั้งที่ข้อความเปลี่ยน draft
        case.priority_facts, case.risk_fact_evidence = merge_facts(
            case.priority_facts, case.risk_fact_evidence, ex, f.text, risk_question)
        self._refresh_priority(case)
        if risk_turn:
            case.context_complete = previous_complete

        ctx = case.conversation_context
        ctx.turns.append(ConversationTurn(role="user", text=f.text[:600]))
        ctx.turns = ctx.turns[-8:]
        after = (case.incident.model_dump(), case.personal.model_dump(), case.context_complete,
                 case.pending_people_confirmation, case.context_subject, case.help_status,
                 case.priority_facts.model_dump(), dict(case.risk_fact_evidence))
        if after != before:
            self._data_changed(case)

    def _is_critical(self, case: Case) -> bool:
        return case.priority.level == PriorityLevel.CRITICAL

    def _notify_emergency(self, case: Case, reply: Reply) -> None:
        if not self._is_critical(case) or case.emergency_notified:
            return
        case.emergency_notified = True
        channels = [d for n in ("1669", "1784") if (d := self.kb.hotline(n))]
        reply.add(self.t["priority_emergency"] if channels else self.t["no_kb_info"])
        self._add_channels(reply, channels)

    def _refresh_priority(self, case: Case) -> None:
        previous = case.priority
        current = evaluate_priority(case.priority_facts, case.risk_fact_evidence)
        case.priority = current
        if current.level != PriorityLevel.CRITICAL:
            case.emergency_notified = False
        self.store.add_event(case.case_id, "priority_evaluated", {
            "previous_level": previous.level.value if previous.level else None,
            "level": current.level.value if current.level else None,
            "reason": current.reason, "rule_id": current.rule_id,
            "ruleset_version": current.ruleset_version,
            "evaluated_at": current.evaluated_at.isoformat()})

    def _data_changed(self, case: Case) -> None:
        """Draft เปลี่ยน: เปิดร่างการแก้ไขถ้าเคสยืนยันแล้ว"""
        if case.intake_status == IntakeStatus.CONFIRMED and not case.editing:
            case.editing = True
            case.draft_round += 1
            case.reminder_sent = False
            case.missing_notice_sent = False
            self.store.add_event(case.case_id, "revision_draft_opened", {"base_version": case.current_version,
                                                                         "draft_round": case.draft_round})
        if case.intake_status in (IntakeStatus.CONFIRMED, IntakeStatus.AWAITING_USER_INPUT,
                                  IntakeStatus.AWAITING_CONFIRMATION):
            case.intake_status = IntakeStatus.COLLECTING
        case.touch(changed=True)
        refresh_slots(case)
    # Knowledge-grounded support without severity classification.
    def _support(self, reply: Reply, case: Case | None, ex: Extraction, f: FilterResult) -> bool:
        # A person correction supplies no new consultation question. Keep the
        # acknowledgement and context flow without regenerating medical advice.
        if replaced_person(f.text) and MessageIntent.QUESTION not in ex.intents and not (case and self._is_critical(case)):
            return False
        # Extractor resolves short questions using history. Do not prepend the
        # accumulated problem: it may contain unrelated previous topics.
        count_only = proposed_people(f.text) is not None or explicit_count_list(f.text) is not None
        consulting = case is not None and case.consultation_active and not count_only
        query = ex.question or (case.problem_text if consulting or (case and self._is_critical(case)) else None) or f.text
        # Two candidates: the generator picks the one matching the latest question and the
        # validator drops any sentence whose content is not in the cited document.
        docs = [h.doc for h in self.kb.search(query, limit=2)] if MessageIntent.QUESTION in ex.intents or consulting or (case is not None and self._is_critical(case)) else []
        allowed = {str(getattr(case.incident.affected_people, g)) for g in AFFECTED_GROUPS} if case else set()
        composed = self.generate(f.text, need_empathy=MessageIntent.EMOTIONAL in ex.intents,
                                 docs=docs, allowed_numbers=allowed,
                                 empathy_template=self._empathy_template(f.text),
                                 conversation_context=self._conversation_context(case) if case else "")
        if not case or composed.text != case.conversation_context.last_support:
            reply.add(composed.text)
        if case:
            case.conversation_context.last_support = composed.text
        reply.kb_doc_ids += composed.kb_doc_ids
        if (MessageIntent.QUESTION in ex.intents or consulting) and not docs:
            reply.add(self.t["no_kb_info"])
        return bool(composed.used_template and not composed.kb_doc_ids)

    def _empathy_template(self, text: str) -> str:
        for word, message in self.t["empathy_by_feeling"].items():
            if word in text:
                return message
        return self.t["empathy"]

    def _conversation_context(self, case: Case) -> str:
        # Re-filter on read as well, including names already held in the contact table.
        text = json.dumps({"current_subject": case.context_subject, "current_problem": case.problem_text,
                           "risk_facts": case.priority_facts.model_dump(),
                           "requested_help": {"status": case.help_status, "evidence": case.help_evidence},
                           "pending_question": case.pending_question,
                           "answers": [answer.model_dump() for answer in case.conversation_context.answers
                                       if answer.subject == case.context_subject],
                           "asked_questions": case.conversation_context.asked_questions,
                           "recent_turns": [turn.model_dump() for turn in case.conversation_context.turns],
                           "history_summary": case.conversation_summary}, ensure_ascii=False)
        for value in (case.personal.first_name, case.personal.last_name, case.personal.phone_number):
            if value:
                text = text.replace(value, "[PERSONAL]")
        return filter_text(text).text

    @staticmethod
    def _question_answered(case: Case, question: str) -> bool:
        def key(value):
            value = re.sub(r"ไม่ทราบว่า|ตอนนี้|คุณพ่อ|คุณแม่|แผล|ครับ|ค่ะ|คะ|[\s?？]", "", value)
            return value
        current = key(question)
        return any(len(current) >= 8 and (current in key(item.question) or key(item.question) in current)
                   for item in case.conversation_context.answers
                   if item.subject == case.context_subject and len(key(item.question)) >= 8)

    def _asked_before(self, case: Case, question: str) -> bool:
        key = _question_key(question)
        for asked in case.conversation_context.asked_questions:
            asked_key = _question_key(asked)
            if key == asked_key or (len(key) >= 8 and len(asked_key) >= 8 and (key in asked_key or asked_key in key)):
                return True
        return False

    def _next_context_question(self, case: Case) -> str | None:
        """คำถามเติมบริบทที่ยังไม่เคยถาม: ของโมเดลก่อน แล้วค่อยใช้ template สำรองที่ต่างจากเดิม
        None = ครบเพดานหรือไม่มีคำถามใหม่แล้ว (พักเรื่องรายละเอียด ไม่ถือว่าครบ)"""
        if case.slots[SlotName.PROBLEM].ask_count >= MAX_CONTEXT_ASKS:
            return None
        for question in (case.context_question, self.t["ask"]["problem"], *self.t["ask"]["problem_alt"]):
            if question and not self._asked_before(case, question) and not self._question_answered(case, question):
                return question
        return None

    def _clarify_context(self, case: Case, text: str) -> None:
        """One extra model attempt, using remembered answers; never retry forever."""
        case.context_complete = False
        if case.slots[SlotName.PROBLEM].ask_count >= MAX_CONTEXT_ASKS:
            case.context_question = None
            return
        try:
            retry_context = json.loads(self._conversation_context(case))
            retry_context["clarification_task"] = ("คำตอบก่อนหน้ายังไม่พอ ให้ถามรายละเอียดที่ยังไม่รู้หนึ่งข้อ "
                                                   "ห้ามถามข้อที่มีใน answers หรือ asked_questions ซ้ำ ให้เปลี่ยนมุมคำถาม")
            followup = self.extract(text, case.display_incident(), "clarify_unanswered_detail",
                                    json.dumps(retry_context, ensure_ascii=False))
            question = followup.context_question or ""
            boundary = re.search(r"(คะ|ครับ|ค่ะ|[?？])\s*(?:และ|แล้ว|อีกทั้ง)", question)
            if boundary:
                question = question[:boundary.start() + len(boundary[1])]
            if (question and filter_text(question).text == question and not FORBIDDEN.search(question)
                    and not ADVICE_WORDS.search(question) and not _CONTEXT_ADMIN.search(question)
                    and not self._question_answered(case, question) and not self._asked_before(case, question)):
                case.context_question = question
            else:
                case.context_question = None
        except Exception:
            case.context_question = None

    def _add_channels(self, reply: Reply, docs: list[KBDoc]) -> None:
        for d in docs:
            reply.add(self.t["channel_line"].format(text=d.text))
            reply.kb_doc_ids.append(d.kb_doc_id)

    # ------------------------------------------------------------------
    # Slot Filling / Confirmation
    # ------------------------------------------------------------------

    def _ask_people_confirmation(self, case: Case, reply: Reply) -> None:
        people = case.pending_people_confirmation
        groups = [f"{'ผู้ใหญ่' if group == 'other' else AFFECTED_LABELS_TH[group]} {getattr(people, group)} คน"
                  for group in AFFECTED_GROUPS if getattr(people, group) is not None]
        total = sum(getattr(people, group) or 0 for group in AFFECTED_GROUPS)
        reply.add(self.t["confirm_people"].format(label=case.pending_people_label, groups=", ".join(groups), total=total))
        case.pending_question = "affected_people_confirmation"
        case.intake_status = IntakeStatus.COLLECTING
        reply.quick_replies = [QuickReply(label=self.labels["yes"], data=f"people_confirm=yes:{case.case_id}:{case.draft_rev}"),
                               QuickReply(label=self.labels["no"], data=f"people_confirm=no:{case.case_id}:{case.draft_rev}")]

    def _resolve_people(self, case: Case, accepted: bool, reply: Reply) -> None:
        if accepted:
            case.incident.affected_people = case.pending_people_confirmation
            case.slots[SlotName.AFFECTED_PEOPLE].evidence = "user_confirmed"
        case.pending_people_confirmation = None
        case.pending_people_label = None
        case.pending_question = None
        self._data_changed(case)
        if not accepted:
            reply.add(self.t["ask"]["affected_people_unclear"])
            mark_asked(case, SlotName.AFFECTED_PEOPLE)
        else:
            reply.add(self.t["people_confirmed"])
            self._continue(case, reply)

    def _continue(self, case: Case, reply: Reply, *, resumed: bool = False, count_unclear: bool = False) -> None:
        if case.case_status == CaseStatus.INACTIVE:
            reply.add(self.t["case_closed"])
            return
        if case.intake_status == IntakeStatus.CONFIRMED:
            reply.quick_replies = [QuickReply(label=self.labels["new_case"], data="new_case")]
            return
        self._notify_emergency(case, reply)
        if self._is_critical(case):
            case.consultation_active = False
            if case.personal.location is None and case.slots[SlotName.LOCATION].ask_count < MAX_SLOT_ASKS:
                mark_asked(case, SlotName.LOCATION)
                case.intake_status = IntakeStatus.COLLECTING
                reply.add(self.t["ask"]["location"])
                reply.quick_replies = [QuickReply(type="location", label=self.labels["share_location"])]
                return
        if case.incident.problem and case.help_status == "missing":
            case.pending_question = "requested_help"
            case.intake_status = IntakeStatus.COLLECTING
            reply.add(self.t["ask_requested_help"])
            return
        # Complete the meaning of problem before asking contact/location slots,
        # with a cap and never the same question twice. When out of questions,
        # pause the detail (not marked complete) and collect the other slots.
        if not case.context_complete:
            question = self._next_context_question(case)
            if question is not None:
                case.pending_question = SlotName.PROBLEM.value
                case.intake_status = IntakeStatus.COLLECTING
                reply.add(question)
                case.slots[SlotName.PROBLEM].ask_count += 1
                case.slots[SlotName.PROBLEM].asked_at = now_bkk()
                asked = case.conversation_context.asked_questions
                asked.append(question[:240])
                case.conversation_context.asked_questions = asked[-8:]
                return
            case.context_question = None
            if not case.context_paused:
                case.context_paused = True
                reply.add(self.t["context_paused"])
        if case.pending_people_confirmation is not None:
            self._ask_people_confirmation(case, reply)
            return
        if case.consultation_active:
            case.pending_question = "consultation"
            if not reply.parts:
                reply.add(self.t["support_ack"])
            reply.quick_replies = [QuickReply(label="กรอกข้อมูลส่งต่อ", data=f"intake={case.case_id}")]
            return
        if all_slots_satisfied(case):
            if case.priority.rule_id == "P1":
                self._ask_assessment(case, reply)
                return
            if case.intake_status == IntakeStatus.AWAITING_CONFIRMATION:
                reply.add(self.t["awaiting_confirmation_reminder"])
            else:
                case.intake_status = IntakeStatus.AWAITING_CONFIRMATION
                reply.add(summary_text(case))
            case.pending_question = None
            reply.quick_replies = self._confirm_buttons(case)
            return
        if self._offer_reuse_contact(case, reply):
            return
        slot = (SlotName.AFFECTED_PEOPLE if count_unclear and not case.slots[SlotName.AFFECTED_PEOPLE].satisfied
                else next_slot(case))
        if slot == SlotName.PROBLEM:
            # Detail questions are handled above; a paused problem is reported in the missing notice instead.
            slot = next((s for s in missing_slots(case) if s != SlotName.PROBLEM and case.slots[s].ask_count < MAX_SLOT_ASKS), None)
        if slot is not None:
            question = self.t["ask"]["affected_people_unclear" if count_unclear and slot == SlotName.AFFECTED_PEOPLE else slot.value]
            if resumed and case.pending_question == slot.value:
                question = self.t["resume_prefix"].format(label=SLOT_LABELS_TH[slot]) + question
            mark_asked(case, slot)
            case.intake_status = IntakeStatus.COLLECTING
            reply.add(question)
            if slot == SlotName.LOCATION:
                reply.quick_replies = [QuickReply(type="location", label=self.labels["share_location"])]
            return
        # ถามครบ 2 ครั้งทุก Slot ที่ขาดแล้ว: แจ้งครั้งเดียว ไม่นับเป็นคำถามครั้งที่ 3
        case.pending_question = None
        case.intake_status = IntakeStatus.AWAITING_USER_INPUT
        if not case.missing_notice_sent:
            case.missing_notice_sent = True
            missing = missing_slots(case)
            reply.add(missing_notice(missing))
            if SlotName.LOCATION in missing:
                reply.quick_replies = [QuickReply(type="location", label=self.labels["share_location"])]
        elif not reply.parts:
            reply.add(self.t["missing_idle"])  # ไม่ถามซ้ำ ตอบรับสั้น ๆ เพื่อให้ LINE มีข้อความตอบกลับเสมอ

    def _ask_assessment(self, case: Case, reply: Reply) -> None:
        """P1 is pending, never a final medium assessment or an exportable case."""
        questions = [
            ("in_safe_place", "ตอนนี้ทุกคนอยู่ในจุดที่น้ำไม่ถึงและปลอดภัยไหมคะ?"),
            ("needs_answer", "ตอนนี้ต้องการความช่วยเหลือเพิ่มเติมอะไรบ้างคะ? ถ้าไม่ต้องการ บอกได้เลยค่ะ"),
            ("trapped", "ตอนนี้มีคนติดอยู่ในที่พักหรือออกจากที่พักไม่ได้ไหมคะ?"),
            ("basic_needs_unmet", "ตอนนี้ขาดอาหาร น้ำดื่ม ยาประจำตัว หรือสิ่งจำเป็นอะไรบ้างคะ?"),
            ("cannot_self_evacuate", "มีใครเดินไม่ได้หรือต้องให้คนช่วยพาอพยพไหมคะ?"),
            ("life_threat", "ตอนนี้มีใครมีอาการฉุกเฉิน เช่น หมดสติ หายใจไม่ออก หรือบาดเจ็บสาหัสไหมคะ?"),
        ]
        if case.priority_facts.in_safe_place is False:
            questions.append(("water_rising", "ตอนนี้น้ำกำลังเพิ่มสูงขึ้นไหมคะ?"))
        if re.search(r"ยา|อุปกรณ์การแพทย์|ออกซิเจน", case.problem_text or ""):
            questions.append(("essential_medication_risk", "ยาหรืออุปกรณ์ที่ขาดนั้นขาดไม่ได้หรือทำให้เกิดอันตรายร้ายแรงไหมคะ?"))
        for name, question in questions:
            known = name in case.risk_fact_evidence if name == "needs_answer" else getattr(case.priority_facts, name) is not None
            if known or question in case.conversation_context.asked_questions:
                continue
            case.pending_question = "assessment:" + name
            case.context_question = question
            case.conversation_context.asked_questions.append(question)
            case.intake_status = IntakeStatus.COLLECTING
            reply.add(question)
            return
        case.pending_question = None
        case.intake_status = IntakeStatus.AWAITING_USER_INPUT
        reply.add("ข้อมูลที่มีอยู่ยังไม่เข้าเกณฑ์ประเมินความเร่งด่วนค่ะ เคสยังไม่ถูกส่งต่อ หากทราบสถานการณ์หรือความต้องการเพิ่ม พิมพ์บอกได้เลยนะคะ")

    def _confirm_buttons(self, case: Case) -> list[QuickReply]:
        ref = f"{case.case_id}:{case.draft_rev}"
        return [QuickReply(label=self.labels["confirm"], data=f"confirm={ref}"),
                QuickReply(label=self.labels["edit"], data=f"edit={ref}")]

    def _confirm(self, case: Case, reply: Reply) -> None:
        self._refresh_priority(case)
        if case.priority.rule_id == "P1":
            self._ask_assessment(case, reply)
            return
        previous = self.store.latest_version(case.case_id)
        case.current_version += 1
        case.confirmed_at = now_bkk()
        case.intake_status = IntakeStatus.CONFIRMED
        case.editing = False
        case.pending_question = None
        snapshot = version_snapshot(case)
        self.store.add_version(case, snapshot)
        if previous is None:
            self.store.add_event(case.case_id, "incident_submitted", {"current_version": case.current_version})
        else:
            self.store.add_event(case.case_id, "case_updated", {
                "previous_version": previous["version"], "current_version": case.current_version,
                "changes": diff_snapshots(previous, snapshot), "confirmed_by": "reporter"})
        reply.add(self.t["confirmed"].format(case_id=case.case_id))
        reply.quick_replies = [QuickReply(label=self.labels["new_case"], data="new_case")]

    # ------------------------------------------------------------------
    # Location (LINE Location Message)
    # ------------------------------------------------------------------

    def _location(self, user, loc: Location) -> TurnOutput:
        reply = Reply()
        case = self._current_case(user) or self._open_case(user)
        self.store.add_raw(user["user_ref"], case.case_id, "in", "location", f"{loc.latitude},{loc.longitude}")
        case.last_user_at = now_bkk()
        if case.personal.location is None:
            case.personal.location = loc
            self._data_changed(case)
            reply.add(self.t["location_received"])
        elif case.expect_relocation:
            self._relocate(case, loc, reply)
        else:
            # หมุดใหม่โดยไม่มีบริบท: ถามก่อน ไม่เขียนทับพิกัดเดิม
            case.pending_location = loc
            case.pending_question = "location_choice"
            reply.add(self.t["location_choice"])
            reply.quick_replies = [QuickReply(label=self.labels["relocate"], data="location_choice=relocate"),
                                   QuickReply(label=self.labels["new_case"], data="location_choice=new")]
            return self._finish(user, case, reply)
        self._continue(case, reply)
        return self._finish(user, case, reply)

    def _relocate(self, case: Case, loc: Location, reply: Reply) -> None:
        case.personal.location = loc
        case.pending_location = None
        case.expect_relocation = False
        self._data_changed(case)
        reply.add(self.t["relocated"].format(case_id=case.case_id))

    # ------------------------------------------------------------------
    # Postback (Quick Reply)
    # ------------------------------------------------------------------

    def _postback(self, user, data: str) -> TurnOutput:
        reply = Reply()
        key, _, value = data.partition("=")
        case = self._current_case(user)
        if key == "intake":
            if case is None or case.case_id != value:
                reply.add(self.t["stale_button"])
            else:
                case.consultation_active = False
                self._continue(case, reply)
            return self._finish(user, case, reply)
        if key == "people_confirm":
            parts = value.split(":")
            if (len(parts) != 3 or parts[0] not in {"yes", "no"} or case is None
                    or case.case_id != parts[1] or str(case.draft_rev) != parts[2]
                    or case.pending_question != "affected_people_confirmation"
                    or case.pending_people_confirmation is None):
                reply.add(self.t["stale_button"])
            else:
                self._resolve_people(case, parts[0] == "yes", reply)
            return self._finish(user, case, reply)
        if key == "new_case":
            case = self._open_case(user)
            if self._offer_reuse_contact(case, reply):
                reply.parts.insert(0, self.t["new_case_created"].format(case_id=case.case_id))
            else:
                reply.add(self.t["new_case_started"].format(case_id=case.case_id))
            return self._finish(user, case, reply)
        if key == "case_select":
            selected = self.store.load_case(value)
            if selected is None or selected.user_ref != user["user_ref"] or selected.case_status != CaseStatus.ACTIVE:
                reply.add(self.t["stale_button"])
                return self._finish(user, case, reply)
            user["current_case_id"] = selected.case_id
            held, user["held_message_id"] = user["held_message_id"], None
            reply.add(self.t["case_selected"].format(case_id=selected.case_id))
            text = self.store.raw(held) if held else None
            if text:
                return self._text(user, text, message_id=held, reply=reply, selected=True)
            self._continue(selected, reply)
            return self._finish(user, selected, reply)
        if key in ("confirm", "edit"):
            case_id, _, rev = value.partition(":")
            target = self.store.load_case(case_id)
            if (target is None or target.user_ref != user["user_ref"] or target.intake_status != IntakeStatus.AWAITING_CONFIRMATION
                    or str(target.draft_rev) != rev or not all_slots_satisfied(target)):
                reply.add(self.t["stale_button"])
                return self._finish(user, case, reply)
            user["current_case_id"] = target.case_id
            if key == "confirm":
                self._confirm(target, reply)
            else:
                target.intake_status = IntakeStatus.COLLECTING
                reply.add(self.t["edit_prompt"])
            return self._finish(user, target, reply)
        if key == "location_choice" and case is not None and case.pending_location is not None:
            loc = case.pending_location
            case.pending_location = None
            case.pending_question = None
            if value == "relocate":
                self._relocate(case, loc, reply)
                self._continue(case, reply)
                self.store.save_case(case)
                return self._finish(user, case, reply)
            self.store.save_case(case)
            new = self._open_case(user)
            new.personal.location = loc
            self._data_changed(new)
            reply.add(self.t["new_case_started"].format(case_id=new.case_id))
            self._continue(new, reply)
            return self._finish(user, new, reply)
        if key == "reuse_contact":
            parts = value.split(":")
            if (len(parts) != 4 or parts[0] not in {"yes", "no"} or case is None
                    or case.case_id != parts[1] or case.reuse_contact_source != parts[2]
                    or str(case.draft_rev) != parts[3] or case.pending_question != "reuse_contact"):
                reply.add(self.t["stale_button"])
            else:
                self._reuse_contact(case, parts[0] == "yes", reply)
            return self._finish(user, case, reply)
        reply.add(self.t["stale_button"])
        return self._finish(user, case, reply)

    def _reuse_contact(self, case: Case, reuse: bool, reply: Reply) -> bool:
        """คัดลอกชื่อ นามสกุล เบอร์ จาก Personal DB ของเคสก่อนหน้าโดยตรง ไม่ผ่าน LLM"""
        case.pending_question = None
        case.reuse_contact_asked = True
        previous = self.store.load_case(case.reuse_contact_source) if reuse and case.reuse_contact_source else None
        if reuse and (previous is None or previous.user_ref != case.user_ref
                      or previous.draft_rev != case.reuse_contact_source_rev
                      or not all((previous.personal.first_name, previous.personal.last_name, previous.personal.phone_number))):
            case.reuse_contact_asked = False
            case.reuse_contact_source = None
            case.reuse_contact_source_rev = None
            reply.add(self.t["stale_button"])
            if not self._offer_reuse_contact(case, reply):
                self._continue(case, reply)
            return True
        if previous:
            case.personal.first_name = previous.personal.first_name
            case.personal.last_name = previous.personal.last_name
            case.personal.phone_number = previous.personal.phone_number
            self._data_changed(case)
            reply.add(self.t["contact_received"])
        self._continue(case, reply)
        return True

    def _previous_contact(self, case: Case) -> Case | None:
        return self.store.previous_contact_case(case.user_ref, case.case_id)

    def _offer_reuse_contact(self, case: Case, reply: Reply) -> bool:
        if case.reuse_contact_asked or any((case.personal.first_name, case.personal.last_name, case.personal.phone_number)):
            return False
        previous = self._previous_contact(case)
        if previous is None:
            return False
        case.reuse_contact_asked = True
        case.reuse_contact_source = previous.case_id
        case.reuse_contact_source_rev = previous.draft_rev
        case.pending_question = "reuse_contact"
        person = previous.personal
        masked_name = f"{person.first_name[:1]}*** {person.last_name[:1]}***"
        reply.add(self.t["reuse_contact"].format(name=masked_name, phone=mask_phone(person.phone_number)))
        reply.quick_replies = [QuickReply(label=self.labels["yes"], data=f"reuse_contact=yes:{case.case_id}:{previous.case_id}:{case.draft_rev}"),
                               QuickReply(label=self.labels["no"], data=f"reuse_contact=no:{case.case_id}:{previous.case_id}:{case.draft_rev}")]
        return True

    # ------------------------------------------------------------------
    # Multi-Case
    # ------------------------------------------------------------------

    def _current_case(self, user) -> Case | None:
        if not user["current_case_id"]:
            return None
        case = self.store.load_case(user["current_case_id"])
        return case if case and case.case_status == CaseStatus.ACTIVE else None

    def _open_case(self, user) -> Case:
        case = self.store.new_case(user["user_ref"])
        user["current_case_id"] = case.case_id
        self.store.add_event(case.case_id, "case_created", {})
        self._refresh_priority(case)
        return case

    def _ask_case_selection(self, user, actives: list[Case], message_id: str, reply: Reply) -> TurnOutput:
        user["pending_question"] = "case_selection"
        user["held_message_id"] = message_id
        reply.add(self.t["case_selection"])
        reply.quick_replies = [QuickReply(label=f"{c.case_id[-6:]} {(c.problem_text or 'ยังไม่ระบุ')}"[:20],
                                          data=f"case_select={c.case_id}") for c in actives[-13:]]
        return self._finish(user, None, reply)

    # ------------------------------------------------------------------
    # Draft reminder (หัวข้อ 8) — เรียกจาก background job
    # ------------------------------------------------------------------

    def due_reminders(self, now=None) -> list[tuple[str, str, str]]:
        now = now or now_bkk()
        due = []
        for case in self.store.all_active_cases():
            if not case.is_draft_open or case.reminder_sent or case.intake_status == IntakeStatus.CONFIRMED:
                continue
            if now - parse_time(case.last_user_at) < DRAFT_REMINDER_AFTER:
                continue
            reply = Reply()
            reply.add(self.t["draft_reminder"])
            case.reminder_sent = True
            self.store.save_case(case)
            self.store.add_event(case.case_id, "draft_reminder_sent", {"draft_round": case.draft_round})
            due.append((case.user_ref, case.case_id, "\n\n".join(reply.parts)))
        return due

    def close_case(self, case_id: str, *, actor: str, reason: str) -> bool:
        """เรียกโดยเจ้าหน้าที่หลังบ้านเท่านั้น Agent 1 ไม่ปิดเคสเอง"""
        case = self.store.load_case(case_id)
        if case is None or case.case_status == CaseStatus.INACTIVE:
            return False
        case.case_status = CaseStatus.INACTIVE
        case.touch()
        self.store.save_case(case)
        self.store.add_event(case_id, "case_closed", {"closed_by": actor, "reason": reason})
        user = self.store.user(case.user_ref)
        if user["current_case_id"] == case_id:
            user["current_case_id"] = None
            self.store.save_user(user)
        return True

    # ------------------------------------------------------------------

    def _finish(self, user, case: Case | None, reply: Reply) -> TurnOutput:
        if case is not None:
            refresh_slots(case)
            if case.pending_question == SlotName.PROBLEM.value and reply.parts:
                case.conversation_context.turns.append(ConversationTurn(role="assistant", text=reply.parts[-1][:600]))
                case.conversation_context.turns = case.conversation_context.turns[-8:]
            for turn in case.conversation_context.turns:
                for value in (case.personal.first_name, case.personal.last_name, case.personal.phone_number):
                    if value:
                        turn.text = turn.text.replace(value, "[PERSONAL]")
                turn.text = filter_text(turn.text).text[:600]
            for answer in case.conversation_context.answers:
                for field in ("question", "answer"):
                    value = getattr(answer, field)
                    for personal in (case.personal.first_name, case.personal.last_name, case.personal.phone_number):
                        if personal:
                            value = value.replace(personal, "[PERSONAL]")
                    setattr(answer, field, filter_text(value).text[:240 if field == "question" else 600])
            case.conversation_context.last_support = filter_text(case.conversation_context.last_support).text[:600]
            self.store.save_case(case)
        self.store.save_user(user)
        out = self._output(reply, case, user)
        self.store.add_raw(user["user_ref"], case.case_id if case else None, "out", "text", out.output_text)
        return out

    def _output(self, reply: Reply, case: Case | None = None, user=None) -> TurnOutput:
        return TurnOutput(
            output_text="\n\n".join(reply.parts), quick_replies=reply.quick_replies,
            case_id=case.case_id if case else None, case_status=case.case_status if case else None,
            intake_status=case.intake_status if case else None,
            pending_question=(user or {}).get("pending_question") or (case.pending_question if case else None),
            missing_slots=missing_slots(case) if case else [], kb_doc_ids=list(dict.fromkeys(reply.kb_doc_ids)),
        )
