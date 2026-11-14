"""
Agent 1 — Case Manager + Decision & Safety Logic (หัวข้อ 3, 5, 8, 9, 10)

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
    missing_notice,
    missing_slots,
    next_slot,
    refresh_slots,
    summary_text,
    version_snapshot,
)
from .extraction import Extractor, ground, is_medical_or_distress
from .knowledge import KBDoc, KnowledgeBase
from .privacy import FilterResult, filter_text, normalize_text
from .response import ResponseGenerator, FORBIDDEN, ADVICE_WORDS
from .schemas import (
    AFFECTED_GROUPS,
    SLOT_LABELS_TH,
    CaseIntent,
    CaseStatus,
    ConversationTurn,
    Extraction,
    IncidentData,
    IntakeStatus,
    Location,
    MessageIntent,
    QuickReply,
    SlotName,
    SlotStatus,
    TurnOutput,
    now_bkk,
)
from .store import Store, parse_time

APP_DIR = Path(__file__).resolve().parent
DRAFT_REMINDER_AFTER = timedelta(minutes=30)
SUMMARY_MESSAGES = 3
NAME_SLOTS = (SlotName.FIRST_NAME, SlotName.LAST_NAME)
_NAME_PREFIX = re.compile(r"^(?:ผม|ดิฉัน|ฉัน|หนู|เรา)?\s*(?:ชื่อ(?:จริง)?|นามสกุล)?\s*(?:ว่า)?\s*(?:นาย|นางสาว|นาง|น\.ส\.)?\s*")
_POLITE = re.compile(r"\s*(?:ครับ|ค่ะ|คะ|ค่า|จ้า|นะ)+\s*$")
_NOT_A_NAME = ("น้ำ", "ท่วม", "ช่วย", "ไม่", "ทำไม", "อะไร", "ยังไง", "ไหม", "ได้", "แล้ว")
YES = ("ใช่", "ใช้", "ได้", "ok", "โอเค", "เอา")

ExtractFn = Callable[[str, IncidentData, "str | None", str], Extraction]


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
        if text is not None and text.strip() == self.labels["new_case"]:
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

        if case is not None and case.context_complete and not is_medical_or_distress(f.text) and self._direct_answer(case, f, text, reply):
            return self._finish(user, case, reply)

        ex = ground(self.extract(f.text, case.display_incident() if case else IncidentData(),
                                 case.pending_question if case else None,
                                 self._conversation_context(case) if case else ""), f.text)

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
        self._support(reply, case, ex, f)
        if ex.case_intent == CaseIntent.RELOCATE and not ex.problem:
            reply.add(self.t["relocation_ask_pin"])
            reply.quick_replies = [QuickReply(type="location", label=self.labels["share_location"])]
            return self._finish(user, case, reply)
        interjected = bool({MessageIntent.QUESTION, MessageIntent.EMOTIONAL} & set(ex.intents)) and bool(reply.parts)
        self._continue(case, reply, resumed=interjected, count_unclear=ex.count_unclear)
        return self._finish(user, case, reply)

    def _direct_answer(self, case: Case, f: FilterResult, text: str, reply: Reply) -> bool:
        """คำตอบของคำถาม PII บันทึกตรงลง Personal Data โดยไม่ผ่าน LLM"""
        pending = case.pending_question
        if pending == "reuse_contact":
            if text.strip().startswith(("ไม่", "ไม่ใช่")):
                return self._reuse_contact(case, False, reply)
            if text.strip().lower().startswith(YES):
                return self._reuse_contact(case, True, reply)
            return False
        if case.intake_status == IntakeStatus.AWAITING_USER_INPUT and pending is None:
            missing = [s for s in missing_slots(case) if s in NAME_SLOTS]
            pending = missing[0].value if len(missing) == 1 and len(missing_slots(case)) == 1 else None
        if pending == SlotName.PHONE_NUMBER.value and f.phones:
            case.personal.phone_number = f.phones[0]
        elif pending in (SlotName.FIRST_NAME.value, SlotName.LAST_NAME.value) and (parts := self._name_parts(text)):
            if pending == SlotName.FIRST_NAME.value:
                case.personal.first_name = parts[0]
                if len(parts) == 2 and not case.personal.last_name:
                    case.personal.last_name = parts[1]
            else:
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
        t = _POLITE.sub("", _NAME_PREFIX.sub("", normalize_text(text).strip()))
        parts = t.split()
        if is_medical_or_distress(t):
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
        before = (case.incident.model_dump(), case.personal.model_dump(), case.context_complete)
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
        if ex.problem or ex.context_question:
            case.context_complete = bool(ex.context_complete and case.incident.problem)
            question = ex.context_question or ""
            # Clarification is a request for information, never uncited treatment advice.
            filtered_question = filter_text(question).text
            if question and filtered_question == question and not FORBIDDEN.search(question) and not ADVICE_WORDS.search(question):
                case.context_question = question
            else:
                case.context_question = None
            if case.context_complete:
                case.context_question = None
        for group in AFFECTED_GROUPS:
            if (value := getattr(ex.affected_people, group)) is not None:
                setattr(case.incident.affected_people, group, value)
        case.last_user_at = now_bkk()
        case.last_intents = [i.value for i in ex.intents]
        recent = [*filter(None, case.conversation_summary.split("\n")), f.text[:200]]
        case.conversation_summary = "\n".join(recent[-SUMMARY_MESSAGES:])

        ctx = case.conversation_context
        ctx.turns.append(ConversationTurn(role="user", text=f.text[:600]))
        ctx.turns = ctx.turns[-8:]
        after = (case.incident.model_dump(), case.personal.model_dump(), case.context_complete)
        if after != before:
            self._data_changed(case)

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
    def _support(self, reply: Reply, case: Case | None, ex: Extraction, f: FilterResult) -> None:
        # Search using the current problem as well, so "ทำไงดี" keeps its subject.
        problem = case.problem_text if case else ex.problem
        query = " ".join(filter(None, (problem, ex.question or f.text)))
        docs = [h.doc for h in self.kb.search(query, limit=1)] if ex.question or ex.problem else []
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
        if MessageIntent.QUESTION in ex.intents and not docs:
            reply.add(self.t["no_kb_info"])

    def _empathy_template(self, text: str) -> str:
        for word, message in self.t["empathy_by_feeling"].items():
            if word in text:
                return message
        return self.t["empathy"]

    def _conversation_context(self, case: Case) -> str:
        # Re-filter on read as well, including names already held in the contact table.
        text = case.conversation_context.model_dump_json() + "\n" + case.conversation_summary
        for value in (case.personal.first_name, case.personal.last_name, case.personal.phone_number):
            if value:
                text = text.replace(value, "[PERSONAL]")
        return filter_text(text).text

    def _add_channels(self, reply: Reply, docs: list[KBDoc]) -> None:
        for d in docs:
            reply.add(self.t["channel_line"].format(text=d.text))
            reply.kb_doc_ids.append(d.kb_doc_id)

    # ------------------------------------------------------------------
    # Slot Filling / Confirmation
    # ------------------------------------------------------------------

    def _continue(self, case: Case, reply: Reply, *, resumed: bool = False, count_unclear: bool = False) -> None:
        if case.case_status == CaseStatus.INACTIVE:
            reply.add(self.t["case_closed"])
            return
        if case.intake_status == IntakeStatus.CONFIRMED:
            reply.quick_replies = [QuickReply(label=self.labels["new_case"], data="new_case")]
            return
        # Complete the meaning of problem before asking contact/location slots.
        # Unlike administrative slots, this is not capped at two questions.
        if not case.context_complete:
            case.pending_question = SlotName.PROBLEM.value
            case.intake_status = IntakeStatus.COLLECTING
            question = case.context_question or self.t["ask"]["problem"]
            reply.add(question)
            case.slots[SlotName.PROBLEM].ask_count += 1
            case.slots[SlotName.PROBLEM].asked_at = now_bkk()
            return
        if all_slots_satisfied(case):
            if case.intake_status == IntakeStatus.AWAITING_CONFIRMATION:
                reply.add(self.t["awaiting_confirmation_reminder"])
            else:
                case.intake_status = IntakeStatus.AWAITING_CONFIRMATION
                reply.add(summary_text(case))
            case.pending_question = None
            reply.quick_replies = self._confirm_buttons(case)
            return
        slot = next_slot(case, SlotName.AFFECTED_PEOPLE if count_unclear else None)
        if slot in (SlotName.FIRST_NAME, SlotName.LAST_NAME, SlotName.PHONE_NUMBER) and not case.reuse_contact_asked \
                and all(case.slots[s].status == SlotStatus.MISSING for s in (SlotName.FIRST_NAME, SlotName.LAST_NAME, SlotName.PHONE_NUMBER)) \
                and self._previous_contact(case):
            case.reuse_contact_asked = True
            case.pending_question = "reuse_contact"
            reply.add(self.t["reuse_contact"])
            reply.quick_replies = [QuickReply(label=self.labels["yes"], data="reuse_contact=yes"),
                                   QuickReply(label=self.labels["no"], data="reuse_contact=no")]
            return
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

    def _confirm_buttons(self, case: Case) -> list[QuickReply]:
        ref = f"{case.case_id}:{case.draft_rev}"
        return [QuickReply(label=self.labels["confirm"], data=f"confirm={ref}"),
                QuickReply(label=self.labels["edit"], data=f"edit={ref}")]

    def _confirm(self, case: Case, reply: Reply) -> None:
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
        if key == "new_case":
            case = self._open_case(user)
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
        if key == "reuse_contact" and case is not None:
            self._reuse_contact(case, value == "yes", reply)
            return self._finish(user, case, reply)
        reply.add(self.t["stale_button"])
        return self._finish(user, case, reply)

    def _reuse_contact(self, case: Case, reuse: bool, reply: Reply) -> bool:
        """คัดลอกชื่อ นามสกุล เบอร์ จาก Personal DB ของเคสก่อนหน้าโดยตรง ไม่ผ่าน LLM"""
        case.pending_question = None
        case.reuse_contact_asked = True
        previous = self._previous_contact(case) if reuse else None
        if previous:
            case.personal.first_name = previous.personal.first_name
            case.personal.last_name = previous.personal.last_name
            case.personal.phone_number = previous.personal.phone_number
            self._data_changed(case)
            reply.add(self.t["contact_received"])
        self._continue(case, reply)
        return True

    def _previous_contact(self, case: Case) -> Case | None:
        others = [c for c in self.store.active_cases(case.user_ref) if c.case_id != case.case_id
                  and c.personal.first_name and c.personal.last_name and c.personal.phone_number]
        return others[-1] if others else None

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
