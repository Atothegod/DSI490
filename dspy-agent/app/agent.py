"""
agent.py — FloodAgent: ประกอบ safety + Extraction + memory + conversation เป็น agent ตัวเดียว (ขั้นที่ 5)

ใช้แบบเดียวกับ guy01.ipynb
    agent = FloodAgent(hotlines=HOTLINES)
    output = agent(input_text="น้ำท่วมบ้านค่ะ")
    output.output_text

ขอบเขตตอนนี้ (Notebook เท่านั้น)
- เก็บทุกอย่างในหน่วยความจำของ object หนึ่งตัวต่อผู้ใช้หนึ่งคน (ยังไม่มี repository.py / PostgreSQL)
- ถือว่าส่งข้อความตอบสำเร็จทุกครั้ง (ยังไม่มี LINE)
- ยังไม่ส่ง event ให้ Agent 2 (ยังไม่มี handoff.py) — ดูว่าควรส่งอะไรได้จาก output.next_actions
- ชื่อพื้นที่เป็น unverified ทั้งหมด (ยังไม่มีรายชื่อพื้นที่สิงห์บุรี)
- คำตอบความรู้ยังไม่มี RAG ใช้ข้อความสำรอง knowledge_unavailable
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Iterable

import dspy

from .conversation import (
    PendingConfirmation,
    TurnPlan,
    after_delivery,
    apply_submission,
    plan_turn,
    reconcile_needs,
    render_reply,
)
from .memory import MergeResult, apply_updates, close_unanswered
from .safety import asks_for_contact, keyword_flags, load_templates, merge_urgent_flags, redact_phones, render_contact_answer
from .schemas import (
    AgentTurnOutput,
    ContactRecord,
    ExtractionResult,
    FieldStatus,
    Hotline,
    IncidentFieldName,
    IncidentState,
    MatchingStatus,
    NextAction,
    PendingQuestion,
    TrackedField,
    now_bkk,
)

F = IncidentFieldName
ExtractFn = Callable[[str, IncidentState | None, list[IncidentFieldName] | None], ExtractionResult]


@dataclass
class TurnLog:
    """สิ่งที่เกิดขึ้นใน turn หนึ่ง ใช้ดูย้อนหลังด้วย agent.show_last()"""

    user_text: str
    redacted_text: str
    extraction: ExtractionResult | None = None
    merge: MergeResult | None = None
    keyword_flags: set = field(default_factory=set)
    new_flags: set = field(default_factory=set)
    plan: TurnPlan | None = None
    notes: list[str] = field(default_factory=list)


class FloodAgent(dspy.Module):
    def __init__(
        self,
        hotlines: Iterable[Hotline],
        *,
        templates: dict | None = None,
        extractor: ExtractFn | None = None,
        internal_user_id: str = "usr_demo",
    ):
        super().__init__()
        self.hotlines = list(hotlines)
        if not self.hotlines:
            raise ValueError("ต้องมี hotlines อย่างน้อยหนึ่งรายการ ข้อความเหตุด่วนต้องแสดงช่องทางฉุกเฉินได้เสมอ")
        self.templates = templates or load_templates()
        if extractor is None:
            from .extraction import Extractor  # โหลดเฉพาะเมื่อใช้ Gemini จริง
            extractor = Extractor()
        self.extract = extractor
        self.internal_user_id = internal_user_id
        self.reset()

    # ---------------------------------------------------------------- state
    def reset(self):
        """เริ่มบทสนทนาใหม่ (เหมือนผู้ใช้คนใหม่)"""
        self.conversation_id = f"conv_{self.internal_user_id}"
        self.incidents: dict[str, IncidentState] = {}
        self.active_id: str | None = None
        self.pending_question: PendingQuestion | None = None
        self.pending_confirmation: PendingConfirmation | None = None
        self.contacts: list[ContactRecord] = []  # เบอร์จริงอยู่ที่นี่เท่านั้น ไม่เข้า state และไม่ถึง Gemini
        self.turns: list[TurnLog] = []

    @property
    def state(self) -> IncidentState | None:
        return self.incidents.get(self.active_id) if self.active_id else None

    def _new_incident(self, **kw) -> IncidentState:
        s = IncidentState(internal_user_id=self.internal_user_id, conversation_id=self.conversation_id, **kw)
        self.incidents[s.incident_id] = s
        self.active_id = s.incident_id
        self.pending_question = None
        self.pending_confirmation = None
        return s

    # ---------------------------------------------------------------- turn
    def forward(self, input_text: str) -> AgentTurnOutput:
        now = now_bkk()
        msg_id = f"msg_{len(self.turns) + 1}"
        red = redact_phones(input_text)  # 1. ปิดบังเบอร์ก่อนทุกอย่าง
        log = TurnLog(user_text=input_text, redacted_text=red.text, keyword_flags=keyword_flags(red.text))
        self.turns.append(log)

        state = self.state if (self.state and self.state.is_open) else None
        pending = self.pending_question.field_names if self.pending_question and state else None
        r = self.extract(red.text, state, pending)  # 2. ✦ Gemini
        log.extraction = r

        # 3. เลือก incident ที่ข้อความนี้เป็นของ
        is_urgent_msg = bool(log.keyword_flags or r.urgent_flags)
        has_incident_info = bool(r.updates or is_urgent_msg)
        if state is not None and self._other_incident(r, state):
            if not is_urgent_msg:
                return self._ask_incident_match(log, r, state)
            # ข้อ 3.2 — เหตุด่วนไม่หน่วงด้วยการถาม: สร้าง provisional incident ผูกกับเหตุเดิมแบบ unconfirmed
            state = self._new_incident(matching_status=MatchingStatus.UNCONFIRMED,
                                       possible_related_incident_ids=[state.incident_id])
            log.notes.append("ไม่ชัดว่าเหตุไหน + เหตุด่วน → สร้าง provisional incident (matching_status=unconfirmed)")
        elif state is None:
            if not has_incident_info:
                return self._no_incident_reply(log, r, red.text)
            state = self._new_incident()

        # 4. merge + flags + needs + คำถามที่ไม่ได้คำตอบ
        m = apply_updates(state, r.updates, message_text=red.text, message_id=msg_id, now=now)
        log.merge = m
        fl = merge_urgent_flags(m.state.urgent_flags, log.keyword_flags, r.urgent_flags,
                                detected_at=now, message_id=msg_id)
        log.new_flags = fl.newly_detected
        st = IncidentState.model_validate({**m.state.model_dump(),
                                           "urgent_flags": [x.model_dump() for x in fl.records]})
        st, needs_change = reconcile_needs(st, fl.newly_detected, now=now, message_id=msg_id)
        if needs_change:
            log.notes.append("help_requested ใหม่: เอา info_only ที่ขัดกับคำขอออก (ไม่เดา rescue)")
        if red.has_phone:
            st = self._save_contact(st, red, now)
            log.notes.append(f"พบเบอร์ {len(red.phones)} เบอร์: เก็บใน ContactRecord ไม่ส่ง Gemini")
        st = close_unanswered(st, self.pending_question)

        # 5. วางแผน → ส่งรายงาน → ข้อความตอบ → บันทึกหลังส่งสำเร็จ
        plan = plan_turn(st, extraction=r, merge=m, new_flags=fl.newly_detected,
                         extra_changes=[needs_change] if needs_change else [],
                         pending_confirmation=self.pending_confirmation, message_text=red.text)
        log.plan = plan
        if plan.submit_trigger:
            st = apply_submission(st, now=now)  # TODO handoff.py: บันทึก incident_submitted ลง outbox
        text = render_reply(plan, st, hotlines=self.hotlines, templates=self.templates)

        d = after_delivery(st, plan, message_id=f"{msg_id}_reply", now=now,  # ถือว่า LINE ส่งสำเร็จ
                           pending_question=self.pending_question, pending_confirmation=self.pending_confirmation)
        self.incidents[d.state.incident_id] = d.state
        self.pending_question, self.pending_confirmation = d.pending_question, d.pending_confirmation
        return self._output(text, r, d.state, plan)


    # ---------------------------------------------------------------- กรณีพิเศษ
    @staticmethod
    def _other_incident(r: ExtractionResult, state: IncidentState) -> bool:
        """
        ข้อ 3.2 — ไม่รวมข้อมูลต่างตำแหน่งเข้าเหตุเดิมอัตโนมัติ
        unclear เสมอ หรือ new_incident ที่บอกพื้นที่ไม่ตรงกับเหตุที่กำลังคุย
        (จับคู่ด้วยข้อความดิบชั่วคราว ของจริงใช้รหัสพื้นที่จาก areas.py)
        """
        if r.target == "unclear":
            return True
        if r.target != "new_incident" or state.location_area.value is None:
            return False
        norm = lambda x: "".join(str(x).split())
        active = norm(state.location_area.value.raw)
        for u in r.updates:
            if u.field == F.LOCATION_AREA and u.value:
                name = u.value if isinstance(u.value, str) else u.value.get("name", "")
                if norm(name) not in active and active not in norm(name):
                    return True
        return False

    def _save_contact(self, st: IncidentState, red, now) -> IncidentState:
        """G14 — เบอร์ที่ผู้ใช้ให้มาเอง เก็บแยกใน ContactRecord และบันทึกว่าได้ช่องทางติดต่อแล้ว"""
        rec = ContactRecord(internal_user_id=self.internal_user_id, incident_id=st.incident_id,
                            phone=red.phones[0].normalized, purpose=self.templates["questions"]["contact"][0],
                            consent_at=now)
        self.contacts.append(rec)
        ask = st.contact_ask
        contact = TrackedField[bool](status=FieldStatus.KNOWN, value=True, updated_at=now,
                                     ask_count=ask.ask_count, last_asked_at=ask.last_asked_at)
        return IncidentState.model_validate({**st.model_dump(), "contact_ask": contact.model_dump(),
                                             "contact_ref": rec.contact_ref})

    def _ask_incident_match(self, log: TurnLog, r: ExtractionResult, state: IncidentState) -> AgentTurnOutput:
        # ยังไม่มีขั้นตอนรับคำตอบ "เหตุเดิม/เหตุใหม่" (รอ repository.py) — ข้อความนี้ยังไม่ถูก merge
        log.notes.append(f"target={r.target} คนละพื้นที่/ไม่ชัด: ถามเหตุเดิมหรือเหตุใหม่ ยังไม่ merge ข้อความนี้")
        area = state.location_area.value.raw if state.location_area.value else ""
        text = self.templates["incident_match"].format(active_area=area or "ก่อนหน้านี้")
        return AgentTurnOutput(output_text=text, intents=r.intents, incident_id=state.incident_id,
                               incident_status=state.status, matching_status=state.matching_status,
                               urgent_flags=sorted(state.urgent_flag_types, key=lambda f: f.value),
                               next_actions=[NextAction.ASK_INCIDENT_MATCH], model=self._model())

    def _no_incident_reply(self, log: TurnLog, r: ExtractionResult, text: str) -> AgentTurnOutput:
        """ทักทาย ถามความรู้ หรือถามช่องทางติดต่อ โดยยังไม่มีเหตุ — ไม่สร้าง incident"""
        t, parts, actions = self.templates, [], []
        if asks_for_contact(text):
            parts.append(render_contact_answer(self.hotlines, t))
        if r.question:
            from .safety import format_hotline_text
            parts.append(t["knowledge_unavailable"].format(hotline_text=format_hotline_text(self.hotlines)))
            actions.append(NextAction.ANSWER_QUESTION)
        if not parts:
            parts.append(t["greeting"])
        log.notes.append("ไม่มีข้อมูลเหตุการณ์ ยังไม่สร้าง incident")
        return AgentTurnOutput(output_text="\n\n".join(parts), intents=r.intents, incident_id=None,
                               incident_status=None, next_actions=actions or [NextAction.NONE],
                               used_hotline_template=asks_for_contact(text), model=self._model())

    def _output(self, text: str, r: ExtractionResult, st: IncidentState, plan: TurnPlan) -> AgentTurnOutput:
        return AgentTurnOutput(
            output_text=text, intents=r.intents, incident_id=st.incident_id, incident_status=st.status,
            matching_status=st.matching_status,
            urgent_flags=sorted(st.urgent_flag_types, key=lambda f: f.value),
            missing_fields=plan.missing_fields, next_actions=plan.actions,
            pending_question=self.pending_question, used_hotline_template=bool(plan.hotline_flags),
            model=self._model(),
        )

    @staticmethod
    def _model() -> str | None:
        lm = dspy.settings.lm
        return getattr(lm, "model", None) if lm else None

    # ---------------------------------------------------------------- ดูข้างใน
    def show_state(self):
        s = self.state
        if s is None:
            print("ยังไม่มี incident")
            return
        print(f"incident {s.incident_id}  status={s.status.value}  matching={s.matching_status.value}"
              f"  urgent={sorted(f.value for f in s.urgent_flag_types)}")
        for name in IncidentFieldName:
            f = s.tracked_field(name)
            v = f.value
            val = "" if v is None else json.dumps(
                v.model_dump(mode="json", exclude_none=True) if hasattr(v, "model_dump")
                else [x.value for x in v] if isinstance(v, list) else v, ensure_ascii=False)
            hist = f"  (เปลี่ยน {len(f.history)} ครั้ง)" if f.history else ""
            print(f"  {name.value:22} {f.status.value:11} ถาม {f.ask_count}  {val}{hist}")
        if self.pending_question:
            print("  คำถามที่ค้าง:", [f.value for f in self.pending_question.field_names])
        if self.pending_confirmation:
            print("  รอผู้ใช้ยืนยันสรุป")
        if len(self.incidents) > 1:
            print(f"  incident ทั้งหมด {len(self.incidents)} รายการ:", list(self.incidents))

    def show_last(self):
        """สิ่งที่เกิดใน turn ล่าสุด: Gemini ดึงอะไร, การเปลี่ยนค่า, flag, เหตุผลของการตัดสิน"""
        if not self.turns:
            print("ยังไม่มีข้อความ")
            return
        t = self.turns[-1]
        print(f"👤 {t.user_text}")
        if t.redacted_text != t.user_text:
            print(f"   🔒 ส่ง Gemini เป็น: {t.redacted_text}")
        if t.extraction:
            r = t.extraction
            print(f"   ✦ intents={[i.value for i in r.intents]} target={r.target}")
            for u in r.updates:
                v = "" if u.value is None else json.dumps(u.value, ensure_ascii=False)
                print(f"     - {u.field.value}: {u.status.value} {v}{' (Gemini: แก้ไข)' if u.is_correction else ''}")
        if t.merge:
            for c in t.merge.changes:
                print(f"   • {c.field.value if hasattr(c.field, 'value') else c.field}: {c.change_type.value}")
            for f in t.merge.confirmations:
                print(f"   • {f.value}: confirmation")
            for f, st in t.merge.status_changes.items():
                print(f"   • {f.value}: {st.value}")
            for rej in t.merge.rejected:
                print(f"   ✗ ไม่รับ {rej.update.field.value}: {rej.reason}")
        if t.keyword_flags:
            print(f"   keyword พบ: {sorted(f.value for f in t.keyword_flags)}")
        if t.new_flags:
            print(f"   ⚠️ flag ใหม่: {sorted(f.value for f in t.new_flags)}")
        if t.plan:
            print(f"   ➜ actions={[a.value for a in t.plan.actions]}"
                  f"{' submit=' + t.plan.submit_trigger if t.plan.submit_trigger else ''}")
        for n in [*t.notes, *(t.plan.notes if t.plan else [])]:
            print(f"   ℹ️ {n}")
