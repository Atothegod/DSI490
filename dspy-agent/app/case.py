"""
Case State + Slot Manager + Summary Generator
Case เป็นภาพรวมของเคสหนึ่งเคสที่ store.py โหลดจากและบันทึกลงหลายตาราง
ไม่เรียก LLM ทุกฟังก์ชันให้ผลแน่นอน
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from .priority import RULESET_VERSION, evaluate_priority, initial_priority
from .schemas import (
    AFFECTED_GROUPS,
    AFFECTED_LABELS_TH,
    AffectedPeople,
    CasePriority,
    MAX_SLOT_ASKS,
    RiskFacts,
    SLOT_LABELS_TH,
    CaseStatus,
    ConversationContext,
    IncidentData,
    IntakeStatus,
    Location,
    PersonalData,
    SlotName,
    SlotState,
    SlotStatus,
    StrictModel,
    now_bkk,
    requested_help_from_text,
    requested_help_phrase,
)

SLOT_ORDER = (SlotName.PROBLEM, SlotName.AFFECTED_PEOPLE, SlotName.LOCATION,
              SlotName.FIRST_NAME, SlotName.LAST_NAME, SlotName.PHONE_NUMBER)


class Case(StrictModel):
    case_id: str
    user_ref: str
    case_status: CaseStatus = CaseStatus.ACTIVE
    intake_status: IntakeStatus = IntakeStatus.COLLECTING
    current_version: int = 0  # 0 = ยังไม่มี Confirmed Version
    draft_rev: int = 0  # เพิ่มทุกครั้งที่ Draft เปลี่ยน ใช้กันการกดปุ่มของสรุปเก่า
    created_at: datetime = Field(default_factory=now_bkk)
    updated_at: datetime = Field(default_factory=now_bkk)
    confirmed_at: datetime | None = None

    # Draft (case_drafts)
    incident: IncidentData = Field(default_factory=IncidentData)  # problem รวมสถานการณ์และความต้องการ = ข้อความต้นฉบับของผู้ใช้ (ผ่าน Privacy Filter)
    problem_display: str | None = None  # ฉบับปัจจุบันจาก LLM ใช้แสดงสรุปและส่ง Pipeline หลังผู้ใช้ยืนยัน
    editing: bool = False  # มีร่างการแก้ไขของเคสที่ยืนยันแล้ว
    draft_round: int = 1  # ร่างแรก = 1, ร่างแก้ V1 = 2, ...
    reminder_sent: bool = False  # เตือนได้ 1 ครั้งต่อรอบ ไม่รีเซ็ตเมื่อผู้ใช้ตอบ
    last_user_at: datetime = Field(default_factory=now_bkk)
    address_masked: bool = False

    # Personal Data (case_personal_data) — ไม่ส่งเข้า LLM
    personal: PersonalData = Field(default_factory=PersonalData)
    pending_location: Location | None = None  # หมุดใหม่ที่รอถามว่าย้ายจุดหรือแจ้งเหตุใหม่

    # Conversation State (case_slots)
    slots: dict[SlotName, SlotState] = Field(default_factory=lambda: {s: SlotState() for s in SlotName})
    pending_question: str | None = None
    missing_notice_sent: bool = False
    expect_relocation: bool = False
    reuse_contact_asked: bool = False
    reuse_contact_source: str | None = None
    reuse_contact_source_rev: int | None = None
    last_intents: list[str] = Field(default_factory=list)
    context_complete: bool = False
    context_question: str | None = None
    context_paused: bool = False  # ถามรายละเอียดครบเพดานแล้วยังไม่ครบ: หยุดถาม ไม่ถือว่าครบ แจ้งผู้ใช้ครั้งเดียว

    # Priority ตาม Ruleset 1.1.1: tri-state facts และหลักฐานราย fact เก็บภายใน
    priority: CasePriority = Field(default_factory=initial_priority)
    priority_facts: RiskFacts = Field(default_factory=RiskFacts)
    risk_fact_evidence: dict[str, str] = Field(default_factory=dict)
    emergency_notified: bool = False
    pending_people_confirmation: AffectedPeople | None = None
    pending_people_label: str | None = None
    context_subject: str | None = None
    consultation_active: bool = False
    help_status: Literal["missing", "known", "unsure", "none"] = "missing"
    help_evidence: str | None = None
    conversation_summary: str = ""
    conversation_context: ConversationContext = Field(default_factory=ConversationContext)

    @property
    def problem_text(self) -> str | None:
        text = self.problem_display or self.incident.problem
        if not text or not self.help_evidence:
            return text
        if self.help_status == "known":
            goal = requested_help_phrase(self.help_evidence)
            goal = re.sub(r"^(?:ขอ|อยาก)", "ต้องการ", goal)
            if re.match(r"(?:ต้องการ)?ปรึกษา", goal):
                # The medical/person context is already in the current account;
                # do not reappend an old patient name from a long request quote.
                goal = "ต้องการปรึกษา"
            if not re.match(r"ต้องการ|ช่วย|ปรึกษา", goal):
                goal = "ต้องการ" + goal
        elif self.help_status == "unsure":
            goal = "ยังไม่แน่ใจว่าต้องการความช่วยเหลือแบบใด"
        elif self.help_status == "none":
            goal = "ไม่ต้องการความช่วยเหลือเพิ่มเติม"
        else:
            return text
        def key(value):
            return re.sub(r"ต้องการ|อยาก|ขอ|ให้ช่วย|ก่อน|ครับ|ค่ะ|คะ|[\s,;]", "", value)
        return text if key(goal) in key(text) else text + " " + goal

    def display_incident(self) -> IncidentData:
        """Current corrected account for display, extraction and confirmed snapshots."""
        return self.incident.model_copy(update={"problem": self.problem_text})

    @property
    def is_draft_open(self) -> bool:
        return self.current_version == 0 or self.editing

    @model_validator(mode="before")
    @classmethod
    def _legacy_case(cls, value):
        if not isinstance(value, dict):
            return value
        value = dict(value)
        if value.get("help_status", "missing") == "missing":
            incident = value.get("incident") or {}
            raw_problem = incident.get("problem", "") if isinstance(incident, dict) else incident.problem
            source = raw_problem or value.get("problem_display") or ""
            status = requested_help_from_text(source)
            if status != "missing":
                value["help_status"], value["help_evidence"] = status, source
        old_display = value.pop("needs_display", None)
        if old_display:
            value["problem_display"] = " ".join(filter(None, (value.get("problem_display") or (value.get("incident") or {}).get("problem"), old_display)))
        for key in ("priority_rule_id", "risk_facts", "danger_signals", "channels_shown"):
            value.pop(key, None)
        facts = value.get("priority_facts")
        if isinstance(facts, dict):
            value["priority_facts"] = {k: v for k, v in facts.items() if k in RiskFacts.model_fields}
        priority = value.get("priority")
        if isinstance(priority, dict) and priority.get("ruleset_version") != RULESET_VERSION:
            # Do not reinterpret legacy draft flags as evidenced risk facts.
            # Historical confirmed rows remain unchanged in case_versions.
            value["priority"] = evaluate_priority(RiskFacts.model_validate(value.get("priority_facts", {})),
                                                  value.get("risk_fact_evidence", {}))
        slots = value.get("slots")
        if slots is not None:
            value["slots"] = {k: v for k, v in slots.items() if k != "needs"}
            for slot in SlotName:
                value["slots"].setdefault(slot, SlotState())
        if value.get("pending_question") == "needs" or str(value.get("pending_question", "")).startswith("safety_"):
            value["pending_question"] = "problem"
        return value

    def touch(self, *, changed: bool = False):
        self.updated_at = now_bkk()
        if changed:
            self.draft_rev += 1


# ---------------------------------------------------------------------------
# Slot Manager
# ---------------------------------------------------------------------------


def refresh_slots(case: Case) -> None:
    """ตรวจ Slot จาก Draft และ Personal Data ด้วยโค้ด ไม่ต้องใส่ค่าจริงใน LLM Context"""
    now = now_bkk()
    filled = {
        SlotName.PROBLEM: bool(case.incident.problem) and case.context_complete and case.help_status != "missing",
        SlotName.AFFECTED_PEOPLE: case.incident.affected_people.has_anyone(),
        SlotName.LOCATION: case.personal.location is not None,
        SlotName.FIRST_NAME: bool(case.personal.first_name),
        SlotName.LAST_NAME: bool(case.personal.last_name),
        SlotName.PHONE_NUMBER: bool(case.personal.phone_number),
    }
    for slot, is_filled in filled.items():
        state = case.slots[slot]
        if is_filled is None:
            continue
        new = SlotStatus.FILLED if is_filled else SlotStatus.MISSING
        if new != state.status:
            state.status = new
            state.answered_at = now if is_filled else None


def missing_slots(case: Case) -> list[SlotName]:
    return [s for s in SLOT_ORDER if not case.slots[s].satisfied]


def all_slots_satisfied(case: Case) -> bool:
    refresh_slots(case)
    return not missing_slots(case)


def next_slot(case: Case, prefer: SlotName | None = None) -> SlotName | None:
    """Slot บังคับที่ยังขาดและถามไม่ถึง 2 ครั้ง
    prefer ใช้ถามยืนยันทันทีเมื่อข้อมูลกำกวม เช่น จำนวนผู้ประสบภัย"""
    candidates = [s for s in missing_slots(case) if case.slots[s].ask_count < MAX_SLOT_ASKS]
    if prefer in candidates:
        return prefer
    return candidates[0] if candidates else None


def mark_asked(case: Case, slot: SlotName) -> None:
    state = case.slots[slot]
    state.ask_count += 1
    state.asked_at = now_bkk()
    case.pending_question = slot.value


# ---------------------------------------------------------------------------
# Summary Generator (Template) — หัวข้อ 10
# ---------------------------------------------------------------------------


def mask_phone(phone: str) -> str:
    return f"{phone[:2]}X-XXX-{phone[-4:]}" if len(phone) >= 9 else "XXX"


def people_text(case: Case) -> str:
    people = case.incident.affected_people
    parts = [f"{AFFECTED_LABELS_TH[g]} {getattr(people, g)} คน" for g in AFFECTED_GROUPS if getattr(people, g) is not None]
    return ", ".join(parts)


def append_raw(old: str | None, new: str) -> str:
    """ต่อข้อความต้นฉบับจากหลายข้อความ ไม่ซ้ำส่วนที่มีอยู่แล้ว"""
    squash = lambda t: re.sub(r"\s+", "", t)  # noqa: E731
    if not old or squash(old) in squash(new):
        return new
    return old if squash(new) in squash(old) else f"{old} {new}"


def summary_text(case: Case) -> str:
    p = case.personal
    lines = [
        "สรุปข้อมูลการแจ้งเหตุ",
        f"1. ปัญหาที่พบ: {case.problem_text}",
        f"2. จำนวนผู้ประสบภัย: {people_text(case)}",
        "3. จุดเกิดเหตุ: ได้รับหมุดจุดเกิดเหตุผ่าน LINE แล้ว",
        f"4. ข้อมูลผู้ติดต่อ: {p.first_name} {p.last_name} · {mask_phone(p.phone_number or '')}",
    ]
    if case.address_masked:
        lines.append("หมายเหตุ: ที่อยู่ที่พิมพ์มาไม่ได้บันทึกเป็นข้อมูลเคส ระบบใช้หมุดจุดเกิดเหตุแทน")
    lines.append("กรุณาตรวจสอบข้อมูล หากถูกต้องกด [ยืนยันข้อมูล] หากต้องการเปลี่ยนกด [แก้ไขข้อมูล]")
    return "\n".join(lines)


def missing_notice(slots: list[SlotName]) -> str:
    names = ", ".join(SLOT_LABELS_TH[s] for s in slots)
    return (f"ข้อมูลที่แจ้งไว้เกือบครบแล้วค่ะ ยังขาดข้อมูล {names} ซึ่งจำเป็นสำหรับการยืนยันการแจ้งเหตุ\n\n"
            "เมื่อสะดวก สามารถส่งข้อมูลเพิ่มเติมได้เลยค่ะ ระบบจะเก็บข้อมูลที่แจ้งไว้ก่อนหน้าให้ ไม่จำเป็นต้องเริ่มใหม่")


def version_snapshot(case: Case) -> dict:
    """Confirmed Snapshot v4 — incident_priority เฉพาะ level และ reason"""
    return {
        # Send the current corrected account. Original statements/corrections
        # remain in the draft evidence and case_raw_messages for audit.
        "incident": case.display_incident().model_dump(mode="json"),
        "location": case.personal.location.model_dump() if case.personal.location else None,
        "incident_priority": case.priority.pipeline_payload(),
    }


def diff_snapshots(old: dict, new: dict) -> dict:
    """รายการเปลี่ยนแปลงสำหรับ Event case_updated เช่น {"affected_people.elderly": {"old": 3, "new": 4}}"""
    def flat(d, prefix=""):
        out = {}
        for k, v in (d or {}).items():
            key = f"{prefix}{k}"
            if isinstance(v, dict) and k != "location":
                out.update(flat(v, key + "."))
            else:
                out[key] = v
        return out

    def view(snapshot: dict) -> dict:
        priority = snapshot.get("incident_priority") or snapshot.get("priority") or {}
        return {**flat({**snapshot["incident"], "location": snapshot["location"]}),
                "incident_priority.level": priority.get("level"), "incident_priority.reason": priority.get("reason")}

    a, b = view(old), view(new)
    return {k: {"old": a.get(k), "new": b.get(k)} for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
