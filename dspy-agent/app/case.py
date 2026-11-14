"""
Case State + Slot Manager + Summary Generator (หัวข้อ 5, 8, 10)

Case เป็นภาพรวมของเคสหนึ่งเคสที่ store.py โหลดจากและบันทึกลงหลายตาราง
ไม่เรียก LLM ทุกฟังก์ชันให้ผลแน่นอน
"""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import Field, model_validator

from .schemas import (
    AFFECTED_GROUPS,
    AFFECTED_LABELS_TH,
    MAX_SLOT_ASKS,
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
    problem_display: str | None = None  # ฉบับเรียบเรียงจาก LLM ใช้แสดงผลเท่านั้น ไม่เข้า Confirmed Version
    editing: bool = False  # มีร่างการแก้ไขของเคสที่ยืนยันแล้ว
    draft_round: int = 1  # ร่างแรก = 1, ร่างแก้ V1 = 2, ...
    reminder_sent: bool = False  # เตือนได้ 1 ครั้งต่อรอบ ไม่รีเซ็ตเมื่อผู้ใช้ตอบ
    last_user_at: datetime = Field(default_factory=now_bkk)
    address_masked: bool = False
    channels_shown: list[str] = Field(default_factory=list)

    # Personal Data (case_personal_data) — ไม่ส่งเข้า LLM
    personal: PersonalData = Field(default_factory=PersonalData)
    pending_location: Location | None = None  # หมุดใหม่ที่รอถามว่าย้ายจุดหรือแจ้งเหตุใหม่

    # Conversation State (case_slots)
    slots: dict[SlotName, SlotState] = Field(default_factory=lambda: {s: SlotState() for s in SlotName})
    pending_question: str | None = None
    missing_notice_sent: bool = False
    expect_relocation: bool = False
    reuse_contact_asked: bool = False
    last_intents: list[str] = Field(default_factory=list)
    context_complete: bool = False
    context_question: str | None = None
    conversation_summary: str = ""
    conversation_context: ConversationContext = Field(default_factory=ConversationContext)

    @property
    def problem_text(self) -> str | None:
        return self.problem_display or self.incident.problem

    def display_incident(self) -> IncidentData:
        """Incident ที่ใช้ฉบับเรียบเรียง สำหรับแสดงผลและส่งให้ LLM รวมข้อความเดิมกับใหม่"""
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
        old_display = value.pop("needs_display", None)
        if old_display:
            value["problem_display"] = " ".join(filter(None, (value.get("problem_display") or (value.get("incident") or {}).get("problem"), old_display)))
        for key in ("priority", "priority_rule_id", "risk_facts", "risk_fact_evidence", "danger_signals"):
            value.pop(key, None)
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
        SlotName.PROBLEM: bool(case.incident.problem) and case.context_complete,
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
    """Confirmed Snapshot v2 — problem รวมความต้องการ ไม่มี priority"""
    return {
        "incident": case.incident.model_dump(mode="json"),
        "location": case.personal.location.model_dump() if case.personal.location else None,
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
    a, b = flat({**old["incident"], "location": old["location"]}), flat({**new["incident"], "location": new["location"]})
    return {k: {"old": a.get(k), "new": b.get(k)} for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
