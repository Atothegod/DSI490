"""
Case State + Slot Manager + Summary Generator (หัวข้อ 5, 8, 10)

Case เป็นภาพรวมของเคสหนึ่งเคสที่ store.py โหลดจากและบันทึกลงหลายตาราง
ไม่เรียก LLM ทุกฟังก์ชันให้ผลแน่นอน
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from .schemas import (
    AFFECTED_GROUPS,
    AFFECTED_LABELS_TH,
    MAX_SLOT_ASKS,
    SLOT_LABELS_TH,
    CaseStatus,
    IncidentData,
    IncidentPriority,
    IntakeStatus,
    Location,
    PersonalData,
    PriorityLevel,
    RiskFacts,
    SlotName,
    SlotState,
    SlotStatus,
    StrictModel,
    now_bkk,
)

SLOT_ORDER = (SlotName.PROBLEM, SlotName.AFFECTED_PEOPLE, SlotName.NEEDS, SlotName.LOCATION,
              SlotName.FIRST_NAME, SlotName.LAST_NAME, SlotName.PHONE_NUMBER)


class Case(StrictModel):
    case_id: str
    user_ref: str
    case_status: CaseStatus = CaseStatus.ACTIVE
    intake_status: IntakeStatus = IntakeStatus.COLLECTING
    priority: IncidentPriority = Field(default_factory=IncidentPriority)
    priority_rule_id: str = "P1"
    current_version: int = 0  # 0 = ยังไม่มี Confirmed Version
    draft_rev: int = 0  # เพิ่มทุกครั้งที่ Draft เปลี่ยน ใช้กันการกดปุ่มของสรุปเก่า
    created_at: datetime = Field(default_factory=now_bkk)
    updated_at: datetime = Field(default_factory=now_bkk)
    confirmed_at: datetime | None = None

    # Draft (case_drafts)
    incident: IncidentData = Field(default_factory=IncidentData)
    risk_facts: RiskFacts = Field(default_factory=RiskFacts)
    risk_fact_evidence: dict[str, str] = Field(default_factory=dict)
    danger_signals: list[str] = Field(default_factory=list)
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
    conversation_summary: str = ""

    @property
    def is_draft_open(self) -> bool:
        return self.current_version == 0 or self.editing

    @property
    def critical(self) -> bool:
        return self.priority.level == PriorityLevel.CRITICAL

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
        SlotName.PROBLEM: bool(case.incident.problem),
        SlotName.AFFECTED_PEOPLE: case.incident.affected_people.has_anyone(),
        SlotName.LOCATION: case.personal.location is not None,
        SlotName.FIRST_NAME: bool(case.personal.first_name),
        SlotName.LAST_NAME: bool(case.personal.last_name),
        SlotName.PHONE_NUMBER: bool(case.personal.phone_number),
    }
    needs = case.slots[SlotName.NEEDS]
    if case.incident.needs:
        filled[SlotName.NEEDS] = True
    elif needs.status in (SlotStatus.ANSWERED_NONE, SlotStatus.ANSWERED_UNKNOWN):
        filled[SlotName.NEEDS] = None  # ตอบแล้ว คงสถานะเดิม
    else:
        filled[SlotName.NEEDS] = False
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
    """Slot บังคับที่ยังขาดและถามไม่ถึง 2 ครั้ง เคส Critical ที่ยังไม่มีพิกัดขอหมุดก่อน
    prefer ใช้ถามยืนยันทันทีเมื่อข้อมูลกำกวม เช่น จำนวนผู้ประสบภัย"""
    candidates = [s for s in missing_slots(case) if case.slots[s].ask_count < MAX_SLOT_ASKS]
    if case.critical and SlotName.LOCATION in candidates:
        return SlotName.LOCATION
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


def needs_text(case: Case) -> str:
    status = case.slots[SlotName.NEEDS].status
    if case.incident.needs:
        return case.incident.needs
    if status == SlotStatus.ANSWERED_NONE:
        return "ไม่ต้องการความช่วยเหลือเพิ่มเติม"
    return "ยังไม่ทราบ"


def summary_text(case: Case) -> str:
    p = case.personal
    lines = [
        "สรุปข้อมูลการแจ้งเหตุ",
        f"1. ปัญหาที่พบ: {case.incident.problem}",
        f"2. ความช่วยเหลือที่ต้องการ: {needs_text(case)}",
        f"3. จำนวนผู้ประสบภัย: {people_text(case)}",
        "4. จุดเกิดเหตุ: ได้รับหมุดจุดเกิดเหตุผ่าน LINE แล้ว",
        f"5. ข้อมูลผู้ติดต่อ: {p.first_name} {p.last_name} · {mask_phone(p.phone_number or '')}",
    ]
    if case.address_masked:
        lines.append("หมายเหตุ: ที่อยู่ที่พิมพ์มาไม่ได้บันทึกเป็นข้อมูลเคส ระบบใช้หมุดจุดเกิดเหตุแทน")
    lines.append("กรุณาตรวจสอบข้อมูล หากถูกต้องกด [ยืนยันข้อมูล] หากต้องการเปลี่ยนกด [แก้ไขข้อมูล]")
    return "\n".join(lines)


def missing_notice(slots: list[SlotName]) -> str:
    names = ", ".join(SLOT_LABELS_TH[s] for s in slots)
    return (f"ข้อมูลที่แจ้งไว้เกือบครบแล้วครับ ยังขาดข้อมูล {names} ซึ่งจำเป็นสำหรับการยืนยันการแจ้งเหตุ\n\n"
            "เมื่อสะดวก สามารถส่งข้อมูลเพิ่มเติมได้เลยครับ ระบบจะเก็บข้อมูลที่แจ้งไว้ก่อนหน้าให้ ไม่จำเป็นต้องเริ่มใหม่")


def version_snapshot(case: Case) -> dict:
    """Confirmed Snapshot — ส่วนที่ Pipeline เห็น (ไม่มี Risk Facts หรือ rule_id)"""
    return {
        "incident": case.incident.model_dump(mode="json"),
        "location": case.personal.location.model_dump() if case.personal.location else None,
        "incident_priority": case.priority.model_dump(mode="json"),
        "needs_status": case.slots[SlotName.NEEDS].status.value,
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
