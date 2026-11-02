"""
Agent 1 schemas — อ้างอิงเอกสาร "DSI490 — การออกแบบ Agent 1 ฉบับปรับปรุง" (เลขหัวข้อในโค้ดอ้างตามเอกสารนี้)

แบ่งตามสิทธิ์การเข้าถึง (หัวข้อ 6)
    Incident Data          เข้า LLM ได้                       IncidentData
    Personal Data          ห้ามเข้า LLM                       PersonalData
    Case Metadata          ควบคุมระบบ                         CaseStatus, IntakeStatus, IncidentPriority
    Conversation State     ควบคุมการสนทนา                     SlotName, SlotState
    Priority Internal      ตรวจสอบภายใน ไม่ส่งให้ Pipeline    RiskFacts, PriorityResult
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"
BANGKOK_TZ = timezone(timedelta(hours=7), name="Asia/Bangkok")


def now_bkk() -> datetime:
    return datetime.now(BANGKOK_TZ)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# 6.1 Incident Data
# ---------------------------------------------------------------------------

PositiveCount = int | None


class AffectedPeople(StrictModel):
    """null = ไม่มีจำนวนที่รายงาน ไม่มี total และไม่ใช้ 0"""

    children: PositiveCount = None
    elderly: PositiveCount = None
    disabled: PositiveCount = None
    pregnant_women: PositiveCount = None
    other: PositiveCount = None

    @field_validator("*")
    @classmethod
    def _positive(cls, v):
        if v is not None and (isinstance(v, bool) or v < 1):
            raise ValueError("จำนวนต้องเป็นจำนวนเต็มบวกหรือ null")
        return v

    def has_anyone(self) -> bool:
        return any(v is not None for v in self.model_dump().values())


AFFECTED_GROUPS = tuple(AffectedPeople.model_fields)
AFFECTED_LABELS_TH = {"children": "เด็ก", "elderly": "ผู้สูงอายุ", "disabled": "ผู้พิการ",
                      "pregnant_women": "หญิงตั้งครรภ์", "other": "ผู้ประสบภัยอื่น"}


class IncidentData(StrictModel):
    problem: str | None = None
    needs: str | None = None
    affected_people: AffectedPeople = Field(default_factory=AffectedPeople)


# ---------------------------------------------------------------------------
# 6.2 Personal Data — เก็บใน case_personal_data เท่านั้น
# ---------------------------------------------------------------------------


class Location(StrictModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class PersonalData(StrictModel):
    first_name: str | None = None
    last_name: str | None = None
    phone_number: str | None = None
    location: Location | None = None


# ---------------------------------------------------------------------------
# 6.3 Case Metadata
# ---------------------------------------------------------------------------


class CaseStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"  # เจ้าหน้าที่หลังบ้านปิดเคส


class IntakeStatus(str, Enum):
    COLLECTING = "collecting"
    AWAITING_USER_INPUT = "awaiting_user_input"  # ถามครบ 2 ครั้งทุก Slot ที่ขาดแล้ว
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    CONFIRMED = "confirmed"


class PriorityLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IncidentPriority(StrictModel):
    """ส่วนเดียวของ Priority ที่ Pipeline เห็น"""

    level: PriorityLevel | None = None
    reason: str = "ข้อมูลไม่เพียงพอสำหรับการประเมิน"


# ---------------------------------------------------------------------------
# 6.4 Conversation State + Slot Tracking (หัวข้อ 8)
# ---------------------------------------------------------------------------


class SlotName(str, Enum):
    PROBLEM = "problem"
    NEEDS = "needs"
    AFFECTED_PEOPLE = "affected_people"
    LOCATION = "location"
    FIRST_NAME = "first_name"
    LAST_NAME = "last_name"
    PHONE_NUMBER = "phone_number"


SLOT_LABELS_TH = {
    SlotName.PROBLEM: "ปัญหาที่พบ", SlotName.NEEDS: "ความช่วยเหลือที่ต้องการ",
    SlotName.AFFECTED_PEOPLE: "จำนวนผู้ประสบภัย", SlotName.LOCATION: "หมุดจุดเกิดเหตุ",
    SlotName.FIRST_NAME: "ชื่อ", SlotName.LAST_NAME: "นามสกุล", SlotName.PHONE_NUMBER: "เบอร์โทรศัพท์",
}
MAX_SLOT_ASKS = 2


class SlotStatus(str, Enum):
    MISSING = "missing"
    FILLED = "filled"
    ANSWERED_NONE = "answered_none"  # ใช้กับ needs เท่านั้น
    ANSWERED_UNKNOWN = "answered_unknown"  # ใช้กับ needs เท่านั้น


class SlotState(StrictModel):
    status: SlotStatus = SlotStatus.MISSING
    ask_count: int = Field(default=0, ge=0)
    asked_at: datetime | None = None
    answered_at: datetime | None = None
    evidence: str | None = None  # ไม่เก็บค่า PII ในช่องนี้

    @property
    def satisfied(self) -> bool:
        return self.status != SlotStatus.MISSING


class MessageIntent(str, Enum):
    INCIDENT = "incident"
    QUESTION = "question"
    EMOTIONAL = "emotional"
    POLICE = "police"  # แจ้งเหตุร้ายที่ต้องใช้ตำรวจ ใช้เลือกเบอร์ 191


class CaseIntent(str, Enum):
    SAME_CASE = "same_case"
    NEW_CASE = "new_case"
    RELOCATE = "relocate"
    UNCLEAR = "unclear"


# ---------------------------------------------------------------------------
# 6.5 Priority Internal Data (หัวข้อ 9)
# ---------------------------------------------------------------------------

Tri = Literal["true", "false", "unknown"]
RISK_FACTS = ("life_threat", "trapped", "water_rising", "unsafe_to_evacuate", "cannot_self_evacuate",
              "in_safe_place", "basic_needs_unmet", "essential_medication_risk")


class RiskFacts(StrictModel):
    life_threat: Tri = "unknown"
    trapped: Tri = "unknown"
    water_rising: Tri = "unknown"
    unsafe_to_evacuate: Tri = "unknown"
    cannot_self_evacuate: Tri = "unknown"
    in_safe_place: Tri = "unknown"
    basic_needs_unmet: Tri = "unknown"
    essential_medication_risk: Tri = "unknown"


class PriorityResult(StrictModel):
    priority: IncidentPriority
    rule_id: str
    ruleset_version: str


# ---------------------------------------------------------------------------
# Information Extractor output (หัวข้อ 8)
# ---------------------------------------------------------------------------


class Extraction(StrictModel):
    """ผลจาก Gemini ก่อนตรวจหลักฐาน ทุกค่าที่ไม่มี evidence ในข้อความจะถูกตัดทิ้ง"""

    intents: list[MessageIntent] = Field(default_factory=list)
    case_intent: CaseIntent = CaseIntent.SAME_CASE
    problem: str | None = None
    problem_evidence: str | None = None
    needs: str | None = None
    needs_evidence: str | None = None
    needs_answer: Literal["none", "unknown"] | None = None  # ผู้ใช้ตอบว่าไม่มี / ไม่ทราบ
    affected_people: AffectedPeople = Field(default_factory=AffectedPeople)
    affected_people_evidence: dict[str, str] = Field(default_factory=dict)
    count_unclear: bool = False  # เช่น "หลายคน" "แม่ที่แก่แล้ว 2 คน" → ถามยืนยันจำนวน
    risk_facts: RiskFacts = Field(default_factory=RiskFacts)
    risk_fact_evidence: dict[str, str] = Field(default_factory=dict)
    danger_signals: list[str] = Field(default_factory=list)
    question: str | None = None  # คำถามของผู้ใช้ ใช้ค้น Knowledge Base


# ---------------------------------------------------------------------------
# Turn output
# ---------------------------------------------------------------------------


class QuickReply(StrictModel):
    type: Literal["postback", "location"] = "postback"
    label: str = Field(max_length=20)
    data: str | None = None

    @model_validator(mode="after")
    def _postback_has_data(self) -> "QuickReply":
        if self.type == "postback" and not self.data:
            raise ValueError("ปุ่ม postback ต้องมี data")
        return self


class TurnOutput(StrictModel):
    output_text: str
    quick_replies: list[QuickReply] = Field(default_factory=list)
    case_id: str | None = None
    case_status: CaseStatus | None = None
    intake_status: IntakeStatus | None = None
    pending_question: str | None = None
    missing_slots: list[SlotName] = Field(default_factory=list)
    kb_doc_ids: list[str] = Field(default_factory=list)
