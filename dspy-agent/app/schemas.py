"""
    Incident Data          เข้า LLM ได้                       IncidentData
    Personal Data          ห้ามเข้า LLM                       PersonalData
    Case Metadata          ควบคุมระบบ                         CaseStatus, IntakeStatus
    Conversation State     ควบคุมการสนทนา                     SlotName, SlotState
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

SCHEMA_VERSION = "4.0"  # incident_priority: level + reason; Ruleset 1.1.1
BANGKOK_TZ = timezone(timedelta(hours=7), name="Asia/Bangkok")
_HELP_REQUEST = re.compile(r"ต้องการ(?!อะไร|ไหม)|อยาก(?:ให้ช่วย|ได้|ปรึกษา|อพยพ|กลับ)|ขอ(?:ความช่วยเหลือ|เรือ|อาหาร|น้ำ|ยา|คน|รถ|อพยพ|ปรึกษา)|ช่วย(?:พา|อพยพ|ส่ง|หา|ดู|แนะนำ|เรื่อง)|ปรึกษา")
_HELP_NEGATION = re.compile(r"(?:ไม่\s*|ไม่อยาก\s*|ไม่ต้องการ\s*|ไม่ได้)$")


def requested_help_from_text(text: str, *, answering: bool = False) -> str:
    """Recognize an explicit request/answer, never infer a need from a hazard."""
    text = text.strip()
    if (answering or re.search(r"ต้องการ|ให้ช่วย", text)) and re.search(r"ไม่รู้|ไม่ทราบ|ไม่แน่ใจ|คิดไม่ออก", text):
        return "unsure"
    if re.fullmatch(r"(?:ยัง)?(?:ไม่มี(?:อะไร)?|ไม่ต้อง(?:การ(?:อะไร|ความช่วยเหลือ|เพิ่ม(?:เติม)?)?)?|ไม่)(?:ครับ|ค่ะ|คะ)?", text):
        return "none" if answering or "ต้องการ" in text else "missing"
    for match in _HELP_REQUEST.finditer(text):
        if not _HELP_NEGATION.search(text[max(0, match.start()-10):match.start()]):
            return "known"
    if answering and "ไม่" not in text and re.search(r"เรือ|น้ำดื่ม|อาหาร|ยา|อพยพ|รถ|หมอ|แพทย์|ติดต่อ|ส่งต่อ", text):
        return "known"
    return "missing"


def requested_help_phrase(text: str) -> str:
    for match in _HELP_REQUEST.finditer(text):
        if not _HELP_NEGATION.search(text[max(0, match.start()-10):match.start()]):
            return text[match.start():].strip()
    return text.strip()


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
    affected_people: AffectedPeople = Field(default_factory=AffectedPeople)

    @model_validator(mode="before")
    @classmethod
    def _legacy_needs(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            needs = value.pop("needs", None)
            if needs:
                problem = value.get("problem") or ""
                value["problem"] = problem if needs in problem else " ".join(filter(None, (problem, needs)))
        return value


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


# ---------------------------------------------------------------------------
# 6.4 Conversation State + Slot Tracking (หัวข้อ 8)
# ---------------------------------------------------------------------------


class SlotName(str, Enum):
    PROBLEM = "problem"
    AFFECTED_PEOPLE = "affected_people"
    LOCATION = "location"
    FIRST_NAME = "first_name"
    LAST_NAME = "last_name"
    PHONE_NUMBER = "phone_number"


SLOT_LABELS_TH = {
    SlotName.PROBLEM: "ปัญหาและรายละเอียด",
    SlotName.AFFECTED_PEOPLE: "จำนวนผู้ประสบภัย", SlotName.LOCATION: "หมุดจุดเกิดเหตุ",
    SlotName.FIRST_NAME: "ชื่อ", SlotName.LAST_NAME: "นามสกุล", SlotName.PHONE_NUMBER: "เบอร์โทรศัพท์",
}
MAX_SLOT_ASKS = 2


class SlotStatus(str, Enum):
    MISSING = "missing"
    FILLED = "filled"


class SlotState(StrictModel):
    status: SlotStatus = SlotStatus.MISSING
    ask_count: int = Field(default=0, ge=0)
    filler_acks: int = Field(default=0, ge=0)  # จำนวนครั้งที่ตอบรับข้อความ "โอเค/เดี๋ยวนะ" ระหว่างรอช่องนี้
    asked_at: datetime | None = None
    answered_at: datetime | None = None
    evidence: str | None = None  # ไม่เก็บค่า PII ในช่องนี้

    @property
    def satisfied(self) -> bool:
        return self.status != SlotStatus.MISSING


class ConversationTurn(StrictModel):
    role: Literal["user", "assistant"]
    text: str = Field(max_length=600)


class ContextAnswer(StrictModel):
    question: str = Field(max_length=240)
    answer: str = Field(max_length=600)
    subject: str | None = None


class ConversationContext(StrictModel):
    # Internal only: not part of the confirmed incident or pipeline contract.
    turns: list[ConversationTurn] = Field(default_factory=list)
    last_support: str = ""
    answers: list[ContextAnswer] = Field(default_factory=list)
    asked_questions: list[str] = Field(default_factory=list)  # คำถามเติมบริบทที่ถามไปแล้ว ใช้กันถามซ้ำ

    @model_validator(mode="before")
    @classmethod
    def _legacy_context(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            for key in ("pending_followup", "asked_topics", "safety_focus", "medical_step", "medical_contact_status"):
                value.pop(key, None)
        return value


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
# Information Extractor output
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Priority รายเคส (Design Decisions)
# ---------------------------------------------------------------------------


class PriorityLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RiskFacts(StrictModel):
    """None means unknown, never false. Values require per-fact reporter evidence."""
    life_threat: StrictBool | None = None
    trapped: StrictBool | None = None
    water_rising: StrictBool | None = None
    unsafe_to_evacuate: StrictBool | None = None
    cannot_self_evacuate: StrictBool | None = None
    in_safe_place: StrictBool | None = None
    basic_needs_unmet: StrictBool | None = None
    essential_medication_risk: StrictBool | None = None


class CasePriority(StrictModel):
    """Internal assessment. Audit fields never enter the Pipeline payload."""
    level: PriorityLevel = PriorityLevel.MEDIUM
    reason: str
    rule_id: str
    ruleset_version: str = "1.1.1"
    evaluated_at: datetime

    @model_validator(mode="before")
    @classmethod
    def _legacy_priority(cls, value):
        if isinstance(value, dict) and "reasons" in value:
            return {"level": value.get("level"), "reason": "; ".join(value["reasons"]),
                    "rule_id": "legacy", "ruleset_version": value.get("rules_version", "legacy"),
                    "evaluated_at": value["evaluated_at"]}
        return value

    def pipeline_payload(self) -> dict:
        if self.rule_id == "P1":
            raise ValueError("Priority is pending assessment; do not export this draft")
        return {"level": self.level.value, "reason": self.reason}


class Extraction(StrictModel):
    """ผลจาก Gemini ก่อนตรวจหลักฐาน ทุกค่าที่ไม่มี evidence ในข้อความจะถูกตัดทิ้ง"""

    risk_facts: RiskFacts = Field(default_factory=RiskFacts)
    risk_fact_evidence: dict[str, str] = Field(default_factory=dict)
    needs_answer: Literal["none", "requested", "unknown"] = "unknown"

    intents: list[MessageIntent] = Field(default_factory=list)
    case_intent: CaseIntent = CaseIntent.SAME_CASE
    problem: str | None = None
    problem_evidence: str | None = None
    affected_people: AffectedPeople = Field(default_factory=AffectedPeople)
    affected_people_evidence: dict[str, str] = Field(default_factory=dict)
    count_unclear: bool = False  # เช่น "หลายคน" "แม่ที่แก่แล้ว 2 คน" → ถามยืนยันจำนวน
    context_complete: bool = False
    context_missing: list[str] = Field(default_factory=list, max_length=6)
    context_question: str | None = Field(default=None, max_length=240)
    correction_acknowledgement: str | None = Field(default=None, max_length=160)
    help_status: Literal["missing", "known", "unsure", "none"] = "missing"
    help_evidence: str | None = None
    question: str | None = None  # คำถามของผู้ใช้ ใช้ค้น Knowledge Base

    @model_validator(mode="before")
    @classmethod
    def _legacy_extraction(cls, value):
        if not isinstance(value, dict):
            return value
        value = dict(value)
        needs, evidence = value.pop("needs", None), value.pop("needs_evidence", None)
        if needs and evidence:
            value["problem"] = " ".join(filter(None, (value.get("problem"), needs)))
            value["problem_evidence"] = " ".join(filter(None, (value.get("problem_evidence"), evidence)))
        for key in ("urgent_flags", "need_categories", "situation_tags", "vulnerable_types", "water_descriptor", "priority_evidence", "danger_signals"):
            value.pop(key, None)
        return value


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
