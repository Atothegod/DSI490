"""
DSI490 — Agent 1 (Unified Flood Intake & Support) schemas.

อ้างอิง: DSI490 — Design Decisions v1.1
ไฟล์นี้เป็น "แหล่งความจริงเดียว" ของโครงสร้างข้อมูล ทั้ง Notebook และ FastAPI ต้อง import จากที่นี่

โครงสร้างไฟล์
    1. Enums                     ค่าคงที่ที่ตกลงกันแล้ว
    2. Tracked fields            field ที่มีสถานะ known / unknown / declined / unanswered / not_asked
    3. Incident State            แหล่งความจริงของเหตุการณ์ (ข้อ 2, 3.1, 3.2, 6)
    4. Contact และพิกัด          ข้อมูลติดต่อ และการอนุญาตส่งพิกัด เก็บแยก (ข้อ 2.6, 6.10)
    5. Memory                    Pending Question, Conversation, Message, Summary, Context 5 ชั้น (ข้อ 3.3–3.4)
    6. Handoff                   รายงานและ event ที่ส่งให้ Agent 2 พร้อม ACK (ข้อ 1.3, 5, 6.7)
    7. RAG                       Knowledge chunk, Hotline (ข้อ 4)
    8. Agent I/O                 ผลลัพธ์ที่ LLM ส่งออกมา และผลลัพธ์ต่อ turn

v1.1 เพิ่ม: UrgentFlagRecord, AreaRef, ServiceScope, CloseReason.OUT_OF_SERVICE_AREA, HotlineNotice,
FieldChangeType / FieldChange, ask_count + FieldStatus.UNANSWERED, LocationShareRequest / LocationShareConsent

หลักการ
    - Schema นี้ไม่มี business logic (merge, matching, retention) ส่วนนั้นอยู่ใน Backend Rules
    - Agent 1 ไม่มี field severity ทุกที่ เพราะการประเมินเป็นหน้าที่ของ Agent 2
    - เวลาทั้งหมดต้องมี timezone (แนะนำ Asia/Bangkok)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Annotated, Any, Generic, Literal, TypeVar, Union
from uuid import uuid4

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

SCHEMA_VERSION = "1.1"
BANGKOK_TZ = timezone(timedelta(hours=7), name="Asia/Bangkok")


def now_bkk() -> datetime:
    return datetime.now(BANGKOK_TZ)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class StrictModel(BaseModel):
    """ห้ามมี field แปลกปลอม เพื่อให้ผลลัพธ์จาก LLM ที่ผิดรูปแบบถูกปฏิเสธทันที"""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)


# ---------------------------------------------------------------------------
# 1. Enums
# ---------------------------------------------------------------------------


class FieldStatus(str, Enum):
    """ข้อ 2.2 — สถานะของแต่ละ field"""

    NOT_ASKED = "not_asked"  # ยังไม่ได้ถาม หรือถามยังไม่ครบจำนวนครั้ง → ถามได้ตาม ask_count
    KNOWN = "known"  # ผู้ใช้ให้ข้อมูลแล้ว
    UNKNOWN = "unknown"  # ผู้ใช้ตอบว่าไม่ทราบ → ห้ามถามซ้ำ
    DECLINED = "declined"  # ผู้ใช้ไม่ประสงค์ให้ข้อมูล → ห้ามถามซ้ำ
    UNANSWERED = "unanswered"  # ถามครบจำนวนครั้งแล้วไม่ได้คำตอบ ไม่เท่ากับไม่ทราบ (ข้อ 6.9)


class FieldChangeType(str, Enum):
    """ข้อ 6.7 — ประเภทการเปลี่ยนค่า ตัดสินโดย Backend Rules ไม่ใช่ LLM"""

    INITIAL = "initial"  # field ยังไม่มีค่า
    CONFIRMATION = "confirmation"  # ค่าใหม่เท่ากับค่าเดิมหลังจัดรูปแบบ → ไม่ส่ง event
    CORRECTION = "correction"  # Gemini และ keyword ยืนยันการแก้ตรงกัน
    UPDATE = "update"  # ค่าใหม่ต่างจากเดิมในกรณีอื่นทั้งหมด ไม่สรุปว่าแย่ลงหรือดีขึ้น


class IncidentStatus(str, Enum):
    """ข้อ 3.1 — lifecycle"""

    DRAFT = "draft"
    SUBMITTED = "submitted"
    CLOSED = "closed"
    ABANDONED = "abandoned"


class CloseReason(str, Enum):
    USER_RESOLVED = "user_resolved"
    TIMEOUT = "timeout"  # ไม่ได้แปลว่าปลอดภัย
    USER_WITHDRAWN = "user_withdrawn"
    OUT_OF_SERVICE_AREA = "out_of_service_area"  # ปิดจาก draft เท่านั้น ไม่ส่ง event (ข้อ 6.5)


class ServiceScope(str, Enum):
    """ข้อ 6.5 — อยู่ในขอบเขตบริการ (สิงห์บุรี) หรือไม่"""

    IN_PROVINCE = "in_province"
    OUT_OF_PROVINCE = "out_of_province"  # ผู้ใช้ระบุชัดว่าอยู่นอกจังหวัด
    UNVERIFIED = "unverified"  # ยังยืนยันพื้นที่ไม่ได้ ห้ามถือว่านอกจังหวัด


class AreaLevel(str, Enum):
    """ระดับพื้นที่ตามที่ผู้ใช้ระบุ Gemini ห้ามเดาระดับเอง"""

    VILLAGE = "village"
    TAMBON = "tambon"
    AMPHOE = "amphoe"
    PROVINCE = "province"
    UNSPECIFIED = "unspecified"


class AreaMatchStatus(str, Enum):
    """ข้อ 6.4 — ผลการจับคู่ชื่อพื้นที่กับข้อมูลอ้างอิงสิงห์บุรี"""

    MATCHED = "matched"  # ตรงเพียงแห่งเดียว มีรหัสพื้นที่
    AMBIGUOUS = "ambiguous"  # ตรงหลายแห่ง ต้องถามผู้ใช้
    NOT_FOUND = "not_found"  # ไม่พบ ต้องขอยืนยัน
    UNVERIFIED = "unverified"  # ถามยืนยันครบแล้วยังไม่พบ รับตามที่แจ้ง
    OUT_OF_PROVINCE = "out_of_province"  # ผู้ใช้ระบุชัดว่าอยู่นอกจังหวัด


class UrgentFlagSource(str, Enum):
    """ข้อ 6.3 — แหล่งที่พบ urgent flag"""

    KEYWORD = "keyword"
    GEMINI = "gemini"


class MatchingStatus(str, Enum):
    """ข้อ 3.2"""

    NEW = "new"
    UNCONFIRMED = "unconfirmed"
    LINKED = "linked"


class Relation(str, Enum):
    SAME_INCIDENT = "same_incident"
    RELATED = "related"


class UrgentFlag(str, Enum):
    """ข้อ 1.2 — ข้อเท็จจริงที่ผู้ใช้บอกตรง ๆ ไม่ใช่การประเมินความรุนแรง"""

    PEOPLE_TRAPPED = "people_trapped"
    INJURY_REPORTED = "injury_reported"
    HELP_REQUESTED = "help_requested"


class Need(str, Enum):
    """ข้อ 2.3 — เลือกได้หลายค่า"""

    RESCUE = "rescue"
    EVACUATION = "evacuation"
    MEDICAL = "medical"
    FOOD_WATER = "food_water"
    SUPPLIES = "supplies"
    INFO_ONLY = "info_only"  # แค่แจ้งสถานการณ์ ถือว่าตอบ needs แล้ว
    OTHER = "other"


# needs กลุ่มนี้ทำให้ต้องถามจำนวนคน (ข้อ 2.4)
ASSISTANCE_NEEDS: frozenset[Need] = frozenset({Need.RESCUE, Need.EVACUATION, Need.MEDICAL})
# needs กลุ่มนี้ทำให้ขอพิกัด และส่งพิกัดให้ Agent 2 ได้โดยไม่ต้องมี consent (ข้อ 2.6, 6.10)
COORDINATE_NEEDS: frozenset[Need] = frozenset({Need.RESCUE, Need.EVACUATION})


class SituationTag(str, Enum):
    """ข้อ 2.3 — หมวดของสถานการณ์ เพิ่มค่าได้โดยไม่เปลี่ยน schema_version หลัก"""

    WATER_IN_HOUSE = "water_in_house"
    AREA_FLOODED = "area_flooded"  # น้ำท่วมพื้นที่ เช่น วัด ตลาด ทุ่งนา ถนนในชุมชน
    ROAD_BLOCKED = "road_blocked"
    ISOLATED = "isolated"  # ถูกตัดขาด ออกจากพื้นที่ไม่ได้
    POWER_OUTAGE = "power_outage"
    WATER_RISING = "water_rising"
    STRONG_CURRENT = "strong_current"
    PROPERTY_DAMAGE = "property_damage"
    LIVESTOCK_AFFECTED = "livestock_affected"
    OTHER = "other"


class WaterDescriptor(str, Enum):
    """ข้อ 2.3 — ระดับน้ำแบบที่คนตอบได้ง่าย Agent 1 ไม่แปลงเป็นเซนติเมตรเอง"""

    ANKLE = "ankle"
    KNEE = "knee"
    WAIST = "waist"
    CHEST = "chest"
    NECK = "neck"
    ABOVE_HEAD = "above_head"
    FIRST_FLOOR = "first_floor"  # ท่วมชั้นล่างของบ้าน
    OTHER = "other"


class VulnerableType(str, Enum):
    """ข้อ 2.4"""

    ELDERLY = "elderly"
    CHILD = "child"
    BEDRIDDEN = "bedridden"
    DISABLED = "disabled"
    PREGNANT = "pregnant"
    CHRONIC_ILLNESS = "chronic_illness"
    OTHER = "other"


class Intent(str, Enum):
    """Intent ของข้อความ ข้อความเดียวมีได้หลาย intent"""

    REPORT_FLOOD = "report_flood"
    REQUEST_HELP = "request_help"
    ASK_INFORMATION = "ask_information"
    UPDATE_INCIDENT = "update_incident"
    CORRECT_INFORMATION = "correct_information"
    CONFIRM_REPORT = "confirm_report"  # ผู้ใช้ยืนยันสรุปรายงาน
    RESOLVE_INCIDENT = "resolve_incident"  # น้ำลดแล้ว / ย้ายออกแล้ว
    WITHDRAW_INCIDENT = "withdraw_incident"  # แจ้งผิด / ซ้ำ
    CHECK_STATUS = "check_status"  # ตอบได้เฉพาะสถานะการส่ง (ข้อ 1.3)
    DELETE_MY_DATA = "delete_my_data"
    GENERAL_CONVERSATION = "general_conversation"


class NextAction(str, Enum):
    """
    สิ่งที่ Backend ต้องทำหลังจบ turn

    เปลี่ยนจาก Project Handoff ข้อ 11: ตัด `escalate` และ `request_human_review`
    เพราะ Agent 1 ไม่ประเมิน Severity เหตุเร่งด่วนใช้ submit_report ทันทีตาม urgent_flags
    """

    ASK_FOLLOW_UP = "ask_follow_up"
    ANSWER_QUESTION = "answer_question"
    CONFIRM_REPORT = "confirm_report"  # แสดงสรุปให้ผู้ใช้ยืนยัน
    SUBMIT_REPORT = "submit_report"  # ส่ง incident_submitted
    SEND_UPDATE = "send_update"  # ส่ง incident_updated
    ASK_INCIDENT_MATCH = "ask_incident_match"  # ถามว่าเหตุเดิมหรือเหตุใหม่
    LINK_INCIDENT = "link_incident"  # ส่ง incident_linked
    CLOSE_INCIDENT = "close_incident"  # ส่ง incident_closed
    DELETE_USER_DATA = "delete_user_data"
    NONE = "none"


class EventType(str, Enum):
    """ข้อ 5"""

    INCIDENT_SUBMITTED = "incident_submitted"
    INCIDENT_UPDATED = "incident_updated"
    INCIDENT_LINKED = "incident_linked"
    INCIDENT_CLOSED = "incident_closed"
    DATA_DELETION_REQUESTED = "data_deletion_requested"


class DeliveryStatus(str, Enum):
    """สถานะการส่ง event ไป Agent 2 ใช้ตอบ check_status (ข้อ 1.3–1.4)"""

    PENDING = "pending"  # บันทึกแล้ว รอส่ง
    SENT = "sent"  # ส่งแล้ว รอ ACK
    ACKED = "acked"  # Agent 2 รับและบันทึกสำเร็จ
    FAILED = "failed"  # retry ครบแล้วยังไม่สำเร็จ


class MessageRole(str, Enum):
    USER = "user"
    AGENT = "agent"


class UserStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"  # unfollow ไม่ใช่การลบ (ข้อ 3.5)


class KBCategory(str, Enum):
    A = "A"  # ความรู้คงที่
    B = "B"  # เปลี่ยนตามเวลา ต้องมี valid_until


class KBTopic(str, Enum):
    PREPARATION = "preparation"
    SAFETY = "safety"
    HEALTH = "health"
    RECOVERY = "recovery"
    CONTACT = "contact"


class ReviewStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    NEEDS_EDIT = "needs_edit"
    APPROVED = "approved"
    REJECTED = "rejected"


class IncidentFieldName(str, Enum):
    """ชื่อ field ที่ Agent ถามได้ ใช้ใน PendingQuestion และ missing_fields"""

    LOCATION_AREA = "location_area"
    LOCATION_LANDMARK = "location_landmark"
    LOCATION_COORDINATES = "location_coordinates"
    SITUATION = "situation"
    NEEDS = "needs"
    AFFECTED_PEOPLE = "affected_people"
    VULNERABLE_PEOPLE = "vulnerable_people"
    WATER_LEVEL = "water_level"
    CONTACT = "contact"


# ข้อ 2.1 — field บังคับ และ field ที่ต้องถามแต่ตอบไม่ทราบได้
REQUIRED_FIELDS: tuple[IncidentFieldName, ...] = (
    IncidentFieldName.LOCATION_AREA,
    IncidentFieldName.SITUATION,
    IncidentFieldName.NEEDS,
)
# ชื่อเดิมคงไว้เพื่อไม่ให้ Notebook พัง: field ที่ "ต้องถาม แต่ตอบไม่ทราบได้" (ไม่ได้แปลว่าถามครั้งเดียว)
ASK_ONCE_FIELDS: tuple[IncidentFieldName, ...] = (
    IncidentFieldName.AFFECTED_PEOPLE,
    IncidentFieldName.VULNERABLE_PEOPLE,
)

# ข้อ 6.9 — จำนวนครั้งสูงสุดที่ถามได้ต่อ field ก่อนตั้งเป็น unanswered
DEFAULT_MAX_ASKS = 2
MAX_ASKS: dict[IncidentFieldName, int] = {
    IncidentFieldName.WATER_LEVEL: 1,
    IncidentFieldName.CONTACT: 1,
    IncidentFieldName.LOCATION_COORDINATES: 1,
}


def max_asks(field: IncidentFieldName) -> int:
    return MAX_ASKS.get(field, DEFAULT_MAX_ASKS)


# ---------------------------------------------------------------------------
# 2. Tracked fields
# ---------------------------------------------------------------------------

T = TypeVar("T")


class FieldRevision(StrictModel, Generic[T]):
    """
    ค่าก่อนหน้าของ field (Project Handoff ข้อ 8.7)
    change_type บอกว่าค่าที่มาแทนค่านี้เป็นการเปลี่ยนแบบใด (ข้อ 6.7)
    """

    status: FieldStatus
    value: T | None = None
    raw: str | None = None
    reported_at: AwareDatetime | None = None  # เวลาที่ค่านี้ถูกรายงาน
    changed_at: AwareDatetime  # เวลาที่ค่านี้ถูกแทนที่
    change_type: FieldChangeType
    source_message_id: str | None = None


class TrackedField(StrictModel, Generic[T]):
    """
    field ที่รู้สถานะของตัวเอง

    กฎ: status = known ต้องมี value, status อื่นต้องไม่มี value
        status = unanswered ต้องเคยถามอย่างน้อย 1 ครั้ง
    """

    status: FieldStatus = FieldStatus.NOT_ASKED
    value: T | None = None
    raw: str | None = None  # ข้อความดิบที่ผู้ใช้พิมพ์ ใช้ตรวจย้อนหลัง
    updated_at: AwareDatetime | None = None
    source_message_id: str | None = None
    ask_count: int = Field(default=0, ge=0)  # ข้อ 6.9
    last_asked_at: AwareDatetime | None = None
    history: list[FieldRevision[T]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _value_matches_status(self) -> "TrackedField[T]":
        if self.status == FieldStatus.KNOWN and self.value is None:
            raise ValueError("status 'known' ต้องมี value")
        if self.status != FieldStatus.KNOWN and self.value is not None:
            raise ValueError(f"status '{self.status.value}' ต้องไม่มี value")
        if self.status == FieldStatus.UNANSWERED and self.ask_count < 1:
            raise ValueError("status 'unanswered' ต้องเคยถามอย่างน้อย 1 ครั้ง")
        if self.ask_count > 0 and self.last_asked_at is None:
            raise ValueError("ask_count มากกว่า 0 ต้องมี last_asked_at")
        return self

    @property
    def can_ask(self) -> bool:
        """สถานะยังเปิดให้ถาม — จำนวนครั้งที่เหลือตรวจด้วย max_asks() ใน Backend Rules"""
        return self.status == FieldStatus.NOT_ASKED


class Coordinates(StrictModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    source: Literal["line_location", "user_text"] = "line_location"
    message_id: str | None = None  # ข้อความที่ส่งพิกัด ใช้ผูกกับคำขอพิกัด (ข้อ 6.10)


class AreaMention(StrictModel):
    """สิ่งที่ Gemini ดึงได้: ชื่อและระดับพื้นที่ตามคำผู้ใช้เท่านั้น ห้ามเติมระดับเอง (ข้อ 6.4)"""

    name: str = Field(min_length=1)
    level: AreaLevel = AreaLevel.UNSPECIFIED


class AreaCandidate(StrictModel):
    tambon_code: str | None = None
    tambon: str | None = None
    amphoe_code: str
    amphoe: str


class AreaRef(StrictModel):
    """
    ข้อ 6.4 — พื้นที่หลังจับคู่กับข้อมูลอ้างอิง เก็บข้อความเดิมของผู้ใช้เสมอ
    รหัสและชื่อมาตรฐานมาจากข้อมูลอ้างอิงที่ยืนยันแล้วเท่านั้น
    """

    raw: str = Field(min_length=1)
    stated_level: AreaLevel = AreaLevel.UNSPECIFIED
    match_status: AreaMatchStatus
    tambon_code: str | None = None
    tambon: str | None = None
    amphoe_code: str | None = None
    amphoe: str | None = None
    province_code: str | None = None
    province: str | None = None
    candidates: list[AreaCandidate] = Field(default_factory=list)  # ใช้เมื่อ ambiguous

    @model_validator(mode="after")
    def _match_rules(self) -> "AreaRef":
        has_code = bool(self.tambon_code or self.amphoe_code)
        if self.match_status == AreaMatchStatus.MATCHED and not has_code:
            raise ValueError("matched ต้องมีรหัสตำบลหรืออำเภอจากข้อมูลอ้างอิง")
        if self.match_status == AreaMatchStatus.AMBIGUOUS:
            if len(self.candidates) < 2:
                raise ValueError("ambiguous ต้องมีตัวเลือกอย่างน้อย 2 แห่ง")
            if has_code:
                raise ValueError("ambiguous ห้ามเลือกรหัสพื้นที่เอง")
        elif self.candidates:
            raise ValueError("candidates ใช้ได้เฉพาะ ambiguous")
        if self.match_status in (AreaMatchStatus.NOT_FOUND, AreaMatchStatus.UNVERIFIED) and has_code:
            raise ValueError(f"{self.match_status.value} ต้องไม่มีรหัสพื้นที่")
        return self

    @property
    def area_code(self) -> str | None:
        """รหัสที่ใช้จับคู่เหตุเดิม (ข้อ 6.4)"""
        return self.tambon_code or self.amphoe_code


class Situation(StrictModel):
    text: str = Field(min_length=1, description="ข้อความดิบของผู้ใช้ (ปิดบังเบอร์โทรแล้ว)")
    tags: list[SituationTag] = Field(default_factory=list)


class WaterLevel(StrictModel):
    """ข้อ 2.3 — ต้องมี descriptor หรือ cm อย่างน้อยหนึ่งค่า"""

    raw: str = Field(min_length=1)
    descriptor: WaterDescriptor | None = None
    cm: float | None = Field(default=None, ge=0, le=1000)

    @model_validator(mode="after")
    def _has_some_value(self) -> "WaterLevel":
        if self.descriptor is None and self.cm is None:
            raise ValueError("water_level ต้องมี descriptor หรือ cm")
        return self


class AffectedPeople(StrictModel):
    """
    ข้อ 2.4 — present ตอบ มี/ไม่มี, count ถามต่อเมื่อขอความช่วยเหลือ
    count = None หมายถึงยังไม่ได้ถามจำนวน หรือผู้ใช้ไม่ทราบจำนวน
    """

    present: bool
    count: int | None = Field(default=None, ge=0)
    is_approximate: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> "AffectedPeople":
        if not self.present and self.count not in (None, 0):
            raise ValueError("present=False แต่มี count มากกว่า 0")
        return self


class VulnerableGroup(StrictModel):
    type: VulnerableType
    count: int | None = Field(default=None, ge=1)
    is_approximate: bool = False
    note: str | None = None  # เช่น "ต้องใช้ออกซิเจน" ถ้าผู้ใช้บอกเอง


class VulnerablePeople(StrictModel):
    present: bool
    groups: list[VulnerableGroup] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> "VulnerablePeople":
        if not self.present and self.groups:
            raise ValueError("present=False แต่มี groups")
        return self


# ---------------------------------------------------------------------------
# 3. Incident State — แหล่งความจริงของเหตุการณ์
# ---------------------------------------------------------------------------


class UrgentFlagRecord(StrictModel):
    """ข้อ 6.3 — flag หนึ่งชนิดใน incident พร้อมแหล่งที่พบ (keyword OR Gemini)"""

    flag: UrgentFlag
    sources: list[UrgentFlagSource] = Field(min_length=1)
    detected_at: AwareDatetime
    message_id: str | None = None

    @model_validator(mode="after")
    def _unique_sources(self) -> "UrgentFlagRecord":
        if len(set(self.sources)) != len(self.sources):
            raise ValueError("sources ต้องไม่ซ้ำกัน")
        return self


class HotlineNotice(StrictModel):
    """
    ข้อ 6.6 — บันทึกหลังส่งข้อความที่มีสายด่วนสำเร็จแล้วเท่านั้น
    แยกจาก urgent_flags เพราะ "ตรวจพบ" กับ "แจ้งผู้ใช้สำเร็จ" เป็นคนละสถานะ
    """

    reason: Literal["new_flags", "user_asked"]
    flags: list[UrgentFlag] = Field(default_factory=list)  # flag ที่แจ้งในข้อความนี้ (ว่างได้ถ้าผู้ใช้ถามเอง)
    incident_id: str | None = None  # ผู้ใช้ถามช่องทางติดต่อโดยไม่มี incident ได้
    message_id: str  # ข้อความตอบกลับที่ส่งสำเร็จ
    notified_at: AwareDatetime

    @model_validator(mode="after")
    def _reason_rules(self) -> "HotlineNotice":
        if self.reason == "new_flags" and (not self.flags or self.incident_id is None):
            raise ValueError("new_flags ต้องระบุ flags และ incident_id")
        if len(set(self.flags)) != len(self.flags):
            raise ValueError("flags ต้องไม่ซ้ำกัน")
        return self


class IncidentState(StrictModel):
    """
    State ภายใน Agent 1 (ชั้นที่ 1 ของ Context)
    ไม่ส่งออกตรง ๆ — รายงานที่ส่ง Agent 2 สร้างจาก IncidentReport.from_state() ใน Backend Rules
    """

    incident_id: str = Field(default_factory=lambda: new_id("inc"))
    internal_user_id: str
    conversation_id: str
    schema_version: str = SCHEMA_VERSION

    status: IncidentStatus = IncidentStatus.DRAFT
    incident_version: int = Field(default=1, ge=1)  # เพิ่มทุกครั้งที่ส่ง event ไป Agent 2
    matching_status: MatchingStatus = MatchingStatus.NEW
    possible_related_incident_ids: list[str] = Field(default_factory=list)
    related_incident_id: str | None = None  # เหตุเดิมที่ปิดไปแล้ว (ข้อ 3.1)
    close_reason: CloseReason | None = None
    service_scope: ServiceScope = ServiceScope.UNVERIFIED  # ข้อ 6.5

    # ข้อมูลเหตุการณ์ (ข้อ 2.1)
    location_area: TrackedField[AreaRef] = Field(default_factory=TrackedField[AreaRef])
    location_landmark: TrackedField[str] = Field(default_factory=TrackedField[str])
    location_coordinates: TrackedField[Coordinates] = Field(default_factory=TrackedField[Coordinates])
    situation: TrackedField[Situation] = Field(default_factory=TrackedField[Situation])
    needs: TrackedField[list[Need]] = Field(default_factory=TrackedField[list[Need]])
    affected_people: TrackedField[AffectedPeople] = Field(default_factory=TrackedField[AffectedPeople])
    vulnerable_people: TrackedField[VulnerablePeople] = Field(default_factory=TrackedField[VulnerablePeople])
    water_level: TrackedField[WaterLevel] = Field(default_factory=TrackedField[WaterLevel])
    contact_ask: TrackedField[bool] = Field(default_factory=TrackedField[bool])  # เก็บแค่สถานะการถาม เบอร์อยู่ใน ContactRecord

    urgent_flags: list[UrgentFlagRecord] = Field(default_factory=list)  # ข้อ 6.3
    hotline_notices: list[HotlineNotice] = Field(default_factory=list)  # ข้อ 6.6
    contact_ref: str | None = None  # ชี้ไป ContactRecord ถ้ามี
    location_consent_id: str | None = None  # ชี้ไป LocationShareConsent ที่ยังมีผล (ข้อ 6.10)

    created_at: AwareDatetime = Field(default_factory=now_bkk)
    updated_at: AwareDatetime = Field(default_factory=now_bkk)
    last_activity_at: AwareDatetime = Field(default_factory=now_bkk)  # นับ timeout (ข้อ 3.1)
    submitted_at: AwareDatetime | None = None
    closed_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def _lifecycle_consistent(self) -> "IncidentState":
        if self.status == IncidentStatus.CLOSED:
            if self.close_reason is None or self.closed_at is None:
                raise ValueError("closed ต้องมี close_reason และ closed_at")
        elif self.close_reason is not None:
            raise ValueError("close_reason ใช้ได้เฉพาะ status closed")

        if self.close_reason == CloseReason.OUT_OF_SERVICE_AREA:
            # ข้อ 6.5 — ปิดได้เฉพาะ draft ที่ไม่เคยส่ง ไม่มี flag และยืนยันแล้วว่านอกจังหวัด
            if self.submitted_at is not None:
                raise ValueError("out_of_service_area ใช้ได้เฉพาะ draft ที่ไม่เคยส่ง เหตุที่ส่งแล้วต้องส่ง incident_updated")
            if self.urgent_flags:
                raise ValueError("เหตุที่มี urgent_flags ห้ามปิดด้วย out_of_service_area")
            if self.service_scope != ServiceScope.OUT_OF_PROVINCE:
                raise ValueError("out_of_service_area ต้องมี service_scope = out_of_province")
        elif self.status in (IncidentStatus.SUBMITTED, IncidentStatus.CLOSED) and self.submitted_at is None:
            # abandoned / draft ไม่เคยถูกส่ง, closed ทั่วไปต้องเคยผ่าน submitted
            raise ValueError(f"status '{self.status.value}' ต้องมี submitted_at")

        if self.status == IncidentStatus.ABANDONED and self.urgent_flags:
            raise ValueError("draft ที่มี urgent_flags ต้องถูกส่งทันที จึงเป็น abandoned ไม่ได้")
        if self.matching_status == MatchingStatus.UNCONFIRMED and not self.possible_related_incident_ids:
            raise ValueError("unconfirmed ต้องมี possible_related_incident_ids")
        if self.needs.value is not None and len(self.needs.value) == 0:
            raise ValueError("needs ที่เป็น known ต้องมีอย่างน้อยหนึ่งค่า")

        flag_types = [r.flag for r in self.urgent_flags]
        if len(set(flag_types)) != len(flag_types):
            raise ValueError("urgent_flags หนึ่งชนิดต้องมีได้ record เดียว ให้รวมแหล่งที่มาใน sources")
        known_flags = set(flag_types)
        for notice in self.hotline_notices:
            if notice.incident_id not in (None, self.incident_id):
                raise ValueError("hotline_notices ต้องเป็นของ incident นี้")
            if not set(notice.flags) <= known_flags:
                raise ValueError("แจ้งสายด่วนสำหรับ flag ที่ไม่ได้ตรวจพบไม่ได้")

        area = self.location_area.value
        if area is not None and area.match_status == AreaMatchStatus.OUT_OF_PROVINCE \
                and self.service_scope != ServiceScope.OUT_OF_PROVINCE:
            raise ValueError("พื้นที่ out_of_province ต้องมี service_scope = out_of_province")
        return self

    def tracked_field(self, name: IncidentFieldName) -> TrackedField[Any]:
        """คืน TrackedField ตามชื่อ — contact คืนสถานะการถาม (contact_ask) ไม่ใช่เบอร์"""
        if name == IncidentFieldName.CONTACT:
            return self.contact_ask
        return getattr(self, name.value)

    @property
    def is_open(self) -> bool:
        return self.status in (IncidentStatus.DRAFT, IncidentStatus.SUBMITTED)

    @property
    def urgent_flag_types(self) -> set[UrgentFlag]:
        return {r.flag for r in self.urgent_flags}

    @property
    def notified_flag_types(self) -> set[UrgentFlag]:
        """flag ที่แจ้งสายด่วนสำเร็จแล้ว ใช้หา "flag ใหม่" (ข้อ 6.6)"""
        return {f for n in self.hotline_notices for f in n.flags}


# ---------------------------------------------------------------------------
# 4. Contact และพิกัด — เก็บแยก ไม่ส่ง Agent 2 ตรง ๆ (ข้อ 2.6, 6.10)
# ---------------------------------------------------------------------------


class LocationShareRequest(StrictModel):
    """
    ข้อ 6.10 — คำขอพิกัดที่ส่งให้ผู้ใช้ (ข้อความแจ้งวัตถุประสงค์) เก็บแยกจาก consent
    สร้างเมื่อส่งคำขอสำเร็จ consent สร้างภายหลังเมื่อผู้ใช้แชร์ตอบคำขอนี้เท่านั้น
    """

    request_id: str = Field(default_factory=lambda: new_id("lreq"))
    incident_id: str
    kind: Literal["ask_location", "confirm_unsolicited"]  # ขอแชร์ใหม่ / ขออนุญาตพิกัดที่ส่งมาก่อน
    notice_version: str  # จาก templates_th.json notice_versions
    purpose: str
    request_message_id: str
    requested_at: AwareDatetime
    status: Literal["pending", "granted", "declined", "skipped", "expired"] = "pending"
    response_message_id: str | None = None
    resolved_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def _resolution(self) -> "LocationShareRequest":
        if self.status == "pending":
            if self.response_message_id or self.resolved_at:
                raise ValueError("คำขอที่ยังรอคำตอบต้องไม่มีคำตอบหรือเวลาปิด")
        elif self.resolved_at is None:
            raise ValueError("คำขอที่ปิดแล้วต้องมี resolved_at")
        if self.status == "granted" and not self.response_message_id:
            raise ValueError("granted ต้องมี response_message_id")
        return self


class LocationShareConsent(StrictModel):
    """
    ข้อ 6.10 — การอนุญาตส่งพิกัดสำหรับ incident นี้โดยเฉพาะ แยกจาก needs และ urgent flag
    ไม่เปลี่ยน needs เป็น rescue และไม่หมายถึงมีเจ้าหน้าที่รับงาน
    """

    consent_id: str = Field(default_factory=lambda: new_id("lcon"))
    incident_id: str
    request_id: str
    purpose: str
    notice_version: str
    request_message_id: str
    response_message_id: str
    granted_at: AwareDatetime
    revoked_at: AwareDatetime | None = None
    revoke_reason: Literal["data_deletion", "user_revoked"] | None = None

    @model_validator(mode="after")
    def _consent_rules(self) -> "LocationShareConsent":
        if self.request_message_id == self.response_message_id:
            raise ValueError("คำขอและคำตอบต้องเป็นคนละข้อความ")
        if (self.revoked_at is None) != (self.revoke_reason is None):
            raise ValueError("revoked_at และ revoke_reason ต้องมีคู่กัน")
        return self

    @property
    def is_active(self) -> bool:
        return self.revoked_at is None

    @classmethod
    def from_request(cls, req: LocationShareRequest) -> "LocationShareConsent":
        """สร้าง consent ได้จากคำขอที่ granted แล้วเท่านั้น"""
        if req.status != "granted" or req.response_message_id is None or req.resolved_at is None:
            raise ValueError("สร้าง consent ได้เฉพาะจากคำขอที่ผู้ใช้อนุญาตแล้ว")
        return cls(
            incident_id=req.incident_id,
            request_id=req.request_id,
            purpose=req.purpose,
            notice_version=req.notice_version,
            request_message_id=req.request_message_id,
            response_message_id=req.response_message_id,
            granted_at=req.resolved_at,
        )


class ContactRecord(StrictModel):
    contact_ref: str = Field(default_factory=lambda: new_id("ctc"))
    internal_user_id: str
    incident_id: str
    phone: str | None = None
    name: str | None = None
    purpose: str  # ข้อความแจ้งวัตถุประสงค์ที่แสดงให้ผู้ใช้ (template)
    consent_at: AwareDatetime
    deleted_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def _has_something(self) -> "ContactRecord":
        if self.deleted_at is None and not (self.phone or self.name):
            raise ValueError("ContactRecord ต้องมี phone หรือ name")
        return self


class ContactAccessLog(StrictModel):
    """ทุกครั้งที่มีการเปิดดูข้อมูลติดต่อจริง"""

    contact_ref: str
    accessed_by: str
    accessed_at: AwareDatetime
    reason: str


# ---------------------------------------------------------------------------
# 5. Memory
# ---------------------------------------------------------------------------


class UserRecord(StrictModel):
    """LINE User ID อยู่ที่นี่ที่เดียว ห้ามส่งออกนอก Agent 1"""

    internal_user_id: str = Field(default_factory=lambda: new_id("usr"))
    line_user_id: str
    reporter_ref: str = Field(default_factory=lambda: new_id("rep"))  # ใช้กับ Agent 2
    status: UserStatus = UserStatus.ACTIVE
    created_at: AwareDatetime = Field(default_factory=now_bkk)
    # ข้อ 6.10 — ตั้งทันทีที่ขอลบข้อมูล ระหว่างนี้ห้ามแนบพิกัดทุกเงื่อนไข (รวมเงื่อนไข needs)
    deletion_requested_at: AwareDatetime | None = None

    @property
    def coordinates_forwarding_suspended(self) -> bool:
        return self.deletion_requested_at is not None


class PendingQuestion(StrictModel):
    """ข้อ 3.3 — คำถามที่ Agent ถามค้างไว้ ถามได้หลาย field ในข้อความเดียว"""

    incident_id: str
    field_names: list[IncidentFieldName] = Field(min_length=1)
    asked_at: AwareDatetime
    asked_message_id: str | None = None


class ConversationState(StrictModel):
    conversation_id: str = Field(default_factory=lambda: new_id("conv"))
    internal_user_id: str
    active_incident_id: str | None = None
    pending_question: PendingQuestion | None = None
    last_message_at: AwareDatetime | None = None


class Message(StrictModel):
    message_id: str = Field(default_factory=lambda: new_id("msg"))
    conversation_id: str
    role: MessageRole
    text: str
    created_at: AwareDatetime = Field(default_factory=now_bkk)
    incident_ids: list[str] = Field(default_factory=list)  # incident ที่ข้อความนี้เกี่ยวข้อง
    line_event_id: str | None = None  # ใช้กัน event ซ้ำจาก LINE


class ConversationSummary(StrictModel):
    """ข้อ 3.4 — rolling summary ห้ามมีชื่อ เบอร์ และพิกัด"""

    conversation_id: str
    text: str
    covers_until_message_id: str
    created_at: AwareDatetime = Field(default_factory=now_bkk)


class OpenIncidentBrief(StrictModel):
    """ชั้นที่ 3 — ข้อมูลสั้นของเหตุที่เปิดอยู่ ใช้จับคู่"""

    incident_id: str
    status: IncidentStatus
    location_area: str | None  # ชื่อสำหรับแสดง
    area_code: str | None = None  # รหัสพื้นที่ ใช้จับคู่เหตุเดิมเมื่อระบุได้ (ข้อ 6.4)
    created_at: AwareDatetime


class ContextBundle(StrictModel):
    """
    ข้อ 3.3 — Context 5 ชั้นที่ส่งเข้า DSPy ทุก turn
    ชั้น 1–2 ห้ามตัด, ชั้น 3–5 ตัดได้ตาม token budget
    """

    active_incident: IncidentState | None  # ชั้น 1
    pending_question: PendingQuestion | None  # ชั้น 2
    open_incidents: list[OpenIncidentBrief] = Field(default_factory=list)  # ชั้น 3
    recent_messages: list[Message] = Field(default_factory=list)  # ชั้น 4
    summary: ConversationSummary | None = None  # ชั้น 5
    now: AwareDatetime = Field(default_factory=now_bkk)


# ---------------------------------------------------------------------------
# 6. Handoff — รายงานและ event ที่ส่งให้ Agent 2
# ---------------------------------------------------------------------------


class ReportLocation(StrictModel):
    area: AreaRef | None
    landmark: str | None = None
    # ส่งเฉพาะเมื่อ needs มี rescue/evacuation หรือมี LocationShareConsent ที่ยังมีผล
    # และผู้ใช้ไม่ได้ขอลบข้อมูล (ข้อ 2.6, 6.10)
    coordinates: Coordinates | None = None
    coordinates_basis: Literal["needs", "consent"] | None = None

    @model_validator(mode="after")
    def _basis(self) -> "ReportLocation":
        if (self.coordinates is None) != (self.coordinates_basis is None):
            raise ValueError("coordinates ต้องระบุ coordinates_basis และกลับกัน")
        return self


class ReportedValue(StrictModel, Generic[T]):
    """field ในรายงาน พร้อมสถานะ เพื่อให้ Agent 2 แยก 'ไม่ทราบ' ออกจาก 'ไม่ได้ถาม'"""

    status: FieldStatus
    value: T | None = None
    reported_at: AwareDatetime | None = None


class IncidentReport(StrictModel):
    """
    รายงานที่ Agent 2 เห็น — ไม่มี LINE User ID, ชื่อ, เบอร์โทร
    ทุกค่าเป็น "รายงานจากผู้แจ้ง" ไม่ใช่ข้อเท็จจริงที่ยืนยันแล้ว
    """

    incident_id: str
    incident_version: int
    reporter_ref: str
    source: Literal["line_oa"] = "line_oa"
    status: IncidentStatus
    matching_status: MatchingStatus
    possible_related_incident_ids: list[str] = Field(default_factory=list)
    related_incident_id: str | None = None
    service_scope: ServiceScope = ServiceScope.UNVERIFIED  # ข้อ 6.5

    location: ReportLocation
    situation: ReportedValue[Situation]
    needs: ReportedValue[list[Need]]
    affected_people: ReportedValue[AffectedPeople]
    vulnerable_people: ReportedValue[VulnerablePeople]
    water_level: ReportedValue[WaterLevel]
    urgent_flags: list[UrgentFlag] = Field(default_factory=list)

    contact_available: bool = False
    contact_ref: str | None = None

    reported_at: AwareDatetime  # เวลาที่สร้าง incident
    submitted_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def _contact_consistent(self) -> "IncidentReport":
        if self.contact_available != (self.contact_ref is not None):
            raise ValueError("contact_available ต้องตรงกับการมี contact_ref")
        if len(set(self.urgent_flags)) != len(self.urgent_flags):
            raise ValueError("urgent_flags ต้องไม่ซ้ำกัน")
        area = self.location.area
        if area is not None and area.match_status == AreaMatchStatus.OUT_OF_PROVINCE \
                and self.service_scope != ServiceScope.OUT_OF_PROVINCE:
            raise ValueError("พื้นที่ out_of_province ต้องมี service_scope = out_of_province")
        return self


class _EventBase(StrictModel):
    event_id: str = Field(default_factory=lambda: new_id("evt"))
    schema_version: str = SCHEMA_VERSION
    occurred_at: AwareDatetime = Field(default_factory=now_bkk)


class IncidentSubmittedEvent(_EventBase):
    event_type: Literal[EventType.INCIDENT_SUBMITTED] = EventType.INCIDENT_SUBMITTED
    incident_id: str
    report: IncidentReport
    trigger: Literal["user_confirmed", "urgent_flags"]

    @model_validator(mode="after")
    def _ids_match(self) -> "IncidentSubmittedEvent":
        if self.report.incident_id != self.incident_id:
            raise ValueError("incident_id ไม่ตรงกับ report")
        return self


ChangedFieldName = IncidentFieldName | Literal["urgent_flags", "service_scope"]


class FieldChange(StrictModel):
    """
    ข้อ 6.7 — การเปลี่ยนค่าหนึ่ง field ใน incident_updated
    Agent 2 เห็นค่าเดิมและค่าใหม่พร้อมเวลา แล้วตีความแนวโน้มเอง
    """

    field: ChangedFieldName
    change_type: FieldChangeType
    previous_value: Any = None
    previous_reported_at: AwareDatetime | None = None
    new_value: Any = None
    reported_at: AwareDatetime

    @model_validator(mode="after")
    def _change_rules(self) -> "FieldChange":
        if self.change_type == FieldChangeType.CONFIRMATION:
            raise ValueError("confirmation ไม่ส่งเป็น event")
        if self.change_type == FieldChangeType.INITIAL:
            if self.previous_value is not None or self.previous_reported_at is not None:
                raise ValueError("initial ต้องไม่มีค่าเดิม")
        elif self.previous_value is None or self.previous_reported_at is None:
            raise ValueError(f"{self.change_type.value} ต้องมีค่าเดิมพร้อมเวลา")
        return self


class IncidentUpdatedEvent(_EventBase):
    event_type: Literal[EventType.INCIDENT_UPDATED] = EventType.INCIDENT_UPDATED
    incident_id: str
    incident_version: int = Field(ge=2)
    changes: list[FieldChange] = Field(min_length=1)
    report: IncidentReport  # รายงานเต็มฉบับล่าสุด Agent 2 ไม่ต้อง merge เอง

    @model_validator(mode="after")
    def _ids_match(self) -> "IncidentUpdatedEvent":
        if self.report.incident_id != self.incident_id:
            raise ValueError("incident_id ไม่ตรงกับ report")
        if self.report.incident_version != self.incident_version:
            raise ValueError("incident_version ไม่ตรงกับ report")
        fields = [c.field for c in self.changes]
        if len(set(fields)) != len(fields):
            raise ValueError("field หนึ่งมีได้ change เดียวต่อ event")
        return self

    @property
    def changed_fields(self) -> list[ChangedFieldName]:
        return [c.field for c in self.changes]


class IncidentLinkedEvent(_EventBase):
    event_type: Literal[EventType.INCIDENT_LINKED] = EventType.INCIDENT_LINKED
    incident_id: str
    linked_to: str
    relation: Relation

    @model_validator(mode="after")
    def _not_self(self) -> "IncidentLinkedEvent":
        if self.incident_id == self.linked_to:
            raise ValueError("เชื่อม incident กับตัวเองไม่ได้")
        return self


class IncidentClosedEvent(_EventBase):
    event_type: Literal[EventType.INCIDENT_CLOSED] = EventType.INCIDENT_CLOSED
    incident_id: str
    close_reason: CloseReason
    last_activity_at: AwareDatetime

    @model_validator(mode="after")
    def _not_out_of_service(self) -> "IncidentClosedEvent":
        if self.close_reason == CloseReason.OUT_OF_SERVICE_AREA:
            raise ValueError("out_of_service_area ปิดจาก draft จึงไม่ส่ง event ให้ Agent 2")
        return self


class DeletionScope(str, Enum):
    """
    ข้อมูลส่วนบุคคลที่ขอลบ — all_personal_data ครอบคลุมทุกประเภทข้างล่าง
    และประเภทที่เพิ่มในอนาคต ไม่จำกัดแค่พิกัด
    """

    ALL_PERSONAL_DATA = "all_personal_data"
    CONTACT = "contact"  # ชื่อ เบอร์โทร
    COORDINATES = "coordinates"
    LOCATION_CONSENT = "location_consent"
    FREE_TEXT = "free_text"  # ข้อความอิสระที่อาจมีชื่อหรือที่อยู่ เช่น situation.text, landmark
    MESSAGES = "messages"  # ข้อความสนทนาและ summary (ฝั่ง Agent 1)


class DataDeletionRequestedEvent(_EventBase):
    event_type: Literal[EventType.DATA_DELETION_REQUESTED] = EventType.DATA_DELETION_REQUESTED
    reporter_ref: str
    scope: list[DeletionScope] = Field(default_factory=lambda: [DeletionScope.ALL_PERSONAL_DATA], min_length=1)

    @model_validator(mode="after")
    def _scope_rules(self) -> "DataDeletionRequestedEvent":
        if len(set(self.scope)) != len(self.scope):
            raise ValueError("scope ต้องไม่ซ้ำกัน")
        if DeletionScope.ALL_PERSONAL_DATA in self.scope and len(self.scope) > 1:
            raise ValueError("all_personal_data ครอบคลุมทุกประเภทแล้ว ใส่คู่กับค่าอื่นไม่ได้")
        return self


HandoffEvent = Annotated[
    Union[
        IncidentSubmittedEvent,
        IncidentUpdatedEvent,
        IncidentLinkedEvent,
        IncidentClosedEvent,
        DataDeletionRequestedEvent,
    ],
    Field(discriminator="event_type"),
]


class HandoffAck(StrictModel):
    """Technical ACK จาก Agent 2 = รับและบันทึกสำเร็จ ไม่ใช่มีคนอ่านหรือส่งคนไปแล้ว"""

    event_id: str
    received: bool
    received_at: AwareDatetime
    error: str | None = None


class OutboxRecord(StrictModel):
    """แถวใน escalation_events — บันทึกก่อนส่งเสมอ (ข้อ 1.3)"""

    event_id: str
    event_type: EventType
    incident_id: str | None
    payload: dict[str, Any]
    delivery_status: DeliveryStatus = DeliveryStatus.PENDING
    attempts: int = Field(default=0, ge=0)
    created_at: AwareDatetime = Field(default_factory=now_bkk)
    last_attempt_at: AwareDatetime | None = None
    acked_at: AwareDatetime | None = None
    last_error: str | None = None


class DeletionRequest(StrictModel):
    """
    คำขอ "ลบข้อมูลของฉัน" — ทันทีที่สร้าง: ยกเลิก LocationShareConsent ทั้งหมดของผู้ใช้
    และตั้ง UserRecord.deletion_requested_at เพื่อระงับการแนบพิกัดทุกเงื่อนไขระหว่างรอลบ (ข้อ 6.10)
    """

    request_id: str = Field(default_factory=lambda: new_id("del"))
    internal_user_id: str
    scope: list[DeletionScope] = Field(default_factory=lambda: [DeletionScope.ALL_PERSONAL_DATA], min_length=1)
    requested_at: AwareDatetime = Field(default_factory=now_bkk)
    consents_revoked_at: AwareDatetime | None = None
    local_deleted_at: AwareDatetime | None = None
    event_id: str | None = None  # DataDeletionRequestedEvent ที่ส่งไป Agent 2
    agent2_acked_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def _order(self) -> "DeletionRequest":
        if DeletionScope.ALL_PERSONAL_DATA in self.scope and len(self.scope) > 1:
            raise ValueError("all_personal_data ครอบคลุมทุกประเภทแล้ว ใส่คู่กับค่าอื่นไม่ได้")
        if self.local_deleted_at is not None and self.consents_revoked_at is None:
            raise ValueError("ต้องยกเลิก consent ก่อนลบข้อมูลฝั่ง Agent 1")
        if self.agent2_acked_at is not None and self.event_id is None:
            raise ValueError("ได้ ACK จาก Agent 2 ต้องมี event_id")
        return self


# ---------------------------------------------------------------------------
# 7. RAG
# ---------------------------------------------------------------------------


class Hotline(StrictModel):
    """ข้อ 4.2 — config ชุดเดียว ใช้ทั้ง urgent template และ RAG"""

    key: str  # เช่น "ddpm", "ems"
    name: str
    number: str
    source_url: str
    verified_at: AwareDatetime


class KnowledgeChunk(StrictModel):
    """ข้อ 4.3 — ใช้ตอบได้เฉพาะ review_status = approved"""

    chunk_id: str = Field(default_factory=lambda: new_id("kb"))
    source_org: str
    title: str
    url: str
    published_at: AwareDatetime | None = None
    verified_at: AwareDatetime | None = None
    valid_until: AwareDatetime | None = None
    category: KBCategory
    topic: KBTopic
    language: str = "th"

    original_text: str
    reviewed_text: str | None = None
    review_status: ReviewStatus = ReviewStatus.PENDING_REVIEW
    reviewed_by: str | None = None
    reviewed_at: AwareDatetime | None = None

    is_core_safety: bool = False  # Core Safety Snippet ใช้ได้โดยไม่ผ่าน retrieval (ข้อ 4.4)
    embedding_model: str | None = None

    @model_validator(mode="after")
    def _review_rules(self) -> "KnowledgeChunk":
        if self.category == KBCategory.B and self.valid_until is None:
            raise ValueError("chunk กลุ่ม B ต้องมี valid_until")
        if self.review_status == ReviewStatus.APPROVED:
            if not (self.reviewed_text and self.reviewed_by and self.reviewed_at and self.verified_at):
                raise ValueError("chunk ที่ approved ต้องมี reviewed_text, reviewed_by, reviewed_at, verified_at")
        return self

    def is_usable(self, at: datetime) -> bool:
        if self.review_status != ReviewStatus.APPROVED:
            return False
        return self.valid_until is None or at <= self.valid_until

    @property
    def text(self) -> str:
        return self.reviewed_text or self.original_text


class RetrievedChunk(StrictModel):
    chunk: KnowledgeChunk
    score: float  # คะแนนรวม hybrid search
    vector_score: float | None = None
    keyword_score: float | None = None


# ---------------------------------------------------------------------------
# 8. Agent I/O
# ---------------------------------------------------------------------------


class FieldUpdate(StrictModel):
    """
    ส่วนที่เปลี่ยนจากข้อความล่าสุด ที่ LLM ดึงได้ (delta)
    Backend Rules เป็นผู้ merge เข้า IncidentState — LLM ไม่แก้ state เอง

    - location_area: value เป็น AreaMention (ชื่อ + ระดับตามคำผู้ใช้) Backend จับคู่เป็น AreaRef
    - situation: Backend แทน text ด้วยข้อความต้นฉบับของผู้ใช้ ใช้จาก LLM เฉพาะ tags (ข้อ 6.1)
    - is_correction: เป็นแค่ความเห็นของ Gemini Backend ตัดสิน FieldChangeType เอง (ข้อ 6.7)
    """

    field: IncidentFieldName
    status: Literal[FieldStatus.KNOWN, FieldStatus.UNKNOWN, FieldStatus.DECLINED]
    value: Any = None  # ตรวจชนิดตอน merge ตาม field
    raw: str | None = None  # ข้อความต้นฉบับส่วนที่ใช้ดึงค่า Backend ตรวจว่าอยู่ในข้อความผู้ใช้จริง
    is_correction: bool = False

    @model_validator(mode="after")
    def _value_matches_status(self) -> "FieldUpdate":
        if self.status == FieldStatus.KNOWN and self.value is None:
            raise ValueError("status 'known' ต้องมี value")
        if self.status != FieldStatus.KNOWN and self.value is not None:
            raise ValueError("status unknown/declined ต้องไม่มี value")
        return self


class ExtractionResult(StrictModel):
    """ผลลัพธ์ของ Signature ExtractIntentAndSlots"""

    intents: list[Intent] = Field(min_length=1)
    target: Literal["active_incident", "new_incident", "unclear", "none"]
    mentioned_location: str | None = None  # ใช้กฎ matching ข้อ 3.2
    updates: list[FieldUpdate] = Field(default_factory=list)
    urgent_flags: list[UrgentFlag] = Field(default_factory=list)
    question: str | None = None  # คำถามความรู้ที่ต้องตอบจาก RAG


class AgentTurnOutput(StrictModel):
    """ผลลัพธ์ต่อ turn ที่ API คืน และใช้ทดสอบ Golden Scenarios"""

    output_text: str
    intents: list[Intent]
    incident_id: str | None
    incident_status: IncidentStatus | None
    matching_status: MatchingStatus | None = None
    urgent_flags: list[UrgentFlag] = Field(default_factory=list)
    missing_fields: list[IncidentFieldName] = Field(default_factory=list)
    next_actions: list[NextAction] = Field(min_length=1)  # หนึ่ง turn ทำได้หลายอย่าง เช่น submit + answer
    pending_question: PendingQuestion | None = None
    delivery_status: DeliveryStatus | None = None
    sources: list[str] = Field(default_factory=list)  # chunk_id ที่ใช้ตอบ
    used_hotline_template: bool = False
    model: str | None = None


__all__ = [name for name in dir() if not name.startswith("_")]
