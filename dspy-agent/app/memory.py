"""
DSI490 — Agent 1 Memory Rules (ขั้นที่ 3)

อ้างอิง Design Decisions v1.1 ข้อ 3.3, 6.1, 6.7, 6.9
ไม่เรียก LLM — Gemini ให้แค่ FieldUpdate (delta) ส่วนการตัดสินทั้งหมดอยู่ที่นี่

    1. แปลงค่าจาก Gemini เป็นชนิดของ field        coerce_value()
    2. จัดประเภทการเปลี่ยนค่า                    classify_change()        ข้อ 6.7
    3. merge FieldUpdate เข้า IncidentState         apply_updates()          ข้อ 6.1 ขั้นที่ 8
    4. บันทึกการถาม / ปิดคำถามที่ไม่ได้คำตอบ         record_questions_asked(), close_unanswered()   ข้อ 6.9
    5. field ที่ยังขาด                            missing_required(), can_ask_now()
    6. ประกอบ Context 5 ชั้น                     build_context()          ข้อ 3.3

ทุกฟังก์ชันคืน state ใหม่ ไม่แก้ object เดิม
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Iterable

from pydantic import ValidationError

from .safety import correction_cue_for, normalize_text, raw_in_message
from .schemas import (
    REQUIRED_FIELDS,
    AffectedPeople,
    AreaLevel,
    AreaMatchStatus,
    AreaMention,
    AreaRef,
    ContextBundle,
    ConversationSummary,
    Coordinates,
    FieldChange,
    FieldChangeType,
    FieldRevision,
    FieldStatus,
    FieldUpdate,
    IncidentFieldName,
    IncidentState,
    IncidentStatus,
    Message,
    Need,
    OpenIncidentBrief,
    PendingQuestion,
    Situation,
    TrackedField,
    VulnerablePeople,
    WaterLevel,
    max_asks,
)

F = IncidentFieldName
AreaResolver = Callable[[AreaMention], AreaRef]


def unverified_area_resolver(mention: AreaMention) -> AreaRef:
    """
    ตัวจับคู่พื้นที่ชั่วคราว ระหว่างยังไม่มีข้อมูลอ้างอิงสิงห์บุรี (areas.py)
    ไม่เดารหัส ไม่เติมระดับ เก็บตามคำผู้ใช้เท่านั้น
    """
    return AreaRef(raw=mention.name, stated_level=mention.level, match_status=AreaMatchStatus.UNVERIFIED)


# ---------------------------------------------------------------------------
# 1. แปลงค่าจาก Gemini
# ---------------------------------------------------------------------------


class InvalidValue(ValueError):
    pass


def _clean_text(s: Any) -> str:
    if not isinstance(s, str) or not s.strip():
        raise InvalidValue("ต้องเป็นข้อความที่ไม่ว่าง")
    return re.sub(r"\s+", " ", normalize_text(s)).strip()


def coerce_value(
    name: IncidentFieldName,
    value: Any,
    *,
    message_text: str,
    area_resolver: AreaResolver = unverified_area_resolver,
) -> Any:
    """
    แปลง value ของ FieldUpdate เป็นชนิดที่ field ต้องการ
    - location_area: รับ AreaMention (หรือข้อความ สำหรับ prompt รุ่นเก่า) แล้วจับคู่ด้วย area_resolver
    - situation: text ใช้ข้อความต้นฉบับของผู้ใช้เสมอ ใช้จาก Gemini เฉพาะ tags (ข้อ 6.1)
    """
    try:
        if name == F.LOCATION_AREA:
            mention = AreaMention(name=_clean_text(value)) if isinstance(value, str) else AreaMention.model_validate(value)
            return area_resolver(mention)
        if name == F.LOCATION_LANDMARK:
            return _clean_text(value)
        if name == F.LOCATION_COORDINATES:
            return Coordinates.model_validate(value)
        if name == F.SITUATION:
            tags = value.get("tags", []) if isinstance(value, dict) else []
            return Situation(text=_clean_text(message_text), tags=tags)
        if name == F.NEEDS:
            needs = [Need(v) for v in (value if isinstance(value, list) else [value])]
            if not needs:
                raise InvalidValue("needs ต้องมีอย่างน้อยหนึ่งค่า")
            return _normalize_needs(needs)
        if name == F.AFFECTED_PEOPLE:
            return AffectedPeople.model_validate(value)
        if name == F.VULNERABLE_PEOPLE:
            return VulnerablePeople.model_validate(value)
        if name == F.WATER_LEVEL:
            return WaterLevel.model_validate(value)
    except (ValidationError, ValueError, TypeError, AttributeError) as e:
        raise InvalidValue(f"{name.value}: {e}") from e
    raise InvalidValue(f"{name.value} ไม่ได้เก็บใน IncidentState")


def _normalize_needs(needs: Iterable[Need]) -> list[Need]:
    s = set(needs)
    if len(s) > 1:
        s.discard(Need.INFO_ONLY)  # มีความต้องการจริงแล้ว ไม่ใช่แค่แจ้งให้ทราบ
    return sorted(s, key=lambda n: n.value)


# ---------------------------------------------------------------------------
# 2. จัดประเภทการเปลี่ยนค่า (ข้อ 6.7)
# ---------------------------------------------------------------------------


def _fold(s: str) -> str:
    return re.sub(r"\s+", " ", normalize_text(s)).strip().lower()


def comparison_key(name: IncidentFieldName, value: Any) -> Any:
    """
    ค่าที่ใช้ตัดสินว่า "เท่ากับค่าเดิมหลังจัดรูปแบบ" หรือไม่
    ค่าคนละรูปแบบ (เช่น descriptor กับ cm) ไม่แปลงหน่วยเพื่อเทียบ → ต่างกัน → update
    """
    if name == F.LOCATION_AREA:
        return ("code", value.area_code) if value.area_code else ("raw", _fold(value.raw))
    if name == F.LOCATION_LANDMARK:
        return _fold(value)
    if name == F.LOCATION_COORDINATES:
        return (round(value.latitude, 5), round(value.longitude, 5))
    if name == F.SITUATION:
        return frozenset(value.tags)  # ข้อความเปลี่ยนทุกครั้ง ใช้หมวดตัดสินแทน (scenario R08)
    if name == F.NEEDS:
        return frozenset(value)
    if name == F.WATER_LEVEL:
        return (value.cm, value.descriptor)
    if name == F.AFFECTED_PEOPLE:
        return (value.present, value.count)
    if name == F.VULNERABLE_PEOPLE:
        return (value.present, frozenset((g.type, g.count) for g in value.groups))
    return value


def classify_change(
    name: IncidentFieldName,
    current: TrackedField,
    new_value: Any,
    *,
    gemini_says_correction: bool,
    keyword_cue: bool,
) -> FieldChangeType:
    if current.status != FieldStatus.KNOWN or current.value is None:
        return FieldChangeType.INITIAL  # ยังไม่มีค่า แม้ Gemini บอกว่าแก้ไข (scenario R09)
    if comparison_key(name, current.value) == comparison_key(name, new_value):
        return FieldChangeType.CONFIRMATION
    if gemini_says_correction and keyword_cue:
        return FieldChangeType.CORRECTION
    return FieldChangeType.UPDATE


def merge_value(name: IncidentFieldName, old: Any, new: Any, change: FieldChangeType) -> Any:
    """ค่าที่จะเก็บหลัง merge"""
    if old is None or change == FieldChangeType.CORRECTION:
        return new
    if name == F.NEEDS:
        # Gemini ดึงเฉพาะข้อความล่าสุด ความต้องการใหม่ไม่ได้แปลว่าความต้องการเดิมหายไป
        return _normalize_needs([*old, *new])
    if name == F.SITUATION and change == FieldChangeType.CONFIRMATION:
        if _fold(new.text) in _fold(old.text):
            return old
        return Situation(text=f"{old.text} / {new.text}", tags=old.tags)  # เก็บรายละเอียดเพิ่ม ไม่ส่ง event
    return new


# ---------------------------------------------------------------------------
# 3. merge FieldUpdate เข้า IncidentState
# ---------------------------------------------------------------------------


@dataclass
class RejectedUpdate:
    update: FieldUpdate
    reason: str


@dataclass
class MergeResult:
    state: IncidentState
    changes: list[FieldChange] = field(default_factory=list)  # ส่งเป็น event ได้ (ไม่รวม confirmation)
    confirmations: list[IncidentFieldName] = field(default_factory=list)
    status_changes: dict[IncidentFieldName, FieldStatus] = field(default_factory=dict)  # unknown / declined
    contact_updates: list[FieldUpdate] = field(default_factory=list)  # ส่งต่อให้ส่วนจัดการ ContactRecord
    rejected: list[RejectedUpdate] = field(default_factory=list)

    @property
    def touched(self) -> bool:
        """มีความเคลื่อนไหวของ incident (นับต่ออายุ timeout ข้อ 3.1)"""
        return bool(self.changes or self.confirmations or self.status_changes or self.contact_updates)

    def change_for(self, name: IncidentFieldName) -> FieldChange | None:
        return next((c for c in self.changes if c.field == name), None)


def apply_updates(
    state: IncidentState,
    updates: Iterable[FieldUpdate],
    *,
    message_text: str,
    message_id: str | None,
    now: datetime,
    area_resolver: AreaResolver = unverified_area_resolver,
) -> MergeResult:
    """
    merge ส่วนที่เปลี่ยนจากข้อความเดียวเข้า state
    message_text ต้องเป็นข้อความที่ปิดบังเบอร์แล้ว (ข้อ 6.2)
    ไม่แตะ urgent flags (safety.py) และ lifecycle (conversation.py)
    """
    result = MergeResult(state=state)
    patch: dict[str, TrackedField] = {}
    seen: set[IncidentFieldName] = set()

    for upd in updates:
        name = upd.field
        if name in seen:
            result.rejected.append(RejectedUpdate(upd, "field ซ้ำในข้อความเดียว ใช้รายการแรก"))
            continue
        seen.add(name)

        attr = "contact_ask" if name == F.CONTACT else name.value
        current: TrackedField = patch.get(attr) or state.tracked_field(name)

        if name == F.CONTACT and upd.status == FieldStatus.KNOWN:
            # เบอร์จริงถูกดึงออกตั้งแต่ทางเข้า (safety.redact_phones) และเก็บใน ContactRecord
            # state เก็บแค่สถานะการถาม ผู้เรียกเป็นผู้ตั้ง contact_ask = known เมื่อบันทึก ContactRecord สำเร็จ
            result.contact_updates.append(upd)
            continue
        raw = upd.raw if raw_in_message(upd.raw, message_text) else None

        if upd.status in (FieldStatus.UNKNOWN, FieldStatus.DECLINED):
            if current.status == FieldStatus.KNOWN:
                result.rejected.append(RejectedUpdate(upd, "มีค่าที่ผู้ใช้ให้ไว้แล้ว ไม่ลบเพราะคำว่าไม่ทราบ"))
                continue
            patch[attr] = current.model_copy(update={
                "status": upd.status, "value": None, "raw": raw, "updated_at": now, "source_message_id": message_id,
            })
            result.status_changes[name] = upd.status
            continue

        try:
            new_value = coerce_value(name, upd.value, message_text=message_text, area_resolver=area_resolver)
        except InvalidValue as e:
            result.rejected.append(RejectedUpdate(upd, str(e)))
            continue

        change = classify_change(
            name, current, new_value,
            gemini_says_correction=upd.is_correction,
            keyword_cue=correction_cue_for(raw, message_text) if raw else False,
        )
        stored = merge_value(name, current.value, new_value, change)

        history = list(current.history)
        if change in (FieldChangeType.UPDATE, FieldChangeType.CORRECTION):
            history.append(FieldRevision(
                status=current.status, value=current.value, raw=current.raw, reported_at=current.updated_at,
                changed_at=now, change_type=change, source_message_id=current.source_message_id,
            ))

        patch[attr] = TrackedField(
            status=FieldStatus.KNOWN, value=stored,
            raw=(raw or current.raw) if change == FieldChangeType.CONFIRMATION else raw,
            updated_at=now, source_message_id=message_id,
            ask_count=current.ask_count, last_asked_at=current.last_asked_at, history=history,
        )

        if change == FieldChangeType.CONFIRMATION:
            result.confirmations.append(name)
        else:
            dump = lambda v: v.model_dump(mode="json") if hasattr(v, "model_dump") else (
                [x.value for x in v] if isinstance(v, list) else v)
            result.changes.append(FieldChange(
                field=name, change_type=change,
                previous_value=None if change == FieldChangeType.INITIAL else dump(current.value),
                previous_reported_at=None if change == FieldChangeType.INITIAL else (current.updated_at or now),
                new_value=dump(stored), reported_at=now,
            ))

    if patch or result.contact_updates:
        result.state = IncidentState.model_validate({
            **state.model_dump(),
            **{k: v.model_dump() for k, v in patch.items()},
            "updated_at": now, "last_activity_at": now,
        })
    return result


# ---------------------------------------------------------------------------
# 4. การถามและการปิดคำถามที่ไม่ได้คำตอบ (ข้อ 6.9)
# ---------------------------------------------------------------------------


def can_ask_now(state: IncidentState, name: IncidentFieldName) -> bool:
    f = state.tracked_field(name)
    return f.can_ask and f.ask_count < max_asks(name)


def record_questions_asked(
    state: IncidentState, fields: Iterable[IncidentFieldName], *, now: datetime
) -> IncidentState:
    """เรียกหลังส่งข้อความที่มีคำถามสำเร็จ"""
    patch = {}
    for name in fields:
        f = state.tracked_field(name)
        attr = "contact_ask" if name == F.CONTACT else name.value
        patch[attr] = f.model_copy(update={"ask_count": f.ask_count + 1, "last_asked_at": now}).model_dump()
    return IncidentState.model_validate({**state.model_dump(), **patch}) if patch else state


def close_unanswered(state: IncidentState, pending: PendingQuestion | None) -> IncidentState:
    """
    หลัง merge ข้อความที่ตอบ pending question แล้ว:
    field ที่ถามไปแต่ยังไม่ได้คำตอบ และถามครบจำนวนครั้งแล้ว → unanswered (ไม่ใช่ unknown)
    """
    if pending is None or pending.incident_id != state.incident_id:
        return state
    patch = {}
    for name in pending.field_names:
        f = state.tracked_field(name)
        if f.status == FieldStatus.NOT_ASKED and f.ask_count >= max_asks(name):
            attr = "contact_ask" if name == F.CONTACT else name.value
            patch[attr] = f.model_copy(update={"status": FieldStatus.UNANSWERED}).model_dump()
    return IncidentState.model_validate({**state.model_dump(), **patch}) if patch else state


def pending_is_valid(pending: PendingQuestion | None, state: IncidentState | None) -> bool:
    """ข้อ 3.3 — ใช้ไม่ได้เมื่อ incident ปิด/abandoned หรือไม่มี field ที่ยังรอคำตอบ"""
    if pending is None or state is None or pending.incident_id != state.incident_id or not state.is_open:
        return False
    return any(state.tracked_field(n).status == FieldStatus.NOT_ASKED for n in pending.field_names)


def missing_required(state: IncidentState) -> list[IncidentFieldName]:
    """field บังคับที่ยังไม่มีค่า (รวมที่ unknown / unanswered) ใช้ระบุข้อมูลที่ขาดตอนขอยืนยันรายงาน"""
    return [n for n in REQUIRED_FIELDS if state.tracked_field(n).status != FieldStatus.KNOWN]


# ---------------------------------------------------------------------------
# 5. Context 5 ชั้น (ข้อ 3.3)
# ---------------------------------------------------------------------------

RECENT_WINDOW = timedelta(hours=24)


def select_recent_messages(
    messages: list[Message],
    pending: PendingQuestion | None,
    *,
    now: datetime,
    pending_valid: bool,
    max_messages: int = 10,
) -> list[Message]:
    """
    มี pending question ที่ยังใช้ได้ → เก็บตั้งแต่ข้อความที่ถาม แม้เกิน 24 ชั่วโมง
    ไม่มี → ใช้เฉพาะ 24 ชั่วโมงล่าสุด
    """
    ordered = sorted(messages, key=lambda m: m.created_at)
    if pending is not None and pending_valid:
        start = next((i for i, m in enumerate(ordered) if m.message_id == pending.asked_message_id), None)
        if start is None:  # ไม่รู้ข้อความที่ถาม ใช้เวลาที่ถามแทน
            start = next((i for i, m in enumerate(ordered) if m.created_at >= pending.asked_at), len(ordered))
        window = ordered[start:]
        # ข้อความก่อนคำถามที่อยู่ใน 24 ชั่วโมงยังช่วยเข้าใจบริบทได้
        before = [m for m in ordered[:start] if now - m.created_at <= RECENT_WINDOW]
        window = before + window
    else:
        window = [m for m in ordered if now - m.created_at <= RECENT_WINDOW]
    return window[-max_messages:]


def _brief(state: IncidentState) -> OpenIncidentBrief:
    area = state.location_area.value
    return OpenIncidentBrief(
        incident_id=state.incident_id, status=state.status, created_at=state.created_at,
        location_area=(area.tambon or area.amphoe or area.raw) if area else None,
        area_code=area.area_code if area else None,
    )


def build_context(
    *,
    active: IncidentState | None,
    pending: PendingQuestion | None,
    open_incidents: Iterable[IncidentState],
    messages: list[Message],
    summary: ConversationSummary | None,
    now: datetime,
    max_messages: int = 10,
    max_chars: int = 6000,
) -> ContextBundle:
    """
    ประกอบ Context 5 ชั้น แล้วตัดตาม budget (นับตัวอักษรแทน token)
    ลำดับการตัด: ย่อ summary → ข้อความเก่าในชั้น 4 → ไม่ตัดชั้น 1–2 เด็ดขาด
    """
    valid = pending_is_valid(pending, active)
    pend = pending if valid else None
    briefs = [_brief(s) for s in open_incidents if s.is_open]
    recent = select_recent_messages(messages, pending, now=now, pending_valid=valid, max_messages=max_messages)

    def size(sm: ConversationSummary | None, msgs: list[Message]) -> int:
        fixed = len(active.model_dump_json()) if active else 0
        fixed += len(pend.model_dump_json()) if pend else 0
        fixed += sum(len(b.model_dump_json()) for b in briefs)
        return fixed + (len(sm.text) if sm else 0) + sum(len(m.text) for m in msgs)

    sm = summary
    if sm is not None and size(sm, recent) > max_chars:
        over = size(sm, recent) - max_chars
        keep = max(0, len(sm.text) - over)
        sm = sm.model_copy(update={"text": sm.text[:keep]}) if keep else None
    while len(recent) > 1 and size(sm, recent) > max_chars:
        recent = recent[1:]

    return ContextBundle(
        active_incident=active, pending_question=pend, open_incidents=briefs,
        recent_messages=recent, summary=sm, now=now,
    )


__all__ = [
    "AreaResolver", "InvalidValue", "MergeResult", "RejectedUpdate", "apply_updates", "build_context",
    "can_ask_now", "classify_change", "close_unanswered", "coerce_value", "comparison_key", "merge_value",
    "missing_required", "pending_is_valid", "record_questions_asked", "select_recent_messages",
    "unverified_area_resolver",
]
