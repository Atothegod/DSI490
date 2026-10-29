"""
conversation.py — Backend Rules ขั้นที่ 3 ส่วนบทสนทนา (ข้อ 2.2, 3.1, 6.6, 6.8, 6.9)

หน้าที่
1. ตัดสินว่า turn นี้ต้องทำอะไร: ส่งรายงาน / ส่ง update / ถามต่อ / ขอยืนยัน (plan_turn)
2. เลือกคำถามถัดไปตามลำดับที่ตกลงไว้ (next_questions)
3. สร้างข้อความตอบจาก templates_th.json (render_reply)
4. บันทึกสิ่งที่ "ส่งสำเร็จแล้ว" หลัง LINE ตอบกลับ OK (after_delivery)

ไม่เรียก Gemini ไม่แตะ DB ไม่สร้าง event (เป็นงานของ handoff.py)

ลำดับการเรียกต่อ turn (ดู Notebook ส่วน E)
    redact → extract → memory.apply_updates → safety.merge_urgent_flags
    → reconcile_needs → memory.close_unanswered → plan_turn → apply_submission (ถ้าส่ง)
    → render_reply → [ส่ง LINE สำเร็จ] → after_delivery
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Literal

from pydantic import AwareDatetime

from .memory import MergeResult, can_ask_now, missing_required, record_questions_asked
from .safety import asks_for_contact, flags_to_notify, format_hotline_text, load_templates, render_contact_answer, render_hotline_notice
from .schemas import (
    COORDINATE_NEEDS,
    ExtractionResult,
    FieldChange,
    FieldChangeType,
    FieldRevision,
    FieldStatus,
    Hotline,
    HotlineNotice,
    IncidentFieldName,
    IncidentState,
    IncidentStatus,
    Intent,
    Need,
    NextAction,
    PendingQuestion,
    StrictModel,
    TrackedField,
    UrgentFlag,
)

F = IncidentFieldName
SubmitTrigger = Literal["user_confirmed", "urgent_flags"]


# ---------------------------------------------------------------------------
# 1. คำขอยืนยันรายงานที่ส่งสำเร็จและยังรอคำตอบ (ข้อ 2.2)
# ---------------------------------------------------------------------------


class PendingConfirmation(StrictModel):
    """
    สร้างใน after_delivery เท่านั้น (คือหลังส่งข้อความสรุปสำเร็จ)
    summary_fingerprint ใช้ตรวจว่าข้อมูลยังเหมือนตอนแสดงสรุปหรือไม่ — ถ้าเปลี่ยน ต้องยืนยันสรุปใหม่

    หมายเหตุ: ตอนต่อ DB ควรย้ายไปเก็บใน ConversationState คู่กับ pending_question
    """

    incident_id: str
    summary_fingerprint: str
    asked_at: AwareDatetime
    asked_message_id: str


SUMMARY_FIELDS: tuple[IncidentFieldName, ...] = (
    F.LOCATION_AREA, F.LOCATION_LANDMARK, F.LOCATION_COORDINATES, F.SITUATION,
    F.NEEDS, F.AFFECTED_PEOPLE, F.VULNERABLE_PEOPLE, F.WATER_LEVEL,
)


def summary_fingerprint(state: IncidentState) -> str:
    """ลายนิ้วมือของข้อมูลที่ผู้ใช้เห็นในสรุป ไม่ขึ้นกับถ้อยคำใน template"""
    data = {}
    for name in SUMMARY_FIELDS:
        f = state.tracked_field(name)
        v = f.value
        data[name.value] = [f.status.value, v.model_dump(mode="json") if hasattr(v, "model_dump")
                            else [x.value for x in v] if isinstance(v, list) else v]
    data["urgent_flags"] = sorted(f.value for f in state.urgent_flag_types)
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def confirmation_is_waiting(pc: PendingConfirmation | None, state: IncidentState) -> bool:
    return pc is not None and pc.incident_id == state.incident_id and state.status == IncidentStatus.DRAFT


# ---------------------------------------------------------------------------
# 2. needs ที่ขัดกับคำขอความช่วยเหลือ
# ---------------------------------------------------------------------------


def reconcile_needs(
    state: IncidentState,
    new_flags: Iterable[UrgentFlag],
    *,
    now: datetime,
    message_id: str | None,
) -> tuple[IncidentState, FieldChange | None]:
    """
    ผู้ใช้เคยบอกว่า "แค่แจ้งให้ทราบ" (info_only) แล้วข้อความนี้ขอความช่วยเหลือ (help_requested ใหม่)
    → info_only ขัดกับคำขอ ต้องเอาออก แต่ไม่เดาว่าต้องการ rescue (needs กลับเป็น not_asked เพื่อถามใหม่)

    กรณีข้อความนี้ระบุ need ใหม่เอง เช่น [info_only] + [rescue] → memory.merge_value เอา info_only ออกแล้ว
    จึงเหลือเฉพาะกรณีที่ needs ยังเป็น [info_only] ล้วน
    """
    f = state.needs
    if UrgentFlag.HELP_REQUESTED not in set(new_flags) or f.status != FieldStatus.KNOWN or f.value != [Need.INFO_ONLY]:
        return state, None
    retracted = TrackedField[list[Need]](
        status=FieldStatus.NOT_ASKED, updated_at=now, source_message_id=message_id,
        ask_count=f.ask_count, last_asked_at=f.last_asked_at,
        history=[*f.history, FieldRevision[list[Need]](
            status=f.status, value=f.value, raw=f.raw, reported_at=f.updated_at, changed_at=now,
            change_type=FieldChangeType.UPDATE, source_message_id=f.source_message_id,
        )],
    )
    change = FieldChange(
        field=F.NEEDS, change_type=FieldChangeType.UPDATE,
        previous_value=[Need.INFO_ONLY.value], previous_reported_at=f.updated_at or now,
        new_value=None, reported_at=now,
    )
    new_state = IncidentState.model_validate({**state.model_dump(), "needs": retracted.model_dump(),
                                              "updated_at": now, "last_activity_at": now})
    return new_state, change


# ---------------------------------------------------------------------------
# 3. ลำดับคำถาม (ข้อ 6.9)
# ---------------------------------------------------------------------------
# เหตุทั่วไป: พื้นที่ → สถานการณ์ → ความต้องการ → คน (คู่) → [ตำแหน่งละเอียด ถ้าต้องการ rescue/evacuation] → ระดับน้ำ
# เหตุด่วน (ส่งรายงานแล้ว): [พื้นที่ ถ้ายังไม่มี] → พิกัดหรือจุดสังเกต → คน (คู่) → ช่องทางติดต่อ → ความต้องการ
# field บังคับถามได้ตาม max_asks ส่วน field ที่ไม่บังคับถามครั้งเดียวแล้วพอ (ข้อตกลงคำถามที่ 8)

REQUIRED = (F.LOCATION_AREA, F.SITUATION, F.NEEDS)


def is_urgent(state: IncidentState) -> bool:
    return bool(state.urgent_flags)


def _askable(state: IncidentState, name: IncidentFieldName) -> bool:
    if not can_ask_now(state, name):
        return False
    return name in REQUIRED or state.tracked_field(name).ask_count == 0


def _known(state: IncidentState, name: IncidentFieldName) -> bool:
    return state.tracked_field(name).status == FieldStatus.KNOWN


def _wants_precise_location(state: IncidentState) -> bool:
    needs = set(state.needs.value or [])
    return is_urgent(state) or bool(needs & COORDINATE_NEEDS)


def _steps(state: IncidentState) -> list[tuple[IncidentFieldName, ...]]:
    precise = () if (_known(state, F.LOCATION_COORDINATES) or _known(state, F.LOCATION_LANDMARK)) \
        else (F.LOCATION_COORDINATES, F.LOCATION_LANDMARK)  # ได้อย่างหนึ่งแล้วไม่ต้องถามอีกอย่าง
    people = (F.AFFECTED_PEOPLE, F.VULNERABLE_PEOPLE)
    if is_urgent(state):
        return [(F.LOCATION_AREA,), precise, people, (F.CONTACT,), (F.NEEDS,)]
    steps = [(F.LOCATION_AREA,), (F.SITUATION,), (F.NEEDS,), people]
    if _wants_precise_location(state):
        steps.append(precise)
    steps.append((F.WATER_LEVEL,))
    return steps


def next_questions(state: IncidentState) -> list[IncidentFieldName]:
    """
    field ที่จะถามในข้อความถัดไป (ไม่เกิน 2 และเป็นกลุ่มเดียวกัน) — ว่างคือหมดคำถาม
    เหตุที่ส่งแล้วจากการยืนยันของผู้ใช้ไม่ถามต่อ (จบรอบการถามตั้งแต่ขอยืนยัน)
    """
    if not state.is_open:
        return []
    if state.status == IncidentStatus.SUBMITTED and not is_urgent(state):
        return []
    for step in _steps(state):
        fields = [n for n in step if _askable(state, n)]
        if fields:
            return fields[:2]
    return []


# ---------------------------------------------------------------------------
# 4. แผนของ turn
# ---------------------------------------------------------------------------


@dataclass
class TurnPlan:
    actions: list[NextAction]
    submit_trigger: SubmitTrigger | None = None
    ask_fields: list[IncidentFieldName] = field(default_factory=list)
    show_summary: bool = False
    summary_changed: bool = False  # แสดงสรุปใหม่เพราะข้อมูลเปลี่ยนหลังสรุปครั้งก่อน
    keep_confirmation: bool = False  # คำขอยืนยันเดิมยังใช้ได้ (เช่น ผู้ใช้ถามเรื่องอื่นแทรก)
    hotline_flags: list[UrgentFlag] = field(default_factory=list)
    contact_answer: bool = False
    ack: str | None = None  # key ใน acknowledgements
    answer_question: str | None = None
    missing_fields: list[IncidentFieldName] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # เหตุผลของการตัดสิน ใช้ดูใน Notebook และ log


def _ack_key(merge: MergeResult | None, extra_changes: list[FieldChange]) -> str | None:
    changes = [*(merge.changes if merge else []), *extra_changes]
    statuses = set(merge.status_changes.values()) if merge else set()
    if any(c.change_type == FieldChangeType.CORRECTION for c in changes):
        return "correction"
    if any(c.change_type == FieldChangeType.UPDATE for c in changes):
        return "updated"
    if changes or (merge and merge.confirmations):
        return "received"
    if FieldStatus.DECLINED in statuses:
        return "declined"
    if FieldStatus.UNKNOWN in statuses:
        return "unknown"
    return None


def plan_turn(
    state: IncidentState,
    *,
    extraction: ExtractionResult,
    merge: MergeResult | None,
    new_flags: Iterable[UrgentFlag] = (),
    extra_changes: Iterable[FieldChange] = (),
    pending_confirmation: PendingConfirmation | None = None,
    message_text: str = "",
) -> TurnPlan:
    """
    state: หลัง merge, รวม flag, reconcile_needs และ close_unanswered แล้ว (ยังไม่ apply_submission)
    new_flags: flag ชนิดใหม่ของ incident จาก safety.merge_urgent_flags
    extra_changes: การเปลี่ยนที่ไม่ได้มาจาก merge เช่น reconcile_needs
    ข้อความ target = unclear ต้องให้ผู้เรียกถามเหตุเดิม/เหตุใหม่ก่อน ไม่ส่งเข้ามาที่นี่
    """
    extra_changes = list(extra_changes)
    new_flags = set(new_flags)
    plan = TurnPlan(actions=[])
    plan.hotline_flags = sorted(flags_to_notify(state), key=lambda f: f.value)
    plan.contact_answer = not plan.hotline_flags and asks_for_contact(message_text)
    plan.ack = _ack_key(merge, extra_changes)
    plan.answer_question = extraction.question
    data_changed = bool((merge and (merge.changes or merge.status_changes)) or extra_changes or new_flags)

    if not state.is_open:
        plan.notes.append(f"incident เป็น {state.status.value} ไม่มีการถามหรือส่งรายงาน")
    elif state.status == IncidentStatus.DRAFT and is_urgent(state):
        # ข้อ 1.2 — เหตุด่วนส่งทันที ไม่รอยืนยัน และไม่ขอยืนยันอีก
        plan.submit_trigger = "urgent_flags"
        plan.actions.append(NextAction.SUBMIT_REPORT)
        if Intent.CONFIRM_REPORT in extraction.intents:
            plan.notes.append("มี urgent flag: ส่งทันทีโดยไม่ต้องใช้ confirm_report")
    elif state.status == IncidentStatus.SUBMITTED:
        if data_changed:
            plan.actions.append(NextAction.SEND_UPDATE)
    else:  # draft ที่ไม่ด่วน
        _plan_draft(state, plan, extraction, pending_confirmation, data_changed)

    if plan.submit_trigger != "user_confirmed" and not plan.show_summary and not plan.keep_confirmation:
        plan.ask_fields = next_questions(state) if state.is_open else []
        if plan.ask_fields:
            plan.actions.append(NextAction.ASK_FOLLOW_UP)
        elif state.status == IncidentStatus.DRAFT and not is_urgent(state) and state.is_open:
            # หมดคำถามแล้ว → ขอยืนยันเท่าที่มี พร้อมบอกข้อมูลที่ขาด (ข้อ 6.9)
            plan.show_summary = True
            plan.summary_changed = confirmation_is_waiting(pending_confirmation, state)
            plan.actions.append(NextAction.CONFIRM_REPORT)

    if plan.show_summary:
        plan.missing_fields = missing_required(state)
    if plan.answer_question:
        plan.actions.append(NextAction.ANSWER_QUESTION)
    if not plan.actions:
        plan.actions.append(NextAction.NONE)
    return plan


def _plan_draft(
    state: IncidentState,
    plan: TurnPlan,
    extraction: ExtractionResult,
    pc: PendingConfirmation | None,
    data_changed: bool,
) -> None:
    waiting = confirmation_is_waiting(pc, state)
    unchanged = waiting and summary_fingerprint(state) == pc.summary_fingerprint
    wants_confirm = Intent.CONFIRM_REPORT in extraction.intents

    if wants_confirm and unchanged:
        plan.submit_trigger = "user_confirmed"
        plan.actions.append(NextAction.SUBMIT_REPORT)
        return
    if wants_confirm and not waiting:
        plan.notes.append("ไม่รับ confirm_report: ยังไม่มีคำขอยืนยันที่ส่งสำเร็จและรอคำตอบ")
    if waiting and not unchanged:
        # ข้อมูลเปลี่ยนหลังแสดงสรุป: ถ้ามีคำถามใหม่ให้ถามก่อน มิฉะนั้นแสดงสรุปใหม่ (ไม่ใช้สรุปเดิม)
        plan.notes.append("ข้อมูลเปลี่ยนหลังแสดงสรุป ต้องยืนยันสรุปใหม่")
        return
    if waiting and unchanged and not data_changed:
        if extraction.question:
            plan.keep_confirmation = True  # ผู้ใช้ถามความรู้แทรก — ตอบคำถาม แล้วยังรอคำตอบของสรุปเดิม
            plan.notes.append("ยังรอคำยืนยันสรุปเดิม")
        else:
            plan.show_summary = True  # ตอบอย่างอื่นที่ไม่ใช่การยืนยัน → แสดงสรุปเดิมอีกครั้ง
            plan.actions.append(NextAction.CONFIRM_REPORT)
            plan.notes.append("ยังไม่ได้รับคำยืนยัน แสดงสรุปเดิมอีกครั้ง")


def apply_submission(state: IncidentState, *, now: datetime) -> IncidentState:
    """เรียกเมื่อ handoff.py บันทึก incident_submitted ลง outbox แล้ว"""
    if state.status != IncidentStatus.DRAFT:
        raise ValueError(f"ส่งรายงานได้เฉพาะ draft (ตอนนี้เป็น {state.status.value})")
    return IncidentState.model_validate({**state.model_dump(), "status": IncidentStatus.SUBMITTED,
                                         "submitted_at": now, "updated_at": now})


# ---------------------------------------------------------------------------
# 5. ข้อความตอบ (ข้อ 6.8)
# ---------------------------------------------------------------------------
# [ข้อความด่วน + สายด่วน] [คำรับทราบ] [คำตอบความรู้] [รับรายงานแล้ว] [สรุป หรือ คำถาม]


def _people_text(present: bool, count: int | None, approx: bool, tpl: dict) -> str:
    if not present:
        return tpl["none"]
    if count is None:
        return tpl["present_no_count"]
    return tpl["count"].format(approx=tpl["approx"] if approx else "", count=count)


def _summary_value(state: IncidentState, name: IncidentFieldName, tpl: dict) -> str | None:
    f = state.tracked_field(name)
    if f.status in (FieldStatus.UNKNOWN, FieldStatus.DECLINED):
        return tpl["status"][f.status.value]
    if f.status != FieldStatus.KNOWN:
        return None
    v = f.value
    if name == F.LOCATION_AREA:
        return v.raw
    if name == F.LOCATION_LANDMARK:
        return v
    if name == F.LOCATION_COORDINATES:
        return tpl["coordinates_shared"]
    if name == F.SITUATION:
        return v.text
    if name == F.NEEDS:
        return ", ".join(tpl["needs"][n.value] for n in v)
    if name == F.AFFECTED_PEOPLE:
        return _people_text(v.present, v.count, v.is_approximate, tpl)
    if name == F.VULNERABLE_PEOPLE:
        if not v.present or not v.groups:
            return _people_text(v.present, None, False, tpl)
        return ", ".join(
            f"{tpl['vulnerable_types'][g.type.value]} {_people_text(True, g.count, g.is_approximate, tpl)}"
            for g in v.groups)
    if name == F.WATER_LEVEL:
        return v.raw
    return None


def render_summary(state: IncidentState, templates: dict, missing: Iterable[IncidentFieldName] = ()) -> str:
    """สรุปสำหรับผู้ใช้ ไม่มีเบอร์โทรหรือข้อมูลติดต่อ (usage.privacy)"""
    tpl = templates["report_summary"]
    lines = [tpl["line"].format(label=tpl["labels"][n.value], value=val)
             for n in SUMMARY_FIELDS if (val := _summary_value(state, n, tpl)) is not None]
    missing = list(missing)
    if missing:
        lines.append(tpl["missing"].format(fields=", ".join(tpl["labels"][n.value] for n in missing)))
    return "\n".join(lines)


def render_questions(state: IncidentState, fields: list[IncidentFieldName], templates: dict) -> str:
    """ใช้ combined_questions เมื่อ field ที่ถามตรงกับกลุ่ม มิฉะนั้นใช้คำถามรายข้อ เลือกแบบตามจำนวนครั้งที่ถามแล้ว"""
    names = [n.value for n in fields]
    for group in templates["combined_questions"].values():
        if sorted(group["fields"]) == sorted(names):
            return group["text"]
    parts = []
    for n in fields:
        variants = templates["questions"][n.value]
        parts.append(variants[min(state.tracked_field(n).ask_count, len(variants) - 1)])
    return " ".join(parts)


def render_reply(
    plan: TurnPlan,
    state: IncidentState,
    *,
    hotlines: Iterable[Hotline],
    templates: dict | None = None,
    ack_text: str | None = None,
    answer_text: str | None = None,
) -> str:
    """
    ack_text / answer_text: ข้อความที่ Gemini เรียบเรียงแล้วและผ่านการตรวจ ถ้าไม่มีใช้ template สำรอง
    state ต้องเป็นสถานะก่อน after_delivery (ask_count ยังไม่เพิ่ม)
    """
    t = templates or load_templates()
    hotlines = list(hotlines)
    parts: list[str] = []
    if plan.hotline_flags:
        parts.append(render_hotline_notice(plan.hotline_flags, hotlines, t))
    elif plan.contact_answer:
        parts.append(render_contact_answer(hotlines, t))
    if ack_text or plan.ack:
        parts.append(ack_text or t["acknowledgements"][plan.ack])
    if plan.answer_question:
        # สำรองเมื่อยังไม่มีคำตอบจาก RAG: แนะนำช่องทางติดต่อ (ไม่แสดงเบอร์ซ้ำถ้ามีข้อความด่วนแล้ว)
        parts.append(answer_text or t["knowledge_unavailable"].format(
            hotline_text="" if plan.hotline_flags else format_hotline_text(hotlines)).strip())
    if plan.submit_trigger:
        parts.append(t["report_submitted"][plan.submit_trigger])
    if plan.show_summary:
        summary = t["report_confirmation"].format(report_summary=render_summary(state, t, plan.missing_fields))
        parts.append(f"{t['summary_changed']}\n{summary}" if plan.summary_changed else summary)
    elif plan.ask_fields:
        parts.append(render_questions(state, plan.ask_fields, t))
    return "\n\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# 6. หลังส่งข้อความสำเร็จ
# ---------------------------------------------------------------------------


@dataclass
class DeliveryResult:
    state: IncidentState
    pending_question: PendingQuestion | None
    pending_confirmation: PendingConfirmation | None


def after_delivery(
    state: IncidentState,
    plan: TurnPlan,
    *,
    message_id: str,
    now: datetime,
    pending_question: PendingQuestion | None = None,
    pending_confirmation: PendingConfirmation | None = None,
) -> DeliveryResult:
    """
    เรียกเมื่อ LINE ยืนยันว่าส่งข้อความตอบสำเร็จเท่านั้น
    - นับการถาม (ask_count) และตั้ง pending question
    - สร้าง PendingConfirmation เมื่อข้อความนี้แสดงสรุป (คำขอยืนยันที่ "ส่งสำเร็จ")
    - บันทึก HotlineNotice
    ถ้าส่งไม่สำเร็จ ไม่ต้องเรียก: state เดิมยังถูกต้อง และ turn ถัดไปจะวางแผนใหม่เอง
    """
    if plan.ask_fields:
        state = record_questions_asked(state, plan.ask_fields, now=now)
        pending_question = PendingQuestion(incident_id=state.incident_id, field_names=plan.ask_fields,
                                           asked_at=now, asked_message_id=message_id)
    elif not plan.answer_question:
        pending_question = None  # ไม่ถามอะไรและไม่ได้ถามแทรก → ไม่มีคำถามค้าง

    if plan.show_summary:
        pending_confirmation = PendingConfirmation(incident_id=state.incident_id,
                                                   summary_fingerprint=summary_fingerprint(state),
                                                   asked_at=now, asked_message_id=message_id)
    elif not plan.keep_confirmation:
        pending_confirmation = None

    notices = list(state.hotline_notices)
    if plan.hotline_flags:
        notices.append(HotlineNotice(reason="new_flags", flags=plan.hotline_flags, incident_id=state.incident_id,
                                     message_id=message_id, notified_at=now))
    elif plan.contact_answer:
        notices.append(HotlineNotice(reason="user_asked", incident_id=state.incident_id,
                                     message_id=message_id, notified_at=now))
    if len(notices) != len(state.hotline_notices):
        state = IncidentState.model_validate({**state.model_dump(),
                                              "hotline_notices": [n.model_dump() for n in notices]})
    return DeliveryResult(state, pending_question, pending_confirmation)
