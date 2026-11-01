"""
agent.py — FloodAgent: ประกอบ safety + Extraction + memory + conversation เป็น agent ตัวเดียว (ขั้นที่ 5)

ใช้แบบเดียวกับ guy01.ipynb
    agent = FloodAgent(hotlines=HOTLINES)
    output = agent(input_text="น้ำท่วมบ้านค่ะ")
    output.output_text

แชร์ตำแหน่ง / กดปุ่ม (เตรียมไว้สำหรับ LINE)
    agent(location=(15.01, 100.34))
    agent(postback="location_consent=yes")

Runtime v1.3: repository + transactional outbox + explicit consent + draft priority.
Standalone calls simulate successful reply delivery. LINE uses defer_delivery=True and
confirm_delivery() only after the Reply API succeeds. Reviewed area/KB data are external.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta
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
from .memory import MergeResult, apply_updates, close_unanswered, unverified_area_resolver
from .runtime import RuntimeMixin
from .safety import redact_identifiers
from .safety import asks_for_contact, keyword_flags, load_templates, merge_urgent_flags, redact_phones, render_contact_answer
from .schemas import (
    AgentTurnOutput,
    ContactRecord,
    ContactConsentRequest,
    Message,
    IncidentStatus,
    CloseReason,
    ServiceScope,
    FieldChange,
    FieldChangeType,
    IncidentLinkedEvent,
    AreaMatchStatus,
    new_id,
    Relation,
    Coordinates,
    ExtractionResult,
    FieldStatus,
    FieldUpdate,
    Hotline,
    IncidentFieldName,
    IncidentState,
    Intent,
    LocationShareConsent,
    LocationShareRequest,
    MatchingStatus,
    NextAction,
    PendingQuestion,
    QuickReply,
    StrictModel,
    TrackedField,
    now_bkk,
)

F = IncidentFieldName
ExtractFn = Callable[[str, IncidentState | None, list[IncidentFieldName] | None], ExtractionResult]

# data ของปุ่ม Quick Reply (postback) — Backend อ่านตรง ๆ ไม่ผ่าน Gemini
POSTBACK_SKIP_LOCATION = "location_share=skip"
POSTBACK_ALLOW_LOCATION = "location_consent=yes"
POSTBACK_DENY_LOCATION = "location_consent=no"
LOCATION_REQUEST_TTL = timedelta(hours=24)  # แชร์หลังจากนี้ ถือว่าไม่ชัดว่าตอบคำขอใด


class LocationInput(StrictModel):
    """ตำแหน่งที่ผู้ใช้แชร์ (เทียบได้กับ LINE location message) — address ไม่ใช้"""

    latitude: float
    longitude: float

    @classmethod
    def parse(cls, value) -> "LocationInput":
        if isinstance(value, cls):
            return value
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return cls(latitude=value[0], longitude=value[1])
        if isinstance(value, dict):
            return cls(latitude=value["latitude"], longitude=value["longitude"])
        raise ValueError("location ต้องเป็น (latitude, longitude) หรือ {'latitude': ..., 'longitude': ...}")


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


class FloodAgent(RuntimeMixin, dspy.Module):
    def __init__(
        self,
        hotlines: Iterable[Hotline],
        *,
        templates: dict | None = None,
        extractor: ExtractFn | None = None,
        internal_user_id: str = "usr_demo",
        repository=None, sender=None, composer=None, knowledge=None, area_resolver=None,
    ):
        super().__init__()
        self.hotlines = list(hotlines)
        if not self.hotlines:
            raise ValueError("ต้องมี hotlines อย่างน้อยหนึ่งรายการ ข้อความเหตุด่วนต้องแสดงช่องทางฉุกเฉินได้เสมอ")
        self.templates = templates or load_templates()
        real_extractor = extractor is None
        if extractor is None:
            from .extraction import Extractor  # โหลดเฉพาะเมื่อใช้ Gemini จริง
            extractor = Extractor()
        self.extract = extractor
        self.internal_user_id = internal_user_id
        self.reset()
        if composer is None and real_extractor:
            from .response import ResponseComposer
            composer = ResponseComposer()
        self._runtime_init(repository, sender, composer, knowledge, area_resolver)

    # ---------------------------------------------------------------- state
    def reset(self):
        """เริ่มบทสนทนาใหม่ (เหมือนผู้ใช้คนใหม่)"""
        self.conversation_id = f"conv_{self.internal_user_id}"
        self.incidents: dict[str, IncidentState] = {}
        self.active_id: str | None = None
        self.pending_question: PendingQuestion | None = None
        self.pending_confirmation: PendingConfirmation | None = None
        self.contacts: list[ContactRecord] = []  # เบอร์จริงอยู่ที่นี่เท่านั้น ไม่เข้า state และไม่ถึง Gemini
        self.location_requests: list[LocationShareRequest] = []  # ข้อ 6.10 คำขอพิกัดที่ส่งสำเร็จ
        self.consents: list[LocationShareConsent] = []  # ข้อ 6.10 การอนุญาตส่งพิกัด
        self.turns: list[TurnLog] = []
        if hasattr(self, "repository"):
            self._runtime_defaults()
            self.persist()

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
    def forward(self, input_text=None, *, location=None, postback=None, defer_delivery=False):
        self._defer_delivery = defer_delivery
        self._delivery_pending = None
        self.last_sources = []
        output = self._forward_impl(input_text, location=location, postback=postback)
        return self._public_reply(output)

    def _forward_impl(
        self,
        input_text: str | None = None,
        *,
        location: LocationInput | tuple[float, float] | dict | None = None,
        postback: str | None = None,
    ) -> AgentTurnOutput:
        """
        ข้อความ 1 ข้อความจากผู้ใช้ เลือกอย่างใดอย่างหนึ่ง
        - input_text: ข้อความตัวอักษร
        - location: ผู้ใช้แชร์ตำแหน่ง (LINE location message) เช่น (15.01, 100.34)
        - postback: ผู้ใช้กดปุ่ม Quick Reply เช่น "location_consent=yes"
        """
        if sum(x is not None for x in (input_text, location, postback)) != 1:
            raise ValueError("ส่ง input_text, location หรือ postback อย่างใดอย่างหนึ่ง")
        if location is not None:
            return self._on_location(LocationInput.parse(location))
        if postback is not None:
            return self._on_postback(postback)
        if (pb := self._label_as_postback(input_text)) is not None:
            return self._on_postback(pb)  # ผู้ใช้พิมพ์ข้อความเดียวกับป้ายปุ่ม แทนการกดปุ่ม
        return self._on_text(input_text)

    def _on_text(self, input_text: str) -> AgentTurnOutput:
        now = now_bkk()
        msg_id = self._next_msg_id()
        red = redact_phones(redact_identifiers(input_text))  # filter identifiers before interpreting phone-like sequences
        red = replace(red, text=redact_identifiers(red.text))
        log = TurnLog(user_text=red.text, redacted_text=red.text, keyword_flags=keyword_flags(red.text))
        self.turns.append(log)

        state = self.state if (self.state and self.state.is_open) else None
        pending = self.pending_question.field_names if self.pending_question and state else None
        if input_text.strip() == "ลบข้อมูลของฉัน":
            self.delete_personal_data(now)
            return AgentTurnOutput(output_text=self.templates["data_deleted"], intents=[Intent.DELETE_MY_DATA],
                                   incident_id=None, incident_status=None, next_actions=[NextAction.DELETE_USER_DATA])
        safe_state = state.model_copy(deep=True) if state else None
        if safe_state:
            # Defense in depth: no precise coordinates in an extractor's input object.
            safe_state.location_coordinates = TrackedField[Coordinates]()
        if hasattr(self.extract, "context"):
            self.extract.context = self.context_for_extraction(state, pending, now)
        try:
            r = self.extract(red.text, safe_state, pending)  # 2. ✦ Gemini
        except Exception:
            if not log.keyword_flags:
                raise
            r = ExtractionResult(intents=[Intent.REPORT_FLOOD], target="active_incident" if state else "new_incident",
                                 urgent_flags=sorted(log.keyword_flags, key=lambda f:f.value))
        self.messages.append(Message(conversation_id=self.conversation_id, role="user", text=red.text,
                                     incident_ids=[state.incident_id] if state else []))
        if Intent.DELETE_MY_DATA in r.intents:
            self.delete_personal_data(now)
            return AgentTurnOutput(output_text=self.templates["data_deleted"], intents=r.intents,
                                   incident_id=None, incident_status=None, next_actions=[NextAction.DELETE_USER_DATA])
        if state and Intent.CHECK_STATUS in r.intents:
            status = self.delivery_status(state.incident_id)
            return AgentTurnOutput(output_text=self.templates["delivery_status"][status or "pending"], intents=r.intents,
                                   incident_id=state.incident_id, incident_status=state.status,
                                   delivery_status=status, next_actions=[NextAction.NONE])
        if state and state.status == IncidentStatus.SUBMITTED and (Intent.RESOLVE_INCIDENT in r.intents or Intent.WITHDRAW_INCIDENT in r.intents):
            st = self.close_incident(CloseReason.USER_WITHDRAWN if Intent.WITHDRAW_INCIDENT in r.intents else CloseReason.USER_RESOLVED, now)
            return AgentTurnOutput(output_text=self.templates["incident_closed"], intents=r.intents,
                                   incident_id=st.incident_id, incident_status=st.status, next_actions=[NextAction.CLOSE_INCIDENT])
        log.extraction = r

        # Reject a clearly out-of-province ordinary report before creating an incident.
        if self.area_resolver:
            from .schemas import AreaMention
            area_update = next((u for u in r.updates if u.field == F.LOCATION_AREA and u.status == FieldStatus.KNOWN), None)
            if area_update:
                mention = AreaMention(name=area_update.value) if isinstance(area_update.value, str) else AreaMention.model_validate(area_update.value)
                resolved = self.area_resolver(mention)
                if resolved.match_status.value == "out_of_province" and not (log.keyword_flags or r.urgent_flags) and state is None:
                    from .safety import format_hotline_text
                    return AgentTurnOutput(output_text=self.templates["service_scope"]["outside_general"].format(hotline_text=format_hotline_text(self.hotlines)),
                                           intents=r.intents, incident_id=None, incident_status=None, next_actions=[NextAction.NONE])
        # 3. เลือก incident ที่ข้อความนี้เป็นของ
        is_urgent_msg = bool(log.keyword_flags or r.urgent_flags)
        has_incident_info = bool(r.updates or is_urgent_msg)
        if state is not None and self._other_incident(r, state, red.text):
            if not is_urgent_msg:
                return self._ask_incident_match(log, r, state)
            # ข้อ 3.2 — เหตุด่วนไม่หน่วงด้วยการถาม: สร้าง provisional incident ผูกกับเหตุเดิมแบบ unconfirmed
            state = self._new_incident(created_at=now, updated_at=now, last_activity_at=now, matching_status=MatchingStatus.UNCONFIRMED,
                                       possible_related_incident_ids=[state.incident_id])
            log.notes.append("ไม่ชัดว่าเหตุไหน + เหตุด่วน → สร้าง provisional incident (matching_status=unconfirmed)")
        elif state is None:
            if not has_incident_info:
                return self._no_incident_reply(log, r, red.text)
            state = self._new_incident(created_at=now, updated_at=now, last_activity_at=now)

        # 4. merge + flags + needs + คำถามที่ไม่ได้คำตอบ
        m = apply_updates(state, r.updates, message_text=red.text, message_id=msg_id, now=now,
                          area_resolver=self.area_resolver or unverified_area_resolver)
        log.merge = m
        fl = merge_urgent_flags(m.state.urgent_flags, log.keyword_flags, r.urgent_flags,
                                detected_at=now, message_id=msg_id)
        log.new_flags = fl.newly_detected
        st = IncidentState.model_validate({**m.state.model_dump(),
                                           "urgent_flags": [x.model_dump() for x in fl.records]})
        st, needs_change = reconcile_needs(st, fl.newly_detected, now=now, message_id=msg_id)
        if needs_change:
            log.notes.append("help_requested ใหม่: เอา info_only ที่ขัดกับคำขอออก (ไม่เดา rescue)")
        if red.has_phone and not self.deletion_requested_at:
            st = self._save_contact(st, red, now)
            log.notes.append("พบเบอร์: รอยืนยันอนุญาต ไม่แนบรายงาน ไม่ส่ง Gemini")
        if self.pending_location:
            stored = self.pending_location
            self.pending_location = None
            if now - datetime.fromisoformat(stored["received_at"]) <= LOCATION_REQUEST_TTL:
                loc_merge = apply_updates(st, [FieldUpdate(field=F.LOCATION_COORDINATES, status=FieldStatus.KNOWN, value=stored["coordinates"])],
                                          message_text="[ตำแหน่ง]", message_id=stored["coordinates"]["message_id"], now=now)
                st = loc_merge.state
        st = close_unanswered(st, self.pending_question)
        return self._finish(st, log, r, m, msg_id=msg_id, now=now, new_flags=fl.newly_detected,
                            extra_changes=[needs_change] if needs_change else [], message_text=red.text)

    def _finish(
        self,
        st: IncidentState,
        log: TurnLog,
        r: ExtractionResult,
        m: MergeResult | None,
        *,
        msg_id: str,
        now,
        new_flags=(),
        extra_changes=(),
        message_text: str = "",
        ack_text: str | None = None,
    ) -> AgentTurnOutput:
        """5. วางแผน → ส่งรายงาน → ข้อความตอบ → บันทึกหลังส่งสำเร็จ (ใช้ร่วมกันทุกชนิดข้อความ)"""
        st = st.model_copy(update={"updated_at":now})
        area = st.location_area.value
        if area and area.match_status.value in {"ambiguous", "not_found"} and st.location_area.ask_count >= 2:
            area = area.model_copy(update={"match_status":AreaMatchStatus.UNVERIFIED, "candidates":[]})
            st = st.model_copy(update={"location_area":st.location_area.model_copy(update={"value":area})})
        plan = plan_turn(st, extraction=r, merge=m, new_flags=new_flags, extra_changes=extra_changes,
                         pending_confirmation=self.pending_confirmation, message_text=message_text)
        log.plan = plan
        if st.service_scope == ServiceScope.OUT_OF_PROVINCE and st.status == IncidentStatus.DRAFT and not st.urgent_flags:
            st = IncidentState.model_validate({**st.model_dump(), "status":"closed", "close_reason":"out_of_service_area", "closed_at":now})
            self.incidents[st.incident_id] = st
            from .safety import format_hotline_text
            text = self.templates["service_scope"]["outside_general"].format(hotline_text=format_hotline_text(self.hotlines))
            return self._output(text, r, st, TurnPlan(actions=[NextAction.NONE]))
        if plan.submit_trigger:
            st = apply_submission(st, now=now)
        st = self._report_event(st, plan, m, now)
        unsolicited = bool(st.location_coordinates.value and not st.location_consent_id and not st.coordinates_forwarding_declined
                           and not self._pending_request("ask_location", now) and not self._pending_request("confirm_unsolicited", now))
        if unsolicited and st.urgent_flags and st.location_area.value:
            plan.ask_fields = []
            plan.actions = [a for a in plan.actions if a != NextAction.ASK_FOLLOW_UP]
        ack, questions, answer = self._wording(plan, st, message_text, changes=[*(m.changes if m else []), *extra_changes])
        status = self.delivery_status(st.incident_id)
        delivery_text = self.templates["delivery_status"][status] if status and (plan.submit_trigger or NextAction.SEND_UPDATE in plan.actions) else None
        text = render_reply(plan, st, hotlines=self.hotlines, templates=self.templates, ack_text=ack_text or ack,
                            question_text=questions, answer_text=answer, delivery_text=delivery_text)
        quick = self._location_buttons() if F.LOCATION_COORDINATES in plan.ask_fields else []
        reply_id = f"{msg_id}_reply"
        # Store the plan, not a successful delivery, until LINE confirms.
        self._delivery_pending = json.loads(json.dumps(dict(plan=asdict(plan),
            incident_id=st.incident_id, reply_id=reply_id, now=now.isoformat(),
            ask_location=bool(quick), purpose=text), default=lambda x:x.value))
        if unsolicited and (not st.urgent_flags or st.location_area.value):
            text += "\n\n" + self.templates["location_consent_confirm"]
            quick = self._consent_buttons()
            self._delivery_pending["unsolicited_location"] = True
        contact_req = self._pending_contact()
        if contact_req and not contact_req.request_message_id and (not st.urgent_flags or plan.ask_fields == [F.CONTACT] or (not plan.ask_fields and not unsolicited)):
            if plan.ask_fields == [F.CONTACT]:
                from .conversation import render_questions
                text = text.replace(render_questions(st, plan.ask_fields, self.templates), self.templates["contact_consent"])
            else:
                text += "\n\n" + self.templates["contact_consent"]
            quick += [QuickReply(type="postback", label=self.templates["quick_replies"]["allow_contact"], data=f"contact_consent=yes:{contact_req.request_id}"),
                      QuickReply(type="postback", label=self.templates["quick_replies"]["deny_contact"], data=f"contact_consent=no:{contact_req.request_id}")]
            self._delivery_pending["contact_request_id"] = contact_req.request_id
        if st.matching_status == MatchingStatus.UNCONFIRMED:
            for target in st.possible_related_incident_ids[:2]:
                quick += [QuickReply(type="postback", label="เหตุเดียวกัน", data=f"incident_link=same_incident:{target}"),
                          QuickReply(type="postback", label="เหตุเกี่ยวข้องกัน", data=f"incident_link=related:{target}")]
        if not self._defer_delivery:
            self.confirm_delivery()
        return self._output(text, r, st, plan, quick_replies=quick)

    # ---------------------------------------------------------------- ตำแหน่ง (ข้อ 6.10)
    def _next_msg_id(self) -> str:
        return new_id("msg")

    def _location_buttons(self) -> list[QuickReply]:
        q = self.templates["quick_replies"]
        return [QuickReply(type="location", label=q["share_location"]),
                QuickReply(type="postback", label=q["skip_location"], data=POSTBACK_SKIP_LOCATION)]

    def _consent_buttons(self) -> list[QuickReply]:
        q = self.templates["quick_replies"]
        return [QuickReply(type="postback", label=q["allow_location"], data=POSTBACK_ALLOW_LOCATION),
                QuickReply(type="postback", label=q["deny_location"], data=POSTBACK_DENY_LOCATION)]

    def _open_location_request(self, st: IncidentState, *, kind: str, purpose: str, reply_id: str, now) -> None:
        for i, old in enumerate(self.location_requests):  # คำขอเดิมที่ยังค้าง → หมดอายุ (ตอบได้แค่คำขอล่าสุด)
            if old.incident_id == st.incident_id and old.status == "pending":
                self.location_requests[i] = old.model_copy(update={"status": "expired", "resolved_at": now})
        version_key = "location_share" if kind == "ask_location" else "location_consent_confirm"
        self.location_requests.append(LocationShareRequest(
            incident_id=st.incident_id, kind=kind, notice_version=self.templates["notice_versions"][version_key],
            purpose=purpose, request_message_id=reply_id, requested_at=now))

    def _pending_request(self, kind: str, now) -> LocationShareRequest | None:
        """คำขอล่าสุดของเหตุที่กำลังคุย ที่ยังรอคำตอบและไม่เกิน 24 ชั่วโมง"""
        st = self.state
        if st is None or not st.is_open:
            return None
        for i in range(len(self.location_requests) - 1, -1, -1):
            req = self.location_requests[i]
            if req.incident_id != st.incident_id or req.status != "pending":
                continue
            if now - req.requested_at > LOCATION_REQUEST_TTL:
                self.location_requests[i] = req.model_copy(update={"status": "expired", "resolved_at": now})
                return None
            return req if req.kind == kind else None
        return None

    def _resolve_request(self, req: LocationShareRequest, status: str, *, msg_id: str | None, now) -> LocationShareRequest:
        done = req.model_copy(update={"status": status, "response_message_id": msg_id, "resolved_at": now})
        LocationShareRequest.model_validate(done.model_dump())  # ตรวจกฎของ schema
        self.location_requests[self.location_requests.index(req)] = done
        return done

    def _grant(self, st: IncidentState, req: LocationShareRequest, *, msg_id: str, now) -> IncidentState:
        consent = LocationShareConsent.from_request(self._resolve_request(req, "granted", msg_id=msg_id, now=now))
        self.consents.append(consent)
        return IncidentState.model_validate({**st.model_dump(), "location_consent_id": consent.consent_id})

    def _on_location(self, loc: LocationInput) -> AgentTurnOutput:
        """ผู้ใช้แชร์ตำแหน่ง — ไม่เรียก Gemini (พิกัดไม่ต้องตีความ และไม่ส่ง Gemini)"""
        now, msg_id = now_bkk(), self._next_msg_id()
        log = TurnLog(user_text="[ตำแหน่ง]", redacted_text="[ตำแหน่ง]")
        self.turns.append(log)
        r = ExtractionResult(intents=[Intent.UPDATE_INCIDENT], target="active_incident")
        log.extraction = r

        req = self._pending_request("ask_location", now)  # ต้องหาก่อนสร้างเหตุใหม่
        st = self.state if (self.state and self.state.is_open) else None
        if st is None:
            self.pending_location = {"received_at": now.isoformat(), "coordinates": {"latitude":loc.latitude, "longitude":loc.longitude,
                                     "source":"line_location", "message_id":msg_id}}
            return AgentTurnOutput(output_text=self.templates["location_without_incident"], intents=[Intent.GENERAL_CONVERSATION],
                                   incident_id=None, incident_status=None, next_actions=[NextAction.NONE])
        coords = {"latitude": loc.latitude, "longitude": loc.longitude, "source": "line_location", "message_id": msg_id}
        # address จาก LINE ไม่เก็บ (ข้อมูลส่วนบุคคล และพิกัดเพียงพอสำหรับรายงาน)
        m = apply_updates(st, [FieldUpdate(field=F.LOCATION_COORDINATES, status=FieldStatus.KNOWN, value=coords)],
                          message_text="[ตำแหน่ง]", message_id=msg_id, now=now)
        log.merge = m
        st = m.state

        if req is not None:
            st = self._grant(st, req, msg_id=msg_id, now=now)
            log.notes.append("แชร์ตอบคำขอที่ส่งไปแล้ว → บันทึก LocationShareConsent (แนบพิกัดในรายงานได้)")
            st = close_unanswered(st, self.pending_question)
            return self._finish(st, log, r, m, msg_id=msg_id, now=now,
                                ack_text=self.templates["location_received"])

        # ข้อ 6.10 — ส่งมาเองก่อนเห็นข้อความแจ้ง ยังไม่ถือว่าอนุญาต ต้องถามยืนยันก่อน
        log.notes.append("แชร์มาเองโดยไม่มีคำขอที่รอ → ยังไม่ถือว่าอนุญาต ถามยืนยันก่อน")
        self.incidents[st.incident_id] = st
        reply_id = f"{msg_id}_reply"
        text = f"{self.templates['location_received']}\n\n{self.templates['location_consent_confirm']}"
        self._delivery_pending = dict(unsolicited_location=True, incident_id=st.incident_id, reply_id=reply_id, now=now.isoformat())
        if not self._defer_delivery:
            self.confirm_delivery()
        return AgentTurnOutput(output_text=text, intents=r.intents, incident_id=st.incident_id,
                               incident_status=st.status, matching_status=st.matching_status,
                               urgent_flags=sorted(st.urgent_flag_types, key=lambda f: f.value),
                               next_actions=[NextAction.NONE], pending_question=self.pending_question,
                               quick_replies=self._consent_buttons(), model=self._model())

    def _on_postback(self, data: str) -> AgentTurnOutput:
        """ผู้ใช้กดปุ่ม — Backend ตัดสินเอง ไม่เรียก Gemini"""
        now, msg_id = now_bkk(), self._next_msg_id()
        safe_data = redact_identifiers(redact_phones(data).text)
        log = TurnLog(user_text=f"[กดปุ่ม {safe_data}]", redacted_text=f"[กดปุ่ม {safe_data}]")
        self.turns.append(log)
        r = ExtractionResult(intents=[Intent.UPDATE_INCIDENT], target="active_incident")
        log.extraction = r
        if data.startswith("incident_link="):
            relation, _, target = data.removeprefix("incident_link=").partition(":")
            st = self.state
            if st and st.matching_status == MatchingStatus.UNCONFIRMED and target in st.possible_related_incident_ids and relation in {"same_incident", "related"}:
                st = st.model_copy(update={"matching_status":MatchingStatus.LINKED, "related_incident_id":target,
                                           "incident_version":st.incident_version+1})
                self.incidents[st.incident_id] = st
                event = IncidentLinkedEvent(incident_id=st.incident_id, case_version=st.incident_version, linked_to=target,
                                             relation=relation, occurred_at=now)
                self.persist([event])
                return self._finish(st, log, r, None, msg_id=msg_id, now=now)
        if data in {"incident_match=new", "incident_match=same"} and self.match_candidate:
            candidate = self.match_candidate
            self.match_candidate = None
            r = ExtractionResult.model_validate(candidate["extraction"])
            st = self._new_incident(created_at=now, updated_at=now, last_activity_at=now) if data.endswith("=new") else self.state
            if st:
                m = apply_updates(st, r.updates, message_text=candidate["text"], message_id=msg_id, now=now,
                                  area_resolver=self.area_resolver or unverified_area_resolver)
                return self._finish(m.state, log, r, m, msg_id=msg_id, now=now, message_text=candidate["text"])
        if data.startswith("contact_consent="):
            st = self._contact_postback(data, now, msg_id)
            if st is not None:
                return self._finish(st, log, r, None, msg_id=msg_id, now=now)
        kind = {POSTBACK_SKIP_LOCATION: "ask_location", POSTBACK_ALLOW_LOCATION: "confirm_unsolicited",
                POSTBACK_DENY_LOCATION: "confirm_unsolicited"}.get(data)
        req = self._pending_request(kind, now) if kind else None
        if req is None:
            log.notes.append("ปุ่มของคำขอที่ไม่ได้รอคำตอบแล้ว → ไม่ทำอะไร")
            st = self.state
            return AgentTurnOutput(output_text=self.templates["request_expired"], intents=r.intents,
                                   incident_id=st.incident_id if st else None, incident_status=st.status if st else None,
                                   next_actions=[NextAction.NONE], pending_question=self.pending_question,
                                   model=self._model())
        st = self.state
        if data == POSTBACK_ALLOW_LOCATION:
            st = self._grant(st, req, msg_id=msg_id, now=now)
            log.notes.append("อนุญาตพิกัดที่ส่งมาก่อน → บันทึก LocationShareConsent")
            return self._finish(st, log, r, None, msg_id=msg_id, now=now,
                                ack_text=self.templates["acknowledgements"]["received"])
        if data == POSTBACK_DENY_LOCATION:
            self._resolve_request(req, "declined", msg_id=msg_id, now=now)
            st = self._remove_coordinates(st, msg_id=msg_id, now=now)
            st = st.model_copy(update={"coordinates_forwarding_declined":True})
            log.notes.append("ไม่อนุญาต → ลบพิกัดออกจาก state และไม่ขอพิกัดอีก")
            return self._finish(st, log, r, None, msg_id=msg_id, now=now,
                                ack_text=self.templates["location_consent_declined"])
        # ข้ามการแชร์ตำแหน่ง = ไม่สะดวกให้ข้อมูลส่วนนี้
        self._resolve_request(req, "skipped", msg_id=msg_id, now=now)
        m = apply_updates(st, [FieldUpdate(field=F.LOCATION_COORDINATES, status=FieldStatus.DECLINED)],
                          message_text="[ข้าม]", message_id=msg_id, now=now)
        log.merge = m
        log.notes.append("ข้ามการแชร์ตำแหน่ง → location_coordinates = declined")
        return self._finish(m.state, log, r, m, msg_id=msg_id, now=now)

    def _remove_coordinates(self, st: IncidentState, *, msg_id: str, now) -> IncidentState:
        f = st.location_coordinates
        removed = TrackedField[Coordinates](
            status=FieldStatus.DECLINED, updated_at=now, source_message_id=msg_id,
            ask_count=f.ask_count, last_asked_at=f.last_asked_at)  # ไม่เก็บพิกัดไว้ใน history ด้วย
        return IncidentState.model_validate({**st.model_dump(), "location_coordinates": removed.model_dump()})

    def _label_as_postback(self, text: str) -> str | None:
        """ผู้ใช้พิมพ์ "อนุญาต" / "ไม่อนุญาต" / "ข้าม" เองขณะมีคำขอที่รออยู่ → ถือเป็นการกดปุ่มนั้น"""
        now, q, t = now_bkk(), self.templates["quick_replies"], text.strip()
        contact = self._pending_contact()
        if contact and text.strip() in {self.templates["quick_replies"]["allow_contact"], self.templates["quick_replies"]["deny_contact"]}:
            choice = "yes" if text.strip() == self.templates["quick_replies"]["allow_contact"] else "no"
            return f"contact_consent={choice}:{contact.request_id}"
        if self._pending_request("confirm_unsolicited", now):
            return {q["allow_location"]: POSTBACK_ALLOW_LOCATION, q["deny_location"]: POSTBACK_DENY_LOCATION}.get(t)
        if self._pending_request("ask_location", now):
            return POSTBACK_SKIP_LOCATION if t == q["skip_location"] else None
        return None


    # ---------------------------------------------------------------- กรณีพิเศษ
    def _other_incident(self, r: ExtractionResult, state: IncidentState, message_text="") -> bool:
        if r.target == "unclear":
            return True
        active = state.location_area.value
        if active is None:
            return False
        from .schemas import AreaMention
        from .safety import correction_cue_for
        norm = lambda x: "".join(str(x).split())
        for update in r.updates:
            if update.field != F.LOCATION_AREA or not update.value:
                continue
            if update.is_correction and correction_cue_for(update.raw, message_text):
                continue
            mention = AreaMention(name=update.value) if isinstance(update.value,str) else AreaMention.model_validate(update.value)
            resolved = (self.area_resolver or unverified_area_resolver)(mention)
            if active.area_code and resolved.area_code:
                if active.area_code != resolved.area_code:
                    return True
                continue
            if norm(mention.name) not in norm(active.raw) and norm(active.raw) not in norm(mention.name):
                return True
        return False

    def _save_contact(self, st: IncidentState, red, now) -> IncidentState:
        # Explicit consent is separate from incident confirmation and urgent submission.
        if st.contact_ask.status == FieldStatus.DECLINED or st.contact_ref:
            return st
        for old in self.contact_requests:
            if old.incident_id == st.incident_id and old.status == "pending":
                old.phone, old.status, old.resolved_at = "", "expired", now
        self.contact_requests.append(ContactConsentRequest(incident_id=st.incident_id, phone=red.phones[0].normalized,
            purpose=self.templates["contact_consent"], notice_version=self.templates["notice_versions"]["contact"], requested_at=now))
        return st

    def _ask_incident_match(self, log: TurnLog, r: ExtractionResult, state: IncidentState) -> AgentTurnOutput:
        # ยังไม่มีขั้นตอนรับคำตอบ "เหตุเดิม/เหตุใหม่" (รอ repository.py) — ข้อความนี้ยังไม่ถูก merge
        log.notes.append(f"target={r.target} คนละพื้นที่/ไม่ชัด: ถามเหตุเดิมหรือเหตุใหม่ ยังไม่ merge ข้อความนี้")
        self.match_candidate = {"extraction":r.model_dump(mode="json"), "text":log.redacted_text}
        area = state.location_area.value.raw if state.location_area.value else ""
        text = self.templates["incident_match"].format(active_area=area or "ก่อนหน้านี้")
        return AgentTurnOutput(output_text=text, intents=r.intents, incident_id=state.incident_id,
                               incident_status=state.status, matching_status=state.matching_status,
                               urgent_flags=sorted(state.urgent_flag_types, key=lambda f: f.value),
                               next_actions=[NextAction.ASK_INCIDENT_MATCH], model=self._model(),
                               quick_replies=[QuickReply(type="postback", label="เหตุเดิม", data="incident_match=same"),
                                              QuickReply(type="postback", label="เหตุใหม่", data="incident_match=new")])

    def _no_incident_reply(self, log: TurnLog, r: ExtractionResult, text: str) -> AgentTurnOutput:
        """ทักทาย ถามความรู้ หรือถามช่องทางติดต่อ โดยยังไม่มีเหตุ — ไม่สร้าง incident"""
        t, parts, actions = self.templates, [], []
        if asks_for_contact(text):
            parts.append(render_contact_answer(self.hotlines, t))
        if r.question:
            from .safety import format_hotline_text
            plan = TurnPlan(actions=[NextAction.ANSWER_QUESTION], answer_question=r.question)
            _, _, answer = self._wording(plan, None, text)
            parts.append(answer or t["knowledge_unavailable"].format(hotline_text=format_hotline_text(self.hotlines)))
            actions.append(NextAction.ANSWER_QUESTION)
        if not parts:
            parts.append(t["greeting"])
        log.notes.append("ไม่มีข้อมูลเหตุการณ์ ยังไม่สร้าง incident")
        return AgentTurnOutput(output_text="\n\n".join(parts), intents=r.intents, incident_id=None,
                               incident_status=None, next_actions=actions or [NextAction.NONE],
                               used_hotline_template=asks_for_contact(text), model=self._model())

    def _output(self, text: str, r: ExtractionResult, st: IncidentState, plan: TurnPlan,
                quick_replies: list[QuickReply] | None = None) -> AgentTurnOutput:
        return AgentTurnOutput(
            output_text=text, intents=r.intents, incident_id=st.incident_id, incident_status=st.status,
            matching_status=st.matching_status,
            urgent_flags=sorted(st.urgent_flag_types, key=lambda f: f.value),
            missing_fields=plan.missing_fields, next_actions=plan.actions,
            pending_question=self.pending_question, used_hotline_template=bool(plan.hotline_flags),
            delivery_status=self.delivery_status(st.incident_id),
            quick_replies=quick_replies or [], model=self._model(),
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
        reqs = [r for r in self.location_requests if r.incident_id == s.incident_id]
        if reqs:
            print("  คำขอพิกัด:", [f"{r.kind}:{r.status}" for r in reqs])
        if s.location_consent_id:
            print("  อนุญาตแนบพิกัดแล้ว:", s.location_consent_id)
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
