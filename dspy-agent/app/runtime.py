"""Runtime integration for persistence, consent, handoff and controlled wording."""
from dataclasses import asdict
from datetime import datetime, timedelta
import json
from pathlib import Path

from .schemas import *
from .conversation import TurnPlan, PendingConfirmation, after_delivery
from .handoff import build_report, changes_between, HandoffDispatcher
from .knowledge import KnowledgeBase
from .priority import evaluate_priority
from .repository import Repository
from .response import validate_wording
from .safety import redact_identifiers, redact_phones


class RuntimeMixin:
    def _runtime_init(self, repository=None, sender=None, composer=None, knowledge=None, area_resolver=None):
        self.repository = repository or Repository()
        self.dispatcher = HandoffDispatcher(self.repository, sender)
        self.composer = composer
        self.knowledge = knowledge or KnowledgeBase(Path(__file__).with_name("knowledge_th.csv"))
        self.area_resolver = area_resolver
        self._runtime_defaults()
        snapshot = self.repository.load(self.internal_user_id)
        if snapshot:
            self._restore(snapshot)

    def _runtime_defaults(self):
        self.user_active = True
        self.reporter_ref = new_id("rep")
        self.seen_notice_versions = []
        self.contact_requests = []
        self.pending_location = None
        self.last_reports = {}
        self.messages = []
        self.deletion_requested_at = None
        self._delivery_pending = None
        self._defer_delivery = False
        self.match_candidate = None
        self.area_suggestion = None  # ชื่อพื้นที่ที่เสนอให้ผู้ใช้ยืนยัน {incident_id, codes}
        self.last_sources = []
        self.conversation_summary = None

    def _snapshot(self):
        dump = lambda values: [x.model_dump(mode="json") for x in values]
        return dict(conversation_id=self.conversation_id, reporter_ref=self.reporter_ref, user_active=self.user_active,
                    incidents={k:v.model_dump(mode="json") for k,v in self.incidents.items()}, active_id=self.active_id,
                    pending_question=self.pending_question.model_dump(mode="json") if self.pending_question else None,
                    pending_confirmation=self.pending_confirmation.model_dump(mode="json") if self.pending_confirmation else None,
                    contacts=dump(self.contacts), location_requests=dump(self.location_requests), consents=dump(self.consents),
                    contact_requests=dump(self.contact_requests), seen_notice_versions=self.seen_notice_versions,
                    pending_location=self.pending_location, last_reports={k:v.model_dump(mode="json") for k,v in self.last_reports.items()},
                    messages=dump(self.messages), conversation_summary=self.conversation_summary.model_dump(mode="json") if self.conversation_summary else None, deletion_requested_at=self.deletion_requested_at.isoformat() if self.deletion_requested_at else None,
                    delivery_pending=self._delivery_pending, match_candidate=self.match_candidate,
                    area_suggestion=self.area_suggestion)

    def _restore(self, data):
        self.conversation_id = data["conversation_id"]
        self.reporter_ref = data["reporter_ref"]
        self.user_active = data.get("user_active", True)
        self.incidents = {k:IncidentState.model_validate(v) for k,v in data["incidents"].items()}
        self.active_id = data.get("active_id")
        self.pending_question = PendingQuestion.model_validate(data["pending_question"]) if data.get("pending_question") else None
        self.pending_confirmation = PendingConfirmation.model_validate(data["pending_confirmation"]) if data.get("pending_confirmation") else None
        for key, cls in (("contacts", ContactRecord), ("location_requests", LocationShareRequest),
                         ("consents", LocationShareConsent), ("contact_requests", ContactConsentRequest), ("messages", Message)):
            setattr(self, key, [cls.model_validate(v) for v in data.get(key, [])])
        self.conversation_summary = ConversationSummary.model_validate(data["conversation_summary"]) if data.get("conversation_summary") else None
        self.last_reports = {k:IncidentReport.model_validate(v) for k,v in data.get("last_reports", {}).items()}
        self.seen_notice_versions = data.get("seen_notice_versions", [])
        self.pending_location = data.get("pending_location")
        self.deletion_requested_at = datetime.fromisoformat(data["deletion_requested_at"]) if data.get("deletion_requested_at") else None
        self._delivery_pending = data.get("delivery_pending")
        self.match_candidate = data.get("match_candidate")
        self.area_suggestion = data.get("area_suggestion")

    def persist(self, events=()):
        self.repository.save(self.internal_user_id, self._snapshot(), events)

    def _public_reply(self, output):
        version = self.templates["notice_versions"]["privacy"]
        if version not in self.seen_notice_versions:
            output.output_text += "\n\n" + self.templates["privacy_notice"]
        if self._defer_delivery:
            if self._delivery_pending is None:
                self._delivery_pending = {}
            self._delivery_pending["notice_version"] = version
        else:
            self.confirm_delivery(version=version)
        self.messages.append(Message(conversation_id=self.conversation_id, role="agent", text=output.output_text,
                                     incident_ids=[output.incident_id] if output.incident_id else []))
        output.sources = self.last_sources
        self.last_sources = []
        self.persist()
        self.dispatcher.dispatch(self.internal_user_id)
        return output

    def confirm_delivery(self, *, version=None):
        pending = self._delivery_pending
        if pending and pending.get("plan"):
            st = self.incidents[pending["incident_id"]]
            plan = TurnPlan(**pending["plan"])
            plan.ask_fields = [IncidentFieldName(n) for n in plan.ask_fields]
            plan.hotline_flags = [UrgentFlag(f) for f in plan.hotline_flags]
            result = after_delivery(st, plan, message_id=pending["reply_id"], now=datetime.fromisoformat(pending["now"]),
                                    pending_question=self.pending_question, pending_confirmation=self.pending_confirmation)
            self.incidents[st.incident_id] = result.state
            self.pending_question, self.pending_confirmation = result.pending_question, result.pending_confirmation
            if pending.get("ask_location"):
                self._open_location_request(result.state, kind="ask_location", purpose=pending["purpose"],
                                            reply_id=pending["reply_id"], now=datetime.fromisoformat(pending["now"]))
        if pending and pending.get("unsolicited_location"):
            st = self.incidents[pending["incident_id"]]
            self._open_location_request(st, kind="confirm_unsolicited", purpose=self.templates["location_consent_confirm"],
                                        reply_id=pending["reply_id"], now=datetime.fromisoformat(pending["now"]))
        if pending and pending.get("contact_request_id"):
            for request in self.contact_requests:
                if request.request_id == pending["contact_request_id"]:
                    request.request_message_id = pending["reply_id"]
        version = version or (pending or {}).get("notice_version")
        if version and version not in self.seen_notice_versions:
            self.seen_notice_versions.append(version)
        self._delivery_pending = None
        self.persist()

    def _report_event(self, st, plan, merge, now):
        events = []
        st = evaluate_priority(st, correction=bool(merge and any(c.change_type == FieldChangeType.CORRECTION and c.field.value in {"needs", "situation", "water_level", "vulnerable_people"} for c in merge.changes)), now=now)
        previous = self.last_reports.get(st.incident_id)
        report = build_report(st, self.reporter_ref, self.consents, self.contacts)
        if plan.submit_trigger:
            events.append(IncidentSubmittedEvent(incident_id=st.incident_id, report=report, trigger=plan.submit_trigger,
                                                  case_version=st.incident_version, occurred_at=now))
        elif previous and st.status == IncidentStatus.SUBMITTED:
            correction_fields = [c.field.value for c in merge.changes if c.change_type == FieldChangeType.CORRECTION] if merge else []
            changes = changes_between(previous, report, now=now, correction_fields=correction_fields)
            if changes:
                st = st.model_copy(update={"incident_version": max(st.incident_version, previous.case_version) + 1})
                report = build_report(st, self.reporter_ref, self.consents, self.contacts)
                events.append(IncidentUpdatedEvent(incident_id=st.incident_id, incident_version=st.incident_version,
                                                   case_version=st.incident_version, report=report, changes=changes, occurred_at=now))
                if NextAction.SEND_UPDATE not in plan.actions:
                    plan.actions.insert(0, NextAction.SEND_UPDATE)
            elif NextAction.SEND_UPDATE in plan.actions:
                plan.actions.remove(NextAction.SEND_UPDATE)
                if not plan.actions:
                    plan.actions.append(NextAction.NONE)
        if events:
            self.last_reports[st.incident_id] = report
        self.incidents[st.incident_id] = st
        # Commit before Gemini wording or any network delivery, including urgent intake.
        self.persist(events)
        if plan.submit_trigger == "urgent_flags":
            self.dispatcher.dispatch(self.internal_user_id)
        return st

    def delivery_status(self, incident_id):
        records = [r for r in self.repository.outbox(self.internal_user_id) if r.incident_id == incident_id]
        return records[-1].delivery_status if records else None

    def _wording(self, plan, st, message, changes=()):
        evidence = self.knowledge.evidence_for(plan.answer_question) if plan.answer_question else []
        # คำขอพิกัดเป็นข้อความแจ้งวัตถุประสงค์ที่ LocationShareRequest อ้าง notice_version จึงใช้ template เสมอ (ข้อ 6.10)
        # คำถามชื่อพื้นที่ที่ตรวจสอบไม่ได้ต้องบอกผู้ใช้ตรง ๆ จึงใช้ template เช่นกัน (ข้อ 6.4)
        area = st.location_area.value if st else None
        location_locked = IncidentFieldName.LOCATION_COORDINATES in plan.ask_fields or (
            IncidentFieldName.LOCATION_AREA in plan.ask_fields and area and area.match_status.value in {"not_found", "ambiguous"})
        if location_locked and not evidence:
            return None, None, None
        if not self.composer or not (plan.ask_fields or evidence) or (st and st.urgent_flags):
            return None, None, None
        options = {f.value:[q for q in self.templates["questions"][f.value] if q.endswith("ค่ะ")]
                   for f in plan.ask_fields}
        # Hints describe the planned question; the model can rephrase instead of selecting a template.
        safe_changes = [c.model_dump(mode="json") for c in changes
                        if getattr(c.field, "value", c.field) not in {"location_coordinates", "contact", "location_consent"}]
        payload = dict(message=message, recent_messages=[redact_identifiers(redact_phones(m.text).text) for m in self.messages[-3:]],
                       confirmed_state=st.model_dump(mode="json", exclude={"location_coordinates", "location_consent_id", "contact_ref"}) if st else {},
                       question_hints=options, acknowledgement_hints=list(self.templates["acknowledgements"].values()), changes=safe_changes,
                       max_chars=1200,
                       actions=[a.value for a in plan.actions], evidence=[asdict(e) for e in evidence])
        try:
            result = self.composer(payload)
            if result is None or not validate_wording(result, payload):
                return None, None, None
        except Exception:
            return None, None, None
        self.last_sources = result.source_ids
        answer = result.knowledge_answer
        if answer:
            rows = [e for e in evidence if e.chunk_id in result.source_ids]
            answer += "\nแหล่งข้อมูล: " + "; ".join(f"{row.title} ({row.url})" for row in rows)
        questions = None if location_locked else "\n".join(result.questions.values()) or None
        return result.acknowledgement or None, questions, answer or None

    def _pending_contact(self):
        return next((r for r in reversed(self.contact_requests) if r.status == "pending" and self.state and r.incident_id == self.state.incident_id), None)

    def _contact_postback(self, data, now, msg_id):
        request = self._pending_contact()
        if not request or not request.request_message_id:
            return None
        if data not in {f"contact_consent=yes:{request.request_id}", f"contact_consent=no:{request.request_id}"}:
            return None
        st = self.state
        allowed = data.startswith("contact_consent=yes:")
        if allowed:
            record = ContactRecord(internal_user_id=self.internal_user_id, incident_id=st.incident_id,
                                    phone=request.phone, purpose=request.purpose, consent_at=now)
            self.contacts.append(record)
            st = st.model_copy(update={"contact_ref":record.contact_ref,
                                       "contact_ask":TrackedField[bool](status="known", value=True, updated_at=now,
                                                                        ask_count=st.contact_ask.ask_count, last_asked_at=st.contact_ask.last_asked_at)})
        else:
            st = st.model_copy(update={"contact_ref":None, "contact_ask":TrackedField[bool](status="declined", updated_at=now,
                                                                                        ask_count=st.contact_ask.ask_count, last_asked_at=st.contact_ask.last_asked_at)})
        request.status = "granted" if allowed else "declined"
        request.phone = ""  # never retain a second copy after resolution
        request.response_message_id, request.resolved_at = msg_id, now
        return st

    def delete_personal_data(self, now=None):
        now = now or now_bkk()
        self.deletion_requested_at = now
        self.consents = [c.model_copy(update={"revoked_at":now, "revoke_reason":"data_deletion"}) for c in self.consents]
        events = [DataDeletionRequestedEvent(reporter_ref=self.reporter_ref, occurred_at=now)]
        # Erase current data and all copies in histories, messages and pending requests.
        self.contacts, self.contact_requests, self.location_requests, self.messages, self.turns = [], [], [], [], []
        self.pending_location, self.pending_question, self.pending_confirmation, self.match_candidate = None, None, None, None
        self.area_suggestion = None
        self.conversation_summary = None
        for key, st in self.incidents.items():
            fields = {}
            for name in ("location_coordinates", "location_landmark", "situation", "water_level", "location_area", "vulnerable_people"):
                field = st.tracked_field(IncidentFieldName(name))
                if name in {"location_coordinates", "location_landmark"}:
                    fields[name] = field.model_copy(update={"status":FieldStatus.DECLINED, "value":None, "raw":None, "history":[]})
                else:
                    value = field.value
                    if value and name == "situation": value = value.model_copy(update={"text":"[ลบข้อความส่วนบุคคลแล้ว]"})
                    if value and name == "location_area": value = value.model_copy(update={"raw":"[ลบข้อความแล้ว]"})
                    if value and name == "water_level": value = value.model_copy(update={"raw":"[ลบข้อความแล้ว]"})
                    if value and name == "vulnerable_people": value = value.model_copy(update={"groups":[g.model_copy(update={"note":None}) for g in value.groups]})
                    fields[name] = field.model_copy(update={"value":value, "raw":None, "history":[]})
            for name in ("needs", "affected_people", "contact_ask"):
                field = getattr(st, name)
                fields[name] = field.model_copy(update={"raw":None, "history":[]})
            fields.update(contact_ref=None, location_consent_id=None, deletion_requested_at=now, coordinates_forwarding_declined=True,
                          contact_ask=TrackedField[bool](status="declined"))
            self.incidents[key] = st.model_copy(update=fields)
        self.last_reports = {k:build_report(st, self.reporter_ref) for k,st in self.incidents.items() if st.submitted_at}
        self._delivery_pending = None
        self.repository.scrub_events(self.internal_user_id, all_personal=True)
        self.persist(events)
        self.dispatcher.dispatch(self.internal_user_id)

    def close_incident(self, reason, now=None):
        now = now or now_bkk()
        st = self.state
        if not st or st.status != IncidentStatus.SUBMITTED:
            return None
        st = IncidentState.model_validate({**st.model_dump(), "status":"closed", "close_reason":reason,
                                           "closed_at":now, "updated_at":now, "incident_version":st.incident_version+1})
        self.incidents[st.incident_id] = st
        self.pending_question = self.pending_confirmation = None
        event = IncidentClosedEvent(incident_id=st.incident_id, case_version=st.incident_version, close_reason=reason,
                                     last_activity_at=st.last_activity_at, occurred_at=now)
        self.persist([event])
        return st

    def maintain(self, *, now=None, contact_pending_ttl_hours=None):
        """Retention defaults in 3.1/3.5. Pending phone TTL is opt-in pending policy approval."""
        now = now or now_bkk()
        for key, st in list(self.incidents.items()):
            age = now - st.last_activity_at
            if st.status == IncidentStatus.DRAFT and not st.urgent_flags and age >= timedelta(days=7):
                self.incidents[key] = st = st.model_copy(update={"status":IncidentStatus.ABANDONED, "updated_at":now})
            elif st.status == IncidentStatus.SUBMITTED and age >= timedelta(days=14):
                active = self.active_id
                self.active_id = key
                self.close_incident(CloseReason.TIMEOUT, now)
                st = self.incidents[key]
                self.active_id = active
            if st.status == IncidentStatus.ABANDONED and now - st.updated_at >= timedelta(days=30):
                del self.incidents[key]
                if self.active_id == key:
                    self.active_id = None
                continue
            if st.closed_at and now - st.closed_at >= timedelta(days=30):
                coords = st.location_coordinates.model_copy(update={"value":None,"status":FieldStatus.DECLINED,"raw":None,"history":[]})
                self.incidents[key] = st.model_copy(update={"location_coordinates":coords,"location_consent_id":None,"contact_ref":None})
                self.contacts = [c for c in self.contacts if c.incident_id != key]
                self.contact_requests = [c for c in self.contact_requests if c.incident_id != key]
                self.consents = [c for c in self.consents if c.incident_id != key]
                self.location_requests = [r for r in self.location_requests if r.incident_id != key]
                self.last_reports.pop(key, None)
                self.repository.scrub_events(self.internal_user_id, {key})
                # Turn diagnostics are expendable and can hold nested state/history copies.
                self.turns = []
        self.messages = [m for m in self.messages if now - m.created_at < timedelta(days=90)]
        if self.conversation_summary and (not self.incidents or all(now-st.last_activity_at >= timedelta(days=90) for st in self.incidents.values())):
            self.conversation_summary = None
        if self.pending_location and now - datetime.fromisoformat(self.pending_location["received_at"]) >= timedelta(hours=24):
            self.pending_location = None
        for request in self.contact_requests:
            if contact_pending_ttl_hours is not None and request.status == "pending" and now-request.requested_at >= timedelta(hours=contact_pending_ttl_hours):
                request.phone, request.status, request.resolved_at = "", "expired", now
        if self.state is None or not self.state.is_open:
            self.pending_question = self.pending_confirmation = None
        self.persist()

    def record_operation(self, record):
        st = self.state
        if not st:
            raise ValueError("unknown incident")
        if st.operational_status and record.recorded_at <= st.operational_status[-1].recorded_at:
            raise ValueError("stale operational status")
        self.incidents[st.incident_id] = st.model_copy(update={"operational_status":[*st.operational_status, record]})
        self.persist()

    def override_priority(self, priority):
        st = self.state
        if not st or priority.source == "agent1_rules":
            raise ValueError("manual override required")
        if st.priority and priority.evaluated_at <= st.priority.evaluated_at:
            raise ValueError("stale priority")
        st = st.model_copy(update={"priority":priority, "priority_history":[*st.priority_history, *([st.priority] if st.priority else [])]})
        return self._report_event(st, TurnPlan(actions=[]), None, priority.evaluated_at)

    def context_for_extraction(self, state, pending, now):
        from .memory import build_context
        import os
        # Rolling summary deliberately contains only incident references and enums, no free text.
        if len(self.messages) > int(os.getenv("SUMMARY_MESSAGE_THRESHOLD", "25")):
            covered = self.messages[:-10]
            if covered:
                ids = sorted({i for m in covered for i in m.incident_ids})
                previous = self.conversation_summary.text if self.conversation_summary else ""
                known = set(previous.split()) | set(ids)
                text = " ".join(sorted(x for x in known if x.startswith("inc_") and all(c in "inc_0123456789abcdef" for c in x)))
                self.conversation_summary = ConversationSummary(conversation_id=self.conversation_id, text=text,
                                                                 covers_until_message_id=covered[-1].message_id)
        safe = state.model_copy(deep=True) if state else None
        if safe:
            safe.location_coordinates = TrackedField[Coordinates]()
        return build_context(active=safe, pending=self.pending_question, open_incidents=self.incidents.values(),
                             messages=self.messages, summary=self.conversation_summary, now=now,
                             max_chars=int(os.getenv("CONTEXT_MAX_CHARS", "6000")))
