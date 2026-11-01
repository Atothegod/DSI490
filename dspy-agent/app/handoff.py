"""Versioned reports, summaries and technical ACK delivery. Draft HTTP contract (8.1)."""
from collections import defaultdict
from .schemas import (
    IncidentReport, ReportLocation, ReportedValue, IncidentSubmittedEvent, IncidentUpdatedEvent,
    FieldChange, FieldChangeType, HandoffAck, DeliveryStatus, AreaSummary, AreaSummaryUpdatedEvent,
    FieldStatus, Need, now_bkk,
)


def build_report(state, reporter_ref, consents=(), contacts=()):
    coords, basis = None, None
    allowed = not state.deletion_requested_at and not state.coordinates_forwarding_declined
    if allowed and state.location_coordinates.value:
        active = any(c.consent_id == state.location_consent_id and c.is_active and c.incident_id == state.incident_id for c in consents)
        if active:
            coords, basis = state.location_coordinates.value, "consent"
        elif set(state.needs.value or []) & {Need.RESCUE, Need.EVACUATION}:
            coords, basis = state.location_coordinates.value, "needs"
    contact = next((c for c in contacts if c.contact_ref == state.contact_ref and not c.deleted_at), None)
    values = {n: ReportedValue(status=getattr(state, n).status, value=getattr(state, n).value,
                              reported_at=getattr(state, n).updated_at)
              for n in ("situation", "needs", "affected_people", "vulnerable_people", "water_level")}
    return IncidentReport(incident_id=state.incident_id, incident_version=state.incident_version,
                          case_version=state.incident_version, reporter_ref=reporter_ref,
                          status=state.status, matching_status=state.matching_status,
                          possible_related_incident_ids=state.possible_related_incident_ids,
                          related_incident_id=state.related_incident_id, service_scope=state.service_scope,
                          location=ReportLocation(area=state.location_area.value, landmark=state.location_landmark.value,
                                                  coordinates=coords, coordinates_basis=basis),
                          location_missing=not bool(state.location_area.value or coords or state.location_landmark.value),
                          priority=state.priority, urgent_flags=sorted(state.urgent_flag_types, key=lambda f: f.value),
                          contact_available=contact is not None, contact_ref=contact.contact_ref if contact else None,
                          reported_at=state.created_at, submitted_at=state.submitted_at, **values)


def changes_between(previous, current, *, now=None, correction_fields=()):
    now = now or now_bkk()
    old, new = previous.model_dump(mode="json"), current.model_dump(mode="json")
    mapping = {"location_area": (old["location"]["area"], new["location"]["area"]),
               "location_coordinates": (old["location"]["coordinates"], new["location"]["coordinates"]),
               "location_landmark": (old["location"]["landmark"], new["location"]["landmark"]),
               "location_consent": (old["location"]["coordinates_basis"], new["location"]["coordinates_basis"])}
    mapping.update({k: (old[k], new[k]) for k in ("situation", "needs", "affected_people", "vulnerable_people", "water_level",
                                                "urgent_flags", "service_scope", "priority", "contact_available")})
    result = []
    for field, (a, b) in mapping.items():
        if field in {"situation", "needs", "affected_people", "vulnerable_people", "water_level"}:
            at, bt = a.get("reported_at"), b.get("reported_at")
            av, bv = a["value"], b["value"]
            if field == "situation" and field not in correction_fields:
                equal = (a["status"], (av or {}).get("tags")) == (b["status"], (bv or {}).get("tags"))
            elif field == "water_level" and field not in correction_fields:
                equal = (a["status"], (av or {}).get("descriptor")) == (b["status"], (bv or {}).get("descriptor"))
            else:
                equal = (a["status"], av) == (b["status"], bv)
            if equal:
                continue
            # Include statuses in changes so unknown is never mistaken for absent people.
            pa, pb = {k: v for k,v in a.items() if k != "reported_at"}, {k: v for k,v in b.items() if k != "reported_at"}
            initial = av is None and a["status"] == "not_asked"
        else:
            if a == b:
                continue
            pa, pb, at, bt, initial = a, b, previous.submitted_at, now, a is None
        result.append(FieldChange(field=field, change_type=FieldChangeType.INITIAL if initial else
                                  FieldChangeType.CORRECTION if field in correction_fields else FieldChangeType.UPDATE,
                                  previous_value=None if initial else pa,
                                  previous_reported_at=None if initial else at or previous.submitted_at or now,
                                  new_value=pb, reported_at=bt or now))
    return result


def area_summaries(reports, *, now=None):
    groups = defaultdict(list)
    for report in reports:
        if report.status.value != "submitted":
            continue
        a = report.location.area
        if a and a.match_status.value == "matched" and a.tambon_code:
            keys = [("tambon", a.tambon_code), ("amphoe", a.amphoe_code), ("province", a.province_code)]
        else:
            keys = [("unassigned", None)]
        for key in keys:
            groups[key].append(report)
    result = []
    for (level, code), cases in groups.items():
        counts = defaultdict(int)
        for r in cases:
            counts[r.priority.level.value if r.priority else "needs_review"] += 1
        numbers = [r.affected_people.value.count for r in cases if r.affected_people.value and r.affected_people.value.count is not None]
        result.append(AreaSummary(summarized_at=now or now_bkk(), area_level=level, area_code=code,
                                  case_versions={r.incident_id:r.case_version for r in cases}, priority_counts=dict(counts),
                                  urgent_incident_ids=sorted(r.incident_id for r in cases if r.urgent_flags or
                                                             (r.priority and r.priority.level.value == "red")),
                                  needs=sorted({n for r in cases for n in r.needs.value or []}, key=lambda n:n.value),
                                  reported_people=sum(numbers) if numbers else None))
    return result


class HandoffDispatcher:
    def __init__(self, repository, sender=None, *, max_attempts=5):
        self.repository, self.sender, self.max_attempts = repository, sender, max_attempts

    def dispatch(self, user_id=None):
        if self.sender is None:
            return  # honest pending state until a transport is configured
        for record in self.repository.outbox(user_id):
            if record.delivery_status == DeliveryStatus.ACKED or record.attempts >= self.max_attempts:
                continue
            if record.last_error == "cancelled_by_erasure":
                continue
            record.attempts += 1
            record.last_attempt_at = now_bkk()
            record.delivery_status = DeliveryStatus.SENT
            self.repository.update_outbox(record)
            try:
                ack = HandoffAck.model_validate(self.sender(record.payload))
                if not ack.received or ack.event_id != record.event_id:
                    raise ValueError("ACK rejected or event_id mismatch")
                record.delivery_status, record.acked_at, record.last_error = DeliveryStatus.ACKED, ack.received_at, None
            except Exception as exc:
                record.delivery_status = DeliveryStatus.FAILED
                record.last_error = type(exc).__name__  # no response bodies or personal data in logs
            self.repository.update_outbox(record)


def http_sender(url, token, *, timeout=5):
    import httpx
    def send(payload):
        response = httpx.post(url, json=payload, headers={"Authorization": "Bearer " + token}, timeout=timeout)
        response.raise_for_status()
        return response.json()
    return send
