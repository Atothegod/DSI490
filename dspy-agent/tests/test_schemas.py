"""ทดสอบกฎใน schemas.py ตาม Design Decisions v1.1 (ไม่ใช้ LLM)
กฎใหม่ของ v1.1 อยู่ใน test_schemas_v11.py"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from app.schemas import (
    BANGKOK_TZ,
    AffectedPeople,
    AreaMatchStatus,
    AreaRef,
    FieldChangeType,
    CloseReason,
    ContactRecord,
    DeliveryStatus,
    EventType,
    ExtractionResult,
    FieldRevision,
    FieldStatus,
    FieldUpdate,
    HandoffEvent,
    IncidentFieldName,
    IncidentReport,
    IncidentState,
    IncidentStatus,
    Intent,
    KBCategory,
    KBTopic,
    KnowledgeChunk,
    MatchingStatus,
    Need,
    NextAction,
    ReportedValue,
    ReportLocation,
    ReviewStatus,
    Situation,
    SituationTag,
    TrackedField,
    UrgentFlag,
    VulnerablePeople,
    VulnerableType,
    WaterDescriptor,
    WaterLevel,
)

NOW = datetime(2026, 10, 29, 14, 0, tzinfo=BANGKOK_TZ)


def make_state(**kw) -> IncidentState:
    return IncidentState(internal_user_id="usr_1", conversation_id="conv_1", **kw)


# --- TrackedField ------------------------------------------------------------


def test_tracked_field_defaults_to_not_asked():
    f = TrackedField[str]()
    assert f.status == FieldStatus.NOT_ASKED and f.can_ask


def test_known_requires_value():
    with pytest.raises(ValidationError):
        TrackedField[str](status=FieldStatus.KNOWN)


@pytest.mark.parametrize("status", [FieldStatus.UNKNOWN, FieldStatus.DECLINED, FieldStatus.NOT_ASKED])
def test_non_known_must_not_have_value(status):
    with pytest.raises(ValidationError):
        TrackedField[str](status=status, value="x")


@pytest.mark.parametrize("status", [FieldStatus.UNKNOWN, FieldStatus.DECLINED, FieldStatus.KNOWN])
def test_answered_fields_cannot_be_asked_again(status):
    f = TrackedField[str](status=status, value="x" if status == FieldStatus.KNOWN else None)
    assert not f.can_ask


def test_correction_history_keeps_previous_value():
    f = TrackedField[WaterLevel](
        status=FieldStatus.KNOWN,
        value=WaterLevel(raw="80 เซนติเมตร", cm=80),
        updated_at=NOW,
        history=[
            FieldRevision[WaterLevel](
                status=FieldStatus.KNOWN, value=WaterLevel(raw="50 ซม", cm=50), changed_at=NOW,
                change_type=FieldChangeType.CORRECTION,
            )
        ],
    )
    assert f.history[-1].value.cm == 50 and f.history[-1].change_type == FieldChangeType.CORRECTION


def test_naive_datetime_rejected():
    with pytest.raises(ValidationError):
        TrackedField[str](status=FieldStatus.KNOWN, value="x", updated_at=datetime(2026, 10, 29))


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        Situation(text="น้ำเข้าบ้าน", severity=3)


# --- Value models ------------------------------------------------------------


def test_water_level_needs_descriptor_or_cm():
    with pytest.raises(ValidationError):
        WaterLevel(raw="เยอะมาก")
    assert WaterLevel(raw="ถึงเอว", descriptor=WaterDescriptor.WAIST).cm is None


def test_affected_people_consistency():
    with pytest.raises(ValidationError):
        AffectedPeople(present=False, count=3)
    assert AffectedPeople(present=True, count=5, is_approximate=True).is_approximate


def test_vulnerable_people_consistency():
    with pytest.raises(ValidationError):
        VulnerablePeople(present=False, groups=[{"type": "elderly", "count": 2}])
    vp = VulnerablePeople(present=True, groups=[{"type": "bedridden", "count": 1}])
    assert vp.groups[0].type == VulnerableType.BEDRIDDEN


# --- IncidentState lifecycle -------------------------------------------------


def test_new_incident_is_open_draft():
    s = make_state()
    assert s.status == IncidentStatus.DRAFT and s.is_open and s.matching_status == MatchingStatus.NEW


def test_no_severity_field_anywhere():
    for model in (IncidentState, IncidentReport):
        assert "severity" not in model.model_fields


def test_closed_requires_reason_and_time():
    with pytest.raises(ValidationError):
        make_state(status=IncidentStatus.CLOSED, submitted_at=NOW)
    s = make_state(status=IncidentStatus.CLOSED, submitted_at=NOW, close_reason=CloseReason.TIMEOUT, closed_at=NOW)
    assert not s.is_open


def test_close_reason_only_when_closed():
    with pytest.raises(ValidationError):
        make_state(close_reason=CloseReason.USER_RESOLVED)


def test_submitted_requires_submitted_at():
    with pytest.raises(ValidationError):
        make_state(status=IncidentStatus.SUBMITTED)


def test_urgent_draft_cannot_be_abandoned():
    flag = {"flag": "people_trapped", "sources": ["keyword"], "detected_at": NOW}
    assert make_state(urgent_flags=[flag]).urgent_flag_types == {UrgentFlag.PEOPLE_TRAPPED}
    with pytest.raises(ValidationError, match="abandoned"):
        make_state(status=IncidentStatus.ABANDONED, urgent_flags=[flag])


def test_unconfirmed_requires_candidates():
    with pytest.raises(ValidationError):
        make_state(matching_status=MatchingStatus.UNCONFIRMED)
    s = make_state(matching_status=MatchingStatus.UNCONFIRMED, possible_related_incident_ids=["inc_a"])
    assert s.possible_related_incident_ids == ["inc_a"]


def test_known_needs_cannot_be_empty():
    with pytest.raises(ValidationError):
        make_state(needs=TrackedField[list[Need]](status=FieldStatus.KNOWN, value=[]))


AREA_TABYA = AreaRef(raw="ทับยา", match_status=AreaMatchStatus.MATCHED, tambon_code="170405",
                     tambon="ทับยา", amphoe_code="1704", amphoe="อินทร์บุรี", province="สิงห์บุรี")
# รหัสข้างบนเป็นค่าสมมติสำหรับทดสอบเท่านั้น ไม่ใช่รหัสจริงจากกรมการปกครอง


def test_tracked_field_lookup_by_name():
    s = make_state(location_area=TrackedField[AreaRef](status=FieldStatus.KNOWN, value=AREA_TABYA))
    assert s.tracked_field(IncidentFieldName.LOCATION_AREA).value.tambon == "ทับยา"
    contact = s.tracked_field(IncidentFieldName.CONTACT)
    assert contact.status == FieldStatus.NOT_ASKED and contact.value is None  # สถานะการถามเท่านั้น ไม่มีเบอร์


def test_state_roundtrip_json():
    s = make_state(
        situation=TrackedField[Situation](
            status=FieldStatus.KNOWN, value=Situation(text="ถนนเข้าหมู่บ้านถูกตัดขาด", tags=[SituationTag.ROAD_BLOCKED])
        )
    )
    again = IncidentState.model_validate_json(s.model_dump_json())
    assert again == s


# --- Contact / Report --------------------------------------------------------


def test_contact_requires_phone_or_name():
    with pytest.raises(ValidationError):
        ContactRecord(internal_user_id="u", incident_id="i", purpose="p", consent_at=NOW)


def _report(**kw) -> IncidentReport:
    na = ReportedValue(status=FieldStatus.NOT_ASKED)
    base = dict(
        incident_id="inc_a",
        incident_version=1,
        reporter_ref="rep_x",
        status=IncidentStatus.SUBMITTED,
        matching_status=MatchingStatus.NEW,
        location=ReportLocation(area=AREA_TABYA),
        situation=na,
        needs=na,
        affected_people=na,
        vulnerable_people=na,
        water_level=na,
        reported_at=NOW,
    )
    base.update(kw)
    return IncidentReport(**base)


def test_report_has_no_personal_identifiers():
    fields = set(IncidentReport.model_fields)
    assert not fields & {"line_user_id", "internal_user_id", "phone", "name", "conversation_id"}


def test_report_contact_flag_must_match_ref():
    with pytest.raises(ValidationError):
        _report(contact_available=True)
    assert _report(contact_available=True, contact_ref="ctc_1").contact_ref == "ctc_1"


# --- Handoff events ----------------------------------------------------------

EVENTS = TypeAdapter(HandoffEvent)


def test_event_union_parses_by_type():
    ev = EVENTS.validate_python(
        {
            "event_type": "incident_linked",
            "incident_id": "inc_045",
            "linked_to": "inc_032",
            "relation": "same_incident",
            "occurred_at": NOW.isoformat(),
        }
    )
    assert ev.event_type == EventType.INCIDENT_LINKED and ev.event_id.startswith("evt_")


def test_link_to_self_rejected():
    with pytest.raises(ValidationError):
        EVENTS.validate_python(
            {"event_type": "incident_linked", "incident_id": "a", "linked_to": "a", "relation": "related"}
        )


def test_submitted_event_ids_must_match_report():
    with pytest.raises(ValidationError):
        EVENTS.validate_python(
            {"event_type": "incident_submitted", "incident_id": "other", "trigger": "urgent_flags",
             "report": _report().model_dump(mode="json")}
        )


WATER_CHANGE = {"field": "water_level", "change_type": "update",
                "previous_value": {"raw": "ถึงเอว", "descriptor": "waist"},
                "previous_reported_at": NOW.isoformat(),
                "new_value": {"raw": "80 เซนติเมตร", "cm": 80}, "reported_at": NOW.isoformat()}


def test_updated_event_version_rules():
    rep = _report(incident_version=2).model_dump(mode="json")
    ok = EVENTS.validate_python(
        {"event_type": "incident_updated", "incident_id": "inc_a", "incident_version": 2,
         "changes": [WATER_CHANGE], "report": rep}
    )
    assert ok.changed_fields == [IncidentFieldName.WATER_LEVEL]
    assert ok.changes[0].change_type == FieldChangeType.UPDATE
    with pytest.raises(ValidationError):  # update ต้องเป็น version >= 2
        EVENTS.validate_python(
            {"event_type": "incident_updated", "incident_id": "inc_a", "incident_version": 1,
             "changes": [WATER_CHANGE], "report": _report().model_dump(mode="json")}
        )


def test_event_roundtrip_json():
    ev = EVENTS.validate_python(
        {"event_type": "incident_closed", "incident_id": "inc_a", "close_reason": "timeout",
         "last_activity_at": NOW.isoformat()}
    )
    assert EVENTS.validate_json(EVENTS.dump_json(ev)) == ev


# --- RAG ---------------------------------------------------------------------


def _chunk(**kw) -> KnowledgeChunk:
    base = dict(source_org="ปภ.", title="t", url="https://example.org", category=KBCategory.A,
                topic=KBTopic.SAFETY, original_text="ตัดไฟก่อนน้ำเข้าบ้าน")
    base.update(kw)
    return KnowledgeChunk(**base)


def test_category_b_requires_valid_until():
    with pytest.raises(ValidationError):
        _chunk(category=KBCategory.B)


def test_approved_requires_review_metadata():
    with pytest.raises(ValidationError):
        _chunk(review_status=ReviewStatus.APPROVED)


def test_only_approved_and_unexpired_chunks_usable():
    assert not _chunk().is_usable(NOW)
    approved = dict(review_status=ReviewStatus.APPROVED, reviewed_text="ตัดไฟ", reviewed_by="guy",
                    reviewed_at=NOW, verified_at=NOW)
    assert _chunk(**approved).is_usable(NOW)
    expired = _chunk(**approved, category=KBCategory.B, valid_until=NOW - timedelta(days=1))
    assert not expired.is_usable(NOW)


# --- Agent I/O ---------------------------------------------------------------


def test_field_update_rules():
    with pytest.raises(ValidationError):
        FieldUpdate(field=IncidentFieldName.WATER_LEVEL, status=FieldStatus.KNOWN)
    with pytest.raises(ValidationError):  # LLM ห้ามคืน not_asked
        FieldUpdate(field=IncidentFieldName.WATER_LEVEL, status=FieldStatus.NOT_ASKED)
    u = FieldUpdate(field=IncidentFieldName.AFFECTED_PEOPLE, status=FieldStatus.UNKNOWN)
    assert u.value is None


def test_extraction_result_requires_intent():
    with pytest.raises(ValidationError):
        ExtractionResult(intents=[], target="none")


# --- Golden scenarios file ---------------------------------------------------

SCENARIOS = json.loads((Path(__file__).parent / "golden_scenarios.json").read_text(encoding="utf-8"))


def _walk(obj, key):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                yield v
            yield from _walk(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v, key)


def _values(enum):
    return {e.value for e in enum}


def test_scenario_ids_unique_and_kinds_valid():
    ids = [s["id"] for s in SCENARIOS["scenarios"]]
    assert len(ids) == len(set(ids))
    for s in SCENARIOS["scenarios"]:
        assert s["kind"] in ("conversation", "rule") and s["covers"]
        if s["kind"] == "conversation":
            assert s["turns"]


@pytest.mark.parametrize(
    "key, allowed",
    [
        ("intents_include", _values(Intent)),
        ("next_actions_include", _values(NextAction)),
        ("urgent_flags_include", _values(UrgentFlag)),
        ("must_not_ask", _values(IncidentFieldName)),
        ("pending_fields_include", _values(IncidentFieldName)),
        ("pending_fields_include_any", _values(IncidentFieldName)),
        ("agent_asked", _values(IncidentFieldName)),
        ("pending_fields", _values(IncidentFieldName)),
        ("tags_include", _values(SituationTag)),
        ("if_known_value_include", _values(Need)),
        ("urgent_flags", _values(UrgentFlag)),
    ],
)
def test_scenario_list_values_match_schema(key, allowed):
    for values in _walk(SCENARIOS, key):
        assert set(values) <= allowed, f"{key}: {set(values) - allowed}"


def test_scenario_changes_use_v11_change_types():
    """v1.1 แทน is_correction / changed_fields ใน expect ด้วย changes[] ที่มี change_type
    (is_correction ยังอยู่ใน input ได้ เพราะเป็นค่าที่ Gemini คืนมา ซึ่ง Backend ไม่เชื่อตรง ๆ)"""
    expects = list(_walk(SCENARIOS, "expect"))
    for old_key in ("is_correction", "changed_fields_include"):
        assert not list(_walk(expects, old_key)), f"ยังมี key เดิม {old_key} ใน expect"
    fields = _values(IncidentFieldName) | {"urgent_flags", "service_scope"}
    for changes in _walk(SCENARIOS, "changes_include"):
        for c in changes:
            assert c["field"] in fields
            assert c["change_type"] in _values(FieldChangeType) - {"confirmation"}
    for change_type in _walk(SCENARIOS, "change_type"):
        assert change_type in _values(FieldChangeType)


@pytest.mark.parametrize(
    "key, allowed",
    [
        ("incident_status", _values(IncidentStatus)),
        ("matching_status", _values(MatchingStatus)),
        ("close_reason", _values(CloseReason)),
        ("delivery_status", _values(DeliveryStatus)),
        ("descriptor", _values(WaterDescriptor)),
    ],
)
def test_scenario_scalar_values_match_schema(key, allowed):
    for v in _walk(SCENARIOS, key):
        assert v in allowed, f"{key}: {v}"


def test_scenario_events_and_fields_valid():
    for events in _walk(SCENARIOS, "events"):
        for e in events:
            assert e["type"] in _values(EventType)
    for fields in _walk(SCENARIOS, "fields"):
        assert set(fields) <= _values(IncidentFieldName)
    for needs in _walk(SCENARIOS, "needs"):
        if isinstance(needs, list):
            assert set(needs) <= _values(Need)
