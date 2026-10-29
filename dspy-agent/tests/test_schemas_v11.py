"""ทดสอบกฎใหม่ใน schemas.py ตาม Design Decisions v1.1 หัวข้อ 6 (ไม่ใช้ LLM)"""

from datetime import datetime, timedelta

import pytest
from pydantic import TypeAdapter, ValidationError

from app.schemas import (
    BANGKOK_TZ,
    COORDINATE_NEEDS,
    SCHEMA_VERSION,
    AreaLevel,
    AreaMatchStatus,
    AreaMention,
    AreaRef,
    CloseReason,
    Coordinates,
    DataDeletionRequestedEvent,
    DeletionRequest,
    DeletionScope,
    FieldChange,
    FieldChangeType,
    FieldStatus,
    HandoffEvent,
    HotlineNotice,
    IncidentFieldName,
    IncidentReport,
    IncidentState,
    IncidentStatus,
    LocationShareConsent,
    LocationShareRequest,
    MatchingStatus,
    Need,
    ReportedValue,
    ReportLocation,
    ServiceScope,
    TrackedField,
    UrgentFlag,
    UrgentFlagRecord,
    UrgentFlagSource,
    UserRecord,
    max_asks,
)

NOW = datetime(2026, 10, 30, 10, 0, tzinfo=BANGKOK_TZ)
EVENTS = TypeAdapter(HandoffEvent)

# รหัสพื้นที่ในไฟล์นี้เป็นค่าสมมติสำหรับทดสอบเท่านั้น ไม่ใช่รหัสจริงจากกรมการปกครอง
TABYA = AreaRef(raw="ทับยา", stated_level=AreaLevel.TAMBON, match_status=AreaMatchStatus.MATCHED,
                tambon_code="T01", tambon="ทับยา", amphoe_code="A01", amphoe="อินทร์บุรี", province="สิงห์บุรี")
ANGTHONG = AreaRef(raw="จังหวัดอ่างทอง", stated_level=AreaLevel.PROVINCE,
                   match_status=AreaMatchStatus.OUT_OF_PROVINCE, province="อ่างทอง")


def state(**kw) -> IncidentState:
    return IncidentState(internal_user_id="usr_1", conversation_id="conv_1", **kw)


def flag(f="people_trapped", sources=("keyword",)) -> dict:
    return {"flag": f, "sources": list(sources), "detected_at": NOW}


def test_schema_version_is_1_1():
    assert SCHEMA_VERSION == "1.1"


# --- 6.3 UrgentFlagRecord ----------------------------------------------------


def test_flag_record_requires_unique_nonempty_sources():
    with pytest.raises(ValidationError):
        UrgentFlagRecord(flag=UrgentFlag.PEOPLE_TRAPPED, sources=[], detected_at=NOW)
    with pytest.raises(ValidationError):
        UrgentFlagRecord(flag=UrgentFlag.PEOPLE_TRAPPED, sources=["keyword", "keyword"], detected_at=NOW)
    both = UrgentFlagRecord(flag=UrgentFlag.PEOPLE_TRAPPED, sources=["keyword", "gemini"], detected_at=NOW)
    assert set(both.sources) == {UrgentFlagSource.KEYWORD, UrgentFlagSource.GEMINI}


def test_one_record_per_flag_type_in_incident():
    """flag เดิมจากอีกแหล่งต้องรวมเป็น record เดียว ไม่ใช่ flag ใหม่ (ข้อ 6.6)"""
    with pytest.raises(ValidationError, match="record เดียว"):
        state(urgent_flags=[flag(sources=["keyword"]), flag(sources=["gemini"])])
    s = state(urgent_flags=[flag(sources=["keyword", "gemini"]), flag("help_requested")])
    assert s.urgent_flag_types == {UrgentFlag.PEOPLE_TRAPPED, UrgentFlag.HELP_REQUESTED}


def test_report_sends_flag_types_only():
    """แหล่งที่มาเป็นข้อมูลภายใน รายงานส่งแค่ชนิด flag และห้ามซ้ำ"""
    assert IncidentReport.model_fields["urgent_flags"].annotation == list[UrgentFlag]


# --- 6.4 AreaRef --------------------------------------------------------------


def test_matched_area_requires_reference_code():
    with pytest.raises(ValidationError, match="รหัส"):
        AreaRef(raw="ทับยา", match_status=AreaMatchStatus.MATCHED, tambon="ทับยา")
    assert TABYA.area_code == "T01"


def test_ambiguous_area_needs_candidates_and_no_chosen_code():
    candidates = [{"amphoe_code": "A02", "amphoe": "พรหมบุรี"},
                  {"tambon_code": "T09", "tambon": "พรหมบุรี", "amphoe_code": "A02", "amphoe": "พรหมบุรี"}]
    ok = AreaRef(raw="พรหมบุรี", match_status=AreaMatchStatus.AMBIGUOUS, candidates=candidates)
    assert ok.area_code is None and len(ok.candidates) == 2
    with pytest.raises(ValidationError):  # Gemini/โค้ดห้ามเลือกเอง
        AreaRef(raw="พรหมบุรี", match_status=AreaMatchStatus.AMBIGUOUS, candidates=candidates, amphoe_code="A02")
    with pytest.raises(ValidationError):
        AreaRef(raw="พรหมบุรี", match_status=AreaMatchStatus.AMBIGUOUS, candidates=candidates[:1])


@pytest.mark.parametrize("status", [AreaMatchStatus.NOT_FOUND, AreaMatchStatus.UNVERIFIED])
def test_unmatched_area_must_not_carry_code(status):
    """ห้ามแก้เป็นชื่อใกล้เคียงอัตโนมัติ → ไม่มีรหัสเมื่อยังไม่พบ"""
    with pytest.raises(ValidationError):
        AreaRef(raw="ทัปยา", match_status=status, tambon_code="T01")
    assert AreaRef(raw="ทัปยา", match_status=status).raw == "ทัปยา"


def test_candidates_only_for_ambiguous():
    with pytest.raises(ValidationError):
        AreaRef(raw="ทับยา", match_status=AreaMatchStatus.NOT_FOUND,
                candidates=[{"amphoe_code": "A01", "amphoe": "อินทร์บุรี"}])


def test_area_mention_keeps_stated_level():
    assert AreaMention(name="พรหมบุรี").level == AreaLevel.UNSPECIFIED


# --- 6.5 Service scope และ out_of_service_area ---------------------------------


def test_new_incident_scope_is_unverified():
    assert state().service_scope == ServiceScope.UNVERIFIED


def test_out_of_province_area_requires_matching_scope():
    area = TrackedField[AreaRef](status=FieldStatus.KNOWN, value=ANGTHONG)
    with pytest.raises(ValidationError, match="service_scope"):
        state(location_area=area)
    assert state(location_area=area, service_scope=ServiceScope.OUT_OF_PROVINCE).is_open


def _closed_out(**kw):
    base = dict(status=IncidentStatus.CLOSED, close_reason=CloseReason.OUT_OF_SERVICE_AREA, closed_at=NOW,
                service_scope=ServiceScope.OUT_OF_PROVINCE)
    base.update(kw)
    return state(**base)


def test_draft_can_close_out_of_service_area_without_submission():
    s = _closed_out()
    assert s.submitted_at is None and not s.is_open


def test_out_of_service_area_rejected_after_submission():
    with pytest.raises(ValidationError, match="incident_updated"):
        _closed_out(submitted_at=NOW)


def test_out_of_service_area_rejected_with_urgent_flags():
    with pytest.raises(ValidationError, match="urgent_flags"):
        _closed_out(urgent_flags=[flag()])


def test_out_of_service_area_requires_out_of_province_scope():
    with pytest.raises(ValidationError):
        _closed_out(service_scope=ServiceScope.UNVERIFIED)


def test_other_close_reasons_still_require_submission():
    with pytest.raises(ValidationError, match="submitted_at"):
        state(status=IncidentStatus.CLOSED, close_reason=CloseReason.TIMEOUT, closed_at=NOW)


def test_out_of_service_area_never_sent_as_event():
    with pytest.raises(ValidationError, match="ไม่ส่ง event"):
        EVENTS.validate_python({"event_type": "incident_closed", "incident_id": "inc_a",
                                "close_reason": "out_of_service_area", "last_activity_at": NOW.isoformat()})


# --- 6.6 HotlineNotice ----------------------------------------------------------


def test_new_flag_notice_needs_flags_and_incident():
    with pytest.raises(ValidationError):
        HotlineNotice(reason="new_flags", flags=[], incident_id="inc_a", message_id="m1", notified_at=NOW)
    with pytest.raises(ValidationError):
        HotlineNotice(reason="new_flags", flags=["people_trapped"], message_id="m1", notified_at=NOW)


def test_user_asked_notice_may_have_no_incident():
    n = HotlineNotice(reason="user_asked", message_id="m1", notified_at=NOW)
    assert n.flags == [] and n.incident_id is None


def test_notice_must_match_detected_flags_and_incident():
    s0 = state()
    notice = {"reason": "new_flags", "flags": ["people_trapped"], "incident_id": s0.incident_id,
              "message_id": "m1", "notified_at": NOW}
    with pytest.raises(ValidationError, match="ไม่ได้ตรวจพบ"):
        state(incident_id=s0.incident_id, hotline_notices=[notice])
    with pytest.raises(ValidationError, match="ของ incident นี้"):
        state(urgent_flags=[flag()], hotline_notices=[{**notice, "incident_id": "other"}])
    s = state(incident_id=s0.incident_id, urgent_flags=[flag(), flag("injury_reported")], hotline_notices=[notice])
    # injury_reported ตรวจพบแล้วแต่ยังไม่ได้แจ้ง → เป็น flag ใหม่ที่ต้องแสดงสายด่วน
    assert s.urgent_flag_types - s.notified_flag_types == {UrgentFlag.INJURY_REPORTED}


# --- 6.7 FieldChange ----------------------------------------------------------------


def _change(**kw):
    base = dict(field="water_level", change_type="update", previous_value={"cm": 50},
                previous_reported_at=NOW - timedelta(hours=3), new_value={"cm": 80}, reported_at=NOW)
    base.update(kw)
    return FieldChange(**base)


def test_confirmation_is_never_an_event_change():
    with pytest.raises(ValidationError, match="confirmation"):
        _change(change_type="confirmation")


def test_initial_has_no_previous_value():
    assert _change(change_type="initial", previous_value=None, previous_reported_at=None).field \
        == IncidentFieldName.WATER_LEVEL
    with pytest.raises(ValidationError):
        _change(change_type="initial")


@pytest.mark.parametrize("change_type", ["update", "correction"])
def test_update_and_correction_carry_previous_value_with_time(change_type):
    with pytest.raises(ValidationError):
        _change(change_type=change_type, previous_reported_at=None)
    c = _change(change_type=change_type)
    assert c.previous_reported_at < c.reported_at


def test_scope_and_flag_changes_allowed():
    c = _change(field="service_scope", previous_value="unverified", new_value="out_of_province")
    assert c.field == "service_scope"


def _report_dict(version=2):
    na = ReportedValue(status=FieldStatus.NOT_ASKED)
    return IncidentReport(incident_id="inc_a", incident_version=version, reporter_ref="rep_x",
                          status=IncidentStatus.SUBMITTED, matching_status=MatchingStatus.NEW,
                          location=ReportLocation(area=TABYA), situation=na, needs=na, affected_people=na,
                          vulnerable_people=na, water_level=na, reported_at=NOW).model_dump(mode="json")


def test_updated_event_one_change_per_field():
    change = _change().model_dump(mode="json")
    with pytest.raises(ValidationError, match="change เดียว"):
        EVENTS.validate_python({"event_type": "incident_updated", "incident_id": "inc_a", "incident_version": 2,
                                "changes": [change, change], "report": _report_dict()})


def test_field_revision_uses_change_type():
    f = TrackedField[str](status=FieldStatus.KNOWN, value="ตำบลน้ำตาล", history=[
        {"status": "known", "value": "ตำบลทับยา", "reported_at": NOW - timedelta(hours=1),
         "changed_at": NOW, "change_type": "correction"}])
    assert f.history[0].change_type == FieldChangeType.CORRECTION
    with pytest.raises(ValidationError):  # is_correction ถูกแทนแล้ว
        TrackedField[str](status=FieldStatus.KNOWN, value="x", history=[
            {"status": "known", "value": "y", "changed_at": NOW, "is_correction": True}])


# --- 6.9 ask_count และ unanswered ----------------------------------------------------


def test_unanswered_requires_previous_ask():
    with pytest.raises(ValidationError, match="unanswered"):
        TrackedField[str](status=FieldStatus.UNANSWERED)
    f = TrackedField[str](status=FieldStatus.UNANSWERED, ask_count=1, last_asked_at=NOW)
    assert not f.can_ask and f.value is None


def test_unanswered_is_not_unknown():
    assert FieldStatus.UNANSWERED != FieldStatus.UNKNOWN


def test_ask_count_requires_time():
    with pytest.raises(ValidationError):
        TrackedField[str](ask_count=1)


def test_max_asks_per_field():
    assert max_asks(IncidentFieldName.WATER_LEVEL) == 1
    assert max_asks(IncidentFieldName.CONTACT) == 1
    assert max_asks(IncidentFieldName.LOCATION_COORDINATES) == 1
    assert max_asks(IncidentFieldName.LOCATION_AREA) == 2
    assert max_asks(IncidentFieldName.NEEDS) == 2


def test_contact_ask_state_lives_in_incident_without_phone():
    s = state(contact_ask=TrackedField[bool](status=FieldStatus.DECLINED, ask_count=1, last_asked_at=NOW))
    assert s.tracked_field(IncidentFieldName.CONTACT).status == FieldStatus.DECLINED
    assert "phone" not in IncidentState.model_fields


# --- 6.10 พิกัดและ consent -------------------------------------------------------------


def _request(**kw):
    base = dict(incident_id="inc_a", kind="ask_location", notice_version="location_share_v1",
                purpose="แนบพิกัดไปกับรายงานให้ผู้ดูแลรายงานพิจารณา", request_message_id="m_req", requested_at=NOW)
    base.update(kw)
    return LocationShareRequest(**base)


def test_pending_request_has_no_answer():
    assert _request().status == "pending"
    with pytest.raises(ValidationError):
        _request(response_message_id="m_resp")


def test_granted_request_requires_response():
    with pytest.raises(ValidationError):
        _request(status="granted", resolved_at=NOW)
    assert _request(status="granted", response_message_id="m_resp", resolved_at=NOW).status == "granted"


def test_closed_request_requires_resolved_time():
    with pytest.raises(ValidationError):
        _request(status="skipped")


def test_consent_only_from_granted_request():
    with pytest.raises(ValueError, match="อนุญาตแล้ว"):
        LocationShareConsent.from_request(_request())
    with pytest.raises(ValueError):
        LocationShareConsent.from_request(_request(status="declined", resolved_at=NOW))
    req = _request(status="granted", response_message_id="m_resp", resolved_at=NOW)
    c = LocationShareConsent.from_request(req)
    assert c.is_active and c.notice_version == "location_share_v1"
    assert (c.request_message_id, c.response_message_id) == ("m_req", "m_resp")


def test_consent_request_and_response_are_different_messages():
    with pytest.raises(ValidationError):
        LocationShareConsent(incident_id="i", request_id="r", purpose="p", notice_version="v",
                             request_message_id="m1", response_message_id="m1", granted_at=NOW)


def test_consent_revocation_needs_reason():
    req = _request(status="granted", response_message_id="m_resp", resolved_at=NOW)
    data = LocationShareConsent.from_request(req).model_dump()
    with pytest.raises(ValidationError):
        LocationShareConsent(**{**data, "revoked_at": NOW})
    revoked = LocationShareConsent(**{**data, "revoked_at": NOW, "revoke_reason": "data_deletion"})
    assert not revoked.is_active


def test_consent_does_not_imply_rescue():
    """consent แยกจาก needs: Need กับ COORDINATE_NEEDS ไม่ถูกแก้ และ consent ไม่มี field needs"""
    assert COORDINATE_NEEDS == frozenset({Need.RESCUE, Need.EVACUATION})
    assert "needs" not in LocationShareConsent.model_fields


def test_report_coordinates_must_state_basis():
    coords = Coordinates(latitude=14.95, longitude=100.36)
    with pytest.raises(ValidationError):
        ReportLocation(area=TABYA, coordinates=coords)
    with pytest.raises(ValidationError):
        ReportLocation(area=TABYA, coordinates_basis="consent")
    assert ReportLocation(area=TABYA, coordinates=coords, coordinates_basis="consent").coordinates_basis == "consent"


def test_unsolicited_coordinates_have_no_consent_by_default():
    s = state(location_coordinates=TrackedField[Coordinates](
        status=FieldStatus.KNOWN, value=Coordinates(latitude=14.95, longitude=100.36, message_id="m9")))
    assert s.location_consent_id is None


# --- การลบข้อมูล -----------------------------------------------------------------------------


def test_deletion_request_suspends_coordinate_forwarding():
    user = UserRecord(line_user_id="U1")
    assert not user.coordinates_forwarding_suspended
    user = UserRecord(line_user_id="U1", deletion_requested_at=NOW)
    assert user.coordinates_forwarding_suspended


def test_deletion_scope_defaults_to_all_personal_data():
    assert DeletionRequest(internal_user_id="u").scope == [DeletionScope.ALL_PERSONAL_DATA]
    assert DataDeletionRequestedEvent(reporter_ref="rep_x").scope == [DeletionScope.ALL_PERSONAL_DATA]


def test_deletion_scope_covers_more_than_coordinates():
    assert {"contact", "coordinates", "location_consent", "free_text", "messages"} <= {s.value for s in DeletionScope}


def test_all_personal_data_cannot_be_mixed():
    with pytest.raises(ValidationError):
        DataDeletionRequestedEvent(reporter_ref="r", scope=["all_personal_data", "coordinates"])
    with pytest.raises(ValidationError):
        DeletionRequest(internal_user_id="u", scope=["all_personal_data", "contact"])


def test_consents_revoked_before_local_deletion():
    with pytest.raises(ValidationError, match="consent"):
        DeletionRequest(internal_user_id="u", local_deleted_at=NOW)
    assert DeletionRequest(internal_user_id="u", consents_revoked_at=NOW, local_deleted_at=NOW).local_deleted_at


def test_agent2_ack_requires_event():
    with pytest.raises(ValidationError):
        DeletionRequest(internal_user_id="u", agent2_acked_at=NOW)
