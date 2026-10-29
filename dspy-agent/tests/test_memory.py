"""ทดสอบ memory.py ตาม Design Decisions v1.1 ข้อ 3.3, 6.1, 6.7, 6.9 (ไม่ใช้ LLM)"""

from datetime import datetime, timedelta

import pytest

from app.memory import (
    apply_updates,
    build_context,
    can_ask_now,
    close_unanswered,
    coerce_value,
    InvalidValue,
    missing_required,
    pending_is_valid,
    record_questions_asked,
    select_recent_messages,
    unverified_area_resolver,
)
from app.schemas import (
    BANGKOK_TZ,
    AreaLevel,
    AreaMatchStatus,
    AreaMention,
    AreaRef,
    CloseReason,
    ConversationSummary,
    FieldChangeType,
    FieldStatus,
    FieldUpdate,
    IncidentFieldName as F,
    IncidentState,
    IncidentStatus,
    Message,
    MessageRole,
    Need,
    PendingQuestion,
    SituationTag,
    WaterDescriptor,
)

NOW = datetime(2026, 10, 30, 12, 0, tzinfo=BANGKOK_TZ)
EARLIER = NOW - timedelta(hours=3)


def upd(field, value=None, status="known", raw=None, is_correction=False):
    return FieldUpdate(field=field, status=status, value=value, raw=raw, is_correction=is_correction)


def base_state(**kw) -> IncidentState:
    return IncidentState(internal_user_id="u", conversation_id="c", created_at=EARLIER, updated_at=EARLIER,
                         last_activity_at=EARLIER, **kw)


def merge(state, updates, text, now=NOW):
    return apply_updates(state, updates, message_text=text, message_id="m_new", now=now)


def with_water(cm=None, descriptor=None, raw="ถึงเอว", at=EARLIER):
    s = merge(base_state(), [upd(F.WATER_LEVEL, {"raw": raw, "cm": cm, "descriptor": descriptor}, raw=raw)], raw, now=at)
    return s.state


# --- coerce_value -----------------------------------------------------------------


def test_situation_text_uses_original_message_not_gemini():
    """ข้อ 6.1 — Gemini ตัดข้อความเหลือ "น้ำเข้าบ้าน" ได้ แต่เราเก็บข้อความเต็ม"""
    msg = "น้ำเข้าบ้านที่ตำบลทับยา อินทร์บุรี น้ำถึงเอว มีคนแก่สองคน"
    v = coerce_value(F.SITUATION, {"text": "น้ำเข้าบ้าน", "tags": ["water_in_house"]}, message_text=msg)
    assert v.text == msg and v.tags == [SituationTag.WATER_IN_HOUSE]


def test_area_accepts_mention_or_legacy_string():
    a = coerce_value(F.LOCATION_AREA, {"name": "พรหมบุรี"}, message_text="")
    assert a.raw == "พรหมบุรี" and a.stated_level == AreaLevel.UNSPECIFIED
    assert a.match_status == AreaMatchStatus.UNVERIFIED and a.area_code is None   # ไม่เดารหัส
    assert coerce_value(F.LOCATION_AREA, "ตำบลทับยา", message_text="").raw == "ตำบลทับยา"


def test_area_resolver_is_pluggable():
    def resolver(m: AreaMention) -> AreaRef:
        return AreaRef(raw=m.name, match_status=AreaMatchStatus.MATCHED, tambon_code="T01", tambon="ทับยา",
                       amphoe_code="A01", amphoe="อินทร์บุรี")
    v = coerce_value(F.LOCATION_AREA, {"name": "ทับยา", "level": "tambon"}, message_text="", area_resolver=resolver)
    assert v.area_code == "T01"


def test_info_only_dropped_when_real_need_present():
    assert coerce_value(F.NEEDS, ["info_only", "food_water"], message_text="") == [Need.FOOD_WATER]


@pytest.mark.parametrize("field, value", [
    (F.WATER_LEVEL, {"raw": "เยอะ"}),          # ไม่มีทั้ง descriptor และ cm
    (F.NEEDS, ["fly_me_home"]),
    (F.NEEDS, []),
    (F.LOCATION_LANDMARK, "   "),
    (F.AFFECTED_PEOPLE, {"present": False, "count": 3}),
])
def test_invalid_values_rejected(field, value):
    with pytest.raises(InvalidValue):
        coerce_value(field, value, message_text="x")


# --- change_type (ข้อ 6.7) --------------------------------------------------------


def test_scenario_r09_pending_answer_is_initial_even_if_gemini_says_correction():
    r = merge(base_state(), [upd(F.WATER_LEVEL, {"raw": "80 เซนติเมตร", "cm": 80}, raw="80 เซนติเมตร",
                                 is_correction=True)], "80 เซนติเมตร")
    c = r.change_for(F.WATER_LEVEL)
    assert c.change_type == FieldChangeType.INITIAL and c.previous_value is None
    assert r.state.water_level.value.cm == 80 and r.state.water_level.history == []


def test_scenario_r10_same_value_is_confirmation_without_event_but_activity():
    s = with_water(cm=80, raw="80 เซนติเมตร")
    r = merge(s, [upd(F.WATER_LEVEL, {"raw": "80 เซนติเมตร", "cm": 80}, raw="80 เซนติเมตร")], "ยังประมาณ 80 เซนติเมตรค่ะ")
    assert r.changes == [] and r.confirmations == [F.WATER_LEVEL]
    assert r.touched and r.state.last_activity_at == NOW
    assert r.state.water_level.history == []


def test_scenario_g04_correction_needs_gemini_and_keyword():
    s = with_water(cm=50, raw="50 เซนติเมตร")
    r = merge(s, [upd(F.WATER_LEVEL, {"raw": "80 เซนติเมตร", "cm": 80}, raw="80 เซนติเมตร", is_correction=True)],
              "แก้เป็น 80 เซนติเมตรครับ พิมพ์ผิด")
    c = r.change_for(F.WATER_LEVEL)
    assert c.change_type == FieldChangeType.CORRECTION
    assert c.previous_value["cm"] == 50 and c.previous_reported_at == EARLIER and c.reported_at == NOW
    rev = r.state.water_level.history[-1]
    assert rev.value.cm == 50 and rev.change_type == FieldChangeType.CORRECTION


def test_gemini_alone_is_not_enough_for_correction():
    s = with_water(cm=50, raw="50 เซนติเมตร")
    r = merge(s, [upd(F.WATER_LEVEL, {"raw": "80 เซนติเมตร", "cm": 80}, raw="80 เซนติเมตร", is_correction=True)],
              "ตอนนี้ 80 เซนติเมตรแล้ว")
    assert r.change_for(F.WATER_LEVEL).change_type == FieldChangeType.UPDATE


def test_keyword_alone_is_not_enough_for_correction():
    s = with_water(cm=50, raw="50 เซนติเมตร")
    r = merge(s, [upd(F.WATER_LEVEL, {"raw": "80 เซนติเมตร", "cm": 80}, raw="80 เซนติเมตร", is_correction=False)],
              "แก้เป็น 80 เซนติเมตร")
    assert r.change_for(F.WATER_LEVEL).change_type == FieldChangeType.UPDATE


def test_scenario_g08_situation_change_is_update():
    s = with_water(cm=80, raw="80 เซนติเมตร")
    r = merge(s, [upd(F.WATER_LEVEL, {"raw": "เป็นเมตร", "cm": 100}, raw="เป็นเมตร")], "ตอนนี้น้ำขึ้นเป็นเมตรแล้วครับ")
    c = r.change_for(F.WATER_LEVEL)
    assert c.change_type == FieldChangeType.UPDATE and c.previous_value["cm"] == 80 and c.new_value["cm"] == 100


def test_different_formats_are_update_not_converted():
    """ถึงเอว → 80 เซนติเมตร: ไม่แปลงหน่วยเพื่อเทียบ"""
    s = with_water(descriptor="waist", raw="ถึงเอว")
    r = merge(s, [upd(F.WATER_LEVEL, {"raw": "80 เซนติเมตร", "cm": 80}, raw="80 เซนติเมตร")], "80 เซนติเมตร")
    assert r.change_for(F.WATER_LEVEL).change_type == FieldChangeType.UPDATE
    assert r.state.water_level.value.descriptor is None  # ค่าใหม่แทนที่ ค่าเดิมอยู่ใน history
    assert r.state.water_level.history[-1].value.descriptor == WaterDescriptor.WAIST


def test_scenario_r11_correction_bound_to_its_field():
    s = merge(base_state(), [
        upd(F.LOCATION_AREA, {"name": "ตำบลทับยา", "level": "tambon"}, raw="ตำบลทับยา"),
        upd(F.WATER_LEVEL, {"raw": "50 เซนติเมตร", "cm": 50}, raw="50 เซนติเมตร"),
    ], "ตำบลทับยา 50 เซนติเมตร", now=EARLIER).state
    r = merge(s, [
        upd(F.LOCATION_AREA, {"name": "ตำบลน้ำตาล", "level": "tambon"}, raw="ตำบลน้ำตาล", is_correction=True),
        upd(F.WATER_LEVEL, {"raw": "80 เซนติเมตร", "cm": 80}, raw="น้ำขึ้นเป็น 80 เซนติเมตร", is_correction=True),
    ], "พิมพ์ตำบลผิด เป็นตำบลน้ำตาล แต่น้ำขึ้นเป็น 80 เซนติเมตร")
    assert r.change_for(F.LOCATION_AREA).change_type == FieldChangeType.CORRECTION
    assert r.change_for(F.WATER_LEVEL).change_type == FieldChangeType.UPDATE


def test_raw_not_in_message_is_dropped_and_never_correction():
    s = with_water(cm=50, raw="50 เซนติเมตร")
    r = merge(s, [upd(F.WATER_LEVEL, {"raw": "แปดสิบ", "cm": 80}, raw="แก้เป็นแปดสิบ", is_correction=True)],
              "พิมพ์ผิด 80 ซม.")
    assert r.change_for(F.WATER_LEVEL).change_type == FieldChangeType.UPDATE
    assert r.state.water_level.raw is None


def test_situation_same_tags_is_confirmation_but_keeps_detail():
    """scenario R08: รายละเอียดเพิ่มเล็กน้อย ไม่ส่ง update แต่ไม่ทิ้งข้อความใหม่"""
    s = merge(base_state(), [upd(F.SITUATION, {"text": "x", "tags": ["water_in_house"]})], "น้ำเข้าบ้าน", now=EARLIER).state
    r = merge(s, [upd(F.SITUATION, {"text": "x", "tags": ["water_in_house"]})], "ชั้นล่างเปียกหมด")
    assert r.changes == [] and r.confirmations == [F.SITUATION]
    assert r.state.situation.value.text == "น้ำเข้าบ้าน / ชั้นล่างเปียกหมด"


def test_needs_accumulate_unless_corrected():
    s = merge(base_state(), [upd(F.NEEDS, ["rescue"])], "ช่วยด้วย", now=EARLIER).state
    r = merge(s, [upd(F.NEEDS, ["food_water"])], "ขออาหารด้วย")
    assert set(r.state.needs.value) == {Need.RESCUE, Need.FOOD_WATER}
    r2 = merge(s, [upd(F.NEEDS, ["food_water"], raw="ไม่ใช่กู้ภัย ขออาหาร", is_correction=True)], "ไม่ใช่กู้ภัย ขออาหาร")
    assert r2.state.needs.value == [Need.FOOD_WATER]


# --- unknown / declined / ค่าผิด ----------------------------------------------------


def test_unknown_sets_status_without_value():
    r = merge(base_state(), [upd(F.AFFECTED_PEOPLE, status="unknown")], "ไม่แน่ใจครับ")
    assert r.state.affected_people.status == FieldStatus.UNKNOWN and r.status_changes == {F.AFFECTED_PEOPLE: FieldStatus.UNKNOWN}
    assert r.changes == [] and r.touched


def test_unknown_does_not_erase_known_value():
    s = with_water(cm=80, raw="80 เซนติเมตร")
    r = merge(s, [upd(F.WATER_LEVEL, status="unknown")], "ไม่แน่ใจแล้ว")
    assert r.state.water_level.value.cm == 80 and len(r.rejected) == 1


def test_declined_contact_recorded_in_contact_ask():
    r = merge(base_state(), [upd(F.CONTACT, status="declined")], "ไม่สะดวกให้เบอร์ค่ะ")
    assert r.state.tracked_field(F.CONTACT).status == FieldStatus.DECLINED


def test_known_contact_is_handed_off_not_stored_in_state():
    r = merge(base_state(), [upd(F.CONTACT, "[เบอร์โทร]")], "เบอร์ [เบอร์โทร] ค่ะ")
    assert r.contact_updates and r.state.contact_ask.status == FieldStatus.NOT_ASKED
    assert r.changes == []  # ไม่มีข้อมูลติดต่อหลุดเข้า event


def test_invalid_update_is_rejected_others_still_applied():
    r = merge(base_state(), [upd(F.WATER_LEVEL, {"raw": "เยอะ"}), upd(F.NEEDS, ["info_only"])], "น้ำเยอะ แจ้งไว้ก่อน")
    assert len(r.rejected) == 1 and r.state.needs.value == [Need.INFO_ONLY]


def test_duplicate_field_uses_first():
    r = merge(base_state(), [upd(F.NEEDS, ["rescue"]), upd(F.NEEDS, ["food_water"])], "ช่วยด้วย")
    assert r.state.needs.value == [Need.RESCUE] and len(r.rejected) == 1


def test_no_updates_keeps_same_state_object():
    s = base_state()
    r = merge(s, [], "ต้องตัดไฟตอนไหนคะ")
    assert r.state is s and not r.touched   # scenario R04: คำถามความรู้ไม่ต่ออายุ incident


def test_merge_does_not_touch_lifecycle_or_flags():
    s = base_state(urgent_flags=[{"flag": "people_trapped", "sources": ["gemini"], "detected_at": EARLIER}])
    r = merge(s, [upd(F.NEEDS, ["rescue"])], "ช่วยด้วย")
    assert r.state.status == IncidentStatus.DRAFT and r.state.urgent_flag_types == s.urgent_flag_types


# --- การถาม (ข้อ 6.9) ----------------------------------------------------------------


def test_record_questions_asked_counts_and_limits():
    s = record_questions_asked(base_state(), [F.WATER_LEVEL, F.LOCATION_AREA], now=NOW)
    assert s.water_level.ask_count == 1 and s.water_level.last_asked_at == NOW
    assert not can_ask_now(s, F.WATER_LEVEL)          # water_level ถามครั้งเดียว
    assert can_ask_now(s, F.LOCATION_AREA)            # field อื่นถามได้ 2 ครั้ง
    s = record_questions_asked(s, [F.LOCATION_AREA], now=NOW)
    assert not can_ask_now(s, F.LOCATION_AREA)


def test_contact_ask_counted_in_contact_ask():
    s = record_questions_asked(base_state(), [F.CONTACT], now=NOW)
    assert s.contact_ask.ask_count == 1 and not can_ask_now(s, F.CONTACT)


def test_scenario_r18_skipped_question_becomes_unanswered_not_unknown():
    s = record_questions_asked(base_state(), [F.WATER_LEVEL], now=EARLIER)
    pending = PendingQuestion(incident_id=s.incident_id, field_names=[F.WATER_LEVEL], asked_at=EARLIER)
    s = close_unanswered(merge(s, [], "ต้องตัดไฟตอนไหนคะ").state, pending)
    assert s.water_level.status == FieldStatus.UNANSWERED
    assert not can_ask_now(s, F.WATER_LEVEL)


def test_unanswered_only_after_reaching_limit():
    s = record_questions_asked(base_state(), [F.LOCATION_AREA], now=EARLIER)
    pending = PendingQuestion(incident_id=s.incident_id, field_names=[F.LOCATION_AREA], asked_at=EARLIER)
    s = close_unanswered(s, pending)
    assert s.location_area.status == FieldStatus.NOT_ASKED and can_ask_now(s, F.LOCATION_AREA)


def test_answered_pending_field_not_marked_unanswered():
    s = record_questions_asked(base_state(), [F.WATER_LEVEL], now=EARLIER)
    pending = PendingQuestion(incident_id=s.incident_id, field_names=[F.WATER_LEVEL], asked_at=EARLIER)
    s = merge(s, [upd(F.WATER_LEVEL, {"raw": "ถึงเข่า", "descriptor": "knee"})], "ถึงเข่า").state
    assert close_unanswered(s, pending).water_level.status == FieldStatus.KNOWN


def test_missing_required_reports_unknown_and_unanswered_too():
    s = merge(base_state(), [upd(F.NEEDS, ["info_only"]), upd(F.LOCATION_AREA, status="unknown")], "แจ้งไว้", now=EARLIER).state
    assert missing_required(s) == [F.LOCATION_AREA, F.SITUATION]


# --- pending question และ Context 5 ชั้น (ข้อ 3.3) ------------------------------------


def _msg(i, hours_ago, text="ข้อความ"):
    return Message(message_id=f"m{i}", conversation_id="c", role=MessageRole.USER, text=text,
                   created_at=NOW - timedelta(hours=hours_ago))


def test_pending_invalid_when_incident_closed_or_answered():
    s = base_state()
    p = PendingQuestion(incident_id=s.incident_id, field_names=[F.WATER_LEVEL], asked_at=EARLIER)
    assert pending_is_valid(p, s)
    closed = base_state(incident_id=s.incident_id, status=IncidentStatus.CLOSED, submitted_at=EARLIER,
                        close_reason=CloseReason.USER_RESOLVED, closed_at=NOW)
    assert not pending_is_valid(p, closed)
    answered = merge(s, [upd(F.WATER_LEVEL, {"raw": "ถึงเข่า", "descriptor": "knee"})], "ถึงเข่า").state
    assert not pending_is_valid(p, answered)


def test_recent_messages_keep_pending_question_older_than_24h():
    msgs = [_msg(1, 30), _msg(2, 29), _msg(3, 2)]
    p = PendingQuestion(incident_id="i", field_names=[F.WATER_LEVEL], asked_at=NOW - timedelta(hours=29),
                        asked_message_id="m2")
    assert [m.message_id for m in select_recent_messages(msgs, p, now=NOW, pending_valid=True)] == ["m2", "m3"]
    assert [m.message_id for m in select_recent_messages(msgs, p, now=NOW, pending_valid=False)] == ["m3"]


def test_recent_messages_capped():
    msgs = [_msg(i, 1) for i in range(15)]
    assert len(select_recent_messages(msgs, None, now=NOW, pending_valid=False, max_messages=10)) == 10


def test_context_trims_summary_then_old_messages_never_layers_1_2():
    s = base_state()
    p = PendingQuestion(incident_id=s.incident_id, field_names=[F.WATER_LEVEL], asked_at=NOW - timedelta(hours=1),
                        asked_message_id="m0")
    msgs = [_msg(i, 1 - i * 0.01, "ก" * 300) for i in range(5)]
    summary = ConversationSummary(conversation_id="c", text="ส" * 2000, covers_until_message_id="m0")
    budget = len(s.model_dump_json()) + len(p.model_dump_json()) + 700
    ctx = build_context(active=s, pending=p, open_incidents=[s], messages=msgs, summary=summary, now=NOW,
                        max_chars=budget)
    assert ctx.active_incident == s and ctx.pending_question == p     # ชั้น 1–2 อยู่ครบ
    assert ctx.summary is None                                        # ตัด summary ก่อน
    assert 1 <= len(ctx.recent_messages) < 5 and ctx.recent_messages[-1].message_id == "m4"  # เก็บข้อความล่าสุด


def test_context_drops_invalid_pending_and_closed_incidents():
    s = base_state()
    p = PendingQuestion(incident_id="other", field_names=[F.WATER_LEVEL], asked_at=EARLIER)
    closed = base_state(status=IncidentStatus.CLOSED, submitted_at=EARLIER, close_reason=CloseReason.TIMEOUT, closed_at=NOW)
    ctx = build_context(active=s, pending=p, open_incidents=[s, closed], messages=[], summary=None, now=NOW)
    assert ctx.pending_question is None and [b.incident_id for b in ctx.open_incidents] == [s.incident_id]


def test_open_incident_brief_uses_area_code_when_available():
    s = merge(base_state(), [upd(F.LOCATION_AREA, {"name": "ทับยา"})], "ทับยา").state
    ctx = build_context(active=None, pending=None, open_incidents=[s], messages=[], summary=None, now=NOW)
    assert ctx.open_incidents[0].location_area == "ทับยา" and ctx.open_incidents[0].area_code is None
