"""Test Cases หลัก (T1–T16), Multi-Case (MC), Reminder (R), การหยุดถาม Slot (S) — หัวข้อ 14

ใช้ FakeExtractor แทน Gemini และ Template แทน Response Generator จึงไม่เรียก network
"""

from datetime import timedelta

from app.response import FORBIDDEN
from app.schemas import AffectedPeople, CaseStatus, IntakeStatus, MessageIntent, RiskFacts, SlotName, SlotStatus, now_bkk

U = "usr_test"
T1 = "น้ำท่วมบ้านสูงถึงเอว ออกไม่ได้ มีเด็ก 2 คน คนแก่ 1 คน ต้องการเรือกับอาหาร"
LOC = (14.95, 100.36)


def script_t1(llm):
    llm.on(T1, problem="น้ำท่วมเข้าบ้านสูงถึงระดับเอว และไม่สามารถออกจากบ้านได้", problem_evidence="น้ำท่วมบ้านสูงถึงเอว ออกไม่ได้",
           needs="เรือสำหรับอพยพ และอาหาร", needs_evidence="ต้องการเรือกับอาหาร",
           affected_people=AffectedPeople(children=2, elderly=1),
           affected_people_evidence={"children": "เด็ก 2 คน", "elderly": "คนแก่ 1 คน"},
           risk_facts=RiskFacts(trapped="true", unsafe_to_evacuate="true", basic_needs_unmet="true"),
           risk_fact_evidence={"trapped": "ออกไม่ได้", "unsafe_to_evacuate": "น้ำท่วมบ้านสูงถึงเอว",
                               "basic_needs_unmet": "ต้องการเรือกับอาหาร"},
           danger_signals=["ออกไม่ได้", "น้ำท่วมบ้านสูงถึงเอว"])


def start(agent, user=U):
    return agent.handle(user, postback="consent=yes")


def complete(agent, llm, user=U):
    """แจ้ง T1 แล้วให้ข้อมูลครบ 7 Slot → สรุปรอยืนยัน"""
    script_t1(llm)
    start(agent, user)
    agent.handle(user, text=T1)
    agent.handle(user, location=LOC)
    agent.handle(user, text="สมชาย ใจดี")
    return agent.handle(user, text="0812345678")


def press(agent, out, label, user=U):
    data = next(q.data for q in out.quick_replies if q.label == label)
    return agent.handle(user, postback=data)


def case(agent, case_id="INC-000001"):
    return agent.store.load_case(case_id)


def no_forbidden(text):
    return FORBIDDEN.search(text) is None


# ---------------------------------------------------------------------------
# Consent
# ---------------------------------------------------------------------------


def test_consent_first_and_text_not_stored_before_consent(agent, llm, store):
    out = agent.handle(U, text="น้ำท่วมบ้าน")
    assert "ยินยอม" in out.output_text and {q.data for q in out.quick_replies} == {"consent=yes", "consent=no"}
    assert llm.seen == [] and out.case_id is None
    assert store.user(U)["consent"] is None


def test_consent_declined_gets_emergency_channels(agent):
    out = agent.handle(U, postback="consent=no")
    assert "1669" in out.output_text and "1784" in out.output_text


def test_life_threat_before_consent_still_gets_1669(agent):
    out = agent.handle(U, text="แม่หมดสติ")
    assert "1669" in out.output_text and "ยินยอม" in out.output_text


# ---------------------------------------------------------------------------
# T1–T16
# ---------------------------------------------------------------------------


def test_t1_extraction_and_priority_h5(agent, llm):
    script_t1(llm)
    start(agent)
    out = agent.handle(U, text=T1)
    c = case(agent)
    assert c.incident.affected_people.model_dump() == {"children": 2, "elderly": 1, "disabled": None,
                                                       "pregnant_women": None, "other": None}
    assert c.incident.problem and c.incident.needs
    assert (c.priority.level.value, c.priority_rule_id) == ("high", "H5")
    assert out.pending_question == "location" and out.quick_replies[0].type == "location"
    assert "1784" in out.output_text  # trapped → 1784


def test_t2_missing_people_asks_count_and_cannot_confirm(agent, llm):
    llm.on("น้ำท่วมบ้าน ต้องการอาหาร", problem="น้ำท่วมบ้าน", problem_evidence="น้ำท่วมบ้าน",
           needs="อาหาร", needs_evidence="ต้องการอาหาร")
    start(agent)
    out = agent.handle(U, text="น้ำท่วมบ้าน ต้องการอาหาร")
    assert out.pending_question == "affected_people"
    assert not case(agent).incident.affected_people.has_anyone()
    assert SlotName.AFFECTED_PEOPLE in out.missing_slots


def test_t3_vague_count_not_filled(agent, llm):
    llm.on("มีคนในบ้านหลายคน", affected_people=AffectedPeople(other=3),
           affected_people_evidence={"other": "หลายคน"}, count_unclear=True)
    start(agent)
    out = agent.handle(U, text="มีคนในบ้านหลายคน")
    assert not case(agent).incident.affected_people.has_anyone()
    assert "ขอยืนยันจำนวน" in out.output_text


def test_count_without_evidence_is_dropped(agent, llm):
    llm.on("น้ำท่วมบ้านค่ะ", problem="น้ำท่วมบ้าน", problem_evidence="น้ำท่วมบ้าน",
           affected_people=AffectedPeople(elderly=2), affected_people_evidence={"elderly": "ผู้สูงอายุ 2 คน"})
    start(agent)
    agent.handle(U, text="น้ำท่วมบ้านค่ะ")
    assert case(agent).incident.affected_people.elderly is None


def test_t4_only_me_is_other_1(agent, llm):
    text = "ไม่มีเด็กหรือคนแก่ มีแค่ผมคนเดียว"
    llm.on(text, affected_people=AffectedPeople(other=1), affected_people_evidence={"other": "มีแค่ผมคนเดียว"})
    start(agent)
    agent.handle(U, text=text)
    c = case(agent)
    assert c.incident.affected_people.model_dump() == {"children": None, "elderly": None, "disabled": None,
                                                       "pregnant_women": None, "other": 1}
    assert c.slots[SlotName.AFFECTED_PEOPLE].status == SlotStatus.FILLED


def test_t5_needs_unknown_is_accepted(agent, llm):
    llm.on("น้ำท่วมบ้าน อยู่ 2 คน", problem="น้ำท่วมบ้าน", problem_evidence="น้ำท่วมบ้าน",
           affected_people=AffectedPeople(other=2), affected_people_evidence={"other": "อยู่ 2 คน"})
    start(agent)
    out = agent.handle(U, text="น้ำท่วมบ้าน อยู่ 2 คน")
    assert out.pending_question == "needs"
    out = agent.handle(U, text="ยังไม่รู้เหมือนกัน")
    c = case(agent)
    assert c.slots[SlotName.NEEDS].status == SlotStatus.ANSWERED_UNKNOWN
    assert out.pending_question == "location" and SlotName.NEEDS not in out.missing_slots


def test_t6_first_name_only_cannot_confirm(agent, llm):
    script_t1(llm)
    start(agent)
    agent.handle(U, text=T1)
    agent.handle(U, location=LOC)
    out = agent.handle(U, text="สมชาย")
    assert case(agent).personal.first_name == "สมชาย"
    assert out.pending_question == "last_name" and out.intake_status == IntakeStatus.COLLECTING


def test_t7_phone_not_asked_third_time(agent, llm):
    llm.on("น้ำท่วมบ้าน", problem="น้ำท่วมบ้าน", problem_evidence="น้ำท่วมบ้าน")
    start(agent)
    agent.handle(U, text="น้ำท่วมบ้าน")
    c = case(agent)
    for s in (SlotName.AFFECTED_PEOPLE, SlotName.NEEDS, SlotName.LOCATION, SlotName.FIRST_NAME, SlotName.LAST_NAME):
        c.slots[s].ask_count = 2
    c.slots[SlotName.PHONE_NUMBER].ask_count = 1
    c.slots[SlotName.PHONE_NUMBER].asked_at = now_bkk()
    c.pending_question = "phone_number"
    agent.store.save_case(c)
    out = agent.handle(U, text="ไม่สะดวกให้")
    assert out.pending_question == "phone_number"  # ครั้งที่ 2
    out = agent.handle(U, text="ไม่ให้")
    assert out.pending_question != "phone_number"
    assert case(agent).slots[SlotName.PHONE_NUMBER].ask_count == 2


def test_t8_pii_never_reaches_llm(agent, llm, store):
    llm.on("ผมชื่อ[NAME] เบอร์ [PHONE] [ADDRESS] น้ำท่วมบ้าน", problem="น้ำท่วมบ้าน", problem_evidence="น้ำท่วมบ้าน")
    start(agent)
    agent.handle(U, text="ผมชื่อสมชาย เบอร์ 0812345678 บ้านเลขที่ 12 น้ำท่วมบ้าน")
    assert llm.seen and all("สมชาย" not in t and "0812345678" not in t and "เลขที่ 12" not in t for t in llm.seen)
    c = case(agent)
    assert c.personal.first_name == "สมชาย" and c.personal.phone_number == "0812345678"
    assert "address" not in c.personal.model_dump() and "address" not in c.incident.model_dump()
    assert c.address_masked


def test_t9_interjected_question_then_resume_slot(agent, llm):
    llm.on("น้ำท่วมบ้าน", problem="น้ำท่วมบ้าน", problem_evidence="น้ำท่วมบ้าน")
    llm.on("ควรปิดไฟไหม", intents=[MessageIntent.QUESTION], question="น้ำท่วมบ้านควรตัดไฟไหม")
    start(agent)
    assert agent.handle(U, text="น้ำท่วมบ้าน").pending_question == "affected_people"
    out = agent.handle(U, text="ควรปิดไฟไหม")
    assert "core_electricity_safe_cutoff" in out.kb_doc_ids
    assert out.output_text.index("ตัดกระแสไฟ") < out.output_text.index("ส่วนเรื่องจำนวนผู้ประสบภัย")
    assert out.pending_question == "affected_people"


def test_t10_emotional_support(agent, llm):
    llm.on("กลัวมากเลย ไม่รู้จะทำยังไง", intents=[MessageIntent.EMOTIONAL])
    llm.on("น้ำท่วมบ้าน", problem="น้ำท่วมบ้าน", problem_evidence="น้ำท่วมบ้าน")
    start(agent)
    agent.handle(U, text="น้ำท่วมบ้าน")
    out = agent.handle(U, text="กลัวมากเลย ไม่รู้จะทำยังไง")
    assert "เข้าใจ" in out.output_text and no_forbidden(out.output_text)
    assert "1323" in out.output_text
    assert out.pending_question == "affected_people"


def test_t11_critical(agent, llm, store):
    llm.on("แม่หมดสติ น้ำท่วมถึงอก", problem="แม่หมดสติ น้ำท่วมถึงอก", problem_evidence="แม่หมดสติ น้ำท่วมถึงอก")
    start(agent)
    out = agent.handle(U, text="แม่หมดสติ น้ำท่วมถึงอก")
    c = case(agent)
    assert c.priority.level.value == "critical" and c.priority_rule_id == "C1"
    assert out.output_text.index("1669") < out.output_text.index("หมุด")
    assert out.pending_question == "location"
    assert store.pipeline_rows() == []
    assert no_forbidden(out.output_text)


def test_t12_unknown_shelter_not_fabricated(agent, llm):
    llm.on("ศูนย์พักพิงที่ใกล้ที่สุดอยู่ไหน", intents=[MessageIntent.QUESTION], question="ศูนย์พักพิงที่ใกล้ที่สุด")
    start(agent)
    out = agent.handle(U, text="ศูนย์พักพิงที่ใกล้ที่สุดอยู่ไหน")
    assert "1784" in out.output_text and "kb_emergency_1784" in out.kb_doc_ids
    assert "ศูนย์พักพิง" not in out.output_text  # ไม่แต่งชื่อศูนย์


def test_t12_no_kb_info(agent, llm):
    llm.on("ราคาทองวันนี้เท่าไร", intents=[MessageIntent.QUESTION], question="ราคาทองวันนี้")
    start(agent)
    out = agent.handle(U, text="ราคาทองวันนี้เท่าไร")
    assert "ยังไม่มีข้อมูลที่ยืนยันได้" in out.output_text


def test_t13_resume_after_restart(agent, llm, store):
    from app.agent import Agent1
    llm.on("น้ำท่วมบ้าน", problem="น้ำท่วมบ้าน", problem_evidence="น้ำท่วมบ้าน")
    start(agent)
    agent.handle(U, text="น้ำท่วมบ้าน")
    restarted = Agent1(store, agent.kb, extractor=llm, generator=agent.generate)
    out = restarted.handle(U, text="สวัสดีครับ")
    assert out.case_id == "INC-000001" and out.pending_question == "affected_people"
    assert "เล่าสถานการณ์" not in out.output_text


def test_t14_edits_before_confirm_create_only_v1(agent, llm, store):
    out = complete(agent, llm)
    llm.on("ขอแก้เป็นคนแก่ 2 คน", affected_people=AffectedPeople(elderly=2), affected_people_evidence={"elderly": "คนแก่ 2 คน"})
    llm.on("ขอแก้เป็นคนแก่ 3 คน", affected_people=AffectedPeople(elderly=3), affected_people_evidence={"elderly": "คนแก่ 3 คน"})
    for text in ("ขอแก้เป็นคนแก่ 2 คน", "ขอแก้เป็นคนแก่ 3 คน"):
        assert press(agent, out, "แก้ไขข้อมูล").intake_status == IntakeStatus.COLLECTING
        out = agent.handle(U, text=text)
        assert out.intake_status == IntakeStatus.AWAITING_CONFIRMATION
    assert store.version_count("INC-000001") == 0
    press(agent, out, "ยืนยันข้อมูล")
    rows = store.pipeline_rows()
    assert [r["version"] for r in rows] == [1] and rows[0]["incident"]["affected_people"]["elderly"] == 3


def test_t15_edit_after_confirm_creates_v2(agent, llm, store):
    press(agent, complete(agent, llm), "ยืนยันข้อมูล")
    llm.on("มีคนแก่เพิ่มเป็น 4 คน", affected_people=AffectedPeople(elderly=4), affected_people_evidence={"elderly": "คนแก่เพิ่มเป็น 4 คน"})
    out = agent.handle(U, text="มีคนแก่เพิ่มเป็น 4 คน")
    assert out.intake_status == IntakeStatus.AWAITING_CONFIRMATION
    assert store.pipeline_rows()[0]["version"] == 1  # ระหว่างแก้ Pipeline ยังเห็น V1
    press(agent, out, "ยืนยันข้อมูล")
    assert store.version_count("INC-000001") == 2 and store.pipeline_rows()[0]["version"] == 2
    event = store.events("INC-000001", "case_updated")[0]
    assert event["previous_version"] == 1 and event["current_version"] == 2
    assert event["changes"] == {"affected_people.elderly": {"old": 1, "new": 4}}


def test_t16_silent_user_case_stays_active(agent, llm, store):
    llm.on("น้ำท่วมบ้าน", problem="น้ำท่วมบ้าน", problem_evidence="น้ำท่วมบ้าน")
    start(agent)
    agent.handle(U, text="น้ำท่วมบ้าน")
    agent.due_reminders(now_bkk() + timedelta(days=3))
    agent.due_reminders(now_bkk() + timedelta(days=5))
    assert case(agent).case_status == CaseStatus.ACTIVE and store.pipeline_rows() == []
    assert store.events("INC-000001", "case_closed") == []


# ---------------------------------------------------------------------------
# Confirmation + Data Contract
# ---------------------------------------------------------------------------


def test_summary_list_and_contract(agent, llm, store):
    out = complete(agent, llm)
    text = out.output_text
    assert "1. ปัญหาที่พบ" in text and "เด็ก 2 คน, ผู้สูงอายุ 1 คน" in text and "08X-XXX-5678" in text
    assert "ผู้พิการ" not in text and "high" not in text  # null ไม่แสดง, ไม่แสดง Priority
    press(agent, out, "ยืนยันข้อมูล")
    row = store.pipeline_rows()[0]
    assert set(row) == {"case_id", "version", "schema_version", "case_status", "confirmed_at", "incident", "location",
                        "incident_priority"}
    assert set(row["incident_priority"]) == {"level", "reason"}
    assert "สมชาย" not in str(row) and "0812345678" not in str(row)
    assert store.events("INC-000001", "incident_submitted")


def test_stale_confirm_button_rejected(agent, llm, store):
    out = complete(agent, llm)
    old = out.quick_replies[0].data
    llm.on("เด็ก 3 คน", affected_people=AffectedPeople(children=3), affected_people_evidence={"children": "เด็ก 3 คน"})
    agent.handle(U, text="เด็ก 3 คน")
    assert "ฉบับเก่า" in agent.handle(U, postback=old).output_text
    assert store.version_count("INC-000001") == 0


def test_question_while_awaiting_confirmation(agent, llm):
    complete(agent, llm)
    llm.on("ควรปิดไฟไหม", intents=[MessageIntent.QUESTION], question="ตัดไฟ")
    out = agent.handle(U, text="ควรปิดไฟไหม")
    assert "ตัดกระแสไฟ" in out.output_text and "ยังรอการยืนยัน" in out.output_text
    assert {q.label for q in out.quick_replies} == {"ยืนยันข้อมูล", "แก้ไขข้อมูล"}


def test_priority_audit_rows(agent, llm, store):
    """U22 + U24"""
    llm.on("ขาดอาหารครับ", problem="ขาดอาหาร", problem_evidence="ขาดอาหาร",
           risk_facts=RiskFacts(basic_needs_unmet="true"), risk_fact_evidence={"basic_needs_unmet": "ขาดอาหาร"})
    llm.on("ตอนนี้ยายหมดสติ")
    start(agent)
    agent.handle(U, text="ขาดอาหารครับ")
    agent.handle(U, text="ตอนนี้ยายหมดสติ")
    evaluated = store.events("INC-000001", "priority_evaluated")
    assert [e["rule_id"] for e in evaluated][-2:] == ["M1", "C1"]
    assert all(e["ruleset_version"] == "1.1.0" for e in evaluated)
    assert store.events("INC-000001", "priority_changed")[-1]["new"] == "critical"


# ---------------------------------------------------------------------------
# Multi-Case
# ---------------------------------------------------------------------------


def test_mc1_more_info_updates_same_case(agent, llm, store):
    llm.on("น้ำท่วมบ้าน", problem="น้ำท่วมบ้าน", problem_evidence="น้ำท่วมบ้าน")
    llm.on("มีเด็ก 1 คน", affected_people=AffectedPeople(children=1), affected_people_evidence={"children": "เด็ก 1 คน"})
    start(agent)
    agent.handle(U, text="น้ำท่วมบ้าน")
    out = agent.handle(U, text="มีเด็ก 1 คน")
    assert out.case_id == "INC-000001" and len(store.active_cases(U)) == 1


def test_mc2_new_case_button(agent, llm, store):
    press(agent, complete(agent, llm), "ยืนยันข้อมูล")
    out = agent.handle(U, postback="new_case")
    assert out.case_id == "INC-000002" and store.user(U)["current_case_id"] == "INC-000002"


def test_mc3_report_for_relative_creates_new_case(agent, llm):
    press(agent, complete(agent, llm), "ยืนยันข้อมูล")
    llm.on("ขอแจ้งเหตุบ้านญาติด้วย", case_intent="new_case")
    out = agent.handle(U, text="ขอแจ้งเหตุบ้านญาติด้วย")
    assert out.case_id == "INC-000002"


def test_mc4_new_pin_asks_before_overwriting(agent, llm):
    complete(agent, llm)
    out = agent.handle(U, location=(15.0, 100.4))
    assert out.pending_question == "location_choice"
    assert case(agent).personal.location.latitude == LOC[0]


def test_mc5_relocate_confirmed_case_new_version(agent, llm, store):
    press(agent, complete(agent, llm), "ยืนยันข้อมูล")
    llm.on("ย้ายมาอยู่บนถนนแล้ว", case_intent="relocate")
    out = agent.handle(U, text="ย้ายมาอยู่บนถนนแล้ว")
    assert out.quick_replies[0].type == "location"
    out = agent.handle(U, location=(15.0, 100.4))
    assert out.intake_status == IntakeStatus.AWAITING_CONFIRMATION
    press(agent, out, "ยืนยันข้อมูล")
    assert store.pipeline_rows()[0]["location"] == {"latitude": 15.0, "longitude": 100.4}
    assert store.version_count("INC-000001") == 2


def test_mc6_ambiguous_message_with_two_cases(agent, llm):
    press(agent, complete(agent, llm), "ยืนยันข้อมูล")
    agent.handle(U, postback="new_case")
    llm.on("น้ำขึ้นอีกแล้ว", case_intent="unclear",
           risk_facts=RiskFacts(water_rising="true"), risk_fact_evidence={"water_rising": "น้ำขึ้นอีกแล้ว"})
    out = agent.handle(U, text="น้ำขึ้นอีกแล้ว")
    assert out.pending_question == "case_selection"
    assert {q.data for q in out.quick_replies} == {"case_select=INC-000001", "case_select=INC-000002"}
    out = agent.handle(U, postback="case_select=INC-000001")
    assert out.case_id == "INC-000001" and case(agent).risk_facts.water_rising == "true"


def test_mc7_nearby_pin_but_other_house_is_new_case(agent, llm):
    complete(agent, llm)
    out = agent.handle(U, location=(LOC[0] + 0.0001, LOC[1]))
    out = press(agent, out, "แจ้งเหตุใหม่")
    assert out.case_id == "INC-000002" and case(agent, "INC-000002").personal.location is not None


def test_new_case_offers_reuse_contact(agent, llm):
    press(agent, complete(agent, llm), "ยืนยันข้อมูล")
    agent.handle(U, postback="new_case")
    llm.on("บ้านแม่น้ำท่วม อยู่ 2 คน ไม่ต้องการอะไร", problem="บ้านแม่น้ำท่วม", problem_evidence="บ้านแม่น้ำท่วม",
           affected_people=AffectedPeople(other=2), affected_people_evidence={"other": "อยู่ 2 คน"}, needs_answer="none")
    agent.handle(U, text="บ้านแม่น้ำท่วม อยู่ 2 คน ไม่ต้องการอะไร")
    out = agent.handle(U, location=(15.1, 100.2))
    assert out.pending_question == "reuse_contact"
    out = press(agent, out, "ใช่")
    assert out.intake_status == IntakeStatus.AWAITING_CONFIRMATION
    assert case(agent, "INC-000002").personal.phone_number == "0812345678"


# ---------------------------------------------------------------------------
# Reminder
# ---------------------------------------------------------------------------


def _draft(agent, llm):
    llm.on("น้ำท่วมบ้าน", problem="น้ำท่วมบ้าน", problem_evidence="น้ำท่วมบ้าน")
    start(agent)
    agent.handle(U, text="น้ำท่วมบ้าน")
    return case(agent).last_user_at


def test_r1_r2_reminder_after_30_minutes(agent, llm, store):
    t = _draft(agent, llm)
    assert agent.due_reminders(t + timedelta(minutes=29)) == []
    due = agent.due_reminders(t + timedelta(minutes=30))
    assert len(due) == 1 and "ยังไม่สมบูรณ์" in due[0][2]
    assert store.events("INC-000001", "draft_reminder_sent")


def test_r3_no_second_reminder(agent, llm):
    t = _draft(agent, llm)
    agent.due_reminders(t + timedelta(minutes=30))
    agent.handle(U, text="สวัสดี")
    assert agent.due_reminders(now_bkk() + timedelta(hours=2)) == []


def test_r4_r5_critical_reminder_has_channels_no_pii(agent, llm):
    start(agent)
    agent.handle(U, text="ผมชื่อสมชาย 0812345678 ยายหมดสติ")
    due = agent.due_reminders(now_bkk() + timedelta(minutes=31))
    assert "1669" in due[0][2]
    assert "สมชาย" not in due[0][2] and "0812345678" not in due[0][2] and "14.95" not in due[0][2]


def test_r6_return_after_days(agent, llm):
    t = _draft(agent, llm)
    agent.due_reminders(t + timedelta(minutes=30))
    out = agent.handle(U, text="กลับมาแล้วครับ")
    assert out.case_id == "INC-000001" and out.pending_question == "affected_people"


def test_r7_revision_draft_gets_one_reminder(agent, llm):
    press(agent, complete(agent, llm), "ยืนยันข้อมูล")
    assert agent.due_reminders(now_bkk() + timedelta(hours=1)) == []  # S7: ยืนยันแล้วไม่ได้แก้ ไม่เตือน
    llm.on("เด็กเพิ่มเป็น 3 คน", affected_people=AffectedPeople(children=3), affected_people_evidence={"children": "เด็กเพิ่มเป็น 3 คน"})
    agent.handle(U, text="เด็กเพิ่มเป็น 3 คน")
    assert len(agent.due_reminders(now_bkk() + timedelta(minutes=31))) == 1
    assert agent.due_reminders(now_bkk() + timedelta(hours=3)) == []


# ---------------------------------------------------------------------------
# หยุดถาม Slot
# ---------------------------------------------------------------------------


def _all_but(agent, llm, *missing):
    out = complete(agent, llm)
    c = case(agent)
    for s in missing:
        if s == SlotName.PHONE_NUMBER:
            c.personal.phone_number = None
        if s == SlotName.LOCATION:
            c.personal.location = None
        c.slots[s].ask_count = 2
        c.slots[s].asked_at = now_bkk()
    c.intake_status = IntakeStatus.COLLECTING
    agent.store.save_case(c)
    return out


def test_s1_missing_notice_once(agent, llm):
    _all_but(agent, llm, SlotName.PHONE_NUMBER)
    out = agent.handle(U, text="โอเค")
    assert "ยังขาดข้อมูล เบอร์โทรศัพท์" in out.output_text
    assert out.intake_status == IntakeStatus.AWAITING_USER_INPUT and out.pending_question is None
    assert case(agent).slots[SlotName.PHONE_NUMBER].ask_count == 2
    assert "ยังขาดข้อมูล" not in agent.handle(U, text="โอเค").output_text


def test_s2_s3_notice_lists_all_with_location_button(agent, llm):
    _all_but(agent, llm, SlotName.PHONE_NUMBER, SlotName.LOCATION)
    out = agent.handle(U, text="โอเค")
    assert "หมุดจุดเกิดเหตุ, เบอร์โทรศัพท์" in out.output_text
    assert out.quick_replies[0].type == "location"


def test_s4_question_during_awaiting_input(agent, llm):
    _all_but(agent, llm, SlotName.PHONE_NUMBER)
    agent.handle(U, text="โอเค")
    llm.on("ควรปิดไฟไหม", intents=[MessageIntent.QUESTION], question="ตัดไฟ")
    out = agent.handle(U, text="ควรปิดไฟไหม")
    assert "ตัดกระแสไฟ" in out.output_text and "ยังขาดข้อมูล" not in out.output_text


def test_s5_critical_during_awaiting_input(agent, llm):
    _all_but(agent, llm, SlotName.PHONE_NUMBER)
    agent.handle(U, text="โอเค")
    out = agent.handle(U, text="แม่หมดสติ")
    assert "1669" in out.output_text and case(agent).priority.level.value == "critical"


def test_s6_missing_info_returns_to_summary(agent, llm):
    _all_but(agent, llm, SlotName.PHONE_NUMBER)
    agent.handle(U, text="โอเค")
    out = agent.handle(U, text="เบอร์ 0898765432 ครับ")
    assert out.intake_status == IntakeStatus.AWAITING_CONFIRMATION and "สรุปข้อมูล" in out.output_text


def test_s7_viewing_confirmed_case_creates_no_revision(agent, llm, store):
    press(agent, complete(agent, llm), "ยืนยันข้อมูล")
    agent.handle(U, text="ขอดูเคสหน่อย")
    c = case(agent)
    assert not c.editing and c.intake_status == IntakeStatus.CONFIRMED
    assert store.events("INC-000001", "revision_draft_opened") == []


def test_closed_case_by_backend(agent, llm, store):
    press(agent, complete(agent, llm), "ยืนยันข้อมูล")
    assert agent.close_case("INC-000001", actor="officer1", reason="ช่วยเหลือแล้ว")
    assert store.pipeline_rows()[0]["case_status"] == "inactive"
    assert store.user(U)["current_case_id"] is None
