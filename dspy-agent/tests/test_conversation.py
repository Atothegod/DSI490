"""
ทดสอบ conversation.py (ไม่ใช้ LLM) — คำตอบของ Gemini เขียนไว้ล่วงหน้าเป็น ExtractionResult
จุดหลัก 3 จุด:
  A. needs: ไม่เติม rescue จาก people_trapped เอง และเอา info_only ที่ขัดกับคำขอความช่วยเหลือออก
  B. confirm_report รับเฉพาะเมื่อมีคำขอยืนยันที่ส่งสำเร็จและข้อมูลยังเหมือนตอนแสดงสรุป
  C. urgent flag → ส่งทันที ถามต่อตามลำดับเหตุด่วน ไม่ขอยืนยันอีก
"""

from datetime import datetime, timedelta

import pytest

from app.conversation import (
    PendingConfirmation,
    after_delivery,
    apply_submission,
    next_questions,
    plan_turn,
    reconcile_needs,
    render_reply,
    render_summary,
    summary_fingerprint,
)
from app.memory import apply_updates, close_unanswered, record_questions_asked
from app.safety import keyword_flags, load_templates, merge_urgent_flags, redact_phones
from app.schemas import (
    BANGKOK_TZ,
    ExtractionResult,
    FieldChangeType,
    FieldStatus,
    FieldUpdate,
    Hotline,
    IncidentFieldName as F,
    IncidentState,
    IncidentStatus,
    Need,
    NextAction,
    UrgentFlag,
)

NOW = datetime(2026, 10, 30, 12, 0, tzinfo=BANGKOK_TZ)
T = load_templates()  # ใช้ templates_th.json จริง เพื่อตรวจว่า key ที่ conversation.py ใช้มีครบ
HOTLINES = [Hotline(key="a", name="สายด่วนทดสอบ ก", number="1111", source_url="https://example.org", verified_at=NOW),
            Hotline(key="b", name="สายด่วนทดสอบ ข", number="2222", source_url="https://example.org", verified_at=NOW)]
FORBIDDEN = ["เจ้าหน้าที่กำลังไป", "กำลังส่งเจ้าหน้าที่", "ได้รับความช่วยเหลือแล้ว", "ทีมกู้ภัยกำลังเดินทาง"]
CONFIRM_PHRASE = "กรุณายืนยันให้ส่งรายงาน"


def upd(field, value=None, status="known", raw=None, is_correction=False):
    return FieldUpdate(field=field, status=status, value=value, raw=raw, is_correction=is_correction)


def ex(*updates, intents=("update_incident",), flags=(), question=None, target="active_incident"):
    return ExtractionResult(intents=list(intents), target=target, updates=list(updates),
                            urgent_flags=list(flags), question=question)


AREA = upd(F.LOCATION_AREA, "ตำบลทับยา อำเภออินทร์บุรี")
SITUATION = upd(F.SITUATION, {"text": "x", "tags": ["water_in_house"]})
INFO_ONLY = upd(F.NEEDS, ["info_only"])


class Session:
    """จำลองลำดับเดียวกับ Notebook ส่วน E: redact → extract(ปลอม) → merge → flags → reconcile → plan → ส่ง"""

    def __init__(self):
        self.state: IncidentState | None = None
        self.pq = None
        self.pc: PendingConfirmation | None = None
        self.n = 0

    def turn(self, text: str, extraction: ExtractionResult, *, deliver: bool = True):
        self.n += 1
        now, mid = NOW + timedelta(minutes=self.n), f"m{self.n}"
        red = redact_phones(text)
        kw = keyword_flags(red.text)
        if self.state is None:
            self.state = IncidentState(internal_user_id="u", conversation_id="c", created_at=NOW,
                                       updated_at=NOW, last_activity_at=NOW)
        m = apply_updates(self.state, extraction.updates, message_text=red.text, message_id=mid, now=now)
        fl = merge_urgent_flags(m.state.urgent_flags, kw, extraction.urgent_flags, detected_at=now, message_id=mid)
        st = IncidentState.model_validate({**m.state.model_dump(),
                                           "urgent_flags": [r.model_dump() for r in fl.records]})
        st, extra = reconcile_needs(st, fl.newly_detected, now=now, message_id=mid)
        st = close_unanswered(st, self.pq)
        plan = plan_turn(st, extraction=extraction, merge=m, new_flags=fl.newly_detected,
                         extra_changes=[extra] if extra else [], pending_confirmation=self.pc, message_text=red.text)
        if plan.submit_trigger:
            st = apply_submission(st, now=now)
        reply = render_reply(plan, st, hotlines=HOTLINES, templates=T)
        if deliver:
            d = after_delivery(st, plan, message_id=f"{mid}_reply", now=now,
                               pending_question=self.pq, pending_confirmation=self.pc)
            st, self.pq, self.pc = d.state, d.pending_question, d.pending_confirmation
        self.state = st
        self.last = (plan, reply, extra, m)
        return plan, reply

    def to_summary(self):
        """เหตุทั่วไปที่ตอบครบจนถึงขั้นขอยืนยัน"""
        self.turn("น้ำเข้าบ้าน ตำบลทับยา อำเภออินทร์บุรี แค่อยากแจ้งไว้", ex(AREA, SITUATION, INFO_ONLY,
                                                                          intents=["report_flood"]))
        self.turn("ไม่ทราบค่ะ", ex(upd(F.AFFECTED_PEOPLE, status="unknown"), upd(F.VULNERABLE_PEOPLE, status="unknown")))
        plan, reply = self.turn("ไม่ทราบค่ะ", ex(upd(F.WATER_LEVEL, status="unknown")))
        assert plan.show_summary and NextAction.CONFIRM_REPORT in plan.actions
        return plan, reply


def no_forbidden(text):
    assert not [c for c in FORBIDDEN if c in text], text


# ---------------------------------------------------------------------------
# A. needs
# ---------------------------------------------------------------------------


class TestNeeds:
    def test_people_trapped_does_not_add_rescue(self):
        s = Session()
        s.turn("น้ำเข้าบ้าน แค่อยากแจ้งไว้", ex(AREA, SITUATION, INFO_ONLY, intents=["report_flood"]))
        s.turn("ยายติดอยู่บนหลังคา", ex(SITUATION, flags=["people_trapped"]))
        assert UrgentFlag.PEOPLE_TRAPPED in s.state.urgent_flag_types
        assert UrgentFlag.HELP_REQUESTED not in s.state.urgent_flag_types
        assert s.state.needs.value == [Need.INFO_ONLY]  # ไม่เดา rescue และไม่ลบ info_only เพราะผู้ใช้ไม่ได้ขอช่วย

    def test_help_request_removes_conflicting_info_only_without_guessing(self):
        s = Session()
        s.turn("น้ำเข้าบ้าน แค่อยากแจ้งไว้", ex(AREA, SITUATION, INFO_ONLY, intents=["report_flood"]))
        s.turn("ช่วยด้วยค่ะ", ex(intents=["request_help"], flags=["help_requested"]))
        _, _, extra, _ = s.last
        assert s.state.needs.status == FieldStatus.NOT_ASKED and s.state.needs.value is None
        assert s.state.needs.history[-1].value == [Need.INFO_ONLY]
        assert extra.change_type == FieldChangeType.UPDATE and extra.previous_value == ["info_only"]
        assert extra.new_value is None

    def test_new_need_replaces_info_only_not_appended(self):
        s = Session()
        s.turn("น้ำเข้าบ้าน แค่อยากแจ้งไว้", ex(AREA, SITUATION, INFO_ONLY, intents=["report_flood"]))
        s.turn("ช่วยด้วย ขอเรือออกจากบ้าน", ex(upd(F.NEEDS, ["rescue"]), intents=["request_help"],
                                              flags=["help_requested"]))
        _, _, extra, m = s.last
        assert s.state.needs.value == [Need.RESCUE]
        assert extra is None  # merge เอา info_only ออกแล้ว ไม่สร้าง change ซ้ำ
        change = m.change_for(F.NEEDS)
        assert change.change_type == FieldChangeType.UPDATE
        assert change.previous_value == ["info_only"] and change.new_value == ["rescue"]

    def test_info_only_dropped_when_given_with_real_need(self):
        s = Session()
        s.turn("แจ้งไว้ก่อน แต่อยากอพยพ", ex(AREA, SITUATION, upd(F.NEEDS, ["info_only", "evacuation"]),
                                            intents=["report_flood"]))
        assert s.state.needs.value == [Need.EVACUATION]

    def test_help_request_keeps_existing_real_need(self):
        s = Session()
        s.turn("ขออพยพ", ex(AREA, SITUATION, upd(F.NEEDS, ["evacuation"]), intents=["report_flood"]))
        s.turn("ช่วยด้วยค่ะ", ex(intents=["request_help"], flags=["help_requested"]))
        assert s.state.needs.value == [Need.EVACUATION]
        assert s.last[2] is None


# ---------------------------------------------------------------------------
# B. การยืนยันรายงาน
# ---------------------------------------------------------------------------


class TestConfirmation:
    def test_confirm_without_request_is_ignored(self):
        s = Session()
        plan, _ = s.turn("น้ำเข้าบ้าน ยืนยันค่ะ", ex(SITUATION, intents=["report_flood", "confirm_report"]))
        assert plan.submit_trigger is None
        assert s.state.status == IncidentStatus.DRAFT
        assert any("ไม่รับ confirm_report" in n for n in plan.notes)
        assert plan.ask_fields == [F.LOCATION_AREA]

    def test_summary_not_delivered_means_no_confirmation(self):
        s = Session()
        s.turn("น้ำเข้าบ้าน ตำบลทับยา แค่แจ้งไว้", ex(AREA, SITUATION, INFO_ONLY, intents=["report_flood"]))
        s.turn("ไม่ทราบ", ex(upd(F.AFFECTED_PEOPLE, status="unknown"), upd(F.VULNERABLE_PEOPLE, status="unknown")))
        plan, _ = s.turn("ไม่ทราบ", ex(upd(F.WATER_LEVEL, status="unknown")), deliver=False)  # LINE ส่งไม่สำเร็จ
        assert plan.show_summary and s.pc is None
        plan, _ = s.turn("ยืนยันค่ะ", ex(intents=["confirm_report"]))
        assert plan.submit_trigger is None and s.state.status == IncidentStatus.DRAFT
        assert plan.show_summary  # แสดงสรุปอีกครั้ง คราวนี้ส่งสำเร็จ
        assert s.pc is not None

    def test_confirm_after_delivered_summary_submits(self):
        s = Session()
        _, reply = s.to_summary()
        assert CONFIRM_PHRASE in reply and "ทับยา" in reply
        assert s.pc is not None and s.pq is None
        plan, reply = s.turn("ถูกต้องค่ะ", ex(intents=["confirm_report"]))
        assert plan.submit_trigger == "user_confirmed" and NextAction.SUBMIT_REPORT in plan.actions
        assert s.state.status == IncidentStatus.SUBMITTED and s.state.submitted_at is not None
        assert plan.ask_fields == [] and not plan.show_summary
        assert T["report_submitted"]["user_confirmed"] in reply
        no_forbidden(reply)
        assert s.pc is None

    def test_change_after_summary_requires_new_summary(self):
        s = Session()
        s.to_summary()
        old = s.pc.summary_fingerprint
        plan, reply = s.turn("ยืนยันค่ะ แต่น้ำถึงเข่าแล้ว", ex(upd(F.WATER_LEVEL, {"raw": "ถึงเข่า", "descriptor": "knee"}),
                                                             intents=["confirm_report", "update_incident"]))
        assert plan.submit_trigger is None and s.state.status == IncidentStatus.DRAFT
        assert plan.show_summary and plan.summary_changed
        assert T["summary_changed"] in reply and "ถึงเข่า" in reply
        assert s.pc.summary_fingerprint != old
        plan, _ = s.turn("ยืนยันค่ะ", ex(intents=["confirm_report"]))
        assert plan.submit_trigger == "user_confirmed"

    def test_change_after_summary_with_new_question_asks_first(self):
        s = Session()
        s.to_summary()
        plan, _ = s.turn("ขออพยพด้วยค่ะ", ex(upd(F.NEEDS, ["evacuation"])))
        assert not plan.show_summary
        assert plan.ask_fields == [F.LOCATION_COORDINATES, F.LOCATION_LANDMARK]  # ข้อมูลใหม่ทำให้มีคำถามใหม่
        assert s.pc is None  # สรุปเดิมใช้ไม่ได้แล้ว
        plan, _ = s.turn("ยืนยันค่ะ", ex(intents=["confirm_report"]))
        assert plan.submit_trigger is None  # ต้องเห็นสรุปใหม่ก่อน
        assert plan.show_summary

    def test_knowledge_question_keeps_waiting_confirmation(self):
        s = Session()
        s.to_summary()
        plan, _ = s.turn("ต้องตัดไฟตอนไหนคะ", ex(intents=["ask_information"], question="ต้องตัดไฟตอนไหน", target="none"))
        assert plan.keep_confirmation and not plan.show_summary
        assert NextAction.ANSWER_QUESTION in plan.actions
        assert s.pc is not None
        plan, _ = s.turn("ยืนยันค่ะ", ex(intents=["confirm_report"]))
        assert plan.submit_trigger == "user_confirmed"

    def test_unrelated_reply_shows_same_summary_again(self):
        s = Session()
        s.to_summary()
        plan, reply = s.turn("โอเค", ex(intents=["general_conversation"], target="none"))
        assert plan.show_summary and not plan.summary_changed and CONFIRM_PHRASE in reply

    def test_confirmation_of_other_incident_is_ignored(self):
        s = Session()
        s.to_summary()
        s.pc = s.pc.model_copy(update={"incident_id": "inc_other"})
        plan, _ = s.turn("ยืนยันค่ะ", ex(intents=["confirm_report"]))
        assert plan.submit_trigger is None and s.state.status == IncidentStatus.DRAFT


# ---------------------------------------------------------------------------
# C. เหตุด่วน
# ---------------------------------------------------------------------------


class TestUrgent:
    def test_urgent_submits_immediately_and_asks_follow_up(self):
        s = Session()
        plan, reply = s.turn("ช่วยด้วย ยายติดอยู่บนหลังคา ตำบลน้ำตาล",
                             ex(upd(F.LOCATION_AREA, "ตำบลน้ำตาล"), SITUATION, intents=["report_flood", "request_help"],
                                flags=["people_trapped", "help_requested"], target="new_incident"))
        assert plan.submit_trigger == "urgent_flags"
        assert plan.actions[:2] == [NextAction.SUBMIT_REPORT, NextAction.ASK_FOLLOW_UP]
        assert NextAction.CONFIRM_REPORT not in plan.actions and not plan.show_summary
        assert s.state.status == IncidentStatus.SUBMITTED
        assert plan.ask_fields == [F.LOCATION_COORDINATES, F.LOCATION_LANDMARK]
        assert "1111" in reply and "2222" in reply
        assert T["report_submitted"]["urgent_flags"] in reply and CONFIRM_PHRASE not in reply
        no_forbidden(reply)

    def test_urgent_follow_up_order_and_never_confirmation(self):
        s = Session()
        s.turn("ช่วยด้วย ยายติดอยู่บนหลังคา", ex(SITUATION, intents=["request_help"],
                                                flags=["people_trapped", "help_requested"], target="new_incident"))
        assert s.pq.field_names == [F.LOCATION_AREA]  # ยังไม่มีพื้นที่ ถามก่อน
        asked = []
        replies = [("ตำบลน้ำตาล", ex(upd(F.LOCATION_AREA, "ตำบลน้ำตาล"))),
                   ("หน้าวัดค่ะ", ex(upd(F.LOCATION_LANDMARK, "หน้าวัด"))),
                   ("ไม่ทราบ", ex(upd(F.AFFECTED_PEOPLE, status="unknown"), upd(F.VULNERABLE_PEOPLE, status="unknown"))),
                   ("ไม่สะดวก", ex(upd(F.CONTACT, status="declined"))),
                   ("ขอเรือค่ะ", ex(upd(F.NEEDS, ["rescue"]))),
                   ("ขอบคุณค่ะ", ex(intents=["general_conversation"], target="none"))]
        for text, e in replies:
            plan, reply = s.turn(text, e)
            asked.append(plan.ask_fields)
            assert plan.submit_trigger is None and not plan.show_summary
            assert NextAction.CONFIRM_REPORT not in plan.actions and CONFIRM_PHRASE not in reply
        assert asked == [
            [F.LOCATION_COORDINATES, F.LOCATION_LANDMARK],
            [F.AFFECTED_PEOPLE, F.VULNERABLE_PEOPLE],  # ได้จุดสังเกตแล้ว ไม่ถามพิกัดอีก
            [F.CONTACT],
            [F.NEEDS],  # help_requested เอา info_only ออกไม่ได้เพราะยังไม่เคยมี needs → ถามท้ายสุด
            [],
            [],
        ]
        assert s.state.needs.value == [Need.RESCUE]

    def test_urgent_updates_after_submit_are_send_update(self):
        s = Session()
        s.turn("ช่วยด้วย ยายติดอยู่บนหลังคา ตำบลน้ำตาล", ex(upd(F.LOCATION_AREA, "ตำบลน้ำตาล"), SITUATION,
                                                           flags=["people_trapped"], target="new_incident"))
        plan, _ = s.turn("หน้าวัดค่ะ", ex(upd(F.LOCATION_LANDMARK, "หน้าวัด")))
        assert NextAction.SEND_UPDATE in plan.actions and plan.submit_trigger is None

    def test_urgent_overrides_pending_confirmation(self):
        s = Session()
        s.to_summary()
        plan, reply = s.turn("ยืนยันค่ะ ตอนนี้มีคนติดอยู่ในบ้าน", ex(SITUATION, intents=["confirm_report"],
                                                                       flags=["people_trapped"]))
        assert plan.submit_trigger == "urgent_flags" and not plan.show_summary
        assert any("urgent flag" in n for n in plan.notes)
        assert s.pc is None and CONFIRM_PHRASE not in reply

    def test_hotline_once_per_flag_type(self):
        s = Session()
        _, first = s.turn("ยายติดอยู่บนหลังคา", ex(SITUATION, flags=["people_trapped"], target="new_incident"))
        _, second = s.turn("ยายยังติดอยู่บนหลังคา", ex(SITUATION, flags=["people_trapped"]))
        assert "1111" in first and "1111" not in second
        assert [n.flags for n in s.state.hotline_notices] == [[UrgentFlag.PEOPLE_TRAPPED]]

    def test_hotline_not_recorded_when_delivery_failed(self):
        s = Session()
        s.turn("ยายติดอยู่บนหลังคา", ex(SITUATION, flags=["people_trapped"], target="new_incident"), deliver=False)
        assert s.state.hotline_notices == []
        _, reply = s.turn("ช่วยด้วย", ex(intents=["request_help"]))
        assert "1111" in reply  # ยังไม่เคยแจ้งสำเร็จ จึงแสดงอีกครั้ง


# ---------------------------------------------------------------------------
# D. ลำดับคำถามเหตุทั่วไป และข้อความตอบ
# ---------------------------------------------------------------------------


class TestGeneralFlow:
    def test_g01_short_report_asks_location_first(self):
        s = Session()
        plan, _ = s.turn("น้ำท่วมบ้านครับ", ex(SITUATION, intents=["report_flood"], target="new_incident"))
        assert plan.ask_fields == [F.LOCATION_AREA] and NextAction.ASK_FOLLOW_UP in plan.actions

    def test_g02_many_fields_asks_only_missing_people(self):
        s = Session()
        plan, _ = s.turn("น้ำเข้าบ้านที่ตำบลทับยา น้ำถึงเอว มีคนแก่สองคน แค่อยากแจ้งไว้ก่อน", ex(
            AREA, SITUATION, INFO_ONLY,
            upd(F.WATER_LEVEL, {"raw": "น้ำถึงเอว", "descriptor": "waist"}),
            upd(F.VULNERABLE_PEOPLE, {"present": True, "groups": [{"type": "elderly", "count": 2}]}),
            intents=["report_flood"], target="new_incident"))
        assert plan.ask_fields == [F.AFFECTED_PEOPLE]

    def test_optional_fields_asked_once_then_summary_lists_missing(self):
        s = Session()
        s.turn("น้ำท่วมถนน", ex(SITUATION, intents=["report_flood"], target="new_incident"))
        s.turn("ถามอะไรนะ", ex(intents=["general_conversation"], target="none"))  # ไม่ตอบพื้นที่ ครั้งที่ 1
        plan, _ = s.turn("อืม", ex(intents=["general_conversation"], target="none"))  # ครั้งที่ 2 → unanswered
        assert s.state.location_area.status == FieldStatus.UNANSWERED
        assert plan.ask_fields == [F.NEEDS]
        s.turn("แค่แจ้งค่ะ", ex(INFO_ONLY))
        plan, _ = s.turn("-", ex(intents=["general_conversation"], target="none"))  # ไม่ตอบเรื่องคน
        assert plan.ask_fields == [F.WATER_LEVEL]  # field ไม่บังคับ ถามครั้งเดียวแล้วไปต่อ
        plan, reply = s.turn("-", ex(intents=["general_conversation"], target="none"))
        assert plan.show_summary and plan.missing_fields == [F.LOCATION_AREA]
        assert "ข้อมูลที่ยังไม่มี: พื้นที่" in reply

    def test_reask_uses_second_variant(self):
        s = Session()
        s.turn("น้ำท่วมบ้าน", ex(SITUATION, intents=["report_flood"], target="new_incident"))
        _, reply = s.turn("ฮะ", ex(intents=["general_conversation"], target="none"))
        assert T["questions"]["location_area"][1] in reply

    def test_summary_has_no_phone(self):
        s = Session()
        s.turn("น้ำเข้าบ้าน ตำบลทับยา โทร 081-234-5678 แค่แจ้งไว้", ex(AREA, upd(F.SITUATION, {"tags": ["water_in_house"]}),
                                                                       INFO_ONLY, intents=["report_flood"]))
        text = render_summary(s.state, T)
        assert "081" not in text and "[เบอร์โทร]" in text  # situation เก็บข้อความที่ปิดบังเบอร์แล้ว

    def test_fingerprint_ignores_ask_count(self):
        s = Session()
        s.turn("น้ำเข้าบ้าน", ex(SITUATION, intents=["report_flood"], target="new_incident"))
        before = summary_fingerprint(s.state)
        asked = record_questions_asked(s.state, [F.NEEDS], now=NOW + timedelta(hours=1))
        assert summary_fingerprint(asked) == before

    def test_submission_only_from_draft(self):
        s = Session()
        s.to_summary()
        s.turn("ยืนยัน", ex(intents=["confirm_report"]))
        with pytest.raises(ValueError):
            apply_submission(s.state, now=NOW)

    def test_new_templates_have_no_forbidden_claims_and_end_politely(self):
        for text in [*T["report_submitted"].values(), T["summary_changed"]]:
            no_forbidden(text)
            assert text.endswith("ค่ะ")
