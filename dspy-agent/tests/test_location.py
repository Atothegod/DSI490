"""
ทดสอบการแชร์ตำแหน่งและความยินยอม (ข้อ 6.10) — ไม่เรียก Gemini
ข้อความตัวอักษรใช้ extractor ปลอม ส่วนตำแหน่งและการกดปุ่มต้องไม่เรียก extractor เลย
"""

import json
import re
from datetime import datetime, timedelta

import pytest

dspy = pytest.importorskip("dspy")  # agent.py เป็น dspy.Module

from app.agent import (  # noqa: E402
    POSTBACK_ALLOW_LOCATION,
    POSTBACK_DENY_LOCATION,
    POSTBACK_SKIP_LOCATION,
    FloodAgent,
)
from app.safety import load_templates  # noqa: E402
from app.schemas import (  # noqa: E402
    BANGKOK_TZ,
    ExtractionResult,
    FieldStatus,
    FieldUpdate,
    Hotline,
    IncidentFieldName as F,
    IncidentStatus,
)

T = load_templates()
NOW = datetime(2026, 10, 30, 12, 0, tzinfo=BANGKOK_TZ)
HOTLINES = [Hotline(key="a", name="สายด่วนทดสอบ", number="1111", source_url="https://example.org", verified_at=NOW)]
LAT, LNG = 14.95123, 100.36456


class FakeGemini:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, message, state, pending):
        self.calls.append((message, state, pending))
        return self.results.pop(0)


def ex(*updates, intents=("update_incident",), target="active_incident", flags=()):
    return ExtractionResult(intents=list(intents), target=target, updates=list(updates), urgent_flags=list(flags))


def u(field, value=None, status="known"):
    return FieldUpdate(field=field, status=status, value=value)


URGENT = ex(u(F.LOCATION_AREA, "ตำบลน้ำตาล"), u(F.SITUATION, {"tags": ["isolated"]}),
            intents=["request_help"], target="new_incident", flags=["people_trapped"])
GENERAL = ex(u(F.SITUATION, {"tags": ["water_in_house"]}), intents=["report_flood"], target="new_incident")


def make(*results):
    g = FakeGemini(*results)
    return FloodAgent(hotlines=HOTLINES, extractor=g), g


def urgent_until_location_asked():
    agent, g = make(URGENT)
    out = agent(input_text="ยายติดอยู่ในบ้าน ตำบลน้ำตาล")
    assert out.incident_status == IncidentStatus.SUBMITTED
    assert F.LOCATION_COORDINATES in out.pending_question.field_names
    return agent, g, out


class TestAskThenShare:
    def test_request_has_location_button_and_skip(self):
        agent, _, out = urgent_until_location_asked()
        assert [(q.type, q.label, q.data) for q in out.quick_replies] == [
            ("location", T["quick_replies"]["share_location"], None),
            ("postback", T["quick_replies"]["skip_location"], POSTBACK_SKIP_LOCATION)]
        req = agent.location_requests[-1]
        assert req.kind == "ask_location" and req.status == "pending"
        assert req.notice_version == T["notice_versions"]["location_share"]
        assert req.purpose in out.output_text  # เก็บข้อความแจ้งวัตถุประสงค์ที่ผู้ใช้เห็นจริง

    def test_r14_share_after_request_records_consent_without_gemini(self):
        agent, g, _ = urgent_until_location_asked()
        out = agent(location=(LAT, LNG))
        assert len(g.calls) == 1  # ตำแหน่งไม่เรียก Gemini
        st = agent.state
        assert st.location_coordinates.status == FieldStatus.KNOWN
        assert (st.location_coordinates.value.latitude, st.location_coordinates.value.longitude) == (LAT, LNG)
        consent = agent.consents[-1]
        assert st.location_consent_id == consent.consent_id
        req = agent.location_requests[-1]
        assert req.status == "granted" and consent.request_message_id != consent.response_message_id
        assert consent.notice_version == T["notice_versions"]["location_share"]
        assert st.needs.status == FieldStatus.NOT_ASKED  # การอนุญาตไม่เปลี่ยน needs (R14)
        assert T["location_received"] not in out.output_text  # urgent replies omit acknowledgement
        assert out.pending_question.field_names == [F.AFFECTED_PEOPLE]  # ถามข้อต่อไป
        assert out.quick_replies == []

    def test_coordinates_never_reach_gemini(self):
        agent, g, _ = urgent_until_location_asked()
        agent(location=(LAT, LNG))
        g.results.append(ex(u(F.AFFECTED_PEOPLE, status="unknown")))
        agent(input_text="ไม่ทราบค่ะ")
        # extractor ได้ state object แต่ Extractor จริงตัด location_coordinates ออกก่อนส่ง (ตรวจใน test ถัดไป)
        from app.extraction import Extractor
        lm = dspy.utils.DummyLM([{"result": json.dumps({"intents": ["update_incident"], "target": "active_incident"})}])
        dspy.configure(lm=lm)
        try:
            Extractor()("ไม่ทราบค่ะ", agent.state, None)
        finally:
            dspy.configure(lm=None)
        prompt = "\n".join(m["content"] for h in lm.history for m in h["messages"])
        assert not re.search(r"14\.95|100\.36", prompt)

    def test_skip_button_declines_and_stops_asking(self):
        agent, g, _ = urgent_until_location_asked()
        out = agent(postback=POSTBACK_SKIP_LOCATION)
        assert len(g.calls) == 1
        assert agent.state.location_coordinates.status == FieldStatus.DECLINED
        assert agent.location_requests[-1].status == "skipped"
        assert F.LOCATION_COORDINATES not in (out.pending_question.field_names if out.pending_question else [])
        assert T["acknowledgements"]["declined"] not in out.output_text

    def test_typing_skip_label_works_like_button(self):
        agent, g, _ = urgent_until_location_asked()
        agent(input_text=T["quick_replies"]["skip_location"])
        assert len(g.calls) == 1 and agent.location_requests[-1].status == "skipped"

    def test_share_after_giving_landmark_still_answers_request(self):
        agent, g, _ = urgent_until_location_asked()
        g.results.append(ex(u(F.LOCATION_LANDMARK, "หลังวัด")))
        agent(input_text="อยู่หลังวัดค่ะ")
        agent(location=(LAT, LNG))  # แชร์ตามมาภายหลัง ยังตอบคำขอเดิมที่รออยู่
        assert agent.state.location_consent_id is not None

    def test_request_older_than_24h_is_not_consent(self):
        agent, _, _ = urgent_until_location_asked()
        req = agent.location_requests[-1]
        agent.location_requests[-1] = req.model_copy(update={"requested_at": req.requested_at - timedelta(hours=25)})
        out = agent(location=(LAT, LNG))
        assert agent.state.location_consent_id is None
        assert agent.location_requests[0].status == "expired"
        assert agent.location_requests[-1].kind == "confirm_unsolicited"
        assert T["location_consent_confirm"] in out.output_text


class TestUnsolicited:
    def test_location_first_asks_about_incident_without_creating_case(self):
        agent, g = make()
        out = agent(location=(LAT, LNG))
        assert g.calls == [] and agent.state is None and agent.incidents == {}
        assert agent.pending_location and agent.consents == []
        assert T["location_without_incident"] in out.output_text
        assert out.quick_replies == []

    def incident_with_unsolicited_location(self):
        agent, g = make(GENERAL)
        agent(location=(LAT,LNG))
        out = agent(input_text="น้ำท่วมบ้านค่ะ")
        assert agent.state.location_consent_id is None
        assert T["location_consent_confirm"] in out.output_text
        return agent,g

    def test_allow_records_consent_then_continues(self):
        agent,_ = self.incident_with_unsolicited_location()
        out = agent(postback=POSTBACK_ALLOW_LOCATION)
        assert agent.state.location_consent_id == agent.consents[-1].consent_id
        assert agent.location_requests[-1].status == "granted"
        assert out.pending_question.field_names == [F.LOCATION_AREA]

    def test_deny_removes_coordinates_without_deleting_incident(self):
        agent,_ = self.incident_with_unsolicited_location()
        incident_id=agent.state.incident_id
        out=agent(postback=POSTBACK_DENY_LOCATION)
        f=agent.state.location_coordinates
        assert f.status == FieldStatus.DECLINED and f.value is None and f.history == []
        assert agent.state.incident_id == incident_id and agent.deletion_requested_at is None
        assert T["location_consent_declined"] in out.output_text
        assert str(LAT) not in agent.state.model_dump_json()

    def test_typing_allow_label_works_like_button(self):
        agent,g=self.incident_with_unsolicited_location()
        agent(input_text=T["quick_replies"]["allow_location"])
        assert len(g.calls)==1 and agent.state.location_consent_id is not None

    def test_location_during_general_report_answers_optional_request(self):
        agent,_=make(GENERAL)
        agent(input_text="น้ำท่วมบ้านค่ะ")
        agent(location=(LAT,LNG))
        assert agent.state.location_consent_id is not None
        assert agent.location_requests[-1].status == "granted"

    def test_stale_button_does_nothing(self):
        agent,_=self.incident_with_unsolicited_location()
        agent(postback=POSTBACK_ALLOW_LOCATION)
        out=agent(postback=POSTBACK_DENY_LOCATION)
        assert T["request_expired"] in out.output_text
        assert agent.state.location_consent_id is not None


def test_rescue_need_in_general_flow_asks_location_with_button():
    agent, _ = make(ex(u(F.LOCATION_AREA, "ตำบลทับยา"), u(F.SITUATION, {"tags": ["water_in_house"]}),
                       u(F.NEEDS, ["evacuation"]), intents=["report_flood"], target="new_incident"),
                    ex(u(F.AFFECTED_PEOPLE, status="unknown"), u(F.VULNERABLE_PEOPLE, status="unknown")))
    out = agent(input_text="น้ำเข้าบ้าน ตำบลทับยา อยากอพยพค่ะ")
    assert F.LOCATION_COORDINATES in out.pending_question.field_names
    assert out.quick_replies[0].type == "location"


def test_exactly_one_input_required():
    agent, _ = make()
    with pytest.raises(ValueError):
        agent(input_text="x", location=(LAT, LNG))
    with pytest.raises(ValueError):
        agent()
