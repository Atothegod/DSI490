"""
ทดสอบ FloodAgent ทั้งสาย ผ่าน DSPy จริง แต่ใช้ DummyLM แทน Gemini (ไม่เรียก API)
คำตอบของ "Gemini" คือ ExtractionResult ที่เขียนไว้ล่วงหน้าในแต่ละ test
ข้ามทั้งไฟล์เมื่อเครื่องไม่มี dspy
"""

import json
from datetime import datetime

import pytest

dspy = pytest.importorskip("dspy")
from dspy.utils import DummyLM  # noqa: E402

from app.agent import FloodAgent  # noqa: E402
from app.extraction import RAW_RULE, SITUATION_NEEDS_RULE  # noqa: E402
from app.safety import load_templates  # noqa: E402
from app.schemas import (  # noqa: E402
    BANGKOK_TZ,
    FieldStatus,
    Hotline,
    IncidentFieldName as F,
    IncidentStatus,
    MatchingStatus,
    Need,
    NextAction,
    UrgentFlag,
)

T = load_templates()
NOW = datetime(2026, 10, 30, 12, 0, tzinfo=BANGKOK_TZ)
HOTLINES = [Hotline(key="a", name="สายด่วนทดสอบ ก", number="1111", source_url="https://example.org", verified_at=NOW)]
FORBIDDEN = ["เจ้าหน้าที่กำลังไป", "กำลังส่งเจ้าหน้าที่", "ได้รับความช่วยเหลือแล้ว", "ทีมกู้ภัยกำลังเดินทาง"]


def g(*updates, intents=("update_incident",), target="active_incident", flags=(), question=None):
    """คำตอบปลอมของ Gemini"""
    return {"result": json.dumps({"intents": list(intents), "target": target, "updates": list(updates),
                                  "urgent_flags": list(flags), "question": question}, ensure_ascii=False)}


def u(field, value=None, status="known", raw=None, is_correction=False):
    return {"field": field, "status": status, "value": value, "raw": raw, "is_correction": is_correction}


def make(*answers):
    lm = DummyLM(list(answers))
    dspy.configure(lm=lm)
    return FloodAgent(hotlines=HOTLINES), lm


def prompts(lm) -> str:
    return "\n".join(m["content"] for h in lm.history for m in h["messages"])


SIT = u("situation", {"text": "x", "tags": ["water_in_house"]})
AREA = u("location_area", "ตำบลทับยา อำเภออินทร์บุรี")


def test_guy01_style_greeting_without_incident():
    agent, _ = make(g(intents=["general_conversation"], target="none"))
    output = agent(input_text="สวัสดีค่ะ")
    assert output.output_text == T["greeting"]
    assert output.incident_id is None and agent.state is None


def test_prompt_contains_new_rules():
    agent, lm = make(g(SIT, intents=["report_flood"], target="new_incident"))
    agent(input_text="น้ำท่วมบ้านค่ะ")
    text = prompts(lm)
    for line in (RAW_RULE + SITUATION_NEEDS_RULE).strip().splitlines():  # DSPy ย่อหน้าแต่ละบรรทัด
        assert line.strip() in text


def test_g01_first_report_asks_location():
    agent, _ = make(g(SIT, intents=["report_flood"], target="new_incident"))
    output = agent(input_text="น้ำท่วมบ้านค่ะ")
    assert output.output_text.endswith(T["questions"]["location_area"][0])
    assert output.incident_status == IncidentStatus.DRAFT
    assert output.pending_question.field_names == [F.LOCATION_AREA]
    assert agent.state.situation.value.text == "น้ำท่วมบ้านค่ะ"  # ข้อความเต็มของผู้ใช้ ไม่ใช่ที่ Gemini ตัด


def test_general_flow_to_confirmation_and_submit():
    agent, _ = make(
        g(SIT, intents=["report_flood"], target="new_incident"),
        g(AREA, u("needs", ["info_only"]), intents=["update_incident", "confirm_report"]),  # Gemini เดา confirm ผิด
        g(u("affected_people", status="unknown"), u("vulnerable_people", status="unknown")),
        g(u("water_level", {"raw": "ถึงเข่า", "descriptor": "knee"}, raw="ถึงเข่า")),
        g(intents=["confirm_report"]),
    )
    agent(input_text="น้ำท่วมบ้านค่ะ")
    out = agent(input_text="ตำบลทับยา อินทร์บุรีค่ะ แค่อยากแจ้งไว้ก่อน")
    assert out.incident_status == IncidentStatus.DRAFT  # confirm_report ไม่มีผลเพราะยังไม่เคยแสดงสรุป
    assert NextAction.SUBMIT_REPORT not in out.next_actions
    agent(input_text="ไม่ทราบค่ะ")
    out = agent(input_text="ถึงเข่าค่ะ")
    assert NextAction.CONFIRM_REPORT in out.next_actions and "ทับยา" in out.output_text
    out = agent(input_text="ถูกต้องค่ะ")
    assert out.incident_status == IncidentStatus.SUBMITTED
    assert T["report_submitted"]["user_confirmed"] in out.output_text
    assert not [c for c in FORBIDDEN if c in out.output_text]


def test_urgent_submits_immediately_and_never_asks_confirmation():
    agent, lm = make(
        g(SIT, intents=["report_flood"], target="new_incident"),
        g(AREA, u("needs", ["info_only"])),
        g(u("situation", {"text": "x", "tags": ["isolated"]}), intents=["request_help"],
          flags=["people_trapped", "help_requested"]),
        g(u("location_landmark", "หน้าวัด")),
    )
    agent(input_text="น้ำท่วมบ้านค่ะ")
    agent(input_text="ตำบลทับยา แค่แจ้งไว้ค่ะ")
    out = agent(input_text="ยายติดอยู่บนหลังคา ช่วยด้วย เบอร์ 081-234-5678")
    assert out.incident_status == IncidentStatus.SUBMITTED
    assert out.next_actions[:2] == [NextAction.SUBMIT_REPORT, NextAction.ASK_FOLLOW_UP]
    assert "1111" in out.output_text and T["report_submitted"]["urgent_flags"] in out.output_text
    assert "กรุณายืนยันให้ส่งรายงาน" not in out.output_text
    assert "ยายติดอยู่บนหลังคา" in agent.state.situation.value.text
    assert agent.state.needs.status == FieldStatus.NOT_ASKED  # info_only ถูกเอาออก ไม่เดา rescue
    # เบอร์: ไม่ถึง Gemini ไม่อยู่ใน state ไม่อยู่ในคำตอบ แต่เก็บใน ContactRecord
    assert "081" not in prompts(lm) and "5678" not in prompts(lm)
    assert "081" not in agent.state.model_dump_json() and "081" not in out.output_text
    assert agent.contacts[0].phone == "0812345678" and agent.state.contact_ask.status == FieldStatus.KNOWN
    out = agent(input_text="หน้าวัดค่ะ")
    assert NextAction.SEND_UPDATE in out.next_actions and "1111" not in out.output_text
    assert "กรุณายืนยันให้ส่งรายงาน" not in out.output_text


def test_correction_needs_raw_and_cue():
    agent, _ = make(
        g(SIT, AREA, u("needs", ["info_only"]), intents=["report_flood"], target="new_incident"),
        g(u("affected_people", status="unknown"), u("vulnerable_people", status="unknown")),
        g(u("water_level", {"raw": "80 เซนติเมตร", "cm": 80}, raw="80 เซนติเมตร", is_correction=True)),
        g(u("water_level", {"raw": "50 เซนติเมตร", "cm": 50}, raw="50 เซนติเมตร", is_correction=True)),
    )
    for text in ["น้ำเข้าบ้าน ตำบลทับยา แค่แจ้งค่ะ", "ไม่ทราบค่ะ", "80 เซนติเมตร", "แก้เป็น 50 เซนติเมตรค่ะ พิมพ์ผิด"]:
        out = agent(input_text=text)
    changes = [c.change_type.value for c in agent.turns[2].merge.changes] + \
              [c.change_type.value for c in agent.turns[3].merge.changes]
    assert changes == ["initial", "correction"]
    assert T["summary_changed"] in out.output_text  # แก้หลังแสดงสรุป → ขอยืนยันสรุปใหม่


def test_other_area_not_merged_without_urgency():
    agent, _ = make(
        g(SIT, AREA, intents=["report_flood"], target="new_incident"),
        g(u("location_area", "อำเภอพรหมบุรี"), SIT, intents=["report_flood"], target="unclear"),
    )
    agent(input_text="น้ำเข้าบ้าน ตำบลทับยาค่ะ")
    out = agent(input_text="บ้านแม่ที่พรหมบุรีน้ำเข้าแล้วค่ะ")
    assert out.next_actions == [NextAction.ASK_INCIDENT_MATCH]
    assert "ตำบลทับยา" in out.output_text
    assert "พรหมบุรี" not in agent.state.location_area.value.raw and len(agent.incidents) == 1


def test_unclear_urgent_creates_provisional_incident():
    agent, _ = make(
        g(SIT, AREA, intents=["report_flood"], target="new_incident"),
        g(u("location_area", "อำเภอพรหมบุรี"), SIT, intents=["request_help"], target="unclear",
          flags=["people_trapped"]),
    )
    agent(input_text="น้ำเข้าบ้าน ตำบลทับยาค่ะ")
    first = agent.active_id
    out = agent(input_text="บ้านแม่ที่พรหมบุรี มีคนติดอยู่ในบ้านค่ะ")
    assert len(agent.incidents) == 2 and agent.active_id != first
    assert out.matching_status == MatchingStatus.UNCONFIRMED
    assert agent.state.possible_related_incident_ids == [first]
    assert out.incident_status == IncidentStatus.SUBMITTED and "1111" in out.output_text
    assert agent.incidents[first].location_area.value.raw.startswith("ตำบลทับยา")  # เหตุเดิมไม่ถูกแก้


def test_new_need_replaces_info_only():
    agent, _ = make(
        g(SIT, AREA, u("needs", ["info_only"]), intents=["report_flood"], target="new_incident"),
        g(u("needs", ["rescue"]), intents=["request_help"], flags=["help_requested"]),
    )
    agent(input_text="น้ำเข้าบ้าน ตำบลทับยา แค่แจ้งค่ะ")
    agent(input_text="ช่วยด้วย ขอเรือออกจากบ้านค่ะ")
    assert agent.state.needs.value == [Need.RESCUE]
    assert UrgentFlag.HELP_REQUESTED in agent.state.urgent_flag_types


def test_knowledge_question_without_incident():
    agent, _ = make(g(intents=["ask_information"], target="none", question="ต้องตัดไฟตอนไหน"))
    out = agent(input_text="ต้องตัดไฟตอนไหนคะ")
    assert out.incident_id is None and NextAction.ANSWER_QUESTION in out.next_actions
    assert "1111" in out.output_text


def test_hotlines_required():
    with pytest.raises(ValueError):
        FloodAgent(hotlines=[], extractor=lambda *a: None)
