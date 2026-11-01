"""ทดสอบ API /chat ด้วย DummyLM แทน Gemini (ไม่เรียก API) — ข้ามเมื่อเครื่องไม่มี fastapi หรือ dspy"""

import json
import re

import pytest

dspy = pytest.importorskip("dspy")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from dspy.utils import DummyLM  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app, load_hotlines  # noqa: E402
from app.safety import load_templates  # noqa: E402

T = load_templates()


def g(*updates, intents=("update_incident",), target="active_incident", flags=()):
    return {"result": json.dumps({"intents": list(intents), "target": target, "updates": list(updates),
                                  "urgent_flags": list(flags)}, ensure_ascii=False)}


SIT = {"field": "situation", "status": "known", "value": {"text": "x", "tags": ["water_in_house"]}}


@pytest.fixture
def client_with(request, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("ENABLE_RESPONSE_COMPOSER", "0")
    def make(*answers):
        lm = DummyLM(list(answers))
        dspy.configure(lm=lm)
        c = TestClient(app)
        c.__enter__()  # เรียก startup
        request.addfinalizer(lambda: (c.__exit__(None, None, None), dspy.configure(lm=None)))
        return c, lm
    return make


def test_urgent_message_goes_through_flood_agent(client_with):
    c, lm = client_with(g(SIT, intents=["report_flood", "request_help"], target="new_incident",
                          flags=["people_trapped", "help_requested"]))
    r = c.post("/chat", json={"user_id": "u1", "input_text": "น้ำท่วมบ้าน ช่วยด้วย ยายติดอยู่ เบอร์ 081-234-5678"})
    assert r.status_code == 200
    body = r.json()
    assert body["incident_status"] == "submitted"
    assert body["next_actions"][:2] == ["submit_report", "ask_follow_up"]
    assert "1784" in body["output_text"] and "1669" in body["output_text"]  # สายด่วนจาก hotlines_th.json
    assert T["delivery_status"]["pending"] in body["output_text"]
    prompt = "\n".join(m["content"] for h in lm.history for m in h["messages"])
    phone = re.compile(r"0?81[-\s]?234[-\s]?5678")  # เลขหมายเต็ม (id สุ่มแบบ hex อาจมี "081" โดยบังเอิญ)
    assert not phone.search(prompt) and not phone.search(r.text)


def test_memory_is_per_user_and_reset(client_with):
    c, _ = client_with(
        g(SIT, intents=["report_flood"], target="new_incident"),
        g(intents=["general_conversation"], target="none"),
        g(intents=["general_conversation"], target="none"),
    )
    first = c.post("/chat", json={"user_id": "u1", "input_text": "น้ำท่วมบ้านค่ะ"}).json()
    assert first["incident_id"] and first["pending_question"]["field_names"] == ["location_area", "location_coordinates"]
    other = c.post("/chat", json={"user_id": "u2", "input_text": "สวัสดีค่ะ"}).json()
    assert other["incident_id"] is None and other["output_text"] == T["greeting"] + "\n\n" + T["privacy_notice"]  # ผู้ใช้อีกคนไม่เห็นเหตุของ u1 (R06)
    assert c.post("/chat/reset", json={"user_id": "u1"}).status_code == 200
    again = c.post("/chat", json={"user_id": "u1", "input_text": "สวัสดีค่ะ"}).json()
    assert again["incident_id"] is None


def test_user_id_required(client_with):
    c, _ = client_with()
    assert c.post("/chat", json={"input_text": "น้ำท่วม"}).status_code == 422


def test_hotline_config_loads():
    hotlines = load_hotlines()
    assert {h.number for h in hotlines} == {"1784", "1669"}


def test_location_and_postback_via_api(client_with):
    c, _ = client_with()  # ไม่ต้องใช้ Gemini เลย
    r = c.post("/chat", json={"user_id": "u9", "location": {"latitude": 14.95, "longitude": 100.36}})
    assert r.status_code == 200
    body = r.json()
    assert body["incident_id"] is None and body["quick_replies"] == []
    r = c.post("/chat", json={"user_id": "u9", "postback": "location_consent=yes"})
    assert r.status_code == 200 and r.json()["incident_id"] is None
    assert T["request_expired"] in r.json()["output_text"]


def test_exactly_one_kind_of_input(client_with):
    c, _ = client_with()
    both = {"user_id": "u9", "input_text": "x", "postback": "location_consent=yes"}
    assert c.post("/chat", json=both).status_code == 422
    assert c.post("/chat", json={"user_id": "u9"}).status_code == 422
