from fastapi.testclient import TestClient

from app.main import app


def test_dev_chat_page_served():
    res = TestClient(app).get("/dev/chat")  # ไม่เข้า lifespan จึงไม่ต้องมี Gemini key
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert 'fetch("/chat"' in res.text


def test_database_requires_admin(monkeypatch):
    monkeypatch.setenv("AGENT_ADMIN_TOKEN", "test-secret")
    response = TestClient(app).get("/dev/database?user_id=alice")
    assert response.status_code == 403


def test_database_reads_saved_rows_for_selected_user(monkeypatch, store):
    import threading

    monkeypatch.setenv("AGENT_ADMIN_TOKEN", "test-secret")
    alice = store.new_case("alice")
    alice.personal.first_name = "ผู้แจ้งทดสอบ"
    store.save_case(alice)
    bob = store.new_case("bob")
    bob.personal.first_name = "ไม่ควรเห็น"
    store.save_case(bob)
    monkeypatch.setattr(app.state, "store", store, raising=False)
    monkeypatch.setattr(app.state, "lock", threading.Lock(), raising=False)
    response = TestClient(app).get("/dev/database?user_id=alice", headers={"Authorization": "Bearer test-secret"})
    assert response.status_code == 200
    tables = response.json()["tables"]
    assert tables["case_personal_data"][0]["first_name"] == "ผู้แจ้งทดสอบ"
    assert len(tables["cases"]) == 1
    assert "ไม่ควรเห็น" not in response.text
    assert tables["case_versions"] == []
