"""v1.3 behavioral regressions. No Gemini or external transport calls."""
import csv
import json
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError
from app.agent import FloodAgent
from app.areas import AreaResolver
from app.handoff import HandoffDispatcher, build_report, area_summaries
from app.area_summary import publish_summaries
from app.knowledge import KnowledgeBase
from app.repository import Repository
from app.response import Wording
from app.schemas import *

NOW = now_bkk()
HOTLINES = [Hotline(key="test", name="ทดสอบ", number="1111", source_url="https://example.org", verified_at=NOW)]


def ex(*updates, flags=(), intents=("report_flood",), target="active_incident", question=None):
    return ExtractionResult(intents=list(intents), target=target, updates=list(updates), urgent_flags=list(flags), question=question)


def u(field, value=None, status="known", **kw):
    return FieldUpdate(field=field, value=value, status=status, **kw)


def make(*answers, **kw):
    queue = list(answers)
    return FloodAgent(hotlines=HOTLINES, extractor=lambda *a: queue.pop(0), **kw)


URGENT = ex(u("situation", {"tags":["isolated"]}), flags=["people_trapped"])
GENERAL = ex(u("situation", {"tags":["water_in_house"]}))
CONTACT_READY = ex(u("location_area","ตำบลทับยา"),u("location_landmark","หลังวัด"),
                   u("situation",{"tags":["isolated"]}),u("affected_people",status="unknown"),
                   u("vulnerable_people",status="unknown"),flags=["people_trapped"])


def test_example_validates_and_has_no_future_values():
    e = TypeAdapter(HandoffEvent).validate_json(Path(__file__).parents[1].joinpath("agent2_input_example.json").read_text())
    assert e.report.priority.level == PriorityLevel.RED
    assert e.case_version == e.report.case_version == e.report.incident_version == 1
    for field in (e.report.situation,e.report.needs,e.report.affected_people,e.report.vulnerable_people,e.report.water_level):
        assert field.reported_at is None or field.reported_at <= e.occurred_at


def test_urgent_is_committed_before_transport_and_never_waits_for_wording():
    repo = Repository()
    sent = []
    def sender(payload):
        assert repo.outbox()[0].event_id == payload["event_id"]
        assert repo.load("usr_demo")["incidents"][payload["incident_id"]]["status"] == "submitted"
        sent.append(payload)
        return {"event_id":payload["event_id"],"received":True,"received_at":now_bkk().isoformat()}
    def compose(_):
        raise AssertionError("urgent reply must use fixed short wording")
    agent = make(URGENT, repository=repo, sender=sender, composer=compose)
    out = agent(input_text="ยายติดอยู่")
    assert len(sent) == 1 and out.delivery_status == DeliveryStatus.ACKED
    assert agent.templates["delivery_status"]["acked"] in out.output_text
    assert len(out.pending_question.field_names) == 1
    assert "รับทราบข้อมูล" not in out.output_text


def test_retry_keeps_event_id_version_and_checks_ack():
    repo = Repository()
    agent = make(URGENT, repository=repo)
    agent(input_text="ยายติดอยู่")
    before = repo.outbox()[0]
    dispatcher = HandoffDispatcher(repo, lambda p: {"event_id":"wrong", "received":True, "received_at":now_bkk().isoformat()})
    dispatcher.dispatch()
    assert repo.outbox()[0].delivery_status == DeliveryStatus.FAILED
    dispatcher.sender = lambda p: {"event_id":p["event_id"],"received":True,"received_at":now_bkk().isoformat()}
    dispatcher.dispatch()
    after = repo.outbox()[0]
    assert after.attempts == 2 and after.event_id == before.event_id and after.payload == before.payload
    dispatcher.dispatch()
    assert repo.outbox()[0].attempts == 2


def test_contact_requires_separate_consent_and_no_raw_logs():
    agent = make(CONTACT_READY)
    out = agent(input_text="ยายติดอยู่ โทร 081-234-5678 เลขบัตร 1234567890123 บ้านเลขที่ 99/7")
    assert agent.contacts == [] and agent.state.contact_ref is None
    assert not agent.repository.outbox()[0].payload["report"]["contact_available"]
    assert "081" not in agent.turns[0].user_text
    assert "1234567890123" not in agent.turns[0].user_text and "99/7" not in agent.turns[0].user_text
    button = next(q for q in out.quick_replies if q.data and q.data.startswith("contact_consent=yes"))
    agent(postback=button.data)
    assert agent.contacts[0].phone == "0812345678"
    assert agent.state.contact_ref == agent.contacts[0].contact_ref
    assert agent.contact_requests[0].phone == ""
    assert agent.repository.outbox()[-1].payload["report"]["contact_available"] is True


def test_contact_denial_erases_pending_phone_without_deleting_incident():
    agent = make(CONTACT_READY)
    out = agent(input_text="ยายติดอยู่ 0812345678")
    button = next(q for q in out.quick_replies if q.data and q.data.startswith("contact_consent=no"))
    incident = agent.state.incident_id
    agent(postback=button.data)
    assert "0812345678" not in json.dumps(agent._snapshot(), ensure_ascii=False)
    assert agent.state.incident_id == incident and agent.state.contact_ask.status == FieldStatus.DECLINED


def test_location_alone_creates_no_incident_then_requires_permission():
    agent = make(GENERAL)
    out = agent(location=(14.95,100.36))
    assert out.incident_id is None and agent.incidents == {} and not agent.consents
    out = agent(input_text="น้ำท่วมบ้านค่ะ")
    assert agent.state and agent.state.location_coordinates.value
    assert not agent.state.location_consent_id
    assert "location_consent=yes" in [q.data for q in out.quick_replies]


def test_general_report_requests_optional_gps_and_privacy_once():
    agent = make(GENERAL, ex(intents=["general_conversation"]))
    first = agent(input_text="น้ำท่วมบ้าน")
    assert first.pending_question.field_names == [IncidentFieldName.LOCATION_AREA, IncidentFieldName.LOCATION_COORDINATES]
    assert first.output_text.count(agent.templates["privacy_notice"]) == 1
    second = agent(input_text="ยังไม่รู้ค่ะ")
    assert agent.templates["privacy_notice"] not in second.output_text
    assert agent.state.location_coordinates.ask_count == 1


def test_failed_line_delivery_does_not_count_asks_or_record_notice():
    agent = make(URGENT)
    out = agent(input_text="ยายติดอยู่", defer_delivery=True)
    assert not agent.state.hotline_notices and not agent.seen_notice_versions
    assert agent.state.location_area.ask_count == 0
    assert agent.repository.outbox()[0].payload["report"]["priority"]["level"] == "red"
    restored = make(repository=agent.repository)
    restored.confirm_delivery()
    assert restored.state.hotline_notices and restored.state.location_area.ask_count == 1
    assert restored.seen_notice_versions == [agent.templates["notice_versions"]["privacy"]]
    restored.confirm_delivery()
    assert restored.state.location_area.ask_count == 1


def test_persistent_snapshot_and_reporter_ref_survive_restart(tmp_path):
    repo = Repository("sqlite:///" + str(tmp_path / "state.db"))
    agent = make(URGENT, repository=repo)
    agent(input_text="ยายติดอยู่")
    restored = make(repository=repo)
    assert restored.state == agent.state and restored.reporter_ref == agent.reporter_ref
    assert restored.last_reports == agent.last_reports


def test_information_changes_send_versioned_full_update_but_text_only_does_not():
    agent = make(URGENT, ex(u("affected_people", {"present":True,"count":1})),
                 ex(u("situation", {"tags":["isolated"]})))
    agent(input_text="ยายติดอยู่")
    agent(input_text="มีคนเดียว")
    rows = agent.repository.outbox()
    update = TypeAdapter(HandoffEvent).validate_python(rows[-1].payload)
    assert update.event_type == EventType.INCIDENT_UPDATED and update.case_version == 2
    assert update.report.case_version == update.report.incident_version == 2
    assert update.changes[0].field == IncidentFieldName.AFFECTED_PEOPLE
    agent(input_text="ยังติดอยู่เช่นเดิม")
    assert len(agent.repository.outbox()) == len(rows)


def test_deny_location_overrides_needs_and_differs_from_delete():
    agent = make(ex(u("situation",{"tags":["isolated"]}),u("needs",["rescue"]),flags=["people_trapped"]))
    agent(input_text="ยายติดอยู่ ขอเรือ")
    agent(location=(14.95,100.36))  # unsolicited because first question was area
    agent(postback="location_consent=no")
    report = build_report(agent.state,agent.reporter_ref,agent.consents,agent.contacts)
    assert report.location.coordinates is None
    assert agent.deletion_requested_at is None and agent.incidents


def test_delete_erases_history_outbox_contact_pending_and_stops_forwarding():
    agent = make(URGENT)
    agent(input_text="ยายติดอยู่ 0812345678")
    agent(location=(14.95123,100.36456))
    agent(postback="location_consent=yes")
    agent(input_text="ลบข้อมูลของฉัน")
    snapshot = json.dumps(agent._snapshot(), ensure_ascii=False)
    events = json.dumps([r.payload for r in agent.repository.outbox()], ensure_ascii=False)
    for value in ("0812345678","14.95123","100.36456"):
        assert value not in snapshot and value not in events
    assert agent.deletion_requested_at and agent.consents[0].revoked_at
    assert agent.repository.outbox()[-1].event_type == EventType.DATA_DELETION_REQUESTED


def test_priority_never_auto_downgrades_and_respects_manual_override():
    from app.priority import evaluate_priority
    st = IncidentState(internal_user_id="u",conversation_id="c",needs=TrackedField[list[Need]](status="known",value=[Need.RESCUE]))
    st = evaluate_priority(st)
    assert st.priority.level == PriorityLevel.RED
    st.needs = TrackedField[list[Need]](status="known",value=[Need.INFO_ONLY])
    assert evaluate_priority(st).priority.level == PriorityLevel.RED
    lowered = evaluate_priority(st,correction=True)
    assert lowered.priority.level == PriorityLevel.WHITE and lowered.priority_history
    lowered.priority = CasePriority(level="yellow",reasons=["manual"],source="official",adjusted_by="officer",evaluated_at=NOW)
    assert evaluate_priority(lowered).priority.source == "official"


@pytest.mark.parametrize("tag",["electrical_hazard","structural_risk"])
def test_new_hazard_tags_raise_yellow(tag):
    from app.priority import evaluate_priority
    st = IncidentState(internal_user_id="u",conversation_id="c",situation=TrackedField[Situation](status="known",value=Situation(text="แจ้งเหตุ",tags=[tag])))
    assert evaluate_priority(st).priority.level == PriorityLevel.YELLOW


def kb_file(tmp_path):
    path = tmp_path/"kb.csv"
    fields = "chunk_id source_org title url verified_at valid_until category review_status reviewed_text reviewed_by reviewed_at example_questions".split()
    base = dict(chunk_id="a",source_org="ปภ.",title="น้ำท่วม",url="https://example.org/a",verified_at=NOW.isoformat(),valid_until="",category="A",review_status="approved",reviewed_text="ติดตามคำแนะนำจากหน่วยงานค่ะ",reviewed_by="reviewer",reviewed_at=NOW.isoformat(),example_questions="น้ำท่วมทำอย่างไร")
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        writer.writerows([base,{**base,"chunk_id":"pending","review_status":"pending_review"},
                         {**base,"chunk_id":"expired","category":"B","valid_until":(NOW-timedelta(days=1)).isoformat()}])
    return path


def test_kb_only_reviewed_valid_rows_are_used(tmp_path):
    kb=KnowledgeBase(kb_file(tmp_path))
    assert [e.chunk_id for e in kb.search("น้ำท่วมทำอย่างไร")] == ["a"]


def test_knowledge_without_incident_uses_cited_reviewed_evidence(tmp_path):
    kb=KnowledgeBase(kb_file(tmp_path))
    def compose(plan):
        e=plan["evidence"][0]
        return Wording(knowledge_answer=e["reviewed_text"],source_ids=[e["chunk_id"]])
    agent=make(ex(intents=["ask_information"],target="none",question="น้ำท่วมทำอย่างไร"),knowledge=kb,composer=compose)
    out=agent(input_text="น้ำท่วมทำอย่างไร")
    assert out.incident_id is None and out.sources == ["a"] and "https://example.org/a" in out.output_text


def test_invalid_wording_or_exception_falls_back_once():
    calls=[]
    def compose(plan):
        calls.append(plan)
        return Wording(questions={"needs":"เจ้าหน้าที่กำลังไปค่ะ"})
    agent=make(GENERAL,composer=compose)
    out=agent(input_text="น้ำท่วมบ้าน")
    assert len(calls)==1 and "เจ้าหน้าที่กำลังไป" not in out.output_text
    assert agent.templates["questions"]["location_coordinates"][0] in out.output_text


def test_area_summary_moves_case_without_merging_or_double_counting():
    agent=make(URGENT)
    agent(input_text="ยายติดอยู่")
    summaries=area_summaries(agent.last_reports.values())
    assert summaries[0].area_level == "unassigned" and len(summaries[0].case_versions)==1
    events=publish_summaries(agent.repository)
    assert len(events)==1
    assert publish_summaries(agent.repository)==[]
    agent.close_incident(CloseReason.USER_RESOLVED)
    events=publish_summaries(agent.repository)
    assert events[0].summary.case_versions=={}


def test_retention_clears_coordinates_from_all_event_copies():
    agent=make(URGENT)
    agent(input_text="ยายติดอยู่")
    agent(location=(14.95123,100.36456));agent(postback="location_consent=yes")
    agent.close_incident(CloseReason.USER_RESOLVED)
    agent.maintain(now=now_bkk()+timedelta(days=31))
    saved=json.dumps(agent._snapshot(),ensure_ascii=False)
    events=json.dumps([r.payload for r in agent.repository.outbox()],ensure_ascii=False)
    assert "14.95123" not in saved+events


def test_draft_abandon_and_submitted_timeout_are_distinct():
    general=make(GENERAL);general(input_text="น้ำท่วมบ้าน")
    general.maintain(now=now_bkk()+timedelta(days=8))
    assert general.state.status==IncidentStatus.ABANDONED
    urgent=make(URGENT);urgent(input_text="ยายติดอยู่")
    urgent.maintain(now=now_bkk()+timedelta(days=15))
    assert urgent.state.status==IncidentStatus.CLOSED and urgent.state.close_reason==CloseReason.TIMEOUT
    assert urgent.repository.outbox()[-1].event_type==EventType.INCIDENT_CLOSED


def test_reporter_operation_cannot_claim_official_completion():
    with pytest.raises(ValidationError):
        OperationalStatusRecord(status="completed",source="reporter",recorded_by="u",reason="บอกว่าออกแล้ว",recorded_at=NOW)
    assert OperationalStatusRecord(status="reporter_resolved",source="reporter",recorded_by="u",reason="ออกแล้ว",recorded_at=NOW)


def test_confirmation_expires_after_24_hours():
    from app.conversation import PendingConfirmation,plan_turn,summary_fingerprint
    st=IncidentState(internal_user_id="u",conversation_id="c")
    pc=PendingConfirmation(incident_id=st.incident_id,summary_fingerprint=summary_fingerprint(st),asked_at=NOW-timedelta(hours=25),asked_message_id="reply")
    plan=plan_turn(st,extraction=ex(intents=["confirm_report"]),merge=None,pending_confirmation=pc)
    assert plan.submit_trigger is None


def test_verified_area_matching_is_exact_and_uses_parent_district(tmp_path):
    path=tmp_path/"areas.csv"
    keys="tambon_code tambon amphoe_code amphoe province_code province source_url verified_at review_status".split()
    rows=[dict(zip(keys,values)) for values in [
        ["170101","ชื่อซ้ำ","1701","อำเภอหนึ่ง","17","สิงห์บุรี","https://example.org",NOW.isoformat(),"approved"],
        ["170201","ชื่อซ้ำ","1702","อำเภอสอง","17","สิงห์บุรี","https://example.org",NOW.isoformat(),"approved"],
        ["170202","ยังไม่ตรวจ","1702","อำเภอสอง","17","สิงห์บุรี","https://example.org",NOW.isoformat(),"pending_review"]]]
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    resolve=AreaResolver(path)
    ambiguous=resolve(AreaMention(name="ตำบลชื่อซ้ำ",level="tambon"))
    assert ambiguous.match_status==AreaMatchStatus.AMBIGUOUS and len(ambiguous.candidates)==2
    assert resolve(AreaMention(name="ตำบลชื่อซ้ำ อำเภออำเภอสอง",level="tambon")).tambon_code=="170201"
    assert resolve(AreaMention(name="ตำบลชื่อช้ำ",level="tambon")).match_status==AreaMatchStatus.NOT_FOUND
    assert resolve(AreaMention(name="ตำบลยังไม่ตรวจ",level="tambon")).match_status==AreaMatchStatus.NOT_FOUND


def test_explicit_outside_province_no_case_unless_urgent():
    area=u("location_area",{"name":"จังหวัดเชียงใหม่","level":"province"})
    general=make(ex(area,u("situation",{"tags":["area_flooded"]})),area_resolver=AreaResolver())
    out=general(input_text="น้ำท่วม จังหวัดเชียงใหม่")
    assert out.incident_id is None and not general.incidents
    urgent=make(ex(area,flags=["people_trapped"]),area_resolver=AreaResolver())
    urgent(input_text="ยายติดอยู่ จังหวัดเชียงใหม่")
    assert urgent.state.status==IncidentStatus.SUBMITTED and urgent.state.service_scope==ServiceScope.OUT_OF_PROVINCE


def test_matching_question_does_not_merge_until_explicit_choice():
    agent=make(ex(u("situation",{"tags":["water_in_house"]}),u("location_area","ตำบลเดิม")),ex(u("location_area","ตำบลอีกแห่ง"),target="new_incident"))
    agent(input_text="น้ำท่วมบ้าน")
    first=agent.state.incident_id
    out=agent(input_text="อีกแห่งที่ตำบลอีกแห่ง")
    assert out.next_actions==[NextAction.ASK_INCIDENT_MATCH] and len(agent.incidents)==1
    agent(postback="incident_match=new")
    assert len(agent.incidents)==2 and agent.state.incident_id!=first
    assert agent.incidents[first].location_area.value.raw == "ตำบลเดิม"


def test_provisional_link_event_keeps_both_cases():
    agent=make(GENERAL,ex(u("location_area","ตำบลอีกแห่ง"),target="unclear",flags=["people_trapped"]))
    agent(input_text="น้ำท่วมบ้าน")
    first=agent.state.incident_id
    out=agent(input_text="ยายติดอยู่ อีกตำบล")
    second=agent.state.incident_id
    button=next(q for q in out.quick_replies if q.data and q.data.startswith("incident_link=same_incident:"))
    agent(postback=button.data)
    assert set(agent.incidents)=={first,second}
    assert agent.state.matching_status==MatchingStatus.LINKED
    assert any(r.event_type==EventType.INCIDENT_LINKED for r in agent.repository.outbox())


@pytest.fixture
def api(monkeypatch):
    import dspy
    from dspy.utils import DummyLM
    from fastapi.testclient import TestClient
    from app.main import app
    monkeypatch.setenv("DATABASE_URL","sqlite:///:memory:")
    monkeypatch.setenv("ENABLE_RESPONSE_COMPOSER","0")
    monkeypatch.setenv("AGENT_ADMIN_TOKEN","test-admin")
    dspy.configure(lm=DummyLM([]))
    with TestClient(app) as client:
        yield client
    dspy.configure(lm=None)


def test_contact_and_operations_api_require_authority_and_audit(api):
    from app.main import get_agent
    agent,_=get_agent("u")
    agent.extract=lambda *a:URGENT
    api.post("/chat",json={"user_id":"u","input_text":"ยายติดอยู่"})
    payload={"user_id":"u","incident_id":agent.state.incident_id,"actor":"operator","reason":"รับเรื่องแล้ว",
             "status":"received","source":"official","recorded_at":now_bkk().isoformat()}
    assert api.post("/incidents/operation",json=payload).status_code==403
    headers={"Authorization":"Bearer test-admin"}
    assert api.post("/incidents/operation",json=payload,headers=headers).status_code==200
    assert agent.state.operational_status[-1].status=="received"
    payload.update(status="completed",source="reporter",recorded_at=now_bkk().isoformat())
    assert api.post("/incidents/operation",json=payload,headers=headers).status_code==422
    assert api.get("/contacts/unknown",params={"user_id":"u","actor":"operator","reason":"ประสานงาน"}).status_code==403
    assert agent.repository.db.execute("SELECT COUNT(*) FROM audit_logs").fetchone()[0]==1


def test_line_deduplication_and_failed_reply_recovery(api,monkeypatch):
    import base64,hashlib,hmac,httpx
    from app.main import get_agent
    from app.line_adapter import internal_user_id
    for key,value in (("LINE_CHANNEL_SECRET","secret"),("LINE_USER_HMAC_SECRET","salt"),("LINE_CHANNEL_ACCESS_TOKEN","token")):
        monkeypatch.setenv(key,value)
    agent,_=get_agent(internal_user_id("line-user","salt"))
    calls=[]
    def extract(*a):
        calls.append(a)
        return URGENT
    agent.extract=extract
    sends=[]
    def send(*args,**kwargs):
        sends.append(kwargs)
        if len(sends)==1:
            raise httpx.ConnectError("offline")
        return httpx.Response(200,request=httpx.Request("POST",args[0]))
    monkeypatch.setattr(httpx,"post",send)
    body=json.dumps({"events":[{"type":"message","webhookEventId":"line-event-1","replyToken":"reply-token",
                                "source":{"type":"user","userId":"line-user"},"message":{"type":"text","text":"ยายติดอยู่"}}]}).encode()
    signature=base64.b64encode(hmac.new(b"secret",body,hashlib.sha256).digest()).decode()
    def request():
        return api.post("/line/webhook",content=body,headers={"x-line-signature":signature})
    assert request().status_code==502
    assert len(calls)==1 and not agent.state.hotline_notices
    assert request().status_code==200
    assert len(calls)==1 and len(sends)==2 and len(agent.state.hotline_notices)==1
    assert request().status_code==200
    assert len(calls)==1 and len(sends)==2
    assert len(agent.repository.outbox(agent.internal_user_id))==1


def test_hmac_identifier_uses_hmac_sha256():
    import hashlib,hmac
    from app.line_adapter import internal_user_id
    assert internal_user_id("line-user","secret")=="usr_"+hmac.new(b"secret",b"line-user",hashlib.sha256).hexdigest()[:20]


def test_event_rejects_values_reported_after_event_time():
    data=json.loads(Path(__file__).parents[1].joinpath("agent2_input_example.json").read_text())
    data["report"]["affected_people"]={"status":"known","value":{"present":True,"count":1},"reported_at":"2026-11-01T18:42:00+07:00"}
    with pytest.raises(ValidationError,match="after occurred_at"):
        IncidentSubmittedEvent.model_validate(data)


def test_urgent_contact_confirmation_waits_its_turn():
    agent=make(URGENT)
    out=agent(input_text="ยายติดอยู่ โทร 0812345678")
    assert agent.contact_requests and not agent.contact_requests[0].request_message_id
    assert not any(q.data and q.data.startswith("contact_consent=") for q in out.quick_replies)
    assert out.pending_question.field_names==[IncidentFieldName.LOCATION_AREA]


def test_composer_dspy_valid_output_and_invalid_format_no_retry():
    import dspy
    from dspy.utils import DummyLM
    from app.response import ResponseComposer
    plan={"question_hints":{"needs":["ต้องการให้ช่วยเรื่องใดค่ะ"]},"acknowledgement_hints":[],"evidence":[]}
    lm=DummyLM([{"result":json.dumps({"questions":{"needs":"ตอนนี้อยากให้ช่วยเรื่องไหนก่อนค่ะ"}},ensure_ascii=False)}])
    dspy.configure(lm=lm)
    try:
        assert ResponseComposer()(plan).questions=={"needs":"ตอนนี้อยากให้ช่วยเรื่องไหนก่อนค่ะ"}
        assert len(lm.history)==1
        bad=DummyLM([{"wrong_field":"invalid format"}])
        dspy.configure(lm=bad)
        assert ResponseComposer()(plan) is None
        assert len(bad.history)==1
    finally:
        dspy.configure(lm=None)


def test_pending_deletion_ack_is_recorded_separately():
    agent=make(URGENT)
    agent(input_text="ยายติดอยู่")
    agent.delete_personal_data()
    row=agent.repository.db.execute("SELECT payload FROM deletion_requests").fetchone()
    assert json.loads(row[0])["agent2_acked_at"] is None
    HandoffDispatcher(agent.repository,lambda p:{"event_id":p["event_id"],"received":True,"received_at":now_bkk().isoformat()}).dispatch()
    row=agent.repository.db.execute("SELECT payload FROM deletion_requests").fetchone()
    assert json.loads(row[0])["agent2_acked_at"] is not None


def test_wording_can_rephrase_questions_and_acknowledgement():
    from app.response import validate_wording
    plan={"question_hints":{"needs":["ตอนนี้ต้องการความช่วยเหลือเรื่องใดค่ะ"]},
          "confirmed_state":{},"evidence":[]}
    wording=Wording(acknowledgement="ขอบคุณที่เล่าสถานการณ์ให้ทราบค่ะ",
                    questions={"needs":"ตอนนี้อยากให้ช่วยเรื่องไหนเป็นอันดับแรกค่ะ"})
    assert wording.questions["needs"] not in plan["question_hints"]["needs"]
    assert validate_wording(wording,plan)


def test_wording_can_paraphrase_multiple_cited_rows():
    from app.response import validate_wording
    plan={"question_hints":{},"confirmed_state":{},"evidence":[
        {"chunk_id":"a","reviewed_text":"เตรียมน้ำดื่มและอาหารแห้งไว้ในที่สูง"},
        {"chunk_id":"b","reviewed_text":"จัดยาและเอกสารสำคัญไว้ในถุงกันน้ำ"}]}
    wording=Wording(knowledge_answer="เก็บอาหารแห้งกับน้ำดื่มขึ้นที่สูง และใส่ยากับเอกสารสำคัญในถุงกันน้ำค่ะ",source_ids=["a","b"])
    assert validate_wording(wording,plan)
    wording.source_ids=["not_retrieved"]
    assert not validate_wording(wording,plan)


@pytest.mark.parametrize("question",[
    "ตอนนี้ต้องการความช่วยเหลือเรื่องไหน และน้ำสูงเท่าไหร่ค่ะ",
    "ตอนนี้ต้องการความช่วยเหลือเรื่องไหน มีผู้ได้รับผลกระทบกี่คนค่ะ",
    "ตอนนี้ต้องการให้ช่วยที่ตำบลสมมติค่ะ",
    "ตอนนี้ต้องการให้ช่วยอพยพ 99 คนค่ะ",
])
def test_rephrasing_cannot_add_questions_places_or_numbers(question):
    from app.response import validate_wording
    plan={"question_hints":{"needs":["ต้องการให้ช่วยเรื่องใดค่ะ"]},"confirmed_state":{},"evidence":[]}
    assert not validate_wording(Wording(questions={"needs":question}),plan)


def test_paraphrase_rejects_known_questions_and_uncited_numeric_facts():
    from app.response import validate_wording
    plan={"question_hints":{"water_level":[]},"confirmed_state":{"water_level":{"status":"known","value":{"cm":80}}},"evidence":[]}
    assert not validate_wording(Wording(questions={"water_level":"น้ำสูงแค่ไหนค่ะ"}),plan)
    plan={"question_hints":{},"confirmed_state":{},"evidence":[{"chunk_id":"a","reviewed_text":"เตรียมน้ำดื่มให้เพียงพอ"}]}
    assert not validate_wording(Wording(knowledge_answer="ควรเตรียมน้ำดื่ม 99 ลิตรค่ะ",source_ids=["a"]),plan)


def test_runtime_uses_new_wording_and_all_citations(tmp_path):
    kb=KnowledgeBase(kb_file(tmp_path))
    def compose(plan):
        assert "question_hints" in plan and "changes" in plan
        return Wording(knowledge_answer="หากเกิดน้ำท่วม ให้ติดตามสิ่งที่หน่วยงานแนะนำค่ะ",source_ids=["a"])
    agent=make(ex(intents=["ask_information"],target="none",question="น้ำท่วมทำอย่างไร"),knowledge=kb,composer=compose)
    out=agent(input_text="น้ำท่วมทำอย่างไร")
    assert "หากเกิดน้ำท่วม" in out.output_text and out.sources==["a"]
