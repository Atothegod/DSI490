"""
FastAPI ของ Agent 1 (Flood Intake & Support)

POST /chat        ส่งข้อความ / ตำแหน่ง / การกดปุ่ม ของผู้ใช้ 1 ครั้ง ได้คำตอบ + สถานะเหตุ (AgentTurnOutput)
POST /chat/reset  เริ่มบทสนทนาใหม่ของผู้ใช้คนนั้น (ใช้ตอนทดสอบ)
GET  /health

Runtime v1.3: PostgreSQL when DATABASE_URL is configured; SQLite for local runs.
LINE replies record delivery only after success. HTTP handoff is a draft transport,
enabled only with AGENT2_HANDOFF_URL/TOKEN. No LINE push is performed.
"""

import json
import logging
import os
import threading
import hmac
from datetime import datetime
from fastapi import Header, Request, Depends
from .repository import Repository
from .handoff import HandoffDispatcher, http_sender
from .knowledge import KnowledgeBase
from .areas import AreaResolver
from .area_summary import publish_summaries
from .schemas import OperationalStatusRecord, CasePriority, ContactAccessLog, now_bkk
from contextlib import asynccontextmanager
from pathlib import Path

import dspy
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from app.agent import FloodAgent
from app.schemas import AgentTurnOutput, Hotline

load_dotenv()
log = logging.getLogger("agent1")

APP_DIR = Path(__file__).resolve().parent
HOTLINES_PATH = APP_DIR / "hotlines_th.json"
DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"


class LocationPayload(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class ChatRequest(BaseModel):
    """ส่งอย่างใดอย่างหนึ่ง: input_text (ข้อความ), location (แชร์ตำแหน่ง) หรือ postback (กดปุ่ม Quick Reply)"""

    user_id: str = Field(min_length=1, description="รหัสผู้ใช้ภายใน 1 คน = 1 บทสนทนา (ห้ามใช้ LINE User ID ตรง ๆ ในระบบจริง)")
    input_text: str | None = Field(default=None, min_length=1)
    location: LocationPayload | None = None
    postback: str | None = Field(default=None, min_length=1, description="data ของปุ่ม เช่น location_consent=yes")

    @model_validator(mode="after")
    def _exactly_one(self) -> "ChatRequest":
        if sum(x is not None for x in (self.input_text, self.location, self.postback)) != 1:
            raise ValueError("ส่ง input_text, location หรือ postback อย่างใดอย่างหนึ่ง")
        return self

    def agent_kwargs(self) -> dict:
        if self.location is not None:
            return {"location": (self.location.latitude, self.location.longitude)}
        if self.postback is not None:
            return {"postback": self.postback}
        return {"input_text": self.input_text}


class ResetRequest(BaseModel):
    user_id: str = Field(min_length=1)


def configure_dspy() -> str:
    if dspy.settings.lm is not None:  # ตั้งไว้แล้ว เช่น ตอนทดสอบด้วย DummyLM
        return getattr(dspy.settings.lm, "model", "configured")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("Missing Gemini API key. Set GEMINI_API_KEY, GOOGLE_API_KEY, or API_KEY.")

    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    lm = dspy.LM(model=model, api_key=api_key)
    dspy.configure(lm=lm)
    return model


def load_hotlines(path: Path = HOTLINES_PATH) -> list[Hotline]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "verified":
        log.warning("hotlines_th.json ยังไม่ได้ตรวจกับแหล่งทางการ (status=%s)", data.get("status"))
    return [Hotline.model_validate(h) for h in data["hotlines"]]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = configure_dspy()
    app.state.hotlines = load_hotlines()
    app.state.repository = Repository(os.getenv("DATABASE_URL", "sqlite:///agent1.sqlite3"))
    app.state.sender = http_sender(os.environ["AGENT2_HANDOFF_URL"], os.environ["AGENT2_HANDOFF_TOKEN"]) if os.getenv("AGENT2_HANDOFF_URL") else None
    app.state.knowledge = KnowledgeBase(os.getenv("KNOWLEDGE_PATH", str(APP_DIR / "knowledge_th.csv")), threshold=float(os.getenv("RAG_THRESHOLD", "0.12")))
    app.state.areas = AreaResolver(os.getenv("AREAS_PATH", str(APP_DIR / "areas_singburi.csv")))
    app.state.agents = {}  # user_id → FloodAgent
    app.state.locks = {}  # user_id → Lock (ข้อความของผู้ใช้คนเดียวกันต้องทำทีละข้อความ)
    app.state.registry_lock = threading.Lock()
    stop = threading.Event()
    def background():
        while not stop.wait(30):
            try:
                for user_id, _ in app.state.repository.all_snapshots():
                    if user_id == "__area_summaries__":
                        continue
                    agent, lock = get_agent(user_id)
                    with lock:
                        ttl = os.getenv("CONTACT_PENDING_TTL_HOURS")
                        agent.maintain(contact_pending_ttl_hours=float(ttl) if ttl else None)
                publish_summaries(app.state.repository)
                HandoffDispatcher(app.state.repository, app.state.sender).dispatch()
            except Exception as exc:
                log.error("maintenance failed: %s", type(exc).__name__)
    worker = threading.Thread(target=background, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join(timeout=35)
        app.state.repository.close()


app = FastAPI(title="DSI490 Flood Agent 1 API", lifespan=lifespan)


def get_agent(user_id: str) -> tuple[FloodAgent, threading.Lock]:
    with app.state.registry_lock:
        if user_id not in app.state.agents:
            app.state.agents[user_id] = FloodAgent(hotlines=app.state.hotlines, internal_user_id=user_id,
                repository=app.state.repository, sender=app.state.sender, knowledge=app.state.knowledge, area_resolver=app.state.areas,
                composer=False if os.getenv("ENABLE_RESPONSE_COMPOSER", "1") == "0" else None)
            app.state.locks[user_id] = threading.Lock()
        return app.state.agents[user_id], app.state.locks[user_id]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": app.state.model}


@app.post("/chat", response_model=AgentTurnOutput)
def chat(payload: ChatRequest) -> AgentTurnOutput:
    agent, lock = get_agent(payload.user_id)
    with lock:
        try:
            return agent(**payload.agent_kwargs())
        except Exception as exc:
            # ไม่ใส่ข้อความผู้ใช้ใน log/response เพราะอาจมีข้อมูลส่วนบุคคล
            log.error("chat failed: %s", type(exc).__name__)
            raise HTTPException(status_code=502, detail=f"Agent failed: {type(exc).__name__}") from exc


@app.post("/chat/reset")
def reset(payload: ResetRequest) -> dict[str, str]:
    agent, lock = get_agent(payload.user_id)
    with lock:
        agent.reset()
    return {"status": "reset", "user_id": payload.user_id}


class AdminChange(BaseModel):
    user_id: str
    incident_id: str
    actor: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class OperationChange(AdminChange):
    status: str
    source: str
    recorded_at: datetime


class PriorityChange(AdminChange):
    priority: CasePriority


def require_admin(authorization: str | None = Header(default=None)):
    token = os.getenv("AGENT_ADMIN_TOKEN")
    if not token:
        raise HTTPException(status_code=503, detail="Admin access is not configured")
    if not authorization or not hmac.compare_digest(authorization, "Bearer " + token):
        raise HTTPException(status_code=403, detail="Forbidden")


@app.post("/incidents/operation", dependencies=[Depends(require_admin)])
def update_operation(payload: OperationChange):
    agent, lock = get_agent(payload.user_id)
    with lock:
        st = agent.incidents.get(payload.incident_id)
        if st is None:
            raise HTTPException(status_code=404, detail="Unknown incident")
        try:
            record = OperationalStatusRecord(status=payload.status, source=payload.source, recorded_by=payload.actor,
                                              reason=payload.reason, recorded_at=payload.recorded_at)
            active = agent.active_id
            agent.active_id = st.incident_id
            agent.record_operation(record)
            agent.active_id = active
            agent.persist()
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid or stale operation")
        finally:
            if "active" in locals():
                agent.active_id = active
                agent.persist()
        app.state.repository.audit({"action":"operation", "actor":payload.actor, "incident_id":st.incident_id,
                                    "reason":payload.reason, "at":now_bkk().isoformat()})
        return {"recorded":True}


@app.post("/incidents/priority", dependencies=[Depends(require_admin)])
def update_priority(payload: PriorityChange):
    agent, lock = get_agent(payload.user_id)
    with lock:
        if payload.incident_id not in agent.incidents:
            raise HTTPException(status_code=404, detail="Unknown incident")
        if payload.priority.adjusted_by != payload.actor or payload.priority.source == "agent1_rules":
            raise HTTPException(status_code=422, detail="Manual source and matching actor required")
        active = agent.active_id
        try:
            agent.active_id = payload.incident_id
            agent.override_priority(payload.priority)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid or stale priority")
        finally:
            agent.active_id = active
            agent.persist()
        app.state.repository.audit({"action":"priority", "actor":payload.actor, "incident_id":payload.incident_id,
                                    "reason":payload.reason, "at":now_bkk().isoformat()})
        return {"recorded":True}


@app.get("/contacts/{contact_ref}", dependencies=[Depends(require_admin)])
def read_contact(contact_ref: str, user_id: str, actor: str, reason: str):
    if not actor.strip() or not reason.strip():
        raise HTTPException(status_code=422, detail="Actor and reason required")
    agent, lock = get_agent(user_id)
    with lock:
        contact = next((c for c in agent.contacts if c.contact_ref == contact_ref and not c.deleted_at), None)
        if contact is None:
            raise HTTPException(status_code=404, detail="Unknown contact")
        audit = ContactAccessLog(contact_ref=contact_ref, accessed_by=actor, accessed_at=now_bkk(), reason=reason)
        app.state.repository.audit(audit.model_dump(mode="json"))
        return {"phone":contact.phone, "name":contact.name}


@app.post("/line/webhook")
async def line_webhook(request: Request, x_line_signature: str | None = Header(default=None)):
    from .line_adapter import verify_signature, parse_webhook, build_messages
    import httpx
    body = await request.body()
    secret, salt, token = (os.getenv(k) for k in ("LINE_CHANNEL_SECRET", "LINE_USER_HMAC_SECRET", "LINE_CHANNEL_ACCESS_TOKEN"))
    if not all((secret, salt, token)):
        raise HTTPException(status_code=503, detail="LINE is not configured")
    if not verify_signature(secret, body, x_line_signature):
        raise HTTPException(status_code=403, detail="Invalid signature")
    try:
        events = parse_webhook(json.loads(body), salt=salt)
    except (ValueError, KeyError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid webhook")
    # Synchronous model work runs outside the event loop.
    from starlette.concurrency import run_in_threadpool
    def handle():
        for event in events:
            if not event.event_id:
                continue
            agent, lock = get_agent(event.user_id)
            with lock:
                if event.kind in {"follow", "unfollow"}:
                    agent.user_active = event.kind == "follow"
                    agent.persist()
                    app.state.repository.record_receipt(event.event_id, event.user_id, {"delivered":True})
                    continue
                if not event.reply_token:
                    continue
                cached = app.state.repository.receipt(event.event_id, event.user_id)
                if cached and cached.get("delivered"):
                    continue
                if cached:
                    output = AgentTurnOutput.model_validate(cached["output"])
                elif event.kind == "unsupported":
                    output = AgentTurnOutput(output_text=agent.templates["unsupported_message"], intents=[],
                                             incident_id=None, incident_status=None, next_actions=["none"])
                else:
                    output = agent(**event.agent_kwargs(), defer_delivery=True)
                app.state.repository.record_receipt(event.event_id, event.user_id, {"output":output.model_dump(mode="json"), "delivered":False})
                try:
                    response = httpx.post("https://api.line.me/v2/bot/message/reply", headers={"Authorization":"Bearer " + token},
                                          json={"replyToken":event.reply_token, "messages":build_messages(output)}, timeout=10)
                    response.raise_for_status()
                except httpx.HTTPError:
                    raise HTTPException(status_code=502, detail="LINE reply failed")
                agent.confirm_delivery()
                app.state.repository.record_receipt(event.event_id, event.user_id, {"output":output.model_dump(mode="json"), "delivered":True})
    await run_in_threadpool(handle)
    return {"ok":True}
