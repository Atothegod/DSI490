"""
FastAPI ของ Agent 1 (Flood Incident Intake & Support)

POST /chat                  ส่งข้อความ / ตำแหน่ง / การกดปุ่ม ของผู้ใช้ 1 ครั้ง (ใช้ทดสอบโดยไม่ผ่าน LINE)
GET  /dev/chat              หน้าแชตจำลอง LINE เรียก /chat (ปุ่ม Quick Reply, ส่งตำแหน่ง, ดูสถานะเคส)
POST /line/webhook          LINE Messaging API ตอบผ่าน Reply API
POST /cases/{case_id}/close เจ้าหน้าที่หลังบ้านปิดเคส (ต้องใช้ admin token) — Agent 1 ไม่ปิดเคสเอง
GET  /health

Pipeline ไม่เรียก API นี้ แต่อ่าน Confirmed Version จาก View pipeline_confirmed_cases ใน PostgreSQL โดยตรง
Background job ส่งข้อความเตือน Draft 1 ครั้งผ่าน LINE Push เมื่อผู้ใช้เงียบครบ 30 นาที
"""

import hmac
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import dspy
import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from .agent import Agent1
from .knowledge import KnowledgeBase
from .response import ResponseGenerator
from .schemas import TurnOutput
from .store import Store

load_dotenv()
log = logging.getLogger("agent1")

APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
LINE_API = "https://api.line.me/v2/bot/message"


class LocationPayload(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class ChatRequest(BaseModel):
    """ส่งอย่างใดอย่างหนึ่ง: input_text, location หรือ postback (data ของปุ่ม เช่น consent=yes)"""

    user_id: str = Field(min_length=1, description="reporter_ref ห้ามใช้ LINE User ID ตรง ๆ")
    input_text: str | None = Field(default=None, min_length=1)
    location: LocationPayload | None = None
    postback: str | None = Field(default=None, min_length=1)

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
        return {"text": self.input_text}


class CloseRequest(BaseModel):
    actor: str = Field(min_length=1)
    reason: str = Field(min_length=1)


def configure_dspy() -> str:
    if dspy.settings.lm is not None:  # ตั้งไว้แล้ว เช่น ตอนทดสอบ
        return getattr(dspy.settings.lm, "model", "configured")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY.")
    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    dspy.configure(lm=dspy.LM(model=model, api_key=api_key))
    return model


def push_line(line_user_id: str, output: TurnOutput) -> None:
    from .line_adapter import build_messages
    token = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
    httpx.post(f"{LINE_API}/push", headers={"Authorization": "Bearer " + token},
               json={"to": line_user_id, "messages": build_messages(output)}, timeout=10).raise_for_status()


def send_reminders(agent: Agent1, store: Store) -> None:
    for user_ref, case_id, text in agent.due_reminders():
        line_user_id = store.line_id(user_ref)
        if line_user_id and os.getenv("LINE_CHANNEL_ACCESS_TOKEN"):
            push_line(line_user_id, TurnOutput(output_text=text, case_id=case_id))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = configure_dspy()
    app.state.store = Store(os.getenv("DATABASE_URL", "sqlite:///agent1.sqlite3"))
    knowledge = KnowledgeBase(os.getenv("KNOWLEDGE_PATH", str(APP_DIR / "knowledge_th.csv")),
                              threshold=float(os.getenv("RAG_THRESHOLD", "0.12")),
                              allow_unreviewed=os.getenv("KB_ALLOW_UNREVIEWED") == "1")
    if not knowledge.docs:
        log.warning("Knowledge Base ว่าง: ยังไม่มีเอกสารที่ผ่านผู้ตรวจ (ตั้ง KB_ALLOW_UNREVIEWED=1 เฉพาะตอนพัฒนา)")
    app.state.agent = Agent1(app.state.store, knowledge,
                             generator=ResponseGenerator(use_llm=os.getenv("ENABLE_RESPONSE_GENERATOR", "1") != "0"))
    app.state.lock = threading.Lock()  # ประมวลผลทีละข้อความ เพื่อไม่ให้ Case State ชนกัน
    stop = threading.Event()

    def background():
        while not stop.wait(60):
            try:
                with app.state.lock:
                    send_reminders(app.state.agent, app.state.store)
            except Exception as exc:
                log.error("reminder job failed: %s", type(exc).__name__)

    worker = threading.Thread(target=background, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join(timeout=65)
        app.state.store.close()


app = FastAPI(title="DSI490 Flood Agent 1 API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": app.state.model}


@app.post("/chat", response_model=TurnOutput)
def chat(payload: ChatRequest) -> TurnOutput:
    with app.state.lock:
        try:
            return app.state.agent.handle(payload.user_id, **payload.agent_kwargs())
        except Exception as exc:
            # ไม่ใส่ข้อความผู้ใช้ใน log เพราะอาจมีข้อมูลส่วนบุคคล
            log.error("chat failed: %s", type(exc).__name__)
            raise HTTPException(status_code=502, detail=f"Agent failed: {type(exc).__name__}") from exc


@app.get("/dev/chat", include_in_schema=False)
def dev_chat() -> FileResponse:
    """หน้าแชตจำลอง LINE สำหรับทดสอบ เรียก POST /chat เหมือนกัน"""
    return FileResponse(APP_DIR / "static" / "chat.html")


def require_admin(authorization: str | None = Header(default=None)):
    token = os.getenv("AGENT_ADMIN_TOKEN")
    if not token:
        raise HTTPException(status_code=503, detail="Admin access is not configured")
    if not authorization or not hmac.compare_digest(authorization, "Bearer " + token):
        raise HTTPException(status_code=403, detail="Forbidden")


@app.get("/dev/database", dependencies=[Depends(require_admin)], include_in_schema=False)
def dev_database(user_id: str):
    """อ่านแถวที่บันทึกจริง เฉพาะผู้ใช้ที่เลือก สำหรับผู้ดูแลหน้าทดสอบ"""
    with app.state.lock:
        store = app.state.store
        tables = {}
        with store.transaction() as c:
            queries = {
                "users": ("SELECT * FROM users WHERE user_ref=?", (user_id,)),
                "cases": ("SELECT * FROM cases WHERE user_ref=?", (user_id,)),
                "case_raw_messages": ("SELECT * FROM case_raw_messages WHERE user_ref=? ORDER BY created_at", (user_id,)),
            }
            for name in ("case_personal_data", "case_drafts", "case_slots", "case_versions",
                         "case_version_contacts", "case_events", "pipeline_confirmed_cases", "pipeline_case_contacts"):
                queries[name] = (f"SELECT * FROM {name} WHERE case_id IN (SELECT case_id FROM cases WHERE user_ref=?)", (user_id,))
            for name, (sql, args) in queries.items():
                cursor = store._exec(c, sql, args)
                columns = [column[0] for column in cursor.description]
                tables[name] = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return {"database": "PostgreSQL" if store.postgres else "SQLite", "user_id": user_id, "tables": tables}


@app.post("/cases/{case_id}/close", dependencies=[Depends(require_admin)])
def close_case(case_id: str, payload: CloseRequest):
    with app.state.lock:
        if not app.state.agent.close_case(case_id, actor=payload.actor, reason=payload.reason):
            raise HTTPException(status_code=404, detail="Unknown or already closed case")
    return {"closed": True}


@app.post("/line/webhook")
async def line_webhook(request: Request, x_line_signature: str | None = Header(default=None)):
    from starlette.concurrency import run_in_threadpool

    from .line_adapter import build_messages, parse_webhook, verify_signature
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

    def handle():
        store, agent = app.state.store, app.state.agent
        for event in events:
            if not event.event_id or event.kind in {"follow", "unfollow"} or not event.reply_token:
                continue
            with app.state.lock:
                cached = store.receipt(event.event_id)
                if cached and cached.get("delivered"):
                    continue
                store.set_line_id(event.user_id, event.line_user_id)
                if cached:
                    output = TurnOutput.model_validate(cached["output"])
                elif event.kind == "unsupported":
                    output = TurnOutput(output_text=agent.t["unsupported_message"])
                else:
                    output = agent.handle(event.user_id, **event.agent_kwargs())
                store.record_receipt(event.event_id, {"output": output.model_dump(mode="json"), "delivered": False})
            try:
                httpx.post(f"{LINE_API}/reply", headers={"Authorization": "Bearer " + token},
                           json={"replyToken": event.reply_token, "messages": build_messages(output)},
                           timeout=10).raise_for_status()
            except httpx.HTTPError:
                raise HTTPException(status_code=502, detail="LINE reply failed")
            store.record_receipt(event.event_id, {"output": output.model_dump(mode="json"), "delivered": True})

    await run_in_threadpool(handle)
    return {"ok": True}
