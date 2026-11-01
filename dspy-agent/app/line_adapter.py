"""
line_adapter.py — แปลงข้อมูลระหว่าง LINE Messaging API กับ FloodAgent (ใช้ร่วมกับ /line/webhook)

ไฟล์นี้ไม่เรียก network เลย มีแค่
- verify_signature  ตรวจว่า webhook มาจาก LINE จริง (header x-line-signature)
- parse_webhook     แปลง webhook body → รายการ LineInput (ข้อความ / ตำแหน่ง / กดปุ่ม)
- internal_user_id  แปลง LINE User ID เป็นรหัสภายใน (ห้ามเก็บหรือส่ง LINE User ID ตรง ๆ — R07)
- build_messages    แปลง AgentTurnOutput → messages สำหรับ Reply API พร้อมปุ่ม Quick Reply

main.py supplies the verified webhook and Reply API transport. This module remains pure.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from dataclasses import dataclass
from typing import Literal

from .schemas import AgentTurnOutput

TEXT_LIMIT = 5000  # LINE text message ยาวได้ไม่เกิน 5,000 ตัวอักษร
MAX_MESSAGES = 5  # Reply API ส่งได้ครั้งละไม่เกิน 5 messages
MAX_QUICK_REPLIES = 13


def verify_signature(channel_secret: str, body: bytes, signature: str | None) -> bool:
    """x-line-signature = base64(HMAC-SHA256(channel secret, request body ดิบ))"""
    if not channel_secret or not signature:
        return False
    digest = hmac.new(channel_secret.encode("utf-8"), body, hashlib.sha256).digest()
    return hmac.compare_digest(base64.b64encode(digest).decode("ascii"), signature)


def internal_user_id(line_user_id: str, salt: str) -> str:
    """รหัสภายในที่ย้อนกลับเป็น LINE User ID ไม่ได้ salt ต้องเก็บเป็นความลับใน .env"""
    if not salt:
        raise ValueError("ต้องตั้ง salt สำหรับแปลง LINE User ID")
    return "usr_" + hmac.new(salt.encode("utf-8"), line_user_id.encode("utf-8"), hashlib.sha256).hexdigest()[:20]


@dataclass(frozen=True)
class LineInput:
    user_id: str  # รหัสภายในแล้ว ไม่ใช่ LINE User ID
    kind: Literal["text", "location", "postback", "unsupported", "follow", "unfollow"]
    reply_token: str | None
    event_id: str | None  # webhookEventId ใช้กัน event ซ้ำ (R05)
    is_redelivery: bool = False
    text: str | None = None
    location: tuple[float, float] | None = None  # address ที่ LINE ส่งมาไม่เก็บ
    postback: str | None = None

    def agent_kwargs(self) -> dict:
        """ใช้เรียก agent(**line_input.agent_kwargs())"""
        if self.kind == "text":
            return {"input_text": self.text}
        if self.kind == "location":
            return {"location": self.location}
        if self.kind == "postback":
            return {"postback": self.postback}
        raise ValueError("ข้อความชนิดนี้ไม่ส่งเข้า agent")


def parse_webhook(body: dict, *, salt: str) -> list[LineInput]:
    """
    รับเฉพาะ event จากผู้ใช้แบบแชตส่วนตัว (source.type = user) และ mode = active
    ข้อความชนิดอื่น เช่น สติกเกอร์ รูปภาพ → kind = unsupported (ให้ตอบด้วย template)
    """
    out: list[LineInput] = []
    for ev in body.get("events", []):
        src = ev.get("source", {})
        if src.get("type") != "user" or not src.get("userId") or ev.get("mode", "active") != "active":
            continue
        common = dict(
            user_id=internal_user_id(src["userId"], salt),
            reply_token=ev.get("replyToken"),
            event_id=ev.get("webhookEventId"),
            is_redelivery=bool(ev.get("deliveryContext", {}).get("isRedelivery", False)),
        )
        if ev.get("type") == "postback":
            out.append(LineInput(kind="postback", postback=ev.get("postback", {}).get("data", ""), **common))
        elif ev.get("type") == "message":
            msg = ev.get("message", {})
            if msg.get("type") == "text":
                out.append(LineInput(kind="text", text=msg.get("text", ""), **common))
            elif msg.get("type") == "location":
                out.append(LineInput(kind="location", location=(float(msg["latitude"]), float(msg["longitude"])),
                                     **common))
            else:
                out.append(LineInput(kind="unsupported", **common))
        elif ev.get("type") in {"follow", "unfollow"}:
            out.append(LineInput(kind=ev["type"], **common))
    return out


def _split_text(text: str) -> list[str]:
    parts, rest = [], text
    while len(rest) > TEXT_LIMIT:
        cut = rest.rfind("\n\n", 0, TEXT_LIMIT)
        cut = cut if cut > 0 else TEXT_LIMIT
        parts.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    parts.append(rest)
    return [p for p in parts if p][:MAX_MESSAGES]


def build_messages(output: AgentTurnOutput) -> list[dict]:
    """messages สำหรับ POST /v2/bot/message/reply — Quick Reply ติดกับข้อความสุดท้าย"""
    messages = [{"type": "text", "text": t} for t in _split_text(output.output_text)]
    if output.quick_replies and messages:
        items = []
        for q in output.quick_replies[:MAX_QUICK_REPLIES]:
            if q.type == "location":
                action = {"type": "location", "label": q.label}
            else:
                action = {"type": "postback", "label": q.label, "data": q.data, "displayText": q.label}
            items.append({"type": "action", "action": action})
        messages[-1]["quickReply"] = {"items": items}
    return messages
