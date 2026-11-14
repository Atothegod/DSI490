"""
line_adapter.py — แปลงข้อมูลระหว่าง LINE Messaging API กับ FloodAgent (ใช้ร่วมกับ /line/webhook)

ไฟล์นี้ไม่เรียก network เลย มีแค่
- verify_signature  ตรวจว่า webhook มาจาก LINE จริง (header x-line-signature)
- parse_webhook     แปลง webhook body → รายการ LineInput (ข้อความ / ตำแหน่ง / กดปุ่ม)
- internal_user_id  แปลง LINE User ID เป็น reporter_ref (LINE User ID จริงเก็บแยกใน user_line_ids เพื่อส่งข้อความเตือนเท่านั้น)
- build_messages    แปลง TurnOutput → messages สำหรับ Reply/Push API พร้อมปุ่ม Quick Reply

main.py supplies the verified webhook and Reply API transport. This module remains pure.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Literal

from .schemas import TurnOutput

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
    user_id: str  # reporter_ref ไม่ใช่ LINE User ID
    line_user_id: str
    kind: Literal["text", "location", "postback", "unsupported", "follow", "unfollow"]
    reply_token: str | None
    event_id: str | None  # webhookEventId ใช้กัน event ซ้ำ (R05)
    text: str | None = None
    location: tuple[float, float] | None = None  # address ที่ LINE ส่งมาไม่เก็บ
    postback: str | None = None

    def agent_kwargs(self) -> dict:
        """ใช้เรียก agent.handle(user_id, **line_input.agent_kwargs())"""
        if self.kind == "text":
            return {"text": self.text}
        if self.kind == "location":
            return {"location": self.location}
        if self.kind == "postback":
            return {"postback": self.postback}
        if self.kind == "follow":
            return {"text": "สวัสดี"}
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
            line_user_id=src["userId"],
            reply_token=ev.get("replyToken"),
            event_id=ev.get("webhookEventId"),
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


def _button_label(reply) -> str:
    """Decorate LINE's native buttons; keep the original postback/display text."""
    data = reply.data or ""
    if reply.type == "location":
        icon = "⌖"
    elif data.startswith("edit="):
        icon = "✎"
    elif data == "consent=yes" or data.startswith(("confirm=", "people_confirm=yes:", "reuse_contact=yes:", "intake=")):
        icon = "✓"
    elif data == "new_case":
        icon = "＋"
    else:
        return reply.label
    label = f"{icon} {reply.label}"
    return label if len(label) <= 20 else reply.label


def _postback_action(reply) -> dict:
    return {"type": "postback", "label": _button_label(reply),
            "data": reply.data, "displayText": reply.label}


def build_flex_message(output: TurnOutput) -> dict | None:
    """Render a short decision as a LINE card; long advice keeps its full text.

    Location remains a native Quick Reply: LINE does not allow location actions
    in Flex buttons. The card never reconstructs case data from a second source.
    """
    replies = output.quick_replies
    if not output.output_text or not replies or len(replies) > 4 or len(output.output_text) > 1800:
        return None
    actions = [q for q in replies if q.type == "postback"]
    data = [q.data or "" for q in replies]
    if "consent=yes" in data:
        title, caption = "ก่อนเริ่มคุยกัน", "คุณเลือกได้ว่าจะให้ใช้ข้อมูลหรือไม่"
    elif any(d.startswith("confirm=") for d in data):
        title, caption = "ตรวจสอบข้อมูลก่อนส่งต่อ", "ตรวจให้ถูกต้อง แล้วกดยืนยันด้านล่าง"
    elif output.pending_question == "affected_people_confirmation":
        title, caption = "ยืนยันจำนวนคน", "ช่วยตรวจจำนวนและกลุ่มผู้ประสบภัย"
    elif output.pending_question == "reuse_contact":
        title, caption = "ใช้ข้อมูลผู้แจ้งเดิมไหม?", "เลือกใช้ข้อมูลเดิม หรือแจ้งข้อมูลใหม่"
    elif any(q.type == "location" for q in replies):
        title, caption = "แชร์จุดที่ต้องการความช่วยเหลือ", "กดปุ่มแชร์ตำแหน่งใต้ข้อความนี้"
    elif output.intake_status and output.intake_status.value == "confirmed":
        title, caption = "บันทึกข้อมูลแล้ว", "แจ้งเพิ่มเติมในแชตนี้ได้เสมอ"
    else:
        title, caption = "ค่อย ๆ ไปทีละขั้น", "เลือกตัวเลือกด้านล่าง หรือพิมพ์ตอบได้เลย"
    contents = {
        "type": "bubble", "size": "mega",
        "header": {"type": "box", "layout": "vertical", "backgroundColor": "#EDF7F2",
                   "paddingAll": "20px", "spacing": "sm", "contents": [
                       {"type": "text", "text": "FLOOD CARE", "size": "xs", "weight": "bold", "color": "#137C50"},
                       {"type": "text", "text": title, "size": "lg", "weight": "bold", "wrap": True, "color": "#19352B"},
                       {"type": "text", "text": caption, "size": "xs", "wrap": True, "color": "#597466"}]},
        "body": {"type": "box", "layout": "vertical", "paddingAll": "20px", "contents": [
            {"type": "text", "text": output.output_text, "size": "sm", "wrap": True, "color": "#294537"}]},
    }
    if actions:
        buttons = []
        for q in actions:
            primary = (q.data == "consent=yes" or (q.data or "").startswith(
                ("confirm=", "people_confirm=yes:", "reuse_contact=yes:", "intake=")))
            button = {"type": "button", "height": "sm", "style": "primary" if primary else "secondary",
                      "action": _postback_action(q)}
            if primary:
                button["color"] = "#137C50"
            buttons.append(button)
        contents["footer"] = {"type": "box", "layout": "vertical", "spacing": "sm",
                              "paddingAll": "16px", "contents": buttons}
    message = {"type": "flex", "altText": f"Flood Care · {title}\n{output.output_text}"[:400], "contents": contents}
    return message if len(json.dumps(contents, ensure_ascii=False).encode()) < 28000 else None


def build_messages(output: TurnOutput, *, style: Literal["flex", "text"] = "flex") -> list[dict]:
    """Render LINE messages with revision-bound actions unchanged."""
    card = build_flex_message(output) if style == "flex" else None
    messages = [card] if card else [{"type": "text", "text": t} for t in _split_text(output.output_text)]
    if output.quick_replies and messages:
        items = []
        for q in output.quick_replies[:MAX_QUICK_REPLIES]:
            if q.type == "location":
                action = {"type": "location", "label": _button_label(q)}
            else:
                action = _postback_action(q)
            items.append({"type": "action", "action": action})
        messages[-1]["quickReply"] = {"items": items}
    return messages
