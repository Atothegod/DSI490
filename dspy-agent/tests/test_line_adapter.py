"""ทดสอบ line_adapter.py — ไม่ใช้ network และไม่ใช้ dspy (ข้อมูล webhook เป็นตัวอย่างตามรูปแบบ LINE Messaging API)"""

import base64
import hashlib
import hmac
import json

import pytest

from app.line_adapter import (
    TEXT_LIMIT,
    build_messages,
    internal_user_id,
    parse_webhook,
    verify_signature,
)
from app.schemas import AgentTurnOutput, NextAction, QuickReply

SECRET, SALT = "test-channel-secret", "test-salt"
LINE_USER = "U0123456789abcdef0123456789abcdef"


def ev(**kw):
    base = {"replyToken": "rt1", "webhookEventId": "evt1", "mode": "active",
            "deliveryContext": {"isRedelivery": False}, "source": {"type": "user", "userId": LINE_USER}}
    base.update(kw)
    return base


def test_signature():
    body = json.dumps({"events": []}).encode()
    good = base64.b64encode(hmac.new(SECRET.encode(), body, hashlib.sha256).digest()).decode()
    assert verify_signature(SECRET, body, good)
    assert not verify_signature(SECRET, body + b" ", good)
    assert not verify_signature(SECRET, body, None) and not verify_signature("", body, good)


def test_internal_user_id_hides_line_id():
    uid = internal_user_id(LINE_USER, SALT)
    assert uid.startswith("usr_") and LINE_USER not in uid
    assert uid == internal_user_id(LINE_USER, SALT) != internal_user_id(LINE_USER, "other-salt")
    with pytest.raises(ValueError):
        internal_user_id(LINE_USER, "")


def test_parse_text_location_postback_and_unsupported():
    body = {"events": [
        ev(type="message", message={"type": "text", "id": "1", "text": "น้ำท่วมบ้าน"}),
        ev(type="message", message={"type": "location", "id": "2", "title": "ตำแหน่ง",
                                    "address": "หมู่ 3 ต.น้ำตาล", "latitude": 14.95, "longitude": 100.36}),
        ev(type="postback", postback={"data": "location_consent=yes"}),
        ev(type="message", message={"type": "sticker", "id": "3"}),
        ev(type="message", message={"type": "text", "text": "x"}, source={"type": "group", "groupId": "G1"}),
        ev(type="message", message={"type": "text", "text": "x"}, mode="standby"),
        ev(type="follow"),
    ]}
    items = parse_webhook(body, salt=SALT)
    assert [i.kind for i in items] == ["text", "location", "postback", "unsupported", "follow"]
    assert items[0].agent_kwargs() == {"input_text": "น้ำท่วมบ้าน"}
    assert items[1].agent_kwargs() == {"location": (14.95, 100.36)}
    assert items[2].agent_kwargs() == {"postback": "location_consent=yes"}
    assert all(i.user_id == internal_user_id(LINE_USER, SALT) for i in items)
    assert "น้ำตาล" not in repr(items[1])  # ไม่เก็บ address
    with pytest.raises(ValueError):
        items[3].agent_kwargs()


def test_redelivery_flag_kept_for_dedupe():
    item = parse_webhook({"events": [ev(type="message", message={"type": "text", "text": "x"},
                                        deliveryContext={"isRedelivery": True})]}, salt=SALT)[0]
    assert item.is_redelivery and item.event_id == "evt1"


def out(text, quick=()):
    return AgentTurnOutput(output_text=text, intents=["update_incident"], incident_id="inc_1", incident_status="draft",
                           next_actions=[NextAction.ASK_FOLLOW_UP], quick_replies=list(quick))


def test_build_messages_with_quick_replies():
    msgs = build_messages(out("ขอแชร์ตำแหน่งค่ะ", [QuickReply(type="location", label="📍 แชร์ตำแหน่ง"),
                                                   QuickReply(type="postback", label="ข้าม", data="location_share=skip")]))
    assert msgs == [{"type": "text", "text": "ขอแชร์ตำแหน่งค่ะ", "quickReply": {"items": [
        {"type": "action", "action": {"type": "location", "label": "📍 แชร์ตำแหน่ง"}},
        {"type": "action", "action": {"type": "postback", "label": "ข้าม", "data": "location_share=skip",
                                      "displayText": "ข้าม"}}]}}]


def test_long_text_split_and_quick_reply_on_last():
    para = "ก" * 3000
    msgs = build_messages(out(f"{para}\n\n{para}", [QuickReply(type="location", label="แชร์")]))
    assert len(msgs) == 2 and all(len(m["text"]) <= TEXT_LIMIT for m in msgs)
    assert "quickReply" not in msgs[0] and "quickReply" in msgs[1]


def test_quick_reply_rules():
    with pytest.raises(ValueError):
        QuickReply(type="postback", label="x")  # postback ต้องมี data
    with pytest.raises(ValueError):
        QuickReply(type="location", label="x" * 21)  # LINE จำกัด 20 ตัวอักษร
