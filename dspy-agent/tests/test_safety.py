"""ทดสอบ safety.py ตาม Design Decisions v1.1 ข้อ 6.2, 6.3, 6.6, 6.7 (ไม่ใช้ LLM)"""

from datetime import datetime

import pytest

from app.safety import (
    PHONE_PLACEHOLDER,
    asks_for_contact,
    correction_cue_for,
    detect_urgent_keywords,
    flags_to_notify,
    format_hotline_text,
    keyword_flags,
    load_keywords,
    merge_urgent_flags,
    normalize_text,
    parse_keywords,
    raw_in_message,
    redact_phones,
    render_contact_answer,
    render_hotline_notice,
)
from app.schemas import BANGKOK_TZ, Hotline, IncidentState, UrgentFlag, UrgentFlagRecord, UrgentFlagSource

NOW = datetime(2026, 10, 30, 11, 0, tzinfo=BANGKOK_TZ)
T, I, H = UrgentFlag.PEOPLE_TRAPPED, UrgentFlag.INJURY_REPORTED, UrgentFlag.HELP_REQUESTED

# ข้อมูลทดสอบ ไม่ใช่ Hotline Config จริง (ของจริงต้องตรวจกับแหล่งทางการและมี verified_at)
HOTLINES = [Hotline(key="a", name="สายด่วนทดสอบ ก", number="1111", source_url="https://example.org", verified_at=NOW),
            Hotline(key="b", name="สายด่วนทดสอบ ข", number="2222", source_url="https://example.org", verified_at=NOW)]
TEMPLATES = {
    "urgent": {
        "people_trapped": "ติดค้าง\n{hotline_text}",
        "injury_reported": "บาดเจ็บ\n{hotline_text}",
        "help_requested": "ขอความช่วยเหลือ\n{hotline_text}",
        "multiple_flags": "หลายเรื่อง\n{hotline_text}",
    },
    "contact_requested": "ช่องทางติดต่อ\n{hotline_text}",
}


# --- 1. ปิดบังเบอร์โทร -----------------------------------------------------------


@pytest.mark.parametrize("text, normalized", [
    ("เบอร์ 081-234-5678", "0812345678"),
    ("โทร 0812345678 ได้เลย", "0812345678"),
    ("081 234 5678", "0812345678"),
    ("08.1234.5678", "0812345678"),
    ("+66 81 234 5678", "0812345678"),
    ("+66812345678", "0812345678"),
    ("๐๘๑๒๓๔๕๖๗๘", "0812345678"),          # ตัวเลขไทย
    ("บ้าน 036-511-234", "036511234"),      # เบอร์บ้าน 9 หลัก
])
def test_phone_formats_are_redacted(text, normalized):
    r = redact_phones(text)
    assert [p.normalized for p in r.phones] == [normalized]
    assert PHONE_PLACEHOLDER in r.text
    assert normalized not in r.text.replace("-", "").replace(" ", "").replace(".", "")


def test_message_from_scenario_g14():
    r = redact_phones("บ้านยายติดน้ำ ออกไม่ได้ อยากให้ช่วยอพยพ ตำบลน้ำตาล เบอร์ 081-234-5678")
    assert r.text.endswith(f"เบอร์ {PHONE_PLACEHOLDER}")
    assert "081" not in r.text and r.phones[0].original == "081-234-5678"


@pytest.mark.parametrize("text", [
    "โทร 1669 หรือ 1784",          # สายด่วนไม่ใช่เบอร์ส่วนตัว
    "น้ำสูง 80 เซนติเมตร",
    "วันที่ 30/10/2026 เวลา 08.30",
    "พิกัด 14.9512, 100.3601",
    "หมู่ 3 บ้านเลขที่ 112/4",
    "ปี 2565 น้ำท่วม 120 ซม.",
])
def test_non_phone_numbers_are_kept(text):
    r = redact_phones(text)
    assert not r.has_phone and r.text == normalize_text(text)


def test_multiple_phones_all_redacted():
    r = redact_phones("ลูก 0812345678 หลาน 0898765432")
    assert len(r.phones) == 2 and r.text.count(PHONE_PLACEHOLDER) == 2


def test_redaction_is_idempotent():
    once = redact_phones("เบอร์ 0812345678").text
    assert redact_phones(once).text == once


# --- 2. คำเร่งด่วน ------------------------------------------------------------------


@pytest.mark.parametrize("text, expected", [
    ("ช่วยด้วยครับ ยายติดอยู่บนหลังคา น้ำขึ้นเร็วมาก", {T, H}),
    ("มีคนบาดเจ็บ ขาโดนของมีคมตอนลุยน้ำ ช่วยด้วย", {I, H}),
    ("บ้านยายติดน้ำ ออกไม่ได้", {T}),
    ("เด็กจมน้ำ", {I}),
    ("โดนไฟดูดตอนเดินในบ้าน", {I}),
    ("SOS ขอเรือด่วน", {H}),
])
def test_urgent_keywords_detected(text, expected):
    assert keyword_flags(text) == expected


@pytest.mark.parametrize("text", [
    "น้ำท่วมบ้านครับ",
    "มีผู้สูงอายุสองคนในบ้าน",             # G17: มีผู้สูงอายุ ≠ ติดค้าง
    "บ้านจมน้ำไปครึ่งหลัง",                # จมน้ำของบ้าน ไม่ใช่คน
    "ฝนตกน้ำท่วมทั้งคืน",                  # ไม่ใช่ "ตกน้ำ"
    "น้ำชักจะขึ้นเรื่อย ๆ",                 # ไม่ใช่ "ชัก"
    "ตายแล้ว น้ำเข้าบ้าน",                 # คำอุทาน
    "ขอบคุณที่มาช่วยค่ะ",
    "ไม่มีคนติดค้างค่ะ ทุกคนออกมาแล้ว",
    "ไม่มีใครบาดเจ็บ",
])
def test_non_urgent_text_has_no_flag(text):
    assert keyword_flags(text) == set()


def test_negation_only_removes_its_own_hit():
    """ปฏิเสธเรื่องหนึ่ง ไม่ลบอีกเรื่องที่ยังเร่งด่วน"""
    assert keyword_flags("ไม่มีคนบาดเจ็บ แต่ยายติดอยู่ชั้นสอง") == {T}


def test_negation_is_narrow_on_purpose():
    """วลีที่มีคำว่า "ไม่มี" แต่ไม่ใช่การปฏิเสธเหตุ ต้องยังจับได้ (พลาดเหตุจริงอันตรายกว่า)"""
    assert keyword_flags("ไม่มีทางออก ติดอยู่ชั้นสอง") == {T}


def test_hits_report_positions():
    hits = detect_urgent_keywords("ช่วยด้วย ติดอยู่")
    assert [(h.flag, h.start) for h in hits] == [(H, 0), (T, 9)]


def test_keywords_file_is_valid_and_covers_every_flag():
    kw = load_keywords()
    assert set(kw.urgent) == set(UrgentFlag)
    assert all(kw.urgent[f] for f in UrgentFlag)


def test_keywords_file_missing_flag_rejected():
    with pytest.raises(ValueError, match="ไม่มีรายการคำ"):
        parse_keywords({"urgent": {"people_trapped": ["ติดอยู่"]}})


# --- 3. รวม flag แบบ OR -------------------------------------------------------------


def test_or_merge_records_sources():
    r = merge_urgent_flags([], keyword_found=[T], gemini_found=[T, H], detected_at=NOW, message_id="m1")
    by = {rec.flag: rec for rec in r.records}
    assert set(by[T].sources) == {UrgentFlagSource.KEYWORD, UrgentFlagSource.GEMINI}
    assert by[H].sources == [UrgentFlagSource.GEMINI]
    assert r.newly_detected == {T, H}


def test_either_source_alone_is_enough():
    assert merge_urgent_flags([], [I], [], detected_at=NOW).flag_types == {I}
    assert merge_urgent_flags([], [], [I], detected_at=NOW).flag_types == {I}


def test_same_flag_from_other_source_is_not_new():
    old = [UrgentFlagRecord(flag=T, sources=["keyword"], detected_at=NOW, message_id="m1")]
    r = merge_urgent_flags(old, keyword_found=[], gemini_found=[T], detected_at=NOW, message_id="m2")
    assert r.newly_detected == set()
    assert len(r.records) == 1 and set(r.records[0].sources) == {UrgentFlagSource.KEYWORD, UrgentFlagSource.GEMINI}
    assert r.records[0].message_id == "m1"  # เก็บข้อความที่พบครั้งแรก


def test_existing_flags_are_never_dropped():
    old = [UrgentFlagRecord(flag=T, sources=["gemini"], detected_at=NOW)]
    assert merge_urgent_flags(old, [], [], detected_at=NOW).flag_types == {T}


def test_merged_records_valid_in_incident():
    r = merge_urgent_flags([], [T, T], [T], detected_at=NOW)
    s = IncidentState(internal_user_id="u", conversation_id="c", urgent_flags=r.records)
    assert s.urgent_flag_types == {T}


# --- 4. สายด่วน ------------------------------------------------------------------------


def _state(flags, notified):
    s0 = IncidentState(internal_user_id="u", conversation_id="c")
    notices = [{"reason": "new_flags", "flags": [f.value for f in notified], "incident_id": s0.incident_id,
                "message_id": "m_sent", "notified_at": NOW}] if notified else []
    return IncidentState(internal_user_id="u", conversation_id="c", incident_id=s0.incident_id,
                         urgent_flags=[{"flag": f.value, "sources": ["keyword"], "detected_at": NOW} for f in flags],
                         hotline_notices=notices)


def test_scenario_r12_no_repeat_for_notified_flag():
    s = _state(flags=[T], notified=[T])
    assert flags_to_notify(s) == set()
    assert render_hotline_notice(flags_to_notify(s), HOTLINES, TEMPLATES) is None


def test_unsent_notice_is_retried():
    """ตรวจพบแล้วแต่ส่งไม่สำเร็จ (ไม่มี HotlineNotice) → ยังต้องแจ้ง"""
    assert flags_to_notify(_state(flags=[T, I], notified=[T])) == {I}


def test_single_new_flag_uses_its_template():
    text = render_hotline_notice({T}, HOTLINES, TEMPLATES)
    assert text.startswith("ติดค้าง") and "1111" in text and "2222" in text


def test_multiple_new_flags_shown_once():
    text = render_hotline_notice({T, I}, HOTLINES, TEMPLATES)
    assert text.startswith("หลายเรื่อง") and text.count("1111") == 1


def test_hotline_text_requires_config():
    with pytest.raises(ValueError, match="Hotline Config"):
        format_hotline_text([])


@pytest.mark.parametrize("text, expected", [
    ("โทรหาใครได้บ้างคะ", True),
    ("ขอเบอร์สายด่วนหน่อย", True),
    ("เบอร์โทร 0812345678 ค่ะ", False),     # ผู้ใช้ให้เบอร์ตัวเอง ไม่ได้ถามช่องทาง
    ("น้ำท่วมบ้าน", False),
])
def test_asks_for_contact(text, expected):
    assert asks_for_contact(text) is expected


def test_contact_answer_always_available():
    assert "1111" in render_contact_answer(HOTLINES, TEMPLATES)


def test_real_template_file_has_required_keys():
    from app.safety import TEMPLATES_PATH, load_templates
    if not TEMPLATES_PATH.exists():
        pytest.skip("ยังไม่มี templates_th.json ในสภาพแวดล้อมนี้")
    t = load_templates()
    assert {f.value for f in UrgentFlag} | {"multiple_flags"} <= set(t["urgent"])
    assert all("{hotline_text}" in v for v in t["urgent"].values())
    assert "{hotline_text}" in t["contact_requested"]


# --- 5. คำบอกการแก้ไขผูกกับ field (ข้อ 6.7) ----------------------------------------------

R11 = "พิมพ์ตำบลผิด เป็นตำบลน้ำตาล แต่น้ำขึ้นเป็น 80 เซนติเมตร"


def test_scenario_r11_cue_bound_to_its_clause():
    assert correction_cue_for("ตำบลน้ำตาล", R11) is True
    assert correction_cue_for("80 เซนติเมตร", R11) is False


@pytest.mark.parametrize("raw, message, expected", [
    ("80 เซนติเมตร", "แก้เป็น 80 เซนติเมตรครับ พิมพ์ผิด", True),
    ("80 เซนติเมตร", "80 เซนติเมตร", False),                       # R09: ตอบคำถาม ไม่ใช่แก้
    ("เป็นเมตร", "ตอนนี้น้ำขึ้นเป็นเมตรแล้ว", False),               # สถานการณ์เปลี่ยน
    ("ตำบลทับยา", "ไม่ใช่ถอนสมอ ตำบลทับยาค่ะ", True),
    ("น้ำเข้าบ้าน", "น้ำเข้าบ้านผิดปกติ", False),                   # "ผิดปกติ" ไม่ใช่คำแก้
])
def test_correction_cues(raw, message, expected):
    assert correction_cue_for(raw, message) is expected


def test_raw_must_come_from_user_message():
    """raw ที่ Gemini สรุปเอง ไม่อยู่ในข้อความ → ผูกไม่ได้ → ผู้เรียกใช้ update"""
    assert raw_in_message("น้ำสูงแปดสิบ", "น้ำสูง 80 ซม.") is False
    assert correction_cue_for("น้ำสูงแปดสิบ", "พิมพ์ผิด น้ำสูง 80 ซม.") is False
    assert raw_in_message(None, "อะไรก็ได้") is False


def test_raw_spanning_clauses_is_unbound():
    assert correction_cue_for("ตำบลน้ำตาล แต่น้ำขึ้น", R11) is False
