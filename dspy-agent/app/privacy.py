"""
Privacy Filter (หัวข้อ 7) — ตรวจและแยก PII ก่อนข้อความใดเข้า Gemini ไม่เรียก LLM

    เบอร์โทร   → [PHONE]    ค่าจริงคืนใน phones เพื่อบันทึกลง case_personal_data
    ชื่อ        → [NAME]     ค่าจริงคืนใน first_names / last_names
    ที่อยู่      → [ADDRESS]  ไม่เก็บเป็น Field อยู่เฉพาะใน Raw Message

Regex จับไม่ได้ทุกแบบ ชุดทดสอบอยู่ที่ tests/test_privacy.py ให้เพิ่มเคสภาษาไทยที่พบจริงต่อ
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

PHONE, NAME, ADDRESS = "[PHONE]", "[NAME]", "[ADDRESS]"
THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿"), None)
_DIGIT_WORDS = {"ศูนย์": "0", "หนึ่ง": "1", "สอง": "2", "สาม": "3", "สี่": "4", "ห้า": "5",
                "หก": "6", "เจ็ด": "7", "แปด": "8", "เก้า": "9"}


def normalize_text(text: str) -> str:
    """NFC, ตัดอักขระความกว้างศูนย์, ตัวเลขไทยเป็นอารบิก"""
    return unicodedata.normalize("NFC", text).translate(_ZERO_WIDTH).translate(THAI_DIGITS)


# เบอร์ไทย มือถือ 10 หลัก เบอร์บ้าน 9 หลัก หรือ +66 คั่นด้วยช่องว่าง - . ได้ทีละหลัก
# ไม่จับเบอร์สายด่วน 3–4 หลัก เช่น 1669 1784
_SEP = r"[ \-.]?"
_PHONE_RE = re.compile(r"(?<![\d+])(?:(?:\+66|66)" + _SEP + r"[1-9](?:" + _SEP + r"\d){7,8}|0" + _SEP + r"[1-9](?:"
                       + _SEP + r"\d){7,8})(?!\d)")
_WORD_PHONE_RE = re.compile(r"(?:(?:" + "|".join(_DIGIT_WORDS) + r")\s*){9,10}")

_NAME_STOP = r"(?=\s|$|[,.!?]|เบอร์|โทร|นามสกุล|อยู่|บ้าน|น้ำ|ค่ะ|ครับ|คะ|นะ)"
_FIRST_NAME_RE = re.compile(r"(?:ชื่อเล่น|ชื่อจริง|ชื่อ|นางสาว|นาย|นาง)\s*(?:ว่า\s*)?([ก-๙A-Za-z]{2,30}?)" + _NAME_STOP)
_LAST_NAME_RE = re.compile(r"นามสกุล\s*(?:ว่า\s*)?([ก-๙A-Za-z]{2,30}?)" + _NAME_STOP)
_NOT_NAMES = {"อะไร", "ไม่รู้", "ไม่ทราบ"}
_ADDRESS_RE = re.compile(
    r"(?:บ้านเลขที่|เลขที่)\s*\d+(?:/\d+)?"
    r"|(?:หมู่ที่|หมู่|ม\.)\s*\d+"
    r"|(?:ซอย|ซ\.)\s*[ก-๙\w\-/]+"
    r"|(?:ถนน|ถ\.)\s*(?!แล้ว|ใหญ่|หน้า|เส้น|ไหน|ครับ|ค่ะ)[ก-๙A-Za-z]+"
)


def normalize_phone(raw: str) -> str | None:
    digits = re.sub(r"\D", "", raw)
    if digits.startswith("66"):
        digits = "0" + digits[2:]
    if not digits.startswith("0") or len(digits) < 2:
        return None
    if digits[1] in "689":
        return digits if len(digits) == 10 else None
    return digits if len(digits) == 9 else None


def _word_phone(raw: str) -> str | None:
    return normalize_phone("".join(_DIGIT_WORDS[w] for w in re.findall("|".join(_DIGIT_WORDS), raw)))


@dataclass
class FilterResult:
    text: str  # ข้อความที่ส่ง Gemini ได้
    phones: list[str] = field(default_factory=list)
    first_names: list[str] = field(default_factory=list)
    last_names: list[str] = field(default_factory=list)
    address_masked: bool = False


def _replace(regex, text, placeholder, convert=None, sink=None, group=0):
    def sub(m):
        value = m.group(group)
        if convert is not None:
            value = convert(value)
            if value is None:
                return m.group(0)
        if sink is not None:
            if group and value in _NOT_NAMES:
                return m.group(0)
            sink.append(value)
        if group:
            return m.group(0)[: m.start(group) - m.start(0)] + placeholder + m.group(0)[m.end(group) - m.start(0):]
        return placeholder
    return regex.sub(sub, text)


def filter_text(text: str) -> FilterResult:
    out = FilterResult(text=normalize_text(text))
    out.text = _replace(_PHONE_RE, out.text, PHONE, normalize_phone, out.phones)
    out.text = _replace(_WORD_PHONE_RE, out.text, PHONE, _word_phone, out.phones)
    out.text = _replace(_LAST_NAME_RE, out.text, NAME, sink=out.last_names, group=1)
    out.text = _replace(_FIRST_NAME_RE, out.text, NAME, sink=out.first_names, group=1)
    masked = _ADDRESS_RE.sub(ADDRESS, out.text)
    out.address_masked = masked != out.text
    out.text = masked
    return out


def contains_pii(text: str) -> bool:
    """ใช้ใน Response Validator: คำตอบที่จะส่งต้องไม่มีเบอร์โทรส่วนบุคคล"""
    return bool(_PHONE_RE.search(normalize_text(text)))
