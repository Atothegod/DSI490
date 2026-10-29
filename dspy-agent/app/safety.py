"""
DSI490 — Agent 1 Safety Rules (ขั้นที่ 3)

อ้างอิง Design Decisions v1.1 หัวข้อ 6.2, 6.3, 6.6, 6.7
ไม่เรียก LLM ทุกฟังก์ชันให้ผลแน่นอน ทดสอบได้ด้วย pytest

    1. ปิดบังเบอร์โทร            redact_phones()            ข้อ 6.2 — ทำที่ทางเข้า ก่อนบันทึกและก่อนส่ง Gemini
    2. ตรวจคำเร่งด่วน             detect_urgent_keywords()   ข้อ 6.3
    3. รวม urgent flag แบบ OR      merge_urgent_flags()       ข้อ 6.3 — keyword หรือ Gemini พบ = มี flag
    4. สายด่วน                    flags_to_notify(), render_hotline_notice()  ข้อ 6.6
    5. คำบอกการแก้ไขต่อ field      correction_cue_for()       ข้อ 6.7 — ใช้คู่กับ is_correction ของ Gemini
    6. ผู้ใช้ถามช่องทางติดต่อ      asks_for_contact()         ข้อ 6.6

รายการคำอยู่ใน safety_keywords_th.json ทีมแก้คำได้โดยไม่ต้องแก้โค้ด
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from .schemas import Hotline, IncidentState, UrgentFlag, UrgentFlagRecord, UrgentFlagSource

APP_DIR = Path(__file__).resolve().parent
KEYWORDS_PATH = APP_DIR / "safety_keywords_th.json"
TEMPLATES_PATH = APP_DIR / "templates_th.json"

PHONE_PLACEHOLDER = "[เบอร์โทร]"
THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿"), None)


# ---------------------------------------------------------------------------
# การจัดรูปข้อความ
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """
    จัดรูปข้อความก่อนตรวจ: NFC, ตัดอักขระความกว้างศูนย์, ตัวเลขไทยเป็นอารบิก
    ความยาวข้อความเปลี่ยนได้ (ตัดอักขระความกว้างศูนย์) จึงต้องใช้ข้อความที่ผ่านฟังก์ชันนี้แล้วทั้งระบบ
    """
    text = unicodedata.normalize("NFC", text).translate(_ZERO_WIDTH)
    return text.translate(THAI_DIGITS)


def _fold(text: str) -> str:
    """ใช้เทียบคำเท่านั้น: ตัวพิมพ์เล็ก ไม่เปลี่ยนความยาว"""
    return text.lower()


# ---------------------------------------------------------------------------
# 1. ปิดบังเบอร์โทร (ข้อ 6.2)
# ---------------------------------------------------------------------------

# เบอร์ไทย: มือถือ 0[6-9]x 10 หลัก, บ้าน 0[2-5,7]x 9 หลัก, รูปแบบ +66
# อนุญาตตัวคั่น ช่องว่าง - . ระหว่างตัวเลข ไม่จับเลขสายด่วน 3–4 หลัก เช่น 1669 1784 191
_SEP = r"[ \-.]?"
_PHONE_RE = re.compile(
    r"(?<![\d+])"
    r"(?:"
    r"(?:\+66|66)" + _SEP + r"[1-9](?:" + _SEP + r"\d){7,8}"
    r"|"
    r"0[1-9](?:" + _SEP + r"\d){7,8}"
    r")"
    r"(?!\d)"
)


@dataclass(frozen=True)
class PhoneMatch:
    original: str  # ตามที่ผู้ใช้พิมพ์
    normalized: str  # 0XXXXXXXXX
    start: int
    end: int


@dataclass(frozen=True)
class RedactionResult:
    text: str  # ข้อความที่ปิดบังแล้ว ใช้บันทึกและส่ง Gemini
    phones: list[PhoneMatch]

    @property
    def has_phone(self) -> bool:
        return bool(self.phones)


def _normalize_phone(raw: str) -> str | None:
    digits = re.sub(r"\D", "", raw)
    if digits.startswith("66"):
        digits = "0" + digits[2:]
    if not digits.startswith("0"):
        return None
    # มือถือ 06/08/09 = 10 หลัก, เบอร์บ้าน = 9 หลัก
    if digits[1] in "689":
        return digits if len(digits) == 10 else None
    return digits if len(digits) == 9 else None


def redact_phones(text: str) -> RedactionResult:
    """
    หาเบอร์โทรแล้วแทนด้วย [เบอร์โทร] — เบอร์จริงคืนใน phones เพื่อเก็บลง ContactRecord เท่านั้น
    ข้อจำกัด: ไม่จับเบอร์ที่เขียนเป็นตัวอักษร เช่น "ศูนย์แปดหนึ่ง..."
    """
    text = normalize_text(text)
    phones: list[PhoneMatch] = []
    out: list[str] = []
    last = 0
    for m in _PHONE_RE.finditer(text):
        normalized = _normalize_phone(m.group())
        if normalized is None:
            continue
        phones.append(PhoneMatch(m.group(), normalized, m.start(), m.end()))
        out.append(text[last:m.start()])
        out.append(PHONE_PLACEHOLDER)
        last = m.end()
    out.append(text[last:])
    return RedactionResult("".join(out), phones)


# ---------------------------------------------------------------------------
# รายการคำ
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SafetyKeywords:
    urgent: dict[UrgentFlag, tuple[str, ...]]
    negations: tuple[str, ...]
    correction_cues: tuple[str, ...]
    correction_patterns: tuple[re.Pattern[str], ...]
    contact_requests: tuple[str, ...]
    version: str = "unknown"


def parse_keywords(data: dict) -> SafetyKeywords:
    urgent = {}
    for name, words in data["urgent"].items():
        urgent[UrgentFlag(name)] = tuple(_fold(normalize_text(w)) for w in words if w.strip())
    missing = set(UrgentFlag) - set(urgent)
    if missing:
        raise ValueError(f"safety keywords ไม่มีรายการคำของ {sorted(f.value for f in missing)}")
    return SafetyKeywords(
        urgent=urgent,
        negations=tuple(_fold(normalize_text(w)) for w in data.get("negations", [])),
        correction_cues=tuple(_fold(normalize_text(w)) for w in data.get("correction_cues", [])),
        correction_patterns=tuple(re.compile(p) for p in data.get("correction_cue_patterns", [])),
        contact_requests=tuple(_fold(normalize_text(w)) for w in data.get("contact_requests", [])),
        version=str(data.get("version", "unknown")),
    )


@lru_cache(maxsize=4)
def load_keywords(path: str | Path = KEYWORDS_PATH) -> SafetyKeywords:
    with open(path, encoding="utf-8") as f:
        return parse_keywords(json.load(f))


# ---------------------------------------------------------------------------
# 2. ตรวจคำเร่งด่วน (ข้อ 6.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KeywordHit:
    flag: UrgentFlag
    keyword: str
    start: int
    end: int


def _find_all(haystack: str, needle: str) -> Iterable[int]:
    start = haystack.find(needle)
    while start != -1:
        yield start
        start = haystack.find(needle, start + 1)


def detect_urgent_keywords(text: str, keywords: SafetyKeywords | None = None) -> list[KeywordHit]:
    """
    คืนตำแหน่งคำเร่งด่วนทั้งหมดในข้อความ
    hit ที่อยู่ภายในวลีปฏิเสธ (เช่น "ไม่มีคนติด") ถูกตัดออกเฉพาะ hit นั้น ไม่ตัดทั้งข้อความ
    """
    kw = keywords or load_keywords()
    folded = _fold(normalize_text(text))
    negated: list[tuple[int, int]] = [
        (i, i + len(n)) for n in kw.negations for i in _find_all(folded, n)
    ]

    def inside_negation(start: int, end: int) -> bool:
        return any(ns <= start < ne or ns < end <= ne for ns, ne in negated)

    hits: list[KeywordHit] = []
    for flag, words in kw.urgent.items():
        for w in words:
            for i in _find_all(folded, w):
                if not inside_negation(i, i + len(w)):
                    hits.append(KeywordHit(flag, w, i, i + len(w)))
    return sorted(hits, key=lambda h: (h.start, h.flag.value))


def keyword_flags(text: str, keywords: SafetyKeywords | None = None) -> set[UrgentFlag]:
    return {h.flag for h in detect_urgent_keywords(text, keywords)}


# ---------------------------------------------------------------------------
# 3. รวม urgent flag แบบ OR (ข้อ 6.3)
# ---------------------------------------------------------------------------


@dataclass
class FlagMergeResult:
    records: list[UrgentFlagRecord]
    newly_detected: set[UrgentFlag] = field(default_factory=set)  # ชนิดที่ incident ไม่เคยมี

    @property
    def flag_types(self) -> set[UrgentFlag]:
        return {r.flag for r in self.records}


def merge_urgent_flags(
    existing: list[UrgentFlagRecord],
    keyword_found: Iterable[UrgentFlag],
    gemini_found: Iterable[UrgentFlag],
    *,
    detected_at: datetime,
    message_id: str | None = None,
) -> FlagMergeResult:
    """
    keyword หรือ Gemini พบ = มี flag (OR)
    flag ชนิดเดิมที่พบจากอีกแหล่ง → เพิ่มแหล่งในระเบียนเดิม ไม่ถือเป็น flag ใหม่ (ข้อ 6.6)
    ไม่ลบ flag เดิม แม้ข้อความนี้ไม่พบ
    """
    sources: dict[UrgentFlag, list[UrgentFlagSource]] = {}
    for found, source in ((keyword_found, UrgentFlagSource.KEYWORD), (gemini_found, UrgentFlagSource.GEMINI)):
        for f in found:  # ชนิดเดียวกันพบหลายคำในข้อความเดียว → นับแหล่งครั้งเดียว
            lst = sources.setdefault(UrgentFlag(f), [])
            if source not in lst:
                lst.append(source)

    records: list[UrgentFlagRecord] = []
    seen: set[UrgentFlag] = set()
    for rec in existing:
        add = [s for s in sources.get(rec.flag, []) if s not in rec.sources]
        records.append(rec.model_copy(update={"sources": [*rec.sources, *add]}) if add else rec)
        seen.add(rec.flag)

    new = {f for f in sources if f not in seen}
    for f in sorted(new, key=lambda x: x.value):
        records.append(UrgentFlagRecord(flag=f, sources=sources[f], detected_at=detected_at, message_id=message_id))
    return FlagMergeResult(records=records, newly_detected=new)


# ---------------------------------------------------------------------------
# 4. สายด่วน (ข้อ 6.6)
# ---------------------------------------------------------------------------


def flags_to_notify(state: IncidentState) -> set[UrgentFlag]:
    """flag ที่ตรวจพบแล้วแต่ยังไม่เคยแจ้งสายด่วนสำเร็จ — ส่งไม่สำเร็จรอบก่อนจะยังอยู่ในรายการนี้"""
    return state.urgent_flag_types - state.notified_flag_types


def asks_for_contact(text: str, keywords: SafetyKeywords | None = None) -> bool:
    kw = keywords or load_keywords()
    folded = _fold(normalize_text(text))
    return any(w in folded for w in kw.contact_requests)


@lru_cache(maxsize=4)
def load_templates(path: str | Path = TEMPLATES_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def format_hotline_text(hotlines: Iterable[Hotline]) -> str:
    """ข้อความเบอร์จาก Hotline Config เท่านั้น ห้ามสร้างเบอร์เอง (ข้อ 4.2)"""
    lines = [f"• {h.name} {h.number}" for h in hotlines]
    if not lines:
        raise ValueError("ไม่มี Hotline Config ที่ตรวจสอบแล้ว ห้ามแสดงข้อความด่วนโดยไม่มีเบอร์")
    return "\n".join(lines)


def render_hotline_notice(
    flags: Iterable[UrgentFlag],
    hotlines: Iterable[Hotline],
    templates: dict | None = None,
) -> str | None:
    """
    ข้อความด่วนสำหรับ flag ใหม่ — หลาย flag ในข้อความเดียวแสดงรวมครั้งเดียว
    คืน None เมื่อไม่มี flag ใหม่ (ไม่แสดงสายด่วนซ้ำ)
    """
    flags = sorted(set(flags), key=lambda f: f.value)
    if not flags:
        return None
    tpl = (templates or load_templates())["urgent"]
    key = flags[0].value if len(flags) == 1 else "multiple_flags"
    return tpl[key].format(hotline_text=format_hotline_text(hotlines))


def render_contact_answer(hotlines: Iterable[Hotline], templates: dict | None = None) -> str:
    """ผู้ใช้ถามช่องทางติดต่อ ตอบได้เสมอแม้เคยแสดงแล้ว"""
    return (templates or load_templates())["contact_requested"].format(hotline_text=format_hotline_text(hotlines))


# ---------------------------------------------------------------------------
# 5. คำบอกการแก้ไข ผูกกับ field (ข้อ 6.7)
# ---------------------------------------------------------------------------

# ตัวช่วยแบ่งช่วงประโยค ไม่ใช่การตัดคำที่สมบูรณ์ — ผูกไม่ชัดให้ผู้เรียกใช้ update
_CLAUSE_SPLIT = re.compile(r"[,\n;!?،]|(?:\s*(?:แต่|และ|ส่วน|แล้วก็)\s*)")


def clauses(text: str) -> list[tuple[int, int]]:
    """ตำแหน่ง (start, end) ของแต่ละช่วงประโยค"""
    spans, start = [], 0
    for m in _CLAUSE_SPLIT.finditer(text):
        if m.start() > start:
            spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def raw_in_message(raw: str | None, message: str) -> bool:
    """ตรวจว่า raw ที่ Gemini คืนมาอยู่ในข้อความผู้ใช้จริง (ข้อ 6.1)"""
    if not raw:
        return False
    norm = lambda s: re.sub(r"\s+", " ", _fold(normalize_text(s))).strip()
    return norm(raw) in norm(message)


def has_correction_cue(text: str, keywords: SafetyKeywords | None = None) -> bool:
    kw = keywords or load_keywords()
    folded = _fold(normalize_text(text))
    return any(c in folded for c in kw.correction_cues) or any(p.search(folded) for p in kw.correction_patterns)


def correction_cue_for(raw: str | None, message: str, keywords: SafetyKeywords | None = None) -> bool:
    """
    มีคำบอกการแก้ไขอยู่ในช่วงประโยคเดียวกับ raw ของ field นี้หรือไม่
    คืน False เมื่อ raw ไม่อยู่ในข้อความ หรือ raw คร่อมหลายช่วง (ผูกไม่ชัด → update)
    """
    if not raw_in_message(raw, message):
        return False
    text = _fold(normalize_text(message))
    needle = _fold(normalize_text(raw)).strip()
    pos = text.find(needle)
    if pos == -1:  # ต่างกันแค่ช่องว่าง: ผูกตำแหน่งไม่ได้แน่นอน
        return False
    end = pos + len(needle)
    for cs, ce in clauses(text):
        if cs <= pos and end <= ce:
            return has_correction_cue(text[cs:ce], keywords)
    return False


__all__ = [
    "PHONE_PLACEHOLDER", "FlagMergeResult", "KeywordHit", "PhoneMatch", "RedactionResult", "SafetyKeywords",
    "asks_for_contact", "clauses", "correction_cue_for", "detect_urgent_keywords", "flags_to_notify",
    "format_hotline_text", "has_correction_cue", "keyword_flags", "load_keywords", "load_templates",
    "merge_urgent_flags", "normalize_text", "parse_keywords", "raw_in_message", "redact_phones",
    "render_contact_answer", "render_hotline_notice",
]
