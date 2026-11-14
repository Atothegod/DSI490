"""Project-defined incident response priority, Ruleset 1.1.1 (not flood severity)."""
from __future__ import annotations

import re

from .schemas import CasePriority, Extraction, RiskFacts, now_bkk

RULESET_VERSION = "1.1.1"
INSUFFICIENT_REASON = "ข้อมูลไม่เพียงพอสำหรับการประเมิน"

# Evidence must report the fact itself; age, family roles and drug names alone
# cannot support a risk fact. This is a conservative gate, not a clinical model.
_ANCHORS = {
    "life_threat": r"จมน้ำ|หมดสติ|ไม่รู้สึกตัว|ไม่หายใจ|หายใจ(?:ไม่ออก|ลำบากรุนแรง|ลำบากมาก)|บาดเจ็บสาหัส|เลือดออกไม่หยุด|เสียเลือดมาก|ตกเลือด|ชัก|ภาวะคุกคามชีวิต|อาการฉุกเฉิน",
    "trapped": r"ออกมาได้แล้ว|ออกจากบ้านได้แล้ว|ติด(?:อยู่|ค้าง|ในบ้าน|บ้าน)|ออก(?:จาก[^ ]{0,25})?ไม่ได้|ออกไม่ไหว",
    "water_rising": r"น้ำ.{0,20}(?:เพิ่ม|ขึ้น|สูงขึ้น)|ระดับน้ำ.{0,15}(?:เพิ่ม|ขึ้น)",
    "unsafe_to_evacuate": r"น้ำ.{0,15}(?:เอว|อก|คอ|เหนือหัว|เชี่ยว|ไหลแรง)|กระแสน้ำแรง|อันตราย.{0,15}อพยพ|อพยพ.{0,15}(?:อันตราย|ไม่ปลอดภัย)",
    "cannot_self_evacuate": r"ติดเตียง|เดินไม่ได้|เดินไม่ไหว|อพยพ(?:เอง)?ไม่ได้|เคลื่อนย้าย(?:เอง)?ไม่ได้|ช่วย(?:เหลือ)?ตัวเองไม่ได้",
    "in_safe_place": r"ทุกคน.{0,40}(?:ปลอดภัย|น้ำไม่ถึง)|ทั้งหมด.{0,40}(?:ปลอดภัย|น้ำไม่ถึง)|ยังไม่(?:อยู่ในจุด)?ปลอดภัย|ไม่ได้อยู่ในจุดปลอดภัย|ไม่อยู่ในจุดปลอดภัย|น้ำถึงตัว",
    "basic_needs_unmet": r"ขาด|หมด|ไม่มี|ไม่พอ|ไม่เหลือ|ยังไม่ได้|ต้องการ|ขอ",
    "essential_medication_risk": r"(?:ยา|อุปกรณ์|เครื่อง|ออกซิเจน).{0,60}(?:ขาดไม่ได้|อันตรายร้ายแรง|จำเป็นต่อชีวิต|เสียชีวิต)|(?:ขาดไม่ได้|อันตรายร้ายแรง|จำเป็นต่อชีวิต).{0,40}(?:ยา|อุปกรณ์|เครื่อง|ออกซิเจน)",
}
_UNCERTAIN = re.compile(r"ไม่รู้|ไม่ทราบ|ไม่แน่ใจ|สมมติ|อาจจะ|น่าจะ|เสี่ยงที่จะ")
_BASIC_ITEMS = re.compile(r"อาหาร|น้ำดื่ม|ยา|สิ่งจำเป็น|ของจำเป็น|อุปกรณ์การแพทย์|นม|ผ้าอ้อม")
_FALSE_REPORTS = {
    "life_threat": r"ไม่มี(?:ภาวะคุกคามชีวิต|อาการฉุกเฉิน|คนหมดสติ|คนจมน้ำ)|ไม่ได้(?:จมน้ำ|หมดสติ|บาดเจ็บสาหัส)",
    "trapped": r"ไม่ได้ติด|ไม่ติด|ไม่มีคนติด|ออก(?:มา|จากบ้าน)?ได้(?:แล้ว)?",
    "water_rising": r"น้ำ.{0,20}(?:ไม่(?:ได้)?(?:เพิ่ม|สูงขึ้น|ขึ้น)|คงที่|ลดลง)",
    "unsafe_to_evacuate": r"อพยพได้อย่างปลอดภัย|ไม่มีอันตราย.{0,15}อพยพ",
    "cannot_self_evacuate": r"อพยพเองได้|เดินได้เอง|ช่วย(?:เหลือ)?ตัวเองได้",
    "in_safe_place": r"ยังไม่(?:อยู่ในจุด)?ปลอดภัย|ไม่ได้อยู่ในจุดปลอดภัย|ไม่อยู่ในจุดปลอดภัย|น้ำถึงตัว",
    "basic_needs_unmet": r"ไม่(?:ได้)?ขาด|มี.{0,30}(?:อาหาร|น้ำดื่ม|สิ่งจำเป็น).{0,20}(?:เพียงพอ|ครบ)|ไม่มีสิ่งจำเป็นที่ขาด",
    "essential_medication_risk": r"ขาดยา.{0,20}ไม่(?:มี)?อันตราย|ไม่มียาที่ขาดไม่ได้|ไม่(?:ได้)?มีความเสี่ยงร้ายแรง",
}


def grounded_risk_update(ex: Extraction, text: str, question: str | None = None) -> tuple[RiskFacts, dict[str, str]]:
    """Keep only facts supported by an exact quote from the latest message.

    A short negative answer requires the corresponding prior question. An unknown
    answer never clears an existing fact and never becomes false.
    """
    kept, evidence = {}, {}
    for name in [*RiskFacts.model_fields, "needs_answer"]:
        value = ex.needs_answer if name == "needs_answer" else getattr(ex.risk_facts, name)
        quote = ex.risk_fact_evidence.get(name, "").strip()
        if value is None or value == "unknown" or not quote or quote not in text:
            continue
        if _UNCERTAIN.search(quote) or re.search(r"ไหม|หรือเปล่า|หรือไม่|[?？]", quote):
            continue
        if name != "essential_medication_risk" and re.search(r"ถ้า|หาก", quote):
            continue
        if name == "needs_answer":
            short_none = (value == "none" and re.fullmatch(r"(?:ไม่มี|ไม่ต้องการ|ไม่)(?:ครับ|ค่ะ|คะ)?", quote)
                          and question and re.search(r"ต้องการ|ความช่วยเหลือ", question))
            if value == "none" and not short_none and not re.search(r"ไม่ต้องการ|ไม่ขอ|ไม่มีความต้องการ|ไม่ต้องช่วย|ไม่ต้องการอะไร", quote):
                continue
            if value == "requested" and not re.search(r"ต้องการ|ขอ|ช่วย", quote):
                continue
        else:
            anchored = bool(re.search(_ANCHORS[name], quote))
            short_negative = (value is False and bool(re.fullmatch(r"(?:ไม่มี|ไม่|เปล่า)(?:ครับ|ค่ะ|คะ)?", quote))
                              and question and re.search(_ANCHORS[name], question))
            short_positive = (name != "life_threat" and value is True
                              and re.fullmatch(r"(?:ใช่|มี)(?:ครับ|ค่ะ|คะ)?", quote)
                              and question and re.search(_ANCHORS[name], question))
            false_report = bool(re.search(_FALSE_REPORTS[name], quote))
            if not anchored and not short_negative and not short_positive and not (value is False and false_report):
                continue
            if value is True and false_report:
                continue
            if name == "basic_needs_unmet" and not _BASIC_ITEMS.search(quote) and not short_negative and not short_positive:
                continue
            if value is False and not false_report and not short_negative:
                continue
            if name == "life_threat" and value is True and re.search(r"ถ้า|หาก|กลัวว่า|เสี่ยง|ไม่(?:ได้)?(?:หมดสติ|จมน้ำ|บาดเจ็บสาหัส)", quote):
                continue
        kept[name], evidence[name] = value, quote
    return RiskFacts(**{k: v for k, v in kept.items() if k in RiskFacts.model_fields}), evidence


def merge_facts(current: RiskFacts, current_evidence: dict[str, str], ex: Extraction,
                text: str, question: str | None = None) -> tuple[RiskFacts, dict[str, str]]:
    update, quotes = grounded_risk_update(ex, text, question)
    values = current.model_dump()
    for name in quotes:
        if name in RiskFacts.model_fields:
            values[name] = getattr(update, name)
    return RiskFacts(**values), {**current_evidence, **quotes}


def evaluate_priority(facts: RiskFacts, evidence: dict[str, str], *, now=None) -> CasePriority:
    """First matching rule wins. Only H2 explicitly allows unknown safe-place status."""
    # Defense in depth: an unsupported value cannot participate in a rule.
    facts = RiskFacts(**{name: value for name, value in facts.model_dump().items()
                        if evidence.get(name)})
    level, rule, reason = "medium", "P1", INSUFFICIENT_REASON
    if facts.life_threat is True:
        level, rule, reason = "critical", "C1", "พบภาวะคุกคามชีวิต: " + evidence["life_threat"]
    elif facts.trapped is True and facts.water_rising is True:
        level, rule, reason = "high", "H1", "ติดอยู่ในที่พักและระดับน้ำกำลังสูงขึ้น"
    elif facts.cannot_self_evacuate is True and facts.in_safe_place is not True:
        level, rule, reason = "high", "H2", "มีผู้ที่อพยพเองไม่ได้และยังไม่อยู่ในจุดปลอดภัย"
    elif facts.water_rising is True and facts.in_safe_place is False:
        level, rule, reason = "high", "H3", "ระดับน้ำกำลังสูงขึ้นและยังไม่อยู่ในจุดปลอดภัย"
    elif facts.essential_medication_risk is True:
        level, rule, reason = "high", "H4", "ขาดยาหรืออุปกรณ์การแพทย์ที่จำเป็นและมีความเสี่ยงร้ายแรง"
    elif facts.trapped is True and facts.unsafe_to_evacuate is True:
        level, rule, reason = "high", "H5", "ติดอยู่ในที่พักและสภาพน้ำไม่ปลอดภัยต่อการอพยพ"
    elif facts.basic_needs_unmet is True:
        level, rule, reason = "medium", "M1", "ขาดสิ่งจำเป็น: " + evidence["basic_needs_unmet"]
    elif facts.trapped is True:
        level, rule, reason = "medium", "M2", "ติดอยู่ในที่พัก แต่ไม่มีรายงานความเสี่ยงเพิ่มเติม"
    elif facts.in_safe_place is True and bool(re.search(r"ไม่ต้องการ|ไม่ขอ|ไม่มีความต้องการ|ไม่ต้องช่วย", evidence.get("needs_answer", "")) or re.fullmatch(r"(?:ไม่มี|ไม่)(?:ครับ|ค่ะ|คะ)?", evidence.get("needs_answer", ""))):
        level, rule, reason = "low", "L1", "ทุกคนอยู่ในจุดปลอดภัยและไม่มีความต้องการเร่งด่วน"
    return CasePriority(level=level, reason=reason, rule_id=rule,
                        ruleset_version=RULESET_VERSION, evaluated_at=now or now_bkk())


def initial_priority() -> CasePriority:
    return evaluate_priority(RiskFacts(), {})
