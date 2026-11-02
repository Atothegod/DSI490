"""
Priority Engine — Ruleset 1.1.0 (หัวข้อ 9)

ตรวจจากบนลงล่าง กฎแรกที่เข้าเงื่อนไขคือผลลัพธ์ ไม่เรียก LLM
unknown ไม่ถือเป็น true หรือ false — กฎดูเฉพาะค่าที่ระบุไว้
"""

from __future__ import annotations

from .schemas import IncidentPriority, PriorityLevel, PriorityResult, RiskFacts, SlotStatus

RULESET_VERSION = "1.1.0"
INSUFFICIENT = "ข้อมูลไม่เพียงพอสำหรับการประเมิน"


def evaluate(facts: RiskFacts, *, needs: str | None = None, needs_status: SlotStatus = SlotStatus.MISSING,
             danger_signals: list[str] | None = None) -> PriorityResult:
    f = facts.model_dump()

    def is_(name, value="true"):
        return f[name] == value

    signals = ", ".join(danger_signals or []) or "ตามที่ผู้แจ้งรายงาน"
    rules = (
        ("C1", is_("life_threat"), PriorityLevel.CRITICAL, f"พบภาวะคุกคามชีวิต: {signals}"),
        ("H1", is_("trapped") and is_("water_rising"), PriorityLevel.HIGH, "ติดอยู่ในที่พักและระดับน้ำกำลังสูงขึ้น"),
        ("H2", is_("cannot_self_evacuate") and not is_("in_safe_place"), PriorityLevel.HIGH,
         "มีผู้ที่อพยพเองไม่ได้และยังไม่อยู่ในจุดปลอดภัย"),
        ("H3", is_("water_rising") and is_("in_safe_place", "false"), PriorityLevel.HIGH,
         "ระดับน้ำกำลังสูงขึ้นและยังไม่อยู่ในจุดปลอดภัย"),
        ("H4", is_("essential_medication_risk"), PriorityLevel.HIGH, "ขาดยาหรืออุปกรณ์การแพทย์ที่จำเป็นและมีความเสี่ยงร้ายแรง"),
        ("H5", is_("trapped") and is_("unsafe_to_evacuate"), PriorityLevel.HIGH, "ติดอยู่ในที่พักและสภาพน้ำไม่ปลอดภัยต่อการอพยพ"),
        ("M1", is_("basic_needs_unmet"), PriorityLevel.MEDIUM, f"ขาดสิ่งจำเป็น: {needs or 'ตามที่ผู้แจ้งรายงาน'}"),
        ("M2", is_("trapped"), PriorityLevel.MEDIUM, "ติดอยู่ในที่พัก แต่ไม่มีรายงานความเสี่ยงเพิ่มเติม"),
        ("L1", is_("in_safe_place") and needs_status == SlotStatus.ANSWERED_NONE, PriorityLevel.LOW,
         "ทุกคนอยู่ในจุดปลอดภัยและไม่มีความต้องการเร่งด่วน"),
    )
    for rule_id, matched, level, reason in rules:
        if matched:
            return PriorityResult(priority=IncidentPriority(level=level, reason=reason), rule_id=rule_id,
                                  ruleset_version=RULESET_VERSION)
    return PriorityResult(priority=IncidentPriority(level=None, reason=INSUFFICIENT), rule_id="P1",
                          ruleset_version=RULESET_VERSION)
