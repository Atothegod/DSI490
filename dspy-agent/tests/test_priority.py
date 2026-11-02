"""Unit Tests ของ Rule Table — Ruleset 1.1.0 (หัวข้อ 14)

Risk Facts ที่ไม่ระบุมีค่า false ส่วน needs ถือว่ามีคำตอบ (filled) เว้นแต่ระบุไว้
"""

import pytest

from app.priority import INSUFFICIENT, RULESET_VERSION, evaluate
from app.schemas import RISK_FACTS, RiskFacts, SlotStatus

NONE, UNKNOWN, FILLED = SlotStatus.ANSWERED_NONE, SlotStatus.ANSWERED_UNKNOWN, SlotStatus.FILLED


def facts(default="false", **kw):
    return RiskFacts(**{f: default for f in RISK_FACTS} | kw)


@pytest.mark.parametrize("case_id, f, needs_status, level, rule", [
    ("U1", facts("unknown", life_threat="true"), FILLED, "critical", "C1"),
    ("U2", facts(life_threat="true", in_safe_place="true"), NONE, "critical", "C1"),
    ("U3", facts(trapped="true", water_rising="true"), FILLED, "high", "H1"),
    ("U4", facts(cannot_self_evacuate="true", in_safe_place="unknown"), FILLED, "high", "H2"),
    ("U5", facts(cannot_self_evacuate="true", in_safe_place="true"), NONE, "low", "L1"),
    ("U6", facts(water_rising="true", in_safe_place="false"), FILLED, "high", "H3"),
    ("U7", facts(water_rising="true", in_safe_place="true"), NONE, "low", "L1"),
    ("U8", facts(essential_medication_risk="true"), FILLED, "high", "H4"),
    ("U9", facts(basic_needs_unmet="true"), FILLED, "medium", "M1"),
    ("U10", facts(trapped="true", water_rising="false"), FILLED, "medium", "M2"),
    ("U11", facts(trapped="true", water_rising="unknown", unsafe_to_evacuate="unknown"), FILLED, "medium", "M2"),
    ("U12", facts(in_safe_place="true"), NONE, "low", "L1"),
    ("U13", facts(in_safe_place="unknown"), NONE, None, "P1"),
    ("U14", facts(in_safe_place="true"), UNKNOWN, None, "P1"),
    ("U15", facts("unknown"), SlotStatus.MISSING, None, "P1"),
    ("U16", facts(in_safe_place="true"), NONE, "low", "L1"),
    ("U17", facts(basic_needs_unmet="true"), FILLED, "medium", "M1"),
    ("U18", facts(trapped="true", unsafe_to_evacuate="true"), FILLED, "high", "H5"),
    ("U19", facts(unsafe_to_evacuate="true", basic_needs_unmet="true"), FILLED, "medium", "M1"),
    ("U20", facts(basic_needs_unmet="true", essential_medication_risk="unknown"), FILLED, "medium", "M1"),
    ("U21", facts(life_threat="true", essential_medication_risk="true"), FILLED, "critical", "C1"),
])
def test_rule_table(case_id, f, needs_status, level, rule):
    result = evaluate(f, needs="อาหาร", needs_status=needs_status)
    assert (result.priority.level.value if result.priority.level else None, result.rule_id) == (level, rule), case_id


def test_u15_reason_and_u24_audit_fields():
    result = evaluate(facts("unknown"))
    assert result.priority.reason == INSUFFICIENT
    assert result.ruleset_version == RULESET_VERSION == "1.1.0"


def test_reason_templates():
    assert evaluate(facts(life_threat="true"), danger_signals=["หมดสติ"]).priority.reason == "พบภาวะคุกคามชีวิต: หมดสติ"
    assert evaluate(facts(basic_needs_unmet="true"), needs="อาหารและน้ำดื่ม").priority.reason == "ขาดสิ่งจำเป็น: อาหารและน้ำดื่ม"


def test_pipeline_contract_has_only_level_and_reason():
    """U23"""
    assert set(evaluate(facts(trapped="true")).priority.model_dump()) == {"level", "reason"}
