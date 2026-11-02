"""
Information Extractor + Intent Classifier (หัวข้อ 8) — DSPy Signature เดียว

Input ต้องผ่าน Privacy Filter แล้วเสมอ (privacy.filter_text) Gemini ไม่เห็นชื่อ เบอร์ ที่อยู่ หรือพิกัด
ผลลัพธ์ตรวจด้วย Pydantic แล้วผ่าน ground() ซึ่งตัดทุกค่าที่ไม่มีหลักฐานในข้อความจริงทิ้ง
"""

from __future__ import annotations

import json
import re

import dspy

from .schemas import AFFECTED_GROUPS, RISK_FACTS, Extraction, IncidentData, MessageIntent

INSTRUCTIONS = """สกัดข้อมูลเหตุน้ำท่วมจากข้อความล่าสุดของผู้ใช้เท่านั้น ห้ามเติมข้อมูลที่ข้อความไม่ได้บอก
ข้อความผ่านการปิดบังข้อมูลส่วนบุคคลแล้ว ([PHONE] [NAME] [ADDRESS]) ห้ามเดาค่าที่ถูกปิดบัง

intents (เลือกได้หลายค่า): incident = แจ้งเหตุหรือให้ข้อมูลเหตุ, question = ถามคำถามหรือขอคำแนะนำ,
emotional = แสดงความรู้สึก เช่น กลัว เครียด เสียใจ, police = แจ้งเหตุร้ายที่ต้องใช้ตำรวจ

case_intent: same_case = ข้อมูลของเคสที่กำลังคุย, new_case = แจ้งเหตุอีกบ้านหรือแจ้งแทนคนอื่นที่อยู่คนละที่,
relocate = ผู้ประสบภัยย้ายไปอยู่จุดอื่น, unclear = ไม่แน่ใจว่าหมายถึงเคสไหน

problem / needs: ข้อความสั้นภาษาไทยจากสิ่งที่ผู้ใช้บอก ถ้า draft มีอยู่แล้วและข้อความใหม่เพิ่มรายละเอียด
ให้ส่งข้อความที่รวมของเดิมกับของใหม่ ถ้าข้อความใหม่ไม่กล่าวถึงให้เป็น null
*_evidence: ข้อความต้นฉบับที่ยกมาตรงตัวจาก latest_message
needs_answer: "none" เมื่อผู้ใช้บอกว่าไม่ต้องการความช่วยเหลือ, "unknown" เมื่อบอกว่ายังไม่รู้ นอกนั้น null

affected_people: นับเฉพาะจำนวนที่ผู้ใช้บอกเป็นตัวเลขหรือคำบอกจำนวนชัดเจน ห้ามใช้ 0 กลุ่มที่ไม่ได้บอกเป็น null
children เด็ก, elderly ผู้สูงอายุ, disabled ผู้พิการ/ติดเตียง, pregnant_women หญิงตั้งครรภ์,
other ผู้ประสบภัยที่ไม่อยู่ใน 4 กลุ่ม เช่น "มีแค่ผมคนเดียว" = other 1
affected_people_evidence: {ชื่อกลุ่ม: ข้อความต้นฉบับ}
count_unclear: true เมื่อจำนวนกำกวม เช่น "หลายคน" หรือ "แม่ที่แก่แล้ว 2 คน" (ไม่ชัดว่ารวมผู้แจ้งหรือไม่) แล้วอย่าเติมจำนวน

risk_facts: ค่า "true" / "false" / "unknown" ต้องมีข้อความที่ผู้ใช้พิมพ์จริงรองรับใน risk_fact_evidence
ห้ามอนุมานจากชื่อยา อายุ หรือกลุ่มของผู้ประสบภัย ไม่มีหลักฐาน = "unknown"
life_threat อาการคุกคามชีวิตตอนนี้ เช่น จมน้ำ หมดสติ หายใจลำบากรุนแรง บาดเจ็บสาหัส
trapped ออกจากที่อยู่ไม่ได้, water_rising ระดับน้ำกำลังสูงขึ้น,
unsafe_to_evacuate น้ำสูงตั้งแต่ระดับเอวขึ้นไปหรือกระแสน้ำแรง, cannot_self_evacuate มีคนติดเตียงหรือเดินไม่ได้,
in_safe_place ทุกคนอยู่ในจุดที่น้ำไม่ถึงตอนนี้, basic_needs_unmet ขาดอาหาร น้ำดื่ม ยา หรือสิ่งจำเป็น,
essential_medication_risk ผู้แจ้งบอกเองว่ายาหรืออุปกรณ์ที่ขาดนั้นขาดไม่ได้หรือจะเกิดอันตรายร้ายแรง
danger_signals: วลีสั้นจากข้อความที่บอกอันตราย
question: คำถามของผู้ใช้ (ถ้ามี) เขียนใหม่ให้ค้นหาได้ ไม่ใส่ข้อมูลส่วนบุคคล"""


class ExtractIncident(dspy.Signature):
    __doc__ = INSTRUCTIONS

    latest_message: str = dspy.InputField(desc="ข้อความล่าสุดที่ผ่าน Privacy Filter แล้ว")
    draft: str = dspy.InputField(desc="Incident Data ปัจจุบัน (JSON)")
    pending_question: str = dspy.InputField(desc="Slot ที่ระบบกำลังถามค้าง หรือ none")
    conversation_summary: str = dspy.InputField(desc="สรุปบทสนทนาที่ไม่มีข้อมูลส่วนบุคคล")
    result: Extraction = dspy.OutputField()


class Extractor:
    def __init__(self):
        self.predict = dspy.Predict(ExtractIncident)

    def __call__(self, text: str, draft: IncidentData, pending_question: str | None, summary: str) -> Extraction:
        out = self.predict(latest_message=text, draft=json.dumps(draft.model_dump(), ensure_ascii=False),
                           pending_question=pending_question or "none", conversation_summary=summary or "")
        return Extraction.model_validate(out.result)


# ---------------------------------------------------------------------------
# ตรวจหลักฐาน — ใช้กับผลจาก LLM ทุกครั้ง
# ---------------------------------------------------------------------------

_COUNT_WORDS = ("หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก", "เจ็ด", "แปด", "เก้า", "สิบ", "คนเดียว", "ทั้งคู่")
# ข้อความเหล่านี้ทำให้ life_threat = true แม้ LLM พลาด (OR กับ LLM) คำที่มี "ไม่" นำหน้าไม่นับ
LIFE_THREAT_WORDS = ("หมดสติ", "จมน้ำ", "หายใจไม่ออก", "ไม่หายใจ", "หายใจลำบาก", "เลือดออกมาก", "ไฟดูด",
                     "บาดเจ็บสาหัส", "เจ็บท้องคลอด", "น้ำเดิน")
NEEDS_NONE_WORDS = ("ไม่ต้องการ", "ไม่ต้องการอะไร", "ไม่มีอะไร", "ไม่มีครับ", "ไม่มีค่ะ", "ไม่มี", "ไม่ต้อง")
NEEDS_UNKNOWN_WORDS = ("ไม่รู้", "ไม่ทราบ", "ยังไม่รู้", "ไม่แน่ใจ")


def _in(evidence: str | None, text: str) -> bool:
    return bool(evidence) and re.sub(r"\s+", "", evidence) in re.sub(r"\s+", "", text)


def _count_supported(value: int, evidence: str | None, text: str) -> bool:
    if not _in(evidence, text):
        return False
    digits = [int(d) for d in re.findall(r"\d+", evidence)]
    if digits:
        return value in digits
    return any(w in evidence for w in _COUNT_WORDS)


def _negated(text: str, start: int) -> bool:
    return "ไม่" in text[max(0, start - 6):start]


def ground(ex: Extraction, text: str) -> Extraction:
    """ตัดทุกค่าที่หลักฐานไม่อยู่ในข้อความ (No Fabrication)"""
    ex = ex.model_copy(deep=True)
    if not _in(ex.problem_evidence, text):
        ex.problem, ex.problem_evidence = None, None
    if not _in(ex.needs_evidence, text):
        ex.needs, ex.needs_evidence = None, None
    for group in AFFECTED_GROUPS:
        value = getattr(ex.affected_people, group)
        if value is not None and not _count_supported(value, ex.affected_people_evidence.get(group), text):
            setattr(ex.affected_people, group, None)
            ex.affected_people_evidence.pop(group, None)
    if ex.count_unclear:
        ex.affected_people = type(ex.affected_people)()
        ex.affected_people_evidence = {}
    for fact in RISK_FACTS:
        if getattr(ex.risk_facts, fact) != "unknown" and not _in(ex.risk_fact_evidence.get(fact), text):
            setattr(ex.risk_facts, fact, "unknown")
            ex.risk_fact_evidence.pop(fact, None)
    ex.risk_fact_evidence = {k: v for k, v in ex.risk_fact_evidence.items() if getattr(ex.risk_facts, k, "unknown") != "unknown"}
    ex.danger_signals = [s for s in ex.danger_signals if _in(s, text)]

    for word in LIFE_THREAT_WORDS:
        pos = text.find(word)
        if pos >= 0 and not _negated(text, pos):
            ex.risk_facts.life_threat = "true"
            ex.risk_fact_evidence.setdefault("life_threat", word)
            if word not in ex.danger_signals:
                ex.danger_signals.append(word)
            if MessageIntent.INCIDENT not in ex.intents:
                ex.intents.append(MessageIntent.INCIDENT)
            break
    return ex


def needs_answer_from_keywords(text: str) -> str | None:
    """ใช้เมื่อระบบเพิ่งถาม needs แล้วผู้ใช้ตอบสั้น ๆ"""
    stripped = text.strip()
    if any(stripped.startswith(w) for w in NEEDS_UNKNOWN_WORDS):
        return "unknown"
    if any(stripped.startswith(w) for w in NEEDS_NONE_WORDS):
        return "none"
    return None
