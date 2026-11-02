"""
Response Generator + Response Validator (หัวข้อ 4, 8)

Gemini เรียบเรียงเฉพาะส่วน "รับฟังความรู้สึก" และ "คำแนะนำจาก Knowledge Base" เท่านั้น
เบอร์ฉุกเฉิน คำถาม Slot และสรุปเคสเป็น Template ที่โค้ดเลือก LLM ไม่ได้แต่ง

Validator (กฎเป็นหลัก)
    - ประโยคคำแนะนำต้องอ้าง kb_doc_id ที่ดึงมาในรอบนี้ ไม่อ้าง = ตัดออก
    - ห้ามมีเบอร์โทรส่วนบุคคล ห้ามมีตัวเลขที่ไม่อยู่ใน Draft หรือเอกสารที่อ้าง
    - ห้ามอ้างสถานะการช่วยเหลือ และห้ามบอกว่าผู้ใช้ปลอดภัยแล้ว
    - ไม่ผ่าน: สร้างใหม่ 1 ครั้ง ยังไม่ผ่านใช้ Template (ข้อความเอกสารตรงตัว)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

import dspy
from pydantic import BaseModel

from .knowledge import KBDoc
from .privacy import contains_pii

# ห้ามอ้างสถานะการช่วยเหลือ หรือรับประกันความปลอดภัย
FORBIDDEN = re.compile(
    r"เจ้าหน้าที่(?:กำลัง|จะ|ได้)|กำลังเดินทาง|กำลังไป|ส่งทีม|ส่งเรือ|ส่งคน|ทีมกู้ภัย(?:กำลัง|จะ)|ประสาน(?:งาน)?(?:ให้)?แล้ว"
    r"|ติดต่อเจ้าหน้าที่(?:ให้)?แล้ว|ได้รับความช่วยเหลือแล้ว|รับเคสแล้ว|ปลอดภัยแล้ว|ไม่ต้องกังวล|ไม่มีอันตราย"
)
ADVICE_WORDS = re.compile(r"ควร|อย่า|ห้าม|ให้(?:ไป|ย้าย|ตัด|อยู่|ปิด|ใช้)|โทร")


class Sentence(BaseModel):
    kind: Literal["empathy", "advice"]
    text: str
    kb_doc_id: str | None = None


class ComposeReply(dspy.Signature):
    """เรียบเรียงคำตอบภาษาไทยสุภาพ ลงท้าย "ครับ" สั้น อบอุ่น ไม่ตัดสิน ไม่ลดทอนความรู้สึก
    empathy: แสดงความเข้าใจความรู้สึก 1 ประโยค เฉพาะเมื่อ need_empathy เป็น true ห้ามมีคำแนะนำหรือข้อเท็จจริง
    advice: ทุกประโยคต้องมาจาก kb_documents เท่านั้น ใส่ kb_doc_id ของเอกสารที่ใช้ ห้ามใช้ความรู้ทั่วไปของโมเดล
    ห้ามกลับความหมายคำห้ามหรือคำเตือน ห้ามใส่เบอร์โทร ชื่อหน่วยงาน หรือจุดพักพิงที่ไม่อยู่ในเอกสาร
    ห้ามบอกว่าเจ้าหน้าที่กำลังไป ส่งทีมแล้ว หรือผู้ใช้ปลอดภัยแล้ว ห้ามถามคำถาม (ระบบถามเอง)"""

    user_message: str = dspy.InputField(desc="ข้อความผู้ใช้ที่ผ่าน Privacy Filter แล้ว")
    need_empathy: bool = dspy.InputField()
    kb_documents: str = dspy.InputField(desc="JSON รายการ {kb_doc_id, text}")
    sentences: list[Sentence] = dspy.OutputField()


@dataclass
class Composed:
    text: str
    kb_doc_ids: list[str]
    used_template: bool


def validate(sentences: list[Sentence], docs: list[KBDoc], allowed_numbers: set[str]) -> tuple[list[Sentence], bool]:
    """คืนประโยคที่ผ่าน และ ok=False เมื่อพบการละเมิดร้ายแรงที่ต้องสร้างใหม่"""
    doc_ids = {d.kb_doc_id: d for d in docs}
    kept = []
    for s in sentences:
        text = s.text.strip()
        if not text:
            continue
        if FORBIDDEN.search(text) or contains_pii(text):
            return [], False
        numbers = set(re.findall(r"\d+", text))
        if s.kind == "advice":
            doc = doc_ids.get(s.kb_doc_id or "")
            if doc is None:
                continue  # อ้างเอกสารไม่ได้ → ตัดออก
            if numbers - allowed_numbers - set(re.findall(r"\d+", doc.text)):
                return [], False
        else:
            if numbers or ADVICE_WORDS.search(text):
                return [], False
        kept.append(Sentence(kind=s.kind, text=text, kb_doc_id=s.kb_doc_id if s.kind == "advice" else None))
    return kept, True


def template_reply(empathy_text: str | None, docs: list[KBDoc]) -> Composed:
    parts = ([empathy_text] if empathy_text else []) + [d.text + "ครับ" for d in docs]
    return Composed(" ".join(parts), [d.kb_doc_id for d in docs], True)


class ResponseGenerator:
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.predict = dspy.Predict(ComposeReply) if use_llm else None

    def __call__(self, text: str, *, need_empathy: bool, docs: list[KBDoc], allowed_numbers: set[str],
                 empathy_template: str) -> Composed:
        if not need_empathy and not docs:
            return Composed("", [], False)
        fallback = template_reply(empathy_template if need_empathy else None, docs)
        if not self.use_llm:
            return fallback
        payload = json.dumps([{"kb_doc_id": d.kb_doc_id, "text": d.text} for d in docs], ensure_ascii=False)
        for _ in range(2):  # สร้างครั้งแรก + สร้างใหม่ 1 ครั้ง
            try:
                out = self.predict(user_message=text, need_empathy=need_empathy, kb_documents=payload)
                sentences = [Sentence.model_validate(s) for s in out.sentences]
            except Exception:
                continue
            kept, ok = validate(sentences, docs, allowed_numbers)
            if ok and kept:
                if need_empathy and not any(s.kind == "empathy" for s in kept):
                    kept.insert(0, Sentence(kind="empathy", text=empathy_template))
                ids = list(dict.fromkeys(s.kb_doc_id for s in kept if s.kb_doc_id))
                return Composed(" ".join(s.text for s in kept), ids, False)
        return fallback
