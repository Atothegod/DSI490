"""
Knowledge Retriever  — Static Knowledge Base + Dynamic Retrieval

- เอกสารทุกชิ้นมี kb_doc_id, แหล่งที่มา, ผู้ตรวจ, วันที่ตรวจ
- status = mock_unreviewed ใช้ได้เฉพาะตอนพัฒนา (allow_unreviewed=True) ห้ามตอบประชาชนจริง
- ค้นใหม่ทุกคำถาม คืนคะแนนความเกี่ยวข้อง ต่ำกว่า threshold = ไม่ตอบเนื้อหานั้น
- ไม่มีการคำนวณ Risk Facts หรือ Priority; คำแนะนำและช่องทางติดต่อมาจากเอกสารที่ค้นได้

คะแนน = trigram ของคำถามที่ถ่วงด้วย IDF (คำที่ทุกเอกสารมี เช่น "น้ำท่วม" แทบไม่มีน้ำหนัก)
เทียบกับเนื้อหาและหัวเรื่อง/trigger ของเอกสาร บวกโบนัสเมื่อตรง trigger
grounded_ratio() ใช้ใน Response Validator ตรวจว่าประโยคคำแนะนำยังอิงเอกสารที่อ้างจริง
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path


KB_VERSION = "mock-0.0"
APPROVED = "approved"
# ค่าเริ่มต้นได้จากการรัน example_questions ทุกข้อ (298 ข้อ) เป็นคำถามทดสอบโดยตัดออกจาก index ก่อน
# และชุดคำตอบสั้นที่พบตอนกรอกข้อมูล เช่น "ใช่ค่ะ" "ที่ขา" "โอเค": top-1 77%, คำตอบสั้นหลุดผ่าน 3/28
# (เดิม 63% และหลุด 24/28) ปรับค่าได้ด้วย RAG_THRESHOLD แต่ควรวัดซ้ำด้วยชุดเดียวกัน
DEFAULT_THRESHOLD = 0.25
# คำที่พบในเกือบทุกเอกสาร ไม่ช่วยแยกเรื่อง ตัดออกจากคำถามก่อนคิดคะแนน
_GENERIC = re.compile(r"(?:ช่วง|ตอน|หลัง|ก่อน|เมื่อ)?(?:น้ำท่วม|น้ำลด)|ยังไง|อย่างไร|ทำไง|ได้ไหม|ไหม|หรือไม่|หรือเปล่า"
                      r"|ต้องทำ|ควรทำ|ทำอะไร|อะไรบ้าง|อะไร|บ้าง|วิธี|ดูแล|ควร|ต้อง|แล้ว|ที่|หลัง(?!คา)|ปรึกษา|อยาก"
                      r"|ครับ|ค่ะ|คะ|นะ|หน่อย|ตอนนี้")
_TRIGGER_MATCH = 0.6  # สัดส่วน trigram ของ trigger (หลังตัดคำทั่วไป) ที่ต้องพบในคำถาม จึงถือว่าตรง trigger
_SMOOTHING = 10.0  # น้ำหนัก IDF รวมที่บวกในตัวหาร (ประมาณ trigram เฉพาะเรื่อง 3 ตัว)
# คำหลักสั้น ๆ ที่ trigram แยกไม่ออก (เช่น "แผล" มี trigram เดียวและพบในหลายเอกสาร)
# นับเป็นโบนัสเมื่อพบทั้งในคำถามและในหัวเรื่อง/trigger ของเอกสาร ไม่ใช่แค่ในเนื้อหา
SUBJECT_TERMS = ("แผล", "ไข้", "งู", "ตะขาบ", "แมงป่อง", "ไฟฟ้า", "ไฟดูด", "ไฟรั่ว", "ไฟดับ", "ปลั๊ก", "น้ำดื่ม", "อาหาร",
                 "ท้องเสีย", "ถ่ายเหลว", "อาเจียน", "ยา", "เบาหวาน", "ความดัน", "ฟอกไต", "ตาแดง", "ผื่น", "คัน", "น้ำกัดเท้า",
                 "เด็ก", "ทารก", "นม", "ผู้สูงอายุ", "คนแก่", "ติดเตียง", "ผู้พิการ", "ตั้งครรภ์", "ส้วม", "ขยะ", "ล้างมือ",
                 "เครียด", "กลัว", "กังวล", "ซึมเศร้า", "นอนไม่หลับ", "อยากตาย", "ฆ่าตัวตาย", "ไม่อยากมีชีวิต", "เหล้า", "สุรา",
                 "จมน้ำ", "ตกน้ำ", "อพยพ", "ติดค้าง", "ติดอยู่", "เรือ", "รถ", "หลังคา", "หมดสติ", "หายใจ", "เจ็บหน้าอก", "ชัก",
                 "ล้ม", "ฟกช้ำ", "เครื่องปั่นไฟ", "ตู้เย็น", "ถุงยังชีพ", "กระเป๋า", "ศูนย์พักพิง", "ล้างบ้าน", "เชื้อรา",
                 "1669", "1784", "1323", "1330", "1129", "1130", "1422")
_SUBJECT_BONUS = 0.15


@dataclass(frozen=True)
class KBDoc:
    kb_doc_id: str
    category: str
    title: str
    text: str
    source_org: str
    url: str
    status: str
    reviewed_by: str
    reviewed_at: str
    example_questions: str
    triggers: str
    hotline: str


@dataclass(frozen=True)
class Hit:
    doc: KBDoc
    score: float


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def _grams(text: str) -> set[str]:
    text = _squash(text)
    return {text[i:i + 3] for i in range(max(0, len(text) - 2))}


def _query_grams(query: str) -> set[str]:
    """trigram ของคำถาม หลังตัดคำทั่วไปที่ทุกเอกสารมีเหมือนกัน"""
    stripped = _GENERIC.sub(" ", query)
    return _grams(stripped) or _grams(query)


def grounded_ratio(sentence: str, source: str) -> float:
    """สัดส่วน trigram ของประโยค (ตัดคำลงท้าย/คำทั่วไป) ที่พบในเอกสารต้นทาง 0..1

    ใช้ตรวจว่าประโยคคำแนะนำที่ LLM เรียบเรียงยังอิงเนื้อหาเอกสารจริง ไม่ใช่ความรู้ของโมเดล"""
    grams = _grams(re.sub(r"ค่ะ|คะ|นะ|ครับ|ด้วย|เลย|ก่อน|หน่อย", " ", sentence))
    if not grams:
        return 0.0
    return len(grams & _grams(source)) / len(grams)


class KnowledgeBase:
    def __init__(self, path: str | Path | None = None, *, threshold: float = DEFAULT_THRESHOLD,
                 allow_unreviewed: bool = False, kb_version: str = KB_VERSION):
        self.threshold = threshold
        self.kb_version = kb_version
        self.docs: list[KBDoc] = []
        if path and Path(path).exists():
            with Path(path).open(encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    doc = KBDoc(**{k: row.get(k, "") or "" for k in KBDoc.__dataclass_fields__})
                    reviewed = doc.status == APPROVED and doc.reviewed_by and doc.reviewed_at
                    if reviewed or (allow_unreviewed and doc.status == "mock_unreviewed"):
                        self.docs.append(doc)
        self.by_id = {d.kb_doc_id: d for d in self.docs}
        self._reindex()

    def _reindex(self) -> None:
        """trigram ต่อเอกสาร + IDF: trigram ที่พบในเกือบทุกเอกสาร (เช่น "น้ำท่วม") แทบไม่มีน้ำหนัก"""
        self._index: dict[str, tuple[set[str], set[str], list[tuple[str, set[str]]], frozenset[str]]] = {}
        frequency: Counter[str] = Counter()
        for doc in self.docs:
            triggers = [t.strip() for t in doc.triggers.split(";") if t.strip()]
            # subject = หัวเรื่องและ trigger ที่คนเขียนเอกสารระบุ, body = เนื้อหาทั้งหมด
            heading = " ".join((doc.title, *triggers))
            subject = _grams(heading)
            body = _grams(" ".join((doc.title, doc.example_questions, doc.text, *triggers)))
            # trigger เทียบแบบตัดคำทั่วไป: "ท้องเสียช่วงน้ำท่วม" ตรงกับคำถาม "ท้องเสียหลายรอบ"
            self._index[doc.kb_doc_id] = (subject, body, [(_squash(t), _grams(_GENERIC.sub(" ", t))) for t in triggers],
                                          frozenset(term for term in SUBJECT_TERMS if term in heading))
            frequency.update(body)
        n = len(self.docs)
        self._idf = {g: math.log((n + 1) / (f + 0.5)) for g, f in frequency.items()}
        # trigram ที่ไม่พบในเอกสารใดเลย (คำผัน คำสะกดเพี้ยน) ให้น้ำหนักเท่าคำที่พบบ่อยที่สุด
        # จึงแทบไม่ถ่วงคำถามสั้นที่มีคำสำคัญคำเดียว เช่น "แผลที่ขาโดนน้ำท่วม"
        self._unseen_idf = min(self._idf.values(), default=1.0)

    def _weight(self, gram: str) -> float:
        return self._idf.get(gram, self._unseen_idf)

    def _overlap(self, query: set[str], doc_grams: set[str], total: float) -> float:
        # บวก _SMOOTHING ในตัวหาร: คำสั้น ๆ เช่น "ใช่ค่ะ" ที่มี trigram ไม่กี่ตัวจะได้คะแนนต่ำ
        # แม้ trigram ทั้งหมดจะพบในเอกสารใดเอกสารหนึ่ง
        return sum(self._weight(g) for g in query & doc_grams) / (total + _SMOOTHING) if total else 0.0

    def search(self, query: str, *, limit: int = 3, include_emergency: bool = True) -> list[Hit]:
        q = _query_grams(query)
        total = sum(self._weight(g) for g in q)
        squashed_query = _squash(query)
        query_terms = {term for term in SUBJECT_TERMS if term in squashed_query}
        hits = []
        for doc in self.docs:
            if doc.category == "emergency" and not include_emergency:
                continue
            subject, body, triggers, terms = self._index[doc.kb_doc_id]
            # ตรง trigger เมื่อข้อความ trigger อยู่ในคำถาม หรือ trigram ส่วนใหญ่ของ trigger พบในคำถาม
            matches = sum(1 for text, grams in triggers
                          if text in squashed_query
                          or (len(grams) >= 3 and len(grams & q) / len(grams) >= _TRIGGER_MATCH))
            score = (0.5 * self._overlap(q, body, total) + 0.5 * self._overlap(q, subject, total)
                     + 0.15 * min(matches, 2) + _SUBJECT_BONUS * min(len(query_terms & terms), 2))
            if score >= self.threshold:
                hits.append(Hit(doc, score))
        return sorted(hits, key=lambda h: h.score, reverse=True)[:limit]

    def hotline(self, number: str) -> KBDoc | None:
        # A hotline field may list several numbers. Never use an entire article
        # as a contact card, or select a medical/prevention article by number.
        candidates = [d for d in self.docs if d.category == "emergency"
                      and number in [n.strip() for n in d.hotline.split(";")]]
        candidates.sort(key=lambda d: (number not in d.title, d.hotline != number))
        if not candidates:
            return None
        doc = candidates[0]
        return replace(doc, hotline=number, text=f"{doc.title} — {number}")
