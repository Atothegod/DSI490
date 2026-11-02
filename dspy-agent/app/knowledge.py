"""
Knowledge Retriever (หัวข้อ 4) — Static Knowledge Base + Dynamic Retrieval

- เอกสารทุกชิ้นมี kb_doc_id, แหล่งที่มา, ผู้ตรวจ, วันที่ตรวจ
- status = mock_unreviewed ใช้ได้เฉพาะตอนพัฒนา (allow_unreviewed=True) ห้ามตอบประชาชนจริง
- ค้นใหม่ทุกคำถาม คืนคะแนนความเกี่ยวข้อง ต่ำกว่า threshold = ไม่ตอบเนื้อหานั้น
- เบอร์ฉุกเฉินเลือกด้วยโค้ดจาก Risk Facts และเจตนา (emergency_channels) ไม่ใช่ให้ LLM เลือก

Retrieval Threshold ต้องกำหนดจากผลรัน kb_eval_questions ไม่ใช่ค่าที่เดา
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from .schemas import MessageIntent, RiskFacts

KB_VERSION = "mock-0.0"
APPROVED = "approved"


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


def _grams(text: str) -> set[str]:
    text = re.sub(r"\s+", "", text.lower())
    return {text[i:i + 3] for i in range(max(0, len(text) - 2))}


class KnowledgeBase:
    def __init__(self, path: str | Path | None = None, *, threshold: float = 0.12, allow_unreviewed: bool = False,
                 kb_version: str = KB_VERSION):
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

    def search(self, query: str, *, limit: int = 3, include_emergency: bool = True) -> list[Hit]:
        q = _grams(query)
        hits = []
        for doc in self.docs:
            if doc.category == "emergency" and not include_emergency:
                continue
            triggers = [t for t in doc.triggers.split(";") if t.strip()]
            if any(t in query for t in triggers):
                score = 1.0
            else:
                score = len(q & _grams(" ".join((doc.title, doc.example_questions, doc.text)))) / max(1, len(q))
            if score >= self.threshold:
                hits.append(Hit(doc, score))
        return sorted(hits, key=lambda h: h.score, reverse=True)[:limit]

    def hotline(self, number: str) -> KBDoc | None:
        return next((d for d in self.docs if d.hotline == number), None)

    def emergency_channels(self, facts: RiskFacts, intents: list[MessageIntent], *, flood_related: bool = True) -> list[KBDoc]:
        """โค้ดเลือกเบอร์ตามตารางช่องทางฉุกเฉิน แสดงเฉพาะที่ตรงสถานการณ์"""
        numbers = []
        if facts.life_threat == "true":
            numbers.append("1669")
        if flood_related and (facts.life_threat == "true" or "true" in (facts.trapped, facts.unsafe_to_evacuate, facts.water_rising)):
            numbers.append("1784")
        if MessageIntent.POLICE in intents:
            numbers.append("191")
        if MessageIntent.EMOTIONAL in intents:
            numbers.append("1323")
        return [doc for n in numbers if (doc := self.hotline(n))]

    def fallback_channels(self) -> list[KBDoc]:
        """ค้นไม่พบข้อมูลที่เชื่อถือได้ → แนะนำช่องทางทางการที่อยู่ใน Knowledge Base"""
        return [doc for doc in (self.hotline("1784"),) if doc]
