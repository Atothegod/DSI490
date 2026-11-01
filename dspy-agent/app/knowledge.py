"""Small CSV retrieval for MVP. Approval and expiration are enforced before search."""
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from .schemas import now_bkk


@dataclass(frozen=True)
class Evidence:
    chunk_id: str
    title: str
    reviewed_text: str
    url: str
    source_org: str


def terms(text):
    text = re.sub(r"\s+", "", text.lower())
    return {text[i:i+3] for i in range(max(0, len(text)-2))}


class KnowledgeBase:
    def __init__(self, path=None, *, threshold=0.12):
        self.threshold = threshold
        self.rows = []
        if path and Path(path).exists():
            with Path(path).open(encoding="utf-8-sig", newline="") as f:
                self.rows = list(csv.DictReader(f))

    def search(self, query, *, now=None, limit=3):
        now = now or now_bkk()
        q, hits = terms(query), []
        for row in self.rows:
            if row.get("review_status") != "approved" or row.get("category") not in {"A", "B"}:
                continue
            if not all(row.get(k) for k in ("chunk_id", "source_org", "title", "url", "reviewed_text", "reviewed_by", "reviewed_at", "verified_at")):
                continue
            try:
                expiry = datetime.fromisoformat(row["valid_until"]) if row.get("valid_until") else None
                if row["category"] == "B" and (not expiry or expiry.tzinfo is None or expiry <= now):
                    continue
                if expiry and (expiry.tzinfo is None or expiry <= now):
                    continue
            except ValueError:
                continue
            t = terms(" ".join(row.get(k, "") for k in ("title", "example_questions", "reviewed_text")))
            score = len(q & t) / max(1, len(q))
            if score >= self.threshold:
                hits.append((score, Evidence(**{k: row[k] for k in Evidence.__dataclass_fields__})))
        return [e for _, e in sorted(hits, key=lambda pair: pair[0], reverse=True)[:limit]]
