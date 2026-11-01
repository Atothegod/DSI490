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

    def _eligible(self, row, now):
        if row.get("review_status") != "approved" or row.get("category") not in {"A", "B"}:
            return False
        if not all(row.get(k) for k in ("chunk_id", "source_org", "title", "url", "reviewed_text", "reviewed_by", "reviewed_at", "verified_at")):
            return False
        try:
            for key in ("verified_at", "reviewed_at"):
                value = datetime.fromisoformat(row[key])
                if value.tzinfo is None or value > now:
                    return False
            expiry = datetime.fromisoformat(row["valid_until"]) if row.get("valid_until") else None
            if row["category"] == "B" and not expiry:
                return False
            if expiry and (expiry.tzinfo is None or expiry <= now):
                return False
        except ValueError:
            return False
        return True

    @staticmethod
    def _evidence(row):
        return Evidence(**{key: row[key] for key in Evidence.__dataclass_fields__})

    def search(self, query, *, now=None, limit=3):
        now = now or now_bkk()
        q, hits = terms(query), []
        for row in self.rows:
            if not self._eligible(row, now):
                continue
            t = terms(" ".join(row.get(k, "") for k in ("title", "example_questions", "reviewed_text")))
            score = len(q & t) / max(1, len(q))
            if score >= self.threshold:
                hits.append((score, self._evidence(row)))
        return [e for _, e in sorted(hits, key=lambda pair: pair[0], reverse=True)[:limit]]

    def core_snippets(self, query=None, *, now=None, limit=3):
        """Curated approved safety facts, available without a retrieval score (4.4).

        Query filtering uses explicit reviewed triggers; these snippets are evidence,
        not a reason to broadcast safety instructions on every urgent message.
        Approval, reviewer metadata and expiry are still required.
        """
        now = now or now_bkk()
        hits = []
        for row in self.rows:
            if row.get("is_core_safety", "").strip().lower() != "true" or row.get("category") != "A":
                continue
            if not self._eligible(row, now):
                continue
            triggers = [part.strip() for part in row.get("core_triggers", "").split(";") if part.strip()]
            if query is not None and not any(trigger in query for trigger in triggers):
                continue
            hits.append(self._evidence(row))
        return hits[:limit]

    def evidence_for(self, query, *, now=None, limit=3):
        # Prioritize relevant curated safety evidence and deduplicate ordinary hits.
        evidence = self.core_snippets(query, now=now, limit=limit) + self.search(query, now=now, limit=limit)
        return list({item.chunk_id:item for item in evidence}.values())[:limit]
