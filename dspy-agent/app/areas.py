"""Exact area matching from human-verified CSV. No fuzzy auto-correction or GPS guessing."""
import csv
import re
from pathlib import Path
from .schemas import AreaRef, AreaCandidate, AreaMatchStatus


class AreaResolver:
    def __init__(self, path=None):
        self.rows = []
        if path and Path(path).exists():
            with Path(path).open(encoding="utf-8-sig", newline="") as f:
                self.rows = [r for r in csv.DictReader(f) if r.get("review_status") == "approved" and
                             r.get("source_url") and r.get("verified_at")]

    def __call__(self, mention):
        raw = mention.name
        province = re.search(r"(?:จังหวัด|จ\.)\s*([^\s,]+)", raw)
        if province and province.group(1) != "สิงห์บุรี":
            return AreaRef(raw=raw, stated_level=mention.level, match_status="out_of_province")
        if not self.rows:
            return AreaRef(raw=raw, stated_level=mention.level, match_status="unverified")
        normalized = re.sub(r"(?:ตำบล|ต\.|อำเภอ|อ\.|จังหวัด|จ\.)\s*", " ", raw).strip()
        parts = set(re.split(r"\s+", normalized))
        tambon = re.search(r"(?:ตำบล|ต\.)\s*([^\s,]+)", raw)
        amphoe = re.search(r"(?:อำเภอ|อ\.)\s*([^\s,]+)", raw)
        rows = [r for r in self.rows if (not tambon or r["tambon"] == tambon.group(1)) and
                (not amphoe or r["amphoe"] == amphoe.group(1)) and
                (tambon or amphoe or r["tambon"] in parts or r["amphoe"] in parts)]
        if not rows:
            return AreaRef(raw=raw, stated_level=mention.level, match_status="not_found")
        # An explicitly stated district can be represented without inventing a tambon.
        if mention.level.value == "amphoe":
            rows = list({r["amphoe_code"]:r for r in rows}.values())
        if len(rows) > 1:
            return AreaRef(raw=raw, stated_level=mention.level, match_status="ambiguous",
                           candidates=[AreaCandidate(**{k:r[k] for k in ("tambon_code","tambon","amphoe_code","amphoe")}) for r in rows])
        keys = ("amphoe_code", "amphoe", "province_code", "province") if mention.level.value == "amphoe" else (
                "tambon_code", "tambon", "amphoe_code", "amphoe", "province_code", "province")
        return AreaRef(raw=raw, stated_level=mention.level, match_status="matched", **{k:rows[0][k] for k in keys})
