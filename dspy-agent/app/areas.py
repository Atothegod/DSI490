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
        tambon = re.search(r"(?:ตำบล|ต\.)\s*([^\s,]+)", raw)
        amphoe = re.search(r"(?:อำเภอ|อ\.)\s*([^\s,]+)", raw)
        if tambon or amphoe:
            rows = [r for r in self.rows if (not tambon or r["tambon"] == tambon.group(1)) and
                    (not amphoe or r["amphoe"] == amphoe.group(1))]
        else:
            rows = self._match_unprefixed(raw)
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

    def _match_unprefixed(self, raw):
        """
        Names without ต./อ. prefixes, with or without spaces ("ทับยาอินทร์บุรี", "น้ำตาลค่ะ").
        A name inside a longer name is ignored ("สิงห์" in "สิงห์บุรี", "บางระจัน" in "ค่ายบางระจัน").
        A tambon plus a different amphoe narrows to rows matching both; otherwise or on conflict,
        every row matching either name is returned so the caller asks the user.
        """
        text = re.sub(r"\s+", "", re.sub(r"(?:จังหวัด|จ\.)", "", raw))
        names = {("tambon", r["tambon"]) for r in self.rows} | {("amphoe", r["amphoe"]) for r in self.rows}
        names |= {("province", r["province"]) for r in self.rows}
        hits = [(m.start(), m.start() + len(n), kind, n) for kind, n in names if n
                for m in re.finditer(re.escape(n), text)]
        hits = [h for h in hits if not any(o[0] <= h[0] and h[1] <= o[1] and o[1] - o[0] > h[1] - h[0] for o in hits)]
        tambons = {(s, e, n) for s, e, kind, n in hits if kind == "tambon"}
        amphoes = {(s, e, n) for s, e, kind, n in hits if kind == "amphoe"}
        rows = [r for r in self.rows if r["tambon"] in {n for *_, n in tambons} or r["amphoe"] in {n for *_, n in amphoes}]
        # A tambon named like its district ("อินทร์บุรี") only narrows when another name is given.
        distinct = {n for s, e, n in tambons if not any((s, e) == (a, b) for a, b, _ in amphoes)}
        if distinct and amphoes:
            narrowed = [r for r in self.rows if r["tambon"] in distinct and r["amphoe"] in {n for *_, n in amphoes}]
            return narrowed or rows
        return rows
