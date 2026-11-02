"""
Persistent Memory + Versioning (หัวข้อ 8, 11, 12) — PostgreSQL ตอนใช้งาน, SQLite ตอนรันในเครื่อง/ทดสอบ

ตารางตามหัวข้อ 11 ข้อมูลส่วนบุคคลอยู่ใน case_personal_data, case_version_contacts และ user_line_ids เท่านั้น
case_versions และ case_events เขียนเพิ่มอย่างเดียว ไม่แก้ย้อนหลัง
Pipeline อ่านผ่าน View pipeline_confirmed_cases (ไม่มีชื่อ/เบอร์) และ pipeline_case_contacts (Role จำกัด)
สิทธิ์ Role ของ PostgreSQL อยู่ที่ db/pipeline_roles.sql
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from uuid import uuid4

from .case import Case
from .schemas import SCHEMA_VERSION, CaseStatus, IncidentPriority, Location, PersonalData, SlotName, SlotState, now_bkk

TABLES = (
    """CREATE TABLE IF NOT EXISTS users (
        user_ref TEXT PRIMARY KEY, consent TEXT, consent_at TEXT, current_case_id TEXT,
        pending_question TEXT, held_message_id TEXT, created_at TEXT NOT NULL)""",
    "CREATE TABLE IF NOT EXISTS user_line_ids (user_ref TEXT PRIMARY KEY, line_user_id TEXT NOT NULL)",
    """CREATE TABLE IF NOT EXISTS cases (
        case_id TEXT PRIMARY KEY, seq INTEGER NOT NULL, user_ref TEXT NOT NULL, case_status TEXT NOT NULL,
        intake_status TEXT NOT NULL, priority_level TEXT, priority_reason TEXT NOT NULL, current_version INTEGER NOT NULL,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL, confirmed_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS case_personal_data (
        case_id TEXT PRIMARY KEY, first_name TEXT, last_name TEXT, phone_number TEXT,
        latitude DOUBLE PRECISION, longitude DOUBLE PRECISION, pending_latitude DOUBLE PRECISION, pending_longitude DOUBLE PRECISION)""",
    """CREATE TABLE IF NOT EXISTS case_raw_messages (
        message_id TEXT PRIMARY KEY, user_ref TEXT NOT NULL, case_id TEXT, direction TEXT NOT NULL,
        kind TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL)""",
    "CREATE TABLE IF NOT EXISTS case_drafts (case_id TEXT PRIMARY KEY, draft_rev INTEGER NOT NULL, payload TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS case_slots (case_id TEXT PRIMARY KEY, pending_question TEXT, payload TEXT NOT NULL)",
    """CREATE TABLE IF NOT EXISTS case_versions (
        case_id TEXT NOT NULL, version INTEGER NOT NULL, schema_version TEXT NOT NULL, confirmed_at TEXT NOT NULL,
        incident TEXT NOT NULL, latitude DOUBLE PRECISION NOT NULL, longitude DOUBLE PRECISION NOT NULL,
        priority_level TEXT, priority_reason TEXT NOT NULL, PRIMARY KEY (case_id, version))""",
    """CREATE TABLE IF NOT EXISTS case_version_contacts (
        case_id TEXT NOT NULL, version INTEGER NOT NULL, first_name TEXT NOT NULL, last_name TEXT NOT NULL,
        phone_number TEXT NOT NULL, PRIMARY KEY (case_id, version))""",
    """CREATE TABLE IF NOT EXISTS case_events (
        event_id TEXT PRIMARY KEY, case_id TEXT, event TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL)""",
    "CREATE TABLE IF NOT EXISTS webhook_receipts (event_id TEXT PRIMARY KEY, response TEXT)",
)

_LATEST = """SELECT v.case_id, v.version, v.schema_version, c.case_status, v.confirmed_at, v.incident,
    v.latitude, v.longitude, v.priority_level, v.priority_reason
    FROM case_versions v JOIN cases c ON c.case_id = v.case_id
    WHERE v.version = (SELECT MAX(v2.version) FROM case_versions v2 WHERE v2.case_id = v.case_id)"""
_CONTACTS = """SELECT k.case_id, k.version, k.first_name, k.last_name, k.phone_number FROM case_version_contacts k
    WHERE k.version = (SELECT MAX(k2.version) FROM case_version_contacts k2 WHERE k2.case_id = k.case_id)"""


class Store:
    def __init__(self, url: str = "sqlite:///:memory:"):
        self.lock = threading.RLock()
        self.postgres = url.startswith(("postgresql://", "postgres://"))
        if self.postgres:
            import psycopg
            self.db = psycopg.connect(url)
        else:
            self.db = sqlite3.connect(url.removeprefix("sqlite:///"), check_same_thread=False)
        with self.transaction() as c:
            for sql in TABLES:
                self._exec(c, sql)
            create = "CREATE OR REPLACE VIEW" if self.postgres else "CREATE VIEW IF NOT EXISTS"
            self._exec(c, f"{create} pipeline_confirmed_cases AS {_LATEST}")
            self._exec(c, f"{create} pipeline_case_contacts AS {_CONTACTS}")

    @contextmanager
    def transaction(self):
        with self.lock:
            c = self.db.cursor()
            try:
                yield c
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise
            finally:
                c.close()

    def _exec(self, c, sql, args=()):
        return c.execute(sql.replace("?", "%s") if self.postgres else sql, args)

    def close(self):
        self.db.close()

    # ---- users -------------------------------------------------------------

    def user(self, user_ref: str) -> dict:
        with self.transaction() as c:
            row = self._exec(c, "SELECT consent, consent_at, current_case_id, pending_question, held_message_id FROM users WHERE user_ref=?",
                             (user_ref,)).fetchone()
            if row is None:
                self._exec(c, "INSERT INTO users (user_ref, created_at) VALUES (?,?)", (user_ref, now_bkk().isoformat()))
                row = (None,) * 5
        keys = ("consent", "consent_at", "current_case_id", "pending_question", "held_message_id")
        return dict(user_ref=user_ref, **dict(zip(keys, row)))

    def save_user(self, user: dict) -> None:
        with self.transaction() as c:
            self._exec(c, "UPDATE users SET consent=?, consent_at=?, current_case_id=?, pending_question=?, held_message_id=? WHERE user_ref=?",
                       (user["consent"], user["consent_at"], user["current_case_id"], user["pending_question"],
                        user["held_message_id"], user["user_ref"]))

    def set_line_id(self, user_ref: str, line_user_id: str) -> None:
        with self.transaction() as c:
            self._exec(c, "INSERT INTO user_line_ids VALUES (?,?) ON CONFLICT(user_ref) DO UPDATE SET line_user_id=excluded.line_user_id",
                       (user_ref, line_user_id))

    def line_id(self, user_ref: str) -> str | None:
        with self.transaction() as c:
            row = self._exec(c, "SELECT line_user_id FROM user_line_ids WHERE user_ref=?", (user_ref,)).fetchone()
        return row[0] if row else None

    # ---- cases -------------------------------------------------------------

    def new_case(self, user_ref: str) -> Case:
        with self.transaction() as c:
            seq = self._exec(c, "SELECT COALESCE(MAX(seq), 0) + 1 FROM cases").fetchone()[0]
            case = Case(case_id=f"INC-{seq:06d}", user_ref=user_ref)
            self._write_case(c, case, seq=seq)
        return case

    def load_case(self, case_id: str) -> Case | None:
        with self.transaction() as c:
            row = self._exec(c, "SELECT user_ref, case_status, intake_status, priority_level, priority_reason, current_version, "
                                "created_at, updated_at, confirmed_at FROM cases WHERE case_id=?", (case_id,)).fetchone()
            if row is None:
                return None
            p = self._exec(c, "SELECT first_name, last_name, phone_number, latitude, longitude, pending_latitude, pending_longitude "
                              "FROM case_personal_data WHERE case_id=?", (case_id,)).fetchone()
            draft = self._exec(c, "SELECT draft_rev, payload FROM case_drafts WHERE case_id=?", (case_id,)).fetchone()
            slots = self._exec(c, "SELECT pending_question, payload FROM case_slots WHERE case_id=?", (case_id,)).fetchone()
        d, s = json.loads(draft[1]), json.loads(slots[1])
        return Case(
            case_id=case_id, user_ref=row[0], case_status=row[1], intake_status=row[2],
            priority=IncidentPriority(level=row[3], reason=row[4]), current_version=row[5],
            created_at=row[6], updated_at=row[7], confirmed_at=row[8], draft_rev=draft[0],
            personal=PersonalData(first_name=p[0], last_name=p[1], phone_number=p[2],
                                  location=Location(latitude=p[3], longitude=p[4]) if p[3] is not None else None),
            pending_location=Location(latitude=p[5], longitude=p[6]) if p[5] is not None else None,
            pending_question=slots[0], slots={SlotName(k): SlotState.model_validate(v) for k, v in s.pop("slots").items()},
            **d, **s,
        )

    def save_case(self, case: Case) -> None:
        with self.transaction() as c:
            self._write_case(c, case)

    def _write_case(self, c, case: Case, seq: int | None = None) -> None:
        iso = lambda t: t.isoformat() if t else None  # noqa: E731
        if seq is not None:
            self._exec(c, "INSERT INTO cases VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (case.case_id, seq, case.user_ref, case.case_status.value, case.intake_status.value,
                        None, case.priority.reason, 0, iso(case.created_at), iso(case.updated_at), None))
        else:
            self._exec(c, "UPDATE cases SET case_status=?, intake_status=?, priority_level=?, priority_reason=?, current_version=?, "
                          "updated_at=?, confirmed_at=? WHERE case_id=?",
                       (case.case_status.value, case.intake_status.value, case.priority.level.value if case.priority.level else None,
                        case.priority.reason, case.current_version, iso(case.updated_at), iso(case.confirmed_at), case.case_id))
        p, loc, pend = case.personal, case.personal.location, case.pending_location
        personal = (p.first_name, p.last_name, p.phone_number, loc and loc.latitude, loc and loc.longitude,
                    pend and pend.latitude, pend and pend.longitude)
        draft = case.model_dump(mode="json", include={
            "incident", "risk_facts", "risk_fact_evidence", "danger_signals", "editing", "draft_round", "reminder_sent",
            "last_user_at", "address_masked", "channels_shown", "priority_rule_id"})
        slots = case.model_dump(mode="json", include={
            "slots", "missing_notice_sent", "expect_relocation", "reuse_contact_asked", "last_intents", "conversation_summary"})
        upsert = "INSERT INTO {t} VALUES ({q}) ON CONFLICT(case_id) DO UPDATE SET {s}"
        self._exec(c, upsert.format(t="case_personal_data", q="?,?,?,?,?,?,?,?", s=", ".join(
            f"{k}=excluded.{k}" for k in ("first_name", "last_name", "phone_number", "latitude", "longitude",
                                          "pending_latitude", "pending_longitude"))), (case.case_id, *personal))
        self._exec(c, upsert.format(t="case_drafts", q="?,?,?", s="draft_rev=excluded.draft_rev, payload=excluded.payload"),
                   (case.case_id, case.draft_rev, json.dumps(draft, ensure_ascii=False)))
        self._exec(c, upsert.format(t="case_slots", q="?,?,?", s="pending_question=excluded.pending_question, payload=excluded.payload"),
                   (case.case_id, case.pending_question, json.dumps(slots, ensure_ascii=False)))

    def active_cases(self, user_ref: str) -> list[Case]:
        with self.transaction() as c:
            ids = [r[0] for r in self._exec(c, "SELECT case_id FROM cases WHERE user_ref=? AND case_status=? ORDER BY seq",
                                            (user_ref, CaseStatus.ACTIVE.value)).fetchall()]
        return [self.load_case(i) for i in ids]

    def all_active_cases(self) -> list[Case]:
        with self.transaction() as c:
            ids = [r[0] for r in self._exec(c, "SELECT case_id FROM cases WHERE case_status=?", (CaseStatus.ACTIVE.value,)).fetchall()]
        return [self.load_case(i) for i in ids]

    # ---- versions & events -------------------------------------------------

    def add_version(self, case: Case, snapshot: dict) -> None:
        loc, p = case.personal.location, case.personal
        with self.transaction() as c:
            self._exec(c, "INSERT INTO case_versions VALUES (?,?,?,?,?,?,?,?,?)",
                       (case.case_id, case.current_version, SCHEMA_VERSION, case.confirmed_at.isoformat(),
                        json.dumps(snapshot["incident"], ensure_ascii=False), loc.latitude, loc.longitude,
                        case.priority.level.value if case.priority.level else None, case.priority.reason))
            self._exec(c, "INSERT INTO case_version_contacts VALUES (?,?,?,?,?)",
                       (case.case_id, case.current_version, p.first_name, p.last_name, p.phone_number))
            self._write_case(c, case)

    def latest_version(self, case_id: str) -> dict | None:
        rows = [r for r in self.pipeline_rows() if r["case_id"] == case_id]
        return rows[0] if rows else None

    def version_count(self, case_id: str) -> int:
        with self.transaction() as c:
            return self._exec(c, "SELECT COUNT(*) FROM case_versions WHERE case_id=?", (case_id,)).fetchone()[0]

    def pipeline_rows(self) -> list[dict]:
        """สิ่งที่ Pipeline เห็นผ่าน View — ใช้ในการทดสอบ Data Contract"""
        with self.transaction() as c:
            rows = self._exec(c, "SELECT * FROM pipeline_confirmed_cases").fetchall()
        return [{
            "case_id": r[0], "version": r[1], "schema_version": r[2], "case_status": r[3], "confirmed_at": r[4],
            "incident": json.loads(r[5]), "location": {"latitude": r[6], "longitude": r[7]},
            "incident_priority": {"level": r[8], "reason": r[9]},
        } for r in rows]

    def add_event(self, case_id: str | None, event: str, payload: dict | None = None) -> None:
        with self.transaction() as c:
            self._exec(c, "INSERT INTO case_events VALUES (?,?,?,?,?)",
                       ("evt_" + uuid4().hex[:16], case_id, event, json.dumps(payload or {}, ensure_ascii=False),
                        now_bkk().isoformat()))

    def events(self, case_id: str | None = None, event: str | None = None) -> list[dict]:
        sql, args = "SELECT case_id, event, payload, created_at FROM case_events WHERE 1=1", []
        if case_id:
            sql, args = sql + " AND case_id=?", args + [case_id]
        if event:
            sql, args = sql + " AND event=?", args + [event]
        with self.transaction() as c:
            rows = self._exec(c, sql + " ORDER BY created_at", tuple(args)).fetchall()
        return [{"case_id": r[0], "event": r[1], **json.loads(r[2]), "created_at": r[3]} for r in rows]

    # ---- raw messages ------------------------------------------------------

    def add_raw(self, user_ref: str, case_id: str | None, direction: str, kind: str, content: str) -> str:
        message_id = "msg_" + uuid4().hex[:16]
        with self.transaction() as c:
            self._exec(c, "INSERT INTO case_raw_messages VALUES (?,?,?,?,?,?,?)",
                       (message_id, user_ref, case_id, direction, kind, content, now_bkk().isoformat()))
        return message_id

    def raw(self, message_id: str) -> str | None:
        with self.transaction() as c:
            row = self._exec(c, "SELECT content FROM case_raw_messages WHERE message_id=?", (message_id,)).fetchone()
        return row[0] if row else None

    def attach_raw(self, message_id: str, case_id: str) -> None:
        with self.transaction() as c:
            self._exec(c, "UPDATE case_raw_messages SET case_id=? WHERE message_id=? AND case_id IS NULL", (case_id, message_id))

    # ---- LINE webhook idempotency -----------------------------------------

    def receipt(self, event_id: str) -> dict | None:
        with self.transaction() as c:
            row = self._exec(c, "SELECT response FROM webhook_receipts WHERE event_id=?", (event_id,)).fetchone()
        return json.loads(row[0]) if row and row[0] else None

    def record_receipt(self, event_id: str, response: dict) -> None:
        with self.transaction() as c:
            self._exec(c, "INSERT INTO webhook_receipts VALUES (?,?) ON CONFLICT(event_id) DO UPDATE SET response=excluded.response",
                       (event_id, json.dumps(response, ensure_ascii=False)))


def parse_time(value: str | datetime) -> datetime:
    return value if isinstance(value, datetime) else datetime.fromisoformat(value)
