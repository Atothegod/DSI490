"""Durable snapshots and transactional outbox; SQLite for local use, PostgreSQL for deployment.

No raw user messages or LINE IDs are stored here. Handoff events and user state commit
in the same transaction. Retry changes delivery metadata, never the event ID/version.
"""
import json
import sqlite3
import threading
from contextlib import contextmanager
from .schemas import OutboxRecord, DeliveryStatus, now_bkk


class Repository:
    def __init__(self, url="sqlite:///:memory:"):
        self.lock = threading.RLock()
        self.postgres = url.startswith(("postgresql://", "postgres://"))
        if self.postgres:
            import psycopg
            self.db = psycopg.connect(url)
        else:
            path = url.removeprefix("sqlite:///")
            self.db = sqlite3.connect(path, check_same_thread=False)
        with self.transaction() as c:
            for sql in (
                "CREATE TABLE IF NOT EXISTS agent_snapshots (user_id TEXT PRIMARY KEY, payload TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS escalation_events (event_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, payload TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS event_inbox (event_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, response TEXT)",
                "CREATE TABLE IF NOT EXISTS deletion_requests (event_id TEXT PRIMARY KEY, payload TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS audit_logs (audit_id TEXT PRIMARY KEY, payload TEXT NOT NULL)",
            ):
                c.execute(sql)

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

    def execute(self, c, sql, args=()):
        return c.execute(sql.replace("?", "%s") if self.postgres else sql, args)

    def load(self, user_id):
        with self.transaction() as c:
            row = self.execute(c, "SELECT payload FROM agent_snapshots WHERE user_id=?", (user_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, user_id, snapshot, events=()):
        with self.transaction() as c:
            self.execute(c, "INSERT INTO agent_snapshots VALUES (?,?) ON CONFLICT(user_id) DO UPDATE SET payload=excluded.payload",
                         (user_id, json.dumps(snapshot, ensure_ascii=False)))
            for event in events:
                payload = event.model_dump(mode="json")
                record = OutboxRecord(event_id=event.event_id, event_type=event.event_type,
                                      incident_id=payload.get("incident_id"), payload=payload)
                self.execute(c, "INSERT INTO escalation_events VALUES (?,?,?) ON CONFLICT(event_id) DO NOTHING",
                             (event.event_id, user_id, record.model_dump_json()))
                if payload["event_type"] == "data_deletion_requested":
                    from .schemas import DeletionRequest
                    request = DeletionRequest(internal_user_id=user_id, requested_at=event.occurred_at,
                                              consents_revoked_at=event.occurred_at, local_deleted_at=event.occurred_at, event_id=event.event_id)
                    self.execute(c, "INSERT INTO deletion_requests VALUES (?,?) ON CONFLICT(event_id) DO NOTHING",
                                 (event.event_id, request.model_dump_json()))

    def outbox(self, user_id=None):
        with self.transaction() as c:
            sql = "SELECT payload FROM escalation_events"
            rows = self.execute(c, sql + (" WHERE user_id=?" if user_id else ""), (user_id,) if user_id else ()).fetchall()
        return [OutboxRecord.model_validate_json(r[0]) for r in rows]

    def update_outbox(self, record):
        with self.transaction() as c:
            self.execute(c, "UPDATE escalation_events SET payload=? WHERE event_id=?", (record.model_dump_json(), record.event_id))
            if record.event_type.value == "data_deletion_requested" and record.acked_at:
                row = self.execute(c, "SELECT payload FROM deletion_requests WHERE event_id=?", (record.event_id,)).fetchone()
                if row:
                    request = json.loads(row[0])
                    request["agent2_acked_at"] = record.acked_at.isoformat()
                    self.execute(c, "UPDATE deletion_requests SET payload=? WHERE event_id=?", (json.dumps(request), record.event_id))

    def all_snapshots(self):
        with self.transaction() as c:
            return [(u, json.loads(s)) for u, s in c.execute("SELECT user_id,payload FROM agent_snapshots").fetchall()]

    def receipt(self, event_id, user_id):
        with self.transaction() as c:
            row = self.execute(c, "SELECT response FROM event_inbox WHERE event_id=? AND user_id=?", (event_id, user_id)).fetchone()
        return json.loads(row[0]) if row and row[0] else None

    def record_receipt(self, event_id, user_id, response):
        with self.transaction() as c:
            self.execute(c, "INSERT INTO event_inbox VALUES (?,?,?) ON CONFLICT(event_id) DO UPDATE SET response=excluded.response",
                         (event_id, user_id, json.dumps(response, ensure_ascii=False)))

    def audit(self, payload):
        from .schemas import new_id
        with self.transaction() as c:
            self.execute(c, "INSERT INTO audit_logs VALUES (?,?)", (new_id("audit"), json.dumps(payload, ensure_ascii=False)))

    def scrub_events(self, user_id, incident_ids=None, *, all_personal=False):
        """Remove historical coordinates including changes and outbox copies (6.10)."""
        def scrub(v, key=""):
            if key in {"coordinates", "location_coordinates", "contact_ref", "landmark", "location_landmark"}:
                return None
            if key == "coordinates_basis":
                return None
            if all_personal and key in {"text", "raw", "note", "name", "phone"}:
                return "[ลบข้อมูลแล้ว]" if isinstance(v, str) else None
            if isinstance(v, dict):
                if v.get("field") in {"location_coordinates", "location_landmark", "contact_available", "location_consent"}:
                    return None
                out = {k: scrub(x, k) for k, x in v.items()}
                if "contact_available" in out:
                    out["contact_available"] = False
                if isinstance(out.get("location"), dict):
                    loc = out["location"]
                    out["location_missing"] = not bool(loc.get("area") or loc.get("coordinates") or loc.get("landmark"))
                return out
            if isinstance(v, list):
                return [r for x in v if (r := scrub(x)) is not None]
            return v
        for record in self.outbox(user_id):
            if incident_ids is not None and record.incident_id not in incident_ids:
                continue
            record.payload = scrub(record.payload)
            # Old unsent updates may be emptied by erasure. Never deliver stale PII.
            if record.payload.get("event_type") == "incident_updated" and not record.payload.get("changes"):
                record.delivery_status = DeliveryStatus.FAILED
                record.last_error = "cancelled_by_erasure"
            self.update_outbox(record)
        # Cached API responses can include report excerpts; erase them as well.
        with self.transaction() as c:
            self.execute(c, "DELETE FROM event_inbox WHERE user_id=?", (user_id,))

    def close(self):
        self.db.close()
