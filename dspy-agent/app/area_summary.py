"""Build cross-user area rollups after case delivery; urgent dispatch never waits for them."""
import json
from .handoff import area_summaries
from .schemas import IncidentReport, AreaSummary, AreaSummaryUpdatedEvent, now_bkk


def publish_summaries(repository):
    reports = []
    for user_id, snapshot in repository.all_snapshots():
        if user_id == "__area_summaries__":
            continue
        for value in snapshot.get("last_reports", {}).values():
            # Only current, open cases; stale historical reports must not keep a case counted.
            st = snapshot.get("incidents", {}).get(value["incident_id"])
            if st and st["status"] == "submitted":
                reports.append(IncidentReport.model_validate(value))
    previous = repository.load("__area_summaries__") or {"groups":{}, "version":0}
    summaries = area_summaries(reports)
    groups = {f"{s.area_level}:{s.area_code}":s for s in summaries}
    # Emit an empty snapshot to retire a group after a move/close/delete.
    for key, old in previous["groups"].items():
        if key not in groups:
            groups[key] = AreaSummary(summarized_at=now_bkk(), area_level=old["area_level"], area_code=old["area_code"],
                                      case_versions={}, priority_counts={}, needs=[])
    def fingerprint(value):
        return {k:v for k,v in value.items() if k != "summarized_at"}
    payload = {k:s.model_dump(mode="json") for k,s in groups.items()}
    if {k:fingerprint(v) for k,v in payload.items()} == {k:fingerprint(v) for k,v in previous["groups"].items()}:
        return []
    version = previous["version"]+1
    events = [AreaSummaryUpdatedEvent(case_version=version, summary=s) for key,s in groups.items()
              if key not in previous["groups"] or fingerprint(payload[key]) != fingerprint(previous["groups"][key])]
    repository.save("__area_summaries__", {"version":version, "groups":payload}, events)
    return events
