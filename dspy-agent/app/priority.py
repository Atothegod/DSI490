"""Draft case-level rules (Design Decisions 7.1); never decide area severity."""
from .schemas import CasePriority, IncidentState, PriorityLevel, now_bkk

RANK = {"white": 0, "green": 1, "yellow": 2, "red": 3}


def evaluate_priority(state: IncidentState, *, correction=False, now=None) -> IncidentState:
    now = now or now_bkk()
    old = state.priority
    if old and old.source != "agent1_rules":
        return state  # manual overrides are authoritative
    flags = {x.value for x in state.urgent_flag_types}
    needs = {x.value for x in state.needs.value or []}
    tags = {x.value for x in state.situation.value.tags} if state.situation.value else set()
    red = sorted(flags & {"people_trapped", "injury_reported"}) + ["needs:" + n for n in sorted(needs & {"rescue", "medical"})]
    yellow = ["needs:evacuation"] if "evacuation" in needs else []
    yellow += sorted(tags & {"water_rising", "strong_current", "isolated", "electrical_hazard", "structural_risk"})
    if "water_in_house" in tags and state.vulnerable_people.value:
        yellow += ["vulnerable:" + g.type.value for g in state.vulnerable_people.value.groups
                   if g.type.value in {"elderly", "bedridden", "disabled", "pregnant", "chronic_illness"}]
    water = state.water_level.value
    if water and water.descriptor and water.descriptor.value in {"chest", "neck", "above_head", "first_floor"}:
        yellow.append("water:" + water.descriptor.value)
    green = sorted(tags & {"water_in_house", "road_blocked", "area_flooded"})
    green += ["needs:" + n for n in sorted(needs & {"food_water", "supplies"})]
    if red:
        level, reasons = PriorityLevel.RED, red
    elif yellow:
        level, reasons = PriorityLevel.YELLOW, yellow
    elif flags == {"help_requested"} and not (green or needs - {"info_only"}):
        level, reasons = PriorityLevel.REVIEW, ["help_requested_only"]
    elif green:
        level, reasons = PriorityLevel.GREEN, green
    elif needs == {"info_only"}:
        level, reasons = PriorityLevel.WHITE, ["info_only"]
    else:
        level, reasons = PriorityLevel.REVIEW, ["insufficient_information"]
    if old and not correction and old.level.value in RANK:
        if RANK.get(level.value, -1) < RANK[old.level.value]:
            return state
    current = CasePriority(level=level, reasons=reasons, evaluated_at=now)
    if old and (old.level, old.reasons) == (current.level, current.reasons):
        return state
    return state.model_copy(update={"priority": current,
                                    "priority_history": [*state.priority_history, *([old] if old else [])]})
