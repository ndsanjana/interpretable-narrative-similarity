from __future__ import annotations
import json
import sys
from collections import Counter
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, model_validator


class PlotType(str, Enum):
    OVERCOMING_THE_MONSTER = "Overcoming the Monster"
    RAGS_TO_RICHES = "Rags to Riches"
    QUEST = "Quest"
    VOYAGE_AND_RETURN = "Voyage and Return"
    COMEDY = "Comedy"
    TRAGEDY = "Tragedy"
    REBIRTH = "Rebirth"
    TRICKSTER = "Trickster"
    CAUTIONARY_TALE = "Cautionary Tale"
    OTHER = "Other"


class NarrativeStatus(str, Enum):
    FULL_NARRATIVE = "full_narrative"
    FRAGMENT = "fragment"
    COMPOSITE = "composite"
    NON_NARRATIVE = "non_narrative"


class OutcomeKind(str, Enum):
    RESOLUTION = "resolution"
    FATE = "fate"
    MORAL = "moral"


WORD_BUDGETS = {
    "theme": (3, 8),
    "role": (2, 6),
    "motivation": (3, 10),
    "conflict_summary": (10, 30),
    "action_description": (5, 20),
    "action_function": (2, 4),
    "outcome_description": (5, 20),
}
ACTOR_BAND = {
    NarrativeStatus.FULL_NARRATIVE: (2, 5),
    NarrativeStatus.FRAGMENT: (1, 5),
    NarrativeStatus.COMPOSITE: (2, 6),
    NarrativeStatus.NON_NARRATIVE: (0, 0),
}
ACTION_BAND = {
    NarrativeStatus.FULL_NARRATIVE: (5, 15),
    NarrativeStatus.FRAGMENT: (2, 5),
    NarrativeStatus.COMPOSITE: (5, 25),
    NarrativeStatus.NON_NARRATIVE: (0, 0),
}


def _wc(text: str) -> int:
    return len(text.split())


class Actor(BaseModel):
    id: int = Field(..., ge=1, description="1, 2, 3, ... in order of importance")
    role: str = Field(
        ...,
        min_length=1,
        description="abstract role description, 2 to 6 words, never a proper name",
    )
    motivation: str = Field(
        ..., min_length=1, description="what this actor wants or fears, 3 to 10 words"
    )


class Conflict(BaseModel):
    summary: str = Field(
        ..., description="one sentence, 10 to 30 words; empty only for non_narrative"
    )
    protagonist_ids: List[int] = Field(default_factory=list)
    antagonist_ids: List[int] = Field(default_factory=list)


class Action(BaseModel):
    id: int = Field(..., ge=1, description="1, 2, 3, ... in story order, no gaps")
    description: str = Field(
        ...,
        min_length=1,
        description="one clause, 5 to 20 words, naming the acting role, no pronouns pointing outside the string",
    )
    function: str = Field(
        ..., min_length=1, description="narrative function in 2 to 4 words, plain reusable wording"
    )
    actor_ids: List[int] = Field(default_factory=list)
    caused_by: List[int] = Field(
        default_factory=list,
        description="ids of EARLIER actions that directly cause this one; never mere sequence",
    )


class Outcome(BaseModel):
    description: str = Field(..., min_length=1, description="5 to 20 words")
    kind: OutcomeKind
    actor_ids: List[int] = Field(default_factory=list, description="empty for a general moral")


class NarrativeExtraction(BaseModel):
    narrative_status: NarrativeStatus
    abstract_theme: List[str] = Field(default_factory=list)
    actors: List[Actor] = Field(default_factory=list)
    central_conflict: Conflict
    course_of_action: List[Action] = Field(default_factory=list)
    outcome: List[Outcome] = Field(default_factory=list)
    plot_type: PlotType

    @model_validator(mode="after")
    def _check_structure(self) -> "NarrativeExtraction":
        if self.narrative_status is NarrativeStatus.NON_NARRATIVE:
            if self.abstract_theme or self.actors or self.course_of_action or self.outcome:
                raise ValueError(
                    "non_narrative records must have empty abstract_theme, actors, course_of_action and outcome"
                )
            if self.central_conflict.summary.strip():
                raise ValueError("non_narrative records must have an empty conflict summary")
            if self.plot_type is not PlotType.OTHER:
                raise ValueError("non_narrative records must have plot_type Other")
            return self
        if not self.actors:
            raise ValueError("a narrative record needs at least one actor")
        if len(self.course_of_action) < 2:
            raise ValueError("a narrative record needs at least two actions")
        if not self.outcome:
            raise ValueError("a narrative record needs at least one outcome")
        if not self.abstract_theme:
            raise ValueError("a narrative record needs at least one theme")
        if not self.central_conflict.summary.strip():
            raise ValueError("a narrative record needs a non-empty conflict summary")
        return self

    @model_validator(mode="after")
    def _check_references(self) -> "NarrativeExtraction":
        actor_ids = [a.id for a in self.actors]
        if len(set(actor_ids)) != len(actor_ids):
            raise ValueError("duplicate actor id")
        known_actors = set(actor_ids)
        action_ids = [a.id for a in self.course_of_action]
        if action_ids != list(range(1, len(action_ids) + 1)):
            raise ValueError("course_of_action ids must be 1..n in order with no gaps")
        for name in ("protagonist_ids", "antagonist_ids"):
            for ref in getattr(self.central_conflict, name):
                if ref not in known_actors:
                    raise ValueError(f"central_conflict.{name} references unknown actor {ref}")
        for act in self.course_of_action:
            for ref in act.actor_ids:
                if ref not in known_actors:
                    raise ValueError(f"action {act.id} references unknown actor {ref}")
            if len(set(act.caused_by)) != len(act.caused_by):
                raise ValueError(f"action {act.id} has a duplicate caused_by id")
            for ref in act.caused_by:
                if ref >= act.id:
                    raise ValueError(f"action {act.id} caused_by {ref} is not an earlier action")
                if ref < 1:
                    raise ValueError(f"action {act.id} has an invalid caused_by id {ref}")
        for i, out in enumerate(self.outcome):
            for ref in out.actor_ids:
                if ref not in known_actors:
                    raise ValueError(f"outcome {i} references unknown actor {ref}")
        return self

    def actor_role(self, actor_id: int) -> Optional[str]:
        for a in self.actors:
            if a.id == actor_id:
                return a.role
        return None

    def causal_edges(self) -> List[Tuple[int, int]]:
        return [(src, act.id) for act in self.course_of_action for src in act.caused_by]

    def is_usable(self) -> bool:
        return self.narrative_status is not NarrativeStatus.NON_NARRATIVE


class RepairReport(BaseModel):
    repairs: Dict[str, int] = Field(default_factory=dict)

    def note(self, kind: str, n: int = 1) -> None:
        if n:
            self.repairs[kind] = self.repairs.get(kind, 0) + n

    @property
    def clean(self) -> bool:
        return not self.repairs


def repair(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], RepairReport]:
    rep = RepairReport()
    data = json.loads(json.dumps(raw))
    actors = data.get("actors") or []
    seen, kept = (set(), [])
    for a in actors:
        aid = a.get("id")
        if aid in seen:
            rep.note("duplicate_actor_id")
            continue
        seen.add(aid)
        kept.append(a)
    data["actors"] = kept
    known_actors = seen
    actions = data.get("course_of_action") or []
    old_ids = [a.get("id") for a in actions]
    new_ids = list(range(1, len(actions) + 1))
    if old_ids != new_ids:
        rep.note("renumbered_action_ids")
    id_map = {old: new for old, new in zip(old_ids, new_ids)}
    for a, new_id in zip(actions, new_ids):
        a["id"] = new_id
    for a in actions:
        refs, dropped = ([], 0)
        for r in a.get("actor_ids") or []:
            if r in known_actors and r not in refs:
                refs.append(r)
            else:
                dropped += 1
        rep.note("dangling_action_actor_ref", dropped)
        a["actor_ids"] = refs
        causes, bad_dir, unknown, dup = ([], 0, 0, 0)
        for r in a.get("caused_by") or []:
            mapped = id_map.get(r)
            if mapped is None:
                unknown += 1
                continue
            if mapped >= a["id"]:
                bad_dir += 1
                continue
            if mapped in causes:
                dup += 1
                continue
            causes.append(mapped)
        rep.note("unknown_caused_by_ref", unknown)
        rep.note("forward_or_self_caused_by", bad_dir)
        rep.note("duplicate_caused_by", dup)
        a["caused_by"] = causes
    data["course_of_action"] = actions
    conflict = data.get("central_conflict") or {}
    for name in ("protagonist_ids", "antagonist_ids"):
        refs, dropped = ([], 0)
        for r in conflict.get(name) or []:
            if r in known_actors and r not in refs:
                refs.append(r)
            else:
                dropped += 1
        rep.note("dangling_conflict_actor_ref", dropped)
        conflict[name] = refs
    data["central_conflict"] = conflict
    outcomes = data.get("outcome") or []
    for o in outcomes:
        refs, dropped = ([], 0)
        for r in o.get("actor_ids") or []:
            if r in known_actors and r not in refs:
                refs.append(r)
            else:
                dropped += 1
        rep.note("dangling_outcome_actor_ref", dropped)
        o["actor_ids"] = refs
    data["outcome"] = outcomes
    if data.get("narrative_status") == NarrativeStatus.NON_NARRATIVE.value:
        if data.get("plot_type") != PlotType.OTHER.value:
            rep.note("forced_plot_type_other")
            data["plot_type"] = PlotType.OTHER.value
        if (conflict.get("summary") or "").strip():
            rep.note("cleared_non_narrative_summary")
            conflict["summary"] = ""
    return (data, rep)


def parse_and_repair(raw: Dict[str, Any]) -> Tuple[NarrativeExtraction, RepairReport]:
    fixed, rep = repair(raw)
    return (NarrativeExtraction.model_validate(fixed), rep)


def quality_report(rec: NarrativeExtraction) -> Dict[str, Any]:
    v: Counter = Counter()

    def budget(kind: str, text: str) -> None:
        lo, hi = WORD_BUDGETS[kind]
        n = _wc(text)
        if n < lo:
            v[f"{kind}_under_budget"] += 1
        elif n > hi:
            v[f"{kind}_over_budget"] += 1

    for t in rec.abstract_theme:
        budget("theme", t)
    for a in rec.actors:
        budget("role", a.role)
        budget("motivation", a.motivation)
    if rec.central_conflict.summary.strip():
        budget("conflict_summary", rec.central_conflict.summary)
    for act in rec.course_of_action:
        budget("action_description", act.description)
        budget("action_function", act.function)
    for o in rec.outcome:
        budget("outcome_description", o.description)
    lo, hi = ACTOR_BAND[rec.narrative_status]
    if not lo <= len(rec.actors) <= hi:
        v["actor_count_out_of_band"] += 1
    lo, hi = ACTION_BAND[rec.narrative_status]
    if not lo <= len(rec.course_of_action) <= hi:
        v["action_count_out_of_band"] += 1
    if not 3 <= len(rec.abstract_theme) <= 8 and rec.is_usable():
        v["theme_count_out_of_band"] += 1
    n_act = len(rec.course_of_action)
    chain_like = sum((1 for a in rec.course_of_action if a.caused_by == [a.id - 1]))
    rootless = sum((1 for a in rec.course_of_action if not a.caused_by))
    return {
        "narrative_status": rec.narrative_status.value,
        "plot_type": rec.plot_type.value,
        "n_themes": len(rec.abstract_theme),
        "n_actors": len(rec.actors),
        "n_actions": n_act,
        "n_outcomes": len(rec.outcome),
        "n_causal_edges": len(rec.causal_edges()),
        "chain_fraction": round(chain_like / n_act, 4) if n_act else 0.0,
        "rootless_fraction": round(rootless / n_act, 4) if n_act else 0.0,
        "outcome_kinds": dict(Counter((o.kind.value for o in rec.outcome))),
        "soft_violations": dict(v),
    }


EXAMPLE_RECORD: Dict[str, Any] = {
    "narrative_status": "full_narrative",
    "abstract_theme": [
        "kindness repaid by unexpected aid",
        "the least regarded sibling prevailing",
        "cunning defeating brute force",
    ],
    "actors": [
        {
            "id": 1,
            "role": "the youngest and poorest son",
            "motivation": "to survive and provide for himself",
        },
        {
            "id": 2,
            "role": "the withholding elder brothers",
            "motivation": "to keep the little they have",
        },
        {
            "id": 3,
            "role": "a disguised supernatural helper",
            "motivation": "to reward whoever shows kindness",
        },
        {
            "id": 4,
            "role": "a giant blocking the road",
            "motivation": "to keep travellers and treasure from passing",
        },
    ],
    "central_conflict": {
        "summary": "The youngest son, denied support by his brothers, must get past a giant who blocks the only road.",
        "protagonist_ids": [1],
        "antagonist_ids": [2, 4],
    },
    "course_of_action": [
        {
            "id": 1,
            "description": "The elder brothers refuse the youngest son any bread.",
            "function": "aid denied by kin",
            "actor_ids": [2, 1],
            "caused_by": [],
        },
        {
            "id": 2,
            "description": "The youngest son shares his food with the disguised helper on the road.",
            "function": "kindness to a stranger",
            "actor_ids": [1, 3],
            "caused_by": [],
        },
        {
            "id": 3,
            "description": "The disguised helper gives the youngest son a whistle.",
            "function": "magical gift received",
            "actor_ids": [3, 1],
            "caused_by": [2],
        },
        {
            "id": 4,
            "description": "A giant blocks the youngest son's road.",
            "function": "obstacle confronts hero",
            "actor_ids": [4, 1],
            "caused_by": [],
        },
        {
            "id": 5,
            "description": "The youngest son blows the whistle and the giant falls asleep.",
            "function": "magical object defeats obstacle",
            "actor_ids": [1, 4],
            "caused_by": [3, 4],
        },
        {
            "id": 6,
            "description": "The youngest son carries the sleeping giant's gold home.",
            "function": "treasure won",
            "actor_ids": [1],
            "caused_by": [5],
        },
    ],
    "outcome": [
        {
            "description": "The youngest son returns home wealthy.",
            "kind": "resolution",
            "actor_ids": [1],
        },
        {
            "description": "Kindness freely given is repaid when it is needed.",
            "kind": "moral",
            "actor_ids": [],
        },
    ],
    "plot_type": "Rags to Riches",
}
NON_NARRATIVE_RECORD: Dict[str, Any] = {
    "narrative_status": "non_narrative",
    "abstract_theme": [],
    "actors": [],
    "central_conflict": {"summary": "", "protagonist_ids": [], "antagonist_ids": []},
    "course_of_action": [],
    "outcome": [],
    "plot_type": "Other",
}


def _smoke() -> int:
    failures: List[str] = []

    def check(name: str, cond: bool) -> None:
        print(("  PASS  " if cond else "  FAIL  ") + name)
        if not cond:
            failures.append(name)

    print("1. strict validation of the hand-written example")
    rec = NarrativeExtraction.model_validate(EXAMPLE_RECORD)
    check("record validates strictly", True)
    check("plot_type is the enum member", rec.plot_type is PlotType.RAGS_TO_RICHES)
    check("actor lookup by id works", rec.actor_role(3) == "a disguised supernatural helper")
    check("causal edges extracted", sorted(rec.causal_edges()) == [(2, 3), (3, 5), (4, 5), (5, 6)])
    check("record is usable", rec.is_usable())
    print("2. round trip through JSON")
    again = NarrativeExtraction.model_validate_json(rec.model_dump_json())
    check("round trip is identical", again == rec)
    print("3. repair pass on a clean record is a no-op")
    _, rep = parse_and_repair(EXAMPLE_RECORD)
    check("no repairs needed", rep.clean)
    print("4. soft quality report")
    q = quality_report(rec)
    check("6 actions counted", q["n_actions"] == 6)
    check("4 causal edges counted", q["n_causal_edges"] == 4)
    check("chain fraction is not degenerate", q["chain_fraction"] < 0.5)
    check("no soft violations", q["soft_violations"] == {})
    print("5. non_narrative record")
    nn = NarrativeExtraction.model_validate(NON_NARRATIVE_RECORD)
    check("empty record validates", not nn.is_usable())
    print("6. structural failures are rejected")
    bad = json.loads(json.dumps(NON_NARRATIVE_RECORD))
    bad["plot_type"] = "Quest"
    try:
        NarrativeExtraction.model_validate(bad)
        check("non_narrative with a real plot_type is rejected", False)
    except Exception:
        check("non_narrative with a real plot_type is rejected", True)
    bad2 = json.loads(json.dumps(EXAMPLE_RECORD))
    bad2["plot_type"] = "Bildungsroman"
    try:
        NarrativeExtraction.model_validate(bad2)
        check("plot_type outside the enum is rejected", False)
    except Exception:
        check("plot_type outside the enum is rejected", True)
    print("7. referential damage is repaired, not fatal")
    broken = json.loads(json.dumps(EXAMPLE_RECORD))
    broken["course_of_action"][2]["actor_ids"] = [3, 9]
    broken["central_conflict"]["antagonist_ids"] = [2, 4, 77]
    broken["outcome"][0]["actor_ids"] = [1, 42]
    for i, a in enumerate(broken["course_of_action"]):
        a["id"] = (i + 1) * 10
    broken["course_of_action"][2]["caused_by"] = [20]
    broken["course_of_action"][3]["caused_by"] = [40]
    broken["course_of_action"][4]["caused_by"] = [30, 30, 60, 99]
    try:
        NarrativeExtraction.model_validate(broken)
        check("damaged record fails strict validation", False)
    except Exception:
        check("damaged record fails strict validation", True)
    fixed, rep2 = parse_and_repair(broken)
    check("damaged record validates after repair", True)
    check("action ids renumbered", [a.id for a in fixed.course_of_action] == [1, 2, 3, 4, 5, 6])
    check("old-id caused_by remapped", fixed.course_of_action[2].caused_by == [2])
    check("dangling actor ref dropped", fixed.course_of_action[2].actor_ids == [3])
    check("self reference dropped", fixed.course_of_action[3].caused_by == [])
    check("forward and duplicate causes dropped", fixed.course_of_action[4].caused_by == [3])
    check("dangling conflict ref dropped", fixed.central_conflict.antagonist_ids == [2, 4])
    check("dangling outcome ref dropped", fixed.outcome[0].actor_ids == [1])
    check("repairs were reported", not rep2.clean)
    print("       repairs: " + json.dumps(rep2.repairs, sort_keys=True))
    print("8. the json schema handed to vLLM builds")
    schema = NarrativeExtraction.model_json_schema()
    check(
        "schema has all seven top-level keys",
        set(schema["properties"])
        == {
            "narrative_status",
            "abstract_theme",
            "actors",
            "central_conflict",
            "course_of_action",
            "outcome",
            "plot_type",
        },
    )
    all_props = set(schema["properties"])
    for sub in (schema.get("$defs") or {}).values():
        all_props |= set(sub.get("properties") or {})
    check(
        "no setting field anywhere",
        not any(("setting" in p or "place" in p or "time" in p for p in all_props)),
    )
    check("plot_type enum has 10 values", len(list(PlotType)) == 10)
    print()
    if failures:
        print("SMOKE TEST FAILED: " + "; ".join(failures))
        return 1
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(_smoke())
