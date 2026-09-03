from __future__ import annotations
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import torch
from torch_geometric.data import HeteroData

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from schemas import (
    NarrativeExtraction,
    NarrativeStatus,
    OutcomeKind,
    PlotType,
    RepairReport,
    parse_and_repair,
)

NODE_TYPES: Tuple[str, ...] = (
    "story",
    "theme",
    "actor",
    "motivation",
    "conflict",
    "action",
    "outcome",
)
TEXT_NODE_TYPES: Tuple[str, ...] = ("theme", "actor", "motivation", "conflict", "action", "outcome")
AUX_TEXT_NODE_TYPES: Tuple[str, ...] = ("action",)
FORWARD_RELATIONS: Tuple[Tuple[str, str, str], ...] = (
    ("theme", "theme_of", "story"),
    ("actor", "actor_in", "story"),
    ("conflict", "conflict_of", "story"),
    ("action", "action_in", "story"),
    ("outcome", "outcome_of", "story"),
    ("actor", "motivated_by", "motivation"),
    ("actor", "performs", "action"),
    ("actor", "experiences", "outcome"),
    ("actor", "protagonist_of", "conflict"),
    ("actor", "antagonist_of", "conflict"),
    ("action", "precedes", "action"),
    ("action", "causes", "action"),
    ("conflict", "resolved_by", "outcome"),
    ("conflict", "drives", "action"),
)
REVERSE_PREFIX = "rev_"


def reverse_relation(rel: Tuple[str, str, str]) -> Tuple[str, str, str]:
    src, name, dst = rel
    return (dst, REVERSE_PREFIX + name, src)


REVERSE_RELATIONS: Tuple[Tuple[str, str, str], ...] = tuple(
    (reverse_relation(r) for r in FORWARD_RELATIONS)
)
EDGE_TYPES: Tuple[Tuple[str, str, str], ...] = FORWARD_RELATIONS + REVERSE_RELATIONS


def metadata(add_reverse: bool = True) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    edges = list(EDGE_TYPES) if add_reverse else list(FORWARD_RELATIONS)
    return (list(NODE_TYPES), edges)


METADATA = metadata(True)
PLOT_TYPE_VALUES: Tuple[str, ...] = tuple((p.value for p in PlotType))
PLOT_TYPE_INDEX: Dict[str, int] = {v: i for i, v in enumerate(PLOT_TYPE_VALUES)}
N_PLOT_TYPES = len(PLOT_TYPE_VALUES)
NARRATIVE_STATUS_VALUES: Tuple[str, ...] = tuple((s.value for s in NarrativeStatus))
NARRATIVE_STATUS_INDEX: Dict[str, int] = {v: i for i, v in enumerate(NARRATIVE_STATUS_VALUES)}
N_NARRATIVE_STATUS = len(NARRATIVE_STATUS_VALUES)
OUTCOME_KIND_VALUES: Tuple[str, ...] = tuple((k.value for k in OutcomeKind))
OUTCOME_KIND_INDEX: Dict[str, int] = {v: i for i, v in enumerate(OUTCOME_KIND_VALUES)}
N_OUTCOME_KINDS = len(OUTCOME_KIND_VALUES)
STORY_FEATURE_FIELDS: Tuple[str, ...] = (
    "n_actions_scaled",
    "n_actors_scaled",
    "n_themes_scaled",
    "n_outcomes_scaled",
    "max_fanout_scaled",
    "mean_parent_count_scaled",
    "max_parent_count_scaled",
    "chain_fraction",
    "rootless_fraction",
    "causal_density_scaled",
    "causal_component_fraction",
    "branch_action_fraction",
    "join_action_fraction",
    "protagonist_fraction",
    "antagonist_fraction",
    "mean_actor_participation",
    "outcome_resolution_fraction",
    "outcome_fate_fraction",
    "outcome_moral_fraction",
    "is_degenerate",
    "was_repaired",
    "max_fanout_x_full_narrative",
    "max_fanout_x_fragment",
    "max_fanout_x_composite",
    "max_fanout_x_non_narrative",
    "mean_parent_x_full_narrative",
    "mean_parent_x_fragment",
    "mean_parent_x_composite",
    "mean_parent_x_non_narrative",
)
STORY_FEATURE_DIM = len(STORY_FEATURE_FIELDS)
THEME_SCALAR_FIELDS: Tuple[str, ...] = ("importance_rank",)
ACTOR_SCALAR_FIELDS: Tuple[str, ...] = (
    "importance_rank",
    "participation_fraction",
    "is_protagonist",
    "is_antagonist",
)
MOTIVATION_SCALAR_FIELDS: Tuple[str, ...] = ("importance_rank",)
CONFLICT_SCALAR_FIELDS: Tuple[str, ...] = ("protagonist_fraction", "antagonist_fraction")
ACTION_SCALAR_FIELDS: Tuple[str, ...] = (
    "position",
    "is_first",
    "is_last",
    "fanout_scaled",
    "parent_count_scaled",
)
OUTCOME_SCALAR_FIELDS: Tuple[str, ...] = ("order_rank",)
SCALAR_FIELDS: Dict[str, Tuple[str, ...]] = {
    "story": STORY_FEATURE_FIELDS,
    "theme": THEME_SCALAR_FIELDS,
    "actor": ACTOR_SCALAR_FIELDS,
    "motivation": MOTIVATION_SCALAR_FIELDS,
    "conflict": CONFLICT_SCALAR_FIELDS,
    "action": ACTION_SCALAR_FIELDS,
    "outcome": OUTCOME_SCALAR_FIELDS,
}
SCALAR_DIMS: Dict[str, int] = {k: len(v) for k, v in SCALAR_FIELDS.items()}
CAP_ACTIONS = 25.0
CAP_ACTORS = 6.0
CAP_THEMES = 8.0
CAP_OUTCOMES = 5.0
CAP_FANOUT = 5.0
CAP_PARENTS = 5.0
CAP_MEAN_PARENTS = 2.0
CAP_CAUSAL_DENSITY = 2.0


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _edge_tensor(pairs: Sequence[Tuple[int, int]]) -> torch.Tensor:
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(list(pairs), dtype=torch.long).t().contiguous()


def _normalise_pairs(pairs: Sequence[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], int]:
    uniq = sorted(set(pairs))
    return (uniq, len(pairs) - len(uniq))


def _components(n: int, edges: Sequence[Tuple[int, int]]) -> int:
    if n == 0:
        return 0
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u, v in edges:
        ru, rv = (find(u), find(v))
        if ru != rv:
            parent[ru] = rv
    return len({find(i) for i in range(n)})


@dataclass
class TaleMeta:
    tale_id: str
    atu_type: Optional[str] = None
    split: Optional[str] = None
    title: Optional[str] = None
    n_words: Optional[int] = None
    source: Optional[str] = None


@dataclass
class GraphTexts:
    texts: Dict[str, List[str]] = field(default_factory=dict)
    aux_texts: Dict[str, List[str]] = field(default_factory=dict)

    def counts(self) -> Dict[str, int]:
        return {k: len(v) for k, v in self.texts.items()}


@dataclass
class GraphRecord:
    data: HeteroData
    texts: GraphTexts
    meta: TaleMeta
    narrative_status: str
    plot_type: str
    is_degenerate: bool
    was_repaired: bool
    repairs: Dict[str, int] = field(default_factory=dict)
    node_counts: Dict[str, int] = field(default_factory=dict)
    edge_counts: Dict[str, int] = field(default_factory=dict)
    duplicate_edges: Dict[str, int] = field(default_factory=dict)
    isolated_nodes: Dict[str, int] = field(default_factory=dict)
    conflict_drives_rule: str = "no_sides"


@dataclass
class BuildReport:
    n_records: int = 0
    n_built: int = 0
    n_failed: int = 0
    n_degenerate: int = 0
    n_repaired: int = 0
    status_counts: Counter = field(default_factory=Counter)
    plot_type_counts: Counter = field(default_factory=Counter)
    node_counts: Counter = field(default_factory=Counter)
    edge_counts: Counter = field(default_factory=Counter)
    repair_counts: Counter = field(default_factory=Counter)
    duplicate_edge_counts: Counter = field(default_factory=Counter)
    isolated_node_counts: Counter = field(default_factory=Counter)
    conflict_drives_rule_counts: Counter = field(default_factory=Counter)
    failures: List[Tuple[str, str]] = field(default_factory=list)

    def note_record(self) -> None:
        self.n_records += 1

    def note_failure(self, tale_id: str, reason: str) -> None:
        self.n_failed += 1
        self.failures.append((tale_id, reason))

    def add(self, rec: GraphRecord) -> None:
        self.n_built += 1
        self.status_counts[rec.narrative_status] += 1
        self.plot_type_counts[rec.plot_type] += 1
        if rec.is_degenerate:
            self.n_degenerate += 1
        if rec.was_repaired:
            self.n_repaired += 1
        for k, v in rec.repairs.items():
            self.repair_counts[k] += v
        for k, v in rec.node_counts.items():
            self.node_counts[k] += v
        for k, v in rec.edge_counts.items():
            self.edge_counts[k] += v
        for k, v in rec.duplicate_edges.items():
            self.duplicate_edge_counts[k] += v
        for k, v in rec.isolated_nodes.items():
            self.isolated_node_counts[k] += v
        self.conflict_drives_rule_counts[rec.conflict_drives_rule] += 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_records": self.n_records,
            "n_built": self.n_built,
            "n_failed": self.n_failed,
            "n_degenerate": self.n_degenerate,
            "n_repaired": self.n_repaired,
            "status_counts": dict(self.status_counts),
            "plot_type_counts": dict(self.plot_type_counts),
            "node_counts": dict(self.node_counts),
            "edge_counts": dict(self.edge_counts),
            "repair_counts": dict(self.repair_counts),
            "duplicate_edge_counts": dict(self.duplicate_edge_counts),
            "isolated_node_counts": dict(self.isolated_node_counts),
            "conflict_drives_rule_counts": dict(self.conflict_drives_rule_counts),
            "failures": [list(f) for f in self.failures],
        }

    def format_text(self) -> str:
        lines: List[str] = []
        lines.append("graph build report")
        lines.append(
            "  records %d, built %d, failed %d, degenerate %d, repaired %d"
            % (self.n_records, self.n_built, self.n_failed, self.n_degenerate, self.n_repaired)
        )
        lines.append("  narrative_status: " + json.dumps(dict(self.status_counts), sort_keys=True))
        lines.append("  nodes: " + json.dumps(dict(self.node_counts), sort_keys=True))
        lines.append("  edges (forward relations):")
        for src, name, dst in FORWARD_RELATIONS:
            lines.append("    %-16s %s -> %s: %d" % (name, src, dst, self.edge_counts.get(name, 0)))
        if self.repair_counts:
            lines.append("  repairs: " + json.dumps(dict(self.repair_counts), sort_keys=True))
        if self.duplicate_edge_counts:
            lines.append(
                "  duplicate edges dropped: "
                + json.dumps(dict(self.duplicate_edge_counts), sort_keys=True)
            )
        if self.isolated_node_counts:
            lines.append(
                "  isolated nodes: " + json.dumps(dict(self.isolated_node_counts), sort_keys=True)
            )
        lines.append(
            "  conflict_drives rule: "
            + json.dumps(dict(self.conflict_drives_rule_counts), sort_keys=True)
        )
        if self.failures:
            lines.append("  failures:")
            for tale_id, reason in self.failures:
                lines.append("    %s: %s" % (tale_id, reason))
        return "\n".join(lines)


def _story_features(
    rec: NarrativeExtraction,
    fanout: List[int],
    parents: List[int],
    causal_pairs: List[Tuple[int, int]],
    participation: List[int],
    was_repaired: bool,
) -> List[float]:
    n_act = len(rec.course_of_action)
    n_actor = len(rec.actors)
    n_theme = len(rec.abstract_theme)
    n_out = len(rec.outcome)
    max_fanout = max(fanout) if fanout else 0
    max_parent = max(parents) if parents else 0
    mean_parent = sum(parents) / n_act if n_act else 0.0
    n_causal = len(causal_pairs)
    chain_like = sum((1 for a in rec.course_of_action if a.caused_by == [a.id - 1]))
    rootless = sum((1 for a in rec.course_of_action if not a.caused_by))
    branchy = sum((1 for f in fanout if f >= 2))
    joiny = sum((1 for p in parents if p >= 2))
    n_comp = _components(n_act, causal_pairs)
    kinds = Counter((o.kind.value for o in rec.outcome))
    max_fanout_scaled = _clip01(max_fanout / CAP_FANOUT)
    mean_parent_scaled = _clip01(mean_parent / CAP_MEAN_PARENTS)
    base = [
        _clip01(math.log1p(n_act) / math.log1p(CAP_ACTIONS)),
        _clip01(n_actor / CAP_ACTORS),
        _clip01(n_theme / CAP_THEMES),
        _clip01(n_out / CAP_OUTCOMES),
        max_fanout_scaled,
        mean_parent_scaled,
        _clip01(max_parent / CAP_PARENTS),
        _clip01(chain_like / n_act) if n_act else 0.0,
        _clip01(rootless / n_act) if n_act else 0.0,
        _clip01(n_causal / max(1, n_act - 1) / CAP_CAUSAL_DENSITY) if n_act else 0.0,
        _clip01(n_comp / n_act) if n_act else 0.0,
        _clip01(branchy / n_act) if n_act else 0.0,
        _clip01(joiny / n_act) if n_act else 0.0,
        _clip01(len(rec.central_conflict.protagonist_ids) / max(1, n_actor)),
        _clip01(len(rec.central_conflict.antagonist_ids) / max(1, n_actor)),
        _clip01(sum(participation) / n_actor / n_act if n_actor and n_act else 0.0),
        _clip01(kinds.get("resolution", 0) / n_out) if n_out else 0.0,
        _clip01(kinds.get("fate", 0) / n_out) if n_out else 0.0,
        _clip01(kinds.get("moral", 0) / n_out) if n_out else 0.0,
        1.0 if rec.narrative_status is NarrativeStatus.NON_NARRATIVE else 0.0,
        1.0 if was_repaired else 0.0,
    ]
    status_onehot = [0.0] * N_NARRATIVE_STATUS
    status_onehot[NARRATIVE_STATUS_INDEX[rec.narrative_status.value]] = 1.0
    inter = [max_fanout_scaled * h for h in status_onehot]
    inter += [mean_parent_scaled * h for h in status_onehot]
    out = base + inter
    assert len(out) == STORY_FEATURE_DIM
    return out


def build_graph(
    rec: NarrativeExtraction,
    meta: TaleMeta,
    add_reverse: bool = True,
    repairs: Optional[Dict[str, int]] = None,
) -> GraphRecord:
    repairs = dict(repairs or {})
    was_repaired = bool(repairs)
    degenerate = rec.narrative_status is NarrativeStatus.NON_NARRATIVE
    themes = list(rec.abstract_theme)
    actors = sorted(rec.actors, key=lambda a: a.id)
    actions = sorted(rec.course_of_action, key=lambda a: a.id)
    outcomes = list(rec.outcome)
    has_conflict = bool(rec.central_conflict.summary.strip())
    actor_pos = {a.id: i for i, a in enumerate(actors)}
    action_pos = {a.id: i for i, a in enumerate(actions)}
    n_theme = len(themes)
    n_actor = len(actors)
    n_action = len(actions)
    n_outcome = len(outcomes)
    n_conflict = 1 if has_conflict else 0
    causal_pairs: List[Tuple[int, int]] = [
        (action_pos[src], action_pos[act.id])
        for act in actions
        for src in act.caused_by
        if src in action_pos
    ]
    fanout = [0] * n_action
    parents = [0] * n_action
    for u, v in causal_pairs:
        fanout[u] += 1
        parents[v] += 1
    participation = [0] * n_actor
    for act in actions:
        for aid in set(act.actor_ids):
            if aid in actor_pos:
                participation[actor_pos[aid]] += 1
    prot = set(rec.central_conflict.protagonist_ids)
    ant = set(rec.central_conflict.antagonist_ids)
    raw: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    raw["theme_of"] = [(i, 0) for i in range(n_theme)]
    raw["actor_in"] = [(i, 0) for i in range(n_actor)]
    raw["conflict_of"] = [(0, 0)] if has_conflict else []
    raw["action_in"] = [(i, 0) for i in range(n_action)]
    raw["outcome_of"] = [(i, 0) for i in range(n_outcome)]
    raw["motivated_by"] = [(i, i) for i in range(n_actor)]
    for act in actions:
        for aid in act.actor_ids:
            if aid in actor_pos:
                raw["performs"].append((actor_pos[aid], action_pos[act.id]))
    for oi, out in enumerate(outcomes):
        for aid in out.actor_ids:
            if aid in actor_pos:
                raw["experiences"].append((actor_pos[aid], oi))
    if has_conflict:
        for aid in rec.central_conflict.protagonist_ids:
            if aid in actor_pos:
                raw["protagonist_of"].append((actor_pos[aid], 0))
        for aid in rec.central_conflict.antagonist_ids:
            if aid in actor_pos:
                raw["antagonist_of"].append((actor_pos[aid], 0))
    raw["precedes"] = [(i, i + 1) for i in range(max(0, n_action - 1))]
    raw["causes"] = list(causal_pairs)
    if has_conflict:
        for oi, out in enumerate(outcomes):
            if out.kind is OutcomeKind.RESOLUTION:
                raw["resolved_by"].append((0, oi))
    if not has_conflict or (not prot and (not ant)):
        drives_rule = "no_sides"
        selected: List[int] = []
    elif prot and ant:
        drives_rule = "both_sides"
        selected = [
            action_pos[act.id]
            for act in actions
            if set(act.actor_ids) & prot and set(act.actor_ids) & ant
        ]
    else:
        drives_rule = "single_side"
        side = prot or ant
        selected = [action_pos[act.id] for act in actions if set(act.actor_ids) & side]
    raw["drives"] = [(0, i) for i in selected]
    data = HeteroData()
    counts = {
        "story": 1,
        "theme": n_theme,
        "actor": n_actor,
        "motivation": n_actor,
        "conflict": n_conflict,
        "action": n_action,
        "outcome": n_outcome,
    }
    for nt in NODE_TYPES:
        data[nt].num_nodes = counts[nt]
    story_vec = _story_features(rec, fanout, parents, causal_pairs, participation, was_repaired)
    data["story"].x_scalar = torch.tensor([story_vec], dtype=torch.float)
    data["story"].plot_type = torch.tensor([PLOT_TYPE_INDEX[rec.plot_type.value]], dtype=torch.long)
    data["story"].narrative_status = torch.tensor(
        [NARRATIVE_STATUS_INDEX[rec.narrative_status.value]], dtype=torch.long
    )

    def _scalar(nt: str, rows: List[List[float]]) -> None:
        dim = SCALAR_DIMS[nt]
        if rows:
            data[nt].x_scalar = torch.tensor(rows, dtype=torch.float)
        else:
            data[nt].x_scalar = torch.empty((0, dim), dtype=torch.float)

    _scalar("theme", [[i / max(1, n_theme - 1) if n_theme > 1 else 0.0] for i in range(n_theme)])
    _scalar(
        "actor",
        [
            [
                i / max(1, n_actor - 1) if n_actor > 1 else 0.0,
                _clip01(participation[i] / n_action) if n_action else 0.0,
                1.0 if actors[i].id in prot else 0.0,
                1.0 if actors[i].id in ant else 0.0,
            ]
            for i in range(n_actor)
        ],
    )
    _scalar(
        "motivation", [[i / max(1, n_actor - 1) if n_actor > 1 else 0.0] for i in range(n_actor)]
    )
    _scalar(
        "conflict",
        (
            [[_clip01(len(prot) / max(1, n_actor)), _clip01(len(ant) / max(1, n_actor))]]
            if has_conflict
            else []
        ),
    )
    _scalar(
        "action",
        [
            [
                i / max(1, n_action - 1) if n_action > 1 else 0.0,
                1.0 if i == 0 else 0.0,
                1.0 if i == n_action - 1 else 0.0,
                _clip01(fanout[i] / CAP_FANOUT),
                _clip01(parents[i] / CAP_PARENTS),
            ]
            for i in range(n_action)
        ],
    )
    _scalar(
        "outcome", [[i / max(1, n_outcome - 1) if n_outcome > 1 else 0.0] for i in range(n_outcome)]
    )
    data["outcome"].kind = (
        torch.tensor([OUTCOME_KIND_INDEX[o.kind.value] for o in outcomes], dtype=torch.long)
        if outcomes
        else torch.empty((0,), dtype=torch.long)
    )
    edge_counts: Dict[str, int] = {}
    duplicate_edges: Dict[str, int] = {}
    degree: Dict[str, List[int]] = {nt: [0] * counts[nt] for nt in NODE_TYPES}
    for src, name, dst in FORWARD_RELATIONS:
        pairs, dups = _normalise_pairs(raw.get(name, []))
        data[src, name, dst].edge_index = _edge_tensor(pairs)
        edge_counts[name] = len(pairs)
        if dups:
            duplicate_edges[name] = dups
        for u, v in pairs:
            degree[src][u] += 1
            degree[dst][v] += 1
        if add_reverse:
            rsrc, rname, rdst = reverse_relation((src, name, dst))
            data[rsrc, rname, rdst].edge_index = _edge_tensor([(v, u) for u, v in pairs])
            edge_counts[rname] = len(pairs)
    isolated = {
        nt: sum((1 for d in degree[nt] if d == 0))
        for nt in NODE_TYPES
        if any((d == 0 for d in degree[nt]))
    }
    texts = GraphTexts(
        texts={
            "theme": list(themes),
            "actor": [a.role for a in actors],
            "motivation": [a.motivation for a in actors],
            "conflict": [rec.central_conflict.summary] if has_conflict else [],
            "action": [a.description for a in actions],
            "outcome": [o.description for o in outcomes],
        },
        aux_texts={"action": [a.function for a in actions]},
    )
    data.tale_id = meta.tale_id
    data.atu_type = meta.atu_type
    data.split = meta.split
    data.narrative_status_str = rec.narrative_status.value
    data.plot_type_str = rec.plot_type.value
    data.is_degenerate = degenerate
    data.was_repaired = was_repaired
    return GraphRecord(
        data=data,
        texts=texts,
        meta=meta,
        narrative_status=rec.narrative_status.value,
        plot_type=rec.plot_type.value,
        is_degenerate=degenerate,
        was_repaired=was_repaired,
        repairs=repairs,
        node_counts=counts,
        edge_counts=edge_counts,
        duplicate_edges=duplicate_edges,
        isolated_nodes=isolated,
        conflict_drives_rule=drives_rule,
    )


def build_from_raw(
    raw_record: Dict[str, Any],
    meta: TaleMeta,
    add_reverse: bool = True,
    report: Optional[BuildReport] = None,
    allow_repair: bool = True,
) -> Optional[GraphRecord]:
    if report is not None:
        report.note_record()
    try:
        if allow_repair:
            rec, rep = parse_and_repair(raw_record)
            repairs = dict(rep.repairs)
        else:
            rec = NarrativeExtraction.model_validate(raw_record)
            repairs = {}
    except Exception as exc:
        if report is not None:
            report.note_failure(meta.tale_id, type(exc).__name__ + ": " + str(exc)[:200])
        return None
    grec = build_graph(rec, meta, add_reverse=add_reverse, repairs=repairs)
    if report is not None:
        report.add(grec)
    return grec


def build_corpus(
    items: Iterable[Tuple[TaleMeta, Dict[str, Any]]],
    add_reverse: bool = True,
    allow_repair: bool = True,
) -> Tuple[List[GraphRecord], BuildReport]:
    out: List[GraphRecord] = []
    report = BuildReport()
    for meta, raw_record in items:
        grec = build_from_raw(
            raw_record, meta, add_reverse=add_reverse, report=report, allow_repair=allow_repair
        )
        if grec is not None:
            out.append(grec)
    return (out, report)


def _demo() -> int:
    from schemas import EXAMPLE_RECORD, NON_NARRATIVE_RECORD

    items = [
        (TaleMeta(tale_id="demo__full", atu_type="000", split="train"), EXAMPLE_RECORD),
        (TaleMeta(tale_id="demo__none", atu_type="000", split="train"), NON_NARRATIVE_RECORD),
    ]
    graphs, report = build_corpus(items)
    print(report.format_text())
    print()
    for g in graphs:
        print(
            g.meta.tale_id,
            g.narrative_status,
            g.plot_type,
            "nodes",
            sum(g.node_counts.values()),
            "forward edges",
            sum((g.edge_counts[n] for _, n, _ in FORWARD_RELATIONS)),
        )
    print()
    print("story feature dim", STORY_FEATURE_DIM)
    print("node types", len(NODE_TYPES), "edge types", len(EDGE_TYPES))
    return 0


if __name__ == "__main__":
    sys.exit(_demo())
