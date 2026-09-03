from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

PROMPT_EXAMPLE_TALES = ("306__2", "750A__7")
BARS = [
    ("chain_collapse_pct", "below", 65.0, "actions whose parent set is exactly [prev]"),
    ("multi_parent_pct", "above", 15.0, "actions with more than one parent"),
    ("nonadjacent_links_per_action", "above", 0.25, "non-adjacent causal links per action"),
    ("recall_certain", "above", 0.4, "recall on certain gold links"),
    ("precision_lenient", "atleast", 0.75, "precision, certain plus plausible tier"),
]


def load_gold(path: str) -> Dict[str, Dict[str, Any]]:
    gold: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            certain: Set[Tuple[int, int]] = set()
            any_tier: Set[Tuple[int, int]] = set()
            for lk in r["gold_links"]:
                pair = (int(lk["child"]), int(lk["parent"]))
                any_tier.add(pair)
                if lk["certainty"] == "certain":
                    certain.add(pair)
            gold[r["tale_id"]] = {
                "n_actions": int(r["n_actions"]),
                "certain": certain,
                "any": any_tier,
                "extractor": [list(x) for x in r["extractor_caused_by"]],
            }
    return gold


def load_predictions(path: str) -> Dict[str, Dict[int, List[int]]]:
    preds: Dict[str, Dict[int, List[int]]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tid = row.get("tale_id")
            if not tid or not row.get("ok", True):
                continue
            if isinstance(row.get("parents"), dict):
                preds[tid] = {int(k): [int(x) for x in v] for k, v in row["parents"].items()}
            elif isinstance(row.get("record"), dict):
                actions = row["record"].get("course_of_action") or []
                preds[tid] = {
                    int(a["id"]): [int(x) for x in a.get("caused_by") or []] for a in actions
                }
    return preds


def predictions_from_gold(
    gold: Dict[str, Dict[str, Any]], which: str
) -> Dict[str, Dict[int, List[int]]]:
    out: Dict[str, Dict[int, List[int]]] = {}
    for tid, g in gold.items():
        n = g["n_actions"]
        if which == "extractor":
            out[tid] = {i + 1: list(ps) for i, ps in enumerate(g["extractor"])}
        else:
            parents: Dict[int, List[int]] = {i: [] for i in range(1, n + 1)}
            for child, parent in sorted(g["certain"]):
                parents[child].append(parent)
            out[tid] = parents
    return out


def score(
    gold: Dict[str, Dict[str, Any]],
    preds: Dict[str, Dict[int, List[int]]],
    skip: Tuple[str, ...] = (),
) -> Dict[str, Any]:
    n_tales = n_actions = 0
    n_pred_links = n_chain = n_multi = n_rootless = n_nonadj = 0
    tp_certain = tp_any = 0
    tp_certain_adj = tp_certain_nonadj = 0
    n_gold_certain = n_gold_any = 0
    n_gold_certain_adj = n_gold_certain_nonadj = 0
    missing: List[str] = []
    mismatched: List[str] = []
    per_tale: List[Dict[str, Any]] = []
    for tid, g in sorted(gold.items()):
        if tid in skip:
            continue
        p = preds.get(tid)
        if p is None:
            missing.append(tid)
            continue
        n = g["n_actions"]
        if sorted(p) != list(range(1, n + 1)):
            mismatched.append(tid)
            continue
        n_tales += 1
        n_actions += n
        gc, ga = (g["certain"], g["any"])
        n_gold_certain += len(gc)
        n_gold_any += len(ga)
        gc_adj = {(c, pa) for c, pa in gc if pa == c - 1}
        n_gold_certain_adj += len(gc_adj)
        n_gold_certain_nonadj += len(gc) - len(gc_adj)
        t_tp = t_pred = 0
        for aid in range(1, n + 1):
            ps = p[aid]
            n_pred_links += len(ps)
            t_pred += len(ps)
            if ps == [aid - 1]:
                n_chain += 1
            if len(ps) > 1:
                n_multi += 1
            if not ps:
                n_rootless += 1
            for parent in ps:
                if parent != aid - 1:
                    n_nonadj += 1
                pair = (aid, parent)
                if pair in gc:
                    tp_certain += 1
                    t_tp += 1
                    if parent == aid - 1:
                        tp_certain_adj += 1
                    else:
                        tp_certain_nonadj += 1
                if pair in ga:
                    tp_any += 1
        per_tale.append(
            {
                "tale_id": tid,
                "n_actions": n,
                "pred_links": t_pred,
                "tp_certain": t_tp,
                "gold_certain": len(gc),
            }
        )
    a = max(1, n_actions)
    pl = max(1, n_pred_links)
    return {
        "n_tales": n_tales,
        "n_actions": n_actions,
        "n_pred_links": n_pred_links,
        "links_per_action": round(n_pred_links / a, 4),
        "chain_collapse_pct": round(100.0 * n_chain / a, 2),
        "multi_parent_pct": round(100.0 * n_multi / a, 2),
        "rootless_pct": round(100.0 * n_rootless / a, 2),
        "nonadjacent_links_per_action": round(n_nonadj / a, 4),
        "nonadjacent_link_pct": round(100.0 * n_nonadj / pl, 2),
        "precision_certain": round(tp_certain / pl, 4),
        "precision_lenient": round(tp_any / pl, 4),
        "recall_certain": round(tp_certain / max(1, n_gold_certain), 4),
        "recall_lenient": round(tp_any / max(1, n_gold_any), 4),
        "recall_certain_adjacent": round(tp_certain_adj / max(1, n_gold_certain_adj), 4),
        "recall_certain_nonadjacent": round(tp_certain_nonadj / max(1, n_gold_certain_nonadj), 4),
        "f1_certain": round(2 * tp_certain / max(1, pl + n_gold_certain), 4),
        "n_gold_certain": n_gold_certain,
        "n_gold_any": n_gold_any,
        "n_gold_certain_nonadjacent": n_gold_certain_nonadj,
        "missing_tales": missing,
        "action_count_mismatch": mismatched,
        "per_tale": per_tale,
    }


def bar_results(m: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for key, sense, target, label in BARS:
        v = m[key]
        if sense == "below":
            ok = v < target
            txt = "< %.2f" % target
        elif sense == "above":
            ok = v > target
            txt = "> %.2f" % target
        else:
            ok = v >= target
            txt = ">= %.2f" % target
        out.append({"metric": key, "label": label, "value": v, "target": txt, "pass": ok})
    return out


def report(m: Dict[str, Any], title: str, fh=sys.stdout) -> bool:
    bars = bar_results(m)
    print("", file=fh)
    print("=== %s ===" % title, file=fh)
    print(
        "tales %d, actions %d, predicted links %d (%.3f per action)"
        % (m["n_tales"], m["n_actions"], m["n_pred_links"], m["links_per_action"]),
        file=fh,
    )
    if m["missing_tales"]:
        print(
            "MISSING predictions for %d tales: %s"
            % (len(m["missing_tales"]), ", ".join(m["missing_tales"])),
            file=fh,
        )
    if m["action_count_mismatch"]:
        print("ACTION COUNT MISMATCH, skipped: %s" % ", ".join(m["action_count_mismatch"]), file=fh)
    print("", file=fh)
    for b in bars:
        print(
            "  %-4s %-44s %8.4f  target %s"
            % ("PASS" if b["pass"] else "FAIL", b["label"], b["value"], b["target"]),
            file=fh,
        )
    print("", file=fh)
    print("  diagnostics (not bars):", file=fh)
    print(
        "    recall, certain NON-ADJACENT gold links   %.4f   (first pass 0.085, study target 0.40)"
        % m["recall_certain_nonadjacent"],
        file=fh,
    )
    print(
        "    recall, certain adjacent gold links       %.4f   (first pass 0.938)"
        % m["recall_certain_adjacent"],
        file=fh,
    )
    print(
        "    precision, certain tier only              %.4f   (first pass 0.883)"
        % m["precision_certain"],
        file=fh,
    )
    print("    recall, certain plus plausible            %.4f" % m["recall_lenient"], file=fh)
    print("    F1, certain tier                          %.4f" % m["f1_certain"], file=fh)
    print(
        "    actions with no parent                    %.2f pct   (gold 16.7 pct)"
        % m["rootless_pct"],
        file=fh,
    )
    print(
        "    non-adjacent share of predicted links     %.2f pct   (gold 37.6 pct)"
        % m["nonadjacent_link_pct"],
        file=fh,
    )
    all_pass = all((b["pass"] for b in bars))
    print("", file=fh)
    print(
        "  OVERALL: %s (%d of %d bars)"
        % ("PASS" if all_pass else "FAIL", sum((1 for b in bars if b["pass"])), len(bars)),
        file=fh,
    )
    return all_pass


def _selftest(gold_path: str) -> int:
    failures: List[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(
            ("  PASS  " if cond else "  FAIL  ") + name + ("  [" + detail + "]" if detail else "")
        )
        if not cond:
            failures.append(name)

    gold = load_gold(gold_path)
    print("1. gold file loads and matches the study's totals")
    n_actions = sum((g["n_actions"] for g in gold.values()))
    n_certain = sum((len(g["certain"]) for g in gold.values()))
    n_any = sum((len(g["any"]) for g in gold.values()))
    check("22 tales", len(gold) == 22, str(len(gold)))
    check("240 actions", n_actions == 240, str(n_actions))
    check("282 certain links", n_certain == 282, str(n_certain))
    check("334 certain plus plausible links", n_any == 334, str(n_any))
    print("2. gold certain links fed back as the prediction score perfectly")
    m = score(gold, predictions_from_gold(gold, "certain"))
    check("recall_certain == 1.0", m["recall_certain"] == 1.0, str(m["recall_certain"]))
    check("precision_certain == 1.0", m["precision_certain"] == 1.0, str(m["precision_certain"]))
    check("precision_lenient == 1.0", m["precision_lenient"] == 1.0, str(m["precision_lenient"]))
    check(
        "recall on non-adjacent certain links == 1.0",
        m["recall_certain_nonadjacent"] == 1.0,
        str(m["recall_certain_nonadjacent"]),
    )
    check("all five bars pass on the gold answer", all((b["pass"] for b in bar_results(m))))
    check(
        "gold chain collapse reproduces the study's 49.6 pct",
        abs(m["chain_collapse_pct"] - 49.6) < 0.5,
        str(m["chain_collapse_pct"]),
    )
    check(
        "gold multi-parent reproduces the study's 27.5 pct",
        abs(m["multi_parent_pct"] - 27.5) < 0.5,
        str(m["multi_parent_pct"]),
    )
    check(
        "gold non-adjacent links per action reproduces 0.44",
        abs(m["nonadjacent_links_per_action"] - 0.44) < 0.01,
        str(m["nonadjacent_links_per_action"]),
    )
    print("3. the first pass scores exactly what the study published")
    b = score(gold, predictions_from_gold(gold, "extractor"))
    check("197 extractor links", b["n_pred_links"] == 197, str(b["n_pred_links"]))
    check(
        "chain collapse 77.9 pct",
        abs(b["chain_collapse_pct"] - 77.9) < 0.5,
        str(b["chain_collapse_pct"]),
    )
    check("multi-parent 0.00 pct", b["multi_parent_pct"] == 0.0, str(b["multi_parent_pct"]))
    check(
        "precision certain 0.883",
        abs(b["precision_certain"] - 0.883) < 0.01,
        str(b["precision_certain"]),
    )
    check(
        "precision lenient 0.944",
        abs(b["precision_lenient"] - 0.944) < 0.01,
        str(b["precision_lenient"]),
    )
    check("recall certain 0.617", abs(b["recall_certain"] - 0.617) < 0.01, str(b["recall_certain"]))
    check(
        "recall on non-adjacent certain links 0.085",
        abs(b["recall_certain_nonadjacent"] - 0.085) < 0.01,
        str(b["recall_certain_nonadjacent"]),
    )
    check(
        "recall on adjacent certain links 0.938",
        abs(b["recall_certain_adjacent"] - 0.938) < 0.01,
        str(b["recall_certain_adjacent"]),
    )
    check("the first pass FAILS the acceptance bars", not all((x["pass"] for x in bar_results(b))))
    print("4. a missing tale and a wrong action count are reported, not scored")
    preds = predictions_from_gold(gold, "certain")
    victim = sorted(preds)[0]
    del preds[victim]
    other = sorted(preds)[0]
    preds[other] = dict(list(preds[other].items())[:-1])
    m2 = score(gold, preds)
    check("missing tale reported", m2["missing_tales"] == [victim])
    check("action count mismatch reported", m2["action_count_mismatch"] == [other])
    check("both tales excluded from the totals", m2["n_tales"] == 20)
    print("5. the prompt-example exclusion works")
    m3 = score(gold, predictions_from_gold(gold, "certain"), skip=PROMPT_EXAMPLE_TALES)
    check("two tales removed", m3["n_tales"] == 20)
    check("still perfect on the rest", m3["recall_certain"] == 1.0)
    print()
    if failures:
        print("SELFTEST FAILED: " + "; ".join(failures))
        return 1
    print("SELFTEST PASSED")
    return 0


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    default_gold = os.path.join(here, "..", "data", "causal_gold_annotations.jsonl")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=None, help="causal_pass.jsonl or an extraction jsonl")
    ap.add_argument("--gold", default=default_gold)
    ap.add_argument("--json-out", default=None)
    ap.add_argument(
        "--baseline",
        action="store_true",
        help="score the extractor_caused_by stored in the gold file",
    )
    ap.add_argument(
        "--gold-as-prediction",
        action="store_true",
        help="score the certain gold links against themselves",
    )
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if not os.path.exists(args.gold):
        print("gold file not found: %s" % args.gold)
        return 2
    if args.selftest:
        return _selftest(args.gold)
    gold = load_gold(args.gold)
    if args.baseline:
        preds = predictions_from_gold(gold, "extractor")
        title = "FIRST PASS (extractor_caused_by from the gold file)"
    elif args.gold_as_prediction:
        preds = predictions_from_gold(gold, "certain")
        title = "GOLD CERTAIN LINKS AS PREDICTION (must be perfect)"
    else:
        if not args.pred:
            ap.error(
                "--pred is required unless --baseline, --gold-as-prediction or --selftest is given"
            )
        preds = load_predictions(args.pred)
        title = "CAUSAL PASS: %s" % os.path.basename(args.pred)
    m = score(gold, preds)
    all_pass = report(m, title)
    m_ex = score(gold, preds, skip=PROMPT_EXAMPLE_TALES)
    report(
        m_ex, "same, excluding the two prompt-example tales (%s)" % ", ".join(PROMPT_EXAMPLE_TALES)
    )
    if args.json_out:
        json.dump(
            {
                "all_pass": all_pass,
                "bars": bar_results(m),
                "metrics": m,
                "metrics_excluding_prompt_examples": m_ex,
            },
            open(args.json_out, "w", encoding="utf-8"),
            indent=1,
            sort_keys=True,
        )
        print("\nwrote %s" % args.json_out)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
