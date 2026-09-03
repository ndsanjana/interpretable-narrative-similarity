from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _load_causal(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tid = row.get("tale_id")
            if tid:
                out[tid] = row
    return out


def _clean_parents(parents: List[int], child: int, n: int) -> Tuple[List[int], Dict[str, int]]:
    rep: Dict[str, int] = {}

    def note(k: str) -> None:
        rep[k] = rep.get(k, 0) + 1

    kept: List[int] = []
    for p in parents:
        try:
            pid = int(p)
        except (TypeError, ValueError):
            note("malformed_parent")
            continue
        if not 1 <= pid <= n:
            note("unknown_parent")
            continue
        if pid >= child:
            note("self_or_forward_parent")
            continue
        if pid in kept:
            note("duplicate_parent")
            continue
        kept.append(pid)
    return (sorted(kept), rep)


def _structure(records: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    n_actions = n_links = n_chain = n_multi = n_root = n_nonadj = 0
    for rec in records:
        for a in rec.get("course_of_action") or []:
            aid = int(a["id"])
            ps = [int(x) for x in a.get(field) or []]
            n_actions += 1
            n_links += len(ps)
            if ps == [aid - 1]:
                n_chain += 1
            if len(ps) > 1:
                n_multi += 1
            if not ps:
                n_root += 1
            n_nonadj += sum((1 for p in ps if p != aid - 1))
    a = max(1, n_actions)
    return {
        "n_actions": n_actions,
        "n_causal_links": n_links,
        "links_per_action": round(n_links / a, 4),
        "chain_collapse_pct": round(100.0 * n_chain / a, 2),
        "multi_parent_pct": round(100.0 * n_multi / a, 2),
        "rootless_pct": round(100.0 * n_root / a, 2),
        "nonadjacent_links_per_action": round(n_nonadj / a, 4),
    }


def merge(extractions: str, causal: str, out: str, validate: bool = True) -> Dict[str, Any]:
    parse_and_repair = None
    if validate:
        try:
            from schemas import parse_and_repair
        except Exception as exc:
            print(
                "WARNING: schemas not importable (%s); skipping validation" % str(exc)[:120],
                flush=True,
            )
            parse_and_repair = None
    causal_rows = _load_causal(causal)
    stats: Dict[str, Any] = {
        "n_rows": 0,
        "n_ok_rows": 0,
        "n_merged": 0,
        "n_no_causal_row": 0,
        "n_causal_failed": 0,
        "n_action_count_mismatch": 0,
        "n_validation_failed": 0,
        "n_forward_parents_not_merged": 0,
        "repairs_by_kind": {},
        "unmerged_tale_ids": [],
    }
    before: List[Dict[str, Any]] = []
    after: List[Dict[str, Any]] = []
    with open(extractions, "r", encoding="utf-8") as fh, open(out, "w", encoding="utf-8") as ofh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            stats["n_rows"] += 1
            if not row.get("ok") or not isinstance(row.get("record"), dict):
                ofh.write(json.dumps(row) + "\n")
                continue
            stats["n_ok_rows"] += 1
            rec = row["record"]
            actions = rec.get("course_of_action") or []
            n = len(actions)
            before.append(json.loads(json.dumps(rec)))
            tid = row["tale_id"]
            cr = causal_rows.get(tid)
            reason: Optional[str] = None
            if cr is None:
                reason = "n_no_causal_row"
            elif not cr.get("ok") or not isinstance(cr.get("parents"), dict):
                reason = "n_causal_failed"
            elif sorted((int(k) for k in cr["parents"])) != list(range(1, n + 1)):
                reason = "n_action_count_mismatch"
            if reason is not None:
                stats[reason] += 1
                stats["unmerged_tale_ids"].append(tid)
                after.append(rec)
                ofh.write(json.dumps(row) + "\n")
                continue
            new_actions = json.loads(json.dumps(actions))
            for a in new_actions:
                aid = int(a["id"])
                ps, rep = _clean_parents(cr["parents"].get(str(aid), []), aid, n)
                a["caused_by"] = ps
                for k, v in rep.items():
                    stats["repairs_by_kind"][k] = stats["repairs_by_kind"].get(k, 0) + v
            stats["n_forward_parents_not_merged"] += sum(
                (len(v) for v in (cr.get("forward_parents") or {}).values())
            )
            new_rec = json.loads(json.dumps(rec))
            new_rec["course_of_action"] = new_actions
            if parse_and_repair is not None:
                try:
                    parse_and_repair(new_rec)
                except Exception as exc:
                    stats["n_validation_failed"] += 1
                    stats["unmerged_tale_ids"].append(tid)
                    print(
                        "VALIDATION FAILED for %s: %s; keeping the original"
                        % (tid, str(exc)[:200]),
                        flush=True,
                    )
                    after.append(rec)
                    ofh.write(json.dumps(row) + "\n")
                    continue
            new_row = dict(row)
            new_row["record"] = new_rec
            new_row["causal_pass"] = True
            stats["n_merged"] += 1
            after.append(new_rec)
            ofh.write(json.dumps(new_row) + "\n")
    stats["before"] = _structure(before, "caused_by")
    stats["after"] = _structure(after, "caused_by")
    stats["delta"] = {
        k: round(stats["after"][k] - stats["before"][k], 4)
        for k in (
            "links_per_action",
            "chain_collapse_pct",
            "multi_parent_pct",
            "rootless_pct",
            "nonadjacent_links_per_action",
        )
    }
    return stats


def _selftest() -> int:
    import tempfile

    failures: List[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(
            ("  PASS  " if cond else "  FAIL  ") + name + ("  [" + detail + "]" if detail else "")
        )
        if not cond:
            failures.append(name)

    try:
        from schemas import EXAMPLE_RECORD
    except Exception as exc:
        print("  FAIL  schemas importable  [%s]" % str(exc)[:120])
        return 1
    tmp = tempfile.mkdtemp(prefix="merge_causal_")
    ex = os.path.join(tmp, "extraction.jsonl")
    ca = os.path.join(tmp, "causal.jsonl")
    ou = os.path.join(tmp, "merged.jsonl")
    rec = json.loads(json.dumps(EXAMPLE_RECORD))
    with open(ex, "w", encoding="utf-8") as fh:
        fh.write(
            json.dumps({"tale_id": "t_merge", "atu_id": "1", "ok": True, "record": rec}) + "\n"
        )
        fh.write(
            json.dumps(
                {
                    "tale_id": "t_nocausal",
                    "atu_id": "2",
                    "ok": True,
                    "record": json.loads(json.dumps(rec)),
                }
            )
            + "\n"
        )
        fh.write(
            json.dumps(
                {
                    "tale_id": "t_mismatch",
                    "atu_id": "3",
                    "ok": True,
                    "record": json.loads(json.dumps(rec)),
                }
            )
            + "\n"
        )
        fh.write(json.dumps({"tale_id": "t_failed", "ok": False, "error": "no response"}) + "\n")
    with open(ca, "w", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "tale_id": "t_merge",
                    "ok": True,
                    "n_actions": 6,
                    "parents": {"1": [], "2": [], "3": [2], "4": [], "5": [3, 4], "6": [1, 5]},
                    "forward_parents": {"4": [5]},
                }
            )
            + "\n"
        )
        fh.write(
            json.dumps(
                {
                    "tale_id": "t_mismatch",
                    "ok": True,
                    "n_actions": 3,
                    "parents": {"1": [], "2": [1], "3": [1]},
                }
            )
            + "\n"
        )
    stats = merge(ex, ca, ou, validate=True)
    rows = [json.loads(l) for l in open(ou, "r", encoding="utf-8")]
    print("1. merge policy")
    check("all four rows written back", len(rows) == 4, str(len(rows)))
    check("one tale merged", stats["n_merged"] == 1, str(stats["n_merged"]))
    check("missing causal row left alone", stats["n_no_causal_row"] == 1)
    check("action count mismatch left alone", stats["n_action_count_mismatch"] == 1)
    check("failed extraction row passed through untouched", rows[3]["ok"] is False)
    print("2. the merged record")
    merged = [r for r in rows if r["tale_id"] == "t_merge"][0]
    cb = [a["caused_by"] for a in merged["record"]["course_of_action"]]
    check("caused_by replaced", cb == [[], [], [2], [], [3, 4], [1, 5]], str(cb))
    check("merge flagged on the row", merged.get("causal_pass") is True)
    check("forward parents counted, not merged", stats["n_forward_parents_not_merged"] == 1)
    check(
        "unmerged tale kept its original caused_by",
        [
            a["caused_by"]
            for a in [r for r in rows if r["tale_id"] == "t_nocausal"][0]["record"][
                "course_of_action"
            ]
        ]
        == [[], [], [2], [], [3, 4], [5]],
    )
    print("3. the merged file still validates through schemas")
    from schemas import parse_and_repair

    ok = True
    for r in rows:
        if r.get("ok"):
            try:
                parse_and_repair(r["record"])
            except Exception:
                ok = False
    check("every merged record parses", ok)
    check("no record needed validation rollback", stats["n_validation_failed"] == 0)
    print("4. invalid parents are dropped, not written")
    with open(ca, "w", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "tale_id": "t_merge",
                    "ok": True,
                    "n_actions": 6,
                    "parents": {"1": [], "2": [2], "3": [99], "4": [5], "5": [3, 3], "6": [1]},
                }
            )
            + "\n"
        )
    stats2 = merge(ex, ca, ou, validate=True)
    rows2 = [json.loads(l) for l in open(ou, "r", encoding="utf-8")]
    merged2 = [r for r in rows2 if r["tale_id"] == "t_merge"][0]
    cb2 = [a["caused_by"] for a in merged2["record"]["course_of_action"]]
    check(
        "self, forward, unknown and duplicate parents dropped",
        cb2 == [[], [], [], [], [3], [1]],
        str(cb2),
    )
    check(
        "drops were counted",
        stats2["repairs_by_kind"].get("self_or_forward_parent") == 2
        and stats2["repairs_by_kind"].get("unknown_parent") == 1
        and (stats2["repairs_by_kind"].get("duplicate_parent") == 1),
        json.dumps(stats2["repairs_by_kind"], sort_keys=True),
    )
    print("5. before and after statistics")
    check(
        "before and after both computed",
        "chain_collapse_pct" in stats["before"] and "chain_collapse_pct" in stats["after"],
    )
    check("delta reported", "nonadjacent_links_per_action" in stats["delta"])
    for f in (ex, ca, ou):
        try:
            os.remove(f)
        except OSError:
            pass
    try:
        os.rmdir(tmp)
    except OSError:
        pass
    print()
    if failures:
        print("SELFTEST FAILED: " + "; ".join(failures))
        return 1
    print("SELFTEST PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractions")
    ap.add_argument("--causal")
    ap.add_argument("--out")
    ap.add_argument("--stats", default=None)
    ap.add_argument("--no-validate", dest="validate", action="store_false", default=True)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    for req in ("extractions", "causal", "out"):
        if not getattr(args, req):
            ap.error("--%s is required unless --selftest is given" % req)
    if args.stats is None:
        args.stats = args.out.replace(".jsonl", "") + "_merge_stats.json"
    stats = merge(args.extractions, args.causal, args.out, args.validate)
    json.dump(stats, open(args.stats, "w", encoding="utf-8"), indent=1, sort_keys=True)
    printable = dict(stats)
    printable["unmerged_tale_ids"] = stats["unmerged_tale_ids"][:20] + (
        ["..."] if len(stats["unmerged_tale_ids"]) > 20 else []
    )
    print(json.dumps(printable, indent=1, sort_keys=True))
    print("\nwrote %s and %s" % (args.out, args.stats))
    return 0 if stats["n_merged"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
