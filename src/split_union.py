from __future__ import annotations
import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List


def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-meta", required=True)
    ap.add_argument(
        "--r1-meta",
        required=True,
        help="round-1 train meta, the only file carrying division labels",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-types", type=int, default=20)
    ap.add_argument("--min-division", type=int, default=6)
    ap.add_argument("--seed", type=int, default=922)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    division: Dict[str, str] = {}
    chapter: Dict[str, str] = {}
    for r in load_jsonl(args.r1_meta):
        for t, c, d in (
            ("anchor_atu", "anchor_chapter", "anchor_division"),
            ("neg_atu", "neg_chapter", "neg_division"),
        ):
            division.setdefault(r[t], r.get(d) or "NA")
            chapter.setdefault(r[t], r.get(c) or "NA")
    rows = load_jsonl(args.train_meta)
    types = sorted(
        {r["anchor_atu"] for r in rows}
        | {r["neg_atu"] for r in rows}
        | {r["pos_atu"] for r in rows}
    )
    by_div = defaultdict(list)
    for t in types:
        by_div[division.get(t, "NA")].append(t)
    eligible = [d for d, ts in sorted(by_div.items()) if len(ts) >= args.min_division and d != "NA"]
    rng.shuffle(eligible)
    val_divisions, used_chapters, n = ([], set(), 0)
    for d in eligible:
        ch = chapter.get(by_div[d][0], "NA")
        if ch in used_chapters:
            continue
        val_divisions.append(d)
        used_chapters.add(ch)
        n += len(by_div[d])
        if n >= args.val_types:
            break
    val_types = {t for d in val_divisions for t in by_div[d]}
    train_rows, val_rows = ([], [])
    for r in rows:
        rec = {
            k: r[k] for k in ("anchor_id", "pos_id", "neg_id", "anchor_atu", "pos_atu", "neg_atu")
        }
        rec["tier"] = r.get("tier", "all")
        rec["stratum"] = r.get("stratum", "")
        sides = {
            "val" if t in val_types else "train"
            for t in (r["anchor_atu"], r["pos_atu"], r["neg_atu"])
        }
        if sides == {"val"}:
            val_rows.append(rec)
        elif sides == {"train"}:
            train_rows.append(rec)
    for name, rs in (("union_train.jsonl", train_rows), ("union_val.jsonl", val_rows)):
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as fh:
            for r in rs:
                fh.write(json.dumps(r) + "\n")
    report = {
        "seed": args.seed,
        "pool_types": len(types),
        "val_types": len(val_types),
        "val_divisions": sorted(val_divisions),
        "train_triplets": len(train_rows),
        "val_triplets": len(val_rows),
        "dropped_mixed": len(rows) - len(train_rows) - len(val_rows),
    }
    json.dump(
        report,
        open(os.path.join(args.out_dir, "union_split_report.json"), "w", encoding="utf-8"),
        indent=1,
        sort_keys=True,
    )
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
