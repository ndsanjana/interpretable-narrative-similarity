from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fusion as fu
from fusion import (
    BENCHMARK_NAMES,
    STRATUM_NAMES,
    STRATUM_ORDER,
    TIER_ORDER,
    PairTable,
    breakdown,
    load_jsonl,
)

TOKEN_RE = re.compile("[a-z]+")


def jaccard_rows(rows: List[dict], texts: Dict[str, str]) -> Optional[np.ndarray]:
    tok = {}
    for tale_id, text in texts.items():
        tok[tale_id] = set(TOKEN_RE.findall(text.lower()))

    def jac(a: str, b: str) -> float:
        sa, sb = (tok.get(a), tok.get(b))
        if not sa or not sb:
            return 0.0
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / union if union else 0.0

    out = []
    for r in rows:
        out.append(jac(r["anchor_id"], r["pos_id"]) > jac(r["anchor_id"], r["neg_id"]))
    return np.array(out, dtype=bool)


def score_checkpoint(
    ckpt: str,
    args,
    rows: List[dict],
    texts: Dict[str, str],
    tvecs: Dict[str, Dict[str, np.ndarray]],
    arms: Optional[dict],
) -> List[dict]:
    gvec, ginfo = fu.graph_embeddings(
        ckpt,
        args.data_dir,
        device=args.device,
        batch_size=args.batch_size,
        knockout_edges=args.knockout_edges,
    )
    story = ginfo.get("story_scalar")
    seed = infer_seed(ckpt)
    out: List[dict] = []
    fusion_text = args.fusion_text
    table_by_text: Dict[str, PairTable] = {}
    for name, tvec in tvecs.items():
        table_by_text[name] = PairTable(rows, gvec, tvec, story_scalar=story)
    primary = table_by_text[fusion_text]
    base = {
        "ckpt": os.path.basename(ckpt),
        "seed": seed,
        "n_scored": primary.n,
        "dropped": primary.dropped,
        "knockout_edges": list(args.knockout_edges),
        "ckpt_val": ginfo.get("ckpt_val"),
        "ckpt_epoch": ginfo.get("ckpt_epoch"),
    }
    correct = primary.cg_pos > primary.cg_neg
    out.append(
        dict(
            base,
            scorer="graph-only",
            text_channel=None,
            breakdown=breakdown(correct, primary.tier, primary.stratum),
        )
    )
    for name, table in table_by_text.items():
        add_pos, add_neg = fu.scalar_scores(table, 0.5, 0.5)
        out.append(
            dict(
                base,
                scorer="fusion additive core",
                text_channel=name,
                alpha=0.5,
                beta=0.5,
                breakdown=breakdown(add_pos > add_neg, table.tier, table.stratum),
                decomposition=fu.decomposition_r2(
                    np.concatenate([add_pos, add_neg]),
                    np.concatenate([table.cg_pos, table.cg_neg]),
                    np.concatenate([table.ct_pos, table.ct_neg]),
                ),
            )
        )
    if arms:
        alpha = arms["arms"]["learned_scalar"]["alpha"]
        beta = arms["arms"]["learned_scalar"]["beta"]
        sca_pos, sca_neg = fu.scalar_scores(primary, alpha, beta)
        out.append(
            dict(
                base,
                scorer="fusion learned scalar",
                text_channel=fusion_text,
                alpha=alpha,
                beta=beta,
                breakdown=breakdown(sca_pos > sca_neg, primary.tier, primary.stratum),
                decomposition=fu.decomposition_r2(
                    np.concatenate([sca_pos, sca_neg]),
                    np.concatenate([primary.cg_pos, primary.cg_neg]),
                    np.concatenate([primary.ct_pos, primary.ct_neg]),
                ),
            )
        )
        gate_path = args.gate_ckpt or arms.get("gate_checkpoint")
        if gate_path and os.path.exists(gate_path):
            gck = torch.load(gate_path, map_location="cpu", weights_only=False)
            gate = fu.GateMLP(gck["in_dim"], hidden=gck["hidden"])
            gate.load_state_dict(gck["state_dict"])
            gate.eval()
            if primary.feature_dim != gck["in_dim"]:
                out.append(
                    dict(
                        base,
                        scorer="fusion gated",
                        text_channel=fusion_text,
                        error="gate feature dim %d != table %d"
                        % (gck["in_dim"], primary.feature_dim),
                    )
                )
            else:
                gs = fu.gate_scores(gate, primary)
                row = dict(
                    base,
                    scorer="fusion gated",
                    text_channel=fusion_text,
                    breakdown=breakdown(
                        gs["fused_pos"] > gs["fused_neg"], primary.tier, primary.stratum
                    ),
                    gate_stats=fu.gate_stats(gs["gate_pos"], gs["gate_neg"]),
                    decomposition=fu.decomposition_r2(
                        np.concatenate([gs["fused_pos"], gs["fused_neg"]]),
                        np.concatenate([primary.cg_pos, primary.cg_neg]),
                        np.concatenate([primary.ct_pos, primary.ct_neg]),
                    ),
                )
                out.append(row)
                if args.gate_values_out:
                    write_gate_values(args.gate_values_out, primary, gs)
    return out


def write_gate_values(path: str, table: PairTable, gs: Dict[str, np.ndarray]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(table.n):
            fh.write(
                json.dumps(
                    {
                        "anchor_id": table.anchor_id[i],
                        "pos_id": table.pos_id[i],
                        "neg_id": table.neg_id[i],
                        "tier": table.tier[i],
                        "stratum": table.stratum[i],
                        "stratum_name": STRATUM_NAMES.get(table.stratum[i], table.stratum[i]),
                        "gate_pos": round(float(gs["gate_pos"][i]), 6),
                        "gate_neg": round(float(gs["gate_neg"][i]), 6),
                        "correct": bool(gs["fused_pos"][i] > gs["fused_neg"][i]),
                        "cos_graph_pos": round(float(table.cg_pos[i]), 6),
                        "cos_graph_neg": round(float(table.cg_neg[i]), 6),
                        "cos_text_pos": round(float(table.ct_pos[i]), 6),
                        "cos_text_neg": round(float(table.ct_neg[i]), 6),
                    }
                )
                + "\n"
            )


def infer_seed(path: str) -> Optional[int]:
    m = re.search("seed(\\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


BREAKDOWN_KEYS = (
    ["overall"] + ["tier_" + t for t in TIER_ORDER] + ["stratum_" + s for s in STRATUM_ORDER]
)


def aggregate(rows: List[dict]) -> List[dict]:
    groups: Dict[Tuple, List[dict]] = {}
    for r in rows:
        if "breakdown" not in r:
            continue
        key = (r["scorer"], r.get("text_channel"), tuple(r.get("knockout_edges", [])))
        groups.setdefault(key, []).append(r)
    agg = []
    for (scorer, text_channel, knock), members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        cell = {}
        for k in BREAKDOWN_KEYS:
            vals = [
                m["breakdown"][k]["acc"]
                for m in members
                if k in m["breakdown"] and m["breakdown"][k]["acc"] is not None
            ]
            if not vals:
                continue
            cell[k] = {
                "mean": round(float(np.mean(vals)), 2),
                "sd": round(float(np.std(vals, ddof=1)), 2) if len(vals) > 1 else 0.0,
                "n_seeds": len(vals),
                "n": members[0]["breakdown"][k]["n"],
            }
        best = max(members, key=lambda m: m["breakdown"]["overall"]["acc"])
        agg.append(
            {
                "scorer": scorer,
                "text_channel": text_channel,
                "knockout_edges": list(knock),
                "seeds": [m.get("seed") for m in members],
                "n_seeds": len(members),
                "mean": cell,
                "best_seed": {
                    "seed": best.get("seed"),
                    "ckpt": best.get("ckpt"),
                    "overall": best["breakdown"]["overall"]["acc"],
                    "hard": best["breakdown"].get("tier_hard", {}).get("acc"),
                },
            }
        )
    return agg


def markdown_table(agg: List[dict], benchmark_key: str, note: str = "") -> str:
    bench = BENCHMARK_NAMES.get(benchmark_key, benchmark_key)
    head = [
        "| scorer | text channel | seeds | overall | hard | medium | easy | text-embedding-mined (a_gemini) | LLM-judged (b_judged) | component-space-mined (c_composite) | motif-graded (d_r1) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def fmt(cell: Optional[dict]) -> str:
        if not cell:
            return "-"
        if cell.get("n_seeds", 1) > 1:
            return "%.2f (sd %.2f)" % (cell["mean"], cell["sd"])
        return "%.2f" % cell["mean"]

    lines = []
    for row in agg:
        m = row["mean"]
        cells = [
            row["scorer"],
            row["text_channel"] or "-",
            str(row["n_seeds"]),
            fmt(m.get("overall")),
        ]
        cells += [fmt(m.get("tier_" + t)) for t in TIER_ORDER]
        cells += [fmt(m.get("stratum_" + s)) for s in STRATUM_ORDER]
        lines.append("| " + " | ".join(cells) + " |")
    body = "\n".join(head + lines)
    header = "Benchmark: %s\n\n" % bench
    if note:
        header += note.rstrip() + "\n\n"
    return header + body + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Canonical scoring for the new-version stack.")
    ap.add_argument(
        "--ckpt", action="append", default=[], help="trained checkpoint; repeat for seeds"
    )
    ap.add_argument("--ckpt-glob", default=None, help="glob of checkpoints (alternative to --ckpt)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--eval-meta", required=True, help="eval triplet rows: ids, tier, stratum")
    ap.add_argument("--eval-texts", default=None, help="eval triplet jsonl carrying the texts")
    ap.add_argument("--tales-json", default=None)
    ap.add_argument("--benchmark", default="atu_union", help="benchmark key for the report name")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="score")
    ap.add_argument("--fusion-arms", default=None, help="fusion arms json")
    ap.add_argument("--gate-ckpt", default=None)
    ap.add_argument("--gate-values-out", default=None)
    ap.add_argument("--fusion-text", default="gemini", help="text channel used inside fusion")
    ap.add_argument(
        "--text-sources",
        default="gemini,embgemma",
        help="comma list of text channels to score; names must include --fusion-text",
    )
    ap.add_argument("--gemini-cache", default="cache/gemini_tale_cache.jsonl")
    ap.add_argument("--embgemma-cache", default="cache/embgemma_tale_cache.npz")
    ap.add_argument(
        "--text-npz",
        default=None,
        help="extra text channel from an npz, registered under --text-npz-name",
    )
    ap.add_argument("--text-npz-name", default="extra")
    ap.add_argument("--text-device", default="cpu")
    ap.add_argument("--max-seq-length", type=int, default=None)
    ap.add_argument("--no-surface", action="store_true")
    ap.add_argument(
        "--knockout-edges",
        action="append",
        default=[],
        help="forward relation names removed at inference (causal knockout)",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    if args.ckpt_glob:
        import glob

        args.ckpt += sorted(glob.glob(args.ckpt_glob))
    if not args.ckpt:
        raise SystemExit("no checkpoints: pass --ckpt or --ckpt-glob")
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    rows = load_jsonl(args.eval_meta)
    texts: Dict[str, str] = {}
    if args.tales_json:
        texts.update(fu.tale_texts_from_json(args.tales_json))
    if args.eval_texts:
        texts.update(fu.tale_texts_from_triplet_files(args.eval_texts, args.eval_meta))
    if not texts:
        raise SystemExit("no tale texts: pass --tales-json or --eval-texts")
    wanted = [s.strip() for s in args.text_sources.split(",") if s.strip()]
    tvecs: Dict[str, Dict[str, np.ndarray]] = {}
    tinfo: Dict[str, dict] = {}
    for name in wanted:
        if name == "gemini":
            vec, missing = fu.gemini_vectors(args.gemini_cache, texts)
            tinfo[name] = {
                "source": "gemini-embedding-001 (cached, no API calls)",
                "missing": len(missing),
                "missing_ids": missing[:10],
            }
        elif name == "embgemma":
            vec = fu.embeddinggemma_vectors(
                texts,
                cache_path=args.embgemma_cache,
                device=args.text_device,
                max_seq_length=args.max_seq_length,
            )
            tinfo[name] = {
                "source": "google/embeddinggemma-300m (STS prompt), zero-shot",
                "missing": len(texts) - len(vec),
            }
        else:
            raise SystemExit("unknown text source %r" % name)
        if vec:
            tvecs[name] = vec
    if args.text_npz:
        vec = fu.load_vector_npz(args.text_npz)
        tvecs[args.text_npz_name] = {t: vec[t] for t in texts if t in vec}
        tinfo[args.text_npz_name] = {
            "source": "npz " + args.text_npz,
            "disclosure": "fine-tuned text channel unless stated otherwise; not comparable to zero-shot text baselines",
        }
    if args.fusion_text not in tvecs:
        raise SystemExit(
            "fusion text channel %r not available (have %s)" % (args.fusion_text, sorted(tvecs))
        )
    arms = None
    if args.fusion_arms and os.path.exists(args.fusion_arms):
        with open(args.fusion_arms, "r", encoding="utf-8") as fh:
            arms = json.load(fh)
    all_rows: List[dict] = []
    dummy_gvec = {t: np.ones(2, dtype=np.float32) for t in texts}
    for name, tvec in tvecs.items():
        tbl = PairTable(rows, dummy_gvec, tvec)
        correct = tbl.ct_pos > tbl.ct_neg
        all_rows.append(
            {
                "scorer": "text-only",
                "text_channel": name,
                "ckpt": None,
                "seed": None,
                "n_scored": tbl.n,
                "dropped": tbl.dropped,
                "knockout_edges": [],
                "breakdown": breakdown(correct, tbl.tier, tbl.stratum),
            }
        )
    if not args.no_surface:
        tbl = PairTable(rows, dummy_gvec, {t: np.ones(2, dtype=np.float32) for t in texts})
        jac = jaccard_rows(tbl.rows, texts)
        all_rows.append(
            {
                "scorer": "surface baseline (token Jaccard)",
                "text_channel": None,
                "ckpt": None,
                "seed": None,
                "n_scored": tbl.n,
                "dropped": tbl.dropped,
                "knockout_edges": [],
                "breakdown": breakdown(jac, tbl.tier, tbl.stratum),
            }
        )
    for ckpt in args.ckpt:
        all_rows += score_checkpoint(ckpt, args, rows, texts, tvecs, arms)
    agg = aggregate(all_rows)
    rows_path = os.path.join(args.out_dir, args.tag + "_rows.jsonl")
    with open(rows_path, "w", encoding="utf-8") as fh:
        for r in all_rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    agg_path = os.path.join(args.out_dir, args.tag + "_aggregate.json")
    with open(agg_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "benchmark": BENCHMARK_NAMES.get(args.benchmark, args.benchmark),
                "benchmark_key": args.benchmark,
                "stratum_names": STRATUM_NAMES,
                "eval_meta": args.eval_meta,
                "checkpoints": args.ckpt,
                "text_channels": tinfo,
                "fusion_arms": args.fusion_arms,
                "knockout_edges": list(args.knockout_edges),
                "aggregate": agg,
                "wall_seconds": round(time.time() - t0, 1),
            },
            fh,
            indent=1,
            sort_keys=True,
        )
    md_path = os.path.join(args.out_dir, args.tag + "_table.md")
    md = markdown_table(agg, args.benchmark, args.note)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)
    print("wrote %s, %s, %s" % (rows_path, agg_path, md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
