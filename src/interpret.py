from __future__ import annotations
import argparse
import json
import os
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


def decomposition_report(table: PairTable, alpha: float, beta: float) -> Dict:
    dg, dt = table.margins()
    ga, ta = (alpha * dg, beta * dt)
    fused = ga + ta
    graph_right = dg > 0
    text_right = dt > 0
    fused_right = fused > 0
    cells = {}
    for gname, gmask in (("graph_right", graph_right), ("graph_wrong", ~graph_right)):
        for tname, tmask in (("text_right", text_right), ("text_wrong", ~text_right)):
            m = gmask & tmask
            n = int(m.sum())
            cells["%s_%s" % (gname, tname)] = {
                "n": n,
                "share": round(100.0 * n / max(table.n, 1), 2),
                "fused_acc": round(100.0 * float(fused_right[m].mean()), 2) if n else None,
            }
    contrib = np.abs(ga) / (np.abs(ga) + np.abs(ta) + 1e-12)
    add_pos, add_neg = fu.scalar_scores(table, alpha, beta)
    r2 = fu.decomposition_r2(
        np.concatenate([add_pos, add_neg]),
        np.concatenate([table.cg_pos, table.cg_neg]),
        np.concatenate([table.ct_pos, table.ct_neg]),
    )
    return {
        "alpha": alpha,
        "beta": beta,
        "n": table.n,
        "identity_r2": r2,
        "accuracy": {
            "graph_only": round(100.0 * float(graph_right.mean()), 2),
            "text_only": round(100.0 * float(text_right.mean()), 2),
            "fused": round(100.0 * float(fused_right.mean()), 2),
        },
        "who_decides": cells,
        "channel_agreement": round(100.0 * float((graph_right == text_right).mean()), 2),
        "graph_contribution_share": {
            "mean": round(float(contrib.mean()), 4),
            "median": round(float(np.median(contrib)), 4),
            "p10": round(float(np.percentile(contrib, 10)), 4),
            "p90": round(float(np.percentile(contrib, 90)), 4),
        },
        "margin_scale": {
            "graph_mean_abs": round(float(np.abs(dg).mean()), 6),
            "text_mean_abs": round(float(np.abs(dt).mean()), 6),
        },
    }


def decomposition_by_stratum(table: PairTable, alpha: float, beta: float) -> Dict:
    dg, dt = table.margins()
    fused = alpha * dg + beta * dt
    out = {}
    groups = [("overall", np.ones(table.n, dtype=bool))]
    groups += [("tier_" + t, table.tier == t) for t in TIER_ORDER]
    groups += [("stratum_" + s, table.stratum == s) for s in STRATUM_ORDER]
    for name, mask in groups:
        n = int(mask.sum())
        if n == 0:
            continue
        gr, tr, fr = (dg[mask] > 0, dt[mask] > 0, fused[mask] > 0)
        rescued = int((~gr & tr & fr).sum() + (gr & ~tr & fr).sum())
        lost = int((gr & ~tr & ~fr).sum() + (~gr & tr & ~fr).sum())
        out[name] = {
            "n": n,
            "name": (
                STRATUM_NAMES.get(name.replace("stratum_", ""), None)
                if name.startswith("stratum_")
                else None
            ),
            "graph_only": round(100.0 * float(gr.mean()), 2),
            "text_only": round(100.0 * float(tr.mean()), 2),
            "fused": round(100.0 * float(fr.mean()), 2),
            "disagreement_share": round(100.0 * float((gr != tr).mean()), 2),
            "rescued_by_fusion": rescued,
            "lost_by_fusion": lost,
        }
    return out


def decision_space_angle(table: PairTable) -> Dict:
    dg, dt = table.margins()
    dg = dg.astype(np.float64)
    dt = dt.astype(np.float64)

    def ang(u: np.ndarray, v: np.ndarray) -> float:
        c = float(u @ v) / (float(np.linalg.norm(u)) * float(np.linalg.norm(v)) + 1e-12)
        return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

    return {
        "n": table.n,
        "angle_deg_uncentered": round(ang(dg, dt), 2),
        "angle_deg_centered": round(ang(dg - dg.mean(), dt - dt.mean()), 2),
        "pearson_r": round(float(np.corrcoef(dg, dt)[0, 1]), 4),
        "sign_agreement_pct": round(100.0 * float(((dg > 0) == (dt > 0)).mean()), 2),
    }


def _pca(mat: np.ndarray, k: int) -> np.ndarray:
    mat = mat - mat.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(mat, full_matrices=False)
    k = min(k, u.shape[1])
    return u[:, :k] * s[:k]


def representation_angles(
    gvec: Dict[str, np.ndarray], tvec: Dict[str, np.ndarray], k: int = 16, seed: int = 922
) -> Dict:
    ids = sorted(set(gvec) & set(tvec))
    if len(ids) < 4 * k:
        k = max(2, len(ids) // 4)
    G = fu.l2norm(np.stack([gvec[i] for i in ids])).astype(np.float64)
    T = fu.l2norm(np.stack([tvec[i] for i in ids])).astype(np.float64)
    Gk, Tk = (_pca(G, k), _pca(T, k))
    Qg, _ = np.linalg.qr(Gk)
    Qt, _ = np.linalg.qr(Tk)
    sv = np.linalg.svd(Qg.T @ Qt, compute_uv=False)
    sv = np.clip(sv, -1.0, 1.0)
    angles = np.degrees(np.arccos(sv))
    types = sorted({i.split("__")[0] for i in ids})
    rng = np.random.RandomState(seed)
    rng.shuffle(types)
    held = set(types[: max(1, len(types) // 5)])
    tr = np.array([i.split("__")[0] not in held for i in ids])
    probe = {"n_train": int(tr.sum()), "n_test": int((~tr).sum()), "held_out_types": len(held)}
    if tr.sum() > k and (~tr).sum() > 1:
        X = np.concatenate([G[tr], np.ones((int(tr.sum()), 1))], axis=1)
        Xt = np.concatenate([G[~tr], np.ones((int((~tr).sum()), 1))], axis=1)
        lam = 0.01
        A = X.T @ X + lam * np.eye(X.shape[1])
        W = np.linalg.solve(A, X.T @ T[tr])
        pred = Xt @ W
        ss_res = float(((T[~tr] - pred) ** 2).sum())
        ss_tot = float(((T[~tr] - T[tr].mean(axis=0, keepdims=True)) ** 2).sum())
        probe["r2_text_from_graph"] = round(1.0 - ss_res / max(ss_tot, 1e-30), 4)
        pn = fu.l2norm(pred)
        tn = fu.l2norm(T[~tr])
        probe["mean_cosine_pred_vs_true"] = round(float((pn * tn).sum(axis=1).mean()), 4)
    return {
        "n_tales": len(ids),
        "k_components": int(k),
        "canonical_correlations": [round(float(s), 4) for s in sv],
        "principal_angles_deg": [round(float(a), 2) for a in angles],
        "mean_principal_angle_deg": round(float(angles.mean()), 2),
        "smallest_principal_angle_deg": round(float(angles.min()), 2),
        "linear_probe": probe,
    }


def gate_analysis(table: PairTable, gs: Dict[str, np.ndarray]) -> Dict:
    g = np.concatenate([gs["gate_pos"], gs["gate_neg"]])
    tier2 = np.concatenate([table.tier, table.tier])
    strat2 = np.concatenate([table.stratum, table.stratum])
    correct = gs["fused_pos"] > gs["fused_neg"]
    dg, dt = table.margins()

    def stats(mask: np.ndarray) -> Optional[dict]:
        n = int(mask.sum())
        if n == 0:
            return None
        sel = g[mask]
        return {
            "n": n,
            "mean": round(float(sel.mean()), 4),
            "median": round(float(np.median(sel)), 4),
            "sd": round(float(sel.std()), 4),
            "frac_graph_favoured": round(float((sel > 0.5).mean()), 4),
        }

    by_tier = {t: stats(tier2 == t) for t in TIER_ORDER if (tier2 == t).any()}
    by_stratum = {}
    for s in STRATUM_ORDER:
        st = stats(strat2 == s)
        if st:
            st["name"] = STRATUM_NAMES.get(s, s)
            by_stratum[s] = st
    gate_pair = 0.5 * (gs["gate_pos"] + gs["gate_neg"])
    return {
        "overall": fu.gate_stats(gs["gate_pos"], gs["gate_neg"]),
        "by_tier": by_tier,
        "by_stratum": by_stratum,
        "by_correctness": {
            "correct": stats(np.concatenate([correct, correct])),
            "incorrect": stats(np.concatenate([~correct, ~correct])),
        },
        "gate_vs_margins": {
            "corr_gate_graph_margin": round(float(np.corrcoef(gate_pair, dg)[0, 1]), 4),
            "corr_gate_text_margin": round(float(np.corrcoef(gate_pair, dt)[0, 1]), 4),
            "corr_gate_margin_gap": round(float(np.corrcoef(gate_pair, dg - dt)[0, 1]), 4),
        },
        "interpretation_note": "gate above 0.5 means the pair is scored mostly by the graph channel; a gate that is flat across strata is evidence that the learned gate found no pair-level rule and the additive core is the honest description of the fusion",
    }


def score_variant(
    ckpt: str,
    data_dir: str,
    rows: List[dict],
    tvec: Dict[str, np.ndarray],
    device: str,
    batch_size: int,
    knockout: Sequence[str],
    alpha: float,
    beta: float,
) -> Dict:
    gvec, ginfo = fu.graph_embeddings(
        ckpt, data_dir, device=device, batch_size=batch_size, knockout_edges=knockout
    )
    tbl = PairTable(rows, gvec, tvec, story_scalar=ginfo.get("story_scalar"))
    fp, fn = fu.scalar_scores(tbl, alpha, beta)
    return {
        "ckpt": os.path.basename(ckpt),
        "knockout_edges": list(knockout),
        "n": tbl.n,
        "graph_only": breakdown(tbl.cg_pos > tbl.cg_neg, tbl.tier, tbl.stratum),
        "fused": breakdown(fp > fn, tbl.tier, tbl.stratum),
    }


def delta_table(reference: Dict, variants: List[Dict]) -> List[Dict]:
    out = []
    for v in variants:
        row = {
            "variant": v.get("label") or v["ckpt"],
            "knockout_edges": v["knockout_edges"],
            "n": v["n"],
        }
        for channel in ("graph_only", "fused"):
            for key in ("overall", "tier_hard"):
                ref = reference[channel].get(key, {}).get("acc")
                cur = v[channel].get(key, {}).get("acc")
                row["%s_%s" % (channel, key)] = cur
                row["%s_%s_delta" % (channel, key)] = (
                    None if ref is None or cur is None else round(cur - ref, 2)
                )
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 4 interpretability suite.")
    ap.add_argument("--ckpt", required=True, help="reference checkpoint")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--eval-meta", required=True)
    ap.add_argument("--eval-texts", default=None)
    ap.add_argument("--tales-json", default=None)
    ap.add_argument("--benchmark", default="atu_union")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="interpret")
    ap.add_argument("--fusion-arms", default=None)
    ap.add_argument("--gate-ckpt", default=None)
    ap.add_argument("--text-source", default="gemini", choices=["gemini", "embgemma"])
    ap.add_argument("--gemini-cache", default="cache/gemini_tale_cache.jsonl")
    ap.add_argument("--embgemma-cache", default="cache/embgemma_tale_cache.npz")
    ap.add_argument("--text-npz", default=None)
    ap.add_argument("--text-device", default="cpu")
    ap.add_argument("--max-seq-length", type=int, default=None)
    ap.add_argument(
        "--knockout", action="append", default=[], help="relation to knock out at inference; repeat"
    )
    ap.add_argument(
        "--ablation-ckpt",
        action="append",
        default=[],
        help="label=path of a retrained ablation checkpoint; repeat",
    )
    ap.add_argument("--cca-components", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=922)
    args = ap.parse_args()
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
    tvec, tinfo = fu.build_text_vectors(args, texts)
    gvec, ginfo = fu.graph_embeddings(
        args.ckpt, args.data_dir, device=args.device, batch_size=args.batch_size
    )
    story = ginfo.get("story_scalar")
    table = PairTable(rows, gvec, tvec, story_scalar=story)
    if table.n == 0:
        raise SystemExit("no scorable triplets (dropped %s)" % table.dropped)
    alpha, beta = (0.5, 0.5)
    arms = None
    if args.fusion_arms and os.path.exists(args.fusion_arms):
        with open(args.fusion_arms, "r", encoding="utf-8") as fh:
            arms = json.load(fh)
    report = {
        "benchmark": BENCHMARK_NAMES.get(args.benchmark, args.benchmark),
        "benchmark_key": args.benchmark,
        "reference_ckpt": args.ckpt,
        "n_triplets": table.n,
        "dropped": table.dropped,
        "text_channel": tinfo,
        "graph_channel": {k: v for k, v in ginfo.items() if k != "story_scalar"},
        "stratum_names": STRATUM_NAMES,
        "decomposition_additive_core": decomposition_report(table, 0.5, 0.5),
        "decomposition_by_stratum_additive_core": decomposition_by_stratum(table, 0.5, 0.5),
        "decision_space_angle": decision_space_angle(table),
        "representation_angles": representation_angles(
            gvec, tvec, k=args.cca_components, seed=args.seed
        ),
        "per_scorer_breakdown": {
            "graph_only": breakdown(table.cg_pos > table.cg_neg, table.tier, table.stratum),
            "text_only": breakdown(table.ct_pos > table.ct_neg, table.tier, table.stratum),
        },
    }
    add_pos, add_neg = fu.scalar_scores(table, 0.5, 0.5)
    report["per_scorer_breakdown"]["fusion_additive_core"] = breakdown(
        add_pos > add_neg, table.tier, table.stratum
    )
    if arms:
        alpha = arms["arms"]["learned_scalar"]["alpha"]
        beta = arms["arms"]["learned_scalar"]["beta"]
        sca_pos, sca_neg = fu.scalar_scores(table, alpha, beta)
        report["per_scorer_breakdown"]["fusion_learned_scalar"] = breakdown(
            sca_pos > sca_neg, table.tier, table.stratum
        )
        report["decomposition_learned_scalar"] = decomposition_report(table, alpha, beta)
        report["decomposition_by_stratum_learned_scalar"] = decomposition_by_stratum(
            table, alpha, beta
        )
        gate_path = args.gate_ckpt or arms.get("gate_checkpoint")
        if gate_path and os.path.exists(gate_path):
            gck = torch.load(gate_path, map_location="cpu", weights_only=False)
            if gck["in_dim"] != table.feature_dim:
                report["gate_analysis"] = {
                    "error": "gate expects %d features, table has %d"
                    % (gck["in_dim"], table.feature_dim)
                }
            else:
                gate = fu.GateMLP(gck["in_dim"], hidden=gck["hidden"])
                gate.load_state_dict(gck["state_dict"])
                gate.eval()
                gs = fu.gate_scores(gate, table)
                report["per_scorer_breakdown"]["fusion_gated"] = breakdown(
                    gs["fused_pos"] > gs["fused_neg"], table.tier, table.stratum
                )
                report["gate_analysis"] = gate_analysis(table, gs)
                report["gate_departure_from_additivity"] = fu.decomposition_r2(
                    np.concatenate([gs["fused_pos"], gs["fused_neg"]]),
                    np.concatenate([table.cg_pos, table.cg_neg]),
                    np.concatenate([table.ct_pos, table.ct_neg]),
                )
    ref_pos, ref_neg = fu.scalar_scores(table, alpha, beta)
    reference = {
        "graph_only": report["per_scorer_breakdown"]["graph_only"],
        "fused": breakdown(ref_pos > ref_neg, table.tier, table.stratum),
        "fusion_weights": {"alpha": alpha, "beta": beta},
    }
    variants = []
    for relation in args.knockout:
        v = score_variant(
            args.ckpt,
            args.data_dir,
            rows,
            tvec,
            args.device,
            args.batch_size,
            [relation],
            alpha,
            beta,
        )
        v["label"] = "inference knockout: %s" % relation
        variants.append(v)
    for spec in args.ablation_ckpt:
        if "=" not in spec:
            raise SystemExit("--ablation-ckpt takes label=path, got %r" % spec)
        label, path = spec.split("=", 1)
        v = score_variant(
            path, args.data_dir, rows, tvec, args.device, args.batch_size, [], alpha, beta
        )
        v["label"] = "retrained ablation: %s" % label
        variants.append(v)
    if variants:
        report["knockout"] = {
            "reference": reference,
            "variants": variants,
            "deltas": delta_table(reference, variants),
        }
    report["wall_seconds"] = round(time.time() - t0, 1)
    out_path = os.path.join(args.out_dir, args.tag + "_report.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
    md = render_markdown(report)
    md_path = os.path.join(args.out_dir, args.tag + "_report.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)
    print("wrote %s and %s" % (out_path, md_path))
    return 0


def render_markdown(report: Dict) -> str:
    lines = [
        "# Interpretability report",
        "",
        "Benchmark: %s" % report["benchmark"],
        "Reference checkpoint: %s" % os.path.basename(report["reference_ckpt"]),
        "Triplets scored: %d" % report["n_triplets"],
        "",
    ]
    dec = report["decomposition_additive_core"]
    lines += [
        "## Channel decomposition (additive core)",
        "",
        "Identity check: R^2 of the fused score on the two channel cosines = %.12f (max abs residual %.3e). The additive core is an identity, not a fit."
        % (dec["identity_r2"]["r2"], dec["identity_r2"]["max_abs_residual"]),
        "",
        "| cell | n | share pct | fused accuracy |",
        "|---|---|---|---|",
    ]
    for k, v in dec["who_decides"].items():
        lines.append(
            "| %s | %d | %.2f | %s |"
            % (
                k.replace("_", " "),
                v["n"],
                v["share"],
                "-" if v["fused_acc"] is None else "%.2f" % v["fused_acc"],
            )
        )
    lines += [
        "",
        "Accuracy: graph-only %.2f, text-only %.2f, fused %.2f."
        % (dec["accuracy"]["graph_only"], dec["accuracy"]["text_only"], dec["accuracy"]["fused"]),
        "Mean graph contribution share of the fused margin: %.4f."
        % dec["graph_contribution_share"]["mean"],
        "",
    ]
    ang = report["decision_space_angle"]
    rep = report["representation_angles"]
    lines += [
        "## Channel angles",
        "",
        "Decision space: the two per-triplet margin vectors sit at %.2f degrees (centered %.2f, Pearson r %.4f); the channels agree on the sign of the decision on %.2f percent of triplets."
        % (
            ang["angle_deg_uncentered"],
            ang["angle_deg_centered"],
            ang["pearson_r"],
            ang["sign_agreement_pct"],
        ),
        "",
        "Representation space over %d shared tales, %d components: mean principal angle %.2f degrees, smallest %.2f, top canonical correlation %.4f."
        % (
            rep["n_tales"],
            rep["k_components"],
            rep["mean_principal_angle_deg"],
            rep["smallest_principal_angle_deg"],
            rep["canonical_correlations"][0] if rep["canonical_correlations"] else float("nan"),
        ),
        "",
    ]
    probe = rep.get("linear_probe", {})
    if "r2_text_from_graph" in probe:
        lines += [
            "Type-disjoint linear probe (graph embedding -> text embedding): R^2 %.4f on %d held-out tales."
            % (probe["r2_text_from_graph"], probe["n_test"]),
            "",
        ]
    lines += [
        "## Per-scorer breakdown",
        "",
        "| scorer | overall | hard | medium | easy | text-embedding-mined | LLM-judged | component-space-mined | motif-graded |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, bd in report["per_scorer_breakdown"].items():
        cells = [name.replace("_", " ")]
        for key in (
            ["overall"]
            + ["tier_" + t for t in TIER_ORDER]
            + ["stratum_" + s for s in STRATUM_ORDER]
        ):
            acc = bd.get(key, {}).get("acc")
            cells.append("-" if acc is None else "%.2f" % acc)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    if "gate_analysis" in report and "error" not in report["gate_analysis"]:
        ga = report["gate_analysis"]
        lines += [
            "## Gate analysis",
            "",
            "Gate mean %.4f (sd %.4f); the graph channel is favoured on %.1f percent of scored pairs."
            % (
                ga["overall"]["mean"],
                ga["overall"]["sd"],
                100.0 * ga["overall"]["frac_graph_favoured"],
            ),
            "",
            "| stratum | n | gate mean | pct graph-favoured |",
            "|---|---|---|---|",
        ]
        for s, v in ga["by_stratum"].items():
            lines.append(
                "| %s (%s) | %d | %.4f | %.1f |"
                % (v["name"], s, v["n"], v["mean"], 100.0 * v["frac_graph_favoured"])
            )
        r2 = report.get("gate_departure_from_additivity", {}).get("r2")
        if r2 is not None:
            lines += [
                "",
                "Departure from additivity: the gated score regresses on the two cosines with R^2 %.6f, so %.2f percent of its variance is not expressible as fixed channel mixing."
                % (r2, 100.0 * (1.0 - r2)),
            ]
        lines.append("")
    if "knockout" in report:
        w = report["knockout"]["reference"].get("fusion_weights", {})
        lines += [
            "## Causal edge knockout",
            "",
            "Fused columns use alpha %.4f / beta %.4f on both the reference and every variant."
            % (w.get("alpha", 0.5), w.get("beta", 0.5)),
            "",
            "| variant | graph overall | delta | graph hard | delta | fused overall | delta |",
            "|---|---|---|---|---|---|---|",
        ]
        for d in report["knockout"]["deltas"]:
            lines.append(
                "| %s | %s | %s | %s | %s | %s | %s |"
                % (
                    d["variant"],
                    _f(d.get("graph_only_overall")),
                    _f(d.get("graph_only_overall_delta")),
                    _f(d.get("graph_only_tier_hard")),
                    _f(d.get("graph_only_tier_hard_delta")),
                    _f(d.get("fused_overall")),
                    _f(d.get("fused_overall_delta")),
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _f(x) -> str:
    return "-" if x is None else "%.2f" % x


if __name__ == "__main__":
    raise SystemExit(main())
