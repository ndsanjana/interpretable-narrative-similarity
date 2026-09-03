from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_NAMES = {"atu_union": "ATU-Union benchmark"}
STRATUM_NAMES = {
    "a_gemini": "text-embedding-mined",
    "b_judged": "LLM-judged",
    "c_composite": "component-space-mined",
    "d_r1": "motif-graded",
    "none": "not-hard-tier",
}
TIER_ORDER = ("hard", "medium", "easy")
STRATUM_ORDER = ("a_gemini", "b_judged", "c_composite", "d_r1")
STORY_FEATURE_IDX = (0, 4, 7, 9, 19)


def load_jsonl(path: str) -> List[dict]:
    import gzip

    op = gzip.open if path.endswith(".gz") else open
    out = []
    with op(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def l2norm(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        return mat / (np.linalg.norm(mat) + 1e-08)
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-08)


def tale_texts_from_triplet_files(jsonl_path: str, meta_path: str) -> Dict[str, str]:
    rows = load_jsonl(jsonl_path)
    meta = load_jsonl(meta_path)
    if len(rows) != len(meta):
        raise ValueError(
            "triplet file and meta sidecar differ in length: %d vs %d" % (len(rows), len(meta))
        )
    texts: Dict[str, str] = {}
    for r, m in zip(rows, meta):
        pos_text = r["text_a"] if r["text_a_is_closer"] else r["text_b"]
        neg_text = r["text_b"] if r["text_a_is_closer"] else r["text_a"]
        texts.setdefault(m["anchor_id"], r["anchor_text"])
        texts.setdefault(m["pos_id"], pos_text)
        texts.setdefault(m["neg_id"], neg_text)
    return texts


def tale_texts_from_json(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as fh:
        obj = json.load(fh)
    if isinstance(obj, dict):
        return {str(k): v if isinstance(v, str) else v["text"] for k, v in obj.items()}
    return {str(rec["tale_id"]): rec["text"] for rec in obj}


def graph_embeddings(
    ckpt_path: str,
    data_dir: str,
    device: str = "cpu",
    batch_size: int = 32,
    knockout_edges: Sequence[str] = (),
) -> Tuple[Dict[str, np.ndarray], dict]:
    from dataset import GraphCorpus
    from model import ModelConfig, NarrativeGNN

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = dict(ckpt["cfg"])
    cfg = ModelConfig(**cfg_dict)
    corpus = GraphCorpus(data_dir)
    cfg.text_dim = corpus.text_dim
    model = NarrativeGNN(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    if knockout_edges:
        drop = set(knockout_edges)
        kept = [
            et
            for et in model.edge_types
            if et[1] not in drop and et[1] not in {"rev_" + d for d in drop}
        ]
        removed = len(model.edge_types) - len(kept)
        if removed == 0:
            raise ValueError("knockout %r removed no edge type" % (knockout_edges,))
        model.edge_types = kept
    ids = corpus.tale_ids
    out = []
    with torch.no_grad():
        for i in range(0, len(ids), batch_size):
            batch = corpus.batch(ids[i : i + batch_size]).to(device)
            out.append(model(batch).detach().cpu().numpy())
    emb = np.concatenate(out, axis=0).astype(np.float32)
    story_scalar = {}
    for tale_id in ids:
        sc = corpus.get(tale_id)["story"].x_scalar
        story_scalar[tale_id] = np.asarray(sc.view(-1).numpy(), dtype=np.float32)
    info = {
        "ckpt": ckpt_path,
        "data_dir": data_dir,
        "n_tales": len(ids),
        "dim": int(emb.shape[1]),
        "encoder": corpus.encoder_name,
        "ckpt_epoch": ckpt.get("epoch"),
        "ckpt_val": ckpt.get("val"),
        "knockout_edges": list(knockout_edges),
        "story_scalar": story_scalar,
    }
    return ({t: emb[i] for i, t in enumerate(ids)}, info)


def save_vector_npz(path: str, vecs: Dict[str, np.ndarray]) -> None:
    ids = sorted(vecs)
    np.savez_compressed(
        path,
        tale_ids=np.array(ids, dtype=object),
        emb=np.stack([vecs[t] for t in ids]).astype(np.float32),
    )


def load_vector_npz(path: str) -> Dict[str, np.ndarray]:
    npz = np.load(path, allow_pickle=True)
    ids = [str(t) for t in npz["tale_ids"]]
    emb = np.asarray(npz["emb"], dtype=np.float32)
    return {t: emb[i] for i, t in enumerate(ids)}


def gemini_vectors(
    cache_path: str, texts: Dict[str, str]
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    want = {sha(t): tale_id for tale_id, t in texts.items()}
    got: Dict[str, np.ndarray] = {}
    with open(cache_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            tale_id = want.get(rec["h"])
            if tale_id is not None:
                got[tale_id] = np.asarray(rec["emb"], dtype=np.float32)
    missing = sorted((t for t in texts if t not in got))
    return (got, missing)


def embeddinggemma_vectors(
    texts: Dict[str, str],
    cache_path: Optional[str] = None,
    device: str = "cpu",
    model_name: str = "google/embeddinggemma-300m",
    prompt_name: str = "STS",
    batch_size: int = 8,
    max_seq_length: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    have: Dict[str, np.ndarray] = {}
    if cache_path and os.path.exists(cache_path):
        have = load_vector_npz(cache_path)
    todo = sorted((t for t in texts if t not in have))
    if todo:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device=device)
        if max_seq_length:
            model.max_seq_length = int(max_seq_length)
        vecs = model.encode(
            [texts[t] for t in todo],
            prompt_name=prompt_name,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for i, t in enumerate(todo):
            have[t] = np.asarray(vecs[i], dtype=np.float32)
        if cache_path:
            save_vector_npz(cache_path, have)
    return {t: have[t] for t in texts if t in have}


class PairTable:

    def __init__(
        self,
        rows: List[dict],
        gvec: Dict[str, np.ndarray],
        tvec: Dict[str, np.ndarray],
        story_scalar: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        gn = {k: l2norm(v) for k, v in gvec.items()}
        tn = {k: l2norm(v) for k, v in tvec.items()}
        keep, dropped = ([], {"graph": 0, "text": 0})
        for r in rows:
            ids = (r["anchor_id"], r["pos_id"], r["neg_id"])
            if not all((i in gn for i in ids)):
                dropped["graph"] += 1
                continue
            if not all((i in tn for i in ids)):
                dropped["text"] += 1
                continue
            keep.append(r)
        self.rows = keep
        self.dropped = dropped
        self.n = len(keep)
        self.tier = np.array([r.get("tier", "unknown") for r in keep], dtype=object)
        self.stratum = np.array([r.get("stratum", "none") for r in keep], dtype=object)
        self.anchor_id = np.array([r["anchor_id"] for r in keep], dtype=object)
        self.pos_id = np.array([r["pos_id"] for r in keep], dtype=object)
        self.neg_id = np.array([r["neg_id"] for r in keep], dtype=object)

        def cosines(cand_key: str, store: Dict[str, np.ndarray]) -> np.ndarray:
            return np.array(
                [float(store[r["anchor_id"]] @ store[r[cand_key]]) for r in keep], dtype=np.float32
            )

        self.cg_pos = cosines("pos_id", gn)
        self.cg_neg = cosines("neg_id", gn)
        self.ct_pos = cosines("pos_id", tn)
        self.ct_neg = cosines("neg_id", tn)
        self.story_scalar = story_scalar
        self.feat_pos = self._features(self.cg_pos, self.ct_pos, "pos_id")
        self.feat_neg = self._features(self.cg_neg, self.ct_neg, "neg_id")

    def _features(self, cg: np.ndarray, ct: np.ndarray, cand_key: str) -> np.ndarray:
        base = np.stack([cg, ct, cg - ct, np.abs(cg - ct)], axis=1)
        if not self.story_scalar:
            return base.astype(np.float32)
        idx = list(STORY_FEATURE_IDX)
        anc = np.stack([self.story_scalar[r["anchor_id"]][idx] for r in self.rows])
        cnd = np.stack([self.story_scalar[r[cand_key]][idx] for r in self.rows])
        return np.concatenate([base, anc, cnd], axis=1).astype(np.float32)

    @property
    def feature_dim(self) -> int:
        return int(self.feat_pos.shape[1])

    def margins(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self.cg_pos - self.cg_neg, self.ct_pos - self.ct_neg)


def breakdown(correct: np.ndarray, tier: np.ndarray, stratum: np.ndarray) -> Dict[str, dict]:
    correct = np.asarray(correct, dtype=bool)

    def cell(mask: np.ndarray) -> dict:
        n = int(mask.sum())
        if n == 0:
            return {"n": 0, "acc": None}
        return {"n": n, "acc": round(100.0 * float(correct[mask].sum()) / n, 2)}

    out = {"overall": cell(np.ones_like(correct, dtype=bool))}
    for t in TIER_ORDER:
        m = tier == t
        if m.any():
            out["tier_" + t] = cell(m)
    for t in sorted(set(tier.tolist())):
        if t not in TIER_ORDER and t != "unknown":
            out["tier_" + str(t)] = cell(tier == t)
    for s in STRATUM_ORDER:
        m = stratum == s
        if m.any():
            out["stratum_" + s] = cell(m)
    return out


def scalar_scores(table: PairTable, alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
    return (alpha * table.cg_pos + beta * table.ct_pos, alpha * table.cg_neg + beta * table.ct_neg)


def scalar_accuracy(dg: np.ndarray, dt: np.ndarray, w: float) -> float:
    m = w * dg + (1.0 - w) * dt
    return float((m > 0).mean())


def fit_scalar_exact(dg: np.ndarray, dt: np.ndarray) -> Dict[str, float]:
    dg = np.asarray(dg, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64)
    denom = dt - dg
    with np.errstate(divide="ignore", invalid="ignore"):
        breaks = np.where(np.abs(denom) > 1e-12, dt / denom, np.nan)
    breaks = breaks[np.isfinite(breaks)]
    breaks = breaks[(breaks > 0.0) & (breaks < 1.0)]
    grid = np.unique(np.concatenate([[0.0, 1.0], breaks]))
    cands = [0.0, 1.0, 0.5]
    cands += [0.5 * (grid[i] + grid[i + 1]) for i in range(len(grid) - 1)]
    cands = sorted(set((float(c) for c in cands if 0.0 <= c <= 1.0)))
    best_w, best_acc = (0.5, -1.0)
    for w in cands:
        acc = scalar_accuracy(dg, dt, w)
        if acc > best_acc + 1e-12 or (
            abs(acc - best_acc) <= 1e-12 and abs(w - 0.5) < abs(best_w - 0.5)
        ):
            best_w, best_acc = (w, acc)
    return {
        "alpha": round(best_w, 6),
        "beta": round(1.0 - best_w, 6),
        "val_acc": round(100.0 * best_acc, 2),
        "fit": "exact_breakpoint",
        "n_val": int(dg.shape[0]),
    }


def fit_scalar_logistic(
    dg: np.ndarray,
    dt: np.ndarray,
    temp: float = 0.02,
    steps: int = 2000,
    lr: float = 0.05,
    seed: int = 922,
) -> Dict[str, float]:
    torch.manual_seed(seed)
    g = torch.tensor(np.asarray(dg, dtype=np.float32))
    t = torch.tensor(np.asarray(dt, dtype=np.float32))
    raw = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=lr)
    for _ in range(steps):
        w = torch.sigmoid(raw)
        m = w * g + (1.0 - w) * t
        loss = F.softplus(-m / temp).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    w = float(torch.sigmoid(raw).detach())
    return {
        "alpha": round(w, 6),
        "beta": round(1.0 - w, 6),
        "val_acc": round(100.0 * scalar_accuracy(dg, dt, w), 2),
        "fit": "logistic_gd",
        "temp": temp,
        "steps": steps,
    }


class GateMLP(nn.Module):

    def __init__(self, in_dim: int, hidden: int = 16, bias_init: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        with torch.no_grad():
            self.net[-1].bias.fill_(bias_init)

    def gate(self, feats: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(feats).squeeze(-1))

    def forward(
        self, feats: torch.Tensor, cg: torch.Tensor, ct: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g = self.gate(feats)
        return (g * cg + (1.0 - g) * ct, g)


def _gate_tensors(table: PairTable) -> Dict[str, torch.Tensor]:
    return {
        "feat_pos": torch.tensor(table.feat_pos),
        "feat_neg": torch.tensor(table.feat_neg),
        "cg_pos": torch.tensor(table.cg_pos),
        "cg_neg": torch.tensor(table.cg_neg),
        "ct_pos": torch.tensor(table.ct_pos),
        "ct_neg": torch.tensor(table.ct_neg),
    }


def fit_gate(
    train: PairTable,
    val: PairTable,
    hidden: int = 16,
    epochs: int = 300,
    lr: float = 0.005,
    temp: float = 0.02,
    weight_decay: float = 0.0001,
    seed: int = 922,
    device: str = "cpu",
) -> Tuple[GateMLP, Dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GateMLP(train.feature_dim, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    tr = {k: v.to(device) for k, v in _gate_tensors(train).items()}
    va = {k: v.to(device) for k, v in _gate_tensors(val).items()}
    best_state, best_acc, best_epoch = (None, -1.0, -1)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        f_pos, _ = model(tr["feat_pos"], tr["cg_pos"], tr["ct_pos"])
        f_neg, _ = model(tr["feat_neg"], tr["cg_neg"], tr["ct_neg"])
        loss = F.softplus(-(f_pos - f_neg) / temp).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vp, gp = model(va["feat_pos"], va["cg_pos"], va["ct_pos"])
            vn, gn = model(va["feat_neg"], va["cg_neg"], va["ct_neg"])
            acc = float((vp > vn).float().mean())
            gate_mean = float(torch.cat([gp, gn]).mean())
        history.append(
            {
                "epoch": epoch,
                "loss": round(float(loss.detach()), 6),
                "val_acc": round(100.0 * acc, 2),
                "gate_mean": round(gate_mean, 4),
            }
        )
        if acc > best_acc + 1e-12:
            best_acc, best_epoch = (acc, epoch)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    info = {
        "hidden": hidden,
        "epochs": epochs,
        "lr": lr,
        "temp": temp,
        "weight_decay": weight_decay,
        "seed": seed,
        "feature_dim": train.feature_dim,
        "best_epoch": best_epoch,
        "val_acc": round(100.0 * best_acc, 2),
        "n_train": train.n,
        "n_val": val.n,
        "history_tail": history[-5:],
    }
    return (model, info)


@torch.no_grad()
def gate_scores(model: GateMLP, table: PairTable, device: str = "cpu") -> Dict[str, np.ndarray]:
    t = {k: v.to(device) for k, v in _gate_tensors(table).items()}
    f_pos, g_pos = model(t["feat_pos"], t["cg_pos"], t["ct_pos"])
    f_neg, g_neg = model(t["feat_neg"], t["cg_neg"], t["ct_neg"])
    return {
        "fused_pos": f_pos.cpu().numpy(),
        "fused_neg": f_neg.cpu().numpy(),
        "gate_pos": g_pos.cpu().numpy(),
        "gate_neg": g_neg.cpu().numpy(),
    }


def gate_stats(gate_pos: np.ndarray, gate_neg: np.ndarray) -> Dict[str, float]:
    g = np.concatenate([gate_pos, gate_neg])
    return {
        "n": int(g.size),
        "mean": round(float(g.mean()), 4),
        "sd": round(float(g.std()), 4),
        "min": round(float(g.min()), 4),
        "p10": round(float(np.percentile(g, 10)), 4),
        "median": round(float(np.median(g)), 4),
        "p90": round(float(np.percentile(g, 90)), 4),
        "max": round(float(g.max()), 4),
        "frac_graph_favoured": round(float((g > 0.5).mean()), 4),
        "frac_saturated_low": round(float((g < 0.05).mean()), 4),
        "frac_saturated_high": round(float((g > 0.95).mean()), 4),
    }


def decomposition_r2(fused: np.ndarray, cg: np.ndarray, ct: np.ndarray) -> Dict[str, float]:
    y = np.asarray(fused, dtype=np.float64)
    X = np.stack(
        [np.asarray(cg, dtype=np.float64), np.asarray(ct, dtype=np.float64), np.ones_like(y)],
        axis=1,
    )
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 if ss_tot <= 1e-30 else 1.0 - ss_res / ss_tot
    return {
        "r2": float(r2),
        "a_graph": float(coef[0]),
        "b_text": float(coef[1]),
        "intercept": float(coef[2]),
        "max_abs_residual": float(np.abs(y - pred).max()),
    }


def build_text_vectors(args, texts: Dict[str, str]) -> Tuple[Dict[str, np.ndarray], dict]:
    if args.text_npz:
        vec = load_vector_npz(args.text_npz)
        return ({t: vec[t] for t in texts if t in vec}, {"source": "npz", "path": args.text_npz})
    if args.text_source == "gemini":
        vec, missing = gemini_vectors(args.gemini_cache, texts)
        return (
            vec,
            {
                "source": "gemini-embedding-001 (cached, no API calls)",
                "cache": args.gemini_cache,
                "missing": len(missing),
                "missing_ids": missing[:10],
            },
        )
    vec = embeddinggemma_vectors(
        texts,
        cache_path=args.embgemma_cache,
        device=args.text_device,
        max_seq_length=args.max_seq_length,
    )
    return (
        vec,
        {
            "source": "google/embeddinggemma-300m (STS prompt)",
            "cache": args.embgemma_cache,
            "missing": len(texts) - len(vec),
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit the three Phase 3 fusion arms.")
    ap.add_argument("--ckpt", required=True, help="trained graph checkpoint")
    ap.add_argument("--data-dir", required=True, help="dir with graphs.pt + node_text_emb.npz")
    ap.add_argument("--train-meta", required=True, help="train triplet rows (ids + tier)")
    ap.add_argument("--val-meta", required=True, help="TYPE-DISJOINT val triplet rows")
    ap.add_argument("--train-texts", default=None, help="triplet jsonl carrying train texts")
    ap.add_argument("--val-texts", default=None, help="triplet jsonl carrying val texts")
    ap.add_argument("--tales-json", default=None, help="tale_id -> text json (preferred)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="fusion")
    ap.add_argument("--text-source", default="gemini", choices=["gemini", "embgemma"])
    ap.add_argument("--gemini-cache", default="cache/gemini_tale_cache.jsonl")
    ap.add_argument("--embgemma-cache", default="cache/embgemma_tale_cache.npz")
    ap.add_argument(
        "--text-npz", default=None, help="precomputed tale vectors, overrides --text-source"
    )
    ap.add_argument("--text-device", default="cpu")
    ap.add_argument("--max-seq-length", type=int, default=None)
    ap.add_argument("--graph-emb-npz", default=None, help="reuse precomputed graph vectors")
    ap.add_argument("--save-graph-emb", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--gate-hidden", type=int, default=16)
    ap.add_argument("--gate-epochs", type=int, default=300)
    ap.add_argument("--gate-lr", type=float, default=0.005)
    ap.add_argument("--gate-temp", type=float, default=0.02)
    ap.add_argument("--gate-features", default="cos_struct", choices=["cos", "cos_struct"])
    ap.add_argument("--seed", type=int, default=922)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    train_rows = load_jsonl(args.train_meta)
    val_rows = load_jsonl(args.val_meta)
    texts: Dict[str, str] = {}
    if args.tales_json:
        texts.update(tale_texts_from_json(args.tales_json))
    if args.train_texts:
        texts.update(tale_texts_from_triplet_files(args.train_texts, args.train_meta))
    if args.val_texts:
        texts.update(tale_texts_from_triplet_files(args.val_texts, args.val_meta))
    if not texts:
        raise SystemExit("no tale texts: pass --tales-json or --train-texts/--val-texts")
    if args.graph_emb_npz and os.path.exists(args.graph_emb_npz):
        gvec = load_vector_npz(args.graph_emb_npz)
        ginfo = {"ckpt": args.ckpt, "graph_emb_npz": args.graph_emb_npz, "story_scalar": {}}
    else:
        gvec, ginfo = graph_embeddings(
            args.ckpt, args.data_dir, device=args.device, batch_size=args.batch_size
        )
        if args.save_graph_emb:
            save_vector_npz(args.save_graph_emb, gvec)
    story = ginfo.get("story_scalar") if args.gate_features == "cos_struct" else None
    tvec, tinfo = build_text_vectors(args, texts)
    train = PairTable(train_rows, gvec, tvec, story_scalar=story or None)
    val = PairTable(val_rows, gvec, tvec, story_scalar=story or None)
    if train.n == 0 or val.n == 0:
        raise SystemExit(
            "empty pair table: train %d val %d (dropped %s / %s)"
            % (train.n, val.n, train.dropped, val.dropped)
        )
    dg_val, dt_val = val.margins()
    exact = fit_scalar_exact(dg_val, dt_val)
    logistic = fit_scalar_logistic(dg_val, dt_val, temp=args.gate_temp, seed=args.seed)
    gate_model, gate_info = fit_gate(
        train,
        val,
        hidden=args.gate_hidden,
        epochs=args.gate_epochs,
        lr=args.gate_lr,
        temp=args.gate_temp,
        seed=args.seed,
        device=args.device,
    )
    gate_path = os.path.join(args.out_dir, args.tag + "_gate.pt")
    torch.save(
        {
            "state_dict": gate_model.state_dict(),
            "in_dim": train.feature_dim,
            "hidden": args.gate_hidden,
            "features": args.gate_features,
        },
        gate_path,
    )
    gs = gate_scores(gate_model, val, device=args.device)
    gate_info["val_gate_stats"] = gate_stats(gs["gate_pos"], gs["gate_neg"])
    add_pos, add_neg = scalar_scores(val, 0.5, 0.5)
    sca_pos, sca_neg = scalar_scores(val, exact["alpha"], exact["beta"])
    decomp = {
        "additive_core": decomposition_r2(
            np.concatenate([add_pos, add_neg]),
            np.concatenate([val.cg_pos, val.cg_neg]),
            np.concatenate([val.ct_pos, val.ct_neg]),
        ),
        "learned_scalar": decomposition_r2(
            np.concatenate([sca_pos, sca_neg]),
            np.concatenate([val.cg_pos, val.cg_neg]),
            np.concatenate([val.ct_pos, val.ct_neg]),
        ),
        "gated": decomposition_r2(
            np.concatenate([gs["fused_pos"], gs["fused_neg"]]),
            np.concatenate([val.cg_pos, val.cg_neg]),
            np.concatenate([val.ct_pos, val.ct_neg]),
        ),
    }
    val_acc = {
        "graph_only": round(100.0 * float((val.cg_pos > val.cg_neg).mean()), 2),
        "text_only": round(100.0 * float((val.ct_pos > val.ct_neg).mean()), 2),
        "additive_core": round(100.0 * float((add_pos > add_neg).mean()), 2),
        "learned_scalar": exact["val_acc"],
        "gated": gate_info["val_acc"],
    }
    out = {
        "tag": args.tag,
        "arms": {
            "additive_core": {"alpha": 0.5, "beta": 0.5, "fit": "none"},
            "learned_scalar": exact,
            "learned_scalar_logistic": logistic,
            "gated": {k: v for k, v in gate_info.items() if k != "history_tail"},
        },
        "decomposition": decomp,
        "val_accuracy": val_acc,
        "gate_checkpoint": gate_path,
        "gate_features": args.gate_features,
        "graph_channel": {k: v for k, v in ginfo.items() if k != "story_scalar"},
        "text_channel": tinfo,
        "coverage": {
            "train_rows": len(train_rows),
            "train_kept": train.n,
            "train_dropped": train.dropped,
            "val_rows": len(val_rows),
            "val_kept": val.n,
            "val_dropped": val.dropped,
        },
        "wall_seconds": round(time.time() - t0, 1),
    }
    arms_path = os.path.join(args.out_dir, args.tag + "_arms.json")
    with open(arms_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    gates_path = os.path.join(args.out_dir, args.tag + "_val_gate_values.jsonl")
    with open(gates_path, "w", encoding="utf-8") as fh:
        for i in range(val.n):
            fh.write(
                json.dumps(
                    {
                        "anchor_id": val.anchor_id[i],
                        "pos_id": val.pos_id[i],
                        "neg_id": val.neg_id[i],
                        "tier": val.tier[i],
                        "stratum": val.stratum[i],
                        "gate_pos": round(float(gs["gate_pos"][i]), 6),
                        "gate_neg": round(float(gs["gate_neg"][i]), 6),
                        "cos_graph_pos": round(float(val.cg_pos[i]), 6),
                        "cos_graph_neg": round(float(val.cg_neg[i]), 6),
                        "cos_text_pos": round(float(val.ct_pos[i]), 6),
                        "cos_text_neg": round(float(val.ct_neg[i]), 6),
                    }
                )
                + "\n"
            )
    print("FUSION ARMS %s" % json.dumps(val_acc))
    print(
        "DECOMPOSITION R2 additive %.12f scalar %.12f gated %.6f"
        % (decomp["additive_core"]["r2"], decomp["learned_scalar"]["r2"], decomp["gated"]["r2"])
    )
    print("wrote %s and %s" % (arms_path, gates_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
