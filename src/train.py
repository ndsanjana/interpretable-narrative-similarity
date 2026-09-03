from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import GraphCorpus, TripletSet, load_jsonl, make_loader
from model import ModelConfig, NarrativeGNN, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def embed_all(model, corpus: GraphCorpus, device: str, batch_size: int = 64) -> torch.Tensor:
    model.eval()
    out = []
    ids = corpus.tale_ids
    for i in range(0, len(ids), batch_size):
        batch = corpus.batch(ids[i : i + batch_size]).to(device)
        out.append(model(batch).detach())
    model.train()
    return torch.cat(out, dim=0)


@torch.no_grad()
def triplet_accuracy(model, corpus: GraphCorpus, rows: List[dict], device: str) -> Dict[str, float]:
    emb = embed_all(model, corpus, device)
    idx = corpus.index
    hit = defaultdict(int)
    tot = defaultdict(int)
    for r in rows:
        if not all((r[k] in idx for k in ("anchor_id", "pos_id", "neg_id"))):
            continue
        a = emb[idx[r["anchor_id"]]]
        sp = float(a @ emb[idx[r["pos_id"]]])
        sn = float(a @ emb[idx[r["neg_id"]]])
        ok = 1 if sp > sn else 0
        hit["overall"] += ok
        tot["overall"] += 1
        hit[r.get("tier", "all")] += ok
        tot[r.get("tier", "all")] += 1
    return {k: round(100.0 * hit[k] / tot[k], 2) for k in tot if tot[k]}


def loss_fn(za, zp, zn, kind: str, tau: float, margin: float) -> torch.Tensor:
    cos_p = (za * zp).sum(-1)
    cos_n = (za * zn).sum(-1)
    if kind == "softmargin":
        return F.softplus(cos_n - cos_p + margin).mean()
    if kind == "infonce":
        cands = torch.cat([zp, zn], dim=0)
        logits = za @ cands.t() / tau
        target = torch.arange(za.shape[0], device=za.device)
        return F.cross_entropy(logits, target)
    raise ValueError("unknown loss %r" % kind)


def build_config(args) -> ModelConfig:
    return ModelConfig(
        text_dim=args.text_dim,
        hidden=args.hidden,
        out_dim=args.out_dim,
        layers=args.layers,
        conv=args.conv,
        heads=args.heads,
        dropout=args.dropout,
        residual=not args.no_residual,
        use_text=not args.no_text,
        drop_relations=tuple((x for x in args.drop_relations.split(",") if x.strip())),
        readout=args.readout,
        head_layers=args.head_layers,
        pool_residual=args.pool_residual,
        pool_residual_gate_init=args.pool_gate_init,
        readout_concat_input=args.concat_input,
        attn_uniform_init=args.attn_uniform_init,
        node_text_dropout=args.node_text_dropout,
        aggr=args.aggr,
        use_description=not args.no_description,
        use_function=not args.no_function,
        use_causal=not args.no_causal,
        use_reverse=not args.no_reverse,
        use_story_scalar=not args.no_story_scalar,
        use_story_interaction=not args.no_story_interaction,
        use_plot_type=not args.no_plot_type,
    )


def run(args) -> Dict:
    device = args.device
    set_seed(args.seed)
    corpus = GraphCorpus(args.data_dir)
    train_rows = load_jsonl(args.train)
    val_rows = load_jsonl(args.val)
    val_rows = [
        r for r in val_rows if all((r[k] in corpus for k in ("anchor_id", "pos_id", "neg_id")))
    ]
    if args.train_type_frac < 1.0:
        types = sorted({r["anchor_atu"] for r in train_rows})
        rng = random.Random(args.seed)
        rng.shuffle(types)
        keep_t = set(types[: max(1, int(round(len(types) * args.train_type_frac)))])
        train_rows = [
            r
            for r in train_rows
            if r["anchor_atu"] in keep_t and r["pos_atu"] in keep_t and (r["neg_atu"] in keep_t)
        ]
    if args.train_frac < 1.0:
        pairs = sorted({(r["anchor_id"], r["pos_id"]) for r in train_rows})
        rng = random.Random(args.seed)
        rng.shuffle(pairs)
        keep = set(pairs[: max(1, int(round(len(pairs) * args.train_frac)))])
        train_rows = [r for r in train_rows if (r["anchor_id"], r["pos_id"]) in keep]
    cfg = build_config(args)
    cfg.text_dim = corpus.text_dim
    train_set = TripletSet(corpus, train_rows, hard_resample=args.hard_resample)
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    loader = make_loader(train_set, args.batch_size, True, gen)
    model = NarrativeGNN(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = None
    if args.lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    elif args.lr_schedule == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    val0 = triplet_accuracy(model, corpus, val_rows, device)
    print("epoch 0 (untrained) val %s" % json.dumps(val0), flush=True)
    best = {"overall": -1.0}
    best_epoch = -1
    bad = 0
    history = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        if args.hard_resample and epoch > 1:
            train_set.resample(model, device)
        model.train()
        total, nb = (0.0, 0)
        for a_ids, p_ids, n_ids, _ in loader:
            ids = list(a_ids) + list(p_ids) + list(n_ids)
            batch = corpus.batch(ids).to(device)
            z = model(batch)
            b = len(a_ids)
            za, zp, zn = (z[:b], z[b : 2 * b], z[2 * b :])
            loss = loss_fn(za, zp, zn, args.loss, args.tau, args.margin)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            total += float(loss.detach())
            nb += 1
        val = triplet_accuracy(model, corpus, val_rows, device)
        if sched is not None:
            if args.lr_schedule == "plateau":
                sched.step(val["overall"])
            else:
                sched.step()
        history.append({"epoch": epoch, "loss": round(total / max(1, nb), 4), "val": val})
        print(
            "epoch %d loss %.4f val %s" % (epoch, total / max(1, nb), json.dumps(val)), flush=True
        )
        if val["overall"] > best["overall"] + 1e-09:
            best = val
            best_epoch = epoch
            bad = 0
            if args.save_checkpoint:
                torch.save(
                    {
                        "cfg": cfg.to_dict(),
                        "state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val": val,
                    },
                    args.save_checkpoint,
                )
        else:
            bad += 1
            if epoch >= args.min_epochs and bad >= args.patience:
                break
    train_acc = triplet_accuracy(model, corpus, train_set.rows[:1500], device)
    result = {
        "run_id": args.run_id,
        "group": args.group,
        "seed": args.seed,
        "loss": args.loss,
        "tau": args.tau,
        "margin": args.margin,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "lr_schedule": args.lr_schedule,
        "hard_resample": bool(args.hard_resample),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "val": best,
        "val_epoch0": val0,
        "val_last": history[-1]["val"] if history else {},
        "train_acc_sample": train_acc,
        "params": count_parameters(model),
        "pool_gate": float(model.pool_gate.detach()) if hasattr(model, "pool_gate") else None,
        "config": cfg.to_dict(),
        "n_train_triplets": len(train_set),
        "train_frac": args.train_frac,
        "train_type_frac": args.train_type_frac,
        "n_val_triplets": len(val_rows),
        "encoder": corpus.encoder_name,
        "wall_seconds": round(time.time() - t0, 1),
        "device": device,
        "host": os.uname().nodename if hasattr(os, "uname") else "",
        "history": history,
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--group", default="main")
    ap.add_argument("--results", default=None)
    ap.add_argument("--save-checkpoint", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=922)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--min-epochs", type=int, default=6)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--weight-decay", type=float, default=0.0001)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--lr-schedule", default="none", choices=["none", "cosine", "plateau"])
    ap.add_argument("--loss", default="softmargin", choices=["softmargin", "infonce"])
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--hard-resample", action="store_true")
    ap.add_argument("--train-frac", type=float, default=1.0)
    ap.add_argument("--train-type-frac", type=float, default=1.0)
    ap.add_argument("--text-dim", type=int, default=768)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--out-dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--conv", default="sage", choices=["sage", "gatv2"])
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument(
        "--readout", default="multi", choices=["multi", "story", "multi_sum", "multi_attn"]
    )
    ap.add_argument("--head-layers", type=int, default=1)
    ap.add_argument(
        "--pool-residual",
        action="store_true",
        help="final = pooled input node text + gate * GNN delta",
    )
    ap.add_argument("--pool-gate-init", type=float, default=0.0)
    ap.add_argument(
        "--concat-input",
        action="store_true",
        help="concatenate pooled input node text into the readout",
    )
    ap.add_argument("--attn-uniform-init", action="store_true")
    ap.add_argument("--node-text-dropout", type=float, default=0.0)
    ap.add_argument("--aggr", default="sum")
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--no-text", action="store_true")
    ap.add_argument(
        "--drop-relations",
        default="",
        help="comma list of relation NAMES to drop, e.g. theme_of,rev_theme_of",
    )
    ap.add_argument("--no-description", action="store_true")
    ap.add_argument("--no-function", action="store_true")
    ap.add_argument("--no-causal", action="store_true")
    ap.add_argument("--no-reverse", action="store_true")
    ap.add_argument("--no-story-scalar", action="store_true")
    ap.add_argument("--no-story-interaction", action="store_true")
    ap.add_argument("--no-plot-type", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    markers = os.path.join(args.out_dir, "markers")
    os.makedirs(markers, exist_ok=True)
    if args.results is None:
        args.results = os.path.join(args.out_dir, "results.jsonl")
    done = os.path.join(markers, args.run_id + ".done")
    failed = os.path.join(markers, args.run_id + ".failed")
    if os.path.exists(done):
        print("run %s already done, skipping" % args.run_id)
        return 0
    if os.path.exists(failed):
        os.remove(failed)
    try:
        result = run(args)
    except Exception:
        tb = traceback.format_exc()
        with open(failed, "w", encoding="utf-8") as fh:
            fh.write(tb)
        print(tb, flush=True)
        return 1
    with open(args.results, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(result) + "\n")
        fh.flush()
    with open(done, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"run_id": args.run_id, "val": result["val"]}) + "\n")
    print(
        "RESULT %s"
        % json.dumps(
            {k: result[k] for k in ("run_id", "val", "best_epoch", "epochs_run", "wall_seconds")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
