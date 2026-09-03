from __future__ import annotations
import argparse
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_builder as gb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractions", required=True)
    ap.add_argument("--tales", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--encoder", default="google/embeddinggemma-300m")
    ap.add_argument("--env", default=".env")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--drop-degenerate", dest="drop_degenerate", action="store_true", default=True)
    ap.add_argument(
        "--keep-degenerate",
        dest="drop_degenerate",
        action="store_false",
        help="keep non_narrative graphs (1 Story node, all 28 edge types empty). Needed for SCORING corpora: a triplet whose tale has no graph cannot be scored at all, and silently skipping it inflates accuracy.",
    )
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tales = {t["tale_id"]: t for t in json.load(open(args.tales, "r", encoding="utf-8"))}
    items = []
    n_failed_extraction = 0
    with open(args.extractions, "r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if not row.get("ok"):
                n_failed_extraction += 1
                continue
            tale = tales.get(row["tale_id"])
            if tale is None:
                continue
            meta = gb.TaleMeta(
                tale_id=row["tale_id"],
                atu_type=tale.get("atu_id"),
                split=tale.get("side"),
                title=tale.get("title"),
                n_words=tale.get("n_words"),
                source="atu_trilogy",
            )
            items.append((meta, row["record"]))
    graphs, report = gb.build_corpus(items)
    print(report.format_text(), flush=True)
    kept = []
    n_degenerate_dropped = 0
    for g in graphs:
        if args.drop_degenerate and g.is_degenerate:
            n_degenerate_dropped += 1
            continue
        kept.append(g)
    graphs = kept
    uniq = {}
    order = []

    def note(s: str) -> None:
        if s not in uniq:
            uniq[s] = len(order)
            order.append(s)

    for g in graphs:
        for nt in gb.TEXT_NODE_TYPES:
            for s in g.texts.texts.get(nt, []):
                note(s)
        for s in g.texts.aux_texts.get("action", []):
            note(s)
    print("unique node texts: %d" % len(order), flush=True)
    from dotenv import load_dotenv

    load_dotenv(args.env, override=True)
    tok = os.getenv("HF_TOKEN")
    if tok:
        os.environ["HF_TOKEN"] = tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.encoder, device=args.device)
    prompts = getattr(model, "prompts", None) or {}
    prompt_name = None
    for cand in ("STS", "sts", "Retrieval-document"):
        if cand in prompts:
            prompt_name = cand
            break
    kw = {
        "batch_size": args.batch_size,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": False,
    }
    if prompt_name:
        kw["prompt_name"] = prompt_name
    print("encoder %s prompt=%r" % (args.encoder, prompt_name), flush=True)
    mat = model.encode(order, **kw).astype(np.float16)
    dim = int(mat.shape[1])
    print("encoded %s -> dim %d" % (mat.shape, dim), flush=True)
    tale_ids = [g.meta.tale_id for g in graphs]
    npz = {
        "tale_ids": np.array(tale_ids, dtype=object),
        "dim": np.array([dim]),
        "encoder": np.array([args.encoder]),
        "prompt_name": np.array([str(prompt_name)]),
    }
    channels = [(nt, lambda g, nt=nt: g.texts.texts.get(nt, [])) for nt in gb.TEXT_NODE_TYPES]
    channels.append(("action_fn", lambda g: g.texts.aux_texts.get("action", [])))
    for name, getter in channels:
        rows, offs = ([], [0])
        for g in graphs:
            texts = getter(g)
            for s in texts:
                rows.append(uniq[s])
            offs.append(len(rows))
        idx = np.array(rows, dtype=np.int64)
        block = mat[idx] if len(idx) else np.zeros((0, dim), dtype=np.float16)
        npz["emb_" + name] = block
        npz["off_" + name] = np.array(offs, dtype=np.int64)
    np.savez_compressed(os.path.join(args.out_dir, "node_text_emb.npz"), **npz)
    torch.save({g.meta.tale_id: g.data for g in graphs}, os.path.join(args.out_dir, "graphs.pt"))
    meta_out = {
        "n_graphs": len(graphs),
        "n_failed_extraction": n_failed_extraction,
        "n_degenerate_dropped": n_degenerate_dropped,
        "encoder": args.encoder,
        "prompt_name": str(prompt_name),
        "text_dim": dim,
        "unique_texts": len(order),
        "build_report": report.as_dict(),
        "tale_ids": tale_ids,
    }
    json.dump(
        meta_out,
        open(os.path.join(args.out_dir, "graph_data_meta.json"), "w", encoding="utf-8"),
        indent=1,
        sort_keys=True,
    )
    print("wrote %d graphs, text dim %d" % (len(graphs), dim), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
