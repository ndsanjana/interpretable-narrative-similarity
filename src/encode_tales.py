from __future__ import annotations
import argparse
import json
import os
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tales", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="google/embeddinggemma-300m")
    ap.add_argument("--env", default=".env")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    tales = json.load(open(args.tales, "r", encoding="utf-8"))
    ids = [t["tale_id"] for t in tales]
    texts = [t.get("text", "") for t in tales]
    from dotenv import load_dotenv

    load_dotenv(args.env, override=True)
    tok = os.getenv("HF_TOKEN")
    if tok:
        os.environ["HF_TOKEN"] = tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.encoder, device=args.device)
    prompts = getattr(model, "prompts", None) or {}
    pn = "STS" if "STS" in prompts else None
    kw = {
        "batch_size": args.batch_size,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": False,
    }
    if pn:
        kw["prompt_name"] = pn
    emb = model.encode(texts, **kw).astype(np.float32)
    np.savez_compressed(
        args.out,
        tale_ids=np.array(ids, dtype=object),
        emb=emb,
        encoder=np.array([args.encoder]),
        prompt_name=np.array([str(pn)]),
    )
    print("wrote %s: %d tales, dim %d, prompt %r" % (args.out, emb.shape[0], emb.shape[1], pn))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
