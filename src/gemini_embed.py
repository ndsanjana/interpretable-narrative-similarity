import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from dotenv import load_dotenv

MODEL = "gemini-embedding-001"
DIM = 2048


def sha(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tales", required=True)
    ap.add_argument("--cache", default="cache/gemini_tale_cache.jsonl")
    ap.add_argument("--env", default=".env")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--give-up-after-s", type=int, default=1500)
    args = ap.parse_args()
    load_dotenv(args.env, override=True)
    from google import genai

    tales = json.load(open(args.tales, encoding="utf-8"))
    ordered = []
    seen = set()
    for t in tales:
        if t["text"] not in seen:
            seen.add(t["text"])
            ordered.append(t["text"])
    done = set()
    os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
    if os.path.exists(args.cache):
        with open(args.cache, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["h"])
    todo = [t for t in ordered if sha(t) not in done]
    print("unique texts %d, cached %d, todo %d" % (len(ordered), len(done), len(todo)), flush=True)
    if not todo:
        return 0
    fout = open(args.cache, "a", encoding="utf-8")
    lock = threading.Lock()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    progress = {"n": 0, "last": time.time()}

    def embed_one(text: str) -> bool:
        delay = 5.0
        for _ in range(9):
            try:
                r = client.models.embed_content(
                    model=MODEL, contents=text, config={"output_dimensionality": DIM}
                )
                v = np.asarray(r.embeddings[0].values, dtype=np.float32)
                with lock:
                    fout.write(json.dumps({"h": sha(text), "emb": v.tolist()}) + "\n")
                    fout.flush()
                    progress["n"] += 1
                    progress["last"] = time.time()
                    if progress["n"] % 50 == 0:
                        print("embedded %d/%d" % (progress["n"], len(todo)), flush=True)
                return True
            except Exception:
                if time.time() - progress["last"] > args.give_up_after_s:
                    return False
                time.sleep(delay)
                delay = min(delay * 1.8, 90)
        return False

    ok = 0
    with ThreadPoolExecutor(args.concurrency) as ex:
        for fut in as_completed([ex.submit(embed_one, t) for t in todo]):
            ok += bool(fut.result())
    fout.close()
    print("embedded %d/%d" % (ok, len(todo)), flush=True)
    return 0 if ok == len(todo) else 3


if __name__ == "__main__":
    sys.exit(main())
