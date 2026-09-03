from __future__ import annotations
import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schemas import NarrativeExtraction, parse_and_repair

FENCE = re.compile("```\\n(.*?)```", re.S)


def load_prompt(doc_path: str) -> Tuple[str, str]:
    text = open(doc_path, "r", encoding="utf-8").read()
    head = text.split("## 2. Schema summary")[0]
    blocks = FENCE.findall(head)
    if len(blocks) < 2:
        raise SystemExit("prompt doc: expected two fenced blocks, found %d" % len(blocks))
    system = blocks[0].strip()
    user = blocks[1]
    if "{story_text}" not in user:
        raise SystemExit("prompt doc: user block has no {story_text} placeholder")
    return (system, user)


def extract_json(blob: str) -> Optional[Dict[str, Any]]:
    blob = blob.strip()
    if not blob:
        return None
    try:
        return json.loads(blob)
    except Exception:
        pass
    start = blob.find("{")
    end = blob.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(blob[start : end + 1])
        except Exception:
            return None
    return None


class Runner:

    def __init__(self, args) -> None:
        self.args = args
        self.system, self.user_tmpl = load_prompt(args.prompt_doc)
        self.lock = threading.Lock()
        self.out_fh = open(args.out, "a", encoding="utf-8")
        self.done: Dict[str, bool] = {}
        self.stats = {
            "requested": 0,
            "ok": 0,
            "invalid_json": 0,
            "schema_fail": 0,
            "repaired": 0,
            "http_errors": 0,
            "retries": 0,
            "truncated_input": 0,
            "repairs_by_kind": {},
            "status_counts": {},
            "empty_content": 0,
            "mode": args.mode,
        }
        if os.path.exists(args.out):
            with open(args.out, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if rec.get("tale_id"):
                        self.done[rec["tale_id"]] = True
        self.schema = NarrativeExtraction.model_json_schema()

    def payload(self, story: str) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.args.model,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": self.user_tmpl.replace("{story_text}", story)},
            ],
            "temperature": 0.0,
            "max_tokens": self.args.max_tokens,
        }
        if self.stats["mode"] == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "narrative_extraction", "schema": self.schema},
            }
        elif self.stats["mode"] == "guided_json":
            body["guided_json"] = self.schema
        else:
            body["response_format"] = {"type": "json_object"}
        return body

    def call(self, story: str) -> Tuple[Optional[str], Optional[str]]:
        data = json.dumps(self.payload(story)).encode("utf-8")
        req = urllib.request.Request(
            self.args.url, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.args.timeout) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        msg = obj["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or "").strip()
        return (content, reasoning)

    def handle(self, tale: Dict[str, Any]) -> None:
        tale_id = tale["tale_id"]
        if self.done.get(tale_id):
            return
        words = tale["text"].split()
        truncated = False
        if len(words) > self.args.max_words:
            words = words[: self.args.max_words]
            truncated = True
        story = " ".join(words)
        raw: Optional[Dict[str, Any]] = None
        err = ""
        for attempt in range(self.args.retries + 1):
            try:
                content, reasoning = self.call(story)
            except Exception as exc:
                err = "%s: %s" % (type(exc).__name__, str(exc)[:200])
                with self.lock:
                    self.stats["http_errors"] += 1
                time.sleep(2.0 * (attempt + 1))
                continue
            if not content and reasoning:
                with self.lock:
                    self.stats["empty_content"] += 1
                content = reasoning
            raw = extract_json(content)
            if raw is not None:
                break
            err = "unparseable content"
            with self.lock:
                self.stats["invalid_json"] += 1
            if attempt < self.args.retries:
                with self.lock:
                    self.stats["retries"] += 1
        row: Dict[str, Any]
        if raw is None:
            row = {
                "tale_id": tale_id,
                "atu_id": tale.get("atu_id"),
                "ok": False,
                "error": err or "no response",
            }
        else:
            try:
                rec, rep = parse_and_repair(raw)
                repairs = dict(rep.repairs)
                row = {
                    "tale_id": tale_id,
                    "atu_id": tale.get("atu_id"),
                    "side": tale.get("side"),
                    "ok": True,
                    "record": json.loads(rec.model_dump_json()),
                    "repairs": repairs,
                    "truncated_input": truncated,
                }
                with self.lock:
                    self.stats["ok"] += 1
                    if repairs:
                        self.stats["repaired"] += 1
                    for k, v in repairs.items():
                        self.stats["repairs_by_kind"][k] = (
                            self.stats["repairs_by_kind"].get(k, 0) + v
                        )
                    st = row["record"].get("narrative_status")
                    self.stats["status_counts"][st] = self.stats["status_counts"].get(st, 0) + 1
            except Exception as exc:
                row = {
                    "tale_id": tale_id,
                    "atu_id": tale.get("atu_id"),
                    "ok": False,
                    "error": "schema: %s: %s" % (type(exc).__name__, str(exc)[:300]),
                    "raw": raw,
                }
                with self.lock:
                    self.stats["schema_fail"] += 1
        with self.lock:
            self.stats["requested"] += 1
            if truncated:
                self.stats["truncated_input"] += 1
            self.out_fh.write(json.dumps(row) + "\n")
            self.out_fh.flush()
            self.done[tale_id] = True
            n = self.stats["requested"]
            if n % 25 == 0:
                print(
                    "  %d done, ok %d, invalid %d, schema_fail %d"
                    % (n, self.stats["ok"], self.stats["invalid_json"], self.stats["schema_fail"]),
                    flush=True,
                )

    def probe(self, tales: List[Dict[str, Any]]) -> None:
        if self.args.mode != "auto":
            return
        sample = min(tales, key=lambda t: len(t["text"]))
        for mode in ("json_schema", "guided_json", "json_object"):
            self.stats["mode"] = mode
            try:
                content, reasoning = self.call(" ".join(sample["text"].split()[:400]))
            except Exception as exc:
                print("probe %s: transport error %s" % (mode, str(exc)[:160]), flush=True)
                continue
            raw = extract_json(content or reasoning)
            if raw is None:
                print("probe %s: no JSON" % mode, flush=True)
                continue
            try:
                parse_and_repair(raw)
            except Exception as exc:
                print("probe %s: schema fail %s" % (mode, str(exc)[:160]), flush=True)
                continue
            print("probe: structured-output mode = %s" % mode, flush=True)
            return
        raise SystemExit("no working structured-output mode")

    def run(self, tales: List[Dict[str, Any]]) -> None:
        todo = [t for t in tales if not self.done.get(t["tale_id"])]
        print(
            "tales %d, already done %d, to extract %d"
            % (len(tales), len(tales) - len(todo), len(todo)),
            flush=True,
        )
        self.probe(tales)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=self.args.concurrency) as pool:
            list(pool.map(self.handle, todo))
        dt = time.time() - t0
        self.stats["wall_seconds"] = round(dt, 1)
        self.stats["rate_per_hour"] = round(len(todo) / dt * 3600, 0) if dt > 0 else 0
        n = max(1, self.stats["ok"] + self.stats["schema_fail"])
        self.stats["validity_pct"] = round(100.0 * self.stats["ok"] / n, 2)
        self.stats["repair_rate_pct"] = round(
            100.0 * self.stats["repaired"] / max(1, self.stats["ok"]), 2
        )
        json.dump(
            self.stats, open(self.args.stats, "w", encoding="utf-8"), indent=1, sort_keys=True
        )
        print(json.dumps(self.stats, indent=1, sort_keys=True), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tales", required=True)
    ap.add_argument("--prompt-doc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--url", default="http://127.0.0.1:11501/v1/chat/completions")
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--max-words", type=int, default=6000)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument(
        "--mode", default="auto", choices=["auto", "json_schema", "guided_json", "json_object"]
    )
    args = ap.parse_args()
    if args.stats is None:
        args.stats = args.out.replace(".jsonl", "") + "_stats.json"
    tales = json.load(open(args.tales, "r", encoding="utf-8"))
    Runner(args).run(tales)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
