from __future__ import annotations
import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

FENCE = re.compile("```\\n(.*?)```", re.S)


def load_prompt(doc_path: str) -> Tuple[str, str]:
    text = open(doc_path, "r", encoding="utf-8").read()
    head = text.split("\n## 2. ")[0]
    blocks = FENCE.findall(head)
    if len(blocks) < 2:
        raise SystemExit(
            "prompt doc: expected two fenced blocks before section 2, found %d" % len(blocks)
        )
    system = blocks[0].strip()
    user = blocks[1]
    for ph in ("{story_text}", "{action_list}"):
        if ph not in user:
            raise SystemExit("prompt doc: user block has no %s placeholder" % ph)
    return (system, user)


def render_action_list(descriptions: Sequence[str]) -> str:
    return "\n".join(("%d. %s" % (i, d) for i, d in enumerate(descriptions, 1)))


def causal_schema(n_actions: int, pin_length: bool = True) -> Dict[str, Any]:
    links_items = {
        "type": "object",
        "properties": {
            "action_id": {"type": "integer"},
            "parents": {"type": "array", "items": {"type": "integer"}},
            "why": {"type": "string"},
        },
        "required": ["action_id", "parents", "why"],
        "additionalProperties": False,
    }
    links: Dict[str, Any] = {"type": "array", "items": links_items}
    if pin_length:
        links["minItems"] = n_actions
        links["maxItems"] = n_actions
    return {
        "type": "object",
        "properties": {
            "standing_conditions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action_id": {"type": "integer"},
                        "condition": {"type": "string"},
                    },
                    "required": ["action_id", "condition"],
                    "additionalProperties": False,
                },
            },
            "links": links,
        },
        "required": ["standing_conditions", "links"],
        "additionalProperties": False,
    }


def break_cycles(
    parents: Dict[int, List[int]], order: Sequence[int]
) -> Tuple[Dict[int, List[int]], int]:
    kept: Dict[int, List[int]] = {c: [] for c in order}
    dropped = 0

    def reachable_from(node: int) -> set:
        seen: set = set()
        stack = list(kept.get(node, []))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(kept.get(cur, []))
        return seen

    for child in order:
        for parent in parents.get(child, []):
            if parent == child:
                dropped += 1
                continue
            if parent in kept and child in reachable_from(parent):
                dropped += 1
                continue
            kept[child].append(parent)
    return (kept, dropped)


def validate_links(
    raw: Dict[str, Any], n_actions: int
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[str, int]]:
    rep: Dict[str, int] = {}

    def note(kind: str, n: int = 1) -> None:
        if n:
            rep[kind] = rep.get(kind, 0) + n

    valid = set(range(1, n_actions + 1))
    seen_children: List[int] = []
    parents: Dict[int, List[int]] = {}
    forward: Dict[int, List[int]] = {}
    for entry in raw.get("links") or []:
        if not isinstance(entry, dict):
            note("malformed_link_entry")
            continue
        try:
            child = int(entry.get("action_id"))
        except (TypeError, ValueError):
            note("malformed_action_id")
            continue
        if child not in valid:
            note("extra_action")
            continue
        if child in parents:
            note("duplicate_action")
            continue
        seen_children.append(child)
        back: List[int] = []
        fwd: List[int] = []
        for p in entry.get("parents") or []:
            try:
                pid = int(p)
            except (TypeError, ValueError):
                note("malformed_parent")
                continue
            if pid not in valid:
                note("unknown_parent")
                continue
            if pid == child:
                note("self_loop")
                continue
            if pid > child:
                note("forward_parent")
                if pid not in fwd:
                    fwd.append(pid)
                continue
            if pid in back:
                note("duplicate_parent")
                continue
            back.append(pid)
        parents[child] = back
        if fwd:
            forward[child] = fwd
    for aid in range(1, n_actions + 1):
        if aid not in parents:
            note("missing_action")
            parents[aid] = []
            seen_children.append(aid)
    order = seen_children
    parents, n_cycles = break_cycles(parents, order)
    note("cycle_break", n_cycles)
    parents = {aid: parents.get(aid, []) for aid in range(1, n_actions + 1)}
    return (parents, forward, rep)


def structure_stats(parents_by_tale: Dict[str, Dict[int, List[int]]]) -> Dict[str, Any]:
    n_actions = n_links = n_chain = n_multi = n_rootless = n_nonadj = 0
    for parents in parents_by_tale.values():
        for aid, ps in parents.items():
            n_actions += 1
            n_links += len(ps)
            if ps == [aid - 1]:
                n_chain += 1
            if len(ps) > 1:
                n_multi += 1
            if not ps:
                n_rootless += 1
            n_nonadj += sum((1 for p in ps if p != aid - 1))
    a = max(1, n_actions)
    return {
        "n_tales": len(parents_by_tale),
        "n_actions": n_actions,
        "n_links": n_links,
        "links_per_action": round(n_links / a, 4),
        "chain_collapse_pct": round(100.0 * n_chain / a, 2),
        "multi_parent_pct": round(100.0 * n_multi / a, 2),
        "rootless_pct": round(100.0 * n_rootless / a, 2),
        "nonadjacent_links_per_action": round(n_nonadj / a, 4),
    }


def extract_json(blob: str) -> Optional[Dict[str, Any]]:
    blob = (blob or "").strip()
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


def load_items(extractions: str, tales_path: str, only: Optional[set]) -> List[Dict[str, Any]]:
    tales = {t["tale_id"]: t for t in json.load(open(tales_path, "r", encoding="utf-8"))}
    items: List[Dict[str, Any]] = []
    with open(extractions, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("ok"):
                continue
            tid = row["tale_id"]
            if only is not None and tid not in only:
                continue
            rec = row.get("record") or {}
            actions = rec.get("course_of_action") or []
            if len(actions) < 2:
                continue
            tale = tales.get(tid)
            if tale is None:
                continue
            items.append(
                {
                    "tale_id": tid,
                    "atu_id": row.get("atu_id") or tale.get("atu_id"),
                    "side": row.get("side") or tale.get("side"),
                    "text": tale.get("text") or "",
                    "descriptions": [a.get("description", "") for a in actions],
                    "old_caused_by": [list(a.get("caused_by") or []) for a in actions],
                }
            )
    return items


class Runner:

    def __init__(self, args) -> None:
        self.args = args
        self.system, self.user_tmpl = load_prompt(args.prompt_doc)
        self.lock = threading.Lock()
        self.out_fh = open(args.out, "a", encoding="utf-8")
        self.done: Dict[str, bool] = {}
        self.parents_by_tale: Dict[str, Dict[int, List[int]]] = {}
        self.stats: Dict[str, Any] = {
            "requested": 0,
            "ok": 0,
            "invalid_json": 0,
            "http_errors": 0,
            "retries": 0,
            "truncated_input": 0,
            "empty_content": 0,
            "failed": 0,
            "repairs_by_kind": {},
            "mode": args.mode,
            "pin_length": True,
        }
        if os.path.exists(args.out):
            with open(args.out, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    tid = rec.get("tale_id")
                    if not tid:
                        continue
                    self.done[tid] = True
                    if rec.get("ok") and isinstance(rec.get("parents"), dict):
                        self.parents_by_tale[tid] = {
                            int(k): list(v) for k, v in rec["parents"].items()
                        }

    def payload(self, item: Dict[str, Any], story: str) -> Dict[str, Any]:
        n = len(item["descriptions"])
        user = self.user_tmpl.replace("{story_text}", story)
        user = user.replace("{action_list}", render_action_list(item["descriptions"]))
        body: Dict[str, Any] = {
            "model": self.args.model,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": self.args.max_tokens,
        }
        schema = causal_schema(n, self.stats["pin_length"])
        mode = self.stats["mode"]
        if mode == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "causal_pass", "schema": schema},
            }
        elif mode == "guided_json":
            body["guided_json"] = schema
        else:
            body["response_format"] = {"type": "json_object"}
        return body

    def call(self, item: Dict[str, Any], story: str) -> Tuple[str, str]:
        data = json.dumps(self.payload(item, story)).encode("utf-8")
        req = urllib.request.Request(
            self.args.url, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.args.timeout) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        msg = obj["choices"][0]["message"]
        return ((msg.get("content") or "").strip(), (msg.get("reasoning_content") or "").strip())

    def handle(self, item: Dict[str, Any]) -> None:
        tale_id = item["tale_id"]
        if self.done.get(tale_id):
            return
        words = item["text"].split()
        truncated = False
        if len(words) > self.args.max_words:
            words = words[: self.args.max_words]
            truncated = True
        story = " ".join(words)
        n = len(item["descriptions"])
        raw: Optional[Dict[str, Any]] = None
        err = ""
        for attempt in range(self.args.retries + 1):
            try:
                content, reasoning = self.call(item, story)
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
        if raw is None:
            row: Dict[str, Any] = {
                "tale_id": tale_id,
                "atu_id": item.get("atu_id"),
                "ok": False,
                "n_actions": n,
                "error": err or "no response",
            }
            with self.lock:
                self.stats["failed"] += 1
        else:
            parents, forward, repairs = validate_links(raw, n)
            conds = []
            for c in raw.get("standing_conditions") or []:
                if isinstance(c, dict):
                    try:
                        cid = int(c.get("action_id"))
                    except (TypeError, ValueError):
                        continue
                    if 1 <= cid <= n:
                        conds.append(
                            {"action_id": cid, "condition": str(c.get("condition", ""))[:200]}
                        )
            why = {}
            for e in raw.get("links") or []:
                if isinstance(e, dict) and e.get("why"):
                    try:
                        cid = int(e.get("action_id"))
                    except (TypeError, ValueError):
                        continue
                    if 1 <= cid <= n:
                        why[str(cid)] = str(e["why"])[:300]
            row = {
                "tale_id": tale_id,
                "atu_id": item.get("atu_id"),
                "side": item.get("side"),
                "ok": True,
                "n_actions": n,
                "parents": {str(k): v for k, v in sorted(parents.items())},
                "forward_parents": {str(k): v for k, v in sorted(forward.items())},
                "standing_conditions": conds,
                "why": why,
                "repairs": repairs,
                "truncated_input": truncated,
            }
            with self.lock:
                self.stats["ok"] += 1
                self.parents_by_tale[tale_id] = parents
                for k, v in repairs.items():
                    self.stats["repairs_by_kind"][k] = self.stats["repairs_by_kind"].get(k, 0) + v
        with self.lock:
            self.stats["requested"] += 1
            if truncated:
                self.stats["truncated_input"] += 1
            self.out_fh.write(json.dumps(row) + "\n")
            self.out_fh.flush()
            self.done[tale_id] = True
            m = self.stats["requested"]
            if m % 25 == 0:
                print(
                    "  %d done, ok %d, failed %d, invalid_json %d"
                    % (m, self.stats["ok"], self.stats["failed"], self.stats["invalid_json"]),
                    flush=True,
                )

    def probe(self, items: List[Dict[str, Any]]) -> None:
        if self.args.mode != "auto":
            return
        sample = min(items, key=lambda t: len(t["text"]))
        attempts = (
            ("json_schema", True),
            ("json_schema", False),
            ("guided_json", True),
            ("guided_json", False),
            ("json_object", False),
        )
        for mode, pin in attempts:
            self.stats["mode"] = mode
            self.stats["pin_length"] = pin
            try:
                content, reasoning = self.call(sample, " ".join(sample["text"].split()[:400]))
            except Exception as exc:
                print(
                    "probe %s pin_length=%s: transport error %s" % (mode, pin, str(exc)[:160]),
                    flush=True,
                )
                continue
            raw = extract_json(content or reasoning)
            if raw is None or "links" not in raw:
                print("probe %s pin_length=%s: no usable JSON" % (mode, pin), flush=True)
                continue
            print("probe: structured-output mode = %s, pin_length = %s" % (mode, pin), flush=True)
            return
        raise SystemExit("no working structured-output mode")

    def run(self, items: List[Dict[str, Any]]) -> int:
        todo = [t for t in items if not self.done.get(t["tale_id"])]
        print(
            "tales %d, already done %d, to run %d"
            % (len(items), len(items) - len(todo), len(todo)),
            flush=True,
        )
        if todo:
            self.probe(todo)
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=self.args.concurrency) as pool:
                list(pool.map(self.handle, todo))
            dt = time.time() - t0
            self.stats["wall_seconds"] = round(dt, 1)
            self.stats["rate_per_hour"] = round(len(todo) / dt * 3600, 0) if dt > 0 else 0
        n = max(1, self.stats["ok"] + self.stats["failed"])
        self.stats["validity_pct"] = round(100.0 * self.stats["ok"] / n, 2)
        self.stats["structure"] = structure_stats(self.parents_by_tale)
        json.dump(
            self.stats, open(self.args.stats, "w", encoding="utf-8"), indent=1, sort_keys=True
        )
        print(json.dumps(self.stats, indent=1, sort_keys=True), flush=True)
        return 0 if self.stats["failed"] == 0 else 1


def _selftest() -> int:
    failures: List[str] = []

    def check(name: str, cond: bool) -> None:
        print(("  PASS  " if cond else "  FAIL  ") + name)
        if not cond:
            failures.append(name)

    print("1. schema builds and pins the links length")
    s = causal_schema(7)
    check(
        "links minItems == maxItems == n",
        s["properties"]["links"]["minItems"] == 7 and s["properties"]["links"]["maxItems"] == 7,
    )
    check("required keys present", set(s["required"]) == {"standing_conditions", "links"})
    check("schema is JSON serialisable", isinstance(json.dumps(s), str))
    s2 = causal_schema(7, pin_length=False)
    check(
        "unpinned fallback schema drops the length bounds",
        "minItems" not in s2["properties"]["links"] and "maxItems" not in s2["properties"]["links"],
    )
    check(
        "unpinned fallback keeps the item shape",
        s2["properties"]["links"]["items"]["required"] == ["action_id", "parents", "why"],
    )
    print("2. validation of a hand-written good response")
    good = {
        "standing_conditions": [{"action_id": 1, "condition": "a bag that never fills"}],
        "links": [
            {"action_id": 1, "parents": [], "why": "opening gift"},
            {"action_id": 2, "parents": [1], "why": "equipped, so he travels"},
            {"action_id": 3, "parents": [], "why": "background state"},
            {"action_id": 4, "parents": [3], "why": "the offer answers the mystery"},
            {"action_id": 5, "parents": [1, 4], "why": "pelt enables, offer motivates"},
        ],
    }
    parents, forward, rep = validate_links(good, 5)
    check("no repairs on a clean response", rep == {})
    check("parents preserved", parents == {1: [], 2: [1], 3: [], 4: [3], 5: [1, 4]})
    check("no forward parents", forward == {})
    print("3. validation repairs a damaged response")
    bad = {
        "standing_conditions": [{"action_id": 99, "condition": "out of range"}],
        "links": [
            {"action_id": 1, "parents": [1], "why": "self loop"},
            {"action_id": 2, "parents": [1, 1, 9], "why": "dup and unknown"},
            {"action_id": 3, "parents": [5], "why": "forward"},
            {"action_id": 3, "parents": [2], "why": "duplicate entry"},
            {"action_id": 9, "parents": [], "why": "extra action"},
        ],
    }
    parents, forward, rep = validate_links(bad, 5)
    check("self loop dropped", rep.get("self_loop") == 1 and parents[1] == [])
    check("duplicate parent dropped", rep.get("duplicate_parent") == 1)
    check("unknown parent dropped", rep.get("unknown_parent") == 1)
    check(
        "forward parent recorded not dropped",
        rep.get("forward_parent") == 1 and forward == {3: [5]} and (parents[3] == []),
    )
    check("duplicate action entry dropped", rep.get("duplicate_action") == 1)
    check("extra action dropped", rep.get("extra_action") == 1)
    check(
        "missing actions filled",
        rep.get("missing_action") == 2 and parents[4] == [] and (parents[5] == []),
    )
    check("all ids present 1..n", sorted(parents) == [1, 2, 3, 4, 5])
    print("4. cycle breaking")
    cyc = {1: [3], 2: [1], 3: [2]}
    kept, dropped = break_cycles(cyc, [1, 2, 3])
    check("one edge dropped", dropped == 1)
    check(
        "the latest-added edge is the one dropped",
        kept[1] == [3] and kept[2] == [1] and (kept[3] == []),
    )
    check("result is acyclic", _is_acyclic(kept))
    cyc2 = {1: [5], 2: [1], 3: [2], 4: [3, 1], 5: [4]}
    kept2, dropped2 = break_cycles(cyc2, [1, 2, 3, 4, 5])
    check("long cycle broken", _is_acyclic(kept2) and dropped2 >= 1)
    check("long-range edge 4 -> 1 survives", 1 in kept2[4])
    kept3, dropped3 = break_cycles({1: [], 2: [2, 1]}, [1, 2])
    check("self loop counted and removed", dropped3 == 1 and kept3[2] == [1])
    dag = {1: [], 2: [1], 3: [1, 2], 4: [1]}
    kept4, dropped4 = break_cycles(dag, [1, 2, 3, 4])
    check("acyclic input untouched", dropped4 == 0 and kept4 == dag)
    print("5. structural statistics")
    st = structure_stats({"t": {1: [], 2: [1], 3: [1], 4: [1, 3], 5: [4]}})
    check("5 actions counted", st["n_actions"] == 5)
    check("5 links counted", st["n_links"] == 5)
    check("chain collapse 40 pct (actions 2 and 5)", abs(st["chain_collapse_pct"] - 40.0) < 1e-06)
    check("multi parent 20 pct", abs(st["multi_parent_pct"] - 20.0) < 1e-06)
    check("non-adjacent per action 0.4", abs(st["nonadjacent_links_per_action"] - 0.4) < 1e-06)
    print("6. prompt doc parses and both placeholders survive")
    here = os.path.dirname(os.path.abspath(__file__))
    doc = ""
    for cand in (
        os.path.join(here, "..", "prompts", "causal_pass_prompt.md"),
        os.path.join(here, "causal_pass_prompt.md"),
    ):
        if os.path.exists(cand):
            doc = cand
            break
    if doc:
        system, user = load_prompt(doc)
        check("system message non-empty", len(system) > 20)
        check(
            "user template has both placeholders",
            "{story_text}" in user and "{action_list}" in user,
        )
        check(
            "worked examples are in the user block",
            "WORKED EXAMPLE 1" in user and "WORKED EXAMPLE 2" in user,
        )
        rendered = user.replace("{story_text}", "TALE BODY").replace(
            "{action_list}", render_action_list(["a", "b"])
        )
        check(
            "rendering leaves no placeholder",
            "{story_text}" not in rendered and "{action_list}" not in rendered,
        )
    else:
        check("prompt doc found next to the script", False)
    print("7. action list rendering")
    check("1-based numbering", render_action_list(["x", "y"]) == "1. x\n2. y")
    print()
    if failures:
        print("SELFTEST FAILED: " + "; ".join(failures))
        return 1
    print("SELFTEST PASSED")
    return 0


def _is_acyclic(parents: Dict[int, List[int]]) -> bool:
    colour: Dict[int, int] = {}

    def visit(node: int) -> bool:
        state = colour.get(node, 0)
        if state == 1:
            return False
        if state == 2:
            return True
        colour[node] = 1
        for p in parents.get(node, []):
            if not visit(p):
                return False
        colour[node] = 2
        return True

    return all((visit(n) for n in parents))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractions")
    ap.add_argument("--tales")
    ap.add_argument("--prompt-doc")
    ap.add_argument("--out")
    ap.add_argument("--stats", default=None)
    ap.add_argument(
        "--tale-ids-file", default=None, help="one tale id per line; restrict the run to these"
    )
    ap.add_argument("--url", default="http://127.0.0.1:11501/v1/chat/completions")
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--max-words", type=int, default=6000)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument(
        "--mode", default="auto", choices=["auto", "json_schema", "guided_json", "json_object"]
    )
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    for req in ("extractions", "tales", "prompt_doc", "out"):
        if not getattr(args, req):
            ap.error("--%s is required unless --selftest is given" % req.replace("_", "-"))
    if args.stats is None:
        args.stats = args.out.replace(".jsonl", "") + "_stats.json"
    only = None
    if args.tale_ids_file:
        only = {ln.strip() for ln in open(args.tale_ids_file, "r", encoding="utf-8") if ln.strip()}
        print("restricted to %d tale ids" % len(only), flush=True)
    items = load_items(args.extractions, args.tales, only)
    if only is not None:
        missing = only - {i["tale_id"] for i in items}
        if missing:
            print(
                "WARNING: %d requested tale ids have no usable extraction: %s"
                % (len(missing), ", ".join(sorted(missing)[:10])),
                flush=True,
            )
    if not items:
        print("nothing to do", flush=True)
        return 2
    return Runner(args).run(items)


if __name__ == "__main__":
    raise SystemExit(main())
