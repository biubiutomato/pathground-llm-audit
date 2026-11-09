#!/usr/bin/env python3
"""Run PathCheck prompts against OpenAI Chat Completions. Key from env only."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = json.loads((ROOT / "results" / "prompts.json").read_text())
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

API_URL = "https://api.openai.com/v1/chat/completions"
MODEL = os.environ.get("PATHCHECK_MODEL", "gpt-4.1")
KEY = os.environ.get("OPENAI_API_KEY", "")
TEMPERATURE = 0.0
MAX_RETRIES = 5
WORKERS = 3
LOCK = threading.Lock()

SYSTEM = (
    "You are a molecular biologist. Return JSON only. "
    "Do not invent pathway names that do not exist in KEGG or Reactome. "
    "Only claim a pathway if multiple input genes participate in it."
)


def parse_pathways(text: str) -> list:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # isolate first JSON object
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        text = m.group(0)
    data = json.loads(text)
    paths = data.get("pathways") or data.get("pathway") or data
    if isinstance(paths, dict):
        paths = [paths]
    out = []
    for p in paths:
        if isinstance(p, str):
            out.append({"name": p, "reason": ""})
        elif isinstance(p, dict):
            name = p.get("name") or p.get("pathway") or p.get("title") or ""
            reason = p.get("reason") or p.get("explanation") or p.get("why") or ""
            if name:
                out.append({"name": str(name).strip(), "reason": str(reason).strip()})
    return out


def call(prompt: str) -> tuple[list, dict]:
    headers = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    body = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": 450,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(API_URL, headers=headers, json=body, timeout=40)
            if r.status_code == 429:
                wait = min(3 + attempt * 3, 20)
                print(f"429 backoff {wait}s", flush=True)
                time.sleep(wait)
                last_err = "429"
                continue
            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                last_err = f"{r.status_code} {r.text[:200]}"
                continue
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
            payload = r.json()
            text = payload["choices"][0]["message"]["content"]
            usage = payload.get("usage") or {}
            return parse_pathways(text), {
                "raw": text,
                "usage": usage,
                "model": payload.get("model", MODEL),
            }
        except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError) as e:
            last_err = str(e)
            time.sleep(2 ** attempt)
    raise RuntimeError(last_err or "unknown failure")


def main():
    if not KEY.startswith("sk-"):
        raise SystemExit("OPENAI_API_KEY missing")
    out_path = OUT_DIR / f"openai_{MODEL.replace('/', '_')}_outputs.json"
    meta_path = OUT_DIR / f"openai_{MODEL.replace('/', '_')}_meta.json"
    results = {}
    meta = {"model": MODEL, "items": {}, "prompt_tokens": 0, "completion_tokens": 0}
    if out_path.exists():
        results = json.loads(out_path.read_text())
    if meta_path.exists():
        meta.update(json.loads(meta_path.read_text()))

    todo = [p for p in PROMPTS if p["id"] not in results]
    logf = (OUT_DIR / "openai_progress.txt").open("a")
    def log(msg):
        line = msg
        logf.write(line + "\n")
        logf.flush()
        print(line, flush=True)

    log(f"model={MODEL} done={len(results)} todo={len(todo)} total={len(PROMPTS)} workers={WORKERS}")

    def work(p):
        t0 = time.time()
        paths, info = call(p["prompt"])
        return p["id"], paths, info, time.time() - t0

    def save():
        out_path.write_text(json.dumps(results, indent=2))
        meta_path.write_text(json.dumps(meta, indent=2))

    BATCH = 6
    for bi in range(0, len(todo), BATCH):
        chunk = todo[bi: bi + BATCH]
        with ThreadPoolExecutor(max_workers=min(WORKERS, len(chunk))) as ex:
            futs = [ex.submit(work, p) for p in chunk]
            for fut in as_completed(futs):
                try:
                    cid, paths, info, sec = fut.result()
                except Exception as e:
                    log(f"FAIL {type(e).__name__} {str(e)[:240]}")
                    continue
                u = info["usage"]
                results[cid] = paths
                meta["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
                meta["completion_tokens"] += int(u.get("completion_tokens") or 0)
                meta["items"][cid] = {
                    "n": len(paths),
                    "usage": u,
                    "sec": round(sec, 2),
                    "model": info.get("model"),
                }
                log(
                    f"[{len(results)}/{len(PROMPTS)}] {cid} n={len(paths)} "
                    f"pt={u.get('prompt_tokens')} ct={u.get('completion_tokens')} {sec:.1f}s"
                )
        save()
    save()
    log(f"DONE prompt_tok={meta['prompt_tokens']} cmpl_tok={meta['completion_tokens']}")
    logf.close()


if __name__ == "__main__":
    main()
