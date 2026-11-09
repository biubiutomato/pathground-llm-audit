#!/usr/bin/env python3
"""GPT-4.1 stability under gene-list perturbations. Resumable."""
from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathcheck import (
    OUT_DIR,
    DATA_DIR,
    match_pathway,
    parse_enrichr,
    PROMPT,
)
from run_level3_stability import perturb, predicted_keys, jaccard

import os
KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = "gpt-4.1"
RNG = random.Random(20260927)
SYSTEM = (
    "You are a molecular biologist. Return JSON only of the form "
    '{"pathways":[{"name":"...","reason":"..."}]} with 3 to 5 items. '
    "Use real KEGG or Reactome pathway names. Only claim a pathway if multiple input genes participate."
)

OUT = OUT_DIR / "gpt41_stability_outputs.json"
META = OUT_DIR / "gpt41_stability_meta.json"
PROG = OUT_DIR / "gpt41_stability_progress.txt"


def parse(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        m = re.search(r"\{.*\}", text, re.S)
        data = json.loads(m.group(0) if m else text)
        paths = data.get("pathways") or []
        out = []
        for p in paths:
            if isinstance(p, dict) and p.get("name"):
                out.append({"name": str(p["name"]).strip(), "reason": str(p.get("reason") or "").strip()})
        if out:
            return out
    except Exception:
        pass
    names = re.findall(r'"name"\s*:\s*"([^"]+)"', text)
    return [{"name": n, "reason": ""} for n in names]


def call(prompt):
    body = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 350,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }
    last = None
    for attempt in range(4):
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
            json=body,
            timeout=40,
        )
        if r.status_code == 429:
            time.sleep(4 + attempt * 4)
            last = "429"
            continue
        if r.status_code != 200:
            last = f"{r.status_code} {r.text[:160]}"
            time.sleep(2)
            continue
        payload = r.json()
        text = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage") or {}
        return parse(text), usage
    raise RuntimeError(last)


def pick_cases(cases):
    by = {"clean": [], "noisy": [], "mixed": [], "decoy": [], "geo_like": []}
    for c in cases:
        by[c["split"]].append(c)
    picked = []
    picked += by["decoy"]
    picked += by["mixed"]
    picked += by["geo_like"]
    picked += by["noisy"][:10]
    # diverse clean: first 20 is fine (covers many pathways)
    picked += by["clean"][:16]
    return picked


def main():
    kegg = parse_enrichr(DATA_DIR / "KEGG_2021_Human.txt", "KEGG")
    reactome = parse_enrichr(DATA_DIR / "Reactome_2022.txt", "Reactome")
    library = {**kegg, **reactome}
    universe = sorted({g for rec in library.values() for g in rec["genes"]})
    cases = json.loads((OUT_DIR / "benchmark.json").read_text())
    base_preds = json.loads((OUT_DIR / "openai_gpt-4.1_outputs.json").read_text())
    picked = pick_cases(cases)
    modes = ["shuffle", "drop10", "drop20", "add10"]
    jobs = []
    for c in picked:
        for mode in modes:
            jobs.append((c, mode))

    store = json.loads(OUT.read_text()) if OUT.exists() else {}
    meta = json.loads(META.read_text()) if META.exists() else {"prompt_tokens": 0, "completion_tokens": 0}
    done_keys = set(store)
    todo = [(c, m) for c, m in jobs if f"{c['id']}::{m}" not in done_keys]
    line = f"stab start picked={len(picked)} jobs={len(jobs)} todo={len(todo)}"
    print(line, flush=True)
    PROG.write_text((PROG.read_text() + "\n" if PROG.exists() else "") + line + "\n")

    for i, (c, mode) in enumerate(todo, 1):
        key = f"{c['id']}::{mode}"
        g2 = perturb(c["genes"], mode, universe)
        prompt = PROMPT.format(n=len(g2), genes=", ".join(g2))
        t0 = time.time()
        try:
            paths, usage = call(prompt)
        except Exception as e:
            msg = f"FAIL {key} {e}"
            print(msg, flush=True)
            with PROG.open("a") as f:
                f.write(msg + "\n")
            continue
        store[key] = {"id": c["id"], "mode": mode, "n_genes": len(g2), "pathways": paths}
        meta["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        meta["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        if i % 4 == 0 or i == len(todo):
            OUT.write_text(json.dumps(store, indent=2))
            META.write_text(json.dumps(meta, indent=2))
        msg = f"[{len(store)}/{len(jobs)}] {key} n={len(paths)} {time.time()-t0:.1f}s"
        print(msg, flush=True)
        with PROG.open("a") as f:
            f.write(msg + "\n")
        time.sleep(0.05)

    OUT.write_text(json.dumps(store, indent=2))
    META.write_text(json.dumps(meta, indent=2))

    # score
    by_mode = {m: [] for m in modes}
    for c in picked:
        p0 = predicted_keys(base_preds[c["id"]], library)
        for mode in modes:
            rec = store.get(f"{c['id']}::{mode}")
            if not rec:
                continue
            p1 = predicted_keys(rec["pathways"], library)
            by_mode[mode].append(jaccard(p0, p1))
    summary = {}
    print("GPT-4.1 stability")
    for m in modes:
        vals = by_mode[m]
        mu = sum(vals) / len(vals) if vals else 0.0
        summary[m] = {"n": len(vals), "jaccard": round(mu, 4)}
        print(m, len(vals), f"{mu:.3f}")
    (OUT_DIR / "gpt41_stability_summary.json").write_text(json.dumps({"picked": len(picked), "modes": summary, "usage": meta}, indent=2))
    print("DONE pt", meta["prompt_tokens"], "ct", meta["completion_tokens"])


if __name__ == "__main__":
    main()
