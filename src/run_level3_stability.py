#!/usr/bin/env python3
"""Level-3 overclaim audit + Paper18-style explanation stability."""
from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathcheck import (
    OUT_DIR,
    DATA_DIR,
    evaluate_output,
    match_pathway,
    parse_enrichr,
)
from run_eval import closed_book_interpret, ora_interpret

RNG = random.Random(20260927)

OVERCLAIM_PAT = re.compile(
    r"\b("
    r"activat(?:e|ed|es|ion|ing)|"
    r"inactivat(?:e|ed|ion)|"
    r"up-?regulat(?:e|ed|ion)|"
    r"down-?regulat(?:e|ed|ion)|"
    r"induc(?:e|ed|es|tion)|"
    r"repress(?:ed|ion)|"
    r"driv(?:e|es|en|ing)|"
    r"causal(?:ly)?|"
    r"cause[sd]?|"
    r"trigger(?:s|ed|ing)?|"
    r"responsible for|"
    r"leading to|"
    r"results in|"
    r"switch(?:ed|es)? on|"
    r"turned on|"
    r"pathway is on|"
    r"indicating activation|"
    r"suggests activation|"
    r"demonstrates activation"
    r")\b",
    re.I,
)

MEMBER_PAT = re.compile(
    r"\b("
    r"involved in|participate|members? of|components? of|"
    r"annotated|belong(?:s|ing)? to|consistent with|"
    r"associated with|mapped to|from the (?:input )?set|"
    r"key (?:components?|genes)|according to (?:KEGG|Reactome)"
    r")\b",
    re.I,
)


def overclaim_label(reason: str) -> str:
    r = reason or ""
    oc = bool(OVERCLAIM_PAT.search(r))
    mb = bool(MEMBER_PAT.search(r))
    if oc and not mb:
        return "overclaim"
    if oc and mb:
        return "mixed"
    return "conservative"


def predicted_keys(preds, library, min_score=0.62):
    keys = []
    for p in preds:
        name = p["name"] if isinstance(p, dict) else str(p)
        hits = match_pathway(name, library, topk=3)
        if hits and hits[0]["score"] >= min_score:
            keys.append(hits[0]["key"])
    return keys


def jaccard(a, b):
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def perturb(genes, mode, universe):
    g = list(genes)
    n = len(g)
    if mode == "shuffle":
        h = g[:]
        RNG.shuffle(h)
        return h
    if mode == "drop10":
        k = max(1, int(round(0.10 * n)))
        drop = set(RNG.sample(g, min(k, n - 1)))
        return [x for x in g if x not in drop]
    if mode == "drop20":
        k = max(1, int(round(0.20 * n)))
        drop = set(RNG.sample(g, min(k, n - 1)))
        return [x for x in g if x not in drop]
    if mode == "add10":
        k = max(1, int(round(0.10 * n)))
        extra = [u for u in universe if u not in set(g)]
        return g + RNG.sample(extra, min(k, len(extra)))
    if mode == "add30":
        k = max(1, int(round(0.30 * n)))
        extra = [u for u in universe if u not in set(g)]
        return g + RNG.sample(extra, min(k, len(extra)))
    raise ValueError(mode)


def main():
    kegg = parse_enrichr(DATA_DIR / "KEGG_2021_Human.txt", "KEGG")
    reactome = parse_enrichr(DATA_DIR / "Reactome_2022.txt", "Reactome")
    library = {**kegg, **reactome}
    universe = sorted({g for rec in library.values() for g in rec["genes"]})
    universe_n = len(universe)
    cases = json.loads((OUT_DIR / "benchmark.json").read_text())
    gpt = json.loads((OUT_DIR / "openai_gpt-4.1_outputs.json").read_text())
    gpt_eval = json.loads((OUT_DIR / "eval_gpt41.json").read_text())
    eval_by_id = {r["id"]: r for r in gpt_eval}

    # ---- Level 3 on GPT-4.1 reasons
    tax = defaultdict(int)
    by_support = defaultdict(lambda: defaultdict(int))
    by_split = defaultdict(lambda: defaultdict(int))
    examples = {"overclaim": [], "conservative": [], "mixed": []}
    n_claims = 0
    for case in cases:
        ev = eval_by_id[case["id"]]
        item_by_name = {it["predicted_name"]: it for it in ev["items"]}
        for pred in gpt[case["id"]]:
            lab = overclaim_label(pred.get("reason") or "")
            tax[lab] += 1
            n_claims += 1
            it = item_by_name.get(pred["name"], {})
            sl = it.get("label", "unknown")
            by_support[sl][lab] += 1
            by_split[case["split"]][lab] += 1
            if len(examples[lab]) < 6:
                examples[lab].append(
                    {
                        "id": case["id"],
                        "split": case["split"],
                        "name": pred["name"],
                        "support": sl,
                        "reason": pred.get("reason", "")[:240],
                    }
                )

    overclaim_rate = (tax["overclaim"] + tax["mixed"]) / max(n_claims, 1)
    strict_overclaim = tax["overclaim"] / max(n_claims, 1)
    level3 = {
        "n_claims": n_claims,
        "labels": dict(tax),
        "overclaim_or_mixed": overclaim_rate,
        "strict_overclaim": strict_overclaim,
        "conservative": tax["conservative"] / max(n_claims, 1),
        "by_support": {k: dict(v) for k, v in by_support.items()},
        "by_split": {k: dict(v) for k, v in by_split.items()},
        "examples": examples,
    }
    (OUT_DIR / "level3_overclaim.json").write_text(json.dumps(level3, indent=2))
    print("LEVEL3", json.dumps({k: level3[k] for k in ["n_claims", "labels", "overclaim_or_mixed", "strict_overclaim", "conservative"]}, indent=2))

    # ---- Stability: closed-book + ORA on perturbations of all 229
    modes = ["shuffle", "drop10", "drop20", "add10", "add30"]
    stab = {m: {"cb": [], "ora": []} for m in modes}
    for case in cases:
        base_cb = predicted_keys(closed_book_interpret(case["genes"]), library)
        base_ora = predicted_keys(ora_interpret(case["genes"], library, universe_n), library)
        for mode in modes:
            g2 = perturb(case["genes"], mode, universe)
            cb2 = predicted_keys(closed_book_interpret(g2), library)
            ora2 = predicted_keys(ora_interpret(g2, library, universe_n), library)
            stab[mode]["cb"].append(jaccard(base_cb, cb2))
            stab[mode]["ora"].append(jaccard(base_ora, ora2))

    summary = {}
    print(f"{'mode':10} {'cb_mean':>8} {'ora_mean':>8}")
    for mode in modes:
        cb_m = sum(stab[mode]["cb"]) / len(stab[mode]["cb"])
        ora_m = sum(stab[mode]["ora"]) / len(stab[mode]["ora"])
        summary[mode] = {
            "closed_book": round(cb_m, 4),
            "ora": round(ora_m, 4),
            "n": len(stab[mode]["cb"]),
        }
        print(f"{mode:10} {cb_m:8.3f} {ora_m:8.3f}")
    (OUT_DIR / "stability.json").write_text(json.dumps(summary, indent=2))
    print("wrote level3 + stability")


if __name__ == "__main__":
    main()
