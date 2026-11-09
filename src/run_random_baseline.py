#!/usr/bin/env python3
"""Random-title baseline and exact-match ablation of the closed-book baseline."""
import json
import random
from pathcheck import DATA_DIR, OUT_DIR, parse_enrichr, evaluate_output, match_pathway

RNG = random.Random(7)


def mean(rows, k):
    return sum(r[k] for r in rows) / len(rows)


def main():
    kegg = parse_enrichr(DATA_DIR / "KEGG_2021_Human.txt", "KEGG")
    reactome = parse_enrichr(DATA_DIR / "Reactome_2022.txt", "Reactome")
    library = {**kegg, **reactome}
    universe_n = len({g for rec in library.values() for g in rec["genes"]})
    cases = json.loads((OUT_DIR / "benchmark.json").read_text())
    cb_out = json.loads((OUT_DIR / "model_outputs.json").read_text())["closed_book"]

    names = [rec["name"] for rec in library.values()]
    rnd_rows = []
    for case in cases:
        preds = [{"name": RNG.choice(names)} for _ in range(4)]
        rnd_rows.append(evaluate_output(case, preds, library, universe_n))
    (OUT_DIR / "eval_random.json").write_text(json.dumps(rnd_rows))

    exact_rows = []
    for case in cases:
        forced = []
        for p in cb_out[case["id"]]:
            hits = match_pathway(p["name"], library, topk=3)
            forced.append({"name": hits[0]["name"] if hits and hits[0]["exact"] else "___NOT_A_PATHWAY___"})
        exact_rows.append(evaluate_output(case, forced, library, universe_n))

    for label, rows in [("random", rnd_rows), ("closed-book exact-only", exact_rows)]:
        print(label, {k: round(mean(rows, k), 3) for k in ["validity_rate", "supported_rate", "recovery"]})


if __name__ == "__main__":
    main()
