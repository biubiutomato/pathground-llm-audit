#!/usr/bin/env python3
"""Score the stored GPT-4.1 outputs with the PathGround checker -> results/eval_gpt41.json."""
import json
from pathcheck import DATA_DIR, OUT_DIR, parse_enrichr, evaluate_output


def main():
    kegg = parse_enrichr(DATA_DIR / "KEGG_2021_Human.txt", "KEGG")
    reactome = parse_enrichr(DATA_DIR / "Reactome_2022.txt", "Reactome")
    library = {**kegg, **reactome}
    universe_n = len({g for rec in library.values() for g in rec["genes"]})
    cases = json.loads((OUT_DIR / "benchmark.json").read_text())
    outputs = json.loads((OUT_DIR / "openai_gpt-4.1_outputs.json").read_text())
    rows = [evaluate_output(c, outputs[c["id"]], library, universe_n) for c in cases if c["id"] in outputs]
    (OUT_DIR / "eval_gpt41.json").write_text(json.dumps(rows))
    n = len(rows)
    for k in ["validity_rate", "supported_rate", "unsupported_rate", "hallucination_rate", "recovery"]:
        print(f"{k:20} {sum(r[k] for r in rows) / n:.3f}")


if __name__ == "__main__":
    main()
