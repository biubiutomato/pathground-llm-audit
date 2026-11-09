# PathGround

Code, benchmark, and model outputs for:

> Ye Ye. **PathGround: Auditing Existence, Gene Support, and Overclaim in LLM Pathway Interpretation.**
> IEEE BIBM 2026 Workshops (paper S20211).

PathGround is a deterministic, retrieval-free audit of LLM pathway interpretations. After a model names
pathways for a gene set, each claim is checked for

1. **existence** – does the title map to a KEGG or Reactome entry?
2. **gene support** – do enough input genes belong to the mapped pathway (≥3 genes and hypergeometric *p* ≤ 0.05 or ≥8% of the input)?
3. **overclaim** – does the rationale describe membership as activation or causation?
4. **stability** – does the answer change when the gene list is shuffled or slightly edited?

## Repository layout

| Path | Contents |
|---|---|
| `src/pathcheck.py` | Checker (title matching, gene support) and benchmark builder |
| `src/run_eval.py` | Closed-book and ORA reference generators |
| `src/run_random_baseline.py` | Random-title baseline and exact-match ablation |
| `src/run_openai.py` | GPT-4.1 generation (needs `OPENAI_API_KEY`) |
| `src/evaluate_gpt41.py` | Scores the stored GPT-4.1 outputs |
| `src/run_level3_stability.py` | Overclaim tagging and baseline stability |
| `src/run_gpt41_stability.py` | GPT-4.1 perturbation re-queries (needs `OPENAI_API_KEY`) |
| `src/sensitivity_analysis.py` | Support-threshold sensitivity and per-split stability |
| `results/benchmark.json` | The 229 benchmark gene sets |
| `results/prompts.json` | Prompts sent to the model |
| `results/openai_gpt-4.1_outputs.json` | GPT-4.1 answers (`gpt-4.1-2025-04-14`, temperature 0) |
| `results/gpt41_stability_outputs.json` | GPT-4.1 answers to perturbed gene lists (60 cases × 4 perturbations) |
| `results/model_outputs.json` | Closed-book and ORA answers |
| `results/eval_*.json` | Per-case scores for each generator |
| `figures/` | Figures in the paper |

## Data

The reference library is the union of the Enrichr **KEGG_2021_Human** and **Reactome_2022** gene-set
libraries. They are not redistributed here; see `data/README.md` for download instructions.

## Reproduce the paper's numbers (no API calls)

```bash
pip install -r requirements.txt
# put KEGG_2021_Human.txt and Reactome_2022.txt in data/ (see data/README.md)
cd src
python evaluate_gpt41.py        # GPT-4.1 existence / support / recovery
python run_random_baseline.py   # random titles and exact-match ablation
python run_eval.py              # closed-book and ORA
python run_level3_stability.py  # overclaim and baseline stability
python sensitivity_analysis.py  # threshold sensitivity and per-split GPT-4.1 stability
```

All scoring scripts read the stored model outputs in `results/`, so no API key is needed.
Re-running generation (`run_openai.py`, `run_gpt41_stability.py`) calls the OpenAI API and will not
reproduce the stored answers exactly.

**Benchmark note.** `python pathcheck.py` rebuilds the benchmark from the library files with a fixed seed.
It regenerates the same 229 gene sets, but the gene *order* within the mixed split can differ between
Python runs. Because model answers depend on gene order, use the stored `results/benchmark.json` to
reproduce the paper. Re-running `run_eval.py` reproduces the stored closed-book and ORA scores; a few
closed-book rationale strings can differ, but every reported metric is identical.

## Citation

```bibtex
@inproceedings{ye2026pathground,
  author    = {Ye, Ye},
  title     = {PathGround: Auditing Existence, Gene Support, and Overclaim in LLM Pathway Interpretation},
  booktitle = {Proc. IEEE International Conference on Bioinformatics and Biomedicine (BIBM) Workshops},
  year      = {2026}
}
```

## License

Code: MIT (see `LICENSE`). KEGG and Reactome content remains subject to the licenses of those resources.
