#!/usr/bin/env python3
"""
PathCheck: database-grounded evaluation of LLM pathway interpretation.

No RAG. Deterministic whitelist matching + gene-set support + decoy test.
Sources: Enrichr KEGG 2021 Human and Reactome 2022 gene-set libraries.
"""
from __future__ import annotations

import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG = random.Random(20260927)
NP_RNG = np.random.default_rng(20260927)

# ---------------------------------------------------------------------------
# Parsing and normalization
# ---------------------------------------------------------------------------

STOPWORDS = {
    "pathway", "pathways", "signaling", "signalling", "signal",
    "cascade", "network", "system", "systems", "process", "processes",
    "of", "the", "and", "in", "to", "by", "via", "for", "on", "with",
    "human", "homo", "sapiens", "hsa", "reactome", "kegg",
    "regulation", "regulatory", "mediated", "dependent",
}

ALIAS_RULES = [
    (r"\bpi3k[- ]?akt\b", "pi3k akt"),
    (r"\bpi3k/akt\b", "pi3k akt"),
    (r"\bakt/pkb\b", "akt"),
    (r"\bnf[- ]?κb\b", "nfkb"),
    (r"\bnf[- ]?kappab\b", "nfkb"),
    (r"\bnfkb\b", "nfkb"),
    (r"\btnf[- ]?alpha\b", "tnf"),
    (r"\btnfα\b", "tnf"),
    (r"\bil[- ]?6\b", "il6"),
    (r"\bwnt/β[- ]?catenin\b", "wnt"),
    (r"\bwnt/beta[- ]?catenin\b", "wnt"),
    (r"\bmapk/erk\b", "mapk erk"),
    (r"\bras[- ]?raf[- ]?mek[- ]?erk\b", "mapk"),
    (r"\bjak[- /]?stat\b", "jak stat"),
    (r"\btgf[- ]?β\b", "tgf beta"),
    (r"\btgf[- ]?beta\b", "tgf beta"),
    (r"\bmtorc1\b", "mtor"),
    (r"\bmtorc2\b", "mtor"),
    (r"\bp53\b", "tp53 p53"),
    (r"\btp53\b", "tp53 p53"),
    (r"\boxphos\b", "oxidative phosphorylation"),
    (r"\btca cycle\b", "citrate cycle tca"),
    (r"\bkreb'?s cycle\b", "citrate cycle tca"),
    (r"\becm[- ]?receptor interaction\b", "ecm receptor"),
]


def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def normalize_name(name: str) -> str:
    s = strip_accents(name).lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"r-hsa-\d+", " ", s)
    s = re.sub(r"hsa\d+", " ", s)
    s = s.replace("_", " ").replace("/", " ").replace("-", " ")
    for pat, repl in ALIAS_RULES:
        s = re.sub(pat, repl, s)
    s = re.sub(r"[^a-z0-9+ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokens(name: str) -> Set[str]:
    t = [w for w in normalize_name(name).split() if w and w not in STOPWORDS and len(w) > 1]
    return set(t)


def parse_enrichr(path: Path, source: str) -> Dict[str, dict]:
    lib = {}
    with path.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            raw_name = parts[0].strip()
            genes = [g.strip().upper() for g in parts[2:] if g.strip()]
            # drop empty placeholders
            genes = [g for g in genes if g and g != "NONE"]
            if len(genes) < 8:
                continue
            key = f"{source}::{raw_name}"
            lib[key] = {
                "source": source,
                "name": raw_name,
                "genes": sorted(set(genes)),
                "norm": normalize_name(raw_name),
                "tokens": sorted(tokens(raw_name)),
            }
    return lib


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def token_overlap(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def match_pathway(query: str, library: Dict[str, dict], topk: int = 5) -> List[dict]:
    qn = normalize_name(query)
    qt = tokens(query)
    hits = []
    for key, rec in library.items():
        exact = qn == rec["norm"]
        # containment of normalized strings
        contained = qn in rec["norm"] or rec["norm"] in qn
        jac = jaccard(qt, set(rec["tokens"]))
        ov = token_overlap(qt, set(rec["tokens"]))
        score = 0.0
        if exact:
            score = 1.0
        elif contained and jac >= 0.4:
            score = 0.92
        else:
            score = 0.55 * jac + 0.45 * ov
        if score >= 0.55 or exact:
            hits.append(
                {
                    "key": key,
                    "name": rec["name"],
                    "source": rec["source"],
                    "score": round(score, 4),
                    "exact": exact,
                }
            )
    hits.sort(key=lambda x: (-x["score"], x["name"]))
    # keep best per source-normalized name
    return hits[:topk]


def hypergeom_sf(k: int, K: int, n: int, N: int) -> float:
    """P(X >= k) for X ~ Hypergeometric(N, K, n). Upper tail."""
    if n == 0 or K == 0 or N == 0:
        return 1.0
    k = max(k, 0)
    # use survival via cumulative
    from math import lgamma

    def logC(a, b):
        if b < 0 or b > a:
            return float("-inf")
        return lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1)

    # sum C(K,i)*C(N-K,n-i) / C(N,n) for i=k..min(K,n)
    lo = k
    hi = min(K, n)
    if lo > hi:
        return 1.0
    log_den = logC(N, n)
    total = 0.0
    for i in range(lo, hi + 1):
        val = math.exp(logC(K, i) + logC(N - K, n - i) - log_den)
        total += val
    return min(1.0, max(0.0, total))


def gene_support(
    query_genes: Sequence[str],
    pathway_genes: Sequence[str],
    background_n: int,
    min_overlap: int = 3,
    max_p: float = 0.05,
    min_frac: float = 0.08,
) -> dict:
    q = set(g.upper() for g in query_genes)
    p = set(g.upper() for g in pathway_genes)
    inter = sorted(q & p)
    k = len(inter)
    frac_q = k / max(len(q), 1)
    pval = hypergeom_sf(k, len(p), len(q), background_n)
    supported = (k >= min_overlap) and (pval <= max_p or frac_q >= min_frac)
    return {
        "overlap": k,
        "overlap_genes": inter[:20],
        "frac_query": round(frac_q, 4),
        "frac_pathway": round(k / max(len(p), 1), 4),
        "pvalue": float(f"{pval:.4g}"),
        "supported": bool(supported),
    }


# ---------------------------------------------------------------------------
# Benchmark construction
# ---------------------------------------------------------------------------

# Well-separated, recognizable pathways used as seeds
SEED_KEGG = [
    "Apoptosis",
    "TNF signaling pathway",
    "PI3K-Akt signaling pathway",
    "Wnt signaling pathway",
    "Jak-STAT signaling pathway",
    "NF-kappa B signaling pathway",
    "p53 signaling pathway",
    "MAPK signaling pathway",
    "HIF-1 signaling pathway",
    "Toll-like receptor signaling pathway",
    "T cell receptor signaling pathway",
    "Cell cycle",
    "Oxidative phosphorylation",
    "Citrate cycle (TCA cycle)",
    "Glycolysis / Gluconeogenesis",
    "Fatty acid degradation",
    "Complement and coagulation cascades",
    "ECM-receptor interaction",
    "Focal adhesion",
    "Hippo signaling pathway",
    "Notch signaling pathway",
    "Hedgehog signaling pathway",
    "mTOR signaling pathway",
    "FoxO signaling pathway",
    "IL-17 signaling pathway",
    "Chemokine signaling pathway",
    "Natural killer cell mediated cytotoxicity",
    "Antigen processing and presentation",
    "Ferroptosis",
    "Autophagy - animal",
]

SEED_REACTOME = [
    "Innate Immune System",
    "Adaptive Immune System",
    "Interferon Signaling",
    "Cytokine Signaling in Immune system",
    "Apoptosis",
    "Cell Cycle",
    "DNA Repair",
    "Signaling by WNT",
    "Signaling by Notch",
    "Signaling by Hedgehog",
    "Signaling by ERBB2",
    "Signaling by EGFR",
    "Signaling by Interleukins",
    "Neutrophil degranulation",
    "Complement cascade",
    "Extracellular matrix organization",
    "Axon guidance",
    "Neuronal System",
    "Transmission across Chemical Synapses",
    "The citric acid (TCA) cycle and respiratory electron transport",
    "Glycolysis",
    "Fatty acid metabolism",
    "Translation",
    "mRNA Splicing",
    "RHO GTPase cycle",
]


def find_by_name(library: Dict[str, dict], name: str, source: str) -> Optional[str]:
    target = normalize_name(name)
    # exact first
    for k, rec in library.items():
        if rec["source"] == source and rec["norm"] == target:
            return k
    # unique containment
    cands = [
        k
        for k, rec in library.items()
        if rec["source"] == source and (target == rec["norm"] or target in rec["norm"])
    ]
    if len(cands) == 1:
        return cands[0]
    # highest token jaccard
    best, best_s = None, 0.0
    tt = tokens(name)
    for k, rec in library.items():
        if rec["source"] != source:
            continue
        s = jaccard(tt, set(rec["tokens"]))
        if s > best_s:
            best, best_s = k, s
    return best if best_s >= 0.7 else None


def sample_genes(genes: Sequence[str], n: int) -> List[str]:
    g = list(genes)
    if n >= len(g):
        return g[:]
    return RNG.sample(g, n)


def construct_benchmark(kegg: Dict[str, dict], reactome: Dict[str, dict]) -> List[dict]:
    all_lib = {**kegg, **reactome}
    universe = sorted({g for rec in all_lib.values() for g in rec["genes"]})
    cases = []
    cid = 0

    def add(case):
        nonlocal cid
        cid += 1
        case["id"] = f"PC{cid:04d}"
        cases.append(case)

    kegg_keys = []
    for name in SEED_KEGG:
        k = find_by_name(kegg, name, "KEGG")
        if k:
            kegg_keys.append(k)
    reac_keys = []
    for name in SEED_REACTOME:
        k = find_by_name(reactome, name, "Reactome")
        if k:
            reac_keys.append(k)

    # A. Clean pathway sets: 10 / 20 / 30 genes
    for src_keys, src in ((kegg_keys, "KEGG"), (reac_keys, "Reactome")):
        for key in src_keys:
            rec = all_lib[key]
            for n in (10, 20, 30):
                if len(rec["genes"]) < n:
                    continue
                genes = sample_genes(rec["genes"], n)
                add(
                    {
                        "split": "clean",
                        "source": src,
                        "true_keys": [key],
                        "true_names": [rec["name"]],
                        "n_genes": n,
                        "genes": genes,
                        "decoy_keys": [],
                        "decoy_names": [],
                    }
                )

    # B. Noisy: 70% pathway + 30% random
    for src_keys, src in ((kegg_keys[:18], "KEGG"), (reac_keys[:15], "Reactome")):
        for key in src_keys:
            rec = all_lib[key]
            n_true, n_noise = 14, 6
            if len(rec["genes"]) < n_true:
                continue
            true_g = sample_genes(rec["genes"], n_true)
            pool = [g for g in universe if g not in rec["genes"]]
            noise = RNG.sample(pool, n_noise)
            add(
                {
                    "split": "noisy",
                    "source": src,
                    "true_keys": [key],
                    "true_names": [rec["name"]],
                    "n_genes": n_true + n_noise,
                    "genes": RNG.sample(true_g + noise, n_true + n_noise),
                    "decoy_keys": [],
                    "decoy_names": [],
                }
            )

    # C. Mixed: 50% A + 50% B, disjoint-ish pathways
    pairs_k = [
        ("Apoptosis", "Cell cycle"),
        ("TNF signaling pathway", "Wnt signaling pathway"),
        ("Oxidative phosphorylation", "Glycolysis / Gluconeogenesis"),
        ("T cell receptor signaling pathway", "Toll-like receptor signaling pathway"),
        ("PI3K-Akt signaling pathway", "p53 signaling pathway"),
        ("Notch signaling pathway", "Hedgehog signaling pathway"),
        ("Complement and coagulation cascades", "ECM-receptor interaction"),
        ("Jak-STAT signaling pathway", "MAPK signaling pathway"),
    ]
    pairs_r = [
        ("Innate Immune System", "Adaptive Immune System"),
        ("Signaling by WNT", "Signaling by Notch"),
        ("Glycolysis", "Fatty acid metabolism"),
        ("DNA Repair", "Cell Cycle"),
        ("Interferon Signaling", "Complement cascade"),
        ("Axon guidance", "Extracellular matrix organization"),
    ]
    for pairs, src, finder_lib in (
        (pairs_k, "KEGG", kegg),
        (pairs_r, "Reactome", reactome),
    ):
        for a, b in pairs:
            ka = find_by_name(finder_lib, a, src)
            kb = find_by_name(finder_lib, b, src)
            if not ka or not kb:
                continue
            ga = sample_genes(all_lib[ka]["genes"], 12)
            gb = sample_genes(all_lib[kb]["genes"], 12)
            add(
                {
                    "split": "mixed",
                    "source": src,
                    "true_keys": [ka, kb],
                    "true_names": [all_lib[ka]["name"], all_lib[kb]["name"]],
                    "n_genes": len(set(ga + gb)),
                    "genes": RNG.sample(list(set(ga + gb)), len(set(ga + gb))),
                    "decoy_keys": [],
                    "decoy_names": [],
                }
            )

    # D. Decoy: 15 true + 5 from a flashy but wrong pathway
    decoy_pairs = [
        ("Apoptosis", "TNF signaling pathway"),
        ("Cell cycle", "p53 signaling pathway"),
        ("Oxidative phosphorylation", "TNF signaling pathway"),
        ("Wnt signaling pathway", "NF-kappa B signaling pathway"),
        ("Citrate cycle (TCA cycle)", "Jak-STAT signaling pathway"),
        ("Glycolysis / Gluconeogenesis", "Toll-like receptor signaling pathway"),
        ("Hedgehog signaling pathway", "TNF signaling pathway"),
        ("Notch signaling pathway", "IL-17 signaling pathway"),
        ("Fatty acid degradation", "Chemokine signaling pathway"),
        ("ECM-receptor interaction", "p53 signaling pathway"),
        ("Hippo signaling pathway", "TNF signaling pathway"),
        ("mTOR signaling pathway", "Complement and coagulation cascades"),
        ("FoxO signaling pathway", "Natural killer cell mediated cytotoxicity"),
        ("Ferroptosis", "T cell receptor signaling pathway"),
        ("Autophagy - animal", "IL-17 signaling pathway"),
    ]
    for true_n, decoy_n in decoy_pairs:
        kt = find_by_name(kegg, true_n, "KEGG")
        kd = find_by_name(kegg, decoy_n, "KEGG")
        if not kt or not kd:
            continue
        true_only = [g for g in kegg[kt]["genes"] if g not in set(kegg[kd]["genes"])]
        decoy_only = [g for g in kegg[kd]["genes"] if g not in set(kegg[kt]["genes"])]
        if len(true_only) < 15 or len(decoy_only) < 5:
            continue
        genes = sample_genes(true_only, 15) + sample_genes(decoy_only, 5)
        add(
            {
                "split": "decoy",
                "source": "KEGG",
                "true_keys": [kt],
                "true_names": [kegg[kt]["name"]],
                "n_genes": 20,
                "genes": RNG.sample(genes, 20),
                "decoy_keys": [kd],
                "decoy_names": [kegg[kd]["name"]],
            }
        )

    # E. External-style GEO-like mixed inflammatory / cancer signatures
    # Constructed as realistic DE-like unions of 3 related pathways + noise
    geo_specs = [
        {
            "label": "GEO-like_TNFa_inflammation",
            "paths": ["TNF signaling pathway", "NF-kappa B signaling pathway", "IL-17 signaling pathway"],
        },
        {
            "label": "GEO-like_Tcell_activation",
            "paths": ["T cell receptor signaling pathway", "Jak-STAT signaling pathway", "Chemokine signaling pathway"],
        },
        {
            "label": "GEO-like_hypoxia_metabolism",
            "paths": ["HIF-1 signaling pathway", "Glycolysis / Gluconeogenesis", "Oxidative phosphorylation"],
        },
        {
            "label": "GEO-like_apoptosis_p53",
            "paths": ["Apoptosis", "p53 signaling pathway", "Cell cycle"],
        },
        {
            "label": "GEO-like_wnt_development",
            "paths": ["Wnt signaling pathway", "Notch signaling pathway", "Hippo signaling pathway"],
        },
        {
            "label": "GEO-like_ecm_adhesion",
            "paths": ["ECM-receptor interaction", "Focal adhesion", "PI3K-Akt signaling pathway"],
        },
    ]
    for spec in geo_specs:
        keys, names, genes = [], [], []
        for pn in spec["paths"]:
            k = find_by_name(kegg, pn, "KEGG")
            if not k:
                continue
            keys.append(k)
            names.append(kegg[k]["name"])
            genes.extend(sample_genes(kegg[k]["genes"], 8))
        pool = [g for g in universe if g not in set(genes)]
        genes = list(dict.fromkeys(genes)) + RNG.sample(pool, 8)
        add(
            {
                "split": "geo_like",
                "source": "KEGG",
                "true_keys": keys,
                "true_names": names,
                "n_genes": len(genes),
                "genes": RNG.sample(genes, len(genes)),
                "decoy_keys": [],
                "decoy_names": [],
                "label": spec["label"],
            }
        )

    return cases, universe, all_lib


# ---------------------------------------------------------------------------
# Evaluation of one model output
# ---------------------------------------------------------------------------

def evaluate_output(
    case: dict,
    predicted_pathways: Sequence[dict],
    library: Dict[str, dict],
    universe_n: int,
) -> dict:
    """
    predicted_pathways: list of {name, reason?}
    """
    items = []
    n_pred = 0
    n_valid = 0
    n_supported = 0
    n_halluc = 0
    n_unsup = 0
    sources_hit = set()
    recovered = set()
    decoy_pref = 0
    named_decoy = 0

    for pred in predicted_pathways:
        name = pred["name"] if isinstance(pred, dict) else str(pred)
        n_pred += 1
        hits = match_pathway(name, library, topk=8)
        valid = bool(hits) and hits[0]["score"] >= 0.62
        best = hits[0] if hits else None
        support = None
        supported = False
        if valid and best:
            rec = library[best["key"]]
            support = gene_support(case["genes"], rec["genes"], universe_n)
            supported = support["supported"]
            sources_hit.add(rec["source"])
            # recovery of planted truth
            if best["key"] in case["true_keys"]:
                recovered.add(best["key"])
            else:
                # also recover if matched another true name alias
                for tk in case["true_keys"]:
                    if jaccard(set(library[tk]["tokens"]), tokens(name)) >= 0.7:
                        recovered.add(tk)
            if case.get("decoy_keys") and best["key"] in case["decoy_keys"]:
                named_decoy += 1
        if not valid:
            n_halluc += 1
            label = "hallucinated"
        elif not supported:
            n_unsup += 1
            n_valid += 1
            label = "unsupported"
        else:
            n_valid += 1
            n_supported += 1
            label = "supported"

        items.append(
            {
                "predicted_name": name,
                "label": label,
                "match": best,
                "support": support,
                "reason": pred.get("reason") if isinstance(pred, dict) else None,
            }
        )

    # decoy preference: among valid predictions, did decoy outrank true?
    if case.get("decoy_keys") and items:
        true_best = 0
        decoy_best = 0
        for it in items:
            if it["match"] and it["match"]["key"] in case["true_keys"] and it["support"] and it["support"]["supported"]:
                true_best += 1
            if it["match"] and it["match"]["key"] in case["decoy_keys"]:
                decoy_best += 1
        if decoy_best > true_best:
            decoy_pref = 1
        elif decoy_best > 0 and true_best == 0:
            decoy_pref = 1

    return {
        "id": case["id"],
        "split": case["split"],
        "source": case["source"],
        "n_pred": n_pred,
        "n_valid": n_valid,
        "n_supported": n_supported,
        "n_unsupported": n_unsup,
        "n_hallucinated": n_halluc,
        "validity_rate": n_valid / n_pred if n_pred else 0.0,
        "supported_rate": n_supported / n_pred if n_pred else 0.0,
        "unsupported_rate": n_unsup / n_pred if n_pred else 0.0,
        "hallucination_rate": n_halluc / n_pred if n_pred else 0.0,
        "database_agreement": int(len(sources_hit) >= 2) if n_valid else 0,
        "recovery": len(recovered) / max(len(case["true_keys"]), 1),
        "recovered_keys": sorted(recovered),
        "decoy_named": named_decoy,
        "decoy_preference": decoy_pref,
        "items": items,
    }


PROMPT = """You are a molecular biologist. Given the human gene set below, identify the most relevant biological pathways and briefly explain why.

Rules:
- List 3 to 5 pathways.
- Prefer canonical pathway names used in KEGG or Reactome.
- Do not invent pathway names that do not exist in curated databases.
- Only claim a pathway if multiple input genes participate in it.
- Return JSON only, of the form:
{{"pathways": [{{"name": "...", "reason": "..."}}]}}

Gene set ({n} genes):
{genes}
"""


def main_build():
    kegg = parse_enrichr(DATA_DIR / "KEGG_2021_Human.txt", "KEGG")
    reactome = parse_enrichr(DATA_DIR / "Reactome_2022.txt", "Reactome")
    print(f"KEGG pathways (>=8 genes): {len(kegg)}")
    print(f"Reactome pathways (>=8 genes): {len(reactome)}")

    cases, universe, library = construct_benchmark(kegg, reactome)
    print(f"Benchmark cases: {len(cases)}")
    print(Counter(c["split"] for c in cases))
    print(f"Universe genes: {len(universe)}")

    # persist library index (names only + gene counts, not full if huge — keep full genes for eval)
    slim_lib = {
        k: {
            "source": v["source"],
            "name": v["name"],
            "norm": v["norm"],
            "tokens": v["tokens"],
            "n_genes": len(v["genes"]),
            "genes": v["genes"],
        }
        for k, v in library.items()
    }
    (OUT_DIR / "library.json").write_text(json.dumps(slim_lib))
    (OUT_DIR / "benchmark.json").write_text(json.dumps(cases, indent=2))
    prompts = []
    for c in cases:
        prompts.append(
            {
                "id": c["id"],
                "split": c["split"],
                "prompt": PROMPT.format(n=len(c["genes"]), genes=", ".join(c["genes"])),
                "genes": c["genes"],
            }
        )
    (OUT_DIR / "prompts.json").write_text(json.dumps(prompts, indent=2))
    print("Wrote library, benchmark, prompts.")
    return kegg, reactome, cases, universe, library


if __name__ == "__main__":
    main_build()
