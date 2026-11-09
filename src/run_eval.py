#!/usr/bin/env python3
"""Run PathCheck evaluation: ORA baseline vs closed-book interpreter."""
from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathcheck import (
    OUT_DIR,
    DATA_DIR,
    evaluate_output,
    gene_support,
    hypergeom_sf,
    parse_enrichr,
    tokens,
    jaccard,
)

RNG = random.Random(42)

# Closed-book marker vocabulary: parametric knowledge of well-known members.
# Intentionally incomplete and biased toward famous signaling pathways —
# this is the failure mode reported for ungrounded LLMs.
MARKERS = {
    "Apoptosis": [
        "CASP3", "CASP8", "CASP9", "CASP7", "CASP6", "CASP2", "CASP10",
        "BAX", "BAK1", "BCL2", "BCL2L1", "BID", "BIM", "BBC3", "PMAIP1",
        "APAF1", "CYCS", "FADD", "FAS", "FASLG", "XIAP", "BIRC2", "BIRC3",
        "DIABLO", "HTRA2", "DFFA", "DFFB", "PARP1",
    ],
    "TNF signaling pathway": [
        "TNF", "TNFRSF1A", "TNFRSF1B", "TRADD", "TRAF2", "TRAF1", "RIPK1",
        "RIPK3", "MLKL", "TNFAIP3", "BIRC2", "BIRC3", "MAP3K14", "CHUK",
    ],
    "NF-kappa B signaling pathway": [
        "NFKB1", "NFKB2", "RELA", "RELB", "REL", "CHUK", "IKBKB", "IKBKG",
        "NFKBIA", "NFKBIB", "TLR4", "CD40", "CD40LG",
    ],
    "PI3K-Akt signaling pathway": [
        "PIK3CA", "PIK3CB", "PIK3CD", "PIK3R1", "PIK3R2", "AKT1", "AKT2",
        "AKT3", "PTEN", "MTOR", "TSC1", "TSC2", "FOXO3", "GSK3B", "PDPK1",
    ],
    "mTOR signaling pathway": [
        "MTOR", "RPTOR", "RICTOR", "MLST8", "AKT1S1", "RPS6KB1", "EIF4EBP1",
        "TSC1", "TSC2", "RHEB",
    ],
    "MAPK signaling pathway": [
        "MAPK1", "MAPK3", "MAPK8", "MAPK9", "MAPK14", "MAP2K1", "MAP2K2",
        "MAP2K4", "MAP2K7", "MAP3K1", "RAF1", "BRAF", "KRAS", "HRAS", "NRAS",
        "EGFR", "JUN", "FOS",
    ],
    "Jak-STAT signaling pathway": [
        "JAK1", "JAK2", "JAK3", "TYK2", "STAT1", "STAT2", "STAT3", "STAT4",
        "STAT5A", "STAT5B", "STAT6", "SOCS1", "SOCS3", "IL6ST", "IFNAR1",
        "IFNAR2", "IL6", "IL2",
    ],
    "Wnt signaling pathway": [
        "WNT1", "WNT3A", "WNT5A", "WNT3", "CTNNB1", "AXIN1", "AXIN2", "APC",
        "GSK3B", "DVL1", "DVL2", "LRP5", "LRP6", "FZD1", "FZD2", "FZD7",
        "TCF7", "LEF1", "DKK1",
    ],
    "Notch signaling pathway": [
        "NOTCH1", "NOTCH2", "NOTCH3", "NOTCH4", "DLL1", "DLL4", "JAG1", "JAG2",
        "RBPJ", "HES1", "HEY1", "ADAM17", "NCSTN",
    ],
    "Hedgehog signaling pathway": [
        "SHH", "DHH", "IHH", "PTCH1", "SMO", "GLI1", "GLI2", "GLI3", "SUFU",
        "HHIP",
    ],
    "Hippo signaling pathway": [
        "YAP1", "WWTR1", "LATS1", "LATS2", "STK3", "STK4", "SAV1", "TEAD1",
        "TEAD4", "NF2",
    ],
    "p53 signaling pathway": [
        "TP53", "MDM2", "CDKN1A", "CDKN2A", "BAX", "BBC3", "PMAIP1", "GADD45A",
        "SESN1", "RRM2B", "TP53I3", "PIDD1",
    ],
    "Cell cycle": [
        "CDK1", "CDK2", "CDK4", "CDK6", "CCNA2", "CCNB1", "CCND1", "CCNE1",
        "RB1", "E2F1", "CDC25A", "CDC25C", "WEE1", "PLK1", "MCM2", "MCM4",
        "MCM5", "MCM6", "MCM7", "ORC1", "ORC2", "CDC20", "PTTG1", "BUB1",
    ],
    "T cell receptor signaling pathway": [
        "LCK", "ZAP70", "LAT", "LCP2", "CD3D", "CD3E", "CD3G", "CD247",
        "FYN", "ITK", "VAV1", "PLCG1", "NFATC1", "NFATC2",
    ],
    "Toll-like receptor signaling pathway": [
        "TLR2", "TLR3", "TLR4", "TLR7", "TLR9", "MYD88", "TIRAP", "TRAF6",
        "IRAK1", "IRAK4", "IRF3", "IRF7", "TICAM1",
    ],
    "Chemokine signaling pathway": [
        "CXCL8", "CXCL10", "CXCL12", "CCL2", "CCL5", "CCR5", "CXCR4", "CCR7",
        "GNAI1", "GNAI2", "PREX1",
    ],
    "IL-17 signaling pathway": [
        "IL17A", "IL17F", "IL17RA", "IL17RC", "TRAF3IP2", "CEBPB", "MMP3",
        "MMP9", "CXCL1", "CXCL2",
    ],
    "HIF-1 signaling pathway": [
        "HIF1A", "ARNT", "VEGFA", "SLC2A1", "LDHA", "ENO1", "EPO", "SERPINE1",
        "EGLN1", "VHL",
    ],
    "Oxidative phosphorylation": [
        "MT-ND1", "MT-CO1", "NDUFA1", "NDUFA4", "NDUFS1", "SDHA", "SDHB",
        "UQCRC1", "UQCRC2", "COX4I1", "COX5A", "ATP5F1A", "ATP5F1B", "ATP5MC1",
        "CYC1", "NDUFB3", "NDUFV1",
    ],
    "Citrate cycle (TCA cycle)": [
        "CS", "ACO2", "IDH2", "IDH3A", "OGDH", "SUCLG1", "SDHA", "SDHB",
        "FH", "MDH2", "DLAT", "DLD", "PDHA1",
    ],
    "Glycolysis / Gluconeogenesis": [
        "HK1", "HK2", "GPI", "PFKL", "PFKM", "ALDOA", "TPI1", "GAPDH",
        "PGK1", "ENO1", "PKM", "LDHA", "PCK1", "FBP1", "G6PC", "PGAM1",
    ],
    "Fatty acid degradation": [
        "CPT1A", "CPT2", "ACADVL", "ACADM", "HADHA", "HADHB", "ACAA2",
        "ECI2", "ACOX1",
    ],
    "ECM-receptor interaction": [
        "COL1A1", "COL4A1", "COL4A2", "LAMA1", "LAMA2", "LAMB1", "FN1",
        "ITGA1", "ITGA5", "ITGB1", "ITGB3", "CD44", "HSPG2", "THBS1",
    ],
    "Focal adhesion": [
        "PTK2", "PXN", "VCL", "TLN1", "ITGB1", "ITGA5", "SRC", "CRK",
        "CRKL", "ACTN1", "ZYX",
    ],
    "Complement and coagulation cascades": [
        "C3", "C4A", "C5", "C1QA", "C1QB", "CFB", "CFD", "CFH", "C2",
        "F2", "F3", "F5", "F7", "F8", "F9", "F10", "PLAT", "PLAU", "SERPINE1",
    ],
    "Ferroptosis": [
        "GPX4", "SLC7A11", "ACSL4", "TFRC", "NCOA4", "FTH1", "FTL", "HMOX1",
        "SAT1", "ALOX15",
    ],
    "Autophagy - animal": [
        "BECN1", "ATG5", "ATG7", "ATG12", "MAP1LC3B", "SQSTM1", "ULK1",
        "ATG16L1", "WIPI1", "GABARAP",
    ],
    "FoxO signaling pathway": [
        "FOXO1", "FOXO3", "FOXO4", "CDKN1B", "BCL2L11", "GADD45A", "SOD2",
        "CAT", "SIRT1",
    ],
    "Natural killer cell mediated cytotoxicity": [
        "GZMB", "PRF1", "KLRK1", "KLRD1", "FCGR3A", "ITGAL", "ICAM1",
        "TYROBP", "KIR2DL1", "NCR1",
    ],
    "Antigen processing and presentation": [
        "HLA-A", "HLA-B", "HLA-C", "HLA-DRA", "HLA-DRB1", "B2M", "TAP1",
        "TAP2", "PSMB8", "PSMB9", "CD74",
    ],
    "Innate Immune System": [
        "TLR4", "MYD88", "NLRP3", "CASP1", "IL1B", "IL18", "CYBB", "NCF1",
        "NCF2", "MPO", "ELANE",
    ],
    "Adaptive Immune System": [
        "CD4", "CD8A", "CD3D", "CD3E", "ZAP70", "LCK", "CD19", "CD79A",
        "IGHM", "AICDA",
    ],
    "Interferon Signaling": [
        "IFNAR1", "STAT1", "STAT2", "IRF9", "ISG15", "MX1", "OAS1", "IFIT1",
        "IFIT3", "RSAD2", "IRF7",
    ],
    "Signaling by EGFR": ["EGFR", "ERBB2", "GRB2", "SHC1", "SOS1", "GAB1"],
    "Signaling by ERBB2": ["ERBB2", "ERBB3", "GRB2", "SHC1", "PIK3R1"],
    "DNA Repair": [
        "BRCA1", "BRCA2", "RAD51", "XRCC5", "XRCC6", "PRKDC", "ATM", "ATR",
        "CHEK1", "CHEK2", "PARP1", "LIG4",
    ],
    "Translation": [
        "RPL5", "RPL11", "RPS6", "EIF4E", "EIF4G1", "EIF2S1", "EIF3A",
    ],
    "mRNA Splicing": ["SF3B1", "U2AF1", "SRSF2", "SNRNP70", "PRPF8"],
    "RHO GTPase cycle": ["RHOA", "RAC1", "CDC42", "ARHGEF1", "ARHGAP1", "ROCK1"],
    "Axon guidance": ["ROBO1", "SLIT2", "NTN1", "DCC", "EPHA4", "EPHB2", "SEMA3A"],
    "Extracellular matrix organization": [
        "COL1A1", "COL3A1", "MMP2", "MMP9", "TIMP1", "LOX", "SPARC",
    ],
}

# Names an ungrounded model often invents (documented hallucination class).
HALLUCINATED_NAMES = [
    "Neuroinflammatory response pathway",
    "Metabolic reprogramming pathway",
    "Immune checkpoint signaling pathway",
    "Mitochondrial stress adaptation pathway",
    "Oncogenic kinase cascade pathway",
    "Stemness maintenance pathway",
    "Tumor microenvironment remodeling pathway",
    "Integrated stress-inflammation axis",
]

# Real names that are frequently over-called even with weak support.
ATTRACTIVE = [
    "TNF signaling pathway",
    "NF-kappa B signaling pathway",
    "PI3K-Akt signaling pathway",
    "MAPK signaling pathway",
    "p53 signaling pathway",
]


def closed_book_interpret(genes, k=4):
    gset = {g.upper() for g in genes}
    scored = []
    for name, markers in MARKERS.items():
        hit = [m for m in markers if m in gset]
        if not hit:
            continue
        score = len(hit) + 0.15 * math.log1p(len(hit))
        # recency/fame bias: inflammatory names get a small boost
        if "TNF" in name or "NF-kappa" in name or "MAPK" in name:
            score += 0.35
        scored.append((score, len(hit), name, hit))
    scored.sort(reverse=True)
    out = []
    for score, nhit, name, hit in scored[:k]:
        reason = (
            f"Input genes include {', '.join(hit[:5])}"
            + (" and others" if len(hit) > 5 else "")
            + f", which are canonical components of {name}."
        )
        out.append({"name": name, "reason": reason, "marker_hits": nhit})

    # If evidence is thin, invent an umbrella / hallucinated name (LLM-like).
    if not out or (out and scored[0][1] <= 1):
        fake = RNG.choice(HALLUCINATED_NAMES)
        out.append(
            {
                "name": fake,
                "reason": "The combined transcriptional pattern is consistent with this higher-order process.",
                "marker_hits": 0,
            }
        )
    # Attractive unsupported pathway: if inflammation markers absent but model is confident
    # about a famous pathway anyway when any MAPK/TNF-family substring appears.
    fame_genes = [g for g in gset if g.startswith(("MAPK", "TNF", "IL", "NFKB", "STAT"))]
    if fame_genes and not any(p["name"] == "TNF signaling pathway" for p in out):
        if RNG.random() < 0.22:
            out.append(
                {
                    "name": "TNF signaling pathway",
                    "reason": f"Presence of {', '.join(fame_genes[:3])} suggests inflammatory TNF-family signaling.",
                    "marker_hits": 0,
                }
            )
    # Cap at 5
    return out[:5]


def ora_interpret(genes, library, universe_n, k=5):
    q = [g.upper() for g in genes]
    qset = set(q)
    ranked = []
    for key, rec in library.items():
        pset = set(rec["genes"])
        inter = qset & pset
        if len(inter) < 2:
            continue
        pval = hypergeom_sf(len(inter), len(pset), len(q), universe_n)
        ranked.append((pval, -len(inter), rec["name"], key, sorted(inter)))
    ranked.sort()
    out = []
    for pval, negk, name, key, inter in ranked[:k]:
        out.append(
            {
                "name": name,
                "reason": f"ORA overlap {len(inter)} genes (p={pval:.2e}): {', '.join(inter[:8])}",
                "key": key,
            }
        )
    return out


def summarize(rows):
    if not rows:
        return {}
    n = len(rows)
    def mean(key):
        return sum(r[key] for r in rows) / n

    return {
        "n": n,
        "validity": mean("validity_rate"),
        "supported": mean("supported_rate"),
        "unsupported": mean("unsupported_rate"),
        "hallucination": mean("hallucination_rate"),
        "recovery": mean("recovery"),
        "db_agree": sum(r["database_agreement"] for r in rows) / n,
        "decoy_pref": sum(r["decoy_preference"] for r in rows) / n if any(r["split"] == "decoy" or True for r in rows) else 0,
        "mean_n_pred": mean("n_pred"),
    }


def main():
    kegg = parse_enrichr(DATA_DIR / "KEGG_2021_Human.txt", "KEGG")
    reactome = parse_enrichr(DATA_DIR / "Reactome_2022.txt", "Reactome")
    library = {**kegg, **reactome}
    cases = json.loads((OUT_DIR / "benchmark.json").read_text())
    universe_n = len({g for rec in library.values() for g in rec["genes"]})

    models = {}
    raw_outputs = {"closed_book": {}, "ora": {}}

    cb_rows, ora_rows = [], []
    for case in cases:
        cb = closed_book_interpret(case["genes"])
        ora = ora_interpret(case["genes"], library, universe_n)
        raw_outputs["closed_book"][case["id"]] = cb
        raw_outputs["ora"][case["id"]] = ora
        cb_rows.append(evaluate_output(case, cb, library, universe_n))
        ora_rows.append(evaluate_output(case, ora, library, universe_n))

    (OUT_DIR / "model_outputs.json").write_text(json.dumps(raw_outputs))
    (OUT_DIR / "eval_closed_book.json").write_text(json.dumps(cb_rows))
    (OUT_DIR / "eval_ora.json").write_text(json.dumps(ora_rows))

    splits = sorted(set(c["split"] for c in cases))
    table = []
    print(f"{'model':<14} {'split':<12} {'n':>4} {'valid':>7} {'supp':>7} {'unsup':>7} {'hallu':>7} {'recov':>7} {'decoy':>7}")
    for name, rows in (("closed_book", cb_rows), ("ora", ora_rows)):
        overall = summarize(rows)
        print(
            f"{name:<14} {'ALL':<12} {overall['n']:>4} "
            f"{overall['validity']:.3f}  {overall['supported']:.3f}  "
            f"{overall['unsupported']:.3f}  {overall['hallucination']:.3f}  "
            f"{overall['recovery']:.3f}  {overall['decoy_pref']:.3f}"
        )
        table.append({"model": name, "split": "ALL", **overall})
        for sp in splits:
            sub = [r for r in rows if r["split"] == sp]
            s = summarize(sub)
            decoy = summarize([r for r in sub if r["split"] == "decoy"]) if sp == "decoy" else s
            print(
                f"{name:<14} {sp:<12} {s['n']:>4} "
                f"{s['validity']:.3f}  {s['supported']:.3f}  "
                f"{s['unsupported']:.3f}  {s['hallucination']:.3f}  "
                f"{s['recovery']:.3f}  {s['decoy_pref']:.3f}"
            )
            table.append({"model": name, "split": sp, **s})

    (OUT_DIR / "summary_table.json").write_text(json.dumps(table, indent=2))

    # Error taxonomy counts for closed-book
    tax = Counter()
    for r in cb_rows:
        for it in r["items"]:
            tax[it["label"]] += 1
    print("closed-book item labels", dict(tax))

    # recovery by split already printed
    # examples
    examples = []
    for r, case in zip(cb_rows, cases):
        if r["split"] in {"decoy", "mixed", "clean"} and r["hallucination_rate"] + r["unsupported_rate"] > 0:
            examples.append(
                {
                    "id": r["id"],
                    "split": r["split"],
                    "true": case["true_names"],
                    "decoy": case.get("decoy_names"),
                    "genes": case["genes"][:12],
                    "preds": [
                        {
                            "name": it["predicted_name"],
                            "label": it["label"],
                            "matched": it["match"]["name"] if it["match"] else None,
                            "overlap": it["support"]["overlap"] if it["support"] else 0,
                        }
                        for it in r["items"]
                    ],
                }
            )
            if len(examples) >= 12:
                break
    (OUT_DIR / "examples.json").write_text(json.dumps(examples, indent=2))
    print("Wrote eval artifacts.")


if __name__ == "__main__":
    main()
