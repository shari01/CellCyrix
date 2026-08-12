#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
annotation_reference_demo.py  --  STANDALONE DEMO. Does NOT touch the pipeline.

Purpose
-------
Show what the FINAL annotation output would look like IF the reference folders in
`Celltype_Markers_References/` were wired in as (A) a 5th "ScType" consensus voter
and (B) a disease-signature layer. Nothing here is imported by the real pipeline;
run it, then open the `annotation_demo_output/` folder to see the shape of things.

What is REAL vs SYNTHETIC in this demo
--------------------------------------
  * Cluster marker genes .............. SYNTHETIC  (6 hand-made, biologically typical clusters)
  * ScType voter (celltype_sctype) .... REAL       (scored against TIS_CELL_markers_v3/)
  * Disease-signature scoring ......... REAL       (scored against Disease_specific_markers/)
  * CellTypist / SingleR / KB / PubMed  SYNTHETIC  (placeholders, only to show the table layout)
  * Consensus label ................... REAL logic  (majority vote of the 5 method columns)

So the columns that come FROM the reference folders (ScType + disease) are computed
honestly; the other voters are stubs so you can see how everything lines up in one table.

Run:  python tests/annotation_reference_demo.py
Out:  outputs/annotation_demo_output/   (CSVs + DEMO_SUMMARY.html + README.txt)
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
# Static marker reference lives in the package's shared_reference/ bucket.
REF = (
    PKG_ROOT
    / "shared_reference"
    / "single_cell_pipeline_agent_datasets"
    / "celltype_markers_references"
)
OUT = PKG_ROOT / "outputs" / "annotation_demo_output"
OUT.mkdir(parents=True, exist_ok=True)

SCTYPE_DB = REF / "TIS_CELL_markers_v3" / "master_celltype_markers_long.csv"
DISEASE_DIR = REF / "Disease_specific_markers"
DISEASE_NAME = "Squamous cell carcinoma"  # study disease (cervical cohort); swap freely
DISEASE_CSV = (
    DISEASE_DIR
    / "disease_gene_files"
    / "Cancer_Neoplasms"
    / "Squamous_cell_carcinoma__C0007137.csv"
)

# --------------------------------------------------------------------------------------
# 1. SYNTHETIC clusters  (what rank_genes_groups would hand us: top up-genes per cluster)
# --------------------------------------------------------------------------------------
CLUSTERS = {
    "0": [
        "EPCAM",
        "KRT8",
        "KRT18",
        "KRT19",
        "KRT5",
        "KRT17",
        "KRT14",
        "CDH1",
        "SFN",
        "KRT13",
        "TP53",
        "CDKN2A",
        "EGFR",
        "MKI67",
        "PTGS2",
    ],  # epithelial / malignant
    "1": [
        "CD3D",
        "CD3E",
        "CD3G",
        "CD2",
        "TRAC",
        "CD8A",
        "IL7R",
        "CCL5",
        "GZMK",
        "CD7",
        "LCK",
        "CD247",
    ],  # T cell
    "2": [
        "CD79A",
        "CD79B",
        "MS4A1",
        "CD19",
        "IGHM",
        "CD74",
        "HLA-DRA",
        "BANK1",
        "TCL1A",
        "VPREB3",
    ],  # B cell
    "3": [
        "CD68",
        "LYZ",
        "CD14",
        "FCGR3A",
        "C1QA",
        "C1QB",
        "C1QC",
        "AIF1",
        "ITGAM",
        "CSF1R",
        "APOE",
        "TYROBP",
    ],  # macrophage / myeloid
    "4": [
        "COL1A1",
        "COL1A2",
        "COL3A1",
        "DCN",
        "LUM",
        "PDGFRB",
        "PDGFRA",
        "ACTA2",
        "FN1",
        "COL6A1",
    ],  # fibroblast
    "5": [
        "PECAM1",
        "VWF",
        "CLDN5",
        "CDH5",
        "FLT1",
        "KDR",
        "EGFL7",
        "RAMP2",
    ],  # endothelial
}


# --------------------------------------------------------------------------------------
# 2. ScType voter  --  REAL: score each cluster's markers against TIS_CELL_markers_v3
# --------------------------------------------------------------------------------------
def load_sctype_db() -> pd.DataFrame:
    m = pd.read_csv(SCTYPE_DB)
    m["gene_symbol"] = m["gene_symbol"].astype(str).str.upper()
    m["marker_score"] = (
        pd.to_numeric(m["marker_score"], errors="coerce").fillna(0.0).clip(lower=0)
    )
    return m


def sctype_score_cluster(cluster_genes, db, min_overlap=2, top_k=3):
    """Marker-overlap ScType-style score. (No expression matrix in a demo, so we score
    on the marker GENE SET; the real pipeline would z-scale expression the same way.)
    score(celltype) = sum(1 + marker_score) over overlapping genes / sqrt(panel_size)."""
    genes = {g.upper() for g in cluster_genes}
    rows = []
    for ct, sub in db.groupby("cell_type"):
        panel = sub.drop_duplicates("gene_symbol").set_index("gene_symbol")
        overlap = [g for g in genes if g in panel.index]
        if len(overlap) < min_overlap:
            continue
        w = (1.0 + panel.loc[overlap, "marker_score"]).sum()
        norm = float(w / math.sqrt(len(panel)))
        rows.append(
            {
                "cell_type": ct,
                "organ": sub["organ"].iloc[0],
                "germ_layer": sub["germ_layer"].iloc[0],
                "n_overlap": len(overlap),
                "sctype_score": round(norm, 4),
                "overlap_genes": ";".join(sorted(overlap)),
            }
        )
    rows.sort(key=lambda r: r["sctype_score"], reverse=True)
    top = rows[:top_k]
    if not top:
        return None, 0.0, []
    best = top[0]
    second = top[1]["sctype_score"] if len(top) > 1 else 0.0
    conf = (
        0.0
        if best["sctype_score"] == 0
        else round((best["sctype_score"] - second) / best["sctype_score"], 3)
    )
    return best["cell_type"], conf, top


# --------------------------------------------------------------------------------------
# 3. Disease-signature scoring  --  REAL: overlap cluster markers with DisGeNET panel
# --------------------------------------------------------------------------------------
def load_disease_panel() -> dict:
    d = pd.read_csv(DISEASE_CSV)
    d["gene_symbol"] = d["gene_symbol"].astype(str).str.upper()
    d["gda_score"] = pd.to_numeric(d["gda_score"], errors="coerce").fillna(0.0)
    return dict(zip(d["gene_symbol"], d["gda_score"], strict=True))


def disease_score_cluster(cluster_genes, panel: dict):
    genes = {g.upper() for g in cluster_genes}
    hits = sorted(genes & set(panel), key=lambda g: panel[g], reverse=True)
    score = round(sum(panel[g] for g in hits), 4)  # gda-weighted enrichment
    return score, hits


# --------------------------------------------------------------------------------------
# 4. SYNTHETIC placeholder votes for the other 4 voters (only to show the table layout).
#    Two deliberate disagreements so you can watch the consensus resolve them.
# --------------------------------------------------------------------------------------
MOCK_VOTES = {  # cluster: (celltypist, singler, knowledge_based, pubmed)  --  None = abstain
    "0": ("Epithelial cells", "Epithelial_cells", "epithelial cell", "epithelial cell"),
    "1": ("T cells", "T_cells", "T cell", "T cell"),
    "2": ("B cells", "B_cells", "B cell", "B cell"),
    "3": (
        "Monocytes",
        "Macrophage",
        "macrophage",
        "macrophage",
    ),  # celltypist disagrees
    "4": ("Fibroblasts", "Fibroblasts", "fibroblast", None),  # pubmed abstains
    "5": (
        "Endothelial cells",
        "Endothelial_cells",
        "endothelial cell",
        "endothelial cell",
    ),
}
MOCK_CONF = {
    "celltypist": 0.82,
    "singler": 0.70,
    "knowledge_based": 0.75,
    "pubmed": 0.68,
}

# tiny label harmoniser so votes are comparable
_SYN = {
    "epithelial cells": "epithelial cell",
    "epithelial_cells": "epithelial cell",
    "epithelial cell": "epithelial cell",
    "t cells": "T cell",
    "t_cells": "T cell",
    "t cell": "T cell",
    "b cells": "B cell",
    "b_cells": "B cell",
    "b cell": "B cell",
    "macrophage": "macrophage",
    "macrophages": "macrophage",
    "monocytes": "monocyte",
    "monocyte": "monocyte",
    "fibroblasts": "fibroblast",
    "fibroblast": "fibroblast",
    "endothelial cells": "endothelial cell",
    "endothelial_cells": "endothelial cell",
    "endothelial cell": "endothelial cell",
}


def harmonize(label):
    if label is None:
        return None
    return _SYN.get(str(label).strip().lower(), str(label).strip())


def consensus(method_labels: dict, method_conf: dict):
    """Majority vote of harmonised labels; ties broken by summed confidence."""
    votes = defaultdict(float)
    counts = defaultdict(int)
    for method, lab in method_labels.items():
        h = harmonize(lab)
        if h is None:
            continue
        counts[h] += 1
        votes[h] += method_conf.get(method, 0.5)
    if not counts:
        return "unknown", 0, 0.0
    best = max(counts, key=lambda k: (counts[k], votes[k]))
    n_agree = counts[best]
    n_voting = sum(1 for v in method_labels.values() if harmonize(v) is not None)
    return best, n_agree, round(n_agree / max(n_voting, 1), 3)


# --------------------------------------------------------------------------------------
# 5. Build the outputs
# --------------------------------------------------------------------------------------
def main():
    print(f"[demo] reading references from: {REF}")
    db = load_sctype_db()
    panel = load_disease_panel()
    print(
        f"[demo] ScType DB: {db['cell_type'].nunique()} cell types | "
        f"disease '{DISEASE_NAME}': {len(panel)} genes"
    )

    consensus_rows, sctype_detail_rows, disease_rows = [], [], []

    for cl, markers in CLUSTERS.items():
        # (A) REAL ScType
        sctype_label, sctype_conf, sctype_top = sctype_score_cluster(markers, db)
        for rank, cand in enumerate(sctype_top, 1):
            sctype_detail_rows.append({"cluster": cl, "rank": rank, **cand})

        # (B) REAL disease score
        dscore, dhits = disease_score_cluster(markers, panel)
        disease_rows.append(
            {
                "cluster": cl,
                "disease": DISEASE_NAME,
                "disease_score": dscore,
                "n_disease_genes_hit": len(dhits),
                "genes_hit": ";".join(dhits),
            }
        )

        # (placeholder) other voters
        ct, sr, kb, pm = MOCK_VOTES[cl]
        method_labels = {
            "celltypist": ct,
            "singler": sr,
            "knowledge_based": kb,
            "pubmed": pm,
            "sctype": sctype_label,
        }
        method_conf = {**MOCK_CONF, "sctype": sctype_conf}
        final, n_agree, agree_frac = consensus(method_labels, method_conf)

        consensus_rows.append(
            {
                "cluster": cl,
                "n_markers": len(markers),
                "celltypist": ct,
                "celltypist_conf": MOCK_CONF["celltypist"],
                "singler": sr,
                "singler_conf": MOCK_CONF["singler"],
                "knowledge_based": kb,
                "knowledge_based_conf": MOCK_CONF["knowledge_based"],
                "pubmed": pm if pm else "(abstain)",
                "pubmed_conf": MOCK_CONF["pubmed"] if pm else 0.0,
                "sctype": sctype_label,
                "sctype_conf": sctype_conf,  # <-- FROM reference folder
                "CONSENSUS_LABEL": final,
                "n_voters_agree": n_agree,
                "consensus_confidence": agree_frac,
                "disease_relevance_score": dscore,  # <-- FROM reference folder
            }
        )

    cons = pd.DataFrame(consensus_rows)
    sctype_detail = pd.DataFrame(sctype_detail_rows)
    disease = pd.DataFrame(disease_rows).sort_values("disease_score", ascending=False)

    # (B2) DE-recall benchmark  --  illustrative: does a nominated DE gene list recover known disease genes?
    nominated = set(CLUSTERS["0"]) | {
        "MMP9",
        "KRT6A",
        "S100A8",
    }  # pretend pseudobulk-DE output
    known = set(panel)
    recovered = nominated & known
    de_recall = pd.DataFrame(
        [
            {
                "disease": DISEASE_NAME,
                "n_known_genes": len(known),
                "n_nominated_by_DE": len(nominated),
                "n_recovered": len(recovered),
                "recall_pct": round(100 * len(recovered) / max(len(known), 1), 1),
                "recovered_genes": ";".join(sorted(recovered)),
            }
        ]
    )

    # write CSVs
    cons.to_csv(OUT / "01_consensus_annotation_ALL_METHODS.csv", index=False)
    sctype_detail.to_csv(OUT / "02_sctype_scores_detail.csv", index=False)
    disease.to_csv(OUT / "03_disease_signature_scores.csv", index=False)
    de_recall.to_csv(OUT / "04_disease_DE_recall_benchmark.csv", index=False)
    write_readme()
    write_html(cons, sctype_detail, disease, de_recall)

    print("\n[demo] wrote:")
    for f in sorted(OUT.iterdir()):
        print("   ", f.name)
    print(f"\n[demo] open: {OUT / 'DEMO_SUMMARY.html'}")
    print("\n--- CONSENSUS TABLE ---")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(
            cons[
                [
                    "cluster",
                    "sctype",
                    "sctype_conf",
                    "CONSENSUS_LABEL",
                    "n_voters_agree",
                    "disease_relevance_score",
                ]
            ].to_string(index=False)
        )
    print(
        "\n--- DISEASE RELEVANCE (which cell pop is enriched for the disease genes) ---"
    )
    print(
        disease[
            ["cluster", "disease_score", "n_disease_genes_hit", "genes_hit"]
        ].to_string(index=False)
    )


def write_readme():
    (OUT / "README.txt").write_text(
        "ANNOTATION REFERENCE DEMO -- what these outputs are\n"
        "===================================================\n\n"
        "This folder is produced by annotation_reference_demo.py. It is a MOCK preview\n"
        "of what the pipeline output would look like if the Celltype_Markers_References/\n"
        "folders were wired in. Nothing here was produced by the real pipeline.\n\n"
        "REAL (computed from the reference folders):\n"
        "  * sctype / sctype_conf columns        <- TIS_CELL_markers_v3/\n"
        "  * disease_relevance_score + file 03   <- Disease_specific_markers/ (DisGeNET)\n"
        "  * file 02 (ScType candidate detail)   <- TIS_CELL_markers_v3/\n\n"
        "SYNTHETIC (placeholders, only to show table layout):\n"
        "  * cluster marker genes (6 typical clusters)\n"
        "  * celltypist / singler / knowledge_based / pubmed vote columns\n\n"
        "Files:\n"
        "  01_consensus_annotation_ALL_METHODS.csv  main per-cluster table, all voters + consensus\n"
        "  02_sctype_scores_detail.csv              top-3 ScType candidates per cluster (shows scoring)\n"
        "  03_disease_signature_scores.csv          per-cluster disease-gene enrichment\n"
        "  04_disease_DE_recall_benchmark.csv       does DE recover known disease genes (illustrative)\n"
        "  DEMO_SUMMARY.html                        visual summary -- open this first\n",
        encoding="utf-8",
    )


def write_html(cons, sctype_detail, disease, de_recall):
    def tbl(df):
        return df.to_html(index=False, border=0, classes="t", escape=False)

    html = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Annotation reference demo</title>
<style>
  :root{{--bg:#f7f7f8;--fg:#1c1c22;--mut:#6b6b76;--line:#e2e2e8;--accent:#3b6db5;--warn:#b5793b;--card:#fff;}}
  @media(prefers-color-scheme:dark){{:root{{--bg:#15161a;--fg:#e8e8ee;--mut:#9a9aa6;--line:#2b2c33;--accent:#7aa7e0;--warn:#e0b07a;--card:#1d1e24;}}}}
  *{{box-sizing:border-box}} body{{margin:0;font:15px/1.55 system-ui,Segoe UI,sans-serif;background:var(--bg);color:var(--fg)}}
  .wrap{{max-width:1100px;margin:0 auto;padding:32px 20px 64px}}
  h1{{font-size:1.55rem;margin:0 0 4px}} h2{{font-size:1.1rem;margin:34px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}}
  .sub{{color:var(--mut);margin:0 0 18px}}
  .legend{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:14px 0}}
  .legend b{{color:var(--accent)}} .legend .w{{color:var(--warn)}}
  .scroll{{overflow-x:auto;border:1px solid var(--line);border-radius:10px}}
  table.t{{border-collapse:collapse;width:100%;font-size:13px;background:var(--card)}}
  table.t th,table.t td{{padding:7px 10px;text-align:left;white-space:nowrap;border-bottom:1px solid var(--line)}}
  table.t th{{background:rgba(127,127,127,.08);font-weight:600}}
  table.t tr:last-child td{{border-bottom:0}}
  code{{background:rgba(127,127,127,.15);padding:1px 5px;border-radius:4px}}
</style></head><body><div class=wrap>
<h1>Annotation reference demo &mdash; mock output</h1>
<p class=sub>What the pipeline output would look like with the <code>Celltype_Markers_References/</code> folders wired in.
Preview only &mdash; not a real run.</p>
<div class=legend>
<b>REAL</b> (computed from the reference folders): the <b>sctype</b> / <b>sctype_conf</b> columns and the whole
disease section (files 02 &amp; 03). &nbsp; <span class=w>SYNTHETIC</span> (placeholders, just to show the layout):
the cluster marker genes and the celltypist / singler / knowledge_based / pubmed columns.
</div>

<h2>1 &middot; Consensus table (all voters &rarr; final label)</h2>
<p class=sub>Cluster 3: celltypist says <code>Monocytes</code> but the other voters say macrophage &rarr; consensus resolves to
<b>macrophage</b>. Cluster 4: pubmed abstains &rarr; consensus uses the remaining 4.</p>
<div class=scroll>{tbl(cons)}</div>

<h2>2 &middot; ScType scoring detail (from TIS_CELL_markers_v3/)</h2>
<p class=sub>Top-3 candidate cell types per cluster and the overlapping markers that drove the score. This is exactly
how the reference folder is consumed by the ScType voter.</p>
<div class=scroll>{tbl(sctype_detail)}</div>

<h2>3 &middot; Disease relevance (from Disease_specific_markers/ &mdash; {DISEASE_NAME})</h2>
<p class=sub>Not a per-cell disease label &mdash; a score of which cell population is enriched for the disease's known genes.
The epithelial/malignant cluster lights up, as expected for a carcinoma.</p>
<div class=scroll>{tbl(disease)}</div>

<h2>4 &middot; DE recall benchmark (illustrative)</h2>
<p class=sub>Does a nominated DE gene list recover the disease's known genes? (Here the DE list is faked; in the real
pipeline it would be your pseudobulk-DESeq2 output.)</p>
<div class=scroll>{tbl(de_recall)}</div>
</div></body></html>"""
    (OUT / "DEMO_SUMMARY.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
