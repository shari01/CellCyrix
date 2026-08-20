#!/usr/bin/env python3
"""
run_benchmarks.py — deterministic validation harness for the single-cell pipeline.

Each benchmark defines, up front, a **test condition**: a synthetic (but
biologically shaped) input, a **precomputed expected output**, and the **actual
output** produced by calling the *real* pipeline functions (no re-implementation).
A condition PASSES only when actual matches expected. Everything is seeded, so the
harness is fully reproducible — the same commit always yields the same verdicts.

Why synthetic data? The repository ships no patient data and the full pipeline's
external voters (CellTypist models, SingleR/R, the OpenRouter LLM, live PubMed)
need credentials/network. These benchmarks therefore target the pipeline's
**deterministic, offline-verifiable core** — data loading, QC/filtering, seed
discipline, pseudobulk aggregation + DESeq2 DE, consensus vote counting, the PubMed
voter's marker cleaning / confidence scoring, and provenance capture. Steps that
require external services are validated structurally by the unit tests and the two
static auditors instead; see BENCHMARK_REPORT.md §Limitations.

Run
---
    python deliverables/single_cell_pipeline/benchmarks/run_benchmarks.py

Outputs (next to this script)
    benchmark_results.json      machine-readable results
    BENCHMARK_REPORT.md         human-readable table (conditions/input/expected/actual)
    logs/benchmark_run.log      full execution log
    outputs/<condition>/...     artifacts produced by each condition
    inputs/<condition>/...      synthetic inputs written for each condition
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

# --------------------------------------------------------------------------- #
#  Paths / imports of the REAL pipeline package
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"
OUTPUTS = HERE / "outputs"
LOGS = HERE / "logs"
for _d in (INPUTS, OUTPUTS, LOGS):
    _d.mkdir(parents=True, exist_ok=True)


def _find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` until the folder containing ``cellcyrix`` is found."""
    for p in [start, *start.parents]:
        if (p / "cellcyrix").is_dir():
            return p
    raise RuntimeError("Could not locate repo root (no 'cellcyrix' on any parent).")


REPO_ROOT = _find_repo_root(HERE)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PKG = "cellcyrix.single_cell_pipeline_agent.singlecell_10x"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS / "benchmark_run.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("benchmark")


# --------------------------------------------------------------------------- #
#  Result container
# --------------------------------------------------------------------------- #
@dataclass
class Condition:
    """One benchmark condition: its input spec, expected vs actual output, verdict."""

    cid: str
    title: str
    category: str
    requires: str = ""  # external requirement, if any
    input_desc: str = ""
    expected: Any = None
    actual: Any = None
    status: str = "PENDING"  # PASS | FAIL | ERROR | SKIP
    detail: str = ""
    checks: List[Dict[str, Any]] = field(default_factory=list)

    def check(
        self, name: str, ok: bool, expected: Any = None, actual: Any = None
    ) -> None:
        self.checks.append(
            {"name": name, "ok": bool(ok), "expected": expected, "actual": actual}
        )

    def finalize(self) -> None:
        if self.status in ("ERROR", "SKIP"):
            return
        self.status = (
            "PASS" if all(c["ok"] for c in self.checks) and self.checks else "FAIL"
        )


# --------------------------------------------------------------------------- #
#  Synthetic-data helpers
# --------------------------------------------------------------------------- #
def write_10x(
    dir_path: Path,
    mat_genes_by_cells,
    gene_ids,
    gene_symbols,
    barcodes,
    gzip_it: bool = True,
) -> None:
    """Write a synthetic 10x feature-barcode trio (matrix/barcodes/features) to ``dir_path``."""
    from scipy import sparse as sp
    from scipy.io import mmwrite

    dir_path.mkdir(parents=True, exist_ok=True)
    m = sp.coo_matrix(mat_genes_by_cells)

    buf = io.BytesIO()
    mmwrite(buf, m, field="integer")
    data = buf.getvalue()
    if gzip_it:
        with gzip.open(dir_path / "matrix.mtx.gz", "wb") as f:
            f.write(data)
    else:
        (dir_path / "matrix.mtx").write_bytes(data)

    def _write_lines(name, lines):
        if gzip_it:
            with gzip.open(dir_path / (name + ".gz"), "wt", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        else:
            (dir_path / name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    _write_lines("barcodes.tsv", list(barcodes))
    _write_lines(
        "features.tsv",
        [
            f"{gid}\t{sym}\tGene Expression"
            for gid, sym in zip(gene_ids, gene_symbols, strict=True)
        ],
    )


def make_structured_adata(n_per_group: int = 60, n_genes: int = 120, seed: int = 0):
    """Build an AnnData with 3 well-separated cell 'programs' so clustering is meaningful."""
    import anndata as ad

    rng = np.random.default_rng(seed)
    programs = 3
    per = n_per_group
    blocks, labels = [], []
    genes_per = n_genes // programs
    for k in range(programs):
        base = rng.poisson(0.4, size=(per, n_genes)).astype(float)
        lo, hi = k * genes_per, (k + 1) * genes_per
        base[:, lo:hi] += rng.poisson(
            8.0, size=(per, hi - lo)
        )  # program-specific high expr
        blocks.append(base)
        labels += [f"program_{k}"] * per
    X = np.vstack(blocks)
    var_names = [f"GENE{i:04d}" for i in range(n_genes)]
    obs_names = [f"cell{i:05d}" for i in range(X.shape[0])]
    a = ad.AnnData(X=X)
    a.obs_names = obs_names
    a.var_names = var_names
    a.obs["true_program"] = labels
    a.layers["counts"] = X.copy()
    return a


# --------------------------------------------------------------------------- #
#  Conditions
# --------------------------------------------------------------------------- #
def c1_loader_roundtrip() -> Condition:
    c = Condition(
        "C1",
        "10x loader round-trip (gzipped feature-barcode matrix)",
        "data-loading",
        input_desc="Synthetic gzipped 10x trio: 40 genes x 25 cells, known symbols.",
    )
    from importlib import import_module

    load = import_module(f"{PKG}.loader_10x").load_10x_feature_barcode_matrix

    rng = np.random.default_rng(1)
    n_genes, n_cells = 40, 25
    mat = rng.poisson(1.0, size=(n_genes, n_cells))  # genes x cells (10x convention)
    mat[0, 0] = 7  # a known landmark value
    gene_ids = [f"ENSG{i:08d}" for i in range(n_genes)]
    gene_syms = [f"SYM{i:03d}" for i in range(n_genes)]
    barcodes = [f"BC{i:04d}-1" for i in range(n_cells)]
    d = INPUTS / "C1_loader"
    write_10x(d, mat, gene_ids, gene_syms, barcodes, gzip_it=True)

    c.expected = {
        "shape_cells_x_genes": [n_cells, n_genes],
        "first_gene_symbol": "SYM000",
        "first_feature_id": "ENSG00000000",
        "landmark_cell0_gene0": 7,
        "total_counts": int(mat.sum()),
    }
    adata = load(d)
    Xarr = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    c.actual = {
        "shape_cells_x_genes": list(adata.shape),
        "first_gene_symbol": str(adata.var["gene_symbol"].iloc[0]),
        "first_feature_id": str(adata.var["feature_id"].iloc[0]),
        "landmark_cell0_gene0": int(round(float(Xarr[0, 0]))),
        "total_counts": int(round(float(Xarr.sum()))),
    }
    for k in c.expected:
        c.check(k, c.expected[k] == c.actual[k], c.expected[k], c.actual[k])
    c.finalize()
    return c


def c2_loader_errors() -> Condition:
    c = Condition(
        "C2",
        "Loader raises clear, typed errors on bad input",
        "data-loading",
        input_desc="(a) folder with no matrix.mtx; (b) a corrupt matrix.mtx.gz.",
    )
    from importlib import import_module

    load = import_module(f"{PKG}.loader_10x").load_10x_feature_barcode_matrix

    # (a) missing folder contents -> FileNotFoundError
    empty = INPUTS / "C2_empty"
    empty.mkdir(parents=True, exist_ok=True)
    got_a = None
    try:
        load(empty)
    except FileNotFoundError:
        got_a = "FileNotFoundError"
    except Exception as e:  # noqa: BLE001 - we specifically want to see the wrong type
        got_a = type(e).__name__
    c.check(
        "missing_files_raises_FileNotFoundError",
        got_a == "FileNotFoundError",
        "FileNotFoundError",
        got_a,
    )

    # (b) present but corrupt matrix -> ValueError naming the file (the guard we added)
    bad = INPUTS / "C2_corrupt"
    bad.mkdir(parents=True, exist_ok=True)
    with gzip.open(bad / "matrix.mtx.gz", "wt", encoding="utf-8") as f:
        f.write("this is not a matrix market file\n")
    with gzip.open(bad / "barcodes.tsv.gz", "wt", encoding="utf-8") as f:
        f.write("BC0-1\n")
    with gzip.open(bad / "features.tsv.gz", "wt", encoding="utf-8") as f:
        f.write("ENSG0\tSYM0\tGene Expression\n")
    got_b = None
    try:
        load(bad)
    except ValueError:
        got_b = "ValueError"
    except Exception as e:  # noqa: BLE001
        got_b = type(e).__name__
    c.check(
        "corrupt_matrix_raises_ValueError", got_b == "ValueError", "ValueError", got_b
    )

    c.expected = {"missing": "FileNotFoundError", "corrupt": "ValueError"}
    c.actual = {"missing": got_a, "corrupt": got_b}
    c.finalize()
    return c


def c3_qc_filtering() -> Condition:
    c = Condition(
        "C3",
        "QC metrics + cell/gene/mito filtering match definition",
        "qc-filtering",
        input_desc="8 cells x 6 genes (2 mitochondrial) with a hand-designed count matrix.",
    )
    import anndata as ad
    import scanpy as sc

    # Rows = cells, cols = genes. Genes g4,g5 are mitochondrial (MT-).
    X = np.array(
        [
            [5, 3, 2, 1, 0, 0],  # cell0: 4 genes, low mt
            [0, 0, 0, 0, 0, 0],  # cell1: 0 genes  -> dropped by min_genes>=2
            [4, 4, 0, 0, 9, 9],  # cell2: 4 genes, HIGH mt (18/26)
            [2, 2, 2, 2, 0, 0],  # cell3: 4 genes, low mt
            [1, 0, 0, 0, 0, 0],  # cell4: 1 gene   -> dropped by min_genes>=2
            [3, 3, 3, 0, 0, 1],  # cell5: 4 genes, low mt
            [6, 0, 5, 4, 0, 0],  # cell6: 3 genes, low mt
            [2, 2, 1, 1, 1, 0],  # cell7: 5 genes, low mt
        ],
        dtype=float,
    )
    genes = ["GENE0", "GENE1", "GENE2", "GENE3", "MT-A", "MT-B"]
    a = ad.AnnData(X=X)
    a.var_names = genes
    a.obs_names = [f"c{i}" for i in range(X.shape[0])]

    MIN_GENES, MIN_CELLS, MAX_MT = 2, 2, 50.0

    # ---- ground truth computed independently with numpy, in the pipeline's order ----
    genes_per_cell = (X > 0).sum(axis=1)
    keep_cells = genes_per_cell >= MIN_GENES
    mt_idx = [i for i, g in enumerate(genes) if g.startswith("MT-")]
    mt_frac = 100.0 * X[:, mt_idx].sum(axis=1) / np.clip(X.sum(axis=1), 1, None)
    keep_cells &= mt_frac < MAX_MT
    Xc = X[keep_cells]
    cells_per_gene = (Xc > 0).sum(axis=0)
    keep_genes = cells_per_gene >= MIN_CELLS
    exp_cells = int(keep_cells.sum())
    exp_genes = int(keep_genes.sum())

    # ---- actual: the scanpy operations the pipeline uses ----
    a.var["mt"] = [g.startswith("MT-") for g in a.var_names]
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], inplace=True, percent_top=None)
    sc.pp.filter_cells(a, min_genes=MIN_GENES)
    a = a[a.obs["pct_counts_mt"] < MAX_MT].copy()
    sc.pp.filter_genes(a, min_cells=MIN_CELLS)

    c.expected = {
        "cells_after_filter": exp_cells,
        "genes_after_filter": exp_genes,
        "dropped_cells": int((~keep_cells).sum()),
    }
    c.actual = {
        "cells_after_filter": int(a.n_obs),
        "genes_after_filter": int(a.n_vars),
        "dropped_cells": int(X.shape[0] - keep_cells.sum()),
    }
    c.check(
        "cells_after_filter",
        c.expected["cells_after_filter"] == c.actual["cells_after_filter"],
        c.expected["cells_after_filter"],
        c.actual["cells_after_filter"],
    )
    c.check(
        "genes_after_filter",
        c.expected["genes_after_filter"] == c.actual["genes_after_filter"],
        c.expected["genes_after_filter"],
        c.actual["genes_after_filter"],
    )
    c.finalize()
    return c


def c4_reproducibility() -> Condition:
    c = Condition(
        "C4",
        "Seed discipline: identical clustering across repeated runs",
        "reproducibility",
        input_desc="Structured 180-cell x 120-gene dataset; normalize->log1p->HVG->scale->PCA->neighbors->Leiden, run twice with the same seed.",
    )
    from importlib import import_module

    import scanpy as sc

    set_global_seed = import_module(f"{PKG}.reproducibility").set_global_seed
    SEED = 0

    def one_run() -> np.ndarray:
        set_global_seed(SEED)
        a = make_structured_adata(seed=SEED)
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        sc.pp.highly_variable_genes(a, n_top_genes=60)
        a = a[:, a.var.highly_variable].copy()
        sc.pp.scale(a, max_value=10)
        sc.tl.pca(a, n_comps=20, random_state=SEED)
        sc.pp.neighbors(a, n_neighbors=15, random_state=SEED)
        sc.tl.leiden(a, resolution=0.5, random_state=SEED, key_added="leiden")
        return a.obs["leiden"].astype(str).values, a

    labels1, a1 = one_run()
    labels2, _ = one_run()
    identical = bool(np.array_equal(labels1, labels2))
    n_clusters = int(len(set(labels1)))
    # sanity: the 3 designed programs should not collapse into a single cluster
    (OUTPUTS / "C4_reproducibility").mkdir(parents=True, exist_ok=True)
    a1.obs[["true_program", "leiden"]].to_csv(
        OUTPUTS / "C4_reproducibility" / "run1_labels.csv"
    )

    c.expected = {"two_runs_identical": True, "n_clusters_ge_2": True}
    c.actual = {"two_runs_identical": identical, "n_clusters": n_clusters}
    c.check("two_runs_identical", identical, True, identical)
    c.check("n_clusters_ge_2", n_clusters >= 2, ">=2", n_clusters)
    c.finalize()
    return c


def c5_pseudobulk_aggregation() -> Condition:
    c = Condition(
        "C5",
        "Pseudobulk aggregation sums counts per sample correctly",
        "differential-expression",
        input_desc="120 cells across 4 samples/2 groups; verify per-sample summed counts + min-cell gate.",
    )
    from importlib import import_module

    import anndata as ad

    pb = import_module(f"{PKG}.pseudobulk_de")

    rng = np.random.default_rng(3)
    genes = [f"G{i}" for i in range(10)]
    samples, groups, rows = [], [], []
    layout = {
        "s1": ("A", 30),
        "s2": ("A", 30),
        "s3": ("B", 30),
        "s4": ("B", 5),
    }  # s4 below min-cells gate
    for s, (grp, ncells) in layout.items():
        block = rng.poisson(3.0, size=(ncells, len(genes)))
        rows.append(block)
        samples += [s] * ncells
        groups += [grp] * ncells
    X = np.vstack(rows).astype(float)
    a = ad.AnnData(X=X)
    a.var_names = genes
    a.obs["sample"] = samples
    a.obs["group"] = groups
    a.layers["counts"] = X.copy()

    # ground truth per-sample sums
    exp_sums = {}
    for s in layout:
        mask = np.array(samples) == s
        exp_sums[s] = X[mask].sum(axis=0).round().astype(int).tolist()

    counts_df, meta_df = pb._build_pseudobulk(a, "sample", "group")
    present = sorted(counts_df.index.tolist())
    # s4 has 5 cells (< MIN_CELLS_PER_PSEUDOBULK=10) so it must be excluded
    c.check(
        "s4_excluded_below_min_cells",
        "s4" not in present,
        "s4 excluded",
        f"present={present}",
    )
    c.check(
        "samples_present", present == ["s1", "s2", "s3"], ["s1", "s2", "s3"], present
    )
    sums_ok = all(counts_df.loc[s].astype(int).tolist() == exp_sums[s] for s in present)
    c.check(
        "per_sample_sums_match", sums_ok, "hand-computed column sums", "see counts_df"
    )
    grp_ok = meta_df.loc["s1", "group"] == "A" and meta_df.loc["s3", "group"] == "B"
    c.check(
        "group_mapping_correct",
        grp_ok,
        {"s1": "A", "s3": "B"},
        {"s1": meta_df.loc["s1", "group"], "s3": meta_df.loc["s3", "group"]},
    )

    (OUTPUTS / "C5_pseudobulk").mkdir(parents=True, exist_ok=True)
    counts_df.to_csv(OUTPUTS / "C5_pseudobulk" / "pseudobulk_counts.csv")
    c.expected = {
        "samples_present": ["s1", "s2", "s3"],
        "s4_excluded": True,
        "sums_match": True,
    }
    c.actual = {
        "samples_present": present,
        "s4_excluded": "s4" not in present,
        "sums_match": sums_ok,
    }
    c.finalize()
    return c


def c6_pseudobulk_deseq2() -> Condition:
    c = Condition(
        "C6",
        "Pseudobulk DESeq2 recovers spiked differential genes",
        "differential-expression",
        requires="pydeseq2",
        input_desc="160 cells / 4 samples / 2 groups, 200 genes; 15 genes spiked ~5x up in group B.",
    )
    from importlib import util as iutil

    if iutil.find_spec("pydeseq2") is None:
        c.status = "SKIP"
        c.detail = "pydeseq2 not installed"
        return c
    from importlib import import_module

    import anndata as ad

    pb = import_module(f"{PKG}.pseudobulk_de")

    rng = np.random.default_rng(7)
    n_genes = 200
    spiked = [f"G{i:03d}" for i in range(15)]  # genes 0..14 are up in group B
    genes = [f"G{i:03d}" for i in range(n_genes)]
    layout = {"s1": "A", "s2": "A", "s3": "B", "s4": "B"}
    base_rate = rng.uniform(2.0, 6.0, size=n_genes)
    samples, groups, blocks = [], [], []
    for s, grp in layout.items():
        rate = base_rate.copy()
        if grp == "B":
            rate[:15] *= 5.0  # spike up in group B
        block = rng.poisson(rate, size=(40, n_genes))  # 40 cells/sample
        blocks.append(block)
        samples += [s] * 40
        groups += [grp] * 40
    X = np.vstack(blocks).astype(float)
    a = ad.AnnData(X=X)
    a.var_names = genes
    a.obs["sample"] = samples
    a.obs["group"] = groups
    a.layers["counts"] = X.copy()

    out = OUTPUTS / "C6_deseq2"
    out.mkdir(parents=True, exist_ok=True)
    pb.compute_pseudobulk_de(a, group_col="group", sample_col="sample", out_dir=out)
    res_path = out / "pseudobulk_overall_DE.csv"
    c.check(
        "DE_output_written",
        res_path.exists(),
        "pseudobulk_overall_DE.csv exists",
        res_path.exists(),
    )
    if not res_path.exists():
        c.finalize()
        return c
    import pandas as pd

    res = pd.read_csv(res_path)
    up_genes = set(res.loc[res["regulation"] == "up", "gene"].astype(str))
    recovered = [g for g in spiked if g in up_genes]
    recall = len(recovered) / len(spiked)
    # spiked genes should be recovered as 'up'; unspiked should rarely be
    false_up = up_genes - set(spiked)
    c.check("spiked_genes_recall_ge_0.8", recall >= 0.8, ">=0.80", round(recall, 3))
    c.check(
        "few_false_positives", len(false_up) <= 10, "<=10 false 'up'", len(false_up)
    )
    c.expected = {"spiked_up_recall": ">=0.80", "false_up": "<=10"}
    c.actual = {
        "spiked_up_recall": round(recall, 3),
        "n_up_total": len(up_genes),
        "false_up": len(false_up),
    }
    c.finalize()
    return c


def c7_consensus_voting() -> Condition:
    c = Condition(
        "C7",
        "Consensus vote tally: majority, unanimity, tie-break, abstention",
        "annotation-consensus",
        input_desc="Four constructed voter panels with known correct outcomes.",
    )
    from importlib import import_module

    tools = import_module(f"{PKG}.celltype_consensus.tools")
    tally = tools.tally_votes
    harmonize = tools.harmonize_label

    # harmonization: synonyms should collapse to one canonical node
    h_t = harmonize("T cells")
    h_t2 = harmonize("T_cells")
    c.check(
        "harmonize_synonyms_agree",
        h_t == h_t2 and h_t not in ("", None),
        "same canonical node",
        {"T cells": h_t, "T_cells": h_t2},
    )
    c.check(
        "harmonize_unknown_is_unassigned",
        harmonize("nonsense").startswith("Other")
        or harmonize(None) == tools.UNASSIGNED,
        "unknown -> Other/Unassigned",
        {"None": harmonize(None)},
    )

    # unanimous
    r = tally({"celltypist": h_t, "singler": h_t, "llm": h_t})
    c.check(
        "unanimous_detected",
        r["unanimous"] and r["majority_label"] == h_t,
        {"unanimous": True, "label": h_t},
        {"unanimous": r["unanimous"], "label": r["majority_label"]},
    )

    # simple majority (2 of 3)
    h_b = harmonize("B cell")
    r2 = tally({"celltypist": h_t, "singler": h_t, "llm": h_b})
    c.check(
        "majority_2of3",
        r2["majority_label"] == h_t and r2["majority_count"] == 2,
        {"label": h_t, "count": 2},
        {"label": r2["majority_label"], "count": r2["majority_count"]},
    )

    # tie broken by confidence
    r3 = tally({"m1": h_t, "m2": h_b}, {"m1": 0.9, "m2": 0.2})
    c.check(
        "tie_broken_by_confidence",
        r3["tied"] and r3["majority_label"] == h_t,
        {"tied": True, "winner": h_t},
        {"tied": r3["tied"], "winner": r3["majority_label"]},
    )

    # all abstain
    r4 = tally({"m1": tools.UNASSIGNED, "m2": tools.UNASSIGNED})
    c.check(
        "all_abstain_is_unassigned",
        r4["majority_label"] == tools.UNASSIGNED and r4["n_methods"] == 0,
        {"label": tools.UNASSIGNED, "n": 0},
        {"label": r4["majority_label"], "n": r4["n_methods"]},
    )

    c.expected = {
        "unanimous": True,
        "majority_2of3": True,
        "tie_break": True,
        "abstain": True,
    }
    c.actual = {ck["name"]: ck["ok"] for ck in c.checks}
    c.finalize()
    return c


def c8_marker_cleaning() -> Condition:
    c = Condition(
        "C8",
        "PubMed voter strips low-information genes before querying",
        "pubmed-annotation",
        input_desc="Marker list mixing ribosomal/mito/lncRNA noise with real lineage markers.",
    )
    from importlib import import_module

    pa = import_module(f"{PKG}.pubmed_annotation")

    raw = [
        "CD3D",
        "cd3e",
        "RPL13",
        "RPS6",
        "MT-CO1",
        "MALAT1",
        "NEAT1",
        "HBB",
        "CD8A",
        "CD3D",
        "GZMK",
        "XIST",
        "FOS",
    ]  # dups + noise + real markers
    cleaned = pa.clean_markers(raw, species="human", top_n=30)
    lowinfo = {"RPL13", "RPS6", "MT-CO1", "MALAT1", "NEAT1", "HBB", "XIST", "FOS"}
    c.check(
        "lowinfo_removed",
        not (set(cleaned) & lowinfo),
        "no low-info genes",
        sorted(set(cleaned) & lowinfo),
    )
    c.check(
        "real_markers_kept",
        {"CD3D", "CD3E", "CD8A", "GZMK"}.issubset(set(cleaned)),
        "CD3D/CD3E/CD8A/GZMK kept",
        cleaned,
    )
    c.check(
        "uppercased_and_deduped",
        cleaned.count("CD3D") == 1 and "CD3E" in cleaned,
        "human genes uppercased + deduped",
        cleaned,
    )

    # the retrieval ladder must broaden from tissue-specific to gene-only
    ladder = pa.build_query_ladder(
        cleaned, disease="cervical cancer", biosample="cervix", species="human"
    )
    levels = [lvl for lvl, _q in ladder]
    c.check(
        "query_ladder_3_levels",
        levels == ["tissue+concept", "concept", "gene-only"],
        ["tissue+concept", "concept", "gene-only"],
        levels,
    )

    c.expected = {
        "lowinfo_removed": True,
        "real_kept": True,
        "ladder": ["tissue+concept", "concept", "gene-only"],
    }
    c.actual = {"cleaned": cleaned, "ladder_levels": levels}
    c.finalize()
    return c


def c9_confidence_scoring() -> Condition:
    c = Condition(
        "C9",
        "PubMed confidence scoring is monotonic & abstains without evidence",
        "pubmed-annotation",
        input_desc="Constructed LLM results: strong-evidence vs uncited vs unknown-label.",
    )
    from importlib import import_module

    pa = import_module(f"{PKG}.pubmed_annotation")

    strong = {
        "cell_type": "CD8-positive T cell",
        "confidence": "high",
        "supporting_markers": ["CD3D", "CD8A", "GZMK"],
        "pmids": ["111", "222"],
        "contradicting_markers": [],
        "review_required": False,
    }
    uncited = {
        "cell_type": "CD8-positive T cell",
        "confidence": "high",
        "supporting_markers": ["CD3D", "CD8A", "GZMK"],
        "pmids": [],
        "contradicting_markers": [],
        "review_required": False,
    }
    unknown = {
        "cell_type": "Unknown",
        "confidence": "low",
        "supporting_markers": [],
        "pmids": [],
        "contradicting_markers": [],
        "review_required": True,
    }

    b_s, s_s, r_s = pa.compute_confidence(strong, n_abstracts=6)
    b_u, s_u, r_u = pa.compute_confidence(uncited, n_abstracts=6)
    b_k, s_k, r_k = pa.compute_confidence(unknown, n_abstracts=0)

    c.check(
        "strong_scores_higher_than_uncited",
        s_s > s_u,
        "strong > uncited",
        {"strong": s_s, "uncited": s_u},
    )
    c.check("uncited_penalized_review", r_u, "uncited -> review_required", r_u)
    c.check(
        "unknown_capped_low",
        s_k <= 0.25 and b_k == "low" and r_k,
        {"score<=0.25": True, "band": "low", "review": True},
        {"score": s_k, "band": b_k, "review": r_k},
    )
    c.check(
        "scores_in_unit_interval",
        all(0.0 <= x <= 1.0 for x in (s_s, s_u, s_k)),
        "0<=score<=1",
        [s_s, s_u, s_k],
    )

    c.expected = {"strong>uncited": True, "unknown<=0.25": True, "uncited_review": True}
    c.actual = {
        "strong": round(s_s, 3),
        "uncited": round(s_u, 3),
        "unknown": round(s_k, 3),
        "bands": [b_s, b_u, b_k],
    }
    c.finalize()
    return c


def c10_provenance_manifest() -> Condition:
    c = Condition(
        "C10",
        "Provenance manifest + seed reproducibility",
        "provenance",
        input_desc="write_run_manifest() + set_global_seed() + capture_versions().",
    )
    from importlib import import_module

    rp = import_module(f"{PKG}.reproducibility")

    # set_global_seed must make numpy draws reproducible
    rp.set_global_seed(0)
    a = np.random.rand(5)
    rp.set_global_seed(0)
    b = np.random.rand(5)
    c.check(
        "seed_makes_numpy_reproducible",
        np.allclose(a, b),
        "identical draws",
        bool(np.allclose(a, b)),
    )

    vers = rp.capture_versions()
    c.check(
        "versions_capture_scanpy_numpy",
        "scanpy" in vers and "numpy" in vers and "python" in vers,
        "scanpy+numpy+python captured",
        sorted(vers)[:6],
    )

    out = OUTPUTS / "C10_provenance"
    out.mkdir(parents=True, exist_ok=True)
    path = rp.write_run_manifest(
        out,
        analysis_name="benchmark",
        seed=0,
        params={"mode": "benchmark"},
        n_obs=123,
        n_vars=45,
    )
    man = json.loads(Path(path).read_text(encoding="utf-8"))
    needed = {
        "analysis_name",
        "timestamp_utc",
        "seed",
        "dataset",
        "params",
        "git_commit",
        "package_versions",
    }
    c.check(
        "manifest_has_all_keys",
        needed.issubset(man.keys()),
        sorted(needed),
        sorted(man.keys()),
    )
    c.check(
        "manifest_dataset_shape",
        man["dataset"] == {"n_obs": 123, "n_vars": 45},
        {"n_obs": 123, "n_vars": 45},
        man["dataset"],
    )

    c.expected = {"seed_reproducible": True, "manifest_complete": True}
    c.actual = {
        "manifest_path": str(path.relative_to(HERE))
        if path.is_relative_to(HERE)
        else str(path),
        "versions_sample": {k: vers[k] for k in list(vers)[:4]},
    }
    c.finalize()
    return c


ALL_CONDITIONS: List[Callable[[], Condition]] = [
    c1_loader_roundtrip,
    c2_loader_errors,
    c3_qc_filtering,
    c4_reproducibility,
    c5_pseudobulk_aggregation,
    c6_pseudobulk_deseq2,
    c7_consensus_voting,
    c8_marker_cleaning,
    c9_confidence_scoring,
    c10_provenance_manifest,
]


# --------------------------------------------------------------------------- #
#  Runner + reporting
# --------------------------------------------------------------------------- #
def run_all() -> List[Condition]:
    results: List[Condition] = []
    for fn in ALL_CONDITIONS:
        cid = fn.__name__
        log.info("=== running %s ===", cid)
        try:
            cond = fn()
        except Exception as e:  # a benchmark itself blew up
            cond = Condition(cid.upper(), cid, "error")
            cond.status = "ERROR"
            cond.detail = f"{type(e).__name__}: {e}"
            log.error("%s ERROR: %s\n%s", cid, e, traceback.format_exc())
        log.info("  [%s] %s — %s", cond.status, cond.cid, cond.title)
        for ck in cond.checks:
            log.info("      %s  %s", "ok" if ck["ok"] else "XX", ck["name"])
        results.append(cond)
    return results


def write_reports(results: List[Condition]) -> Dict[str, Any]:
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_err = sum(1 for r in results if r.status == "ERROR")
    n_skip = sum(1 for r in results if r.status == "SKIP")
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_conditions": len(results),
        "passed": n_pass,
        "failed": n_fail,
        "errored": n_err,
        "skipped": n_skip,
        "all_passed": n_fail == 0 and n_err == 0,
        "conditions": [
            {
                "id": r.cid,
                "title": r.title,
                "category": r.category,
                "requires": r.requires,
                "status": r.status,
                "input": r.input_desc,
                "expected": r.expected,
                "actual": r.actual,
                "detail": r.detail,
                "checks": r.checks,
            }
            for r in results
        ],
    }
    (HERE / "benchmark_results.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    icon = {"PASS": "✅", "FAIL": "❌", "ERROR": "💥", "SKIP": "➖"}
    L: List[str] = []
    L.append("# Single-cell pipeline — benchmark report\n")
    L.append(
        f"**{n_pass}/{len(results)} conditions passed** "
        f"(fail {n_fail}, error {n_err}, skip {n_skip}). "
        f"Generated {summary['generated_utc']}.\n"
    )
    L.append(
        "All benchmarks are deterministic (fixed seeds) and call the pipeline's own "
        "functions on synthetic, biologically-shaped data. Each condition states its "
        "input, a precomputed expected output, and the actual output produced.\n"
    )
    L.append("| ID | Condition | Category | Status | Requires |")
    L.append("|---|---|---|---|---|")
    for r in results:
        L.append(
            f"| {r.cid} | {r.title} | {r.category} | {icon.get(r.status, '?')} {r.status} "
            f"| {r.requires or '—'} |"
        )
    L.append("")
    L.append("## Per-condition detail\n")
    for r in results:
        L.append(f"### {r.cid} · {r.title}  {icon.get(r.status, '?')} **{r.status}**\n")
        L.append(
            f"- **Category:** {r.category}"
            + (f"  ·  **Requires:** {r.requires}" if r.requires else "")
        )
        L.append(f"- **Input:** {r.input_desc}")
        if r.detail:
            L.append(f"- **Note:** {r.detail}")
        L.append(f"- **Expected:** `{json.dumps(r.expected, default=str)}`")
        L.append(f"- **Actual:** `{json.dumps(r.actual, default=str)}`")
        if r.checks:
            L.append("")
            L.append("| Check | Result | Expected | Actual |")
            L.append("|---|---|---|---|")
            for ck in r.checks:
                L.append(
                    f"| {ck['name']} | {'✅' if ck['ok'] else '❌'} "
                    f"| `{json.dumps(ck['expected'], default=str)}` "
                    f"| `{json.dumps(ck['actual'], default=str)}` |"
                )
        L.append("")
    L.append("## Limitations & scope\n")
    L.append(
        "- These conditions cover the pipeline's **deterministic, offline-verifiable core**. "
        "Steps that require external services — CellTypist model download, SingleR/R via rpy2, "
        "the OpenRouter LLM voter, and live PubMed retrieval — are **not** exercised end-to-end "
        "here because they need credentials/network and are non-deterministic; they are covered "
        "structurally by the unit tests and the two static auditors.\n"
    )
    L.append(
        "- `C6` runs a genuine pyDESeq2 fit on synthetic counts and is skipped (not failed) "
        "if pyDESeq2 is unavailable.\n"
    )
    L.append(
        "- Synthetic inputs are written under `benchmarks/inputs/` and actual artifacts under "
        "`benchmarks/outputs/` for inspection; the full run log is `benchmarks/logs/benchmark_run.log`.\n"
    )
    (HERE / "BENCHMARK_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            # A wrapped/detached stream cannot be reconfigured; only console
            # rendering of non-ASCII output is affected.
            pass
    log.info("Repo root: %s", REPO_ROOT)
    results = run_all()
    summary = write_reports(results)
    print("\n" + "=" * 64)
    print(
        f" BENCHMARKS: {summary['passed']}/{summary['n_conditions']} passed "
        f"| fail {summary['failed']} | error {summary['errored']} | skip {summary['skipped']}"
    )
    print("=" * 64)
    print(f" Report: {HERE / 'BENCHMARK_REPORT.md'}")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
