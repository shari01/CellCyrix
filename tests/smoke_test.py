#!/usr/bin/env python3
"""
smoke_test.py — fast, offline end-to-end smoke test for the single-cell 10x pipeline.

    python tests/smoke_test.py

WHAT THIS IS
------------
A smoke test, not a benchmark and not a scientific validation. It answers one
question — "does the whole thing still run, and does it still write the artifacts
every downstream consumer reads?" — in a few minutes, on synthetic data, with no
network, no credentials and no real cohort.

It builds a tiny synthetic 10x cohort in a temp workspace (2 groups x 2 samples,
four transcriptional programs with real T / B / Monocyte / NK marker panels, an
interferon response elevated in CASE, and a composition shift between the arms),
then runs five phases:

  1. IMPORTS      — import every module in the package, so a syntax error or a broken
                    import anywhere in the tree fails here, including in modules the
                    run below never reaches.
  2. CONFIG       — the nested-YAML plumbing in ``main.py`` (``qc:`` / ``clustering:``
                    / ``annotation:`` / ``downstream:`` / ``de:`` -> flat driver
                    kwargs) is checked against the real driver signatures. A key that
                    no longer exists on the driver is silently dropped at runtime, so
                    this drift is invisible without a check like this one.
  3. SINGLE RUN   — the single-sample driver, end to end, on one synthetic sample.
  4. MULTI RUN    — the cohort driver through the REAL entry point (a generated YAML
                    handed to ``main.py``), with bbknn integration, donor-level
                    pseudobulk DESeq2 and group DE.
  5. ARTIFACTS    — the processed .h5ad (obs / layers / obsm schema), the provenance
                    manifest, the data-validation report, QC figures, the cluster
                    marker table, the consensus annotation table, the pseudobulk DE
                    tables + contrast design, the Bisque export — plus a direction
                    positive control: the genes simulated as up in CASE must come out
                    with a positive log2_fold_change in ``CASE_vs_CONTROL``.

OFFLINE BY DESIGN
-----------------
The only annotation voter left on is CellTypist, pinned to a locally cached model
(``CELLTYPIST_MODEL``). The knowledge-based, PubMed and SingleR voters and the report
are OFF: they need OpenRouter credentials, the internet or R, none of which belong in
a smoke test. If the CellTypist cache is missing, that voter abstains and the
consensus degrades — the pipeline's documented behaviour, so this test reports it as
a warning and does not turn it into a failure.

Everything is seeded (``SEED``). Console output stays readable: each phase's pipeline
log goes to ``<workspace>/logs/<phase>.log`` and only the tail of it is printed when
something fails. The workspace is deleted on success and KEPT on failure.

Exit code is 0 only when every check passed, so this can gate a commit or CI with no
CLI of its own.
"""

from __future__ import annotations

import contextlib
import gzip
import importlib
import importlib.util
import inspect
import io
import json
import logging
import os
import pkgutil
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
#  Knobs — edit these. Deliberately no CLI (package convention: run the file).
# --------------------------------------------------------------------------- #
SEED = 0
CELLS_PER_SAMPLE = 180  # per synthetic sample, before QC
N_FILLER_GENES = 500  # background genes on top of the marker panels
RUN_SINGLE = True  # phase 3
RUN_MULTI = True  # phase 4 (the cohort run; also covers main.py)
ENABLE_CELLTYPIST = True  # uses the LOCAL cache below; no download at run time
CELLTYPIST_MODEL = "Immune_All_Low.pkl"
GENERATE_REPORT = False  # report needs OPENROUTER_API_KEY + network
KEEP_WORKSPACE = False  # True = keep the temp workspace even when passing
WORKSPACE_ROOT = None  # None = a fresh temp dir; or set a Path

# Cohort layout written into the workspace: <group>/<sample>/ 10x folders.
COHORT = {"CONTROL": ["ctrl_s1", "ctrl_s2"], "CASE": ["case_s1", "case_s2"]}
SINGLE_SAMPLE = ("CONTROL", "ctrl_s1")  # which cohort sample the single run uses
SINGLE_ANALYSIS_NAME = "smoke_single"  # sample_label -> analysis_name for phase 3
MULTI_ANALYSIS_NAME = "combined_all_samples"  # fixed by main_multi.run_pipeline_multi


# --------------------------------------------------------------------------- #
#  Locate + import the real pipeline package (works from the repo or a copy)
# --------------------------------------------------------------------------- #
def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "cellcyrix").is_dir():
            return p
    raise RuntimeError("repo root with 'cellcyrix' not found")


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PKG = "cellcyrix.single_cell_pipeline_agent.singlecell_10x"


# --------------------------------------------------------------------------- #
#  Tiny check harness (stdlib only; no pytest/unittest dependency)
# --------------------------------------------------------------------------- #
_CHECKS: list[tuple[str, bool, str]] = []
_WARNINGS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    """Record one pass/fail check and print it."""
    ok = bool(ok)
    _CHECKS.append((name, ok, detail))
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""),
        flush=True,
    )
    return ok


def warn(message: str) -> None:
    """Record a non-fatal observation (documented degradation, not a defect)."""
    _WARNINGS.append(message)
    print(f"  [WARN] {message}", flush=True)


def check_file(name: str, path, min_bytes: int = 1) -> bool:
    """Check that ``path`` exists as a non-trivial file."""
    p = Path(path)
    if not p.is_file():
        return check(name, False, f"missing: {p}")
    size = p.stat().st_size
    return check(name, size >= min_bytes, f"{p.name} ({size:,} B)")


def check_glob(name: str, directory, pattern: str, minimum: int = 1) -> bool:
    """Check that at least ``minimum`` files match ``pattern`` under ``directory``."""
    d = Path(directory)
    hits = sorted(d.glob(pattern)) if d.is_dir() else []
    return check(
        name,
        len(hits) >= minimum,
        f"{len(hits)} match {pattern}" if hits else f"none in {d}",
    )


def _reset_logging() -> None:
    """Detach the drivers' log handlers so a closed log file is never written to.

    The drivers call ``logging.basicConfig`` with a ``StreamHandler(sys.stdout)``
    bound at call time. Each phase redirects stdout to its own log file, so leaving
    those handlers attached across phases would write to a closed stream.
    """
    for h in logging.root.handlers[:]:
        try:
            h.close()
        except (OSError, ValueError):
            # Already-closed or detached handler; removal below is what matters.
            pass
        logging.root.removeHandler(h)


def _tail(path: Path, n: int = 30) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(log unavailable)"
    return "\n".join(f"    | {ln}" for ln in lines[-n:])


@contextlib.contextmanager
def _capture_output(log_path: Path):
    """Send everything a phase writes to ``log_path``, at the file-descriptor level.

    ``contextlib.redirect_stdout`` alone is not enough: scanpy's logger binds the
    original stream at import time and would keep writing to the console. Redirecting
    fds 1/2 as well catches that, plus anything a C extension prints.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    with open(log_path, "w", encoding="utf-8", errors="replace") as fh:
        saved = (None, None)
        try:
            saved = (os.dup(1), os.dup(2))
            os.dup2(fh.fileno(), 1)
            os.dup2(fh.fileno(), 2)
            with contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
                try:
                    yield fh
                except BaseException:
                    traceback.print_exc(file=fh)
                    raise
        finally:
            fh.flush()
            for fd, target in zip(saved, (1, 2), strict=True):
                if fd is not None:
                    os.dup2(fd, target)
                    os.close(fd)


@contextlib.contextmanager
def phase(title: str, log_path: Path | None = None):
    """Run a phase, timing it and (optionally) capturing pipeline chatter to a log."""
    print(f"\n=== {title} ===", flush=True)
    if log_path is not None:
        print(f"  (pipeline output -> {log_path})", flush=True)
    t0 = time.perf_counter()
    failed = False
    try:
        if log_path is None:
            yield None
        else:
            with _capture_output(log_path):
                yield None
    except BaseException:
        failed = True
        raise
    finally:
        _reset_logging()
        print(f"--- {title}: {time.perf_counter() - t0:.1f}s", flush=True)
        if failed and log_path is not None:
            print(f"  last lines of {log_path.name}:\n{_tail(log_path)}", flush=True)


# --------------------------------------------------------------------------- #
#  Synthetic fixture — a tiny cohort with known biology
# --------------------------------------------------------------------------- #
# Real marker symbols, so the CellTypist voter has something to match on (the model
# only sees the genes it recognises; the filler genes below are ignored by it).
_PROGRAMS = {
    "T cell": ["CD3D", "CD3E", "CD3G", "TRAC", "IL7R", "CD2", "LCK", "CD27"],
    "B cell": ["MS4A1", "CD79A", "CD79B", "CD19", "TCL1A", "BANK1", "IGHM"],
    "Monocyte": ["LYZ", "CD14", "S100A8", "S100A9", "FCN1", "VCAN", "CST3", "TYROBP"],
    "NK cell": ["GNLY", "NKG7", "KLRD1", "PRF1", "GZMB", "KLRF1", "FGFBP2"],
}
# Elevated in CASE — the known-truth group effect the DE direction check reads.
_RESPONSE = ["ISG15", "IFI6", "IFI44L", "MX1", "OAS1", "STAT1", "IRF7", "LY6E"]
# Present but low, so pct_counts_mt stays well under the 15% QC threshold.
_MITO = ["MT-CO1", "MT-CO2", "MT-CO3", "MT-ND1", "MT-ND2", "MT-ATP6"]
# Composition differs between arms, so the proportion comparison has something to see.
_COMPOSITION = {
    "CONTROL": {"T cell": 0.45, "B cell": 0.20, "Monocyte": 0.25, "NK cell": 0.10},
    "CASE": {"T cell": 0.30, "B cell": 0.15, "Monocyte": 0.40, "NK cell": 0.15},
}
_MARKER_LAMBDA = 18.0  # own-program marker counts
_RESPONSE_LAMBDA = 3.0  # baseline for the response panel
_CASE_RESPONSE_FOLD = 4.0  # x this in CASE
_MITO_LAMBDA = 0.4


def _gene_table() -> tuple[list[str], list[str]]:
    """(feature_ids, gene_symbols) for the synthetic reference."""
    symbols: list[str] = []
    for markers in _PROGRAMS.values():
        symbols.extend(markers)
    symbols.extend(_RESPONSE)
    symbols.extend(_MITO)
    symbols.extend(f"SMOKEG{i:04d}" for i in range(N_FILLER_GENES))
    # Ensembl-style ids alongside symbols, so the loader's id/symbol columns and the
    # gene-name heuristics both see what they see on real 10x output.
    ids = [f"ENSG{i:011d}" for i in range(len(symbols))]
    return ids, symbols


def _simulate_counts(rng, n_cells: int, group: str, symbols: list[str]):
    """Raw integer counts (cells x genes) plus the true program label per cell."""
    idx = {g: i for i, g in enumerate(symbols)}
    n_genes = len(symbols)

    # Gene-specific baseline rates give HVG selection a real mean-variance spread,
    # and keep ~2/3 of the filler genes detected so cells clear qc.min_genes.
    base = rng.lognormal(mean=0.0, sigma=0.45, size=n_genes)
    X = rng.poisson(np.broadcast_to(base, (n_cells, n_genes))).astype(np.int32)

    comp = _COMPOSITION[group]
    programs = list(comp)
    labels = rng.choice(programs, size=n_cells, p=[comp[p] for p in programs])
    for prog in programs:
        rows = np.flatnonzero(labels == prog)
        if rows.size == 0:
            continue
        cols = [idx[g] for g in _PROGRAMS[prog]]
        X[np.ix_(rows, cols)] += rng.poisson(
            _MARKER_LAMBDA, size=(rows.size, len(cols))
        ).astype(np.int32)

    lam = _RESPONSE_LAMBDA * (_CASE_RESPONSE_FOLD if group == "CASE" else 1.0)
    rcols = [idx[g] for g in _RESPONSE]
    X[:, rcols] += rng.poisson(lam, size=(n_cells, len(rcols))).astype(np.int32)

    mcols = [idx[g] for g in _MITO]
    X[:, mcols] += rng.poisson(_MITO_LAMBDA, size=(n_cells, len(mcols))).astype(
        np.int32
    )
    return X, labels


def _write_10x_dir(
    out_dir: Path, counts, ids: list[str], symbols: list[str], barcodes: list[str]
) -> None:
    """Write a gzipped Cell Ranger feature-barcode trio (genes x cells matrix)."""
    from scipy import sparse as sp
    from scipy.io import mmwrite

    out_dir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    mmwrite(buf, sp.coo_matrix(counts.T), field="integer")
    with gzip.open(out_dir / "matrix.mtx.gz", "wb") as f:
        f.write(buf.getvalue())
    with gzip.open(out_dir / "barcodes.tsv.gz", "wt", encoding="utf-8") as f:
        f.write("\n".join(barcodes) + "\n")
    with gzip.open(out_dir / "features.tsv.gz", "wt", encoding="utf-8") as f:
        f.write(
            "\n".join(
                f"{i}\t{s}\tGene Expression" for i, s in zip(ids, symbols, strict=True)
            )
            + "\n"
        )


def build_cohort(workspace: Path) -> Path:
    """Write the whole synthetic cohort tree; return its base directory."""
    base = workspace / "cohort"
    ids, symbols = _gene_table()
    rng = np.random.default_rng(SEED)
    rows = []
    for group, samples in COHORT.items():
        for sample in samples:
            counts, labels = _simulate_counts(rng, CELLS_PER_SAMPLE, group, symbols)
            barcodes = [f"{i:05d}-1" for i in range(counts.shape[0])]
            _write_10x_dir(base / group / sample, counts, ids, symbols, barcodes)
            rows.append(f"{sample},{group}")
            print(
                f"  wrote {group}/{sample}: {counts.shape[0]} cells x "
                f"{counts.shape[1]} genes, "
                f"median {int(np.median((counts > 0).sum(axis=1)))} genes/cell",
                flush=True,
            )
    # sample,group handshake file — exercises the cohort-tool group-map path too.
    (base / "metadata.csv").write_text(
        "sample,group\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    return base


# --------------------------------------------------------------------------- #
#  Phase 1 — every module imports
# --------------------------------------------------------------------------- #
def phase_imports() -> None:
    failures: list[str] = []

    def _onerror(name: str) -> None:
        failures.append(f"{name}: {traceback.format_exc(limit=1).strip()}")

    pkg = importlib.import_module(PKG)
    names = [
        m.name
        for m in pkgutil.walk_packages(pkg.__path__, prefix=f"{PKG}.", onerror=_onerror)
    ]
    imported = 0
    for name in names:
        if name.rsplit(".", 1)[-1] == "__main__":
            continue  # importing a __main__ submodule would run it
        try:
            importlib.import_module(name)
            imported += 1
        except Exception as e:  # noqa: BLE001 - the condition being tested
            failures.append(f"{name}: {type(e).__name__}: {e}")

    check(
        f"all {imported} package modules import",
        not failures,
        "; ".join(failures[:3]) if failures else f"{imported} modules",
    )

    from cellcyrix.single_cell_pipeline_agent.singlecell_10x import (  # noqa: F401
        run_pipeline,
        run_pipeline_multi,
    )

    check(
        "package exports run_pipeline + run_pipeline_multi",
        callable(run_pipeline) and callable(run_pipeline_multi),
    )


# --------------------------------------------------------------------------- #
#  Phase 2 — main.py config plumbing agrees with the driver signatures
# --------------------------------------------------------------------------- #
def _load_entry_module():
    """Import the repo's ``main.py`` as a module (it is not importable by name)."""
    path = REPO_ROOT / "main.py"
    spec = importlib.util.spec_from_file_location("_smoke_entry_main", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def phase_config_plumbing():
    from cellcyrix.single_cell_pipeline_agent.singlecell_10x import (
        run_pipeline,
        run_pipeline_multi,
    )
    from cellcyrix.single_cell_pipeline_agent.singlecell_10x.pipeline_options import (
        PipelineOptions,
    )

    entry = _load_entry_module()
    check(
        "main.py exposes the config helpers",
        all(
            hasattr(entry, a)
            for a in (
                "load_config",
                "flatten_sections",
                "_build_kwargs",
                "_SECTION_KEY_MAP",
            )
        ),
    )

    # The ~50 shared options now live on PipelineOptions and reach a driver through its
    # `**overrides`, so they are not in `inspect.signature`. Accept the declared field
    # names too, or every nested config key reads as orphaned.
    accepted = (
        set(inspect.signature(run_pipeline).parameters)
        | set(inspect.signature(run_pipeline_multi).parameters)
        | set(PipelineOptions.field_names())
    )
    mapped = {
        dst or src
        for keymap in entry._SECTION_KEY_MAP.values()
        for src, dst in keymap.items()
    }
    orphans = sorted(mapped - accepted)
    check(
        "every nested config key maps onto a real driver argument",
        not orphans,
        f"orphaned: {orphans}" if orphans else f"{len(mapped)} keys",
    )

    # A nested section must actually reach the drivers (it is dropped silently if the
    # mapping regresses), and an unknown key must be reported, not applied.
    flat = entry.flatten_sections(
        {
            "qc": {"min_genes": 111, "max_mito_percent": 12.5},
            "common": {
                "clustering": {"leiden_resolution": 0.7},
                "de": {"lfc_threshold": 2.0},
            },
        }
    )
    check(
        "nested sections flatten to driver kwargs",
        flat.get("qc_min_genes") == 111
        and flat.get("qc_max_mito_percent") == 12.5
        and flat.get("leiden_resolution") == 0.7
        and flat.get("de_lfc_threshold") == 2.0,
        str(sorted(flat)),
    )

    kwargs = entry._build_kwargs(
        run_pipeline, {"out_name": "x", "not_a_real_option": 1}
    )
    check(
        "_build_kwargs drops unknown options",
        kwargs == {"out_name": "x"},
        str(sorted(kwargs)),
    )
    return entry


# --------------------------------------------------------------------------- #
#  Phase 3 / 4 — the actual runs
# --------------------------------------------------------------------------- #
def _offline_voter_kwargs() -> dict:
    """Annotation settings that keep the run offline and credential-free."""
    return dict(
        enable_celltypist=ENABLE_CELLTYPIST,
        enable_knowledge_based=False,  # needs OPENROUTER_API_KEY
        enable_singler=False,  # needs R + SingleR + celldex
        enable_pubmed=False,  # needs the internet
        tissue="blood",  # pinned, so nothing is auto-inferred by LLM
        species="human",
        celltypist_model=CELLTYPIST_MODEL,
    )


def run_single(cohort_base: Path, out_dir: Path) -> Path:
    from cellcyrix.single_cell_pipeline_agent.singlecell_10x import run_pipeline

    group, sample = SINGLE_SAMPLE
    return run_pipeline(
        single_10x_dir=cohort_base / group / sample,
        sample_label=SINGLE_ANALYSIS_NAME,
        group_label=group,
        out_name=str(out_dir),
        do_pathway_clustering=False,  # Enrichr needs the network
        do_groupwise_de=False,  # one group
        do_dpt=False,  # root selection would want the LLM
        generate_report=GENERATE_REPORT,
        prepare_for_bisque=True,
        seed=SEED,
        do_doublet_detection=True,
        remove_doublets=False,  # detect (exercise Scrublet) without perturbing counts
        batch_key=None,
        integration_method=None,
        cleanup_pipeline_log=False,
        **_offline_voter_kwargs(),
    )


def _write_multi_config(path: Path, cohort_base: Path, out_dir: Path) -> None:
    """Write the YAML the real entry point consumes for the cohort run."""
    import yaml

    voters = _offline_voter_kwargs()
    config = {
        "mode": "multi",
        "common": {
            "do_pathway_clustering": False,
            "do_dpt": False,
            "generate_report": GENERATE_REPORT,
            "prepare_for_bisque": True,
            "geo_json_path": None,
            "logos_dir": None,
            "seed": SEED,
            "skip_tsne": True,
            "skip_pca_cluster_plots": True,
            "skip_per_celltype_plots": True,
            "skip_per_celltype_csvs": True,
            "skip_per_cluster_marker_csvs": True,
            "cleanup_pipeline_log": False,
            **voters,
        },
        # The five nested sections, so the config -> driver plumbing is exercised
        # by the run itself and not only by phase 2.
        "clustering": {"leiden_resolution": 0.5, "min_cluster_cells": 20},
        "qc": {
            "min_genes": 200,
            "max_genes": 6000,
            "max_mito_percent": 15.0,
            "do_doublet_detection": True,
            "remove_doublets": False,
        },
        "annotation": {"reuse_existing_final_annotation": False},
        "downstream": {"do_pseudobulk_de": True, "exclude_low_confidence_de": False},
        "de": {"reference_group": "CONTROL", "lfc_threshold": 1.0, "alpha": 0.05},
        "multi": {
            "multi_base_dir": str(cohort_base),
            "out_name": str(out_dir),
            "group_map_path": None,
            "do_groupwise_de": True,
            "batch_key": "sample",
            "integration_method": "bbknn",
            "run_per_sample": False,
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def run_multi_via_entry_point(
    entry, cohort_base: Path, out_dir: Path, config_path: Path
) -> Path:
    """Drive the cohort run through ``main.py`` exactly as a user would."""
    _write_multi_config(config_path, cohort_base, out_dir)
    # The repo entry point takes `config_path` (and an `output_root` it resolves a
    # relative out_name against); the standalone one has no CLI and reads the
    # module-level CONFIG_FILE. Support both without editing either. Passed by KEYWORD:
    # a positional list used to land in `config_path` and fail on `.exists()`.
    parameters = inspect.signature(entry.main).parameters
    if "config_path" in parameters:
        entry.main(config_path=config_path)
    elif parameters:
        entry.main([str(config_path)])
    else:
        entry.CONFIG_FILE = config_path
        entry.main()
    return out_dir


# --------------------------------------------------------------------------- #
#  Phase 5 — artifacts
# --------------------------------------------------------------------------- #
def check_common_artifacts(out_dir: Path, analysis_name: str) -> None:
    check_file(
        "processed h5ad",
        out_dir / f"{analysis_name}_processed_scanpy_output.h5ad",
        10_000,
    )
    check_file("provenance manifest", out_dir / "provenance" / "manifest.json", 100)
    check_file(
        "data-validation report",
        out_dir / "00_data_validation" / f"{analysis_name}_data_validation.csv",
    )
    check_file(
        "analysis summary",
        out_dir / "00_analysis_summary" / f"{analysis_name}_analysis_summary.txt",
    )
    check_glob("QC figures", out_dir / "01_qc_and_filtering", "*.png", minimum=3)
    check_glob("HVG figure", out_dir / "02_highly_variable_genes", "*.png")
    check_glob(
        "embedding figures",
        out_dir / "03_dimensionality_reduction_and_embeddings",
        "*.png",
    )
    check_file(
        "cluster marker table",
        out_dir
        / "04_clustering_and_cell_states"
        / "intercluster_analysis_deg"  # lowercased by the Rule 5.1 rename
        / "intercluster_cluster_markers.csv",
    )
    check_file(
        "consensus annotation table",
        out_dir
        / "05_celltype_analysis"
        / "celltype_annotation"
        / f"{analysis_name}_consensus_annotation.csv",
    )
    check_file(
        "Bisque export",
        out_dir / f"bisque_ready_{analysis_name}_processed_scanpy_output.h5ad",
        10_000,
    )
    check_file("pipeline log", out_dir / "pipeline.log", 100)


def check_manifest(out_dir: Path, analysis_name: str, *, expect_seed: int) -> None:
    path = out_dir / "provenance" / "manifest.json"
    if not path.is_file():
        return
    man = json.loads(path.read_text(encoding="utf-8"))
    check(
        "manifest records the run",
        man.get("analysis_name") == analysis_name and man.get("seed") == expect_seed,
        f"name={man.get('analysis_name')!r} seed={man.get('seed')}",
    )
    check(
        "manifest records package versions",
        all(
            k in (man.get("package_versions") or {})
            for k in ("python", "scanpy", "numpy", "pandas")
        ),
    )
    check(
        "manifest records the dataset size",
        int((man.get("dataset") or {}).get("n_obs", 0)) > 0
        and int((man.get("dataset") or {}).get("n_vars", 0)) > 0,
        str(man.get("dataset")),
    )


def check_h5ad_schema(out_dir: Path, analysis_name: str, *, multi: bool) -> None:
    import anndata as ad

    path = out_dir / f"{analysis_name}_processed_scanpy_output.h5ad"
    if not path.is_file():
        return
    a = ad.read_h5ad(path)
    check(
        "h5ad has cells and genes",
        a.n_obs > 0 and a.n_vars > 0,
        f"{a.n_obs} cells x {a.n_vars} genes",
    )
    required = ["sample", "group", "leiden", "include_in_downstream_analysis"]
    missing = [c for c in required if c not in a.obs.columns]
    check(
        "h5ad obs carries the run's design + gating columns",
        not missing,
        f"missing {missing}" if missing else ", ".join(required),
    )
    check("h5ad keeps raw counts in layers['counts']", "counts" in a.layers)
    check("h5ad carries the UMAP embedding", "X_umap" in a.obsm)
    if "celltype" in a.obs.columns:
        labels = sorted(set(a.obs["celltype"].astype(str)))
        check(
            "h5ad carries consensus cell-type labels",
            len(labels) >= 1,
            f"{len(labels)} label(s): {labels[:4]}",
        )
        if labels in (["Unassigned"], ["unassigned"]):
            warn(
                "every cluster is Unassigned — the CellTypist voter abstained "
                "(cached model missing?); the consensus degraded as documented"
            )
    else:
        warn(
            "no 'celltype' column: consensus annotation did not produce labels "
            "(see the phase log); downstream per-cell-type steps were skipped"
        )
    if multi:
        check(
            "multi run kept all 4 samples in 2 groups",
            a.obs["sample"].nunique() == sum(len(v) for v in COHORT.values())
            and a.obs["group"].nunique() == len(COHORT),
            f"{a.obs['sample'].nunique()} samples / {a.obs['group'].nunique()} groups",
        )


def check_group_de(out_dir: Path) -> None:
    import pandas as pd

    pb_dir = out_dir / "06_groupwise_deg" / "pseudobulk_deg"
    overall = pb_dir / "pseudobulk_overall_de.csv"
    if not overall.is_file():  # pre-rename runs
        overall = pb_dir / "pseudobulk_overall_DE.csv"
    if not check_file("pseudobulk overall DE table", overall):
        return
    check_file(
        "pseudobulk contrast design table", pb_dir / "pseudobulk_contrast_design.csv"
    )

    de = pd.read_csv(overall)
    needed = {
        "comparison",
        "log2_fold_change",
        "p_value_adj",
        "unit_of_replication",
        "test",
    }
    check(
        "pseudobulk DE table has the audit columns",
        needed.issubset(de.columns),
        f"missing {sorted(needed - set(de.columns))}"
        if not needed.issubset(de.columns)
        else f"{len(de):,} rows",
    )
    check(
        "pseudobulk unit of replication is the donor",
        "sample"
        in str(
            de.get("unit_of_replication", pd.Series(dtype=str)).iloc[0]
            if len(de)
            else ""
        ),
        str(de["unit_of_replication"].iloc[0]) if len(de) else "empty table",
    )

    # Direction positive control: reference_group=CONTROL is pinned in the config, so
    # a positive log2FoldChange must mean "higher in CASE" — and the response panel was
    # simulated 4x up in CASE. This catches a flipped contrast, which no file-existence
    # check can see.
    contrast = "CASE_vs_CONTROL"
    check(
        "contrast honours the pinned reference group",
        contrast in set(de.get("comparison", [])),
        f"comparisons: {sorted(set(de.get('comparison', [])))}",
    )
    sub = (
        de[de["comparison"] == contrast] if "comparison" in de.columns else de.iloc[0:0]
    )
    gene_col = next(
        (c for c in ("gene", "gene_name", "names", "index") if c in sub.columns), None
    )
    lfc_col = next(
        (c for c in ("log2_fold_change", "log2FoldChange") if c in sub.columns), None
    )
    if gene_col is None or lfc_col is None or sub.empty:
        warn(
            "cannot locate the gene column in the pseudobulk table; "
            "skipped the DE direction control"
        )
        return
    resp = sub[sub[gene_col].astype(str).isin(_RESPONSE)]
    median_lfc = float(resp[lfc_col].median()) if len(resp) else float("nan")
    check(
        "DE direction control: CASE-elevated genes have positive log2FC",
        len(resp) >= 4 and median_lfc > 0,
        f"{len(resp)}/{len(_RESPONSE)} genes, median log2FC={median_lfc:.2f}",
    )

    # Composition + exploratory cell-level DE outputs (only when labels exist).
    ct_dir = out_dir / "06_groupwise_deg" / "celltype_specific_deg"
    if ct_dir.is_dir():
        check_glob("exploratory cell-level DE per cell type", ct_dir, "**/*.csv")
    else:
        warn("no celltype_specific_deg/ — cell-type labels were unavailable")


# --------------------------------------------------------------------------- #
#  Driver
# --------------------------------------------------------------------------- #
def main() -> None:
    # Same reason the drivers do this: a Windows cp1252 console would mangle or raise
    # on the status output.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            # Console encoding only; the analysis is unaffected either way.
            pass

    t0 = time.perf_counter()
    workspace = (
        Path(WORKSPACE_ROOT)
        if WORKSPACE_ROOT
        else Path(tempfile.mkdtemp(prefix="sc_smoke_"))
    )
    workspace.mkdir(parents=True, exist_ok=True)
    logs = workspace / "logs"

    print("single-cell pipeline — SMOKE TEST")
    print(f"repo:      {REPO_ROOT}")
    print(f"python:    {sys.version.split()[0]}")
    print(f"workspace: {workspace}")

    try:
        with phase("Phase 1/5 — imports"):
            phase_imports()

        with phase("Phase 2/5 — main.py config plumbing"):
            entry = phase_config_plumbing()

        with phase("Phase 3/5 — synthetic cohort fixture"):
            cohort_base = build_cohort(workspace)
            check_file(
                "cohort matrix written",
                cohort_base / SINGLE_SAMPLE[0] / SINGLE_SAMPLE[1] / "matrix.mtx.gz",
            )

        if RUN_SINGLE:
            single_out = workspace / "out_single"
            with phase("Phase 4/5 — SINGLE-sample run", logs / "single_run.log"):
                run_single(cohort_base, single_out)
            with phase("Phase 4/5 — SINGLE-sample artifacts"):
                check_common_artifacts(single_out, SINGLE_ANALYSIS_NAME)
                check_manifest(single_out, SINGLE_ANALYSIS_NAME, expect_seed=SEED)
                check_h5ad_schema(single_out, SINGLE_ANALYSIS_NAME, multi=False)
        else:
            warn("single-sample run skipped (RUN_SINGLE=False)")

        if RUN_MULTI:
            multi_out = workspace / "out_multi"
            with phase(
                "Phase 5/5 — MULTI-sample cohort run via main.py",
                logs / "multi_run.log",
            ):
                run_multi_via_entry_point(
                    entry, cohort_base, multi_out, workspace / "smoke_config.yaml"
                )
            with phase("Phase 5/5 — MULTI-sample artifacts"):
                check_common_artifacts(multi_out, MULTI_ANALYSIS_NAME)
                check_manifest(multi_out, MULTI_ANALYSIS_NAME, expect_seed=SEED)
                check_h5ad_schema(multi_out, MULTI_ANALYSIS_NAME, multi=True)
                check_group_de(multi_out)
        else:
            warn("cohort run skipped (RUN_MULTI=False)")

    except BaseException as e:  # noqa: BLE001 - report, then fail the suite
        check(
            f"smoke run completed without raising ({type(e).__name__})", False, str(e)
        )

    failed = [name for name, ok, _ in _CHECKS if not ok]
    print("\n" + "=" * 72)
    print(
        f"SMOKE TEST: {len(_CHECKS) - len(failed)}/{len(_CHECKS)} checks passed, "
        f"{len(_WARNINGS)} warning(s), {time.perf_counter() - t0:.1f}s"
    )
    for w in _WARNINGS:
        print(f"  warn: {w}")
    for name in failed:
        print(f"  FAILED: {name}")

    ok = not failed and bool(_CHECKS)
    if ok and not KEEP_WORKSPACE:
        shutil.rmtree(workspace, ignore_errors=True)
        print(f"workspace removed: {workspace}")
    else:
        print(f"workspace KEPT for inspection: {workspace}")
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
