"""
Close the gap between the shipped alias table and the label sets your installed
references actually emit.

The alias table in spec/aliases.py was transcribed from published label sets, not
parsed from your model artefacts. Run this before trusting it in production.

    # from the installed CellTypist models on disk
    python -m cell_hierarchy.audit_vocabulary --celltypist-models \
        Immune_All_Low.pkl Human_Lung_Atlas.pkl Cells_Intestinal_Tract.pkl

    # from any newline- or CSV-delimited label list
    python -m cell_hierarchy.audit_vocabulary --labels-file singler_labels.txt \
        --source singler_blueprint

    # from an .h5ad obs column
    python -m cell_hierarchy.audit_vocabulary --h5ad adata.h5ad --obs-column cell_type

Outputs
-------
    vocabulary_audit.csv    every input label with its resolution and flags
    alias_stubs.py          paste-ready ALIASES entries for the unresolved ones

No LLM calls. Resolution is the same deterministic path the pipeline uses, so a
label that resolves here resolves identically at run time.
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .resolver import (
    MATCH_COMPARTMENT,
    MATCH_FUZZY,
    MATCH_UNRESOLVED,
    CellHierarchy,
    LabelConservationError,
)

logger = logging.getLogger(__name__)

AUDIT_COLUMNS = [
    "source_vocabulary",
    "raw_label",
    "node_id",
    "canonical_label",
    "level_name",
    "lineage",
    "main_cell_type",
    "match_method",
    "confidence",
    "resolved",
    "needs_review",
    "states",
    "note",
]


def labels_from_celltypist(model_paths: Sequence[str]) -> Dict[str, List[str]]:
    """Read cell_type labels out of CellTypist .pkl models."""
    try:
        from celltypist import models  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "celltypist is not importable in this environment; "
            "use --labels-file with an exported label list instead"
        ) from exc

    out: Dict[str, List[str]] = {}
    for path in model_paths:
        model = models.Model.load(model=path)
        source = f"celltypist:{Path(path).stem}"
        out[source] = sorted(str(c) for c in model.cell_types)
    return out


def labels_from_file(path: str, source: str) -> Dict[str, List[str]]:
    """Read labels from a newline list or a single-column/`label`-column CSV."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".csv", ".tsv"}:
        delim = "\t" if p.suffix.lower() == ".tsv" else ","
        rows = list(csv.reader(text.splitlines(), delimiter=delim))
        if not rows:
            return {source: []}
        header = [c.strip().lower() for c in rows[0]]
        col = 0
        for candidate in ("label", "cell_type", "celltype", "raw_label"):
            if candidate in header:
                col = header.index(candidate)
                break
        else:
            rows = [rows[0]] + rows[1:]  # no header match; treat all as data
        body = rows[1:] if col or "label" in header else rows
        labels = [r[col].strip() for r in body if r and r[col].strip()]
    else:
        labels = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return {source: sorted(set(labels))}


def labels_from_h5ad(path: str, obs_column: str, source: str) -> Dict[str, List[str]]:
    """Read the distinct labels in one ``.obs`` column of an ``.h5ad``.

    Args:
        path: The ``.h5ad`` to open.
        obs_column: Column holding the labels to audit.
        source: Vocabulary name to file the labels under.

    Returns:
        ``{source: [label, ...]}``, so results merge with the other loaders.
    """
    try:
        import anndata  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "anndata is not importable; export the obs column to CSV instead"
        ) from exc
    adata = anndata.read_h5ad(path, backed="r")
    if obs_column not in adata.obs.columns:
        raise SystemExit(
            f"obs column {obs_column!r} not found; available: {list(adata.obs.columns)}"
        )
    labels = sorted({str(v) for v in adata.obs[obs_column].unique()})
    return {source: labels}


def audit(
    h: CellHierarchy,
    label_sets: Dict[str, List[str]],
) -> Tuple[List[Dict[str, object]], Dict[str, List[str]]]:
    """Resolve every label against the hierarchy and report what did not map.

    Args:
        h: The hierarchy to resolve against.
        label_sets: ``{source: [label, ...]}`` as produced by the loaders above.

    Returns:
        ``(rows, unresolved)`` — one row per label with its resolution, and the
        labels that failed to resolve, grouped by source so they can be turned
        into alias stubs.
    """
    rows: List[Dict[str, object]] = []
    unresolved: Dict[str, List[str]] = defaultdict(list)

    for source, labels in label_sets.items():
        # the alias table is keyed by curated namespaces; a model-specific source
        # like 'celltypist:Human_Lung_Atlas' will fall through to the shared
        # indexes, which is the intended behaviour
        resolutions = h.resolve_many(labels, source=source)
        if len(resolutions) != len(labels):
            raise LabelConservationError(
                f"audit dropped labels for source {source!r}: "
                f"in={len(labels)} out={len(resolutions)}"
            )
        for res in resolutions:
            rows.append(
                {
                    "source_vocabulary": source,
                    "raw_label": res.raw_label,
                    "node_id": res.node_id,
                    "canonical_label": res.canonical_label,
                    "level_name": res.level_name,
                    "lineage": res.lineage,
                    "main_cell_type": res.main_cell_type,
                    "match_method": res.match_method,
                    "confidence": res.confidence,
                    "resolved": res.resolved,
                    "needs_review": res.needs_review,
                    "states": ";".join(res.states),
                    "note": res.note,
                }
            )
            if res.match_method in {MATCH_UNRESOLVED, MATCH_FUZZY, MATCH_COMPARTMENT}:
                unresolved[source].append(res.raw_label)
    return rows, dict(unresolved)


def emit_alias_stubs(unresolved: Dict[str, List[str]]) -> str:
    """Render unresolved labels as a paste-ready ``spec/aliases.py`` fragment."""
    lines = [
        '"""Auto-generated alias stubs. Fill in the node_id for each label, then',
        "merge into cell_hierarchy/spec/aliases.py. Labels are grouped by source.",
        "A label that belongs to no node in the tree needs a new node in the spec,",
        'not an alias — do not force-map it."""',
        "",
        "ALIAS_STUBS = {",
    ]
    for source in sorted(unresolved):
        lines.append(f"    # ---- {source}")
        for label in sorted(set(unresolved[source])):
            lines.append(f'    # "<node_id>": {{"{source}": ["{label}"]}},')
    lines.append("}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 = every label resolved)."""
    # Configured HERE rather than at module scope so importing this module does
    # not reconfigure the host application's logging (Rule: nothing runs at
    # import time). The bare "%(message)s" format keeps the summary below
    # readable as the aligned table it was written to be.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Audit voter vocabularies against the hierarchy."
    )
    parser.add_argument(
        "--celltypist-models",
        nargs="*",
        default=[],
        help="paths to CellTypist .pkl models",
    )
    parser.add_argument("--labels-file", help="newline or CSV list of labels")
    parser.add_argument("--h5ad", help="AnnData file to read labels from")
    parser.add_argument(
        "--obs-column", default="cell_type", help="obs column for --h5ad"
    )
    parser.add_argument(
        "--source",
        default="unspecified",
        help="vocabulary name for --labels-file/--h5ad",
    )
    parser.add_argument(
        "--outdir", default=".", help="where to write the audit outputs"
    )
    parser.add_argument("--fuzzy-threshold", type=float, default=0.88)
    args = parser.parse_args(argv)

    label_sets: Dict[str, List[str]] = {}
    if args.celltypist_models:
        label_sets.update(labels_from_celltypist(args.celltypist_models))
    if args.labels_file:
        label_sets.update(labels_from_file(args.labels_file, args.source))
    if args.h5ad:
        label_sets.update(labels_from_h5ad(args.h5ad, args.obs_column, args.source))
    if not label_sets:
        parser.error(
            "supply at least one of --celltypist-models, --labels-file, --h5ad"
        )

    h = CellHierarchy.from_spec(fuzzy_threshold=args.fuzzy_threshold)
    rows, unresolved = audit(h, label_sets)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "vocabulary_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (outdir / "alias_stubs.py").write_text(
        emit_alias_stubs(unresolved), encoding="utf-8"
    )

    methods = Counter(str(r["match_method"]) for r in rows)
    total = len(rows)
    logger.info("labels audited : %d", total)
    for method, count in methods.most_common():
        logger.info("  %-20s %5d  (%.1f%%)", method, count, 100.0 * count / total)
    flagged = sum(1 for r in rows if r["needs_review"])
    logger.info("needs review   : %d (%.1f%%)", flagged, 100.0 * flagged / total)
    logger.info("written to     : %s", outdir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
