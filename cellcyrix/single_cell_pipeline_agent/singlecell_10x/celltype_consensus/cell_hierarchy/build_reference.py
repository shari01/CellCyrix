"""
Emit the reference database from the spec.

    python -m cell_hierarchy.build_reference --outdir data [--xlsx]

Outputs
-------
    cell_type_hierarchy.csv    one row per node, with flattened lineage columns
    cell_type_aliases.csv      one row per (source_vocabulary, raw_label)
    cell_state_vocabulary.csv  state axis, orthogonal to identity
    coverage_report.csv        per-vocabulary and per-lineage coverage counts
    cell_type_reference.xlsx   optional multi-sheet workbook of the above

Deterministic: identical spec in, byte-identical CSVs out.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Sequence

from .resolver import CellHierarchy
from .spec import ALIASES, COMPARTMENT_LABELS, LEVEL_NAMES, SOURCES, STATES, flat_nodes

logger = logging.getLogger(__name__)

HIERARCHY_COLUMNS = [
    "node_id",
    "canonical_label",
    "level",
    "level_name",
    "parent_id",
    "lineage",
    "class",
    "main_cell_type",
    "subtype",
    "fine_subtype",
    "cl_id",
    "tissue_scope",
    "marker_genes_core",
    "n_children",
    "n_descendants",
    "is_terminal",
    "path_node_ids",
    "path_labels",
    "n_aliases",
    "alias_sources",
]

ALIAS_COLUMNS = [
    "source_vocabulary",
    "raw_label",
    "normalized_label",
    "node_id",
    "canonical_label",
    "level_name",
    "lineage",
]

STATE_COLUMNS = ["state_id", "axis", "display_label", "surface_forms"]

COVERAGE_COLUMNS = [
    "scope_type",
    "scope",
    "n_nodes",
    "n_aliases",
    "n_nodes_with_alias",
    "pct_nodes_with_alias",
]


def build_hierarchy_rows(h: CellHierarchy) -> List[Dict[str, object]]:
    """One row per node, with the lineage path flattened into columns."""
    alias_count: Dict[str, int] = {}
    alias_sources: Dict[str, set] = {}
    for node_id, by_source in ALIASES.items():
        for source, labels in by_source.items():
            alias_count[node_id] = alias_count.get(node_id, 0) + len(labels)
            alias_sources.setdefault(node_id, set()).add(source)
    for _label, node_id in COMPARTMENT_LABELS.items():
        alias_count[node_id] = alias_count.get(node_id, 0) + 1
        alias_sources.setdefault(node_id, set()).add("_compartment")

    rows: List[Dict[str, object]] = []
    for node_id in sorted(h.nodes):
        node = h.nodes[node_id]
        levels = h.level_labels(node_id)
        rows.append(
            {
                "node_id": node.node_id,
                "canonical_label": node.canonical_label,
                "level": node.level,
                "level_name": node.level_name,
                "parent_id": node.parent_id,
                "lineage": levels[LEVEL_NAMES[0]],
                "class": levels[LEVEL_NAMES[1]],
                "main_cell_type": levels[LEVEL_NAMES[2]],
                "subtype": levels[LEVEL_NAMES[3]],
                "fine_subtype": levels[LEVEL_NAMES[4]],
                "cl_id": node.cl_id,
                "tissue_scope": node.tissue_scope,
                "marker_genes_core": node.markers,
                "n_children": len(h._children.get(node_id, ())),
                "n_descendants": len(h.subtree(node_id)) - 1,
                "is_terminal": node.is_terminal,
                "path_node_ids": "|".join(node.path_ids),
                "path_labels": " > ".join(
                    h.nodes[n].canonical_label for n in node.path_ids
                ),
                "n_aliases": alias_count.get(node_id, 0),
                "alias_sources": ";".join(sorted(alias_sources.get(node_id, ()))),
            }
        )
    return rows


def build_alias_rows(h: CellHierarchy) -> List[Dict[str, object]]:
    """One row per ``(source_vocabulary, raw_label)`` alias, with its target node."""
    from .resolver import normalize

    rows: List[Dict[str, object]] = []
    for node_id, by_source in ALIASES.items():
        node = h.nodes[node_id]
        levels = h.level_labels(node_id)
        for source in sorted(by_source):
            for raw in by_source[source]:
                rows.append(
                    {
                        "source_vocabulary": source,
                        "raw_label": raw,
                        "normalized_label": normalize(raw).normalized,
                        "node_id": node_id,
                        "canonical_label": node.canonical_label,
                        "level_name": node.level_name,
                        "lineage": levels[LEVEL_NAMES[0]],
                    }
                )
    for raw, node_id in COMPARTMENT_LABELS.items():
        node = h.nodes[node_id]
        levels = h.level_labels(node_id)
        rows.append(
            {
                "source_vocabulary": "_compartment",
                "raw_label": raw,
                "normalized_label": normalize(raw).normalized,
                "node_id": node_id,
                "canonical_label": node.canonical_label,
                "level_name": node.level_name,
                "lineage": levels[LEVEL_NAMES[0]],
            }
        )
    rows.sort(key=lambda r: (r["source_vocabulary"], r["raw_label"].lower()))
    return rows


def build_state_rows() -> List[Dict[str, object]]:
    """One row per cell state. States are an axis orthogonal to identity, so they
    are emitted separately and never enter the hierarchy tree."""
    return [
        {
            "state_id": state_id,
            "axis": axis,
            "display_label": display,
            "surface_forms": ";".join(forms),
        }
        for state_id, (axis, display, forms) in sorted(STATES.items())
    ]


def build_coverage_rows(
    hierarchy_rows: Sequence[Dict[str, object]],
    alias_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Per-vocabulary and per-lineage coverage counts, for spotting thin areas."""
    rows: List[Dict[str, object]] = []
    total_nodes = len(hierarchy_rows)

    for source in list(SOURCES) + ["_compartment"]:
        subset = [r for r in alias_rows if r["source_vocabulary"] == source]
        nodes_hit = {r["node_id"] for r in subset}
        rows.append(
            {
                "scope_type": "vocabulary",
                "scope": source,
                "n_nodes": total_nodes,
                "n_aliases": len(subset),
                "n_nodes_with_alias": len(nodes_hit),
                "pct_nodes_with_alias": round(100.0 * len(nodes_hit) / total_nodes, 2)
                if total_nodes
                else 0.0,
            }
        )

    by_lineage: Dict[str, List[Dict[str, object]]] = {}
    for row in hierarchy_rows:
        by_lineage.setdefault(str(row["lineage"]), []).append(row)
    for lineage, nodes in sorted(by_lineage.items()):
        node_ids = {r["node_id"] for r in nodes}
        subset = [r for r in alias_rows if r["node_id"] in node_ids]
        hit = {r["node_id"] for r in subset}
        rows.append(
            {
                "scope_type": "lineage",
                "scope": lineage,
                "n_nodes": len(nodes),
                "n_aliases": len(subset),
                "n_nodes_with_alias": len(hit),
                "pct_nodes_with_alias": round(100.0 * len(hit) / len(nodes), 2),
            }
        )

    by_level: Dict[str, List[Dict[str, object]]] = {}
    for row in hierarchy_rows:
        by_level.setdefault(str(row["level_name"]), []).append(row)
    for level_name in LEVEL_NAMES:
        nodes = by_level.get(level_name, [])
        if not nodes:
            continue
        node_ids = {r["node_id"] for r in nodes}
        subset = [r for r in alias_rows if r["node_id"] in node_ids]
        hit = {r["node_id"] for r in subset}
        rows.append(
            {
                "scope_type": "level",
                "scope": level_name,
                "n_nodes": len(nodes),
                "n_aliases": len(subset),
                "n_nodes_with_alias": len(hit),
                "pct_nodes_with_alias": round(100.0 * len(hit) / len(nodes), 2),
            }
        )
    return rows


def _write_csv(
    path: Path, columns: Sequence[str], rows: Sequence[Dict[str, object]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def _write_xlsx(path: Path, sheets: Dict[str, tuple]) -> None:
    import pandas as pd
    from openpyxl.styles import Font

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, (columns, rows) in sheets.items():
            pd.DataFrame(rows, columns=list(columns)).to_excel(
                writer, sheet_name=sheet_name[:31], index=False
            )
        for sheet_name, (columns, _rows) in sheets.items():
            ws = writer.sheets[sheet_name[:31]]
            ws.freeze_panes = "A2"
            for idx, col in enumerate(columns, start=1):
                letter = ws.cell(row=1, column=idx).column_letter
                ws.column_dimensions[letter].width = min(max(12, len(str(col)) + 4), 48)
            for cell in ws[1]:
                cell.font = Font(bold=True, name="Arial")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Writes the reference CSVs and returns an exit code."""
    # Configured HERE rather than at module scope so importing this module does
    # not reconfigure the host application's logging (Rule: nothing runs at
    # import time). The bare "%(message)s" format keeps the summary below
    # readable as the aligned table it was written to be.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Build the cell-type hierarchy reference."
    )
    parser.add_argument("--outdir", default="data", help="output directory")
    parser.add_argument(
        "--xlsx", action="store_true", help="also emit a multi-sheet workbook"
    )
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes = flat_nodes()
    h = CellHierarchy(nodes, ALIASES, COMPARTMENT_LABELS)

    hierarchy_rows = build_hierarchy_rows(h)
    alias_rows = build_alias_rows(h)
    state_rows = build_state_rows()
    coverage_rows = build_coverage_rows(hierarchy_rows, alias_rows)

    _write_csv(outdir / "cell_type_hierarchy.csv", HIERARCHY_COLUMNS, hierarchy_rows)
    _write_csv(outdir / "cell_type_aliases.csv", ALIAS_COLUMNS, alias_rows)
    _write_csv(outdir / "cell_state_vocabulary.csv", STATE_COLUMNS, state_rows)
    _write_csv(outdir / "coverage_report.csv", COVERAGE_COLUMNS, coverage_rows)

    if args.xlsx:
        _write_xlsx(
            outdir / "cell_type_reference.xlsx",
            {
                "hierarchy": (HIERARCHY_COLUMNS, hierarchy_rows),
                "aliases": (ALIAS_COLUMNS, alias_rows),
                "states": (STATE_COLUMNS, state_rows),
                "coverage": (COVERAGE_COLUMNS, coverage_rows),
            },
        )

    logger.info("nodes           : %d", len(hierarchy_rows))
    logger.info("aliases         : %d", len(alias_rows))
    logger.info("states          : %d", len(state_rows))
    logger.info(
        "lineages (L0)   : %d", sum(1 for r in hierarchy_rows if r["level"] == 0)
    )
    logger.info(
        "terminal nodes  : %d", sum(1 for r in hierarchy_rows if r["is_terminal"])
    )
    logger.info("written to      : %s", outdir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
