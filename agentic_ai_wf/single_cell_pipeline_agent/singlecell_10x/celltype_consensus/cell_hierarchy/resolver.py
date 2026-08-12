"""
Cell-type hierarchy resolver — PURE LOGIC.

Contract enforced by this module:
  * no LLM / network / API calls of any kind
  * no disease strings, and no branching on disease or tissue
  * deterministic: same input -> same output, always
  * total_in == total_out: resolve_many() returns exactly one Resolution per
    input label. Unresolvable labels come back flagged, never dropped.

Tissue is metadata. `tissue` is accepted only as an optional *tie-breaker among
already-valid candidates* and never as a gate that can change a confident match.

Public surface
--------------
    normalize(label)                        -> NormalizedLabel
    CellHierarchy.from_spec()               -> CellHierarchy
    CellHierarchy.from_csv(dir)             -> CellHierarchy
    h.resolve(label, source=, tissue=)      -> Resolution
    h.resolve_many(labels, ...)             -> list[Resolution]   (len preserved)
    h.rollup(node_id, level)                -> node_id | None
    h.lowest_common_ancestor([node_id, ...])-> node_id | None
    h.consensus(votes)                      -> Consensus
    h.subtree(node_id)                      -> list[node_id]
    h.describe(node_id)                     -> dict
"""

from __future__ import annotations

import csv
import difflib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Base class for this module's error type. This package is a self-contained
# sub-library (invariant 4 in cell_hierarchy/__init__.py: pure logic, no pipeline
# dependencies) and is documented as importable standalone via
# ``from cell_hierarchy import CellHierarchy``. Under that import there is no parent
# package, so the pipeline-relative import raises ImportError — hence the fallback
# rather than an unconditional import. PipelineComputationError derives from
# RuntimeError, so the two bases are behaviourally interchangeable for callers.
try:
    from ...exceptions import PipelineComputationError as _ERROR_BASE
except ImportError:  # pragma: no cover - the standalone-import path
    _ERROR_BASE = RuntimeError  # type: ignore[assignment,misc]

from .spec import (
    ALIASES,
    COMPARTMENT_LABELS,
    LEVEL_NAMES,
    PROTECTED_TOKENS,
    FlatNode,
    flat_nodes,
    state_lookup,
)

# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #

_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-")
_GREEK = {
    "\u03b1": " alpha ",
    "\u03b2": " beta ",
    "\u03b3": " gamma ",
    "\u03b4": " delta ",
    "\u03b5": " epsilon ",
    "\u03ba": " kappa ",
    "\u03bb": " lambda ",
    "\u03bc": " mu ",
    "\u03c6": " phi ",
    "\u03c8": " psi ",
}
_PUNCT_RE = re.compile(r"[^a-z0-9+\- ]+")
_WS_RE = re.compile(r"\s+")
_PLURAL_EXCEPT = frozenset(
    {
        "cells",
        "vessels",  # handled by token rules below
    }
)

# token-level rewrites applied after punctuation stripping
_TOKEN_REWRITES: Dict[str, str] = {
    "cells": "cell",
    "lymphocytes": "lymphocyte",
    "lymphocyte": "cell",
    "monos": "monocyte",
    "mono": "monocyte",
    "monocytes": "monocyte",
    "macs": "macrophage",
    "mac": "macrophage",
    "macrophages": "macrophage",
    "mo": "monocyte",
    "fibroblasts": "fibroblast",
    "ecs": "endothelial",
    "ec": "endothelial",
    "tcells": "t cell",
    "bcells": "b cell",
    "pos": "+",
    "positive": "+",
    "neg": "-",
    "negative": "-",
    "hi": "high",
    "lo": "low",
    "prolif": "proliferating",
}

# multi-word rewrites applied to the whole string (longest first at build time)
_PHRASE_REWRITES: List[Tuple[str, str]] = [
    ("t lymphocyte", "t cell"),
    ("b lymphocyte", "b cell"),
    ("natural killer", "nk"),
    ("nk t cell", "nkt cell"),
    ("red blood cell", "erythrocyte"),
    ("white blood cell", "leukocyte"),
    ("dendritic cell", "dc"),
    ("smooth muscle cell", "smooth muscle"),
    ("alveolar type i", "at1"),
    ("alveolar type ii", "at2"),
    ("alveolar type 1", "at1"),
    ("alveolar type 2", "at2"),
    ("type i pneumocyte", "at1"),
    ("type ii pneumocyte", "at2"),
    ("type 1 pneumocyte", "at1"),
    ("type 2 pneumocyte", "at2"),
    ("t helper", "th"),
    ("t regulatory", "regulatory t"),
    ("regulatory t cell", "treg"),
    ("central memory", "tcm"),
    ("effector memory", "tem"),
]
_PHRASE_REWRITES.sort(key=lambda kv: -len(kv[0]))

_STATE_FORMS: List[Tuple[str, str]] = sorted(
    state_lookup().items(), key=lambda kv: -len(kv[0])
)


class LabelConservationError(_ERROR_BASE):
    """A batch label operation returned a different number of rows than it got.

    One resolution per input label is what lets a caller zip the results back
    onto its own rows; a silent drop would misalign every downstream column.

    The base class is resolved above: ``PipelineComputationError`` when this package
    is imported as part of the pipeline (so ``except PipelineError`` catches it like
    every other pipeline failure), and plain ``RuntimeError`` when it is imported
    standalone. ``PipelineComputationError`` inherits ``RuntimeError``, so
    ``except RuntimeError`` behaves identically either way.
    """


@dataclass(frozen=True)
class NormalizedLabel:
    """Result of normalising a raw label."""

    raw: str
    normalized: str  # canonical comparison key
    tokens: Tuple[str, ...]  # normalised token set
    states: Tuple[str, ...]  # state_ids stripped out, if any
    residual: str  # normalized string with states removed


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _basic_clean(label: str) -> str:
    text = label.translate(_DASHES)
    for greek, latin in _GREEK.items():
        text = text.replace(greek, latin)
    text = _strip_accents(text).lower()
    # unify +/- markers so 'CD16+', 'CD16 +' and 'CD16-positive' converge
    text = text.replace("(+)", "+").replace("(-)", "-")
    text = re.sub(r"\bcd(\d+)\s*\+", r"cd\1+", text)
    text = re.sub(r"\b(\w+)\s*-\s*positive\b", r"\1+", text)
    text = re.sub(r"\b(\w+)\s*-\s*negative\b", r"\1-", text)
    # A hyphen is a word separator ('T-cells', 'gamma-delta', 'non-classical')
    # EXCEPT when it trails a token as a negation marker ('CD16-', 'KRT5-'),
    # where it is load-bearing and must survive.
    text = re.sub(r"(?<=[a-z0-9])-(?=[a-z])", " ", text)
    text = text.replace("/", " ").replace("_", " ").replace("&", " ")
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def _apply_phrase_rewrites(text: str) -> str:
    for src, dst in _PHRASE_REWRITES:
        if src in text:
            text = text.replace(src, dst)
    return text


def _rewrite_tokens(text: str) -> List[str]:
    out: List[str] = []
    for tok in text.split():
        out.append(_TOKEN_REWRITES.get(tok, tok))
    # collapse a trailing bare 'cell' duplication ('t cell cell')
    deduped: List[str] = []
    for tok in out:
        if tok == "cell" and deduped and deduped[-1] == "cell":
            continue
        deduped.append(tok)
    return deduped


def _extract_states(text: str) -> Tuple[str, Tuple[str, ...]]:
    """Strip recognised state qualifiers, honouring PROTECTED_TOKENS."""
    found: List[str] = []
    residual = text
    for form, state_id in _STATE_FORMS:
        if form in PROTECTED_TOKENS:
            continue
        pattern = r"(?<![a-z0-9])" + re.escape(form) + r"(?![a-z0-9])"
        if re.search(pattern, residual):
            candidate = _WS_RE.sub(" ", re.sub(pattern, " ", residual)).strip()
            residual = candidate
            if state_id not in found:
                found.append(state_id)
            if not residual:
                # The label was nothing but state qualifiers ('Cycling',
                # 'Doublet'). Report the state and an empty identity rather
                # than inventing one; resolve() turns this into state_only.
                break
    return residual, tuple(found)


def normalize(label: str) -> NormalizedLabel:
    """Deterministic label normalisation. No side effects."""
    raw = label if label is not None else ""
    cleaned = _basic_clean(str(raw))
    cleaned = _apply_phrase_rewrites(cleaned)
    tokens = _rewrite_tokens(cleaned)
    normalized = " ".join(tokens)
    residual, states = _extract_states(normalized)
    residual_tokens = _rewrite_tokens(residual)
    return NormalizedLabel(
        raw=str(raw),
        normalized=normalized,
        tokens=tuple(residual_tokens),
        states=states,
        residual=" ".join(residual_tokens),
    )


# --------------------------------------------------------------------------- #
# Resolution result types
# --------------------------------------------------------------------------- #

MATCH_EXACT_ALIAS = "exact_alias"
MATCH_NORMALIZED_ALIAS = "normalized_alias"
MATCH_CANONICAL_LABEL = "canonical_label"
MATCH_NODE_ID = "node_id"
MATCH_CL_ID = "cl_id"
MATCH_COMPARTMENT = "compartment_label"
MATCH_FUZZY = "fuzzy"
MATCH_STATE_ONLY = "state_only"
MATCH_UNRESOLVED = "unresolved"

CONFIDENCE = {
    MATCH_EXACT_ALIAS: 1.00,
    MATCH_NODE_ID: 1.00,
    MATCH_CL_ID: 1.00,
    MATCH_CANONICAL_LABEL: 0.98,
    MATCH_NORMALIZED_ALIAS: 0.95,
    MATCH_COMPARTMENT: 0.80,
    MATCH_FUZZY: 0.55,
    MATCH_STATE_ONLY: 0.00,
    MATCH_UNRESOLVED: 0.00,
}

UNRESOLVED_NODE = "unknown_cell"
UNASSIGNED_ROOT = "unassigned"


@dataclass
class Resolution:
    """One input label in, one Resolution out. Always."""

    raw_label: str
    source: Optional[str]
    node_id: str
    canonical_label: str
    level: int
    level_name: str
    lineage: str
    main_cell_type: str
    subtype: str
    cl_id: str
    states: Tuple[str, ...]
    match_method: str
    confidence: float
    resolved: bool
    needs_review: bool
    note: str = ""

    def as_dict(self) -> Dict[str, object]:
        """Flatten this resolution into columns for a per-cluster annotation table."""
        return {
            "raw_label": self.raw_label,
            "source": self.source or "",
            "node_id": self.node_id,
            "canonical_label": self.canonical_label,
            "level": self.level,
            "level_name": self.level_name,
            "lineage": self.lineage,
            "main_cell_type": self.main_cell_type,
            "subtype": self.subtype,
            "cl_id": self.cl_id,
            "states": ";".join(self.states),
            "match_method": self.match_method,
            "confidence": self.confidence,
            "resolved": self.resolved,
            "needs_review": self.needs_review,
            "note": self.note,
        }


@dataclass
class Consensus:
    """Multi-voter agreement summary computed on the hierarchy, not on strings."""

    votes: Dict[str, str]  # source -> raw label
    resolutions: List[Resolution] = field(default_factory=list)
    consensus_node_id: str = UNRESOLVED_NODE
    consensus_label: str = ""
    consensus_level: int = -1
    consensus_level_name: str = ""
    agreement_score: float = 0.0
    n_voters: int = 0
    n_resolved: int = 0
    exact_agreement: bool = False
    dissenting_sources: Tuple[str, ...] = ()
    states: Tuple[str, ...] = ()
    note: str = ""

    def as_dict(self) -> Dict[str, object]:
        """Flatten this consensus into columns for a per-cluster annotation table."""
        return {
            "consensus_node_id": self.consensus_node_id,
            "consensus_label": self.consensus_label,
            "consensus_level": self.consensus_level,
            "consensus_level_name": self.consensus_level_name,
            "agreement_score": round(self.agreement_score, 4),
            "n_voters": self.n_voters,
            "n_resolved": self.n_resolved,
            "exact_agreement": self.exact_agreement,
            "dissenting_sources": ";".join(self.dissenting_sources),
            "states": ";".join(self.states),
            "note": self.note,
        }


# --------------------------------------------------------------------------- #
# The hierarchy
# --------------------------------------------------------------------------- #

# Agreement credit by the level at which voters converge. Deeper agreement is
# worth more; agreeing only that a cell is 'haematopoietic' is worth little.
LEVEL_CREDIT = (0.20, 0.40, 0.65, 0.85, 1.00)


class CellHierarchy:
    """In-memory hierarchy with resolution, rollup and consensus operations."""

    def __init__(
        self,
        nodes: Sequence[FlatNode],
        alias_map: Mapping[str, Mapping[str, Sequence[str]]],
        compartment_map: Mapping[str, str],
        fuzzy_threshold: float = 0.88,
    ) -> None:
        self.nodes: Dict[str, FlatNode] = {n.node_id: n for n in nodes}
        self.fuzzy_threshold = fuzzy_threshold
        self._children: Dict[str, List[str]] = {nid: [] for nid in self.nodes}
        for node in nodes:
            if node.parent_id:
                self._children[node.parent_id].append(node.node_id)

        # exact alias index: (source, raw) -> node_id  and  raw -> node_id
        self._alias_exact: Dict[Tuple[str, str], str] = {}
        self._alias_any: Dict[str, List[str]] = {}
        # normalised index: normalized -> [node_id]
        self._norm_index: Dict[str, List[str]] = {}
        self._cl_index: Dict[str, str] = {}

        for node in nodes:
            self._index_norm(node.canonical_label, node.node_id)
            self._index_norm(node.node_id.replace("_", " "), node.node_id)
            if node.cl_id:
                self._cl_index[node.cl_id.upper()] = node.node_id

        conflicts: List[str] = []
        for node_id, by_source in alias_map.items():
            if node_id not in self.nodes:
                raise KeyError(f"alias table references unknown node_id {node_id!r}")
            for source, labels in by_source.items():
                for label in labels:
                    key = (source, label)
                    prior = self._alias_exact.get(key)
                    if prior is not None and prior != node_id:
                        # A single vocabulary cannot mean two things by one label.
                        # Silently keeping the first insertion makes resolution
                        # depend on dict order, so refuse to build instead.
                        conflicts.append(
                            f"{source}:{label!r} claimed by both "
                            f"{prior!r} and {node_id!r}"
                        )
                        continue
                    self._alias_exact[key] = node_id
                    self._alias_any.setdefault(label, []).append(node_id)
                    self._index_norm(label, node_id)
        if conflicts:
            raise ValueError(
                "alias table has ambiguous labels within a single vocabulary:\n  - "
                + "\n  - ".join(sorted(conflicts))
            )

        self._compartment: Dict[str, str] = {}
        for label, node_id in compartment_map.items():
            if node_id not in self.nodes:
                raise KeyError(
                    f"compartment table references unknown node_id {node_id!r}"
                )
            self._compartment[normalize(label).normalized] = node_id

        self._fuzzy_keys: Tuple[str, ...] = tuple(self._norm_index)

    # ---------------------------------------------------------------- builders
    @classmethod
    def from_spec(cls, fuzzy_threshold: float = 0.88) -> "CellHierarchy":
        """Build the hierarchy from the Python spec — the normal path.

        Reads no files: the tree, aliases and compartment labels are compiled from
        ``spec/*.py``, so the resolver works in a checkout with no reference CSVs.

        Args:
            fuzzy_threshold: Minimum similarity for a fuzzy label match to count.
        """
        return cls(flat_nodes(), ALIASES, COMPARTMENT_LABELS, fuzzy_threshold)

    @classmethod
    def from_csv(
        cls, data_dir: str | Path, fuzzy_threshold: float = 0.88
    ) -> "CellHierarchy":
        """Rebuild from emitted CSVs, so the resolver can run without the spec."""
        data_dir = Path(data_dir)
        nodes: List[FlatNode] = []
        with (data_dir / "cell_type_hierarchy.csv").open(
            newline="", encoding="utf-8"
        ) as fh:
            for row in csv.DictReader(fh):
                nodes.append(
                    FlatNode(
                        node_id=row["node_id"],
                        canonical_label=row["canonical_label"],
                        cl_id=row["cl_id"],
                        tissue_scope=row["tissue_scope"],
                        markers=row["marker_genes_core"],
                        level=int(row["level"]),
                        parent_id=row["parent_id"],
                        path_ids=tuple(row["path_node_ids"].split("|"))
                        if row["path_node_ids"]
                        else (),
                        is_terminal=row["is_terminal"].strip().lower()
                        in {"true", "1", "yes"},
                    )
                )
        aliases: Dict[str, Dict[str, List[str]]] = {}
        compartments: Dict[str, str] = {}
        with (data_dir / "cell_type_aliases.csv").open(
            newline="", encoding="utf-8"
        ) as fh:
            for row in csv.DictReader(fh):
                if row["source_vocabulary"] == "_compartment":
                    compartments[row["raw_label"]] = row["node_id"]
                else:
                    aliases.setdefault(row["node_id"], {}).setdefault(
                        row["source_vocabulary"], []
                    ).append(row["raw_label"])
        return cls(nodes, aliases, compartments, fuzzy_threshold)

    # ------------------------------------------------------------------ index
    def _index_norm(self, label: str, node_id: str) -> None:
        key = normalize(label).normalized
        if not key:
            return
        bucket = self._norm_index.setdefault(key, [])
        if node_id not in bucket:
            bucket.append(node_id)

    # ------------------------------------------------------------------ tree
    def ancestors(self, node_id: str, include_self: bool = True) -> List[str]:
        """Node ids from the root down to ``node_id``, or ``[]`` if it is unknown.

        Args:
            node_id: The node to walk up from.
            include_self: Whether ``node_id`` itself terminates the returned path.
        """
        node = self.nodes.get(node_id)
        if node is None:
            return []
        path = list(node.path_ids)
        return path if include_self else path[:-1]

    def subtree(self, node_id: str) -> List[str]:
        """``node_id`` plus every descendant, or ``[]`` if it is unknown."""
        if node_id not in self.nodes:
            return []
        out, stack = [], [node_id]
        while stack:
            cur = stack.pop()
            out.append(cur)
            stack.extend(self._children.get(cur, ()))
        return out

    def rollup(self, node_id: str, level: int) -> Optional[str]:
        """Coarsen a node to the requested level. None if it never reaches it."""
        path = self.ancestors(node_id)
        return path[level] if 0 <= level < len(path) else None

    def level_labels(self, node_id: str) -> Dict[str, str]:
        """The flattened lineage/main_cell_type/subtype view of a node."""
        path = self.ancestors(node_id)
        out = {name: "" for name in LEVEL_NAMES}
        for depth, nid in enumerate(path):
            if depth < len(LEVEL_NAMES):
                out[LEVEL_NAMES[depth]] = self.nodes[nid].canonical_label
        return out

    def lowest_common_ancestor(self, node_ids: Sequence[str]) -> Optional[str]:
        """Deepest node that is an ancestor of every id given.

        This is how disagreeing voters are reconciled: two calls at different
        granularity collapse to the most specific node both actually support.

        Returns:
            The node id, or None when the ids share no ancestor or none are known.
        """
        paths = [self.ancestors(nid) for nid in node_ids if nid in self.nodes]
        paths = [p for p in paths if p]
        if not paths:
            return None
        common = paths[0]
        for path in paths[1:]:
            shared: List[str] = []
            for a, b in zip(
                common, path, strict=False
            ):  # prefix walk: lengths differ by design
                if a != b:
                    break
                shared.append(a)
            common = shared
            if not common:
                return None
        return common[-1]

    def describe(self, node_id: str) -> Dict[str, object]:
        """Full record for one node — labels, level, lineage and path.

        Returns:
            A dict of the node's attributes, or ``{}`` if ``node_id`` is unknown.
        """
        node = self.nodes.get(node_id)
        if node is None:
            return {}
        out: Dict[str, object] = {
            "node_id": node.node_id,
            "canonical_label": node.canonical_label,
            "cl_id": node.cl_id,
            "level": node.level,
            "level_name": node.level_name,
            "parent_id": node.parent_id,
            "path": " > ".join(self.nodes[n].canonical_label for n in node.path_ids),
            "tissue_scope": node.tissue_scope,
            "marker_genes_core": node.markers,
            "n_descendants": len(self.subtree(node_id)) - 1,
        }
        out.update(self.level_labels(node_id))
        return out

    # -------------------------------------------------------------- resolution
    def _pick(
        self, candidates: Sequence[str], tissue: Optional[str]
    ) -> Tuple[str, str]:
        """Choose among equally-scoring candidates. Deterministic."""
        if len(candidates) == 1:
            return candidates[0], ""
        ranked = sorted(candidates)
        if tissue:
            token = normalize(tissue).normalized
            scoped = [
                nid
                for nid in ranked
                if token and token in normalize(self.nodes[nid].tissue_scope).normalized
            ]
            if len(scoped) == 1:
                return scoped[
                    0
                ], f"tissue hint '{tissue}' broke a {len(ranked)}-way tie"
            if scoped:
                ranked = scoped
        # prefer the shallowest node: coarser is the safer default
        ranked.sort(key=lambda nid: (self.nodes[nid].level, nid))
        return ranked[0], f"ambiguous across {len(candidates)} nodes; took shallowest"

    def resolve(
        self,
        label: str,
        source: Optional[str] = None,
        tissue: Optional[str] = None,
    ) -> Resolution:
        """Resolve one raw label. Never raises; never returns None."""
        norm = normalize(label)

        if not norm.normalized:
            return self._make(
                label, source, UNRESOLVED_NODE, norm, MATCH_UNRESOLVED, "empty label"
            )

        # A label consisting only of state qualifiers ('Cycling', 'Doublet')
        # names no cell identity. Report the state and abstain on identity
        # rather than guessing a lineage from a state word.
        if norm.states and not norm.residual:
            node = (
                "technical_artefact"
                if {"doublet", "low_quality"} & set(norm.states)
                else UNRESOLVED_NODE
            )
            return self._make(
                label,
                source,
                node,
                norm,
                MATCH_STATE_ONLY,
                "label is state-only ("
                + ",".join(norm.states)
                + "); no identity asserted",
            )

        # 1. exact alias within the declared source
        if source is not None:
            hit = self._alias_exact.get((source, label))
            if hit:
                return self._make(label, source, hit, norm, MATCH_EXACT_ALIAS)

        # 2. exact alias in any source
        any_hit = self._alias_any.get(label)
        if any_hit:
            node_id, note = self._pick(sorted(set(any_hit)), tissue)
            return self._make(label, source, node_id, norm, MATCH_EXACT_ALIAS, note)

        # 3. direct node_id / CL id
        slug = norm.normalized.replace(" ", "_")
        if slug in self.nodes:
            return self._make(label, source, slug, norm, MATCH_NODE_ID)
        cl_probe = str(label).strip().upper()
        if cl_probe in self._cl_index:
            return self._make(
                label, source, self._cl_index[cl_probe], norm, MATCH_CL_ID
            )

        # 4. normalised index, full label then state-stripped residual
        for key, method in (
            (norm.normalized, MATCH_NORMALIZED_ALIAS),
            (norm.residual, MATCH_NORMALIZED_ALIAS),
        ):
            bucket = self._norm_index.get(key)
            if bucket:
                node_id, note = self._pick(bucket, tissue)
                return self._make(label, source, node_id, norm, method, note)

        # 5. compartment labels (coarse but legitimate)
        for key in (norm.normalized, norm.residual):
            if key in self._compartment:
                return self._make(
                    label,
                    source,
                    self._compartment[key],
                    norm,
                    MATCH_COMPARTMENT,
                    "compartment-level label; resolved to containing node",
                )

        # 6. fuzzy fallback, flagged for review
        probe = norm.residual or norm.normalized
        close = difflib.get_close_matches(
            probe, self._fuzzy_keys, n=3, cutoff=self.fuzzy_threshold
        )
        if close:
            bucket = self._norm_index[close[0]]
            node_id, note = self._pick(bucket, tissue)
            ratio = difflib.SequenceMatcher(None, probe, close[0]).ratio()
            detail = f"fuzzy match to '{close[0]}' (ratio {ratio:.2f})"
            return self._make(
                label,
                source,
                node_id,
                norm,
                MATCH_FUZZY,
                "; ".join(filter(None, [detail, note])),
            )

        return self._make(
            label,
            source,
            UNRESOLVED_NODE,
            norm,
            MATCH_UNRESOLVED,
            "no alias, normalised or fuzzy match",
        )

    def _make(
        self,
        label: str,
        source: Optional[str],
        node_id: str,
        norm: NormalizedLabel,
        method: str,
        note: str = "",
    ) -> Resolution:
        node = self.nodes[node_id]
        levels = self.level_labels(node_id)
        # Anything in the 'unassigned' subtree asserts no identity, however it
        # was matched, so it must not count as a resolved vote.
        in_unassigned = UNASSIGNED_ROOT in node.path_ids
        resolved = (
            method not in {MATCH_UNRESOLVED, MATCH_STATE_ONLY} and not in_unassigned
        )
        return Resolution(
            raw_label=str(label),
            source=source,
            node_id=node_id,
            canonical_label=node.canonical_label,
            level=node.level,
            level_name=node.level_name,
            lineage=levels[LEVEL_NAMES[0]],
            main_cell_type=levels[LEVEL_NAMES[2]] or levels[LEVEL_NAMES[1]],
            subtype=levels[LEVEL_NAMES[3]] or levels[LEVEL_NAMES[4]],
            cl_id=node.cl_id,
            states=norm.states,
            match_method=method,
            confidence=CONFIDENCE[method],
            resolved=resolved,
            needs_review=in_unassigned
            or method
            in {
                MATCH_FUZZY,
                MATCH_UNRESOLVED,
                MATCH_COMPARTMENT,
                MATCH_STATE_ONLY,
            },
            note=note,
        )

    def resolve_many(
        self,
        labels: Iterable[str],
        source: Optional[str] = None,
        tissue: Optional[str] = None,
    ) -> List[Resolution]:
        """Batch resolve, one Resolution per input label.

        Raises:
            LabelConservationError: If the output count differs from the input count.
                Enforced with a raise rather than an ``assert`` so it survives
                ``python -O``.
        """
        items = list(labels)
        out = [self.resolve(lbl, source=source, tissue=tissue) for lbl in items]
        if len(out) != len(items):
            raise LabelConservationError(
                f"resolver dropped rows: total_in={len(items)} total_out={len(out)}"
            )
        return out

    # --------------------------------------------------------------- consensus
    def consensus(
        self,
        votes: Mapping[str, str],
        tissue: Optional[str] = None,
        min_voters: int = 2,
    ) -> Consensus:
        """
        Hierarchy-aware multi-voter agreement.

        `votes` maps source vocabulary -> raw label, e.g.
            {"celltypist_lung": "AT2", "singler_hpca": "Epithelial_cells"}

        Agreement is computed on the tree: voters at different granularities
        that sit on one path are treated as agreeing at the shallower depth,
        rather than as a string mismatch. Unresolved voters are excluded from
        the ancestry maths but still counted in n_voters, so abstentions
        visibly depress the score instead of silently vanishing.
        """
        resolutions = [
            self.resolve(label, source=src, tissue=tissue)
            for src, label in votes.items()
        ]
        result = Consensus(
            votes=dict(votes),
            resolutions=resolutions,
            n_voters=len(resolutions),
        )

        usable = [r for r in resolutions if r.resolved]
        result.n_resolved = len(usable)
        state_counter = Counter(s for r in resolutions for s in r.states)
        result.states = tuple(sorted(state_counter))

        if not usable:
            result.note = "no voter resolved to a hierarchy node"
            return result

        if len(usable) < min_voters:
            node = usable[0]
            result.consensus_node_id = node.node_id
            result.consensus_label = node.canonical_label
            result.consensus_level = node.level
            result.consensus_level_name = node.level_name
            result.agreement_score = 0.0
            result.note = (
                f"only {len(usable)} voter(s) resolved; below min_voters={min_voters}"
            )
            return result

        node_ids = [r.node_id for r in usable]
        lca = self.lowest_common_ancestor(node_ids)
        if lca is None:
            result.note = "voters span disjoint lineages; no common ancestor"
            result.dissenting_sources = tuple(sorted(r.source or "" for r in usable))
            return result

        lca_node = self.nodes[lca]
        result.consensus_node_id = lca
        result.consensus_label = lca_node.canonical_label
        result.consensus_level = lca_node.level
        result.consensus_level_name = lca_node.level_name
        result.exact_agreement = len(set(node_ids)) == 1

        # score = depth credit x resolved fraction x mean voter confidence
        depth_credit = LEVEL_CREDIT[min(lca_node.level, len(LEVEL_CREDIT) - 1)]
        resolved_fraction = len(usable) / len(resolutions)
        mean_conf = sum(r.confidence for r in usable) / len(usable)
        result.agreement_score = depth_credit * resolved_fraction * mean_conf

        # a voter is dissenting if the consensus node is not on its own path
        result.dissenting_sources = tuple(
            sorted(
                (r.source or "") for r in usable if lca not in self.ancestors(r.node_id)
            )
        )
        if result.exact_agreement:
            result.note = "all resolved voters agree exactly"
        else:
            result.note = (
                f"agreement at {lca_node.level_name}: {lca_node.canonical_label}"
            )
        return result
