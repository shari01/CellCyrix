"""
contrasts.py — the single place that decides WHICH group is the reference.

Why this module exists
----------------------
Every group comparison in the pipeline used ``sorted(groups)`` + ``combinations``,
which makes the *alphabetically first* group the reference. That is a coin flip
against biology::

    ["Control", "Tumor"]    -> ref="Control"   correct by luck
    ["Disease", "Healthy"]  -> ref="Disease"   INVERTED  (log2FC>0 = up in Healthy)
    ["AD", "Control"]       -> ref="AD"        INVERTED
    ["Post", "Pre"]         -> ref="Post"      INVERTED

The p-values and gene lists were right; the SIGN was arbitrary. A silent sign flip
is the worst class of bug in a DE pipeline because nothing downstream looks wrong.

This module resolves ONE reference group per run — explicitly configured, or matched
from a baseline vocabulary — and orients every contrast as ``focus_vs_reference``.
It also stamps the literal contrast onto every output row (:func:`stamp_contrast`),
so the direction of a result is auditable from the CSV alone and never has to be
re-derived from a filename or from the column order.

When no baseline can be identified the orientation falls back to alphabetical, the
same as before — but ``reference_selection`` records that the direction is arbitrary
instead of leaving it implicit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation only — this module does no pandas work at import
    import pandas as pd

import logging
import re
from itertools import combinations
from typing import Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Vocabulary for "this arm is the baseline / untreated / unaffected side".
# Deliberately broad: a false positive here only picks the wrong reference in a
# design that has no obvious control, which is exactly the case the explicit
# `reference_group` config setting exists for.
BASELINE_PATTERN = re.compile(
    r"normal|healthy|control|ctrl|baseline|adjacent|non[-_ ]?tumou?r|"
    r"benign|naive|untreated|unstimulated|unstim|vehicle|mock|sham|placebo|"
    r"wild[-_ ]?type|pre[-_ ]?infusion|pre[-_ ]?treatment|pre[-_ ]?therapy",
    re.IGNORECASE,
)

# Standalone tokens that mean "baseline" but are too short to match loosely
# ("wt" would otherwise hit "wtCD8"; "pre" would hit "prednisone").
_BASELINE_EXACT = frozenset({"wt", "pre", "d0", "day0", "t0", "ref", "reference", "nc"})

# Columns :func:`stamp_contrast` writes onto every DE result row.
CONTRAST_COLUMNS: Tuple[str, ...] = (
    "comparison",
    "focus_group",
    "reference_group",
    "contrast_direction",
    "reference_selection",
)


def _is_baseline(label: str) -> bool:
    s = str(label).strip()
    if s.lower() in _BASELINE_EXACT:
        return True
    return bool(BASELINE_PATTERN.search(s))


def resolve_reference_group(
    groups: Sequence[str],
    explicit: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """Pick the reference (control/baseline) arm. Returns ``(reference, reason)``.

    Resolution order:

    1. ``explicit`` when it names a real group — always wins, no heuristics.
    2. exactly one group matching the baseline vocabulary.
    3. several matches -> the alphabetically first match, flagged as ambiguous.
    4. no match -> ``None``; the caller falls back to alphabetical orientation and
       the returned reason says so in words, so the manifest and every CSV record
       that the direction was not biologically determined.
    """
    labels = [str(g) for g in groups]
    if len(labels) < 2:
        return None, f"fewer than 2 groups ({labels}); no contrast to orient."

    if explicit is not None and str(explicit).strip():
        want = str(explicit).strip()
        if want in labels:
            return want, f"explicitly configured reference_group={want!r}."
        # Case-insensitive second chance before giving up.
        ci = [g for g in labels if g.lower() == want.lower()]
        if ci:
            return ci[
                0
            ], f"explicitly configured reference_group={want!r} (matched {ci[0]!r})."
        logger.warning(
            "[CONTRAST] configured reference_group=%r is not one of %s; falling back to automatic baseline detection.",
            want,
            labels,
        )

    hits = sorted(g for g in labels if _is_baseline(g))
    if len(hits) == 1:
        return hits[0], f"baseline vocabulary matched {hits[0]!r} among {labels}."
    if len(hits) > 1:
        return hits[0], (
            f"baseline vocabulary matched several groups {hits}; used {hits[0]!r} "
            f"(alphabetically first match). Set reference_group in the config to "
            f"remove the ambiguity."
        )
    return None, (
        f"no baseline/control group identified among {labels}; contrast direction "
        f"is ALPHABETICAL and therefore arbitrary. Set reference_group in the config "
        f"to make log2FoldChange signs biologically interpretable."
    )


def ordered_contrasts(
    groups: Iterable[str],
    reference: Optional[str] = None,
) -> Tuple[List[Tuple[str, str]], Optional[str], str]:
    """Every group pair as ``(focus, reference)``, oriented against the baseline.

    Returns ``(pairs, reference_group, reason)``. Coverage is unchanged from the
    previous ``combinations(sorted(groups), 2)`` — every pair is still tested; only
    the ORIENTATION of each pair changes. Pairs that do not involve the reference
    group (possible with 3+ arms) keep the alphabetical orientation, since nothing
    identifies which of two non-baseline arms is the "control".
    """
    labels = sorted({str(g) for g in groups})
    ref, reason = resolve_reference_group(labels, explicit=reference)

    pairs: List[Tuple[str, str]] = []
    for a, b in combinations(labels, 2):  # a < b alphabetically
        if ref == a:
            pairs.append((b, a))
        elif ref == b:
            pairs.append((a, b))
        else:
            pairs.append((b, a))  # neither is the baseline
    return pairs, ref, reason


def contrast_label(focus: str, ref: str) -> str:
    """The literal contrast string used in filenames and the ``comparison`` column."""
    return f"{focus}_vs_{ref}"


def direction_note(focus: str, ref: str) -> str:
    """Plain-language reading of a positive log fold change for this contrast."""
    return f"positive log2FoldChange = higher in {focus} than in {ref}"


def stamp_contrast(
    df: pd.DataFrame, *, focus: str, ref: str, reference_selection: str
) -> pd.DataFrame:
    """Write the contrast identity onto every row of a DE table (in place).

    Adds :data:`CONTRAST_COLUMNS`. This is the auditability requirement: a reader
    holding only the CSV can tell which arm is the numerator without consulting the
    filename, the log, or the order of the groups.
    """
    df["comparison"] = contrast_label(focus, ref)
    df["focus_group"] = str(focus)
    df["reference_group"] = str(ref)
    df["contrast_direction"] = direction_note(focus, ref)
    df["reference_selection"] = reference_selection
    return df
