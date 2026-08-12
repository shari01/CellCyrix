"""
safe_names.py — turn a cell-type label into a filename that survives Windows.

Why this exists
---------------
Cell-type labels become filenames all over the pipeline
(``<celltype>_pseudobulk_DE.csv``, ``<celltype>/<celltype>_..._EXPLORATORY.csv``,
volcano PNGs). Two call sites used ``label.replace(" ", "_").replace("/", "_")``,
which covers the space and the forward slash and nothing else. **The colon is the
dangerous one.**

On NTFS, ``a:b`` is not a filename — it is file ``a`` with an *alternate data
stream* named ``b``. ``open("Other: GABAergic interneuron_DE.csv", "w")`` therefore
succeeds, reports no error, and writes real bytes into a stream hanging off a
0-byte file called ``Other``. The data is there, but:

* File Explorer and ``os.listdir`` show only the empty ``Other`` file
* Excel cannot open it
* **copying, zipping or uploading the folder silently discards it**

Measured on GSE157827: nine clusters were labelled ``Other: <raw CellTypist code>``,
so seven pseudobulk DE tables (~30 MB, every neuronal population in the study) were
written into streams on one 0-byte file and were invisible until specifically
hunted for with ``Get-Item -Stream *``. Nothing in the run logged a problem — the
log lines said the files had been written, and technically they had.

This is silent data loss, so sanitizing happens in ONE place with tests, rather
than being re-derived ad hoc at each call site.
"""

from __future__ import annotations

import re as _re

# Reserved on Windows: < > : " / \ | ? *  — plus control characters. Kept as a
# literal so the reason for each is greppable.
_RESERVED = '<>:"/\\|?*'

# Names Windows refuses outright, in any case, with or without an extension.
_RESERVED_STEMS = frozenset(
    ["CON", "PRN", "AUX", "NUL"]
    + [f"COM{i}" for i in range(1, 10)]
    + [f"LPT{i}" for i in range(1, 10)]
)

MAX_COMPONENT_CHARS: int = 120  # leaves room for suffixes under the 255 limit


def safe_filename(label: str, *, fallback: str = "unnamed") -> str:
    """One path COMPONENT (no directory separators) built from a free-text label.

    Replaces every reserved character with ``_``, collapses whitespace runs,
    strips trailing dots and spaces (Windows silently drops those, so ``"A."`` and
    ``"A"`` would collide), avoids the reserved device stems, and truncates.
    Returns ``fallback`` if nothing usable survives.

    Deliberately NOT reversible: this makes a safe name, it does not encode one.
    Two labels differing only in punctuation can collide, which is why the
    human-readable label always stays in a COLUMN of the file, never only in its
    name.
    """
    s = str(label) if label is not None else ""
    s = "".join("_" if (ch in _RESERVED or ord(ch) < 32) else ch for ch in s)
    s = "_".join(s.split())  # collapse whitespace runs to single "_"
    # Collapse runs of "_" too, otherwise "Other: GABAergic" (colon -> "_", then
    # space -> "_") yields "Other__GABAergic".
    s = _re.sub(r"_+", "_", s)
    s = s.strip("_. ")  # Windows drops trailing dots/spaces, and a
    # leading/trailing "_" is just noise
    if len(s) > MAX_COMPONENT_CHARS:
        s = s[:MAX_COMPONENT_CHARS].rstrip("_. ")
    if not s:  # e.g. ":::" reduced to nothing usable
        return fallback
    if s.split(".")[0].upper() in _RESERVED_STEMS:
        s = f"_{s}"
    return s
