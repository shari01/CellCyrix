"""
Cell STATE vocabulary — deliberately orthogonal to cell IDENTITY.

Why this is a separate file: 'proliferating', 'malignant', 'exhausted',
'interferon-stimulated' and 'doublet' are not positions in a differentiation
hierarchy. A proliferating macrophage is a macrophage; a malignant epithelial
cell is an epithelial cell. Folding states into the tree corrupts every rollup
and every lowest-common-ancestor computation downstream.

So the resolver strips a recognised state prefix/suffix, resolves the residual
identity against the tree, and returns the state separately. Both axes travel
together in the output; neither contaminates the other.

`axis` groups states that are mutually exclusive with each other but
independent across groups (a cell can be cycling AND stressed).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# state_id -> (axis, display, surface forms found in labels)
STATES: Dict[str, Tuple[str, str, List[str]]] = {
    "cycling": (
        "cell_cycle",
        "Cycling / proliferating",
        [
            "cycling",
            "proliferating",
            "proliferative",
            "dividing",
            "mki67+",
            "mki67 positive",
            "in cell cycle",
            "cell cycle",
            "mitotic",
        ],
    ),
    "quiescent": ("cell_cycle", "Quiescent", ["quiescent", "resting", "non-cycling"]),
    "activated": (
        "activation",
        "Activated",
        ["activated", "active", "stimulated", "effector"],
    ),
    "naive_state": ("activation", "Naive", ["naive", "naïve"]),
    "memory_state": ("activation", "Memory", ["memory"]),
    "exhausted": (
        "activation",
        "Exhausted / dysfunctional",
        [
            "exhausted",
            "dysfunctional",
            "terminally exhausted",
            "tex",
        ],
    ),
    "senescent": ("activation", "Senescent", ["senescent", "senescence"]),
    "anergic": ("activation", "Anergic", ["anergic"]),
    "interferon_stimulated": (
        "signalling",
        "Interferon-stimulated",
        [
            "interferon-stimulated",
            "interferon stimulated",
            "isg-high",
            "isg high",
            "isg+",
            "ifn-stimulated",
            "ifn stimulated",
            "type i ifn",
        ],
    ),
    "inflammatory_state": (
        "signalling",
        "Inflammatory",
        ["inflammatory", "inflamed", "pro-inflammatory"],
    ),
    "hypoxic": ("signalling", "Hypoxic", ["hypoxic", "hypoxia"]),
    "stressed": (
        "stress",
        "Stressed",
        ["stressed", "stress response", "heat shock", "hsp-high"],
    ),
    "apoptotic": ("stress", "Apoptotic / dying", ["apoptotic", "dying", "necrotic"]),
    "malignant": (
        "transformation",
        "Malignant / aneuploid",
        [
            "malignant",
            "tumour",
            "tumor",
            "cancer",
            "cancerous",
            "neoplastic",
            "aneuploid",
            "transformed",
            "carcinoma",
        ],
    ),
    "premalignant": (
        "transformation",
        "Pre-malignant / dysplastic",
        [
            "premalignant",
            "pre-malignant",
            "dysplastic",
            "dysplasia",
            "atypical epithelial",
        ],
    ),
    "emt": (
        "transformation",
        "Mesenchymal transition",
        [
            "emt",
            "mesenchymal-like",
            "partial emt",
            "hybrid emt",
        ],
    ),
    "tissue_resident": (
        "localisation",
        "Tissue-resident",
        [
            "tissue-resident",
            "tissue resident",
            "resident",
        ],
    ),
    "circulating": (
        "localisation",
        "Circulating",
        ["circulating", "peripheral blood", "blood-derived"],
    ),
    "infiltrating": (
        "localisation",
        "Infiltrating",
        ["infiltrating", "tumour-infiltrating", "tumor-infiltrating"],
    ),
    "doublet": (
        "technical",
        "Doublet / multiplet",
        ["doublet", "doublets", "multiplet", "hybrid barcode"],
    ),
    "low_quality": (
        "technical",
        "Low quality",
        [
            "low quality",
            "low-quality",
            "poor quality",
            "high mito",
            "high-mito",
            "high mitochondrial",
            "debris",
            "ambient",
            "empty droplet",
        ],
    ),
}

# Words that look like state qualifiers but are load-bearing parts of an
# identity label. Never strip these.
PROTECTED_TOKENS = frozenset(
    {
        "memory",  # 'memory B cell', 'central memory T cell' are identities here
        "naive",  # 'naive CD4 T cell' is an identity node
        "naïve",
        "effector",  # 'effector memory T cell' is an identity node
        "activated",  # 'SM activated stress response' (Azimuth lung) is an identity
        "resident",  # 'tissue-resident memory T cell' is an identity node
        "tissue-resident",
        "tissue",
        "inflammatory",  # 'inflammatory fibroblast', 'inflammatory CAF' are identities
    }
)


def state_lookup() -> Dict[str, str]:
    """Surface form -> state_id. Longest forms first is the caller's job."""
    out: Dict[str, str] = {}
    for state_id, (_axis, _display, forms) in STATES.items():
        for form in forms:
            out[form.lower()] = state_id
    return out
