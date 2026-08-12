"""End-to-end demo: hierarchy-aware consensus vs naive string comparison."""

from cell_hierarchy import CellHierarchy

h = CellHierarchy.from_spec()

CASES = [
    (
        "granularity mismatch",
        {
            "celltypist_immune": "CD16+ NK cells",
            "singler_blueprint": "NK cells",
            "azimuth_pbmc": "NK",
        },
    ),
    (
        "real conflict inside a lineage",
        {
            "celltypist_immune": "CD8+ T cells",
            "singler_blueprint": "NK cells",
            "azimuth_pbmc": "CD8 TEM",
        },
    ),
    (
        "disjoint lineages",
        {
            "celltypist_lung": "AT2",
            "singler_blueprint": "CD8+ T-cells",
        },
    ),
    (
        "abstention",
        {
            "celltypist_immune": "Regulatory T cells",
            "singler_blueprint": "Tregs",
            "llm_verifier": "some label nobody has ever used",
        },
    ),
    (
        "state carried separately",
        {
            "celltypist_immune": "Cycling T cells",
            "singler_blueprint": "CD4+ T-cells",
        },
    ),
    (
        "epithelial, tissue-specific",
        {
            "celltypist_intestine": "BEST4+ epithelial",
            "singler_hpca": "Epithelial_cells",
            "llm_verifier": "Colonocyte",
        },
    ),
]

for title, votes in CASES:
    naive = "AGREE" if len(set(votes.values())) == 1 else "DISAGREE"
    c = h.consensus(votes)
    print(f"\n{title}")
    print(f"  votes            : {votes}")
    print(f"  naive string cmp : {naive}")
    print(
        f"  consensus        : {c.consensus_label or '(none)'} [{c.consensus_level_name or '-'}]"
    )
    print(
        f"  score            : {c.agreement_score:.3f}   resolved {c.n_resolved}/{c.n_voters}"
    )
    if c.dissenting_sources:
        print(f"  dissenting       : {', '.join(c.dissenting_sources)}")
    if c.states:
        print(f"  states           : {', '.join(c.states)}")
    print(f"  note             : {c.note}")
