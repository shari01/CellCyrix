"""
Cross-vocabulary alias crosswalk: raw voter label -> canonical node_id.

SOURCES are vocabulary namespaces, not tools. A single tool can emit several
(CellTypist emits a different label set per .pkl model), so the namespace is the
label set, which is what actually needs mapping.

PROVENANCE WARNING
------------------
These alias strings are transcribed from knowledge of the published reference
label sets, NOT parsed from the installed model artefacts on your machine.
Expect near-misses in punctuation, hyphenation and pluralisation. Two mitigations
ship with this package:

  1. resolver.normalize() collapses the usual variation (case, punctuation,
     '+'/'-' suffixes, plurals, unicode dashes) before lookup.
  2. audit_vocabulary.py ingests the *real* label sets from your installed
     CellTypist .pkl models and celldex references, reports every label that
     fails to resolve, and emits paste-ready alias stubs.

Run (2) before trusting this file in production.
"""

from __future__ import annotations

from typing import Dict, List

SOURCES = (
    "celltypist_immune",
    "celltypist_lung",
    "celltypist_intestine",
    "celltypist_skin",
    "singler_blueprint",
    "singler_hpca",
    "singler_monaco",
    "singler_dice",
    "azimuth_pbmc",
    "azimuth_lung",
    "azimuth_kidney",
    "panglaodb",
    "sctype",
    "generic",
)

# ---------------------------------------------------------------------------
# celltypist_skin — the 34 labels emitted by Adult_Human_Skin.pkl, which the
# pipeline selects automatically for skin datasets. Added after a real run left
# 22 of them unresolved: unresolved labels abstain from the hierarchy consensus,
# so a cluster three voters agreed on was scored as a disagreement.
#
# Numbered subsets (F1/F2/F3, VE1-3, Macro_1/2, Pericyte_1/2, Schwann_1/2) are
# study-specific partitions of one cell type; they map to the parent node rather
# than inventing subtype nodes the reference does not define. ILC1_NK is genuinely
# ambiguous between ILC1 and NK, so it maps to the shared parent.
# ---------------------------------------------------------------------------
_CELLTYPIST_SKIN: Dict[str, List[str]] = {
    "cd4_t_cell": ["Th"],
    "cd8_t_cell": ["Tc"],
    "regulatory_t_cell": ["Treg"],
    "nk_cell": ["NK"],
    "innate_lymphoid_cell": ["ILC1_3", "ILC1_NK"],
    "ilc2": ["ILC2"],
    "plasma_cell": ["Plasma"],
    "macrophage": ["Macro_1", "Macro_2", "Inf_mac"],
    "monocyte_derived_macrophage": ["Mono_mac"],
    "monocyte_derived_dc": ["moDC"],
    "conventional_dc1": ["DC1"],
    "conventional_dc2": ["DC2"],
    "dendritic_cell": ["MigDC"],
    "langerhans_cell": ["LC", "migLC"],
    "mast_cell": ["Mast_cell"],
    "keratinocyte": ["Differentiated_KC"],
    "basal_keratinocyte": ["Undifferentiated_KC"],
    "melanocyte": ["Melanocyte"],
    "fibroblast": ["F1", "F2", "F3"],
    "pericyte": ["Pericyte_1", "Pericyte_2"],
    "vascular_endothelial_cell": ["VE1", "VE2", "VE3"],
    "lymphatic_endothelial_cell": ["LE1", "LE2"],
    "schwann_cell": ["Schwann_1", "Schwann_2"],
}

# node_id -> {source: [raw labels]}
ALIASES: Dict[str, Dict[str, List[str]]] = {
    # ---------------------------------------------------------------- lymphoid
    "lymphoid_cell": {
        "singler_blueprint": ["Lymphocytes"],
        "generic": [
            "Lymphocytes",
            "Lymphoid",
            "T/NK cells",
            "T & NK cells",
            "TNK",
            "Lymphoid cells",
        ],
        "panglaodb": ["Lymphocytes"],
    },
    "t_cell": {
        "celltypist_immune": ["T cells"],
        "singler_blueprint": ["T-cells"],
        "singler_hpca": ["T_cells"],
        "singler_monaco": ["T cells"],
        "azimuth_pbmc": ["T"],
        "azimuth_lung": ["T cell lineage"],
        "panglaodb": ["T cells"],
        "sctype": ["T cells", "Naive CD8+ T cells"],
        "generic": ["T cell", "T-cells", "Tcells", "T lymphocyte", "T_cell", "TCells"],
    },
    "cd4_t_cell": {
        "celltypist_immune": ["CD4+ T cells", "Helper T cells"],
        "singler_blueprint": ["CD4+ T-cells"],
        "singler_monaco": ["T cells CD4"],
        "azimuth_pbmc": ["CD4 T"],
        "generic": [
            "CD4 T cells",
            "CD4+ T",
            "CD4 T",
            "CD4-positive T cells",
            "Th cells",
        ],
    },
    "cd4_naive_t_cell": {
        "celltypist_immune": ["Naive CD4+ T cells", "Tcm/Naive helper T cells"],
        "singler_blueprint": ["naive CD4+ T-cells"],
        "singler_monaco": ["T cells CD4 naive"],
        "azimuth_pbmc": ["CD4 Naive"],
        "singler_dice": ["T cells CD4 naive"],
        "generic": ["Naive CD4 T", "CD4 naive", "CD4+ naive T"],
    },
    "cd4_central_memory_t_cell": {
        "singler_blueprint": ["CD4+ Tcm"],
        "singler_monaco": ["T cells CD4 memory TCM"],
        "azimuth_pbmc": ["CD4 TCM"],
        "generic": ["CD4 Tcm", "CD4 central memory"],
    },
    "cd4_effector_memory_t_cell": {
        "celltypist_immune": [
            "Tem/Effector helper T cells",
            "Tem/Effector helper 1 T cells",
        ],
        "singler_blueprint": ["CD4+ Tem"],
        "singler_monaco": ["T cells CD4 memory TEM"],
        "azimuth_pbmc": ["CD4 TEM"],
        "generic": ["CD4 Tem", "CD4 effector memory", "Memory CD4 T cells"],
    },
    "th1_cell": {
        "celltypist_immune": ["Type 1 helper T cells", "Th1 cells", "Th1/Th17 cells"],
        "singler_monaco": ["T cells CD4 Th1", "T cells CD4 Th1/Th17"],
        "singler_dice": ["T cells CD4 Th1"],
        "generic": ["Th1", "TH1 cells"],
    },
    "th2_cell": {
        "celltypist_immune": ["Type 17 helper T cells", "Th2 cells"],
        "singler_monaco": ["T cells CD4 Th2"],
        "singler_dice": ["T cells CD4 Th2"],
        "generic": ["Th2", "TH2 cells"],
    },
    "th17_cell": {
        "celltypist_immune": ["Th17 cells"],
        "singler_monaco": ["T cells CD4 Th17"],
        "singler_dice": ["T cells CD4 Th17"],
        "generic": ["Th17", "TH17 cells", "IL17+ T cells"],
    },
    "tfh_cell": {
        "celltypist_immune": ["Follicular helper T cells", "Tfh"],
        "singler_monaco": ["T cells CD4 follicular helper"],
        "singler_dice": ["T cells CD4 Tfh"],
        "generic": ["Tfh", "Follicular helper T", "GC-Tfh"],
    },
    "regulatory_t_cell": {
        "celltypist_immune": ["Regulatory T cells", "Tregs"],
        "singler_blueprint": ["Tregs"],
        "singler_monaco": ["T cells CD4 Treg", "T regulatory cells"],
        "singler_dice": ["T cells CD4 Treg memory", "T cells CD4 Treg naive"],
        "azimuth_pbmc": ["Treg"],
        "panglaodb": ["T regulatory cells"],
        "generic": ["Treg", "Tregs", "Regulatory T", "FOXP3+ T cells"],
    },
    "cd8_t_cell": {
        "celltypist_immune": ["CD8+ T cells", "Cytotoxic T cells"],
        "singler_blueprint": ["CD8+ T-cells"],
        "singler_monaco": ["T cells CD8"],
        "azimuth_pbmc": ["CD8 T"],
        "panglaodb": ["Cytotoxic T cells"],
        "generic": ["CD8 T cells", "CD8+ T", "CD8 T", "Cytotoxic CD8 T cells", "CTL"],
    },
    "cd8_naive_t_cell": {
        "singler_blueprint": ["CD8+ naive T-cells", "naive CD8+ T-cells"],
        "singler_monaco": ["T cells CD8 naive"],
        "azimuth_pbmc": ["CD8 Naive"],
        "generic": ["Naive CD8 T", "CD8 naive"],
    },
    "cd8_central_memory_t_cell": {
        "celltypist_immune": ["Tcm/Naive cytotoxic T cells"],
        "singler_blueprint": ["CD8+ Tcm"],
        "singler_monaco": ["T cells CD8 memory TCM"],
        "azimuth_pbmc": ["CD8 TCM"],
        "generic": ["CD8 Tcm"],
    },
    "cd8_effector_memory_t_cell": {
        "celltypist_immune": [
            "Tem/Effector cytotoxic T cells",
            "Tem/Temra cytotoxic T cells",
        ],
        "singler_blueprint": ["CD8+ Tem"],
        "singler_monaco": ["T cells CD8 memory TEM"],
        "azimuth_pbmc": ["CD8 TEM"],
        "generic": ["CD8 Tem", "GZMK+ CD8 T cells", "Memory CD8 T cells"],
    },
    "cd8_temra_cell": {
        "singler_monaco": ["T cells CD8 memory TEMRA"],
        "generic": ["CD8 Temra", "TEMRA", "Effector CD8 T cells", "GZMB+ CD8 T cells"],
    },
    "cd8_tissue_resident_memory_t_cell": {
        "celltypist_immune": [
            "Tissue-resident memory T cells",
            "Trm cells",
            "Tem/Trm cytotoxic T cells",
            "Trm cytotoxic T cells",
        ],
        "generic": ["Trm", "CD8 Trm", "Tissue resident memory T"],
    },
    "cd8_exhausted_t_cell": {
        "generic": [
            "Exhausted T cells",
            "Exhausted CD8 T",
            "Tex",
            "Terminally exhausted T cells",
        ],
    },
    "mait_cell": {
        "celltypist_immune": ["MAIT cells"],
        "singler_monaco": ["T cells MAIT"],
        "azimuth_pbmc": ["MAIT"],
        "generic": ["MAIT", "Mucosal associated invariant T cells"],
    },
    "gamma_delta_t_cell": {
        "celltypist_immune": ["gamma-delta T cells", "Tgd cells"],
        "azimuth_pbmc": ["gdT"],
        "generic": ["gdT", "Gamma delta T cells", "γδ T cells", "TCRgd"],
    },
    "vd2_t_cell": {
        "singler_monaco": ["T cells gamma-delta Vd2"],
        "generic": ["Vd2 T cells", "Vδ2 T cells"],
    },
    "vd1_t_cell": {
        "singler_monaco": ["T cells gamma-delta non-Vd2"],
        "generic": ["Vd1 T cells", "Vδ1 T cells"],
    },
    "nkt_cell": {
        "celltypist_immune": ["NKT cells"],
        "panglaodb": ["Natural killer T cells"],
        "generic": ["NKT", "iNKT", "Natural killer T cells"],
    },
    "cd4_cytotoxic_t_cell": {
        "azimuth_pbmc": ["CD4 CTL"],
        "generic": ["CD4 CTL", "Cytotoxic CD4 T cells"],
    },
    "double_negative_t_cell": {
        "azimuth_pbmc": ["dnT"],
        "generic": ["dnT", "DN T cells", "Double-negative T cells"],
    },
    "monocyte_precursor": {
        "celltypist_immune": ["Monocyte precursor"],
        "generic": ["Monocyte precursors", "Promonocyte"],
    },
    "proliferating_t_cell": {
        "celltypist_immune": ["Cycling T cells", "T cells proliferative"],
        "azimuth_pbmc": ["CD4 Proliferating", "CD8 Proliferating"],
        "generic": ["Cycling T", "Proliferating T cells", "MKI67+ T cells"],
    },
    "double_positive_thymocyte": {
        "celltypist_immune": ["Double-positive thymocytes", "DP thymocytes"],
        "generic": ["DP T cells", "CD4+CD8+ thymocytes"],
    },
    "double_negative_thymocyte": {
        "celltypist_immune": ["Double-negative thymocytes"],
        "generic": ["DN thymocytes"],
    },
    # ------------------------------------------------------------------ NK/ILC
    "nk_cell": {
        "celltypist_immune": ["NK cells"],
        "singler_blueprint": ["NK cells"],
        "singler_hpca": ["NK_cell"],
        "singler_monaco": ["NK cells"],
        "azimuth_pbmc": ["NK"],
        "panglaodb": ["NK cells", "Natural killer cells"],
        "sctype": ["Natural killer  cells", "Natural killer cells"],
        "generic": ["NK", "NK cell", "Natural killer cells", "NK-cells"],
    },
    "cd56_dim_nk_cell": {
        "celltypist_immune": ["CD16+ NK cells"],
        "singler_monaco": ["NK cells CD56dim"],
        "generic": ["CD56dim NK", "CD16+ NK", "Cytotoxic NK cells"],
    },
    "cd56_bright_nk_cell": {
        "celltypist_immune": ["CD16- NK cells"],
        "singler_monaco": ["NK cells CD56bright"],
        "azimuth_pbmc": ["NK_CD56bright"],
        "generic": ["CD56bright NK", "CD16- NK", "Regulatory NK cells"],
    },
    "proliferating_nk_cell": {
        "azimuth_pbmc": ["NK Proliferating"],
        "generic": ["Cycling NK cells", "Proliferating NK"],
    },
    "innate_lymphoid_cell": {
        "celltypist_immune": ["ILC", "ILC precursor"],
        "azimuth_pbmc": ["ILC"],
        "generic": ["ILCs", "Innate lymphoid cells"],
    },
    "ilc1": {"celltypist_immune": ["ILC1"], "generic": ["ILC1 cells"]},
    "ilc2": {"celltypist_immune": ["ILC2"], "generic": ["ILC2 cells"]},
    "ilc3": {"celltypist_immune": ["ILC3"], "generic": ["ILC3 cells"]},
    # ------------------------------------------------------------------ B/plasma
    "b_cell": {
        "celltypist_immune": ["B cells"],
        "singler_blueprint": ["B-cells"],
        "singler_hpca": ["B_cell"],
        "singler_monaco": ["B cells"],
        "azimuth_pbmc": ["B"],
        "azimuth_lung": ["B cell lineage"],
        "panglaodb": ["B cells"],
        "sctype": ["B cells"],
        "generic": ["B cell", "B-cells", "Bcells", "B lymphocyte"],
    },
    "naive_b_cell": {
        "celltypist_immune": ["Naive B cells"],
        "singler_blueprint": ["naive B-cells"],
        "singler_monaco": ["B cells naive"],
        "azimuth_pbmc": ["B naive"],
        "generic": ["Naive B", "IgD+ B cells"],
    },
    "memory_b_cell": {
        "celltypist_immune": ["Memory B cells"],
        "singler_blueprint": ["Memory B-cells", "Class-switched memory B-cells"],
        "singler_monaco": ["B cells non-switched memory", "B cells switched memory"],
        "azimuth_pbmc": ["B memory"],
        "panglaodb": ["Memory B cells"],
        "generic": ["Memory B", "Class-switched B cells"],
    },
    "germinal_center_b_cell": {
        "celltypist_immune": ["Germinal center B cells", "GC B cells"],
        "generic": ["GC B cells", "Germinal centre B", "Follicular B cells"],
    },
    "transitional_b_cell": {
        "celltypist_immune": ["Transitional B cells"],
        "azimuth_pbmc": ["B intermediate"],
        "generic": ["Immature B cells", "Transitional B"],
    },
    "atypical_b_cell": {
        "celltypist_immune": ["Age-associated B cells"],
        "generic": ["Atypical B cells", "DN2 B cells", "ABCs", "CD11c+ B cells"],
    },
    "plasma_cell": {
        "celltypist_immune": ["Plasma cells"],
        "singler_blueprint": ["Plasma cells"],
        "azimuth_lung": ["Plasma cells"],
        "panglaodb": ["Plasma cells"],
        "sctype": ["Plasma B cells"],
        "generic": ["Plasma cell", "PC", "Antibody-secreting cells", "ASC"],
    },
    "plasmablast": {
        "celltypist_immune": ["Plasmablasts"],
        "azimuth_pbmc": ["Plasmablast"],
        "generic": ["Plasmablast", "Proliferating plasma cells"],
    },
    "iga_plasma_cell": {
        "celltypist_intestine": ["IgA plasma cell", "IgA plasma cells"],
        "generic": ["IgA+ plasma cells"],
    },
    "igg_plasma_cell": {
        "celltypist_intestine": ["IgG plasma cell", "IgG plasma cells"],
        "generic": ["IgG+ plasma cells"],
    },
    "pro_b_cell": {
        "singler_blueprint": ["Pro-B_cell_CD34+"],
        "generic": ["Pro-B cells"],
    },
    "pre_b_cell": {
        "singler_hpca": ["Pre-B_cell_CD34-"],
        "sctype": ["Pre-B cells"],
        "generic": ["Pre-B cells"],
    },
    # -------------------------------------------------------------- monocyte/mac
    "myeloid_cell": {
        "azimuth_lung": ["Myeloid"],
        "generic": ["Myeloid cells", "Myeloid", "Mononuclear phagocytes", "MNP"],
    },
    "monocyte": {
        "celltypist_immune": ["Monocytes"],
        "singler_blueprint": ["Monocytes"],
        "singler_hpca": ["Monocyte"],
        "singler_monaco": ["Monocytes"],
        "azimuth_pbmc": ["Mono"],
        "panglaodb": ["Monocytes"],
        "sctype": ["Monocytes"],
        "generic": ["Monocyte", "Monocytes", "Mono"],
    },
    "classical_monocyte": {
        "celltypist_immune": ["Classical monocytes"],
        "singler_monaco": ["Monocytes classical"],
        "azimuth_pbmc": ["CD14 Mono"],
        "generic": ["CD14+ monocytes", "CD14 Mono", "cMono", "Classical Mono"],
    },
    "non_classical_monocyte": {
        "celltypist_immune": ["Non-classical monocytes"],
        "singler_monaco": ["Monocytes non-classical"],
        "azimuth_pbmc": ["CD16 Mono"],
        "generic": ["CD16+ monocytes", "FCGR3A+ monocytes", "ncMono"],
    },
    "intermediate_monocyte": {
        "singler_monaco": ["Monocytes intermediate"],
        "generic": ["Intermediate monocytes", "CD14+CD16+ monocytes"],
    },
    "macrophage": {
        "celltypist_immune": ["Macrophages", "Erythrophagocytic macrophages"],
        "singler_blueprint": ["Macrophages", "Macrophages M1", "Macrophages M2"],
        "singler_hpca": ["Macrophage"],
        "azimuth_lung": ["Macrophages"],
        "panglaodb": ["Macrophages"],
        "sctype": ["Macrophages"],
        "generic": [
            "Macrophage",
            "Mac",
            "Mφ",
            "Tissue macrophages",
            "TAM",
            "M1 macrophages",
            "M2 macrophages",
        ],
    },
    "alveolar_macrophage": {
        "celltypist_lung": ["Alveolar macrophages", "AM"],
        "azimuth_lung": ["Alveolar Macrophages"],
        "generic": ["Alveolar macrophage", "FABP4+ macrophages"],
    },
    "interstitial_macrophage": {
        "celltypist_lung": ["Interstitial macrophages"],
        "azimuth_lung": ["Interstitial Macrophages"],
        "generic": ["IM", "Interstitial macrophage"],
    },
    "monocyte_derived_macrophage": {
        "celltypist_lung": ["Monocyte-derived macrophages"],
        "generic": ["MoMac", "Monocyte derived macrophages", "SPP1+ macrophages"],
    },
    "kupffer_cell": {"panglaodb": ["Kupffer cells"], "generic": ["Kupffer cell"]},
    "microglial_cell": {
        "panglaodb": ["Microglia"],
        "sctype": ["Microglial cells"],
        "generic": ["Microglia", "Microglial cells"],
    },
    "osteoclast": {"panglaodb": ["Osteoclasts"], "generic": ["Osteoclast"]},
    "langerhans_cell": {
        "celltypist_immune": ["Langerhans cells"],
        "panglaodb": ["Langerhans cells"],
        "generic": ["Langerhans cell", "LC"],
    },
    "lipid_associated_macrophage": {
        "generic": ["LAM", "Lipid-associated macrophages", "TREM2+ macrophages"],
    },
    "proliferating_macrophage": {
        "celltypist_immune": ["Cycling monocytes"],
        "generic": ["Proliferating macrophages", "Cycling myeloid cells"],
    },
    # ------------------------------------------------------------------------ DC
    "dendritic_cell": {
        "celltypist_immune": ["DC", "Dendritic cells"],
        "singler_blueprint": ["DC"],
        "singler_hpca": ["DC"],
        "singler_monaco": ["Dendritic cells"],
        "azimuth_pbmc": ["DC"],
        "azimuth_lung": ["Dendritic cells"],
        "panglaodb": ["Dendritic cells"],
        "sctype": ["Myeloid Dendritic cells"],
        "generic": ["DC", "Dendritic cell", "cDC"],
    },
    "conventional_dc1": {
        "celltypist_immune": ["DC1"],
        "singler_monaco": ["Myeloid dendritic cells"],
        "azimuth_pbmc": ["cDC1"],
        "generic": ["cDC1", "CLEC9A+ DC", "XCR1+ DC"],
    },
    "conventional_dc2": {
        "celltypist_immune": ["DC2"],
        "azimuth_pbmc": ["cDC2"],
        "generic": ["cDC2", "CD1C+ DC", "CD1c+ dendritic cells"],
    },
    "dc3": {"celltypist_immune": ["DC3"], "generic": ["DC3", "Inflammatory DC"]},
    "plasmacytoid_dc": {
        "celltypist_immune": ["pDC", "Plasmacytoid DC"],
        "singler_monaco": ["Plasmacytoid dendritic cells"],
        "azimuth_pbmc": ["pDC"],
        "panglaodb": ["Plasmacytoid dendritic cells"],
        "sctype": ["Plasmacytoid Dendritic cells"],
        "generic": ["pDC", "pDCs", "Plasmacytoid dendritic cell"],
    },
    "as_dc": {
        "azimuth_pbmc": ["ASDC"],
        "generic": ["ASDC", "AS-DC", "AXL+ DC", "transitional DC"],
    },
    "migratory_dc": {
        "celltypist_immune": ["Migratory DCs"],
        "generic": ["LAMP3+ DC", "mregDC", "Mature DC", "CCR7+ DC"],
    },
    # ---------------------------------------------------------------- granulocyte
    "neutrophil": {
        "celltypist_immune": ["Neutrophils"],
        "singler_blueprint": ["Neutrophils"],
        "singler_hpca": ["Neutrophil"],
        "singler_monaco": ["Neutrophils"],
        "panglaodb": ["Neutrophils"],
        "sctype": ["Neutrophils"],
        "generic": ["Neutrophil", "PMN", "Granulocytes"],
    },
    "eosinophil": {
        "celltypist_immune": ["Eosinophils"],
        "singler_blueprint": ["Eosinophils"],
        "singler_hpca": ["Eosinophil"],
        "panglaodb": ["Eosinophils"],
        "generic": ["Eosinophil"],
    },
    "basophil": {
        "celltypist_immune": ["Basophils"],
        "singler_hpca": ["Basophil"],
        "singler_monaco": ["Basophils"],
        "panglaodb": ["Basophils"],
        "generic": ["Basophil"],
    },
    "mast_cell": {
        "celltypist_immune": ["Mast cells"],
        "azimuth_lung": ["Mast cells"],
        "panglaodb": ["Mast cells"],
        "sctype": ["Mast cells"],
        "generic": ["Mast cell", "Mastocyte", "MC"],
    },
    # -------------------------------------------------- erythroid / megakaryocyte
    "erythroid_cell": {
        "celltypist_immune": ["Erythroid", "Erythrocytes"],
        "singler_blueprint": ["Erythrocytes"],
        "singler_hpca": ["Erythroblast"],
        "azimuth_pbmc": ["Eryth"],
        "panglaodb": ["Erythroid-like and erythroid precursor cells"],
        "generic": ["Erythroid cells", "RBC", "Red blood cells", "Erythrocytes"],
    },
    "proerythroblast": {"generic": ["Proerythroblast", "Early erythroid"]},
    "erythrocyte": {"generic": ["Erythrocyte", "Mature RBC"]},
    "megakaryocytic_cell": {
        "celltypist_immune": ["Megakaryocytes/platelets", "Megakaryocyte precursor"],
        "panglaodb": ["Megakaryocytes"],
        "generic": ["Megakaryocyte/platelet", "MK"],
    },
    "megakaryocyte": {
        "singler_hpca": ["MEP"],
        "generic": ["Megakaryocyte", "Megakaryocytes"],
    },
    "platelet": {
        "azimuth_pbmc": ["Platelet"],
        "panglaodb": ["Platelets"],
        "generic": ["Platelets", "Thrombocytes", "PLT"],
    },
    "hematopoietic_stem_progenitor_cell": {
        "celltypist_immune": ["HSC/MPP", "Early MK"],
        "singler_blueprint": ["Multipotent progenitors"],
        "singler_hpca": ["HSC_CD34+", "HSC_-G-CSF"],
        "azimuth_pbmc": ["HSPC"],
        "generic": [
            "HSPC",
            "CD34+ cells",
            "Progenitor cells",
            "Haematopoietic progenitors",
        ],
    },
    "hematopoietic_stem_cell": {
        "singler_blueprint": ["HSC"],
        "generic": ["HSC", "Haematopoietic stem cells"],
    },
    "common_lymphoid_progenitor": {"singler_blueprint": ["CLP"], "generic": ["CLP"]},
    "common_myeloid_progenitor": {"singler_blueprint": ["CMP"], "generic": ["CMP"]},
    "granulocyte_monocyte_progenitor": {
        "singler_blueprint": ["GMP"],
        "generic": ["GMP"],
    },
    "megakaryocyte_erythroid_progenitor": {
        "singler_blueprint": ["MEP"],
        "generic": ["MEP"],
    },
    # ------------------------------------------------------------------ airway
    "epithelial": {
        "singler_hpca": ["Epithelial_cells"],
        "azimuth_lung": ["Epithelial"],
        "panglaodb": ["Epithelial cells"],
        "sctype": ["Epithelial cells"],
        "generic": ["Epithelial cells", "Epithelium", "EPCAM+ cells"],
    },
    "basal_cell_airway": {
        "celltypist_lung": ["Basal cells", "Basal"],
        "azimuth_lung": ["Basal"],
        "panglaodb": ["Airway basal cells"],
        "generic": ["Basal cells", "KRT5+ basal cells"],
    },
    "club_cell": {
        "celltypist_lung": ["Club cells", "Secretory cells"],
        "azimuth_lung": ["Club"],
        "panglaodb": ["Clara cells", "Club cells"],
        "generic": ["Club cells", "Clara cells", "SCGB1A1+ cells", "Secretory cells"],
    },
    "goblet_cell_airway": {
        "celltypist_lung": ["Goblet cells"],
        "azimuth_lung": ["Goblet"],
        "panglaodb": ["Airway goblet cells"],
        "generic": ["Goblet cells", "MUC5AC+ cells"],
    },
    "ciliated_cell_airway": {
        "celltypist_lung": ["Ciliated cells", "Multiciliated cells"],
        "azimuth_lung": ["Multiciliated"],
        "panglaodb": ["Ciliated cells"],
        "sctype": ["Airway epithelial cells"],
        "generic": ["Ciliated cells", "FOXJ1+ cells", "Multiciliated cells"],
    },
    "pulmonary_ionocyte": {
        "celltypist_lung": ["Ionocytes"],
        "azimuth_lung": ["Ionocyte"],
        "generic": ["Ionocytes", "FOXI1+ cells", "Pulmonary ionocyte"],
    },
    "tuft_cell_airway": {
        "celltypist_lung": ["Tuft cells"],
        "azimuth_lung": ["Tuft"],
        "generic": ["Brush cells", "Tuft cells"],
    },
    "pulmonary_neuroendocrine_cell": {
        "celltypist_lung": ["Neuroendocrine cells"],
        "azimuth_lung": ["Neuroendocrine"],
        "generic": ["PNEC", "Neuroendocrine cells", "Pulmonary neuroendocrine cells"],
    },
    "alveolar_type_1_cell": {
        "celltypist_lung": ["AT1", "Alveolar type 1 cells"],
        "azimuth_lung": ["AT1"],
        "panglaodb": ["Pulmonary alveolar type I cells"],
        "generic": ["AT1", "ATI", "Type 1 pneumocytes", "AGER+ cells"],
    },
    "alveolar_type_2_cell": {
        "celltypist_lung": ["AT2", "Alveolar type 2 cells"],
        "azimuth_lung": ["AT2"],
        "panglaodb": ["Pulmonary alveolar type II cells"],
        "generic": ["AT2", "ATII", "Type 2 pneumocytes", "SFTPC+ cells"],
    },
    "transitional_alveolar_epithelial_cell": {
        "azimuth_lung": ["AT1/AT2"],
        "generic": [
            "KRT8+ ADI",
            "Transitional AT2",
            "Damage-associated transient progenitors",
            "DATP",
        ],
    },
    "aberrant_basaloid_cell": {
        "generic": ["Aberrant basaloid cells", "KRT17+ KRT5- cells", "Basaloid cells"],
    },
    # ---------------------------------------------------------------- intestinal
    "intestinal_stem_cell": {
        "celltypist_intestine": ["Stem cells", "Crypt stem cells"],
        "generic": ["ISC", "LGR5+ stem cells", "Crypt stem cells"],
    },
    "transit_amplifying_cell_intestine": {
        "celltypist_intestine": ["TA", "Transit amplifying cells"],
        "generic": ["TA cells", "Transit-amplifying cells", "Cycling TA"],
    },
    "enterocyte": {
        "celltypist_intestine": ["Enterocytes", "Mature enterocytes"],
        "panglaodb": ["Enterocytes"],
        "generic": ["Enterocyte", "Absorptive cells"],
    },
    "colonocyte": {
        "celltypist_intestine": ["Colonocytes"],
        "generic": ["Colonocyte", "Absorptive colonocytes"],
    },
    "best4_epithelial_cell": {
        "celltypist_intestine": ["BEST4+ epithelial", "BEST4 enterocytes"],
        "generic": ["BEST4+ cells", "BEST4/OTOP2 cells", "CA7+ cells"],
    },
    "goblet_cell_intestine": {
        "celltypist_intestine": ["Goblet cells"],
        "panglaodb": ["Goblet cells"],
        "generic": ["Intestinal goblet cells", "MUC2+ cells"],
    },
    "paneth_cell": {
        "celltypist_intestine": ["Paneth cells"],
        "panglaodb": ["Paneth cells"],
        "generic": ["Paneth cell", "DEFA5+ cells"],
    },
    "enteroendocrine_cell": {
        "celltypist_intestine": ["Enteroendocrine cells", "EEC"],
        "panglaodb": ["Enteroendocrine cells"],
        "generic": ["EEC", "Enteroendocrine", "CHGA+ cells"],
    },
    "tuft_cell_intestine": {
        "celltypist_intestine": ["Tuft cells"],
        "generic": ["Intestinal tuft cells", "POU2F3+ cells"],
    },
    "microfold_cell": {
        "celltypist_intestine": ["M cells"],
        "generic": ["M cell", "Microfold cells"],
    },
    # ------------------------------------------------------------------ squamous
    "squamous_epithelial_cell": {
        "panglaodb": ["Squamous epithelial cells"],
        "generic": ["Squamous cells", "Squamous epithelium"],
    },
    "squamous_basal_cell": {
        "generic": ["Basal squamous cells", "Basal cells (squamous)", "Basal/parabasal"]
    },
    "squamous_parabasal_cell": {"generic": ["Parabasal cells", "Parabasal"]},
    "squamous_intermediate_cell": {
        "generic": ["Intermediate squamous cells", "Intermediate cells"]
    },
    "squamous_superficial_cell": {
        "generic": ["Superficial squamous cells", "Superficial cells"]
    },
    "reserve_cell_cervix": {"generic": ["Reserve cells", "Cervical reserve cells"]},
    "squamocolumnar_junction_cell": {
        "generic": [
            "SCJ cells",
            "Squamocolumnar junction cells",
            "KRT7+ junction cells",
        ]
    },
    "metaplastic_squamous_cell": {
        "generic": ["Squamous metaplasia", "Metaplastic cells"]
    },
    "endocervical_columnar_cell": {
        "generic": ["Endocervical cells", "Columnar cells", "Glandular cervical cells"]
    },
    "keratinocyte": {
        "singler_hpca": ["Keratinocytes"],
        "panglaodb": ["Keratinocytes"],
        "generic": ["Keratinocyte", "Keratinocytes"],
    },
    "basal_keratinocyte": {"generic": ["Basal keratinocytes", "KRT14+ keratinocytes"]},
    "melanocyte": {
        "singler_hpca": ["Melanocytes"],
        "panglaodb": ["Melanocytes"],
        "generic": ["Melanocyte", "Melanocytes"],
    },
    # ---------------------------------------------------------------- glandular
    "luminal_epithelial_cell": {
        "generic": ["Luminal cells", "Luminal epithelium", "Luminal 1", "Luminal 2"]
    },
    "myoepithelial_cell": {
        "panglaodb": ["Myoepithelial cells"],
        "generic": ["Myoepithelial cells", "Basal/myoepithelial"],
    },
    "ductal_epithelial_cell": {
        "panglaodb": ["Ductal cells"],
        "generic": ["Ductal cells", "Duct cells", "KRT19+ ductal cells"],
    },
    "acinar_cell": {
        "panglaodb": ["Acinar cells"],
        "generic": ["Acinar cells", "Pancreatic acinar cells"],
    },
    "mesothelial_cell": {
        "panglaodb": ["Mesothelial cells"],
        "azimuth_lung": ["Mesothelial"],
        "generic": ["Mesothelium", "Mesothelial cells", "MSLN+ cells"],
    },
    "thyroid_follicular_cell": {
        "panglaodb": ["Thyrocytes"],
        "generic": ["Thyrocytes", "Follicular cells"],
    },
    # -------------------------------------------------------------------- renal
    "proximal_tubule_cell": {
        "azimuth_kidney": ["PT", "Proximal Tubule"],
        "panglaodb": ["Proximal tubule cells"],
        "generic": ["PT cells", "Proximal tubule"],
    },
    "loop_of_henle_cell": {
        "azimuth_kidney": ["TAL", "Thick Ascending Limb"],
        "generic": ["TAL", "Loop of Henle", "Thick ascending limb"],
    },
    "distal_convoluted_tubule_cell": {
        "azimuth_kidney": ["DCT"],
        "generic": ["DCT", "Distal convoluted tubule"],
    },
    "principal_cell_kidney": {
        "azimuth_kidney": ["PC", "Principal Cell"],
        "generic": ["Principal cells", "Collecting duct PC"],
    },
    "intercalated_cell_alpha": {
        "azimuth_kidney": ["IC-A"],
        "generic": ["Type A intercalated cells", "IC-A"],
    },
    "intercalated_cell_beta": {
        "azimuth_kidney": ["IC-B"],
        "generic": ["Type B intercalated cells", "IC-B"],
    },
    "podocyte": {
        "azimuth_kidney": ["POD", "Podocyte"],
        "panglaodb": ["Podocytes"],
        "generic": ["Podocytes", "NPHS2+ cells"],
    },
    "urothelial_cell": {
        "panglaodb": ["Urothelial cells"],
        "generic": ["Urothelium", "Urothelial cells", "Transitional epithelium"],
    },
    # ---------------------------------------------------------- hepatobiliary
    "hepatocyte": {
        "singler_hpca": ["Hepatocytes"],
        "panglaodb": ["Hepatocytes"],
        "generic": ["Hepatocyte", "Hepatocytes", "ALB+ cells"],
    },
    "cholangiocyte": {
        "panglaodb": ["Cholangiocytes"],
        "generic": ["Cholangiocytes", "Biliary epithelial cells", "BEC"],
    },
    "beta_cell": {
        "panglaodb": ["Beta cells"],
        "generic": ["Beta cells", "INS+ cells", "Insulin-producing cells"],
    },
    "alpha_cell": {
        "panglaodb": ["Alpha cells"],
        "generic": ["Alpha cells", "GCG+ cells"],
    },
    "delta_cell": {
        "panglaodb": ["Delta cells"],
        "generic": ["Delta cells", "SST+ cells"],
    },
    "pp_cell": {
        "panglaodb": ["Gamma (PP) cells"],
        "generic": ["PP cells", "Gamma cells"],
    },
    # ----------------------------------------------------------------- stromal
    "stromal_mesenchymal": {
        "azimuth_lung": ["Stroma"],
        "generic": ["Stromal cells", "Mesenchymal cells", "Stroma"],
    },
    "fibroblast": {
        "singler_blueprint": ["Fibroblasts"],
        "singler_hpca": ["Fibroblasts"],
        "azimuth_lung": ["Fibroblasts"],
        "panglaodb": ["Fibroblasts"],
        "sctype": ["Fibroblasts"],
        "generic": ["Fibroblast", "Fibroblasts", "COL1A1+ cells"],
    },
    "myofibroblast": {
        "panglaodb": ["Myofibroblasts"],
        "azimuth_lung": ["Myofibroblasts"],
        "generic": ["Myofibroblast", "ACTA2+ fibroblasts", "Activated fibroblasts"],
    },
    "inflammatory_fibroblast": {
        "generic": ["Inflammatory fibroblasts", "IL11+ fibroblasts", "IAF"]
    },
    "cancer_associated_fibroblast": {
        "generic": ["CAF", "CAFs", "Cancer-associated fibroblasts", "FAP+ fibroblasts"]
    },
    "myofibroblastic_caf": {"generic": ["myCAF", "Myofibroblastic CAF"]},
    "inflammatory_caf": {"generic": ["iCAF", "Inflammatory CAF"]},
    "antigen_presenting_caf": {"generic": ["apCAF", "Antigen-presenting CAF"]},
    "pericyte": {
        "azimuth_lung": ["Pericytes"],
        "panglaodb": ["Pericytes"],
        "generic": ["Pericyte", "Pericytes", "RGS5+ cells", "Mural cells"],
    },
    "vascular_smooth_muscle_cell": {
        "azimuth_lung": ["SMC"],
        "panglaodb": ["Smooth muscle cells"],
        "sctype": ["Smooth muscle cells"],
        "generic": ["VSMC", "Smooth muscle cells", "SMC", "vSMC"],
    },
    "hepatic_stellate_cell": {
        "panglaodb": ["Hepatic stellate cells"],
        "generic": ["HSC (hepatic stellate)", "Stellate cells"],
    },
    "preadipocyte": {
        "singler_blueprint": ["Preadipocytes"],
        "generic": ["Preadipocytes", "Pre-adipocytes"],
    },
    "mesangial_cell": {
        "singler_blueprint": ["Mesangial cells"],
        "generic": ["Mesangial cells"],
    },
    "adipocyte": {
        "singler_hpca": ["Adipocytes"],
        "panglaodb": ["Adipocytes"],
        "generic": ["Adipocyte", "Adipocytes"],
    },
    "chondrocyte": {
        "singler_hpca": ["Chondrocytes"],
        "panglaodb": ["Chondrocytes"],
        "generic": ["Chondrocyte", "Chondrocytes"],
    },
    "osteoblast": {
        "singler_hpca": ["Osteoblasts"],
        "panglaodb": ["Osteoblasts"],
        "generic": ["Osteoblast", "Osteoblasts"],
    },
    "mesenchymal_stromal_cell": {
        "singler_hpca": ["MSC"],
        "generic": ["MSC", "Mesenchymal stem cells", "Mesenchymal stromal cells"],
    },
    "fibroblastic_reticular_cell": {"generic": ["FRC", "Fibroblastic reticular cells"]},
    "follicular_dendritic_cell": {"generic": ["FDC", "Follicular dendritic cells"]},
    "intestinal_fibroblast_s1": {
        "generic": ["S1 fibroblasts", "ADAMDEC1+ fibroblasts"]
    },
    "intestinal_fibroblast_s2": {"generic": ["S2 fibroblasts", "NPY+ fibroblasts"]},
    "intestinal_fibroblast_s3": {"generic": ["S3 fibroblasts"]},
    "wnt2b_fibroblast": {"generic": ["WNT2B+ fibroblasts", "WNT2B+ Fos-lo"]},
    "telocyte": {"generic": ["Telocytes", "FOXL1+ telocytes"]},
    # -------------------------------------------------------------- endothelial
    "endothelial": {
        "singler_blueprint": ["Endothelial cells"],
        "singler_hpca": ["Endothelial_cells"],
        "azimuth_lung": ["Endothelial"],
        "panglaodb": ["Endothelial cells"],
        "sctype": ["Endothelial cells"],
        "generic": ["Endothelial cells", "EC", "ECs", "PECAM1+ cells"],
    },
    "arterial_endothelial_cell": {
        "azimuth_lung": ["EC arterial"],
        "generic": ["Arterial EC", "Arterial endothelium"],
    },
    "capillary_endothelial_cell": {
        "singler_blueprint": ["mv Endothelial cells"],
        "generic": ["Capillary EC", "cEC", "Microvascular endothelial cells"],
    },
    "general_capillary_endothelial_cell": {
        "azimuth_lung": ["EC general capillary"],
        "generic": ["gCap", "General capillary"],
    },
    "aerocyte": {
        "azimuth_lung": ["EC aerocyte capillary"],
        "generic": ["aCap", "Aerocytes", "Aerocyte"],
    },
    "venous_endothelial_cell": {
        "azimuth_lung": ["EC venous systemic", "EC venous pulmonary"],
        "generic": ["Venous EC", "ACKR1+ EC"],
    },
    "lymphatic_endothelial_cell": {
        "azimuth_lung": ["Lymphatic EC mature"],
        "panglaodb": ["Lymphatic endothelial cells"],
        "generic": ["LEC", "Lymphatic EC", "Lymphatics"],
    },
    "high_endothelial_venule_cell": {"generic": ["HEV", "High endothelial venules"]},
    "tip_endothelial_cell": {"generic": ["Tip cells", "Angiogenic EC", "ESM1+ EC"]},
    # ------------------------------------------------------------------- muscle
    "skeletal_muscle_cell": {
        "singler_blueprint": ["Skeletal muscle", "Myocytes"],
        "singler_hpca": ["Skeletal_muscle_cells"],
        "panglaodb": ["Skeletal muscle cells"],
        "generic": ["Myocytes", "Skeletal myocytes", "Myofibres"],
    },
    "satellite_cell": {
        "panglaodb": ["Satellite cells"],
        "generic": ["Muscle satellite cells", "PAX7+ cells"],
    },
    "cardiac_muscle_cell": {
        "panglaodb": ["Cardiomyocytes"],
        "generic": ["Cardiomyocytes", "CM", "Cardiac myocytes"],
    },
    "visceral_smooth_muscle_cell": {
        "generic": ["Visceral SMC", "Smooth muscle (visceral)"]
    },
    "airway_smooth_muscle_cell": {
        "azimuth_lung": ["SM activated stress response"],
        "generic": ["ASM", "Airway smooth muscle"],
    },
    "interstitial_cell_of_cajal": {"generic": ["ICC", "Interstitial cells of Cajal"]},
    # ------------------------------------------------------------------- neural
    "neuron": {
        "singler_hpca": ["Neurons"],
        "panglaodb": ["Neurons"],
        "generic": ["Neuron", "Neurons", "Neuronal cells"],
    },
    "excitatory_neuron": {
        "generic": ["Excitatory neurons", "Glutamatergic neurons", "ExN"]
    },
    "inhibitory_neuron": {
        "generic": ["Inhibitory neurons", "GABAergic neurons", "InN", "Interneurons"]
    },
    "astrocyte": {
        "singler_hpca": ["Astrocyte"],
        "panglaodb": ["Astrocytes"],
        "generic": ["Astrocyte", "Astrocytes", "GFAP+ cells"],
    },
    "oligodendrocyte": {
        "panglaodb": ["Oligodendrocytes"],
        "generic": ["Oligodendrocyte", "Oligodendrocytes", "Oligo"],
    },
    "oligodendrocyte_precursor_cell": {
        "panglaodb": ["Oligodendrocyte progenitor cells"],
        "generic": ["OPC", "OPCs", "Oligodendrocyte precursor cells"],
    },
    "schwann_cell": {"panglaodb": ["Schwann cells"], "generic": ["Schwann cells"]},
    "enteric_glial_cell": {"generic": ["Enteric glia", "Enteric glial cells"]},
    "enteric_neuron": {"generic": ["Enteric neurons", "ENS neurons"]},
    "ependymal_cell": {
        "panglaodb": ["Ependymal cells"],
        "generic": ["Ependymal cells"],
    },
    "muller_glial_cell": {
        "panglaodb": ["Muller cells"],
        "generic": ["Muller glia", "Müller cells"],
    },
    "rod_photoreceptor": {"panglaodb": ["Rods"], "generic": ["Rods", "Rod cells"]},
    "cone_photoreceptor": {"panglaodb": ["Cones"], "generic": ["Cones", "Cone cells"]},
    # -------------------------------------------------------- germ / placental
    "sertoli_cell": {"panglaodb": ["Sertoli cells"], "generic": ["Sertoli cells"]},
    "leydig_cell": {"panglaodb": ["Leydig cells"], "generic": ["Leydig cells"]},
    "granulosa_cell": {
        "panglaodb": ["Granulosa cells"],
        "generic": ["Granulosa cells"],
    },
    "spermatogonium": {
        "panglaodb": ["Spermatogonia"],
        "generic": ["Spermatogonia", "SSC"],
    },
    "oocyte": {"panglaodb": ["Oocytes"], "generic": ["Oocytes"]},
    "cytotrophoblast": {"generic": ["CTB", "Cytotrophoblasts", "VCT"]},
    "syncytiotrophoblast": {"generic": ["STB", "Syncytiotrophoblast", "SCT"]},
    "extravillous_trophoblast": {"generic": ["EVT", "Extravillous trophoblasts"]},
    "decidual_stromal_cell": {"generic": ["Decidual stromal cells", "dStromal"]},
    "hofbauer_cell": {"generic": ["Hofbauer cells", "HB"]},
    "decidual_nk_cell": {"generic": ["dNK", "Decidual NK cells"]},
    # ------------------------------------------------------------- unassigned
    "unknown_cell": {
        "generic": [
            "Unknown",
            "Unassigned",
            "NA",
            "Undetermined",
            "Unclassified",
            "Other",
            "None",
        ],
    },
    "mixed_or_ambiguous_cell": {
        "generic": [
            "Mixed",
            "Ambiguous",
            "Doublet-like",
            "Mixed population",
            "Multiplet",
        ],
    },
    "technical_artefact": {
        "generic": [
            "Doublet",
            "Doublets",
            "Low quality",
            "Low-quality cells",
            "Debris",
            "Ambient RNA",
            "Empty droplet",
            "Dying cells",
            "Stressed cells",
        ],
    },
}

# Fold the celltypist_skin crosswalk in. Kept as a separate literal above so the
# 34-label model vocabulary stays readable and reviewable as one block; merged
# here so ALIASES remains the single structure the builder validates.
for _node, _labels in _CELLTYPIST_SKIN.items():
    ALIASES.setdefault(_node, {}).setdefault("celltypist_skin", []).extend(_labels)
del _node, _labels

# Labels that name a compartment rather than a cell type. They resolve to the
# coarsest node that strictly contains them, so rollup/LCA still behaves.
COMPARTMENT_LABELS: Dict[str, str] = {
    "immune cells": "hematopoietic",
    "immune": "hematopoietic",
    "leukocytes": "hematopoietic",
    "white blood cells": "hematopoietic",
    "wbc": "hematopoietic",
    "pbmc": "hematopoietic",
    "haematopoietic cells": "hematopoietic",
    "hematopoietic cells": "hematopoietic",
    "cd45+ cells": "hematopoietic",
    "ptprc+ cells": "hematopoietic",
    "mononuclear phagocytes": "myeloid_cell",
    "myeloid": "myeloid_cell",
    "granulocytes": "granulocyte",
    "t/nk": "lymphoid_cell",
    "t and nk cells": "lymphoid_cell",
    "b/plasma": "b_cell",
    "epithelium": "epithelial",
    "stroma": "stromal_mesenchymal",
    "mesenchyme": "stromal_mesenchymal",
    "vasculature": "endothelial",
    "smooth muscle": "muscle",
    "smooth muscle cells": "muscle",
    "glia": "glial_cell",
    "non-immune": "unassigned",
    "malignant cells": "unassigned",
    "tumour cells": "unassigned",
    "tumor cells": "unassigned",
    "cancer cells": "unassigned",
}
