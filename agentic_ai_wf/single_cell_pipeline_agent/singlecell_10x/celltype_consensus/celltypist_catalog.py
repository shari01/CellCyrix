"""
celltypist_catalog.py — curated catalog of HUMAN CellTypist models.

Pure data (no LLM / no HTTP). The agent layer uses this list to pick a
tissue-appropriate CellTypist model; when no organ is a reasonable match it
falls back to the general immune model below. Selection is by TISSUE/ORGAN
only — never by disease — so the disease-agnostic invariant holds.

To refresh: the authoritative list is at https://www.celltypist.org/models
(`celltypist.models.models_description()`). Add rows here as new human models
ship; the selector only ever chooses a name that exists in this list.
"""

from __future__ import annotations

from typing import Dict, List

# General fallback: pan-tissue immune model. Used when the LLM finds no
# organ-appropriate match, when the LLM is disabled, or when selection fails.
DEFAULT_FALLBACK_MODEL = "Immune_All_Low.pkl"

# Each row: model (exact .pkl name), tissue (organ context), description.
HUMAN_CELLTYPIST_MODELS: List[Dict[str, str]] = [
    # --- general / immune (pan-tissue) ---
    {
        "model": "Immune_All_Low.pkl",
        "tissue": "immune (pan-tissue)",
        "description": "High-resolution immune cell subtypes combined from 20 tissues.",
    },
    {
        "model": "Immune_All_High.pkl",
        "tissue": "immune (pan-tissue)",
        "description": "Broad immune cell types combined from 20 tissues.",
    },
    {
        "model": "Pan_Fetal_Human.pkl",
        "tissue": "fetal (multi-organ)",
        "description": "Stromal and immune populations across the human fetus.",
    },
    # --- epithelial-rich solid tissues (most useful for non-immune cohorts) ---
    {
        "model": "Human_Endometrium_Atlas.pkl",
        "tissue": "endometrium / female reproductive tract",
        "description": "Endometrial epithelial, stromal and immune cell types across the menstrual cycle.",
    },
    {
        "model": "Human_Placenta_Decidua.pkl",
        "tissue": "placenta / decidua / female reproductive tract",
        "description": "First-trimester placenta with matched maternal blood and decidua.",
    },
    {
        "model": "Cells_Adult_Breast.pkl",
        "tissue": "breast",
        "description": "Epithelial, stromal and immune cell types of the adult human breast.",
    },
    {
        "model": "Cells_Intestinal_Tract.pkl",
        "tissue": "intestine / gut",
        "description": "Intestinal epithelial, stromal and immune cells (fetal, pediatric, adult).",
    },
    {
        "model": "Human_Colorectal_Cancer.pkl",
        "tissue": "colon / colorectal",
        "description": "Epithelial, stromal and immune cell types of colon tissue.",
    },
    {
        "model": "Adult_Human_Skin.pkl",
        "tissue": "skin",
        "description": "Keratinocyte, stromal, vascular and immune cell types of healthy adult skin.",
    },
    {
        "model": "Fetal_Human_Skin.pkl",
        "tissue": "skin (fetal)",
        "description": "Cell types of developing human fetal skin.",
    },
    {
        "model": "Cells_Human_Tonsil.pkl",
        "tissue": "tonsil / lymphoid",
        "description": "Tonsillar epithelial and immune cell types (ages 3-65).",
    },
    # --- lung / airway ---
    {
        "model": "Human_Lung_Atlas.pkl",
        "tissue": "lung / respiratory",
        "description": "Integrated Human Lung Cell Atlas (healthy respiratory system).",
    },
    {
        "model": "Cells_Lung_Airway.pkl",
        "tissue": "lung / airway",
        "description": "Cell populations across five locations of human lungs and airways.",
    },
    {
        "model": "Nuclei_Lung_Airway.pkl",
        "tissue": "lung / airway",
        "description": "snRNA-seq cell populations across human lungs and airways.",
    },
    {
        "model": "Human_IPF_Lung.pkl",
        "tissue": "lung",
        "description": "Cell types from idiopathic pulmonary fibrosis, COPD and healthy lungs.",
    },
    {
        "model": "Human_PF_Lung.pkl",
        "tissue": "lung",
        "description": "Cell types from pulmonary-fibrosis lungs of adult humans.",
    },
    {
        "model": "Cells_Fetal_Lung.pkl",
        "tissue": "lung (fetal)",
        "description": "Cell types from human embryonic and fetal lungs.",
    },
    # --- abdominal / endocrine organs ---
    {
        "model": "Healthy_Human_Liver.pkl",
        "tissue": "liver",
        "description": "Cell types from scRNA/snRNA-seq of the adult human liver.",
    },
    {
        "model": "Adult_Human_PancreaticIslet.pkl",
        "tissue": "pancreas",
        "description": "Cell types from pancreatic islets of healthy adults.",
    },
    {
        "model": "Fetal_Human_Pancreas.pkl",
        "tissue": "pancreas (fetal)",
        "description": "Pancreatic cell types from human embryos (9-19 weeks).",
    },
    {
        "model": "Fetal_Human_AdrenalGlands.pkl",
        "tissue": "adrenal gland (fetal)",
        "description": "Cell types of human fetal adrenal glands.",
    },
    {
        "model": "Fetal_Human_Pituitary.pkl",
        "tissue": "pituitary (fetal)",
        "description": "Cell types of human fetal pituitaries.",
    },
    # --- cardiovascular ---
    {
        "model": "Healthy_Adult_Heart.pkl",
        "tissue": "heart",
        "description": "Cell types across eight regions of the healthy adult human heart.",
    },
    {
        "model": "Adult_Human_Vascular.pkl",
        "tissue": "vascular / endothelial (multi-organ)",
        "description": "Vascular populations combined from multiple adult human organs.",
    },
    # --- nervous system / eye / ear ---
    {
        "model": "Adult_Human_MTG.pkl",
        "tissue": "brain (middle temporal gyrus)",
        "description": "Adult human middle temporal gyrus cell types.",
    },
    {
        "model": "Adult_Human_PrefrontalCortex.pkl",
        "tissue": "brain (prefrontal cortex)",
        "description": "Adult human dorsolateral prefrontal cortex cell types.",
    },
    {
        "model": "Human_AdultAged_Hippocampus.pkl",
        "tissue": "brain (hippocampus)",
        "description": "Hippocampal cell types of adult and aged humans.",
    },
    {
        "model": "Human_Longitudinal_Hippocampus.pkl",
        "tissue": "brain (hippocampus)",
        "description": "Adult human anterior/posterior hippocampus cell types.",
    },
    {
        "model": "Developing_Human_Brain.pkl",
        "tissue": "brain (fetal)",
        "description": "First-trimester developing human brain cell types.",
    },
    {
        "model": "Developing_Human_Hippocampus.pkl",
        "tissue": "brain (fetal hippocampus)",
        "description": "Developing human hippocampus cell types.",
    },
    {
        "model": "Fetal_Human_Retina.pkl",
        "tissue": "eye / retina (fetal)",
        "description": "Human fetal neural retina and retinal pigment epithelium.",
    },
    {
        "model": "Human_Developmental_Retina.pkl",
        "tissue": "eye / retina (fetal)",
        "description": "Cell types from human fetal retina.",
    },
    {
        "model": "Nuclei_Human_InnerEar.pkl",
        "tissue": "inner ear",
        "description": "Cell types of the human inner ear.",
    },
    # --- developmental / other ---
    {
        "model": "Developing_Human_Organs.pkl",
        "tissue": "fetal (endoderm-derived organs)",
        "description": "Cell types of five endoderm-derived organs in developing humans.",
    },
    {
        "model": "Developing_Human_Gonads.pkl",
        "tissue": "gonads (fetal)",
        "description": "Human gonadal and adjacent extragonadal cell types (1st-2nd trimester).",
    },
    {
        "model": "Developing_Human_Thymus.pkl",
        "tissue": "thymus",
        "description": "Thymic cell populations from embryonic to adult stages.",
    },
    {
        "model": "Human_Embryonic_YolkSac.pkl",
        "tissue": "yolk sac (embryonic)",
        "description": "Cell types of the human yolk sac (4-8 post-conception weeks).",
    },
    # --- blood / PBMC ---
    {
        "model": "Healthy_COVID19_PBMC.pkl",
        "tissue": "blood / PBMC",
        "description": "Peripheral blood mononuclear cell types (healthy and COVID-19).",
    },
    {
        "model": "Adult_COVID19_PBMC.pkl",
        "tissue": "blood / PBMC",
        "description": "PBMC types from COVID-19 patients and healthy controls.",
    },
    {
        "model": "COVID19_HumanChallenge_Blood.pkl",
        "tissue": "blood",
        "description": "Detailed blood cell states after SARS-CoV-2 challenge.",
    },
    {
        "model": "COVID19_Immune_Landscape.pkl",
        "tissue": "immune (lung/blood)",
        "description": "Immune subtypes from lung and blood of COVID-19 patients.",
    },
]

# Fast membership check for validating an LLM's choice.
VALID_MODEL_NAMES = {m["model"] for m in HUMAN_CELLTYPIST_MODELS}


# ===========================================================================
# SingleR (celldex) reference catalog.
#
# SingleR is an algorithm, not one model — accuracy depends on how well the
# chosen reference matches the SPECIES and TISSUE. The LLM picks a reference by
# species+tissue (never disease); no organ match => the broad general fallback.
# Every name here must be supported by singler_bridge.R's `switch`.
#
# HUMAN ONLY. This list is the complete set of references the pipeline may use,
# and it is deliberately human-only, matching the CellTypist catalog above.
# `celldex::MouseRNAseqData` is therefore NOT listed: SingleR is the only voter
# with a mouse reference available, while CellTypist has human models only and
# the lineage-gate panels are human gene symbols (PTPRC, EPCAM — mouse writes
# Ptprc, Epcam, which the gate's exact match does not find, so it would abstain
# on every cluster). Offering the mouse reference would make SingleR right and
# the rest of the annotation stack wrong on the same run.
#
# To support mouse properly, four things are needed together: mouse CellTypist
# models in the catalog above, mouse (or ortholog-mapped) lineage-gate panels, a
# mouse marker table for lineage_panels.py, and species passed to the mygene
# lookup in gene_names.py. Re-adding the row below on its own is not enough:
#     {"reference": "MouseRNAseqData", "species": "mouse",
#      "tissue": "broad (mouse, all lineages)",
#      "description": "Broad mouse reference across many organs; mouse data only."},
# The name stays in singler_bridge.R's `switch` (harmless — unreachable while it
# is absent here), so re-enabling is a one-row change once the rest is in place.
# ===========================================================================
DEFAULT_SINGLER_REFERENCE = "BlueprintEncodeData"  # human, broad, covers non-immune

SINGLER_REFERENCES: List[Dict[str, str]] = [
    {
        "reference": "HumanPrimaryCellAtlasData",
        "species": "human",
        "tissue": "broad / solid tissue (all lineages)",
        "description": "Widest human reference: immune, epithelial, stromal, endothelial, neural — best for mixed solid tissues.",
    },
    {
        "reference": "BlueprintEncodeData",
        "species": "human",
        "tissue": "broad (immune + stromal + endothelial)",
        "description": "Human immune plus stromal/endothelial; solid-tissue safe general default.",
    },
    {
        "reference": "MonacoImmuneData",
        "species": "human",
        "tissue": "blood / immune",
        "description": "Fine-grained human immune subsets from blood; best for PBMC/immune-only data.",
    },
    {
        "reference": "DatabaseImmuneCellExpressionData",
        "species": "human",
        "tissue": "blood / immune",
        "description": "Human immune cell reference (DICE); immune-focused blood data.",
    },
    {
        "reference": "NovershternHematopoieticData",
        "species": "human",
        "tissue": "bone marrow / hematopoietic",
        "description": "Human hematopoietic/bone-marrow lineages; best for HSPC/marrow data.",
    },
]

# The only reference names the pipeline accepts, from either the "auto" selector
# or an explicit `singler_reference` in config. Anything else is rejected up
# front rather than handed to R, so a typo or a mouse reference cannot quietly
# annotate human data.
VALID_SINGLER_REFERENCES = {r["reference"] for r in SINGLER_REFERENCES}
