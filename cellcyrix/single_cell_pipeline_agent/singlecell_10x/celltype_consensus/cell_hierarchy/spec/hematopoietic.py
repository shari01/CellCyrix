"""Haematopoietic lineage: lymphoid, myeloid, erythroid, megakaryocytic, progenitors."""

from ._dsl import N

PAN = "pan_tissue"

# --------------------------------------------------------------------------- #
# T cells
# --------------------------------------------------------------------------- #
_CD4_T = N(
    "cd4_t_cell",
    "CD4-positive T cell",
    "CL:0000624",
    PAN,
    "CD4,IL7R,CD40LG",
    [
        N(
            "cd4_naive_t_cell",
            "CD4-positive naive T cell",
            "CL:0000895",
            PAN,
            "CCR7,SELL,TCF7,LEF1",
        ),
        N(
            "cd4_central_memory_t_cell",
            "CD4-positive central memory T cell",
            "CL:0000904",
            PAN,
            "CCR7,SELL,IL7R,S100A4",
        ),
        N(
            "cd4_effector_memory_t_cell",
            "CD4-positive effector memory T cell",
            "CL:0000905",
            PAN,
            "IL7R,S100A4,KLRB1",
        ),
        N("th1_cell", "T-helper 1 cell", "CL:0000545", PAN, "TBX21,IFNG,CXCR3"),
        N("th2_cell", "T-helper 2 cell", "CL:0000546", PAN, "GATA3,IL4,IL13,CCR4"),
        N("th17_cell", "T-helper 17 cell", "CL:0000899", PAN, "RORC,IL17A,IL23R,CCR6"),
        N("th22_cell", "T-helper 22 cell", "", PAN, "AHR,IL22,CCR10"),
        N(
            "tfh_cell",
            "T follicular helper cell",
            "CL:0002038",
            "lymph_node,tonsil,spleen,mucosa",
            "CXCR5,BCL6,PDCD1,ICOS",
        ),
        # Treg activation states (naive / effector / tissue-resident) are deliberately
        # NOT nodes. They are the identity 'regulatory_t_cell' crossed with the
        # activation and localisation axes in spec/states.py. Resolving
        # 'Effector Treg' returns node=regulatory_t_cell, states=('activated',),
        # which composes without deepening the tree or corrupting rollup.
        N(
            "regulatory_t_cell",
            "Regulatory T cell",
            "CL:0000815",
            PAN,
            "FOXP3,IL2RA,CTLA4,IKZF2",
        ),
        N("tr1_cell", "Type 1 regulatory T cell", "", PAN, "IL10,LAG3,CD49B"),
        N(
            "cd4_cytotoxic_t_cell",
            "CD4-positive cytotoxic T cell",
            "",
            PAN,
            "GZMB,GNLY,NKG7,CD4",
        ),
    ],
)

_CD8_T = N(
    "cd8_t_cell",
    "CD8-positive T cell",
    "CL:0000625",
    PAN,
    "CD8A,CD8B,CD3D",
    [
        N(
            "cd8_naive_t_cell",
            "CD8-positive naive T cell",
            "CL:0000900",
            PAN,
            "CCR7,SELL,TCF7,LEF1",
        ),
        N(
            "cd8_central_memory_t_cell",
            "CD8-positive central memory T cell",
            "CL:0000907",
            PAN,
            "CCR7,IL7R,SELL",
        ),
        N(
            "cd8_effector_memory_t_cell",
            "CD8-positive effector memory T cell",
            "CL:0000913",
            PAN,
            "GZMK,IL7R,CCL5",
        ),
        N(
            "cd8_temra_cell",
            "CD8-positive terminally differentiated effector memory T cell",
            "",
            PAN,
            "GZMB,GNLY,FGFBP2,NKG7,FCGR3A",
        ),
        N(
            "cd8_tissue_resident_memory_t_cell",
            "CD8-positive tissue-resident memory T cell",
            "",
            PAN,
            "ITGAE,CD69,CXCR6,ZNF683",
        ),
        N(
            "cd8_exhausted_t_cell",
            "CD8-positive exhausted T cell",
            "",
            PAN,
            "PDCD1,HAVCR2,LAG3,TOX,CTLA4",
        ),
        N(
            "cd8_stem_like_t_cell",
            "CD8-positive stem-like progenitor exhausted T cell",
            "",
            PAN,
            "TCF7,PDCD1,SLAMF6",
        ),
    ],
)

_UNCONV_T = [
    N(
        "mait_cell",
        "Mucosal-associated invariant T cell",
        "CL:0000940",
        PAN,
        "SLC4A10,KLRB1,TRAV1-2,ZBTB16",
    ),
    N(
        "gamma_delta_t_cell",
        "Gamma-delta T cell",
        "CL:0000798",
        PAN,
        "TRDC,TRGC1,TRGC2,KLRD1",
        [
            N(
                "vd1_t_cell",
                "V-delta-1 gamma-delta T cell",
                "",
                "mucosa,skin,liver",
                "TRDV1,CD69",
            ),
            N(
                "vd2_t_cell",
                "V-delta-2 gamma-delta T cell",
                "",
                "blood",
                "TRDV2,TRGV9,GZMB",
            ),
        ],
    ),
    N(
        "nkt_cell",
        "Natural killer T cell",
        "CL:0000814",
        PAN,
        "CD3D,KLRD1,ZBTB16,NCAM1",
    ),
    N("double_negative_t_cell", "Double-negative T cell", "", PAN, "CD3D,PTPRC"),
    N("proliferating_t_cell", "Proliferating T cell", "", PAN, "MKI67,TOP2A,CD3D"),
]

_THYMIC_T = [
    N(
        "double_negative_thymocyte",
        "Double-negative thymocyte",
        "CL:0002489",
        "thymus",
        "PTCRA,RAG1,CD34",
    ),
    N(
        "double_positive_thymocyte",
        "Double-positive thymocyte",
        "CL:0000809",
        "thymus",
        "CD4,CD8A,RAG1,PTCRA",
    ),
    N(
        "single_positive_thymocyte",
        "Single-positive thymocyte",
        "",
        "thymus",
        "CD3E,CCR7,ITM2A",
    ),
]

T_CELL = N(
    "t_cell",
    "T cell",
    "CL:0000084",
    PAN,
    "CD3D,CD3E,CD3G,TRAC",
    [_CD4_T, _CD8_T, *_UNCONV_T, *_THYMIC_T],
)

# --------------------------------------------------------------------------- #
# NK and innate lymphoid
# --------------------------------------------------------------------------- #
NK_CELL = N(
    "nk_cell",
    "Natural killer cell",
    "CL:0000623",
    PAN,
    "NKG7,GNLY,KLRD1,NCAM1",
    [
        N(
            "cd56_dim_nk_cell",
            "CD56-dim CD16-positive natural killer cell",
            "",
            PAN,
            "FCGR3A,FGFBP2,PRF1,GZMB",
        ),
        N(
            "cd56_bright_nk_cell",
            "CD56-bright CD16-negative natural killer cell",
            "",
            PAN,
            "NCAM1,XCL1,GZMK,SELL",
        ),
        N(
            "adaptive_nk_cell",
            "Adaptive memory-like natural killer cell",
            "",
            PAN,
            "KLRC2,FCER1G,ZEB2",
        ),
        N(
            "tissue_resident_nk_cell",
            "Tissue-resident natural killer cell",
            "",
            "liver,uterus,mucosa",
            "CD69,ITGA1,CXCR6",
        ),
        N(
            "decidual_nk_cell",
            "Decidual natural killer cell",
            "",
            "uterus,decidua",
            "CD9,ITGA1,KIR2DL1,CSF1",
        ),
        N(
            "proliferating_nk_cell",
            "Proliferating natural killer cell",
            "",
            PAN,
            "MKI67,TOP2A,NKG7",
        ),
    ],
)

ILC = N(
    "innate_lymphoid_cell",
    "Innate lymphoid cell",
    "CL:0001065",
    PAN,
    "IL7R,KIT,RORC",
    [
        N("ilc1", "Group 1 innate lymphoid cell", "", PAN, "TBX21,IFNG,IL7R"),
        N(
            "ilc2",
            "Group 2 innate lymphoid cell",
            "",
            "lung,skin,mucosa",
            "GATA3,IL1RL1,PTGDR2,IL13",
        ),
        N(
            "ilc3",
            "Group 3 innate lymphoid cell",
            "",
            "mucosa,tonsil",
            "RORC,IL22,KIT,NCR2",
        ),
        N(
            "lymphoid_tissue_inducer_cell",
            "Lymphoid tissue inducer cell",
            "",
            "mucosa,lymph_node",
            "RORC,LTB,IL7R",
        ),
    ],
)

# --------------------------------------------------------------------------- #
# B cells and plasma cells
# --------------------------------------------------------------------------- #
B_CELL = N(
    "b_cell",
    "B cell",
    "CL:0000236",
    PAN,
    "CD79A,CD79B,MS4A1,CD19",
    [
        N(
            "transitional_b_cell",
            "Transitional B cell",
            "",
            "blood,bone_marrow",
            "MME,CD24,IGHM,VPREB3",
        ),
        N("naive_b_cell", "Naive B cell", "CL:0000788", PAN, "TCL1A,IGHD,IGHM,FCER2"),
        N(
            "memory_b_cell",
            "Memory B cell",
            "CL:0000787",
            PAN,
            "CD27,TNFRSF13B,AIM2,IGHG1",
        ),
        N(
            "germinal_center_b_cell",
            "Germinal centre B cell",
            "CL:0000844",
            "lymph_node,tonsil,spleen,mucosa",
            "BCL6,AICDA,MEF2B,RGS13",
            [
                N(
                    "gc_dark_zone_b_cell",
                    "Germinal centre dark-zone B cell",
                    "",
                    "lymph_node,tonsil",
                    "CXCR4,MKI67,AICDA",
                ),
                N(
                    "gc_light_zone_b_cell",
                    "Germinal centre light-zone B cell",
                    "",
                    "lymph_node,tonsil",
                    "CD83,CD40,BCL2A1",
                ),
            ],
        ),
        N(
            "atypical_b_cell",
            "Atypical memory B cell",
            "",
            PAN,
            "ITGAX,TBX21,FCRL5,ZEB2",
        ),
        N("regulatory_b_cell", "Regulatory B cell", "", PAN, "IL10,CD24,CD27"),
        N(
            "marginal_zone_b_cell",
            "Marginal-zone B cell",
            "CL:0000845",
            "spleen,mucosa",
            "CR2,CD1C,IGHM",
        ),
        N("proliferating_b_cell", "Proliferating B cell", "", PAN, "MKI67,TOP2A,MS4A1"),
    ],
)

PLASMA = N(
    "plasma_cell_lineage",
    "Plasma cell lineage",
    "CL:0000786",
    PAN,
    "MZB1,XBP1,JCHAIN,SDC1",
    [
        N("plasmablast", "Plasmablast", "CL:0000980", PAN, "MKI67,MZB1,XBP1,PRDM1"),
        N(
            "plasma_cell",
            "Plasma cell",
            "CL:0000786",
            PAN,
            "SDC1,MZB1,DERL3,JCHAIN",
            [
                N(
                    "iga_plasma_cell",
                    "IgA-secreting plasma cell",
                    "",
                    "mucosa,gut,salivary_gland",
                    "IGHA1,IGHA2,JCHAIN",
                ),
                N(
                    "igg_plasma_cell",
                    "IgG-secreting plasma cell",
                    "",
                    PAN,
                    "IGHG1,IGHG3,MZB1",
                ),
                N("igm_plasma_cell", "IgM-secreting plasma cell", "", PAN, "IGHM,MZB1"),
                N(
                    "ige_plasma_cell",
                    "IgE-secreting plasma cell",
                    "",
                    "mucosa,nasal",
                    "IGHE,MZB1",
                ),
            ],
        ),
    ],
)

_B_PROGENITORS = N(
    "b_cell_progenitor",
    "B-cell progenitor",
    "CL:0000826",
    "bone_marrow",
    "VPREB1,DNTT,MME",
    [
        N(
            "pro_b_cell",
            "Pro-B cell",
            "CL:0000826",
            "bone_marrow",
            "DNTT,VPREB1,IGLL1,CD34",
        ),
        N(
            "pre_b_cell",
            "Pre-B cell",
            "CL:0000817",
            "bone_marrow",
            "VPREB3,IGLL5,MME,CD79B",
        ),
        N(
            "immature_b_cell",
            "Immature B cell",
            "CL:0000816",
            "bone_marrow",
            "IGHM,MME,CD24",
        ),
    ],
)

LYMPHOID = N(
    "lymphoid_cell",
    "Lymphoid cell",
    "CL:0000542",
    PAN,
    "PTPRC,IL7R",
    [
        T_CELL,
        NK_CELL,
        ILC,
        B_CELL,
        PLASMA,
        _B_PROGENITORS,
        N(
            "common_lymphoid_progenitor",
            "Common lymphoid progenitor",
            "CL:0000051",
            "bone_marrow",
            "IL7R,CD34,DNTT",
        ),
    ],
)

# --------------------------------------------------------------------------- #
# Monocytes / macrophages
# --------------------------------------------------------------------------- #
MONOCYTE = N(
    "monocyte",
    "Monocyte",
    "CL:0000576",
    PAN,
    "LYZ,CD14,FCN1,VCAN",
    [
        N(
            "classical_monocyte",
            "Classical CD14-positive monocyte",
            "CL:0000860",
            PAN,
            "CD14,S100A8,S100A9,VCAN",
        ),
        N(
            "non_classical_monocyte",
            "Non-classical CD16-positive monocyte",
            "CL:0000875",
            PAN,
            "FCGR3A,LST1,CDKN1C,MS4A7",
        ),
        N(
            "intermediate_monocyte",
            "Intermediate monocyte",
            "CL:0002393",
            PAN,
            "CD14,FCGR3A,HLA-DRA",
        ),
        N(
            "monocyte_precursor",
            "Monocyte precursor",
            "",
            "bone_marrow,blood",
            "MPO,PRTN3,CD14,LYZ",
        ),
    ],
)

MACROPHAGE = N(
    "macrophage",
    "Macrophage",
    "CL:0000235",
    PAN,
    "CD68,CD163,MRC1,CSF1R",
    [
        N(
            "alveolar_macrophage",
            "Alveolar macrophage",
            "CL:0000583",
            "lung",
            "MARCO,FABP4,MCEMP1,PPARG",
        ),
        N(
            "interstitial_macrophage",
            "Interstitial macrophage",
            "",
            "lung,heart,kidney",
            "C1QA,LYVE1,CD163",
        ),
        N(
            "kupffer_cell",
            "Kupffer cell",
            "CL:0000091",
            "liver",
            "CD5L,MARCO,TIMD4,VSIG4",
        ),
        N(
            "microglial_cell",
            "Microglial cell",
            "CL:0000129",
            "brain,spinal_cord,retina",
            "P2RY12,TMEM119,CX3CR1,CSF1R",
        ),
        N("osteoclast", "Osteoclast", "CL:0000092", "bone", "ACP5,CTSK,MMP9,TNFRSF11A"),
        N(
            "langerhans_cell",
            "Langerhans cell",
            "CL:0000453",
            "skin,mucosa",
            "CD1A,CD207,EPCAM",
        ),
        N(
            "peritoneal_macrophage",
            "Serosal cavity macrophage",
            "",
            "peritoneum,pleura",
            "GATA6,LYVE1,TIMD4",
        ),
        N(
            "perivascular_macrophage",
            "Perivascular macrophage",
            "",
            PAN,
            "LYVE1,MRC1,CD163,STAB1",
        ),
        N(
            "lipid_associated_macrophage",
            "Lipid-associated macrophage",
            "",
            "adipose,liver,plaque",
            "TREM2,SPP1,GPNMB,LPL",
        ),
        N(
            "monocyte_derived_macrophage",
            "Monocyte-derived macrophage",
            "",
            PAN,
            "CCR2,FCN1,CD68,S100A8",
        ),
        N(
            "hemophagocytic_macrophage",
            "Erythrophagocytic macrophage",
            "",
            "spleen,bone_marrow,liver",
            "CD5L,SPIC,HMOX1,SLC40A1",
        ),
        N(
            "multinucleated_giant_cell",
            "Multinucleated giant cell",
            "",
            PAN,
            "CD68,ACP5,MMP9",
        ),
        N(
            "proliferating_macrophage",
            "Proliferating macrophage",
            "",
            PAN,
            "MKI67,TOP2A,CD68",
        ),
    ],
)

DC = N(
    "dendritic_cell",
    "Dendritic cell",
    "CL:0000451",
    PAN,
    "HLA-DRA,CD74,FLT3",
    [
        N(
            "conventional_dc1",
            "Conventional type 1 dendritic cell",
            "CL:0000990",
            PAN,
            "CLEC9A,XCR1,CADM1,IDO1",
        ),
        N(
            "conventional_dc2",
            "Conventional type 2 dendritic cell",
            "CL:0002399",
            PAN,
            "CD1C,FCER1A,CLEC10A",
        ),
        N("dc3", "DC3 inflammatory dendritic cell", "", PAN, "CD14,CD1C,S100A8,IL1B"),
        N(
            "plasmacytoid_dc",
            "Plasmacytoid dendritic cell",
            "CL:0000784",
            PAN,
            "LILRA4,IL3RA,CLEC4C,GZMB",
        ),
        N(
            "as_dc",
            "AXL-positive SIGLEC6-positive dendritic cell",
            "",
            PAN,
            "AXL,SIGLEC6,PPP1R14A",
        ),
        N(
            "migratory_dc",
            "Migratory maturation-state dendritic cell",
            "",
            PAN,
            "LAMP3,CCR7,CCL19,IDO1",
        ),
        N(
            "monocyte_derived_dc",
            "Monocyte-derived dendritic cell",
            "",
            PAN,
            "CD1C,CD14,FCER1A,MRC1",
        ),
        N(
            "follicular_dendritic_cell_placeholder",
            "Follicular dendritic cell (stromal, cross-referenced)",
            "CL:0000442",
            "lymph_node,tonsil",
            "CR2,CXCL13,FDCSP",
        ),
    ],
)

GRANULOCYTE = N(
    "granulocyte",
    "Granulocyte",
    "CL:0000094",
    PAN,
    "S100A8,S100A9,FCGR3B",
    [
        N(
            "neutrophil",
            "Neutrophil",
            "CL:0000775",
            PAN,
            "FCGR3B,CSF3R,ELANE,S100A8",
            [
                N(
                    "mature_neutrophil",
                    "Mature neutrophil",
                    "",
                    PAN,
                    "FCGR3B,CXCR2,SELL",
                ),
                N(
                    "immature_neutrophil",
                    "Immature neutrophil",
                    "",
                    "bone_marrow,blood",
                    "MPO,ELANE,DEFA3,CAMP",
                ),
                N(
                    "interferon_stimulated_neutrophil",
                    "Interferon-stimulated neutrophil",
                    "",
                    PAN,
                    "ISG15,IFIT1,IFIT3",
                ),
            ],
        ),
        N("eosinophil", "Eosinophil", "CL:0000771", PAN, "CLC,PRG2,IL5RA,EPX"),
        N("basophil", "Basophil", "CL:0000767", PAN, "CPA3,HDC,MS4A2,GATA2"),
    ],
)

MAST = N(
    "mast_cell",
    "Mast cell",
    "CL:0000097",
    PAN,
    "TPSAB1,TPSB2,CPA3,KIT",
    [
        N(
            "tryptase_only_mast_cell",
            "Tryptase-only mast cell",
            "",
            "lung,mucosa",
            "TPSAB1,CMA1",
        ),
        N(
            "tryptase_chymase_mast_cell",
            "Tryptase-chymase mast cell",
            "",
            "skin,gut_submucosa",
            "TPSAB1,CMA1,CTSG",
        ),
    ],
)

MYELOID_SUPPRESSIVE = N(
    "myeloid_suppressive_cell",
    "Suppressive myeloid cell",
    "",
    PAN,
    "S100A8,S100A9,ARG1,CD84",
    [
        N(
            "monocytic_suppressive_myeloid_cell",
            "Monocytic suppressive myeloid cell",
            "",
            PAN,
            "CD14,S100A9,IL4I1",
        ),
        N(
            "granulocytic_suppressive_myeloid_cell",
            "Granulocytic suppressive myeloid cell",
            "",
            PAN,
            "FCGR3B,ARG1,CXCR2",
        ),
    ],
)

MYELOID = N(
    "myeloid_cell",
    "Myeloid cell",
    "CL:0000763",
    PAN,
    "LYZ,ITGAM,CSF1R",
    [
        MONOCYTE,
        MACROPHAGE,
        DC,
        GRANULOCYTE,
        MAST,
        MYELOID_SUPPRESSIVE,
        N(
            "common_myeloid_progenitor",
            "Common myeloid progenitor",
            "CL:0000049",
            "bone_marrow",
            "CD34,MPO,GATA2",
        ),
        N(
            "granulocyte_monocyte_progenitor",
            "Granulocyte-monocyte progenitor",
            "CL:0000557",
            "bone_marrow",
            "MPO,ELANE,CD34,AZU1",
        ),
    ],
)

# --------------------------------------------------------------------------- #
# Erythroid, megakaryocytic, stem/progenitor
# --------------------------------------------------------------------------- #
ERYTHROID = N(
    "erythroid_cell",
    "Erythroid cell",
    "CL:0000764",
    "bone_marrow,blood,liver,spleen",
    "HBB,HBA1,ALAS2,GYPA",
    [
        N(
            "proerythroblast",
            "Proerythroblast",
            "CL:0000547",
            "bone_marrow",
            "GATA1,KIT,CD36,TFRC",
        ),
        N(
            "basophilic_erythroblast",
            "Basophilic erythroblast",
            "CL:0000549",
            "bone_marrow",
            "HBB,GYPA,ALAS2",
        ),
        N(
            "polychromatic_erythroblast",
            "Polychromatic erythroblast",
            "CL:0000551",
            "bone_marrow",
            "HBB,HBA1,SLC4A1",
        ),
        N(
            "orthochromatic_erythroblast",
            "Orthochromatic erythroblast",
            "CL:0000552",
            "bone_marrow",
            "HBB,SLC4A1,AHSP",
        ),
        N(
            "reticulocyte",
            "Reticulocyte",
            "CL:0000558",
            "bone_marrow,blood",
            "HBB,SLC4A1,BNIP3L",
        ),
        N("erythrocyte", "Erythrocyte", "CL:0000232", "blood", "HBB,HBA1,HBA2"),
    ],
)

MEGAKARYOCYTIC = N(
    "megakaryocytic_cell",
    "Megakaryocytic cell",
    "CL:0000556",
    "bone_marrow,blood,lung",
    "PPBP,PF4,ITGA2B,GP9",
    [
        N(
            "megakaryocyte",
            "Megakaryocyte",
            "CL:0000556",
            "bone_marrow,lung",
            "PF4,VWF,ITGA2B,GP1BA",
        ),
        N("platelet", "Platelet", "CL:0000233", "blood", "PPBP,PF4,TUBB1,GP9"),
    ],
)

HSPC = N(
    "hematopoietic_stem_progenitor_cell",
    "Haematopoietic stem and progenitor cell",
    "CL:0000037",
    "bone_marrow,cord_blood,blood",
    "CD34,SPINK2,PROM1",
    [
        N(
            "hematopoietic_stem_cell",
            "Haematopoietic stem cell",
            "CL:0000037",
            "bone_marrow,cord_blood",
            "CD34,AVP,HLF,CRHBP",
        ),
        N(
            "multipotent_progenitor",
            "Multipotent progenitor",
            "CL:0000837",
            "bone_marrow",
            "CD34,SPINK2,MLLT3",
        ),
        N(
            "megakaryocyte_erythroid_progenitor",
            "Megakaryocyte-erythroid progenitor",
            "CL:0000050",
            "bone_marrow",
            "GATA1,KLF1,ITGA2B",
        ),
        N(
            "lymphoid_primed_multipotent_progenitor",
            "Lymphoid-primed multipotent progenitor",
            "",
            "bone_marrow",
            "CD34,IL7R,FLT3,DNTT",
        ),
    ],
)

HEMATOPOIETIC = N(
    "hematopoietic",
    "Haematopoietic cell",
    "CL:0000988",
    PAN,
    "PTPRC",
    [LYMPHOID, MYELOID, ERYTHROID, MEGAKARYOCYTIC, HSPC],
)

BRANCH = [HEMATOPOIETIC]
