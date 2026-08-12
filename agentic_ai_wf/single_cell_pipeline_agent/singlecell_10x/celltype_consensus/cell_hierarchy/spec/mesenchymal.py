"""Stromal/mesenchymal, endothelial and muscle lineages."""

from ._dsl import N

PAN = "pan_tissue"

# --------------------------------------------------------------------------- #
# Fibroblasts and related stroma
# --------------------------------------------------------------------------- #
FIBROBLAST = N(
    "fibroblast",
    "Fibroblast",
    "CL:0000057",
    PAN,
    "COL1A1,COL1A2,DCN,LUM,PDGFRA",
    [
        N(
            "adventitial_fibroblast",
            "Adventitial fibroblast",
            "",
            PAN,
            "SFRP2,PI16,MFAP5,SERPINF1",
        ),
        N(
            "alveolar_fibroblast",
            "Alveolar fibroblast",
            "",
            "lung",
            "SPINT2,GPC3,FGF10,MACF1",
        ),
        N(
            "peribronchial_fibroblast",
            "Peribronchial fibroblast",
            "",
            "lung",
            "ASPN,SCARA5,WNT5A",
        ),
        N(
            "myofibroblast",
            "Myofibroblast",
            "CL:0000186",
            PAN,
            "ACTA2,TAGLN,POSTN,CTHRC1",
        ),
        N(
            "inflammatory_fibroblast",
            "Inflammatory fibroblast",
            "",
            PAN,
            "IL6,CXCL1,CXCL8,PDPN,IL11",
        ),
        N(
            "matrix_fibroblast",
            "Matrix-producing fibroblast",
            "",
            PAN,
            "COL3A1,COL5A1,FN1,COMP",
        ),
        N(
            "lipofibroblast",
            "Lipofibroblast",
            "",
            "lung,adipose",
            "PLIN2,FABP4,PPARG,TCF21",
        ),
        N(
            "dermal_papilla_fibroblast",
            "Dermal papilla fibroblast",
            "",
            "skin",
            "WIF1,SOX2,CRABP1",
        ),
        N(
            "dermal_sheath_fibroblast",
            "Dermal sheath fibroblast",
            "",
            "skin",
            "ACTA2,COL11A1,SOX2",
        ),
        N(
            "intestinal_fibroblast_s1",
            "Intestinal S1 (crypt-associated) fibroblast",
            "",
            "small_intestine,colon",
            "ADAMDEC1,CCL11,APOE",
        ),
        N(
            "intestinal_fibroblast_s2",
            "Intestinal S2 (villus/surface) fibroblast",
            "",
            "small_intestine,colon",
            "F3,NPY,SOX6,WNT5A",
        ),
        N(
            "intestinal_fibroblast_s3",
            "Intestinal S3 (submucosal) fibroblast",
            "",
            "small_intestine,colon",
            "C7,MFAP5,PI16",
        ),
        N(
            "wnt2b_fibroblast",
            "WNT2B-positive fibroblast",
            "",
            "small_intestine,colon",
            "WNT2B,RSPO3,GREM1",
        ),
        N(
            "telocyte",
            "Telocyte",
            "",
            "small_intestine,colon,heart",
            "FOXL1,PDGFRA,WNT5A,BMP5",
        ),
        N(
            "cancer_associated_fibroblast",
            "Cancer-associated fibroblast",
            "",
            PAN,
            "FAP,POSTN,COL11A1,THBS2",
            [
                N(
                    "myofibroblastic_caf",
                    "Myofibroblastic CAF",
                    "",
                    PAN,
                    "ACTA2,TAGLN,CTHRC1,POSTN",
                ),
                N(
                    "inflammatory_caf",
                    "Inflammatory CAF",
                    "",
                    PAN,
                    "IL6,CXCL12,PDGFRA,CFD",
                ),
                N(
                    "antigen_presenting_caf",
                    "Antigen-presenting CAF",
                    "",
                    PAN,
                    "CD74,HLA-DRA,SLPI",
                ),
            ],
        ),
        N(
            "fibroblastic_reticular_cell",
            "Fibroblastic reticular cell",
            "",
            "lymph_node,tonsil,spleen",
            "CCL19,CCL21,PDPN,DES",
        ),
        N(
            "follicular_dendritic_cell",
            "Follicular dendritic cell",
            "CL:0000442",
            "lymph_node,tonsil,spleen",
            "CR2,CXCL13,FDCSP,CLU",
        ),
        N(
            "synovial_fibroblast",
            "Synovial fibroblast",
            "",
            "synovium,joint",
            "PRG4,THY1,CDH11,HAS1",
        ),
    ],
)

PERIVASCULAR = N(
    "perivascular_cell",
    "Perivascular mural cell",
    "CL:0000669",
    PAN,
    "PDGFRB,RGS5,NOTCH3",
    [
        N("pericyte", "Pericyte", "CL:0000669", PAN, "RGS5,PDGFRB,KCNJ8,HIGD1B"),
        N(
            "vascular_smooth_muscle_cell",
            "Vascular smooth muscle cell",
            "CL:0000359",
            PAN,
            "ACTA2,MYH11,TAGLN,DES",
        ),
        N(
            "mesangial_cell",
            "Mesangial cell",
            "CL:0000650",
            "kidney",
            "PDGFRB,GATA3,ITGA8,REN",
        ),
        N(
            "juxtaglomerular_cell",
            "Juxtaglomerular (renin-producing) cell",
            "",
            "kidney",
            "REN,AKR1B1,ACTA2",
        ),
    ],
)

STELLATE = N(
    "stellate_cell",
    "Stellate cell",
    "CL:0000632",
    "liver,pancreas",
    "PDGFRB,DES,RGS5,COL1A1",
    [
        N(
            "hepatic_stellate_cell",
            "Hepatic stellate cell",
            "CL:0000632",
            "liver",
            "RGS5,DCN,COLEC11,LRAT",
        ),
        N(
            "pancreatic_stellate_cell",
            "Pancreatic stellate cell",
            "",
            "pancreas",
            "PDGFRB,ACTA2,COL1A1,RGS5",
        ),
    ],
)

SKELETAL_STROMA = N(
    "skeletal_stromal_cell",
    "Skeletal and connective-tissue stromal cell",
    "",
    "bone,cartilage,joint,adipose",
    "COL1A1,RUNX2,SOX9",
    [
        N("osteoblast", "Osteoblast", "CL:0000062", "bone", "RUNX2,SP7,BGLAP,COL1A1"),
        N("osteocyte", "Osteocyte", "CL:0000137", "bone", "SOST,DMP1,MEPE"),
        N(
            "chondrocyte",
            "Chondrocyte",
            "CL:0000138",
            "cartilage,joint",
            "ACAN,COL2A1,SOX9,COMP",
        ),
        N("tenocyte", "Tenocyte", "", "tendon", "SCX,TNMD,COL1A1,THBS4"),
        N(
            "adipocyte",
            "Adipocyte",
            "CL:0000136",
            "adipose,bone_marrow,breast",
            "ADIPOQ,LEP,PLIN1,FABP4",
            [
                N(
                    "white_adipocyte",
                    "White adipocyte",
                    "CL:0000448",
                    "adipose",
                    "ADIPOQ,LEP,PLIN1",
                ),
                N(
                    "beige_brown_adipocyte",
                    "Beige/brown adipocyte",
                    "CL:0000449",
                    "adipose",
                    "UCP1,CIDEA,PPARGC1A",
                ),
                N(
                    "preadipocyte",
                    "Preadipocyte",
                    "",
                    "adipose",
                    "PDGFRA,DLK1,CD34,PPARG",
                ),
            ],
        ),
        N(
            "mesenchymal_stromal_cell",
            "Mesenchymal stromal cell",
            "CL:0000134",
            PAN,
            "NT5E,THY1,ENG,PDGFRB",
        ),
        N(
            "bone_marrow_stromal_cell",
            "Bone-marrow stromal (CAR) cell",
            "",
            "bone_marrow",
            "CXCL12,LEPR,KITLG,VCAM1",
        ),
    ],
)

STROMAL = N(
    "stromal_mesenchymal",
    "Stromal / mesenchymal cell",
    "CL:0000499",
    PAN,
    "COL1A1,DCN,PDGFRB",
    [FIBROBLAST, PERIVASCULAR, STELLATE, SKELETAL_STROMA],
)

# --------------------------------------------------------------------------- #
# Endothelium
# --------------------------------------------------------------------------- #
VASCULAR_EC = N(
    "vascular_endothelial_cell",
    "Vascular endothelial cell",
    "CL:0002139",
    PAN,
    "PECAM1,VWF,CDH5,CLDN5",
    [
        N(
            "arterial_endothelial_cell",
            "Arterial endothelial cell",
            "CL:1000413",
            PAN,
            "GJA5,HEY1,SEMA3G,EFNB2",
        ),
        N(
            "capillary_endothelial_cell",
            "Capillary endothelial cell",
            "CL:0002144",
            PAN,
            "CA4,RGCC,SGK1,FCN3",
            [
                N(
                    "general_capillary_endothelial_cell",
                    "General capillary endothelial cell",
                    "",
                    "lung",
                    "FCN3,IL7R,EDN1,GPIHBP1",
                ),
                N(
                    "aerocyte",
                    "Aerocyte (aCap) endothelial cell",
                    "",
                    "lung",
                    "HPGD,EDNRB,SOSTDC1,TBX2",
                ),
            ],
        ),
        N(
            "venous_endothelial_cell",
            "Venous endothelial cell",
            "CL:0002543",
            PAN,
            "ACKR1,NR2F2,VCAM1,SELP",
        ),
        N(
            "high_endothelial_venule_cell",
            "High endothelial venule cell",
            "",
            "lymph_node,tonsil",
            "CCL21,CHST4,SELP,ACKR1",
        ),
        N(
            "sinusoidal_endothelial_cell",
            "Sinusoidal endothelial cell",
            "",
            "liver,spleen,bone_marrow",
            "STAB2,LYVE1,CLEC4G,FCN2",
        ),
        N(
            "glomerular_endothelial_cell",
            "Glomerular endothelial cell",
            "",
            "kidney",
            "EHD3,PLAT,SOST,EMCN",
        ),
        N(
            "tip_endothelial_cell",
            "Angiogenic tip endothelial cell",
            "",
            PAN,
            "ESM1,APLN,ANGPT2,CXCR4",
        ),
        N(
            "proliferating_endothelial_cell",
            "Proliferating endothelial cell",
            "",
            PAN,
            "MKI67,TOP2A,PECAM1",
        ),
    ],
)

LYMPHATIC_EC = N(
    "lymphatic_endothelial_cell",
    "Lymphatic endothelial cell",
    "CL:0002138",
    PAN,
    "PROX1,PDPN,LYVE1,CCL21",
    [
        N(
            "lymphatic_capillary_endothelial_cell",
            "Lymphatic capillary endothelial cell",
            "",
            PAN,
            "LYVE1,CCL21,MMRN1",
        ),
        N(
            "lymphatic_collecting_endothelial_cell",
            "Lymphatic collecting-vessel endothelial cell",
            "",
            PAN,
            "FOXC2,GJA4,ACTA2",
        ),
    ],
)

ENDOTHELIAL = N(
    "endothelial",
    "Endothelial cell",
    "CL:0000115",
    PAN,
    "PECAM1,CDH5,CLDN5",
    [
        VASCULAR_EC,
        LYMPHATIC_EC,
        N("endocardial_cell", "Endocardial cell", "", "heart", "NPR3,NFATC1,PECAM1"),
    ],
)

# --------------------------------------------------------------------------- #
# Muscle
# --------------------------------------------------------------------------- #
MUSCLE = N(
    "muscle",
    "Muscle cell",
    "CL:0000187",
    PAN,
    "DES,ACTA2,MYH11,TTN",
    [
        N(
            "skeletal_muscle_cell",
            "Skeletal muscle cell",
            "CL:0000188",
            "skeletal_muscle,tongue,diaphragm",
            "ACTA1,MYH1,MYH2,DES",
            [
                N(
                    "type_1_myofibre",
                    "Slow-twitch (type I) myofibre",
                    "",
                    "skeletal_muscle",
                    "MYH7,TNNT1,ATP2A2",
                ),
                N(
                    "type_2_myofibre",
                    "Fast-twitch (type II) myofibre",
                    "",
                    "skeletal_muscle",
                    "MYH1,MYH2,TNNT3,ATP2A1",
                ),
                N(
                    "satellite_cell",
                    "Skeletal muscle satellite cell",
                    "CL:0000680",
                    "skeletal_muscle",
                    "PAX7,MYF5,CALCR",
                ),
                N(
                    "myoblast",
                    "Myoblast",
                    "CL:0000056",
                    "skeletal_muscle",
                    "MYOD1,MYOG,DES",
                ),
            ],
        ),
        N(
            "cardiac_muscle_cell",
            "Cardiomyocyte",
            "CL:0000746",
            "heart",
            "TNNT2,MYH7,NPPA,ACTC1",
            [
                N(
                    "atrial_cardiomyocyte",
                    "Atrial cardiomyocyte",
                    "",
                    "heart",
                    "NPPA,MYL7,SLN",
                ),
                N(
                    "ventricular_cardiomyocyte",
                    "Ventricular cardiomyocyte",
                    "",
                    "heart",
                    "MYL2,MYH7,FHL2",
                ),
                N(
                    "conduction_system_cell",
                    "Cardiac conduction system cell",
                    "",
                    "heart",
                    "HCN4,SHOX2,CACNA1G",
                ),
            ],
        ),
        N(
            "visceral_smooth_muscle_cell",
            "Visceral smooth muscle cell",
            "CL:0000192",
            "gut,airway,bladder,uterus,vessel",
            "MYH11,DES,ACTG2,CNN1",
            [
                N(
                    "airway_smooth_muscle_cell",
                    "Airway smooth muscle cell",
                    "",
                    "lung,bronchus",
                    "MYH11,DES,CNN1,ACTG2",
                ),
                N(
                    "intestinal_smooth_muscle_cell",
                    "Intestinal smooth muscle cell",
                    "",
                    "small_intestine,colon",
                    "ACTG2,MYH11,DES",
                ),
                N(
                    "myometrial_smooth_muscle_cell",
                    "Myometrial smooth muscle cell",
                    "",
                    "uterus",
                    "OXTR,ACTG2,MYH11",
                ),
            ],
        ),
        N(
            "interstitial_cell_of_cajal",
            "Interstitial cell of Cajal",
            "",
            "gut,stomach",
            "KIT,ANO1,PDGFRA",
        ),
        N("myopericyte", "Myopericyte", "", PAN, "RGS5,ACTA2,NOTCH3"),
    ],
)

BRANCH = [STROMAL, ENDOTHELIAL, MUSCLE]
