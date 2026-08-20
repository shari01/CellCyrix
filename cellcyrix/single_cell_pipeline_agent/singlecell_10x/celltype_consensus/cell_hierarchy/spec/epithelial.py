"""Epithelial lineage: airway, alveolar, intestinal, squamous, glandular, renal, hepatobiliary."""

from ._dsl import N

PAN = "pan_tissue"

# --------------------------------------------------------------------------- #
# Airway epithelium
# --------------------------------------------------------------------------- #
AIRWAY = N(
    "airway_epithelial_cell",
    "Airway epithelial cell",
    "CL:0000082",
    "lung,trachea,bronchus,nasal",
    "EPCAM,KRT8,KRT18,SCGB1A1",
    [
        N(
            "basal_cell_airway",
            "Airway basal cell",
            "CL:0002633",
            "lung,trachea,bronchus,nasal",
            "KRT5,KRT14,TP63,DLK2",
            [
                N(
                    "suprabasal_cell_airway",
                    "Airway suprabasal cell",
                    "",
                    "lung,trachea,bronchus",
                    "KRT5,NOTCH3,SERPINB3",
                ),
                N(
                    "proliferating_basal_cell_airway",
                    "Proliferating airway basal cell",
                    "",
                    "lung,trachea,bronchus",
                    "MKI67,TOP2A,KRT5",
                ),
                N(
                    "hillock_cell",
                    "Hillock luminal cell",
                    "",
                    "lung,trachea",
                    "KRT13,KRT4,S100A2,SPRR3",
                ),
            ],
        ),
        N(
            "club_cell",
            "Club (secretory) cell",
            "CL:0000158",
            "lung,bronchus,bronchiole",
            "SCGB1A1,SCGB3A2,BPIFB1",
        ),
        N(
            "goblet_cell_airway",
            "Airway goblet cell",
            "CL:1000143",
            "lung,bronchus,nasal",
            "MUC5AC,MUC5B,TFF3,SPDEF",
        ),
        N(
            "ciliated_cell_airway",
            "Airway ciliated cell",
            "CL:0005012",
            "lung,trachea,bronchus,nasal",
            "FOXJ1,TPPP3,PIFO,TUBA1A",
            [
                N(
                    "deuterosomal_cell",
                    "Deuterosomal cell",
                    "",
                    "lung,nasal",
                    "DEUP1,CDC20B,PLK4,FOXN4",
                ),
            ],
        ),
        N(
            "tuft_cell_airway",
            "Airway tuft (brush) cell",
            "",
            "lung,trachea",
            "POU2F3,ASCL2,AVIL,TRPM5",
        ),
        N(
            "pulmonary_ionocyte",
            "Pulmonary ionocyte",
            "",
            "lung,trachea,nasal",
            "FOXI1,CFTR,ASCL3,ATP6V1B1",
        ),
        N(
            "pulmonary_neuroendocrine_cell",
            "Pulmonary neuroendocrine cell",
            "CL:0000082",
            "lung,bronchus",
            "CALCA,ASCL1,CHGA,SYP",
        ),
        N(
            "serous_cell_airway",
            "Airway submucosal gland serous cell",
            "",
            "trachea,bronchus,nasal",
            "LTF,LYZ,PRR4,ZG16B",
        ),
        N(
            "mucous_cell_airway",
            "Airway submucosal gland mucous cell",
            "",
            "trachea,bronchus,nasal",
            "MUC5B,BPIFB2,TFF3",
        ),
    ],
)

ALVEOLAR = N(
    "alveolar_epithelial_cell",
    "Alveolar epithelial cell",
    "CL:0002062",
    "lung",
    "EPCAM,NKX2-1,SFTPB",
    [
        N(
            "alveolar_type_1_cell",
            "Alveolar type 1 cell",
            "CL:0002062",
            "lung",
            "AGER,PDPN,CAV1,EMP2",
        ),
        N(
            "alveolar_type_2_cell",
            "Alveolar type 2 cell",
            "CL:0002063",
            "lung",
            "SFTPC,SFTPB,SFTPA1,NAPSA",
            [
                N(
                    "alveolar_type_2_signalling_cell",
                    "Alveolar type 2 signalling (AT2-s) cell",
                    "",
                    "lung",
                    "SFTPC,WIF1,CTNNB1,LRRK2",
                ),
                N(
                    "proliferating_alveolar_type_2_cell",
                    "Proliferating alveolar type 2 cell",
                    "",
                    "lung",
                    "MKI67,TOP2A,SFTPC",
                ),
            ],
        ),
        N(
            "transitional_alveolar_epithelial_cell",
            "Transitional (KRT8-high) alveolar epithelial cell",
            "",
            "lung",
            "KRT8,KRT18,CLDN4,SFN",
        ),
        N(
            "respiratory_bronchiole_secretory_cell",
            "Respiratory airway secretory cell",
            "",
            "lung",
            "SCGB3A2,SFTPB,MUC5B",
        ),
        N(
            "aberrant_basaloid_cell",
            "Aberrant basaloid cell",
            "",
            "lung",
            "KRT17,COL1A1,CDH2,ITGB6,PTCHD4",
        ),
    ],
)

# --------------------------------------------------------------------------- #
# Intestinal epithelium
# --------------------------------------------------------------------------- #
INTESTINAL = N(
    "intestinal_epithelial_cell",
    "Intestinal epithelial cell",
    "CL:0002563",
    "small_intestine,colon,rectum,ileum",
    "EPCAM,CDH1,KRT8,KRT18",
    [
        N(
            "intestinal_stem_cell",
            "Intestinal crypt stem cell",
            "CL:0009043",
            "small_intestine,colon",
            "LGR5,OLFM4,ASCL2,RGMB",
        ),
        N(
            "transit_amplifying_cell_intestine",
            "Intestinal transit-amplifying cell",
            "",
            "small_intestine,colon",
            "MKI67,TOP2A,PCNA,OLFM4",
        ),
        N(
            "enterocyte",
            "Enterocyte",
            "CL:0000584",
            "small_intestine,ileum,duodenum",
            "APOA4,FABP2,ALPI,RBP2",
            [
                N(
                    "immature_enterocyte",
                    "Immature enterocyte",
                    "",
                    "small_intestine",
                    "FABP1,ADA,CA2",
                ),
                N(
                    "mature_enterocyte",
                    "Mature enterocyte",
                    "",
                    "small_intestine",
                    "APOA4,ALPI,SI,APOB",
                ),
            ],
        ),
        N(
            "colonocyte",
            "Colonocyte",
            "CL:0011108",
            "colon,rectum",
            "CA1,CA2,AQP8,SLC26A3",
            [
                N(
                    "immature_colonocyte",
                    "Immature colonocyte",
                    "",
                    "colon",
                    "CA2,SLC26A2,MT1G",
                ),
                N(
                    "mature_colonocyte",
                    "Mature colonocyte",
                    "",
                    "colon,rectum",
                    "AQP8,CA1,SLC26A3,GUCA2A",
                ),
                N(
                    "best4_epithelial_cell",
                    "BEST4-positive epithelial cell",
                    "",
                    "small_intestine,colon",
                    "BEST4,OTOP2,CA7,SPIB",
                ),
            ],
        ),
        N(
            "goblet_cell_intestine",
            "Intestinal goblet cell",
            "CL:1000320",
            "small_intestine,colon,rectum",
            "MUC2,TFF3,SPINK4,CLCA1",
            [
                N(
                    "crypt_goblet_cell",
                    "Crypt goblet cell",
                    "",
                    "colon",
                    "MUC2,LYZ,REG4",
                ),
                N(
                    "surface_goblet_cell",
                    "Surface goblet cell",
                    "",
                    "colon",
                    "MUC2,ITLN1,CLCA1,WFDC2",
                ),
            ],
        ),
        N(
            "paneth_cell",
            "Paneth cell",
            "CL:0000510",
            "small_intestine,ileum",
            "DEFA5,DEFA6,LYZ,REG3A",
        ),
        N(
            "enteroendocrine_cell",
            "Enteroendocrine cell",
            "CL:0000164",
            "small_intestine,colon,stomach",
            "CHGA,CHGB,NEUROD1,TPH1",
            [
                N(
                    "enterochromaffin_cell",
                    "Enterochromaffin cell",
                    "",
                    "small_intestine,colon",
                    "TPH1,SLC18A1,CHGA",
                ),
                N("l_cell", "L cell", "", "ileum,colon", "GCG,PYY,INSL5"),
                N("k_cell", "K cell", "", "duodenum,jejunum", "GIP,CHGA"),
                N("d_cell", "D cell", "", "stomach,small_intestine", "SST,CHGA"),
                N("i_cell", "I cell", "", "duodenum,jejunum", "CCK,CHGA"),
                N("n_cell", "N cell", "", "ileum", "NTS,CHGA"),
            ],
        ),
        N(
            "tuft_cell_intestine",
            "Intestinal tuft cell",
            "CL:0002204",
            "small_intestine,colon",
            "POU2F3,TRPM5,AVIL,SH2D6",
        ),
        N(
            "microfold_cell",
            "Microfold (M) cell",
            "CL:0000682",
            "small_intestine,colon",
            "SPIB,CCL20,TNFRSF11A,GP2",
        ),
        N(
            "intestinal_secretory_progenitor",
            "Intestinal secretory progenitor",
            "",
            "small_intestine,colon",
            "ATOH1,DLL1,SPDEF",
        ),
    ],
)

# --------------------------------------------------------------------------- #
# Gastric and oesophageal
# --------------------------------------------------------------------------- #
GASTRIC = N(
    "gastric_epithelial_cell",
    "Gastric epithelial cell",
    "CL:0002178",
    "stomach",
    "EPCAM,TFF1,MUC5AC",
    [
        N(
            "gastric_pit_cell",
            "Gastric foveolar (pit) mucous cell",
            "CL:0002179",
            "stomach",
            "MUC5AC,TFF1,GKN1",
        ),
        N(
            "gastric_chief_cell",
            "Gastric chief (zymogenic) cell",
            "CL:0000155",
            "stomach",
            "PGA5,PGC,LIPF",
        ),
        N(
            "gastric_parietal_cell",
            "Gastric parietal cell",
            "CL:0000162",
            "stomach",
            "ATP4A,ATP4B,GIF",
        ),
        N(
            "gastric_neck_cell",
            "Gastric mucous neck cell",
            "",
            "stomach",
            "MUC6,TFF2,LYZ",
        ),
        N(
            "gastric_isthmus_stem_cell",
            "Gastric isthmus stem cell",
            "",
            "stomach",
            "STMN1,MKI67,SOX9",
        ),
        N("g_cell", "G cell", "", "stomach", "GAST,CHGA"),
    ],
)

# --------------------------------------------------------------------------- #
# Squamous / mucosal squamous (cervix, oesophagus, oral, skin adnexa)
# --------------------------------------------------------------------------- #
SQUAMOUS = N(
    "squamous_epithelial_cell",
    "Squamous epithelial cell",
    "CL:0000076",
    "cervix,esophagus,oral_mucosa,vagina,skin,anus",
    "KRT5,KRT14,TP63,KRT13",
    [
        N(
            "squamous_basal_cell",
            "Squamous basal cell",
            "",
            "cervix,esophagus,oral_mucosa,vagina",
            "KRT5,KRT14,TP63,COL17A1",
        ),
        N(
            "squamous_parabasal_cell",
            "Squamous parabasal cell",
            "",
            "cervix,esophagus,vagina",
            "KRT5,KRT6A,MKI67,SERPINB3",
        ),
        N(
            "squamous_intermediate_cell",
            "Squamous intermediate cell",
            "",
            "cervix,esophagus,vagina",
            "KRT13,KRT4,CRNN,MAL",
        ),
        N(
            "squamous_superficial_cell",
            "Squamous superficial cell",
            "",
            "cervix,esophagus,vagina",
            "SPRR3,CRNN,IVL,TGM3",
        ),
        N(
            "reserve_cell_cervix",
            "Cervical reserve cell",
            "",
            "cervix",
            "KRT5,TP63,KRT17",
        ),
        N(
            "squamocolumnar_junction_cell",
            "Squamocolumnar junction cell",
            "",
            "cervix",
            "KRT7,AGR2,CK17,MMP7",
        ),
        N(
            "metaplastic_squamous_cell",
            "Metaplastic squamous cell",
            "",
            "cervix,bronchus,bladder",
            "KRT5,KRT17,SPRR1B,S100A2",
        ),
    ],
)

KERATINOCYTE = N(
    "keratinocyte",
    "Keratinocyte",
    "CL:0000312",
    "skin",
    "KRT1,KRT10,KRT14,KRT5",
    [
        N(
            "basal_keratinocyte",
            "Basal keratinocyte",
            "CL:0002187",
            "skin",
            "KRT14,KRT5,COL17A1,ITGA6",
        ),
        N("spinous_keratinocyte", "Spinous keratinocyte", "", "skin", "KRT1,KRT10,DSP"),
        N("granular_keratinocyte", "Granular keratinocyte", "", "skin", "FLG,LOR,IVL"),
        N(
            "cornified_keratinocyte",
            "Cornified keratinocyte",
            "",
            "skin",
            "LOR,FLG2,CDSN",
        ),
        N(
            "hair_follicle_keratinocyte",
            "Hair follicle keratinocyte",
            "",
            "skin",
            "KRT75,SOX9,LGR5,KRT71",
        ),
        N(
            "sebaceous_gland_cell",
            "Sebaceous gland cell",
            "CL:0002337",
            "skin",
            "PLIN2,AWAT2,FADS2,SOX9",
        ),
        N(
            "sweat_gland_cell",
            "Sweat gland cell",
            "",
            "skin",
            "DCD,SCGB2A2,MUCL1,KRT77",
        ),
        N(
            "melanocyte_placeholder",
            "Melanocyte (neural-crest, cross-referenced)",
            "CL:0000148",
            "skin,eye,mucosa",
            "MLANA,PMEL,TYRP1,DCT",
        ),
    ],
)

# --------------------------------------------------------------------------- #
# Glandular / secretory epithelium (breast, prostate, salivary, endometrium)
# --------------------------------------------------------------------------- #
GLANDULAR = N(
    "glandular_epithelial_cell",
    "Glandular epithelial cell",
    "CL:0000150",
    PAN,
    "EPCAM,KRT8,KRT18,ELF3",
    [
        N(
            "luminal_epithelial_cell",
            "Luminal epithelial cell",
            "CL:0002326",
            "breast,prostate,salivary_gland",
            "KRT8,KRT18,ELF5,AR",
            [
                N(
                    "luminal_hormone_sensing_cell",
                    "Luminal hormone-sensing cell",
                    "",
                    "breast",
                    "ESR1,PGR,ANKRD30A,TPD52L1",
                ),
                N(
                    "luminal_progenitor_cell",
                    "Luminal progenitor cell",
                    "",
                    "breast",
                    "KIT,ELF5,SLPI,LTF",
                ),
                N(
                    "prostate_luminal_cell",
                    "Prostate luminal cell",
                    "",
                    "prostate",
                    "KLK3,KLK2,MSMB,ACPP",
                ),
                N(
                    "prostate_club_cell",
                    "Prostate club cell",
                    "",
                    "prostate",
                    "SCGB1A1,SCGB3A1,PIGR,LTF",
                ),
            ],
        ),
        N(
            "myoepithelial_cell",
            "Myoepithelial cell",
            "CL:0000185",
            "breast,salivary_gland,sweat_gland,prostate",
            "ACTA2,KRT14,MYLK,OXTR",
        ),
        N(
            "ductal_epithelial_cell",
            "Ductal epithelial cell",
            "CL:0000068",
            "breast,pancreas,salivary_gland,prostate",
            "KRT19,KRT7,SPP1,MMP7",
        ),
        N(
            "acinar_cell",
            "Acinar cell",
            "CL:0000622",
            "pancreas,salivary_gland,lacrimal_gland",
            "PRSS1,CTRB1,CPA1,AMY2A",
        ),
        N(
            "serous_acinar_cell",
            "Serous acinar cell",
            "",
            "salivary_gland,lacrimal_gland",
            "ZG16B,PRR4,LYZ,AZGP1",
        ),
        N(
            "mucous_acinar_cell",
            "Mucous acinar cell",
            "",
            "salivary_gland",
            "MUC5B,BPIFB2,TFF3",
        ),
        N(
            "endometrial_epithelial_cell",
            "Endometrial epithelial cell",
            "",
            "uterus,endometrium",
            "EPCAM,PAX8,FOXA2,MUC1",
            [
                N(
                    "endometrial_ciliated_cell",
                    "Endometrial ciliated cell",
                    "",
                    "uterus,endometrium",
                    "FOXJ1,PIFO,TPPP3",
                ),
                N(
                    "endometrial_secretory_cell",
                    "Endometrial secretory cell",
                    "",
                    "uterus,endometrium",
                    "PAEP,GPX3,SCGB2A1",
                ),
                N(
                    "endometrial_glandular_cell",
                    "Endometrial glandular cell",
                    "",
                    "uterus,endometrium",
                    "MUC1,EPCAM,SOX17",
                ),
            ],
        ),
        N(
            "endocervical_columnar_cell",
            "Endocervical columnar cell",
            "",
            "cervix",
            "KRT7,MUC5B,TFF3,PAX8",
        ),
        N(
            "endocervical_ciliated_cell",
            "Endocervical ciliated cell",
            "",
            "cervix",
            "FOXJ1,TPPP3,PIFO",
        ),
        N(
            "fallopian_tube_epithelial_cell",
            "Fallopian tube epithelial cell",
            "",
            "fallopian_tube",
            "PAX8,OVGP1,FOXJ1",
        ),
        N(
            "thyroid_follicular_cell",
            "Thyroid follicular cell",
            "CL:0002258",
            "thyroid",
            "TG,TPO,TSHR,PAX8",
        ),
        N(
            "parathyroid_chief_cell",
            "Parathyroid chief cell",
            "",
            "parathyroid",
            "PTH,GCM2,CASR",
        ),
        N(
            "mesothelial_cell",
            "Mesothelial cell",
            "CL:0000077",
            "pleura,peritoneum,pericardium",
            "MSLN,UPK3B,KRT19,WT1",
        ),
    ],
)

# --------------------------------------------------------------------------- #
# Renal and urothelial
# --------------------------------------------------------------------------- #
RENAL = N(
    "renal_epithelial_cell",
    "Renal epithelial cell",
    "CL:0002518",
    "kidney",
    "PAX8,EPCAM,CDH16",
    [
        N(
            "proximal_tubule_cell",
            "Proximal tubule epithelial cell",
            "CL:0002306",
            "kidney",
            "LRP2,CUBN,SLC34A1,MIOX",
        ),
        N(
            "loop_of_henle_cell",
            "Loop of Henle epithelial cell",
            "",
            "kidney",
            "UMOD,SLC12A1,CLDN10",
        ),
        N(
            "distal_convoluted_tubule_cell",
            "Distal convoluted tubule cell",
            "CL:1000849",
            "kidney",
            "SLC12A3,TRPM6,CALB1",
        ),
        N(
            "connecting_tubule_cell",
            "Connecting tubule cell",
            "",
            "kidney",
            "SLC8A1,CALB1,AQP2",
        ),
        N(
            "principal_cell_kidney",
            "Collecting duct principal cell",
            "CL:1000714",
            "kidney",
            "AQP2,AQP3,SCNN1G",
        ),
        N(
            "intercalated_cell_alpha",
            "Type A intercalated cell",
            "CL:0005010",
            "kidney",
            "SLC4A1,ATP6V0D2,AQP6",
        ),
        N(
            "intercalated_cell_beta",
            "Type B intercalated cell",
            "CL:0005011",
            "kidney",
            "SLC26A4,ATP6V1B1,INSRR",
        ),
        N("podocyte", "Podocyte", "CL:0000653", "kidney", "NPHS1,NPHS2,WT1,PODXL"),
        N(
            "parietal_epithelial_cell_kidney",
            "Glomerular parietal epithelial cell",
            "",
            "kidney",
            "CLDN1,VCAM1,CFH",
        ),
        N(
            "urothelial_cell",
            "Urothelial cell",
            "CL:0000731",
            "bladder,ureter,renal_pelvis",
            "UPK1B,UPK2,UPK3A,KRT20",
            [
                N(
                    "urothelial_basal_cell",
                    "Urothelial basal cell",
                    "",
                    "bladder,ureter",
                    "KRT5,KRT14,TP63",
                ),
                N(
                    "urothelial_intermediate_cell",
                    "Urothelial intermediate cell",
                    "",
                    "bladder",
                    "KRT8,KRT18,UPK1B",
                ),
                N(
                    "urothelial_umbrella_cell",
                    "Urothelial umbrella cell",
                    "",
                    "bladder",
                    "UPK2,UPK3A,KRT20",
                ),
            ],
        ),
    ],
)

# --------------------------------------------------------------------------- #
# Hepatobiliary and pancreatic islet
# --------------------------------------------------------------------------- #
HEPATOBILIARY = N(
    "hepatobiliary_epithelial_cell",
    "Hepatobiliary epithelial cell",
    "",
    "liver,gallbladder,bile_duct",
    "ALB,KRT19,HNF4A",
    [
        N(
            "hepatocyte",
            "Hepatocyte",
            "CL:0000182",
            "liver",
            "ALB,APOA1,TTR,HNF4A",
            [
                N(
                    "periportal_hepatocyte",
                    "Periportal hepatocyte",
                    "",
                    "liver",
                    "ALB,SDS,ASS1,CPS1",
                ),
                N(
                    "pericentral_hepatocyte",
                    "Pericentral hepatocyte",
                    "",
                    "liver",
                    "CYP2E1,GLUL,CYP3A4,OAT",
                ),
            ],
        ),
        N(
            "cholangiocyte",
            "Cholangiocyte",
            "CL:1000488",
            "liver,bile_duct,gallbladder",
            "KRT19,KRT7,SOX9,EPCAM",
        ),
        N(
            "hepatic_progenitor_cell",
            "Hepatic progenitor cell",
            "",
            "liver",
            "SOX9,KRT19,PROM1",
        ),
    ],
)

ISLET = N(
    "pancreatic_islet_cell",
    "Pancreatic islet cell",
    "CL:0000008",
    "pancreas",
    "CHGA,CHGB,SCG2",
    [
        N(
            "beta_cell",
            "Pancreatic beta cell",
            "CL:0000169",
            "pancreas",
            "INS,IAPP,NKX6-1,MAFA",
        ),
        N(
            "alpha_cell",
            "Pancreatic alpha cell",
            "CL:0000171",
            "pancreas",
            "GCG,ARX,TTR,IRX2",
        ),
        N(
            "delta_cell",
            "Pancreatic delta cell",
            "CL:0000173",
            "pancreas",
            "SST,HHEX,LEPR",
        ),
        N(
            "pp_cell",
            "Pancreatic PP (gamma) cell",
            "CL:0002275",
            "pancreas",
            "PPY,SERTM1,ETV1",
        ),
        N(
            "epsilon_cell",
            "Pancreatic epsilon cell",
            "CL:0005019",
            "pancreas",
            "GHRL,ACSL1",
        ),
    ],
)

ENDOCRINE_OTHER = N(
    "endocrine_epithelial_cell",
    "Other endocrine epithelial cell",
    "CL:0000163",
    PAN,
    "CHGA,SCG2",
    [
        N(
            "adrenal_cortical_cell",
            "Adrenal cortical cell",
            "",
            "adrenal_gland",
            "CYP11B1,CYP17A1,STAR,NR5A1",
        ),
        N(
            "chromaffin_cell",
            "Adrenal chromaffin cell",
            "CL:0000166",
            "adrenal_gland",
            "PNMT,CHGA,TH,DBH",
        ),
        N(
            "pituitary_endocrine_cell",
            "Pituitary endocrine cell",
            "",
            "pituitary",
            "POU1F1,PRL,GH1,POMC",
        ),
        N("pinealocyte", "Pinealocyte", "", "pineal_gland", "TPH1,AANAT,ASMT"),
    ],
)

# --------------------------------------------------------------------------- #
# Thymic and other specialised epithelium
# --------------------------------------------------------------------------- #
SPECIALISED = N(
    "specialised_epithelial_cell",
    "Specialised epithelial cell",
    "",
    PAN,
    "EPCAM,KRT8",
    [
        N(
            "thymic_epithelial_cell",
            "Thymic epithelial cell",
            "CL:0002293",
            "thymus",
            "KRT8,PSMB11,FOXN1",
            [
                N(
                    "cortical_thymic_epithelial_cell",
                    "Cortical thymic epithelial cell",
                    "CL:0002364",
                    "thymus",
                    "PSMB11,PRSS16,CCL25",
                ),
                N(
                    "medullary_thymic_epithelial_cell",
                    "Medullary thymic epithelial cell",
                    "CL:0002365",
                    "thymus",
                    "AIRE,KRT14,CCL19,FEZF2",
                ),
                N(
                    "thymic_tuft_cell",
                    "Thymic tuft cell",
                    "",
                    "thymus",
                    "POU2F3,TRPM5,AVIL",
                ),
            ],
        ),
        N(
            "choroid_plexus_epithelial_cell",
            "Choroid plexus epithelial cell",
            "",
            "brain",
            "TTR,FOLR1,PRLR,AQP1",
        ),
        N(
            "retinal_pigment_epithelial_cell",
            "Retinal pigment epithelial cell",
            "CL:0002586",
            "eye,retina",
            "RPE65,BEST1,TYR,MLANA",
        ),
        N(
            "corneal_epithelial_cell",
            "Corneal epithelial cell",
            "CL:0000575",
            "eye,cornea",
            "KRT12,KRT3,PAX6",
        ),
        N(
            "lens_epithelial_cell",
            "Lens epithelial cell",
            "",
            "eye,lens",
            "CRYAA,CRYBB1,MIP",
        ),
        N(
            "taste_receptor_cell",
            "Taste receptor cell",
            "",
            "tongue,oral_mucosa",
            "TAS1R3,PLCB2,SNAP25",
        ),
        N(
            "olfactory_epithelial_cell",
            "Olfactory epithelial cell",
            "",
            "nasal,olfactory_mucosa",
            "OMP,GNG13,CNGA2",
        ),
        N(
            "sertoli_cell_placeholder",
            "Sertoli cell (gonadal-support, cross-referenced)",
            "CL:0000216",
            "testis",
            "SOX9,AMH,CLDN11,WT1",
        ),
    ],
)

EPITHELIAL = N(
    "epithelial",
    "Epithelial cell",
    "CL:0000066",
    PAN,
    "EPCAM,CDH1,KRT8,KRT18",
    [
        AIRWAY,
        ALVEOLAR,
        INTESTINAL,
        GASTRIC,
        SQUAMOUS,
        KERATINOCYTE,
        GLANDULAR,
        RENAL,
        HEPATOBILIARY,
        ISLET,
        ENDOCRINE_OTHER,
        SPECIALISED,
    ],
)

BRANCH = [EPITHELIAL]
