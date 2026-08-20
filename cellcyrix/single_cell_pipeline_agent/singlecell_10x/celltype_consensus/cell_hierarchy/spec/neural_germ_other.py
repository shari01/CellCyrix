"""Neural/glial, germ-cell/placental, and an explicit unassigned lineage."""

from ._dsl import N

PAN = "pan_tissue"

# --------------------------------------------------------------------------- #
# Neural / glial
# --------------------------------------------------------------------------- #
NEURON = N(
    "neuron",
    "Neuron",
    "CL:0000540",
    "brain,spinal_cord,retina,gut,ganglion",
    "RBFOX3,SNAP25,SYT1,ELAVL3",
    [
        N(
            "excitatory_neuron",
            "Excitatory (glutamatergic) neuron",
            "CL:0000679",
            "brain,spinal_cord",
            "SLC17A7,SATB2,NEUROD6",
        ),
        N(
            "inhibitory_neuron",
            "Inhibitory (GABAergic) neuron",
            "CL:0000617",
            "brain,spinal_cord",
            "GAD1,GAD2,SLC32A1",
            [
                N(
                    "pvalb_interneuron",
                    "PVALB-positive interneuron",
                    "",
                    "brain",
                    "PVALB,GAD1,ERBB4",
                ),
                N(
                    "sst_interneuron",
                    "SST-positive interneuron",
                    "",
                    "brain",
                    "SST,GAD1,NPY",
                ),
                N(
                    "vip_interneuron",
                    "VIP-positive interneuron",
                    "",
                    "brain",
                    "VIP,GAD1,CALB2",
                ),
            ],
        ),
        N(
            "dopaminergic_neuron",
            "Dopaminergic neuron",
            "CL:0000700",
            "brain",
            "TH,SLC6A3,DDC",
        ),
        N("serotonergic_neuron", "Serotonergic neuron", "", "brain", "TPH2,SLC6A4,FEV"),
        N(
            "cholinergic_neuron",
            "Cholinergic neuron",
            "CL:0000108",
            "brain,spinal_cord,gut",
            "CHAT,SLC5A7,ACHE",
        ),
        N(
            "motor_neuron",
            "Motor neuron",
            "CL:0000100",
            "spinal_cord,brainstem",
            "CHAT,MNX1,ISL1",
        ),
        N(
            "sensory_neuron",
            "Sensory neuron",
            "CL:0000101",
            "ganglion,skin",
            "PRPH,NTRK1,SCN10A",
        ),
        N(
            "enteric_neuron",
            "Enteric neuron",
            "",
            "gut,small_intestine,colon",
            "ELAVL4,PHOX2B,RET,NOS1",
        ),
        N(
            "granule_cell",
            "Cerebellar granule cell",
            "",
            "brain,cerebellum",
            "GABRA6,RBFOX3,CBLN3",
        ),
        N(
            "purkinje_cell",
            "Purkinje cell",
            "CL:0000121",
            "brain,cerebellum",
            "CALB1,PCP2,ITPR1",
        ),
        N(
            "photoreceptor_cell",
            "Photoreceptor cell",
            "CL:0000210",
            "eye,retina",
            "RCVRN,RHO,ARR3",
            [
                N(
                    "rod_photoreceptor",
                    "Rod photoreceptor",
                    "CL:0000604",
                    "eye,retina",
                    "RHO,NRL,GNAT1",
                ),
                N(
                    "cone_photoreceptor",
                    "Cone photoreceptor",
                    "CL:0000573",
                    "eye,retina",
                    "ARR3,OPN1SW,GNAT2",
                ),
            ],
        ),
        N(
            "retinal_ganglion_cell",
            "Retinal ganglion cell",
            "CL:0000740",
            "eye,retina",
            "RBPMS,NEFL,SNCG",
        ),
        N(
            "bipolar_cell_retina",
            "Retinal bipolar cell",
            "CL:0000748",
            "eye,retina",
            "VSX2,GRIK1,TRPM1",
        ),
        N(
            "amacrine_cell",
            "Amacrine cell",
            "CL:0000561",
            "eye,retina",
            "GAD1,TFAP2B,SLC6A9",
        ),
        N(
            "horizontal_cell",
            "Horizontal cell",
            "CL:0000745",
            "eye,retina",
            "ONECUT1,LHX1,CALB1",
        ),
    ],
)

GLIA = N(
    "glial_cell",
    "Glial cell",
    "CL:0000125",
    "brain,spinal_cord,retina,gut,nerve",
    "S100B,PLP1,GFAP",
    [
        N(
            "astrocyte",
            "Astrocyte",
            "CL:0000127",
            "brain,spinal_cord,retina",
            "GFAP,AQP4,SLC1A2,ALDH1L1",
            [
                N(
                    "protoplasmic_astrocyte",
                    "Protoplasmic astrocyte",
                    "",
                    "brain",
                    "SLC1A2,GJA1,MFGE8",
                ),
                N(
                    "fibrous_astrocyte",
                    "Fibrous astrocyte",
                    "",
                    "brain,spinal_cord",
                    "GFAP,AQP4,CD44",
                ),
                N(
                    "reactive_astrocyte",
                    "Reactive astrocyte",
                    "",
                    "brain,spinal_cord",
                    "GFAP,VIM,SERPINA3,C3",
                ),
            ],
        ),
        N(
            "oligodendrocyte",
            "Oligodendrocyte",
            "CL:0000128",
            "brain,spinal_cord",
            "MBP,PLP1,MOG,MAG",
        ),
        N(
            "oligodendrocyte_precursor_cell",
            "Oligodendrocyte precursor cell",
            "CL:0002453",
            "brain,spinal_cord",
            "PDGFRA,CSPG4,OLIG1,SOX10",
        ),
        N(
            "ependymal_cell",
            "Ependymal cell",
            "CL:0000065",
            "brain,spinal_cord",
            "FOXJ1,PIFO,HDC,TMEM212",
        ),
        N(
            "schwann_cell",
            "Schwann cell",
            "CL:0002573",
            "nerve,skin,gut",
            "S100B,PLP1,MPZ,SOX10",
            [
                N(
                    "myelinating_schwann_cell",
                    "Myelinating Schwann cell",
                    "CL:0000192",
                    "nerve",
                    "MPZ,MBP,PMP22",
                ),
                N(
                    "non_myelinating_schwann_cell",
                    "Non-myelinating Schwann cell",
                    "",
                    "nerve,skin",
                    "NGFR,L1CAM,SCN7A",
                ),
            ],
        ),
        N(
            "enteric_glial_cell",
            "Enteric glial cell",
            "",
            "gut,small_intestine,colon",
            "S100B,PLP1,SOX10,GFAP",
        ),
        N(
            "satellite_glial_cell",
            "Satellite glial cell",
            "",
            "ganglion",
            "FABP7,APOE,S100B",
        ),
        N(
            "muller_glial_cell",
            "Müller glial cell",
            "CL:0000636",
            "eye,retina",
            "RLBP1,GLUL,SLC1A3,VIM",
        ),
        N(
            "bergmann_glial_cell",
            "Bergmann glial cell",
            "",
            "brain,cerebellum",
            "GDF10,AQP4,SLC1A3",
        ),
        N("tanycyte", "Tanycyte", "", "brain,hypothalamus", "RAX,DIO2,CRYM"),
        N("pituicyte", "Pituicyte", "", "pituitary", "S100B,GFAP,COL25A1"),
    ],
)

NEURAL_CREST = N(
    "neural_crest_derived_cell",
    "Neural-crest-derived cell",
    "",
    "skin,eye,mucosa,adrenal_gland",
    "SOX10,MITF",
    [
        N(
            "melanocyte",
            "Melanocyte",
            "CL:0000148",
            "skin,eye,mucosa",
            "MLANA,PMEL,TYRP1,DCT,MITF",
        ),
        N("merkel_cell", "Merkel cell", "CL:0000242", "skin", "KRT20,SOX2,CHGA,ATOH1"),
        N("glomus_cell", "Glomus (type I) cell", "", "carotid_body", "TH,CHGA,DBH"),
    ],
)

NEURAL = N(
    "neural",
    "Neural cell",
    "CL:0002319",
    "brain,spinal_cord,retina,gut,nerve,ganglion",
    "SNAP25,PLP1,SOX10",
    [
        NEURON,
        GLIA,
        NEURAL_CREST,
        N(
            "neural_progenitor_cell",
            "Neural progenitor cell",
            "CL:0011020",
            "brain,spinal_cord",
            "SOX2,NES,PAX6,VIM",
        ),
        N(
            "radial_glial_cell",
            "Radial glial cell",
            "CL:0000681",
            "brain",
            "VIM,HES1,SLC1A3,SOX2",
        ),
    ],
)

# --------------------------------------------------------------------------- #
# Germ cell and placental
# --------------------------------------------------------------------------- #
GERM = N(
    "germ_cell",
    "Germ cell",
    "CL:0000586",
    "testis,ovary",
    "DDX4,DAZL,MAGEA4",
    [
        N(
            "spermatogonium",
            "Spermatogonium",
            "CL:0000020",
            "testis",
            "UTF1,FGFR3,MAGEA4,ID4",
        ),
        N("spermatocyte", "Spermatocyte", "CL:0000017", "testis", "SYCP3,SPO11,PIWIL1"),
        N("spermatid", "Spermatid", "CL:0000018", "testis", "PRM1,PRM2,TNP1"),
        N(
            "spermatozoon",
            "Spermatozoon",
            "CL:0000019",
            "testis,semen",
            "PRM1,AKAP4,ODF1",
        ),
        N("oocyte", "Oocyte", "CL:0000023", "ovary", "ZP3,GDF9,FIGLA,NLRP7"),
    ],
)

GONADAL_SUPPORT = N(
    "gonadal_somatic_cell",
    "Gonadal somatic support cell",
    "",
    "testis,ovary",
    "GATA4,NR5A1",
    [
        N(
            "sertoli_cell",
            "Sertoli cell",
            "CL:0000216",
            "testis",
            "SOX9,AMH,CLDN11,WT1",
        ),
        N(
            "leydig_cell",
            "Leydig cell",
            "CL:0000178",
            "testis",
            "INSL3,CYP17A1,STAR,HSD3B1",
        ),
        N(
            "granulosa_cell",
            "Granulosa cell",
            "CL:0000501",
            "ovary",
            "AMH,FOXL2,CYP19A1,INHA",
        ),
        N("theca_cell", "Theca cell", "CL:0000503", "ovary", "CYP17A1,STAR,TCF21"),
        N(
            "peritubular_myoid_cell",
            "Peritubular myoid cell",
            "",
            "testis",
            "ACTA2,MYH11,PDGFRB",
        ),
    ],
)

PLACENTAL = N(
    "placental_cell",
    "Placental and decidual cell",
    "",
    "placenta,decidua,uterus",
    "KRT7,GATA3,TFAP2C",
    [
        N(
            "cytotrophoblast",
            "Cytotrophoblast",
            "CL:0000351",
            "placenta",
            "PAGE4,PEG10,TP63,ITGA6",
        ),
        N(
            "syncytiotrophoblast",
            "Syncytiotrophoblast",
            "CL:0000525",
            "placenta",
            "CGA,CGB3,CSH1,ERVFRD-1",
        ),
        N(
            "extravillous_trophoblast",
            "Extravillous trophoblast",
            "CL:0008036",
            "placenta,decidua",
            "HLA-G,ITGA5,PAPPA2,MMP2",
        ),
        N(
            "decidual_stromal_cell",
            "Decidual stromal cell",
            "",
            "decidua,uterus",
            "IGFBP1,PRL,DKK1,LUM",
        ),
        N("hofbauer_cell", "Hofbauer cell", "", "placenta", "CD163,LYVE1,F13A1,CD14"),
    ],
)

GERM_PLACENTAL = N(
    "germ_placental",
    "Germ-cell and placental lineage",
    "",
    "testis,ovary,placenta,decidua",
    "DDX4,KRT7",
    [GERM, GONADAL_SUPPORT, PLACENTAL],
)

# --------------------------------------------------------------------------- #
# Explicit unassigned lineage (never a silent drop)
# --------------------------------------------------------------------------- #
UNASSIGNED = N(
    "unassigned",
    "Unassigned / not resolvable to a lineage",
    "",
    PAN,
    "",
    [
        N("unknown_cell", "Unknown cell", "", PAN, ""),
        N("mixed_or_ambiguous_cell", "Mixed or ambiguous label", "", PAN, ""),
        N("technical_artefact", "Technical artefact label", "", PAN, ""),
    ],
)

BRANCH = [NEURAL, GERM_PLACENTAL, UNASSIGNED]
