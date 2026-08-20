#!/usr/bin/env Rscript
# singler_bridge.R — Annotator B (SingleR) subprocess bridge.
#
# Reads a log-normalized expression matrix (genes x cells, MatrixMarket), a gene
# list, and a per-cell (barcode,cluster) table. Runs SingleR against a celldex
# reference chosen by NAME (tissue-driven, disease-agnostic) in per-cluster mode,
# and writes cluster,label,score to --out.
#
# Fails loudly: any error calls stop(), which exits non-zero so the Python caller
# captures stderr (no silent fallback). Chosen reference must cover non-immune
# lineages (BlueprintEncode / HumanPrimaryCellAtlas do).
#
# Usage:
#   Rscript singler_bridge.R --matrix expr.mtx --genes genes.txt \
#       --clusters clusters.csv --reference BlueprintEncodeData --out labels.csv

suppressWarnings(suppressMessages({
  ok <- requireNamespace("Matrix", quietly = TRUE) &&
        requireNamespace("SingleR", quietly = TRUE) &&
        requireNamespace("celldex", quietly = TRUE) &&
        requireNamespace("SummarizedExperiment", quietly = TRUE)
}))
if (!ok) {
  stop("Missing R packages. Install with: BiocManager::install(c('SingleR','celldex','SummarizedExperiment')) and install.packages('Matrix')")
}

# ---- minimal --flag value argument parser ---------------------------------
parse_args <- function(args) {
  out <- list()
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (startsWith(key, "--")) {
      name <- substring(key, 3)
      val <- if (i + 1 <= length(args)) args[[i + 1]] else ""
      out[[name]] <- val
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
  out
}

argv <- parse_args(commandArgs(trailingOnly = TRUE))
for (req in c("matrix", "genes", "clusters", "reference", "out")) {
  if (is.null(argv[[req]])) stop(sprintf("missing required --%s", req))
}

message(sprintf("[R] reference=%s", argv$reference))

# ---- load reference by name (disease-agnostic; tissue-driven) -------------
ref_fun <- switch(
  argv$reference,
  "BlueprintEncodeData"        = celldex::BlueprintEncodeData,
  "HumanPrimaryCellAtlasData"  = celldex::HumanPrimaryCellAtlasData,
  "MonacoImmuneData"           = celldex::MonacoImmuneData,
  "DatabaseImmuneCellExpressionData" = celldex::DatabaseImmuneCellExpressionData,
  "NovershternHematopoieticData"     = celldex::NovershternHematopoieticData,
  "MouseRNAseqData"                  = celldex::MouseRNAseqData,
  stop(sprintf("Unknown reference '%s'.", argv$reference))
)
ref <- ref_fun()

# ---- load test matrix (genes x cells) -------------------------------------
mat <- Matrix::readMM(argv$matrix)
genes <- readLines(argv$genes)
clusters_df <- utils::read.csv(argv$clusters, stringsAsFactors = FALSE)
if (!all(c("barcode", "cluster") %in% colnames(clusters_df))) {
  stop("clusters csv must have columns: barcode, cluster")
}
if (nrow(mat) != length(genes)) {
  stop(sprintf("gene count mismatch: matrix rows=%d, genes=%d", nrow(mat), length(genes)))
}
if (ncol(mat) != nrow(clusters_df)) {
  stop(sprintf("cell count mismatch: matrix cols=%d, clusters=%d", ncol(mat), nrow(clusters_df)))
}
rownames(mat) <- genes
colnames(mat) <- clusters_df$barcode

# ---- per-cluster SingleR --------------------------------------------------
labels_main <- ref$label.main
pred <- SingleR::SingleR(
  test = mat,
  ref = ref,
  labels = labels_main,
  clusters = clusters_df$cluster
)

score_vec <- tryCatch(
  apply(pred$scores, 1, function(r) max(r, na.rm = TRUE)),
  error = function(e) rep(NA_real_, nrow(pred))
)
final_labels <- pred$labels
if (!is.null(pred$pruned.labels)) {
  # prefer pruned labels where available, else fall back to raw
  pl <- pred$pruned.labels
  final_labels <- ifelse(is.na(pl), pred$labels, pl)
}

out_df <- data.frame(
  cluster = rownames(pred),
  label = final_labels,
  score = as.numeric(score_vec),
  stringsAsFactors = FALSE
)
utils::write.csv(out_df, argv$out, row.names = FALSE)
message(sprintf("[R] wrote %d cluster labels to %s", nrow(out_df), argv$out))
