"""
scanpy_params.py — the numeric parameters of the standard preprocessing path.

Why this exists
---------------
These values were inline literals spread across ``pipeline.py``, which made two of
them wrong to change safely:

* ``n_pcs=30`` appeared at three call sites in ``pipeline.py`` AND was mirrored by
  ``clustering.SILHOUETTE_N_PCS`` under the comment "matches the neighbors n_pcs=30".
  One value, four places: raising the PC count for neighbours while the silhouette
  score kept scoring 30 PCs would have silently compared embeddings that were no
  longer the same embedding.
* ``target_sum=1e4`` is the normalisation depth every downstream log1p/HVG/marker
  step assumes. It belongs next to the seed in the provenance manifest, not buried
  in a call.

Changing a value here changes it for the whole run and it is recorded in the run
manifest, so a run's numbers stay traceable to the parameters that produced them.
"""

from __future__ import annotations

# Counts-per-cell that every library is normalised to before log1p. 1e4 ("counts per
# 10k") is the scanpy convention; the HVG, marker and cell-level DE steps all assume
# a log1p-of-CP10K matrix.
NORMALIZE_TARGET_SUM: float = 1e4

# Ceiling applied when z-scaling the HVG matrix. Caps the influence of a handful of
# extreme cells on PCA without dropping them.
SCALE_MAX_VALUE: float = 10.0

# Principal components computed. 50 is scanpy's default and comfortably more than the
# NEIGHBORS_N_PCS actually consumed downstream, so the elbow stays visible in the
# variance-explained plot.
PCA_N_COMPS: int = 50

# PCs used to build the kNN graph. Also the number scored by the silhouette metric in
# clustering.py — the two MUST agree or the score describes a different embedding than
# the clustering does.
NEIGHBORS_N_PCS: int = 30

# Neighbours per cell in the kNN graph feeding UMAP/t-SNE/Leiden.
NEIGHBORS_N_NEIGHBORS: int = 15

__all__ = [
    "NORMALIZE_TARGET_SUM",
    "SCALE_MAX_VALUE",
    "PCA_N_COMPS",
    "NEIGHBORS_N_PCS",
    "NEIGHBORS_N_NEIGHBORS",
]
