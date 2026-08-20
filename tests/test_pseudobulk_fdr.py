"""
test_pseudobulk_fdr.py — does the donor-level DE actually control what it claims to?

The existing smoke test proves DIRECTION: genes simulated as up in CASE come out with a
positive log2 fold change. That is necessary and not sufficient. The claim the pipeline
makes in a paper is stronger — that `p_value_adj < 0.05` means an expected false
discovery proportion of about 5% — and direction says nothing about it. A method can get
every sign right and still report ten times more significant genes than it should.

Three simulations, each with a known answer:

1. **Null-only.** No gene differs between arms. Anything called significant is a false
   positive, so the count should be ~0. This is the type-I error check, and it is where
   cell-level tests on the same data fail badly.
2. **Mixed.** A known set of true-DE genes among many nulls, repeated over several
   independent seeds. The false discovery proportion is averaged across replicates,
   because FDR bounds the *expectation* — judging it from one simulation would be
   testing sampling noise.
3. **Power.** Strongly-DE genes must actually be recovered, so a method that controls
   FDR by never calling anything cannot pass.

Counts are drawn from a negative binomial with a per-donor mean, which is what gives
donors real biological variability. That variability is the entire reason the pipeline
aggregates to the donor before testing, and a simulation without it would make
cell-level testing look fine.

Marked ``slow``: several DESeq2 fits, ~1-2 minutes total.

    pytest tests/test_pseudobulk_fdr.py -m slow
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.slow

#: Nominal FDR the pipeline is asked to control at.
ALPHA = 0.05

#: Donors per arm. 4v4 is a realistic small cohort and the regime where an
#: inflated-FDR method is most dangerous, because it still looks plausible.
N_DONORS_PER_ARM = 4

#: Cells per donor, aggregated away by the pseudobulk step.
N_CELLS_PER_DONOR = 60

#: Genes per simulation.
N_GENES = 600

#: Fraction of genes given a real effect in the mixed simulation.
TRUE_DE_FRACTION = 0.10

#: Fold change applied to the true-DE genes. Large enough to be detectable at n=4.
TRUE_LOG2_FC = 2.0

#: Biological variability between donors, as the negative binomial dispersion. Without
#: this the simulation has no donor effect and the test proves nothing.
DONOR_DISPERSION = 0.25

#: Independent seeds for the mixed simulation. FDR is an expectation, so it is estimated
#: by averaging the false discovery proportion across replicates.
N_REPLICATES = 5

#: Ceiling on the AVERAGE false discovery proportion in the mixed simulation.
#:
#: Deliberately NOT set at ALPHA, and the gap is not slack for Monte-Carlo error — it
#: is an honest characterisation. Measured on this simulator the mean FDP is ~0.11-0.17
#: against a nominal 0.05, and it does not converge with more genes (600 -> 5000) or
#: more donors (4 -> 12), so it is not a small-sample artifact.
#:
#: The cause is NOT the pipeline's adjustment step, which
#: :func:`test_adjustment_is_exactly_benjamini_hochberg` shows is exact to 4e-14, nor a
#: miscalibrated statistic, which :func:`test_null_pvalues_are_uniform` shows is uniform.
#: It is the far tail: null ``p < 0.01`` fires at ~1.5x its nominal rate at every donor
#: count tested. Two explanations remain live and this test cannot separate them —
#: DESeq2's Wald test using an asymptotic normal approximation on shrunken dispersions,
#: and this simulator drawing a CONSTANT dispersion that DESeq2's ``a/mean + b`` trend
#: cannot represent, so shrinkage toward a misspecified trend under-estimates dispersion
#: for some genes.
#:
#: So the ceiling exists to catch REGRESSION, not to certify nominal control. What is
#: safe to claim in a paper is what the other tests here establish: correct BH
#: application, a uniform null, ~0 discoveries when nothing is true, adequate power,
#: correct direction, and the donor as the unit of replication. Do not write "FDR
#: controlled at 5%" on the strength of this file.
FDP_CEILING = 0.30

#: Null p-values must pass a Kolmogorov-Smirnov uniformity test at this level. A
#: miscalibrated test statistic would make every downstream FDR claim meaningless.
UNIFORMITY_KS_ALPHA = 0.01

#: Tolerance for the pipeline's adjusted p-values against an independent BH computation.
#: Exact to floating point, so this is tight on purpose.
BH_AGREEMENT_TOL = 1e-9

#: A null-only run should call almost nothing. Expressed as a fraction of all genes.
MAX_NULL_ONLY_DISCOVERY_RATE = 0.02


def _simulate(
    *,
    seed: int,
    n_true_de: int,
    log2_fc: float = TRUE_LOG2_FC,
    n_genes: int = N_GENES,
    n_donors_per_arm: int = N_DONORS_PER_ARM,
    n_cells_per_donor: int = N_CELLS_PER_DONOR,
):
    """Build an AnnData cohort with a known set of differentially expressed genes.

    Each donor gets its own gene-wise mean drawn around the arm's mean, which is the
    donor effect. Cells within a donor are then Poisson draws around that donor mean, so
    within-donor variation is small relative to between-donor variation — the structure
    that makes cell-level testing anti-conservative and donor-level testing correct.

    Args:
        seed: Seed for the whole simulation.
        n_true_de: Number of genes given a real effect. 0 for a null-only cohort.
        log2_fc: Effect size applied to those genes in the CASE arm.
        n_genes: Total genes.
        n_donors_per_arm: Donors in each of CONTROL and CASE.
        n_cells_per_donor: Cells per donor.

    Returns:
        ``(adata, true_de_genes)`` — the cohort with raw integer counts in ``X`` and
        ``layers["counts"]``, and the set of gene names given a real effect.
    """
    import anndata

    rng = np.random.default_rng(seed)

    gene_names = [f"GENE{i:04d}" for i in range(n_genes)]
    true_de_index = rng.choice(n_genes, size=n_true_de, replace=False)
    true_de_genes = {gene_names[i] for i in true_de_index}

    # Baseline expression spread over several orders of magnitude, as real data is.
    base_mean = rng.lognormal(mean=2.0, sigma=1.0, size=n_genes)

    effect = np.ones(n_genes, dtype=float)
    effect[true_de_index] = 2.0**log2_fc

    blocks, obs_rows = [], []
    for arm, n_donors in (("CONTROL", n_donors_per_arm), ("CASE", n_donors_per_arm)):
        arm_mean = base_mean * (effect if arm == "CASE" else 1.0)
        for donor in range(n_donors):
            sample = f"{arm}_D{donor}"
            # Donor effect: a gamma multiplier per gene, giving negative-binomial
            # marginals with dispersion DONOR_DISPERSION.
            shape = 1.0 / DONOR_DISPERSION
            donor_mean = arm_mean * rng.gamma(shape, 1.0 / shape, size=n_genes)
            counts = rng.poisson(
                np.broadcast_to(donor_mean, (n_cells_per_donor, n_genes))
            ).astype(np.int32)
            blocks.append(counts)
            obs_rows += [
                {"sample": sample, "group": arm, "celltype_consensus": "T cell"}
            ] * n_cells_per_donor

    matrix = np.vstack(blocks)
    obs = pd.DataFrame(obs_rows, index=[f"cell_{i}" for i in range(matrix.shape[0])])
    var = pd.DataFrame(index=pd.Index(gene_names, name=None))

    adata = anndata.AnnData(X=matrix.astype(np.float32), obs=obs, var=var)
    # The DE step reads raw integer counts from this layer.
    adata.layers["counts"] = matrix.copy()
    return adata, true_de_genes


def _run_de(adata, out_dir: Path) -> pd.DataFrame:
    """Run the pipeline's pseudobulk DE and return the overall table.

    Args:
        adata: Simulated cohort.
        out_dir: Directory the DE step writes into.

    Returns:
        The parsed ``pseudobulk_overall_de.csv``.
    """
    from cellcyrix.single_cell_pipeline_agent.singlecell_10x.pseudobulk_de import (
        compute_pseudobulk_de,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    compute_pseudobulk_de(
        adata,
        group_col="group",
        sample_col="sample",
        out_dir=out_dir,
        reference_group="CONTROL",
        # lfc_threshold 0.0 so the p-values are tested against the null of NO change.
        # A non-zero threshold tests a different (composite) null, and its adjusted
        # p-values are not the quantity an FDR claim is about.
        lfc_threshold=0.0,
        alpha=ALPHA,
    )
    table = next(iter(sorted(out_dir.rglob("pseudobulk_overall_de.csv"))), None)
    assert table is not None, f"no pseudobulk_overall_de.csv written under {out_dir}"
    return pd.read_csv(table)


def _significant(frame: pd.DataFrame, alpha: float = ALPHA) -> set[str]:
    """Gene names called significant at `alpha` on the adjusted p-value."""
    padj = pd.to_numeric(frame["p_value_adj"], errors="coerce")
    return set(frame.loc[padj < alpha, "gene"].astype(str))


# --------------------------------------------------------------------------------------
# 1. Null-only: type-I error
# --------------------------------------------------------------------------------------


def test_null_only_cohort_yields_almost_no_discoveries(tmp_path):
    """With no true effect anywhere, almost nothing may be called significant.

    This is the assertion that separates donor-level testing from cell-level testing.
    Cell-level Wilcoxon on this same cohort treats each of the 480 cells as an
    independent replicate and reports a large fraction of the genome; the donor-level
    test sees n=4 per arm and correctly finds nothing.
    """
    adata, true_de = _simulate(seed=0, n_true_de=0)
    assert not true_de

    frame = _run_de(adata, tmp_path / "null_only")
    discoveries = _significant(frame)
    rate = len(discoveries) / len(frame)

    assert rate <= MAX_NULL_ONLY_DISCOVERY_RATE, (
        f"{len(discoveries)}/{len(frame)} genes ({rate:.1%}) called significant in a "
        f"cohort with NO true differences. Every one is a false positive; the nominal "
        f"FDR of {ALPHA} is not being controlled. Examples: "
        f"{sorted(discoveries)[:10]}"
    )


# --------------------------------------------------------------------------------------
# 2. Mixed: the FDR claim itself
# --------------------------------------------------------------------------------------


def test_null_pvalues_are_uniform(tmp_path):
    """Under a complete null the p-values must be Uniform(0, 1).

    The precondition for every FDR statement: Benjamini-Hochberg assumes a uniform null,
    so a p-value distribution that fails here invalidates the adjusted values regardless
    of how correctly the adjustment itself is computed.
    """
    from scipy.stats import kstest

    adata, _ = _simulate(seed=42, n_true_de=0, n_genes=2000)
    frame = _run_de(adata, tmp_path / "uniformity")

    pvalues = pd.to_numeric(frame["p_value"], errors="coerce").dropna().to_numpy()
    assert pvalues.size > 1000, f"only {pvalues.size} usable p-values"

    result = kstest(pvalues, "uniform")
    assert result.pvalue > UNIFORMITY_KS_ALPHA, (
        f"null p-values are not uniform (KS statistic {result.statistic:.4f}, "
        f"p={result.pvalue:.2e}). Benjamini-Hochberg assumes uniformity, so every "
        f"adjusted p-value the pipeline reports is unreliable. Observed "
        f"P(p<0.05)={float((pvalues < 0.05).mean()):.4f} (expect 0.05), "
        f"P(p<0.01)={float((pvalues < 0.01).mean()):.4f} (expect 0.01)."
    )


def test_adjustment_is_exactly_benjamini_hochberg(tmp_path):
    """The pipeline's ``p_value_adj`` is BH applied to its own ``p_value``, exactly.

    Separates the two things that can go wrong. If the null is uniform and this is
    exact, any residual FDP inflation is a property of the upstream test statistic, not
    of the pipeline's multiple-testing code — which is what makes the characterisation
    on :data:`FDP_CEILING` attributable rather than a mystery.
    """
    from statsmodels.stats.multitest import multipletests

    adata, _ = _simulate(seed=43, n_true_de=int(2000 * TRUE_DE_FRACTION), n_genes=2000)
    frame = _run_de(adata, tmp_path / "bh_exactness")

    usable = frame[
        pd.to_numeric(frame["p_value"], errors="coerce").notna()
        & pd.to_numeric(frame["p_value_adj"], errors="coerce").notna()
    ]
    pvalues = pd.to_numeric(usable["p_value"], errors="coerce").to_numpy(dtype=float)
    reported = pd.to_numeric(usable["p_value_adj"], errors="coerce").to_numpy(
        dtype=float
    )

    _, expected, _, _ = multipletests(pvalues, alpha=ALPHA, method="fdr_bh")
    max_difference = float(np.abs(reported - expected).max())

    assert max_difference <= BH_AGREEMENT_TOL, (
        f"reported p_value_adj differs from Benjamini-Hochberg on the same p-values by "
        f"up to {max_difference:.3e}. The multiple-testing correction is not the method "
        f"the output claims."
    )


def test_false_discovery_proportion_stays_within_the_characterised_ceiling(tmp_path):
    """Record the mean false discovery proportion and fail only on regression.

    Read :data:`FDP_CEILING` before interpreting this. The measured FDP on this
    simulator is well above the nominal 0.05 and does not converge with more genes or
    more donors, and this test cannot attribute that between DESeq2's small-n Wald tail
    and this simulator's constant-dispersion misspecification. It is therefore a
    regression guard, not evidence of nominal FDR control.

    Averaging over replicates is still required: FDR bounds E[FDP], so a single
    simulation's FDP is a noisy draw.
    """
    n_true_de = int(N_GENES * TRUE_DE_FRACTION)
    per_replicate = []

    for replicate in range(N_REPLICATES):
        adata, true_de = _simulate(seed=100 + replicate, n_true_de=n_true_de)
        frame = _run_de(adata, tmp_path / f"mixed_{replicate}")
        discoveries = _significant(frame)

        if not discoveries:
            # No discoveries means FDP is undefined (0/0); record 0 and let the power
            # test below catch a method that never calls anything.
            per_replicate.append(
                {"replicate": replicate, "n_discoveries": 0, "fdp": 0.0, "n_true": 0}
            )
            continue

        false_positives = discoveries - true_de
        per_replicate.append(
            {
                "replicate": replicate,
                "n_discoveries": len(discoveries),
                "n_true": len(discoveries & true_de),
                "fdp": len(false_positives) / len(discoveries),
            }
        )

    summary = pd.DataFrame(per_replicate)
    mean_fdp = float(summary["fdp"].mean())
    total_discoveries = int(summary["n_discoveries"].sum())

    assert total_discoveries > 0, (
        "no discoveries in any replicate; the FDP is undefined and the power test "
        "should also be failing"
    )
    assert mean_fdp <= FDP_CEILING, (
        f"mean false discovery proportion {mean_fdp:.3f} exceeds the characterised "
        f"ceiling {FDP_CEILING} (nominal FDR {ALPHA}). This is a REGRESSION signal: the "
        f"measured value was ~0.11-0.17 when this ceiling was set. Per-replicate:\n"
        f"{summary.to_string(index=False)}"
    )
    # Surfaced so a run of this file reports the number rather than only pass/fail —
    # it is the value to quote, with its caveat, if the paper discusses FDR behaviour.
    print(  # noqa: T201 - tests/** is exempt; this is a measurement, not logging
        f"\n[FDR] mean FDP {mean_fdp:.3f} over {N_REPLICATES} replicates "
        f"(nominal {ALPHA}); see FDP_CEILING for why these differ."
    )


def test_true_effects_are_recovered(tmp_path):
    """Power check: a method that controls FDR by calling nothing must not pass.

    Strong, unambiguous effects at n=4 per arm should be detected for a clear majority
    of the true-DE genes.
    """
    n_true_de = int(N_GENES * TRUE_DE_FRACTION)
    adata, true_de = _simulate(seed=7, n_true_de=n_true_de, log2_fc=3.0)
    frame = _run_de(adata, tmp_path / "power")

    discoveries = _significant(frame)
    recovered = discoveries & true_de
    sensitivity = len(recovered) / len(true_de)

    assert sensitivity >= 0.5, (
        f"only {len(recovered)}/{len(true_de)} truly-DE genes recovered "
        f"({sensitivity:.1%}) at log2FC=3.0 with {N_DONORS_PER_ARM} donors per arm. "
        f"The test is too conservative to be useful, so its FDR control is not "
        f"evidence of anything."
    )


def test_direction_of_recovered_effects_is_correct(tmp_path):
    """Recovered true-DE genes must be positive: they were simulated as up in CASE.

    Overlaps with the smoke test's direction control, kept here because an FDR result is
    meaningless if the sign convention silently inverts — the same discoveries would
    then be reported with the wrong biology.
    """
    n_true_de = int(N_GENES * TRUE_DE_FRACTION)
    adata, true_de = _simulate(seed=7, n_true_de=n_true_de, log2_fc=3.0)
    frame = _run_de(adata, tmp_path / "direction")

    recovered = frame[
        frame["gene"].astype(str).isin(_significant(frame) & true_de)
    ].copy()
    if recovered.empty:
        pytest.skip("no true-DE genes recovered; covered by the power test")

    lfc = pd.to_numeric(recovered["log2_fold_change"], errors="coerce")
    n_positive = int((lfc > 0).sum())
    assert n_positive == len(lfc), (
        f"{len(lfc) - n_positive}/{len(lfc)} recovered CASE-elevated genes have a "
        f"NEGATIVE log2 fold change. The contrast is inverted relative to "
        f"reference_group='CONTROL'."
    )


# --------------------------------------------------------------------------------------
# 3. The unit of replication is the donor, not the cell
# --------------------------------------------------------------------------------------


def test_more_cells_per_donor_does_not_inflate_significance(tmp_path):
    """Tripling cells per donor must not multiply discoveries on a null cohort.

    The signature of using the cell as the unit of replication: significance grows with
    sequencing depth rather than with evidence. Because the pipeline sums to the donor
    first, n stays at 4 per arm and the discovery count stays near zero regardless.
    """
    small, _ = _simulate(seed=11, n_true_de=0, n_cells_per_donor=40)
    large, _ = _simulate(seed=11, n_true_de=0, n_cells_per_donor=120)

    n_small = len(_significant(_run_de(small, tmp_path / "cells_40")))
    n_large = len(_significant(_run_de(large, tmp_path / "cells_120")))

    budget = int(N_GENES * MAX_NULL_ONLY_DISCOVERY_RATE)
    assert n_small <= budget and n_large <= budget, (
        f"null-cohort discoveries scale with cells per donor "
        f"({n_small} at 40 cells -> {n_large} at 120), which means the effective n is "
        f"the CELL count, not the donor count."
    )
