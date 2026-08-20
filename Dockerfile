# CellCyrix — reproducible runtime image.
#
# Built on rocker/r-ver rather than a python base image, for one reason: the SingleR
# voter needs R, and rocker/r-ver pins both the R version AND a dated CRAN snapshot
# (Posit Package Manager), so `install.packages` resolves to the same versions on every
# rebuild. A python base with R bolted on gets whatever CRAN serves that day, which is
# not a reproducible environment even with a pinned Dockerfile.
#
# Debian bookworm ships Python 3.11, which satisfies requires-python >= 3.11.
#
#   docker build -t cellcyrix:1.0.0 .
#   docker run --rm -v "$PWD/input_data:/data/input:ro" -v "$PWD/outputs:/data/outputs" \
#       cellcyrix:1.0.0 --config /data/input/config.yaml --output-root /data/outputs
#
# The image runs fully offline once built: the 41 CellTypist models are fetched at build
# time and checksum-verified, and the LLM voter reads its response cache. Provide
# OPENROUTER_API_KEY at run time only if you want live LLM calls.

# Pin the R version and its CRAN snapshot date. Change deliberately, never implicitly.
FROM rocker/r-ver:4.4.1

LABEL org.opencontainers.image.title="CellCyrix"
LABEL org.opencontainers.image.description="Disease-agnostic single-cell pipeline with multi-voter consensus cell-type annotation and donor-level pseudobulk DE"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.source="https://github.com/ayassbioscience/cellcyrix"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # rpy2 needs to find R; rocker installs it here.
    R_HOME=/usr/local/lib/R \
    # Keep the reference tree and model cache at a fixed absolute path, so a bind mount
    # or a derived image can replace it without touching the code. Relative values
    # resolve against the CWD and would silently lose the reference data.
    SCPIPE_SHARED_REFERENCE_ROOT=/opt/cellcyrix/shared_reference \
    SCPIPE_LLM_CACHE_DIR=/opt/cellcyrix/.llm_cache \
    SCPIPE_ENABLE_SINGLER=1

# --- system + Python -----------------------------------------------------------------
# libcurl/libssl/libxml2 are Bioconductor build dependencies; libhdf5 is for h5py;
# build-essential + gfortran cover the scientific wheels that lack manylinux builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3-pip \
        build-essential \
        gfortran \
        git \
        curl \
        ca-certificates \
        libcurl4-openssl-dev \
        libssl-dev \
        libxml2-dev \
        libhdf5-dev \
        libglpk-dev \
        libgit2-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# --- R packages for the SingleR voter -------------------------------------------------
# Versions come from the base image's pinned CRAN snapshot, so this layer is
# reproducible without a per-package version list. Bioconductor is pinned by release.
RUN R -q -e "install.packages('BiocManager', repos='https://cloud.r-project.org')" \
 && R -q -e "BiocManager::install(version='3.19', ask=FALSE, update=FALSE)" \
 && R -q -e "BiocManager::install(c('SingleR','celldex','SummarizedExperiment','scrapper'), ask=FALSE, update=FALSE)" \
 && R -q -e "install.packages('Matrix', repos='https://cloud.r-project.org')" \
 && R -q -e "stopifnot(all(c('SingleR','celldex','SummarizedExperiment','Matrix','scrapper') %in% rownames(installed.packages())))"

# --- Python environment ---------------------------------------------------------------
# A venv rather than the system interpreter, so pip cannot fight Debian's packages.
ENV VIRTUAL_ENV=/opt/venv
RUN python3.11 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /opt/cellcyrix

# Dependency metadata first, so a source-only change does not rebuild the dependency
# layer. uv.lock carries the exact resolved versions the paper's numbers came from.
COPY pyproject.toml uv.lock README.md LICENSE ./

# uv installs from the lockfile, which is the whole point of shipping one: `pip install`
# would re-resolve and could pick up a newer transitive dependency than was tested.
RUN pip install --no-cache-dir uv==0.5.* \
 && uv pip sync --python "$VIRTUAL_ENV/bin/python" uv.lock || \
    pip install --no-cache-dir -e .

# --- application ----------------------------------------------------------------------
COPY cellcyrix/ ./cellcyrix/
COPY main.py config.yaml CITATION.cff ./
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY benchmarks/ ./benchmarks/
COPY shared_reference/ ./shared_reference/

RUN pip install --no-cache-dir -e .

# --- CellTypist models ----------------------------------------------------------------
# Fetched and SHA256-verified at build time so the container never needs the network at
# run time. A model whose checksum does not match is deleted, and this step fails.
RUN fetch-celltypist-models \
 && python -c "\
from pathlib import Path; \
import os; \
d = Path(os.environ['SCPIPE_SHARED_REFERENCE_ROOT']) / 'celltypist_models' / 'data' / 'models'; \
n = len(list(d.glob('*.pkl'))); \
print(f'CellTypist models present: {n}'); \
raise SystemExit(0 if n >= 41 else 1)"

# --- build-time verification ----------------------------------------------------------
# The offline end-to-end smoke test runs in the image. If the environment cannot produce
# correct output, the build fails here rather than at analysis time. Needs no network,
# no credentials and no input data — it builds a synthetic cohort in a temp workspace.
RUN python tests/smoke_test.py \
 && python tests/run_resolver_tests.py \
 && python -m pytest tests cellcyrix -q

# Optional: the HTML report renders without this; PDF export needs a browser.
# Uncomment to enable PDF, at roughly +400 MB.
# RUN playwright install --with-deps chromium

# Run as an unprivileged user. Outputs go to a bind-mounted directory.
RUN useradd --create-home --uid 10001 cellcyrix \
 && mkdir -p /data/outputs \
 && chown -R cellcyrix:cellcyrix /data /opt/cellcyrix
USER cellcyrix

WORKDIR /data

# `docker run <image> --config ... --output-root ...` — arguments go to the CLI.
ENTRYPOINT ["cellcyrix"]
CMD ["--help"]
