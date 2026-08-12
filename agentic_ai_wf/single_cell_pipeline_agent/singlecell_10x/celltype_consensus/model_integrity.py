"""
model_integrity.py — verify a CellTypist model file before it is unpickled.

Why this exists
---------------
CellTypist models are Python pickles, and ``pickle.load`` executes whatever the
stream tells it to. The bundle ships 41 of them plus a ``SHA256SUMS.txt`` manifest —
but nothing in the pipeline ever read that manifest, so the checksums were
documentation rather than a control. A tampered or half-downloaded ``.pkl`` was loaded
exactly like a good one.

This module closes that gap: ``verify_model_file`` hashes the file and compares it to
the manifest entry BEFORE the caller hands the path to CellTypist. A mismatch raises;
a file the manifest does not list raises; a missing manifest is a warning, because a
user who supplies their own model directory should not be blocked by the absence of
ours.

Hashing cost is ~50 MB read once per model per run, which is negligible next to
loading and applying the model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path

from ..exceptions import PipelineInputError

logger = logging.getLogger(__name__)

MANIFEST_NAME = "SHA256SUMS.txt"
MODELS_INDEX_NAME = "models.json"

# Upstream is a plain static bucket; a generous timeout matters more than retries
# because a partial file is deleted and re-fetched on the next attempt anyway.
DOWNLOAD_TIMEOUT_S = 300

# Read the file in chunks so a large model is never fully resident just to hash it.
_HASH_CHUNK_BYTES = 1 << 20


class ModelIntegrityError(PipelineInputError):
    """A model file's checksum does not match the shipped manifest."""


def _sha256(path: Path) -> str:
    """Hex SHA-256 of a file, read in chunks.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(models_dir: str | Path) -> dict[str, tuple[str, int | None]]:
    """Parse ``SHA256SUMS.txt`` from a model directory.

    The shipped manifest carries three whitespace-separated fields per line —
    ``<sha256>  <filename>  <size_bytes>`` — which is `sha256sum` output plus a size
    column. The size is optional so a plain `sha256sum > SHA256SUMS.txt` file also
    parses.

    Args:
        models_dir: Directory holding the ``.pkl`` models and the manifest.

    Returns:
        Mapping of filename to `(expected_hex_digest, expected_size_or_None)`. Empty
        when no manifest exists.
    """
    manifest_path = Path(models_dir) / MANIFEST_NAME
    if not manifest_path.is_file():
        return {}
    sums: dict[str, tuple[str, int | None]] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        digest = parts[0].lower()
        name = Path(parts[1].lstrip("*")).name
        size: int | None = None
        if len(parts) > 2:
            try:
                size = int(parts[2])
            except ValueError:
                size = None
        sums[name] = (digest, size)
    return sums


def verify_model_file(model_path: str | Path, *, required: bool = False) -> bool:
    """Check a model file against its directory's manifest before it is unpickled.

    Args:
        model_path: The ``.pkl`` about to be loaded.
        required: When True, a missing manifest or missing entry is an error. Use for
            the bundled model directory, where the manifest is always shipped. Leave
            False for a user-supplied directory.

    Returns:
        True when the checksum matched; False when no manifest entry was available and
        `required` is False.

    Raises:
        ModelIntegrityError: If the digest differs from the manifest, or if the
            manifest/entry is absent while `required` is True.
    """
    path = Path(model_path)
    sums = load_manifest(path.parent)

    if not sums:
        message = f"No {MANIFEST_NAME} beside {path.name}; cannot verify before load."
        if required:
            raise ModelIntegrityError(message)
        logger.warning("[MODEL-INTEGRITY] %s", message)
        return False

    entry = sums.get(path.name)
    if entry is None:
        message = (
            f"{path.name} is not listed in {path.parent / MANIFEST_NAME}. Refusing to "
            "unpickle an unlisted model file."
        )
        if required:
            raise ModelIntegrityError(message)
        logger.warning("[MODEL-INTEGRITY] %s", message)
        return False

    expected, expected_size = entry

    # Size is a cheap pre-check that catches the common case (a truncated download)
    # without reading the whole file.
    if expected_size is not None:
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ModelIntegrityError(
                f"Size mismatch for {path.name}: manifest says {expected_size} bytes, "
                f"file is {actual_size}. The download is truncated or the file was "
                "replaced. Re-fetch it with scripts/fetch_celltypist_models.py."
            )

    actual = _sha256(path)
    if actual != expected:
        raise ModelIntegrityError(
            f"Checksum mismatch for {path.name}: manifest says {expected}, file is "
            f"{actual}. The model is corrupt or has been modified — unpickling it "
            "would execute whatever the file contains. Re-fetch it with "
            "scripts/fetch_celltypist_models.py."
        )
    logger.debug("[MODEL-INTEGRITY] %s verified (%s).", path.name, actual[:12])
    return True


def model_urls(models_dir: str | Path) -> dict[str, str]:
    """Map model filename to its upstream URL, read from ``models.json``.

    Args:
        models_dir: Directory holding ``models.json``.

    Returns:
        Mapping of filename to download URL.

    Raises:
        FileNotFoundError: If ``models.json`` is absent.
    """
    index = Path(models_dir) / MODELS_INDEX_NAME
    if not index.is_file():
        raise FileNotFoundError(
            f"{index} is missing; it is tracked in git and lists the upstream URLs."
        )
    data = json.loads(index.read_text(encoding="utf-8"))
    return {
        entry["filename"]: entry["url"]
        for entry in data.get("models", [])
        if entry.get("filename") and entry.get("url")
    }


def fetch_model(model_name: str, url: str, models_dir: str | Path) -> bool:
    """Download one model and verify it, removing the file if verification fails.

    Args:
        model_name: Model filename, e.g. ``Immune_All_Low.pkl``.
        url: Upstream URL to fetch it from.
        models_dir: Destination directory.

    Returns:
        True if the model is present and verified after this call.
    """
    dest = Path(models_dir) / model_name
    # Download to a temp name so an interrupted transfer never occupies the real one.
    tmp = dest.with_name(dest.name + ".part")
    try:
        logger.info("[MODEL-FETCH] downloading %s", model_name)
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response:
            tmp.write_bytes(response.read())
        tmp.replace(dest)
    except (urllib.error.URLError, OSError, TimeoutError):
        logger.exception("[MODEL-FETCH] could not download %s from %s", model_name, url)
        tmp.unlink(missing_ok=True)
        return False

    try:
        verify_model_file(dest, required=True)
    except ModelIntegrityError:
        # A pickle that fails verification must not be left where a loader finds it.
        logger.exception(
            "[MODEL-FETCH] %s failed verification; deleting it", model_name
        )
        dest.unlink(missing_ok=True)
        return False
    logger.info("[MODEL-FETCH] verified %s", model_name)
    return True


def ensure_model_file(
    models_dir: str | Path, model_name: str, *, allow_fetch: bool = True
) -> Path | None:
    """Return a verified model path, fetching it once if the bundle is not local yet.

    The models are referenced externally rather than committed (54 MB of pickles, one
    over the 5 MB per-file ceiling), so a fresh clone has the manifest but not the
    ``.pkl`` files. Rather than making an offline run fail with "model missing" and a
    manual step to discover, this fetches exactly the model being asked for, verifies
    it against the manifest, and caches it in place — so the FIRST run needs the
    network and every run after it does not.

    Args:
        models_dir: Directory holding the manifest and the models.
        model_name: File name of the model, e.g. ``Immune_All_Low.pkl``.
        allow_fetch: Set False to require the model to be present already (a run that
            must not touch the network).

    Returns:
        The verified path, or None when the model is absent and could not be fetched.

    Raises:
        ModelIntegrityError: If a present or freshly-fetched file fails verification.
            A pickle that does not match the manifest is never handed back.
    """
    directory = Path(models_dir)
    target = directory / model_name

    if target.is_file():
        verify_model_file(target, required=True)
        return target

    if not allow_fetch:
        logger.warning(
            "[MODEL-INTEGRITY] %s is not present and fetching is disabled.", model_name
        )
        return None

    expected = load_manifest(directory)
    if model_name not in expected:
        logger.warning(
            "[MODEL-INTEGRITY] %s is not listed in %s; not fetching an unlisted model.",
            model_name,
            MANIFEST_NAME,
        )
        return None

    try:
        url = model_urls(directory).get(model_name)
    except FileNotFoundError:
        logger.warning(
            "[MODEL-INTEGRITY] models.json is missing from %s; cannot fetch %s.",
            directory,
            model_name,
        )
        return None
    if not url:
        logger.warning(
            "[MODEL-INTEGRITY] %s has no URL in models.json; cannot fetch.", model_name
        )
        return None

    logger.info(
        "[MODEL-INTEGRITY] %s not present locally; fetching it once from upstream "
        "and verifying against %s.",
        model_name,
        MANIFEST_NAME,
    )
    if not fetch_model(model_name, url, directory):
        logger.warning("[MODEL-INTEGRITY] fetch of %s failed.", model_name)
        return None
    # fetch_model verifies and deletes on mismatch, so reaching here means it is good.
    return target


__all__ = [
    "ModelIntegrityError",
    "MANIFEST_NAME",
    "MODELS_INDEX_NAME",
    "load_manifest",
    "model_urls",
    "verify_model_file",
    "fetch_model",
    "ensure_model_file",
]
