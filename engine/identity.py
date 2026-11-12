"""Deployment identity and checkpoint pinning; independent of benchmark clients."""

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


def source_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if "tests" in path.relative_to(root).parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def engine_identity(config):
    root = Path(__file__).resolve().parent
    build = {}
    path = os.getenv("RELAY_BUILD_INFO")
    if path:
        build = json.loads(Path(path).read_text())
    revision, dirty = build.get("revision"), build.get("dirty")
    if not revision:
        try:
            revision = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            dirty = bool(
                subprocess.check_output(
                    ["git", "-C", str(root), "status", "--porcelain"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            )
        except (OSError, subprocess.CalledProcessError):
            revision = os.getenv("RELAY_REVISION", "unreported")
    fingerprint = source_sha256(root)
    if build.get("source_sha256") and build["source_sha256"] != fingerprint:
        dirty = True
    return {
        "engine_revision": revision,
        "engine_source_sha256": fingerprint,
        "engine_dirty": dirty,
        "config_sha256": hashlib.sha256(config.model_dump_json().encode()).hexdigest(),
    }


def resolve_revision(repo: str, configured: str | None) -> str:
    """Resolve once, then load weights and tokenizer from that same immutable commit."""
    requested = configured or os.getenv("RELAY_MODEL_REVISION") or "main"
    if re.fullmatch(r"[0-9a-f]{40}", requested):
        return requested
    from huggingface_hub import HfApi

    revision = HfApi().model_info(repo, revision=requested).sha
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Could not resolve an immutable model revision")
    return revision
