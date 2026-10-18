"""Drama 3 tokenizer loading (tokenizer files only, ~20 MB; never the weights).

Adapted from Drama3-nano-vllm/nanovllm-port/nanovllm_drama3/prompt/tokenizer.py,
plus: a local Hugging Face cache snapshot is used without touching the network
when it already holds the tokenizer files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DRAMA3_REPO = "fishaudio/drama-3"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")


def _cached_snapshot(repo: str) -> Path | None:
    """A snapshot directory in the HF cache that already has the tokenizer files."""
    root = Path(os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME", Path.home() / ".cache/huggingface") )
    if "HF_HUB_CACHE" not in os.environ:
        root = root / "hub" if root.name != "hub" else root
    snaps = root / f"models--{repo.replace('/', '--')}" / "snapshots"
    if not snaps.is_dir():
        return None
    for snap in sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if (snap / "tokenizer.json").is_file() and (snap / "tokenizer_config.json").is_file():
            return snap
    return None


def tokenizer_dir(model: str | os.PathLike[str] = DRAMA3_REPO, *, cache_dir: str | None = None) -> Path:
    """Directory holding the tokenizer files: ``model`` itself when it is a
    directory, a cached snapshot when present, else a tokenizer-only download."""
    path = Path(model)
    if path.is_dir():
        return path
    cached = _cached_snapshot(str(model)) if cache_dir is None else None
    if cached is not None:
        return cached
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(model), allow_patterns=list(TOKENIZER_FILES), cache_dir=cache_dir))


def model_revision(model: str | os.PathLike[str] = DRAMA3_REPO) -> str | None:
    """The commit hash of the cached snapshot, when the tokenizer came from the HF cache."""
    try:
        d = tokenizer_dir(model)
    except Exception:
        return None
    if d.parent.name == "snapshots":
        return d.name
    return None


def load_tokenizer(model: str | os.PathLike[str] = DRAMA3_REPO, *, cache_dir: str | None = None) -> Any:
    """``transformers.AutoTokenizer`` for the checkpoint (tokenizer files only)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(tokenizer_dir(model, cache_dir=cache_dir)))


__all__ = ["DRAMA3_REPO", "TOKENIZER_FILES", "load_tokenizer", "model_revision", "tokenizer_dir"]
