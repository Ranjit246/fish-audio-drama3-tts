# SPDX-License-Identifier: Apache-2.0
"""Tokenizer loading for Drama 3 (tokenizer files only, ~20 MB; no weights)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DRAMA3_REPO = "fishaudio/drama-3"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")


def tokenizer_dir(model: str | os.PathLike[str] = DRAMA3_REPO, *, cache_dir: str | None = None) -> Path:
    """Local directory holding the tokenizer files: ``model`` itself when it is a
    directory, otherwise a Hugging Face snapshot restricted to the tokenizer files."""
    path = Path(model)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(model), allow_patterns=list(TOKENIZER_FILES), cache_dir=cache_dir))


def load_tokenizer(model: str | os.PathLike[str] = DRAMA3_REPO, *, cache_dir: str | None = None) -> Any:
    """``transformers.AutoTokenizer`` for the checkpoint (downloads tokenizer files only)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(tokenizer_dir(model, cache_dir=cache_dir)))


__all__ = ["DRAMA3_REPO", "TOKENIZER_FILES", "load_tokenizer", "tokenizer_dir"]
