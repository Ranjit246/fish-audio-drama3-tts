"""Test setup: import paths for the d3align package and the .pth hook module."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "hook"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture
def hook_module(tmp_path, monkeypatch):
    """A fresh, armed import of hook/drama3_align_hook.py writing to tmp_path."""
    monkeypatch.setenv("DRAMA3_ALIGN_SWEEP", "0")  # import inert; tests arm pieces explicitly
    sys.modules.pop("drama3_align_hook", None)
    mod = importlib.import_module("drama3_align_hook")
    mod.STATE.__init__()
    mod.STATE.out_dir = str(tmp_path / "capture")
    mod.STATE.idle_s = 0  # no background thread in tests
    yield mod
    for cls in list(mod.STATE.patched_classes):
        mod.unpatch_class(cls)
    sys.modules.pop("drama3_align_hook", None)


@pytest.fixture(scope="session")
def drama3_tokenizer():
    from d3align.tokenizer import load_tokenizer

    try:
        return load_tokenizer()
    except Exception as e:  # pragma: no cover - offline without a cached tokenizer
        pytest.skip(f"Drama 3 tokenizer unavailable: {e}")
