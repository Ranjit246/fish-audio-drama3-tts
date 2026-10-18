"""Real-weight smoke test on the GPU box (skipped without CUDA or the checkpoint).

    .venv/bin/python -m pytest tests/test_e2e_gpu.py -m gpu -s

Loads the full engine once (~70 GB), synthesizes one manifest line with a cloned
voice and checks duration / code ranges / timing fields.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.gpu

MANIFEST = Path(os.environ.get("DRAMA3_MANIFEST", Path.home() / "drama3" / "manifest.jsonl"))


def _model_dir() -> Path:
    from nanovllm_drama3.engine import default_model_dir

    try:
        return default_model_dir()
    except Exception as exc:  # not cached
        pytest.skip(f"checkpoint not available: {exc!r}")


@pytest.fixture(scope="module")
def engine():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from nanovllm_drama3.engine import Drama3Engine

    return Drama3Engine.load(_model_dir(), device="cuda")


def test_load_report(engine):
    rep = engine.load_report
    assert rep["peak_allocated_gb"] < 90
    assert "MISSING" not in rep["backbone"] and "UNEXPECTED" not in rep["backbone"]
    print(rep)


def test_cloned_line(engine):
    from nanovllm_drama3.cli import manifest_codes, read_manifest

    if not MANIFEST.exists():
        pytest.skip(f"no manifest at {MANIFEST}")
    row = read_manifest(MANIFEST)[0]
    wav, frames, t = engine.synthesize(
        row["text"], ref_text=row["ref_text"], ref_codes=manifest_codes(row), max_new_tokens=500, seed=0,
    )
    assert wav is not None and wav.dtype == torch.float32
    assert frames.shape[1] == 10 and frames.shape[0] >= 10
    assert int(frames[:, 0].min()) >= 1 and int(frames[:, 0].max()) < 4096
    assert int(frames[:, 1:].max()) < 1024
    assert t["stop_reason"] == "eos"
    words = len(row["text"].split())
    assert 0.12 * words < t["audio_seconds"] < 1.5 * words + 2.0  # plausible duration for the text
    assert wav.abs().max() > 0.05  # not silent
    print({k: v for k, v in t.items() if k != "per_step"})
