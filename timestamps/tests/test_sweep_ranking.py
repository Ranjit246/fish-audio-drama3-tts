"""Planted-diagonal head must rank first; parked head must lose despite its path_mass."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from d3align.capture import list_captures, match_captures, read_requests, target_cols_of, target_logits
from d3align.scoring import layer_metrics
from sweep import match as match_mod
from sweep import score_heads

from . import synth

ROOT = Path(__file__).resolve().parents[1]


def test_match_skips_junk_and_pairs_in_order(tmp_path):
    cap_dir, req_path, requests = synth.make_dataset(tmp_path)
    rows, junk, unmatched = match_mod.run(cap_dir, req_path, tmp_path / "matched.jsonl", verbose=False)
    assert len(rows) == len(requests)
    assert not unmatched
    assert len(junk) == 3  # profiling run, 1-row dummy, trailing undecoded prefill
    assert [r["utt"] for r in rows] == [r["utt"] for r in requests]
    for r in rows:
        assert r["steps"] == r["expected_steps"] and not r["notes"]
        assert r["n_text_sliced"] == r["n_text"]
    assert (tmp_path / "matched.jsonl").exists()


def test_target_slice_happens_before_softmax(tmp_path):
    cap_dir, req_path, _ = synth.make_dataset(tmp_path, n_cloned=1, n_zeroshot=0)
    pairs, _, _ = match_captures(list_captures(cap_dir), read_requests(req_path))
    cap, req, _ = pairs[0]
    cols = target_cols_of(req)
    sliced = target_logits(cap, cols)
    assert sliced[7].shape == (cap.steps, synth.N_HEADS, len(cols))
    np.testing.assert_array_equal(sliced[7], cap.logits(7)[:, :, cols])
    mets, target_mass = layer_metrics(cap.logits(7), cols)
    # the planted head puts most of its full-prompt mass on the target columns (>> uniform)
    uniform = len(cols) / cap.prompt_len
    assert target_mass[synth.PLANTED[1]] > 0.5 and target_mass[synth.PLANTED[1]] > 5 * uniform
    mets3, target_mass3 = layer_metrics(cap.logits(3), cols)
    # ... while the reference-reading head does not, even if its sliced matrix is sharp-ish
    assert target_mass3[synth.REFERENCE_READER[1]] < 0.2


def test_planted_head_ranks_first_in_every_condition(tmp_path):
    cap_dir, req_path, _ = synth.make_dataset(tmp_path)
    pairs, _, _ = match_captures(list_captures(cap_dir), read_requests(req_path))
    per_cond, baselines = score_heads.score_pairs(pairs, verbose=False)
    assert set(per_cond) == {"cloned", "zeroshot"}
    for cond, rows in per_cond.items():
        assert len(rows) == len(synth.LAYERS) * synth.N_HEADS
        ranked = sorted(rows, key=lambda k: -rows[k]["score"])
        assert ranked[0] == synth.PLANTED, (cond, ranked[:3])
        planted, parked = rows[synth.PLANTED], rows[synth.PARKED]
        assert planted["monotonicity"] > 0.9 and planted["coverage"] > 0.8
        # the trap: the parked head has a near-perfect path_mass but no coverage
        assert parked["path_mass"] > 0.9
        assert parked["coverage"] < 0.2
        assert parked["score"] < planted["score"] / 2
        assert 0 < baselines[cond] < 1
    heads = score_heads.rank_heads(per_cond, topk=3)
    assert (heads[0]["layer"], heads[0]["head"]) == synth.PLANTED
    assert set(heads[0]) >= {"layer", "head", "score", "target_mass", "cloned", "zeroshot"}


def test_score_heads_cli_writes_files(tmp_path):
    cap_dir, req_path, _ = synth.make_dataset(tmp_path, n_cloned=2, n_zeroshot=2)
    out_dir = tmp_path / "scored"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "sweep" / "score_heads.py"), "--capture-dir", str(cap_dir), "--requests", str(req_path),
         "--out-dir", str(out_dir), "--topk", "2", "--revision", "test-rev", "--quiet"],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    for cond in ("cloned", "zeroshot"):
        d = json.loads((out_dir / f"sweep_{cond}.json").read_text())
        assert len(d["heads"]) == len(synth.LAYERS) * synth.N_HEADS
        assert d["heads"][0]["layer"] == synth.PLANTED[0] and d["heads"][0]["head"] == synth.PLANTED[1]
        assert d["meta"]["condition"] == cond
    heads = json.loads((out_dir / "alignment_heads.json").read_text())
    assert heads["revision"] == "test-rev"
    assert heads["conditions"] == ["cloned", "zeroshot"]
    assert len(heads["heads"]) == 2
    assert (heads["heads"][0]["layer"], heads["heads"][0]["head"]) == synth.PLANTED
    assert "model layer index" in heads["layer_index_space"]
