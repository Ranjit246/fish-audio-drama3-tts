"""Synthetic capture files in the hook's req_<n>.npz format, with planted heads."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LAYERS = (3, 7)
N_HEADS = 4
PLANTED = (7, 2)   # the alignment head: a clean diagonal on the target columns
PARKED = (3, 1)    # the trap: all attention on one target token (path_mass ~ 1, no coverage)
REFERENCE_READER = (3, 3)  # a sharp diagonal that sits on the REFERENCE transcript columns, not the target


def raw_logits(rng, steps: int, prompt_len: int, target: range, ref_cols: range | None, layer: int, sharpness: float = 6.0):
    """[steps, N_HEADS, prompt_len] raw logits for one layer."""
    M = rng.normal(0.0, 0.5, size=(steps, N_HEADS, prompt_len))
    n_text = len(target)
    for h in range(N_HEADS):
        if (layer, h) == PLANTED:
            for s in range(steps):
                j = target.start + min(n_text - 1, (s * n_text) // steps)
                M[s, h, j] += sharpness
                M[s, h, min(prompt_len - 1, j + 1)] += sharpness * 0.3
        elif (layer, h) == PARKED:
            M[:, h, target.start + n_text // 2] += sharpness
        elif (layer, h) == REFERENCE_READER and ref_cols is not None and len(ref_cols) > 1:
            n_ref = len(ref_cols)
            for s in range(steps):
                M[s, h, ref_cols.start + min(n_ref - 1, (s * n_ref) // steps)] += sharpness
        # every other head: noise (the uniform-ish baseline)
    return M.astype(np.float16)


def write_capture(path: Path, counter: int, prompt_len: int, steps: int, target: range, ref_cols: range | None, rng):
    arrays = {
        "prompt_len": np.int64(prompt_len),
        "layers": np.asarray(LAYERS, dtype=np.int64),
        "steps": np.int64(steps),
        "counter": np.int64(counter),
        "pid": np.int64(1),
        "created": np.float64(0.0),
    }
    for layer in LAYERS:
        arrays[f"logits_L{layer}"] = (
            raw_logits(rng, steps, prompt_len, target, ref_cols, layer)
            if steps
            else np.zeros((0, 0, prompt_len), dtype=np.float16)
        )
    np.savez(path, **arrays)


def make_dataset(out_dir: Path, *, n_cloned: int = 3, n_zeroshot: int = 3, seed: int = 0):
    """Captures + requests.jsonl with junk files mixed in. Returns (capture_dir, requests_path, plan)."""
    rng = np.random.default_rng(seed)
    cap_dir = out_dir / "capture"
    cap_dir.mkdir(parents=True)
    requests = []
    counter = 0
    # junk: a profiling run (long prefill, no decode) and a dummy prefill of 1 row
    write_capture(cap_dir / f"req_{counter}.npz", counter, 4096, 0, range(0, 1), None, rng); counter += 1
    write_capture(cap_dir / f"req_{counter}.npz", counter, 1, 0, range(0, 1), None, rng); counter += 1
    order = 0
    texts = [
        "Get the trust fund to the bank early.",
        "We asked over twenty different people today.",
        "The weather will be clear and cold tonight.",
        "Please bring the documents before the meeting.",
        "Seven small boats drifted toward the harbour.",
        "Nobody expected the answer to be that simple.",
    ]
    plan = [("cloned", True)] * n_cloned + [("zeroshot", False)] * n_zeroshot
    for i, (cond, cloned) in enumerate(plan):
        text = texts[i % len(texts)]
        n_text = len(text.split()) + 2
        ref_frames = int(rng.integers(60, 120)) if cloned else 0
        ref_text_rows = 12 if cloned else 0
        # layout: system(9) [+ ref text rows + ref frames] + user_open(3) + speaker(1) + text + close(2) + assistant(4)
        head = 9 + ref_text_rows + ref_frames + 3 + 1
        prompt_len = head + n_text + 2 + 4
        target = range(head, head + n_text)
        ref_cols = range(9 + 6, 9 + ref_text_rows) if cloned else None
        steps = int(rng.integers(50, 90))
        write_capture(cap_dir / f"req_{counter}.npz", counter, prompt_len, steps, target, ref_cols, rng)
        counter += 1
        requests.append(
            {
                "order": order,
                "condition": cond,
                "utt": f"utt{order}",
                "text": text,
                "ref_text": "reference transcript" if cloned else None,
                "prompt_len": prompt_len,
                "target_cols": [target.start, target.stop],
                "n_text": n_text,
                "audio_s": steps * 2048 / 44100,
                "expected_steps": steps,
            }
        )
        order += 1
    # a trailing junk prefill that never decoded (e.g. a request cancelled mid-prefill)
    write_capture(cap_dir / f"req_{counter}.npz", counter, 777, 0, range(0, 1), None, rng)
    req_path = out_dir / "requests.jsonl"
    req_path.write_text("".join(json.dumps(r) + "\n" for r in requests))
    return cap_dir, req_path, requests
