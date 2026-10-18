"""Shared maths and plumbing for Drama 3 attention-derived word timestamps.

Modules
-------
dp            monotonic DTW and path-mass kernels (numba, numpy fallback)
scoring       per-head sweep metrics (path_mass, monotonicity, coverage, score)
prompt_layout reference-bracket-v1 prompt rows / target-text columns (vendored)
tokenizer     Drama 3 tokenizer loading (tokenizer files only)
capture       reading the hook's req_<n>.npz files and matching them to requests
decode        reduce_matrix / words_from_path (the runtime decode maths)

Everything here is CPU-only and has no vLLM dependency.
"""

from .dp import HAVE_NUMBA, dtw_path, monotonic_path_mass  # noqa: F401

# Drama 3 / DAC codec: 44.1 kHz, 2048 samples per frame, one frame per decode step.
CODEC_SAMPLE_RATE = 44100
CODEC_HOP = 2048
FRAME_MS = 1000.0 * CODEC_HOP / CODEC_SAMPLE_RATE  # 46.4399... ms

# Model layer indices of the 10 full-attention layers of fishaudio/drama-3
# (config.json layer_types: every 4th layer starting at 3, of 40).
FULL_ATTENTION_LAYERS = tuple(range(3, 40, 4))
NUM_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 256
