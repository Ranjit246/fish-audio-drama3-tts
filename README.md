# Drama 3 on vLLM-Omni — low-latency serving on one RTX PRO 6000 (96 GB)

Streaming voice-clone TTS with [fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3)
served by [vLLM-Omni](https://github.com/vllm-project/vllm-omni), tuned for
time-to-first-audio (TTFA) on a single NVIDIA RTX PRO 6000 Blackwell (sm_120).

- **48 ms time-to-first-audio at 1 stream, 88 ms at 16** (stock deploy: 80 / 162 ms),
  with WER and speaker similarity unchanged — via a smaller first codec window and a
  load-adaptive window ramp (a vllm-omni patch on the `vllm-omni` branch).
- Word timestamps read from the model's own attention at 46 ms resolution, streamed
  alongside the audio (second patch; no forced aligner, no ASR).
- Reproducible Seed-TTS-Eval benchmark (same protocol as the model card / #8933):
  per-line voice cloning from pre-encoded references, streaming PCM, TTFA / RTF /
  stalls / runaway takes, Whisper-large-v3 WER and WavLM-SV speaker similarity.
- Two startup fixes needed on SM120 that the upstream recipe does not mention.

### How this compares with the H200 numbers in vllm-omni #8933 — read before quoting

| | RTX PRO 6000, this repo | H200, #8933 (stock deploy) | verdict |
|---|---:|---:|---|
| TTFA p50, 1 / 16 / 24 streams | **48 / 88 / 149 ms** | 66 / 104 / – ms | lower here |
| TTFA p50, 32 / 64 streams | 219 / 350 ms | 135 / 198 ms | higher here |
| RTF (throughput), 1 / 16 / 32 / 64 streams | 0.18 / 0.41 / 0.56 / 0.70 | 0.135 / 0.24 / 0.32 / 0.375 | 1.3–1.9× slower here |
| WER / speaker similarity | 1.7–2.2 % / 0.94 | 2.1 % / 0.96 (WavLM-SV) | on par |

The time-to-first-audio win up to ~24 streams is a **deployment** improvement (first
codec window of 2–4 frames instead of 6, chosen by load), not a hardware one: the
same patch on an H200 should land near ~35 ms. On throughput the RTX PRO 6000 is the
slower card (1.6 vs 4.8 TB/s on a bandwidth-bound decode step), and #8933's numbers
stand; one RTX PRO 6000 serves ≤32 streams cleanly, 64 is over its capacity. The
nano-vllm port on the `nano-vllm` branch is a working MVP, not a faster engine
(882 ms to first frame, 73 ms/frame vs 48 ms / 43 ms for vLLM-Omni on the same card).

## Hardware and software

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell **Server Edition**, 96 GB, compute capability 12.0, 600 W |
| Driver / CUDA | 580.178.04 / CUDA 13.0 (system `nvcc` is 12.9 — see fixes) |
| Host | GCP VM, Ubuntu 22.04, 48 vCPU, 176 GB RAM |
| Python | 3.12 (uv venv) |
| vLLM | 0.31.0, torch 2.13.0+cu132 |
| vLLM-Omni | main @ `c470395` (includes #8900 model support, #8933 perf rewrite, #8938 fix) |
| Model | `fishaudio/drama-3` revision `b2b5ee9` (36.9 GiB FP8-expert checkpoint + `codec.pth`) |

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p ~/drama3 && cd ~/drama3
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install vllm==0.31.0 --torch-backend=auto          # pulls the CUDA 13 wheels
uv pip install "git+https://github.com/vllm-project/vllm-omni.git@c470395bde2f8de4f6f29b54adf970f885afa32f"
uv pip install "huggingface_hub[cli]" jiwer soundfile librosa zhon
# fix 1: opencv pulled in by vllm-omni needs libGL; use the headless build
uv pip uninstall opencv-python opencv-python-headless && uv pip install "opencv-python-headless==4.10.0.84"
sudo apt-get install -y libgl1   # optional belt-and-braces
hf download fishaudio/drama-3
```

### Two SM120-specific fixes

1. **`ImportError: libGL.so.1`** at stage-0 start: vllm-omni's dependency tree pulls
   the non-headless `opencv-python`. Replace it with `opencv-python-headless`
   (above) or install `libgl1`.
2. **DeepGEMM JIT fails** (`DeepGEMM SM120 kernels require CUDA 13.0 or newer`):
   vllm-omni registers Drama 3's `bsgemm-moe` quantization as an `Fp8Config`
   subclass, which bypasses vLLM's own "auto-disable DeepGEMM on Blackwell"
   guard, and DeepGEMM then compiles with the system `nvcc` (12.9). Set
   **`VLLM_USE_DEEP_GEMM=0`**. vLLM then selects the **Triton** block-FP8 MoE
   backend with the tuned `E=256,N=512,...RTX_PRO_6000_Blackwell_Server_Edition`
   config, which #8933 found faster than DeepGEMM on H200 anyway.

With those, the startup log shows the kernel set you want:

```
Using FlashInfer GDN prefill kernel (requested=auto, head_k_dim=128)
GDN decode kernel: cuda
Using TRITON Fp8 MoE backend
Using configuration from .../E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition,dtype=fp8_w8a8,block_shape=[128,128].json
Using FLASH_ATTN attention backend ... FlashAttention version 2
```

## Serving

```bash
# stock deploy (vllm_omni/deploy/fish_omni.yaml, loaded automatically)
VLLM_USE_DEEP_GEMM=0 vllm serve fishaudio/drama-3 --omni --port 8091

# latency-tuned overlay (this repo)
VLLM_USE_DEEP_GEMM=0 vllm serve fishaudio/drama-3 --omni --port 8091 \
    --deploy-config deploy/b_admission.yaml
```

Startup takes ~5 min (weights 11 s, torch.compile + CUDA-graph capture for 39
piecewise + 19 full graph sizes, codec graphs 58 s). Stage 0 gets 33.9 GiB of KV
cache (525k tokens) at the stock `gpu_memory_utilization: 0.78`.

`scripts/serve.sh <name> [deploy.yaml]` wraps this: kills the previous server,
starts a fresh one in tmux, waits for `/health`, logs to `logs/server_<name>.log`.

### Request shape that gives the low numbers

```jsonc
POST /v1/audio/speech
{
  "model": "fishaudio/drama-3",
  "input": "Text to speak.",
  "ref_text": "Transcript of the reference clip.",
  "ref_audio_codes": [[...], ...],      // [10 codebooks, frames] — pre-encoded once per voice
  "stream": true, "stream_format": "audio", "response_format": "pcm",
  "max_new_tokens": 500                  // ~23 s; set from expected text length
}
```

- **Send `ref_audio_codes`, not `ref_audio`.** Inline audio is re-encoded inside
  stage 0 on every request (not cached): **+25 ms TTFA at concurrency 1** in our
  measurements, and the encode blocks every other running stream. Encode each
  voice once with `scripts/encode_refs.py` (uses the model's own `codec.pth`).
- Raw PCM (`stream_format: "audio"`) avoids per-chunk base64/JSON on the server's
  event loop. Output is 44.1 kHz s16le mono (`X-Audio-Sample-Rate` header).
- Cap `max_new_tokens` near the expected length: a few takes per hundred keep
  emitting silence until the cap (model behaviour, also on H200).

## Benchmark

Protocol follows vllm-omni #8933 / the model card's production-serving section:
Seed-TTS-Eval **en** (`zhaochenyang20/seed-tts-eval`, 1088 lines), each line clones
its own reference clip, references pre-encoded, streaming PCM, sampling
temperature 1.0 / top_p 0.9 / top_k 30 / max 500 frames, 8 warmup requests
excluded. Metrics: TTFA (first audio byte), RTF (request → last chunk ÷ audio
duration), audio s/s and frames/s (21.5 frames = 1 s), **stall** (a chunk arrives
after the audio already received has played out; strict, and >100 ms), runaway
takes (≥ 20 s). Deviations from #8933 for the time budget: a fixed 300-line
subset per level, and one fresh server per *config* (8 warmups per level) instead
of per level.

The overlays in `deploy/` inherit from the installed stock deploy; point their
`base_config` at your venv once:

```bash
SITE=$(python -c 'import vllm_omni, os; print(os.path.dirname(vllm_omni.__file__))')
sed -i "s#^base_config: .*fish_omni.yaml#base_config: $SITE/deploy/fish_omni.yaml#" deploy/*.yaml
```

```bash
python scripts/encode_refs.py                      # 1088 refs -> manifest.jsonl (32 s on the GPU)
scripts/run_config.sh stock                        # fresh server + c=1,4,16,64 (+ GPU utilization sampler)
LEVELS="1 16" scripts/run_config.sh a_ramp deploy/a_ramp.yaml
python scripts/bench.py --concurrency 1 --n 300 --mode audio --out runs/stock_audio_c1   # inline ref_audio
python scripts/score.py runs/stock_c1 runs/b_admission_c1   # WER + SIM; stop the server first
python scripts/report.py > results/report.md
```

## Results

### Stock `fish_omni.yaml` (tuned upstream for 128 streams on an H200)

300 lines per level, `ref_audio_codes`:

| c | TTFA p50 | TTFA p95 | RTF | audio s/s | frames/s | stalls strict / >100 ms | runaway | H200 (#8933) TTFA p50 / RTF |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **80 ms** | 82 ms | 0.177 | 5.7 | 123 | 0 / 0 % | 2 | 66 ms / 0.135 |
| 4 | **102 ms** | 124 ms | 0.245 | 16.4 | 353 | 0 / 0 % | 0 | 75 ms / 0.157 |
| 16 | **162 ms** | 212 ms | 0.408 | 38.5 | 829 | 0 / 0 % | 1 | 104 ms / 0.239 |
| 64 | **352 ms** | 524 ms | 0.698 | 72.6 | 1563 | 13.7 / 1.3 % | 4 | 198 ms / 0.375 |
| 1, inline `ref_audio` | 105 ms | 114 ms | 0.183 | 5.5 | 119 | 0 / 0 % | 3 | – |

TTFA spread at c=1 is 76–84 ms over 300 requests. The card is comfortable up to
16 streams and compute-bound by 64 (RTF 0.70; 13.7 % of streams see at least one
strict underrun, 1.3 % an audible one). Realistic operating ceiling: 32–48 streams.

### Tuned (pending)

| step | change | c=1 TTFA p50 / p95 | c=16 TTFA p50 / p95 | RTF c=1 / c=16 | stalls | notes |
|---|---|---|---|---|---|---|
| (a) | `codec_chunk_ramp: "2,4,8,16,32"` (first window 2 frames instead of 6) | **49 / 50 ms** (was 80 / 82) | **91 / 140 ms** (was 162 / 212) | 0.177 / 0.408 (unchanged) | c=1: 0 %; c=16: 10.7 % strict, **0 % >100 ms** | GPU util 99–100 %; the 4-frame second window must land inside the 93 ms the first one buys — micro-underruns at 16 streams, none audible |
| (b) | (a) + prefill admission off + `max_num_seqs: 32` | 48 / 50 ms | 89 / 141 ms | 0.177 / 0.407 | c=16: 8.7 % strict, 0 % >100 ms | no latency change at ≤16 streams (admission only engages at ≥32 running); peak GPU memory 84.6 GB vs 88.8 GB |
| (c) | (b) + main-token `top_k` 100 (`fish_omni_max_top_k: 128`) | 48 / 50 ms | 89 / 141 ms | 0.176 / 0.408 | c=16: 5.7 % strict, 0 % >100 ms | latency-neutral; runaway 2 (c=1) / 0 (c=16) vs (b) 0 / 3 — within noise; WER/SIM decides (b) vs (c) |
| (e) | (c) with `codec_chunk_ramp: "3,6,12,24,32"` | 56 / 58 ms | 111 / 167 ms | 0.177 / 0.408 | c=16: 4.3 % strict, 0 % >100 ms | **rejected**: +8 ms (c=1) / +22 ms (c=16) for a 1.4-point drop in inaudible micro-stalls |

### Final: profile (c) on all 1088 lines

| c | TTFA p50 | TTFA p95 | TTFA p99 | RTF | audio s/s | stalls strict / >100 ms | runaway | H200 stock (#8933) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **48 ms** | 50 ms | 51 ms | 0.178 | 5.7 | 0 / 0 % | 1 | 66 ms / 0.135 |
| 16 | **88 ms** | 115 ms | 157 ms | 0.411 | 38.8 | 7.5 / **0 %** | 1 | 104 ms / 0.239 |
| 32 | 117 ms | 150 ms | 243 ms | 0.564 | 55.7 | **96.6 / 63.6 %** | 1 | 135 ms / 0.318 |

**Quality (Whisper-large-v3 WER, WavLM-SV cosine), same 300 utterances for both:**

| c | stock WER (no-runaway) | tuned WER (no-runaway) | stock SIM | tuned SIM | >50 %-WER takes stock / tuned |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.95 % (1.51 %) | **1.70 % (1.70 %)** | 0.943 | 0.943 | 2 / 0 |
| 16 | 1.65 % (1.32 %) | 1.65 % (1.48 %) | 0.944 | 0.942 | 1 / 1 |

On all 1088 lines the tuned profile scores WER 2.16 % (2.07 % without the one runaway
take) and SIM 0.941 — in line with #8933's 2.13 % at 64 streams on H200. The 2-frame
first window and `top_k` 100 cost nothing in intelligibility or speaker similarity.

**Operating envelope — read this before deploying.** Profile (c) is the right choice
up to 16 streams: TTFA below the H200's, no audible stalls. **At 32 streams it must
not be used**: 64 % of streams get an audible underrun, because the 4-frame second
window (~105 ms to generate at ~26 ms/step) cannot arrive inside the 93 ms the
2-frame first window buffers, and with prefill admission off every new arrival's
prefill stretches every running stream's step. The safe first-window size scales
with load; a load-adaptive ramp (`codec_chunk_ramp_by_load`, implemented on the
`vllm-omni` branch under `upstream/`) is the fix. The matrix below shows why no
static profile can be.

### Profile × concurrency matrix (300 lines per cell; TTFA p50 / audible-stall % / RTF)

| profile | first window / admission / `max_num_seqs` | c=1 | c=4 | c=16 | c=32 | c=64 |
|---|---|---:|---:|---:|---:|---:|
| stock | 6 / on / 128 | 80 / 0 % / 0.18 | 102 / 0 % / 0.24 | 162 / 0 % / 0.41 | **219 / 0 % / 0.55** | **352 / 1.3 % / 0.70** |
| a_ramp | 2 / on / 128 | 49 / 0 % / 0.18 | **61 / 0 % / 0.24** | 91 / 0 % / 0.41 | 117 / 44.7 % / 0.55 | 230 / 99 % / 0.70 |
| c_topk | 2 / off / 32 | **48 / 0 % / 0.18** | 61 / 0 % / 0.25 | **88 / 0 % / 0.41** | 117 / 63.6 % / 0.56 | 2463¹ / 59.7 % / 1.09 |
| e_ramp3 | 3 / off / 32 | 56 / 0 % / 0.18 | 71 / 0 % / 0.25 | 111 / 0 % / 0.41 | 144 / 15.3 % / 0.55 | 2515¹ / 22.3 % / 1.11 |
| f_mid | 4 / on / 64 | 64 / 0 % / 0.18 | 82 / 0 % / 0.24 | 126 / 0 % / 0.41 | 169 / 1.7 % / 0.55 | 281 / 91 % / 0.70 |
| H200 stock (#8933) | 6 / on / 128 | 66 | 75 | 104 | 135 | 198 |

¹ queueing, not a ramp effect: `max_num_seqs 32` with 64 concurrent requests parks
half of them in the wait queue.

**Best static profile per level (audible stalls ≤ 1 %):** c=1 `c_topk` 48 ms ·
c=4 `a_ramp` 61 ms · c=16 `c_topk` 88 ms · c=32 `stock` 219 ms · c=64 none within
budget (`stock` at 1.3 % is closest). The first-window size that is safe shrinks
from 6 frames at 64 streams to 2 frames at ≤16 — exactly a function of load, which
is what `codec_chunk_ramp_by_load` encodes (2 frames below 8 in-flight streams, 3
below 16, 4 below 32, stock above). Even 4 frames fails at 64 streams (`f_mid`),
so the top of the table must stay at the stock ramp. Reproduce with
`scripts/matrix.sh` and `scripts/best.py` (`results/best.md`).

### Load-adaptive ramp (`codec_chunk_ramp_by_load`) — one deploy for every level

Implemented as a patch to vllm-omni on the `vllm-omni` branch (`upstream/`): each
request picks its codec window ramp from the number of requests in its first decode
step; the codec stage captures CUDA graphs for the union of window sizes. Table
`16:2,4,8,16,32 ; 24:4,6,12,24,32 ; *:6,8,12,16,24,32`, stock prefill admission,
`max_num_seqs 64`, measured with `PYTHONPATH` shadowing of the patched package:

| c | TTFA p50 | TTFA p95 | RTF | stalls strict / audible | best static profile at this level |
|---:|---:|---:|---:|---:|---|
| 1 | **49 ms** | 50 ms | 0.177 | 0 / 0 % | `c_topk` 48 ms |
| 16 | **88 ms** | 144 ms | 0.408 | 10.0 / **0 %** | `c_topk` 88 ms |
| 24 | **149 ms** | 232 ms | 0.495 | 2.7 / **0 %** | (not measured statically) |
| 32 | **219–222 ms** | 278–328 ms | 0.556 | 0.3 / **0 %** | `stock` 219 ms |
| 64 | 349–350 ms | 529–540 ms | 0.700 | 10–20 / 2.3–10 % | `stock` 352 ms / 1.3 % |

It matches the best static profile at every level up to 32 streams with no
operator decision — the recommended deployment. v1 of the patch selected the ramp
from a lagging in-flight count and failed under burst arrivals (12.7 % audible
stalls at 64); v2 uses the step's batch size.

**64 streams is over this card's capacity, for any profile.** Per-stream RTF is
0.70 (each stream generates only 1.4× real time), and the audible-stall rate is
run-to-run noisy: the adaptive deploy measured 2.3 %, 10 % and 2.7 % across three
runs (the last with `max_num_seqs 128`, which ruled out slot count as the cause);
stock's 1.3 % is a single run. Treat ≤32 streams as the operating envelope of one
RTX PRO 6000 and scale out beyond that.

Quality (WER / SIM on saved WAVs): *pending*.

## Where the remaining latency is (concurrency 1)

Per-frame step time on this card is ~43 ms of GPU-bound decode for a 3B-active
MoE + 13-layer codebook tower (memory-bandwidth bound: ~4.5 GB of weights read
per frame at 1.6 TB/s vs 4.8 TB/s on H200). Host-side work with `ref_audio_codes`
is under ~1 ms per request (API process: pydantic validation of ~2.5k ints,
blake2b of the codes, ~7 tokenizer calls; engine: one `np.stack` per window).
The stock first window of 6 frames costs ~5 decode steps before the codec can
start — that is what step (a) attacks.

## nano-vllm port (branch `nano-vllm`)

A from-scratch runtime on the nano-vllm-fusion scaffold, built to answer "can a
~2k-line engine beat vLLM-Omni's latency on this model?" (feasibility verdict in
`nanovllm-port/FEASIBILITY.md`). Status: **MVP speaks** — real 35B weights (experts
dequantized to bf16, 67.9 GB peak), codebook tower, RAS sampler, Fish DAC codec,
eager B=1, SDPA attention (no flash-attn wheel for torch 2.14 / CUDA 13).

| | nano-vllm MVP | vLLM-Omni (tuned) |
|---|---:|---:|
| WER, same 20 Seed-TTS lines | 2.2–2.65 % | 1.33 % |
| time to first frame / TTFA | 882 ms | 48 ms |
| decode per frame | 73 ms (backbone 60 + tower 12) | ≈43 ms |
| RTF | 1.8–2.0 | 0.18 |

The gap is where the feasibility analysis said it would be: the eager Python MoE
loop is launch-bound (~5,400 `mm` calls per 5 steps at 6.9 µs each, 1,800 `nonzero`
syncs). `nanovllm-port/BENCH.md` ranks the fixes (grouped MoE GEMM, CUDA graphs for
the B=1 step, Triton chunked GDN for prefill). 102 CPU tests pass.

## Known limitations

- One TP / PP rank; prefix caching off. A per-voice prefix cache (prefill only the
  new text; the system prompt + reference codes are identical per voice) would
  need a voice-only `cache_salt` (today the salt hashes the input text, so no two
  requests can ever hit) plus vLLM's `mamba_cache_mode: align` for the GDN state —
  a vllm-omni code change, not a deploy setting.
- Some takes run to `max_tokens` with silence (2–4 of 300 here; model behaviour).
- Word timestamps come from the model's own attention (no forced aligner, no ASR):
  implemented and GPU-validated as patch `0002` on the `vllm-omni` branch — 46.4 ms
  resolution, streaming `speech.alignment.delta` events or an `X-Word-Timestamps`
  header, opt-in via `hf_overrides: fish_omni_word_timestamps: true`. Design, head
  sweep and validation in [TIMESTAMPS.md](TIMESTAMPS.md). Still to measure: per-step
  cost at 64+ streams, preemption/resume, texts over 512 tokens.

## Layout

```
Drama3/
├── README.md
├── sync.sh                 # pull results/logs from the VM; --push to push script edits
├── scripts/
│   ├── serve.sh            # fresh server per config, waits for /health
│   ├── encode_refs.py      # Seed-TTS refs -> ref_audio_codes manifest
│   ├── bench.py            # streaming client: TTFA / RTF / stalls / runaway, saves WAVs
│   ├── run_config.sh       # serve + bench per level + GPU utilization sampler
│   ├── score.py            # Whisper-large-v3 WER (official normalization) + WavLM-SV SIM
│   └── report.py           # runs/*/summary.json -> markdown vs H200 reference
├── deploy/                 # thin overlays on vllm_omni/deploy/fish_omni.yaml
├── results/                # summary.json / results.jsonl / gpu.csv per run (no WAVs)
├── logs/
└── nanovllm-port/FEASIBILITY.md
```

## License

Apache-2.0 for the code in this repository. Vendored files from vllm-omni keep their
Apache-2.0 headers (provenance in `VENDORED.md` files); nano-vllm-fusion is MIT and
pulled in as a submodule. Model weights are not included — `fishaudio/drama-3` is
distributed by Fish Audio under the Fish Audio Community License.

## Acknowledgments

vllm-omni PRs [#8900](https://github.com/vllm-project/vllm-omni/pull/8900)
(Drama 3 support) and [#8933](https://github.com/vllm-project/vllm-omni/pull/8933)
(the runtime rewrite that makes these numbers possible); the Seed-TTS-Eval
protocol from Bytedance; README structure after
[nanovllm-voxcpm](https://github.com/a710128/nanovllm-voxcpm).
