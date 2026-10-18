# Drama 3 on nano-vllm-fusion — first GPU numbers (2026-10-18)

Box: RTX PRO 6000 Blackwell 96 GB (sm_120), driver 580 / CUDA 13.0,
torch 2.14.1+cu130, triton 3.8.0, Python 3.12, venv `~/drama3/nanovllm-port/.venv`.
Engine: `nanovllm_drama3.engine.Drama3Engine` — B = 1, eager, bf16, experts
dequantized to bf16 at load, `lm_head` sliced to 4097 rows, tower + DAC codec as
plain PyTorch, no CUDA graphs, no fused MoE, no flash-attn (see "Attention").

Everything below is from `out/first.log`, `out/b20/` (20 Seed-TTS lines, seed 0),
`out/b20_seed1/` (same lines, seed 1), `out/profile.log` and `out/compile.log` on
the VM (`~/drama3/nanovllm-port/out/`, WAVs included); the JSON/log parts are
copied to `bench_out/` in this repo. Reproduce with

```
.venv/bin/python -m nanovllm_drama3.cli synth --ref-codes-from-manifest <utt> --out x.wav --seed 0 --repeat 2
.venv/bin/python -m nanovllm_drama3.cli batch --limit 20 --out-dir out/b20 --seed 0
.venv/bin/python -m nanovllm_drama3.cli wer --hyp-dir out/b20 --ref-dir ~/drama3/runs/final_c_c1/wavs
```

## Load

| | |
|---|---|
| backbone (8 shards, 62,245 tensors, 30,720 FP8 experts dequantized on the GPU) | **7.8 s** (shards were in the page cache) |
| tower (192 tensors straight from the shards) | 0.3 s |
| codec.pth + tokenizer | ~4 s |
| engine ready | **12.1 s** |
| peak allocated | **67.9 GB** (steady ≈ 67.5 GB: 64 GB experts + 1 GB embed + ~1 GB rest + 2.4 GB tower + 1.6 GB codec fp32; KV cache for 8192 tokens 0.17 GB, GDN slot 0.06 GB) |

Param map, dtypes and GDN state shapes needed no fixes: `load_backbone` reported
0 missing / 0 unexpected on the real checkpoint on the first try.

## Quality: 20 Seed-TTS lines, Whisper-large-v3 WER

Same 20 utterances (`manifest.jsonl` rows 0–19), same transcription pipeline for
both systems (Whisper English normalizer, lowercase, punctuation stripped,
word-level edit distance; `cli.py wer`). The vLLM-Omni WAVs are the c=1 run
`~/drama3/runs/final_c_c1/wavs/` (its own 1088-line score: 2.16 %).

| system | corpus WER | mean per-utt WER | > 50 % | notes |
|---|---|---|---|---|
| nano-vllm port, seed 0 | **2.65 %** | 2.54 % | 0 | 5 lines with one slip each ("the"→"a", dropped "is", "did"→"do", "and the boy"); 15 lines exact |
| nano-vllm port, seed 1 | **2.21 %** | 2.03 % | 0 | 3 lines with one slip each ("fell"→"felt", "sky jumper"→"skyjumper", and the line vLLM-Omni also misses); 17 exact |
| vLLM-Omni c=1 | 1.33 % | 1.21 % | 0 | 2 lines with one slip |

All 20 lines ended with EOS (0 runaways, 0 hit `max_new_tokens`); durations are
0.89–1.28× the vLLM-Omni WAV of the same line (mean 1.07×); semantic code 0 stays in
1..4095 (code 0 excluded from the candidates), residual codes in 0..1023.
The WER gap is within single-seed sampling noise (every miss is a one-word slip on
a different line from vLLM-Omni's own misses), so the STATUS.md item-8 logits
comparison (`norm_topk_prob`, embedding scatter, RoPE, GDN reset) was not needed.

## Speed (one cloned line, 150-row prompt, 47 frames, warm)

| phase | nano-vllm port (eager) | vLLM-Omni c=1 (same card) |
|---|---|---|
| prefill, backbone (150 rows) | 858 ms | — |
| prefill, tower teacher-forcing (149 rows) | 11 ms | — |
| time to first frame | **882 ms** | **48 ms** (TTFA, 2-frame first window) |
| decode step, backbone | 60.3 ms | — |
| decode step, tower (9 heads, sequential) | 12.4 ms | — |
| decode step, sampler (main token, RAS double draw) | 0.5 ms | — |
| **decode step, total** | **73 ms / frame** (46.4 ms of audio) | **≈ 43 ms / frame** |
| codec (47 frames → 2.18 s, fp32, non-streaming) | 11 ms (first call 307 ms) | streaming |
| RTF, whole line | 1.98 (20-line mean 1.81) | 0.18 |

The step time is flat across the 20 lines (72.1–72.7 ms mean per line; the
sequence grows from 150 to ~400 rows without a visible attention cost).

Where the time goes (`out/profile.log`; forward hooks with a sync per module, so
the split carries a few ms of overhead):

| | decode step (ms) | prefill, 150 rows (ms) |
|---|---|---|
| MoE blocks (40 layers) | **46** | **964** |
| GatedDeltaNet (30 layers) | 17 | 121 |
| full attention, SDPA (10 layers) | 6 | 7 |
| whole backbone call | 60–76 | 858–1102 |

The step is **launch-bound, not compute-bound**: over 5 profiled decode steps
(tower + backbone) the profiler records ~500 ms of host time but only ~125 ms
of CUDA kernel time (≈25 ms of GPU work per 73 ms step; the largest kernel bucket,
`aten::mm`, is 5,385 calls of 6.9 µs each). The MoE path is 40 layers × a Python
loop over the 8 routed experts, each a separate `nn.Module` with two tiny GEMMs
(gemv on 2048×1024 and 512×2048) plus `index_add_`, and a `.nonzero().tolist()`
host sync per layer (1,805 `nonzero` calls in 5 steps). GDN decode is a few small
fp32 matmuls on a [32,128,128] state per layer, again many launches.

Prefill is slow for the same reason the design notes predicted: the MoE loop
visits up to 256 expert modules per layer for 150 tokens (964 ms), and fusion's
`chunk_gated_delta_rule` runs a Python loop per 64-token chunk plus a 63-step inner
loop per GDN layer (121 ms).

`--compile` (keeping fusion's `@torch.compile` on the norms / SiLU / rotary) did
not help: 76 ms/step and 960 ms prefill after warm-up (first line 4.1 s prefill,
92 ms/step while compiling) vs 73 ms / 858 ms eager — the compiled pieces are not
where the launches are.

## Attention on this box

No flash-attn wheel exists for torch 2.14 / CUDA 13 (the latest release only ships
cu12 wheels for torch 2.4–2.8) and the VM has only the 12.9 toolkit, so per the
STATUS.md caveat the engine imports fusion with `flash_attn` stubbed and
replaces the two entry points with `torch.nn.functional.scaled_dot_product_attention`
(`engine.py: sdpa_varlen_prefill`, `sdpa_decode_with_kvcache`, bf16, GQA via
`enable_gqa`; tested against the float32 reference fallbacks in
`tests/test_e2e_cpu.py`). The Triton `store_kvcache` kernel is kept. Nothing in
`third_party/` or `backbone/` was edited for this.

## Fixes made during integration

* `engine.py` only. `torch.cuda.set_device(torch.device("cuda"))` rejects an
  index-less device on this torch — the engine now resolves `cuda` →
  `cuda:<current>`.
* No change was needed in packages 1–3: param map, FP8 dequant, GDN state
  shapes, tower names and the prompt layout all matched the checkpoint.
* Pre-existing, not fixed (not an integration issue): `tests/test_backbone_fusion.py::test_import_fusion_cpu`
  asserts that triton is stubbed on a CPU-only run, which is false on the VM
  (real triton installed) even with `CUDA_VISIBLE_DEVICES=""`. All other CPU
  tests (including `tests/test_e2e_cpu.py`) pass there.

## Next optimisation, in order of expected payoff

1. **Fused / grouped MoE for decode** (the 60 ms): replace the per-expert
   `nn.Module` loop with stacked expert weights `[E, 2I, H]` / `[E, H, I]` and a
   grouped GEMM (Triton fused-MoE kernel, or at B = 1 simply
   `torch.bmm` on the 8 gathered experts); drop the `.nonzero().tolist()` sync.
   Expected: backbone step from ~60 ms to ~15–20 ms (memory-bound on 8 × 6 MB of
   bf16 expert weights per layer ≈ 2 GB per step ≈ 1–2 ms at 1.8 TB/s, plus
   launch overhead).
2. **CUDA graphs for the decode step** (B = 1, fixed shapes once the MoE is a
   fixed-shape kernel): removes the ~1,000 launches per step from the critical
   path; this is what gets the tower from 12 ms to a few ms too (9 sequential
   heads × 13 layers of tiny kernels). fusion disables graphs for hybrid models
   because of the GDN state pools; with one slot and static `state_indices` the
   step is graph-safe.
3. **Prefill**: Triton/`fla` chunked gated-delta-rule kernel instead of the
   Python chunk loop (858 ms → tens of ms), and a batched MoE for prefill tokens.
   Together with 1–2 this is what brings TTFF near vLLM-Omni's 48 ms.
4. flash-attn / FlashInfer decode attention once a cu13 wheel exists (SDPA is
   not the bottleneck at these sequence lengths).
5. Streaming codec decode (see `codec/streaming_notes.md`) to turn "time to first
   frame" into time to first audio.
