# Drama 3 on nano-vllm: feasibility (2026-10-18)

Question: should Fish Audio Drama 3 get a nano-vllm runtime, the way
`a710128/nanovllm-voxcpm` did for VoxCPM?

Short answer: **feasible, but it is a new engine rather than a port.** Upstream
nano-vllm supports one dense model (Qwen3) and has none of the five things
Drama 3 needs. Every relevant capability exists only in unmerged PRs or 0–43 star
forks. Rough effort: ~8–10 engineer-weeks to a slow MVP, ~17–23 weeks to approach
vLLM-Omni's current latency (81 ms TTFA on the RTX PRO 6000). Recommended only as
a learning / control project, not as the production path.

## What Drama 3 needs vs what nano-vllm has

| Capability Drama 3 needs | Upstream nano-vllm (`GeeeekExplorer/nano-vllm`, last push 2026-04-26) |
|---|---|
| MoE experts (256, top-8 + shared) | No. Three open PRs (#116, #236, #251), none merged |
| Block-FP8 (128×128) expert weights | No. PR #241 (weight-only fp8, dequant per forward) closed unmerged |
| Gated DeltaNet linear attention (30 of 40 layers), per-request recurrent state | No. PR #232 (Qwen3.5 GDN Triton ops, +8k lines) open |
| Hybrid cache: paged KV for 10 layers + fixed state slots for 30 | No |
| Codebook tower (4 shared + 9 residual layers with own KV) | No analogue; closest is voxcpm's `residual_lm` |
| Second stage: DAC codec, streaming windows | No multi-stage pipeline, no token streaming (`generate()` returns a list) |
| Sampler: top-k / top-p / repetition-aware | No (12-line temperature-only Gumbel sampler) |
| Decode CUDA graphs | Yes (decode only, batch-size buckets); no fork has them for hybrid + tower |
| Prefix caching | Yes (xxhash per full block); recurrent-state invalidation would be new |

### Relevant PRs / issues (nano-vllm, all states)

| # | Type | State | Date | Title |
|---|---|---|---|---|
| 116 | PR | open | 2025-10-16 | FuseMoeLinear + Qwen3-MoE (Unsloth grouped GEMM) |
| 236 | PR | open | 2026-05-29 | qwen3-moe (MoE kernel + CUDA graph + chunked prefill) |
| 251 | PR | open | 2026-07-26 | Qwen3-MoE with fused Triton kernels (vLLM modular_kernel port) |
| 241 | PR | closed | 2026-06-10 | fp8 (e4m3) weight-only quantization |
| 232 | PR | open | 2026-05-13 | Qwen3.5 text+multimodal: GDN Triton ops, persistent GDN slot pool |
| 184 / 229 | PR | open | 2026-03 / 05 | INT8 KV cache |
| 206 | issue | open | 2026-04-20 | Implement Qwen3.5 → points to Wangbiao2/nano-vllm-qwen3.5 |
| 225 | issue | open | 2026-05-07 | Nano-vLLM-Quant: block-FP8 GEMM, FP8 KV, Qwen3.5 dense |
| 73 | issue | open | 2025-07-08 | chunked prefill of new seqs while decoding |

No hits for: mamba, gated delta (outside #232), hybrid, streaming, tts, audio.

## What nanovllm-voxcpm actually did (why it worked for VoxCPM)

`a710128/nanovllm-voxcpm` is a re-architected copy, not a fork. Per its
ARCHITECTURE.md:

- `Sequence.token_ids` is a mixed `int | bytes` stream so audio latents take part
  in prefix caching; `custom_payload` carries per-request state; engine
  `preprocess_seq` / `postprocess_seq` hooks.
- `VoxCPMModel` = base LM + residual LM + local feat encoder + CFM/DiT decoder +
  stop head, returning latent patches per step.
- `runner.py` + `streaming_vae.py`: latents → waveform chunk per step; one
  multiprocess server per GPU, `mp.Queue` IPC, async client yields chunks.
- Decode CUDA graphs plus an optional prefill-diffusion graph; LoRA with
  slot-major weights for graph validity.
- Result: first-audio median 24–28 ms on H800 at concurrency 1 (~35 % better
  than the official engine). VoxCPM is a *small dense* model, which is why a
  1.2k-line runtime could beat the reference.

Its issues/PRs are all VoxCPM-specific; nothing about MoE or other models.

## Forks that already cover pieces of Drama 3's backbone

| Repo | Stars | What it adds |
|---|---|---|
| xiez-njust-xmu/nano-vllm-fusion | 0 | **Qwen3.5-MoE hybrid (GDN + MoE + shared expert), verified on Qwen3.5-35B-A3B** — Drama 3's exact backbone. bf16 only; "CUDA graphs disabled for hybrid models" |
| cuber726579/Nano-vLLM-Quant | 4 | Block-wise FP8 GEMM, FP8 KV, Qwen3.5 dense hybrid |
| Wangbiao2 / Yuezheng-Ling-0412 nano-vllm-qwen3.5 | 1 | GDN hybrid, TP, decode CUDA graph |
| shiguang12321/gemma4-nanovllm-optimization | 43 | Gemma4 26B-A4B MoE, grouped GEMM, CUDA graphs |
| yaLen404/nano-vllm-moe-spec | 0 | Triton grouped-GEMM MoE, fair chunked prefill |
| MciG-ggg/nanovllm-omni | 5 | Thinker → Talker → Code2Wav stage pipeline |

## Effort estimate (one strong engineer)

| Component | Vendor from | Weeks |
|---|---|---|
| Block-FP8 MoE experts (256, top-8 + shared) | vLLM `fused_moe` Triton block-fp8 (PR #251 ports the bf16 kernel) | 3–4 |
| GDN chunked prefill + recurrent decode, state slots | flash-linear-attention / PR #232 ops; FlashInfer GDN decode | 3–4 |
| Hybrid cache manager (paged KV + state slots, prefix-cache invalidation) | nano-vllm-fusion design | 2–3 |
| Codebook tower (own KV, 9 sequential sub-steps per frame) | voxcpm `residual_lm` pattern | 2–3 |
| DAC codec stage + streaming windows | voxcpm `runner.py` / `streaming_vae.py` scaffold, Fish DAC weights | 2 |
| Sampler (top-k/top-p/RAS, vocab 252k) | vLLM sampler logic | 1 |
| Decode CUDA graphs across hybrid + tower | nothing to vendor | 2–3 |
| Loader, config, tests, server | voxcpm server | 2 |
| **Total** | | **~17–23 weeks to parity-ish; ~8–10 to MVP** |

**MVP milestone:** non-streaming, eager mode, concurrency 1, experts dequantized
to bf16 at load (~70 GB, fits the 96 GB card), tower run sequentially in Python,
DAC run offline on the finished frame sequence. Start from nano-vllm-fusion
(backbone already runs Qwen3.5-35B-A3B) grafted onto the voxcpm runtime (mixed
token stream, `custom_payload`, per-step chunk emission).

## Verdict

The per-frame budget is 46 ms (21.5 Hz), and each frame is 40 MoE/GDN layers +
13 tower layers + codec. That cost is dominated by kernel quality (block-FP8
grouped GEMM, GDN recurrent kernel, CUDA graphs over a hybrid model) — exactly
what vLLM already has and what no nano fork has for this combination.
nano-vllm's plausible gain is lower per-step Python overhead at concurrency 1 and
a codebase you fully own; the measured host overhead on the vLLM-Omni path with
`ref_audio_codes` is under ~1 ms per request, so there is little to recover.
You would give up hybrid-aware prefix caching, FP8 kernels, CUDA graphs on
hybrid models, batching maturity, and maintenance (upstream nano-vllm has been
effectively unmaintained since April 2026).

Pursue only if the goal is an embedded, dependency-light runtime or a learning
project. For production latency, keep tuning vLLM-Omni.

## If pursued anyway: first steps

1. Clone `xiez-njust-xmu/nano-vllm-fusion`; confirm it runs `Qwen/Qwen3.5-35B-A3B`
   on the RTX PRO 6000 in bf16 and measure tokens/s at batch 1.
2. Load Drama 3's text backbone weights into it (same architecture; experts
   dequantized from block-FP8 to bf16 at load).
3. Add the codebook tower as a sequential Python module; produce frames
   non-streaming; decode with the Fish DAC offline. Compare WER against the
   vLLM-Omni output on 20 Seed-TTS lines.
4. Only then decide whether to invest in kernels (block-FP8 experts, GDN decode,
   CUDA graphs) — compare per-frame time against vLLM-Omni's ~46 ms budget.
