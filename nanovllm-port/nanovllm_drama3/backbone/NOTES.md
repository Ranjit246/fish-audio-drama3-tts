# nano-vllm-fusion as the Drama 3 backbone — notes

Vendored at `third_party/nano-vllm-fusion` (commit `3e44d30`, MIT; see
`VENDORED.md`). Everything below was read from that commit; nothing in
`third_party/` is edited — all fixes are applied at import time by
`nanovllm_drama3/backbone/`.

## 1. Classes that implement the hybrid backbone

| Piece | File (under `nanovllm/`) | Class / function | Notes |
|---|---|---|---|
| Top-level model | `models/qwen3_5_moe.py` | `Qwen3_5MoeForCausalLM(config, vision_config=None)` | `.model` (`Qwen3_5MoeModel`) + `.lm_head` (`ParallelLMHead`). `forward(input_ids, positions, pixel_values=None, image_grid_thw=None, image_token_mask=None) -> hidden [N, H]`; `compute_logits(hidden)` applies `lm_head` (in prefill only on the last token of each sequence, via `context.cu_seqlens_q`). |
| Decoder layer | same | `Qwen3_5MoeDecoderLayer(config, layer_idx)` | picks `Qwen3_5Attention` or `GatedDeltaNet` from `config.layer_types[layer_idx]`; FFN is `Qwen3_5MoeBlock` unless `layer_idx in config.mlp_only_layers` (or `decoder_sparse_step` says dense). Pre-norm with fused residual (`GemmaRMSNorm(x, residual)`). |
| MoE FFN | same | `Qwen3_5MoeBlock` (**replaced** by `Drama3MoeBlock`, see §5) | `gate: nn.Linear(H, E)`, `experts: ModuleList[Qwen3_5MoeExpert]` (each `gate_up_proj: MergedColumnParallelLinear`, `down_proj: RowParallelLinear`), `shared_expert: _SharedExpertMLP`, `shared_expert_gate`. Routing: softmax → top-k → renormalise → python loop over hit experts with `index_add_`. |
| Full attention | `models/qwen3_5.py` | `Qwen3_5Attention(config, layer_idx)` | `q_proj` outputs `2 * heads * head_dim` (query + per-head output gate), `k_proj`/`v_proj`, `o_proj`; `q_norm`/`k_norm` = `GemmaRMSNorm(head_dim)`; `InterleavedMRoPE` (partial rotary, 1-D positions ⇒ plain RoPE); `Attention` core from `layers/attention.py`. |
| Attention core | `layers/attention.py` | `Attention`, `store_kvcache` | **GPU only**: Triton `store_kvcache_kernel`, `flash_attn_varlen_func` (prefill), `flash_attn_with_kvcache` (decode, paged with `block_table`). Owns `k_cache`/`v_cache` views set by the runner. |
| Linear attention | `layers/gated_delta_net.py` | `GatedDeltaNet(config, layer_idx)` | **pure torch** (no Triton): `in_proj_qkv/z/b/a`, depthwise `conv1d`, `A_log` (float32), `dt_bias`, `norm: RMSNormGated(head_v_dim)`, `out_proj`. Prefill = `chunk_gated_delta_rule` (chunk 64, python loop per sequence and per chunk); decode = `recurrent_gated_delta_rule`. State pools `conv_states [slots, conv_dim, K-1]`, `recurrent_states [slots, Hv, Dk, Dv] f32` indexed by `context.state_indices`. |
| Norms | `layers/layernorm.py` | `GemmaRMSNorm` (1 + w), `RMSNormGated` | `@torch.compile` decorated. |
| Linear / embed | `layers/linear.py`, `layers/embed_head.py` | `ColumnParallelLinear`, `MergedColumnParallelLinear`, `RowParallelLinear`, `VocabParallelEmbedding`, `ParallelLMHead` | Need `torch.distributed` initialised (`dist.get_rank()` in `__init__`), even for TP=1. Each has a `weight_loader(param, tensor[, shard_id])`. |
| Cache / state manager | `engine/model_runner.py`, `engine/scheduler.py`, `engine/block_manager.py`, `engine/sequence.py` | `ModelRunner.allocate_kv_cache` / `allocate_gdn_state`, `StateSlotManager`, `BlockManager`, `Sequence.state_slot_id` | KV blocks (`kvcache_block_size % 256 == 0`) only for layers that have `k_cache`; GDN slots = free list sized from leftover GPU memory. Prefix caching is disabled for hybrid models (`disable_prefix_cache=is_hybrid`); CUDA graphs disabled for hybrid models; decode goes through eager `run_model`. |
| Per-step context | `utils/context.py` | `set_context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, state_indices)` | Global; model code reads it. |
| Loader | `utils/loader.py` | `load_model(model, path)` | Strips `weight_prefix="model.language_model."` → `model.`, packs `gate_proj`/`up_proj` → `gate_up_proj` shards 0/1, silently skips unknown names. Not FP8-aware (we use our own loader, §4). |
| Dispatch | `engine/model_runner.py::_create_model` | `'qwen3_5' in model_type and 'moe' in model_type` → `Qwen3_5MoeForCausalLM` | Our translated config sets `model_type="qwen3_5_moe"`. |

## 2. Config expectations (what the constructor reads)

`FusionTextConfig` in `nanovllm_drama3/config.py` carries exactly these:
`hidden_size, num_hidden_layers, layer_types, num_attention_heads,
num_key_value_heads, head_dim, rms_norm_eps, rope_parameters{rope_theta,
partial_rotary_factor, mrope_section}, max_position_embeddings,
linear_num_value_heads, linear_num_key_heads, linear_key_head_dim,
linear_value_head_dim, linear_conv_kernel_dim, num_experts,
num_experts_per_tok, moe_intermediate_size, shared_expert_intermediate_size,
intermediate_size, vocab_size, tie_word_embeddings, mlp_only_layers,
decoder_sparse_step, dtype, model_type` (+ our `norm_topk_prob`).

Fusion's own `nanovllm/config.py::Config` calls `transformers.AutoConfig`,
which does not know `model_type=fish_omni`; package 4 must build
`Config`-like state from `Drama3Config.fusion_text_config()` instead (e.g.
construct `Config` with a dummy path then overwrite `hf_config`, or bypass
`LLMEngine` and drive `ModelRunner`-style code directly).

Switches the fusion code hardcodes and that Drama 3 matches (the translation
raises `IncompatibleConfig` otherwise): `attn_output_gate=True`,
`use_qk_norm=True`, `layernorm_zero_centered_gamma=True` (GemmaRMSNorm),
`rotary_interleaved=False`, `shared_expert_gate=True`, `hidden_act=silu`,
`rope_type=default` (no scaling), all `mlp_layer_types == "moe"`.

Verified against the safetensors headers: `q_proj (8192, 2048)` = 16×256×2
(gated), `k/v_proj (512, 2048)`, GDN `conv1d (8192, 1, 4)` with
conv_dim = 2·16·128 + 32·128, `A_log/dt_bias (32,)`, GDN `norm (128,)`,
router `gate (256, 2048)`, experts `gate/up (512, 2048)` + `down (2048, 512)`
in `F8_E4M3` with `weight_scale_inv` `(4,16)` / `(16,4)` F32 (block 128×128),
`shared_expert_gate (1, 2048)`, `embed_tokens`/`lm_head (252288, 2048)`.

## 3. Weight naming / loading

Checkpoint `model.language_model.X` → module `model.X`; `gate_proj`/`up_proj`
→ `gate_up_proj` via `MergedColumnParallelLinear.weight_loader(param, t,
shard_id)` (0 = gate, 1 = up); `lm_head.weight` → `lm_head.weight`;
`audio_decoder*` (tower) not part of this module. Full map with counts:
`nanovllm_drama3/weights/param_map.json` (`python -m
nanovllm_drama3.weights.param_map` regenerates it). Dtype notes: `A_log`,
`dt_bias`, GDN `norm.weight` are bf16 in the checkpoint and float32 (A_log,
norm) in the module — `copy_` casts.

## 4. CPU vs GPU-only

| Runs on CPU (pure torch) | GPU only |
|---|---|
| `GatedDeltaNet` prefill + decode, MoE block, shared expert, norms, rotary, embeddings, lm_head, the whole `load_model` path | `layers/attention.py`: Triton `store_kvcache_kernel`, flash-attn varlen (prefill) and kv-cache (decode) calls; `ModelRunner` (NCCL init, `torch.cuda.*`, pinned memory); `@torch.compile` on norms/activation/rotary/sampler (works on CPU but slow/fragile — we disable it off-GPU) |

`nanovllm_drama3.backbone.import_fusion()` makes the model importable on CPU:
puts the checkout on `sys.path`, registers a light `nanovllm` package (skips
`nanovllm/__init__.py`, which imports the engine → transformers, tqdm, PIL,
torchvision), imports `layers/attention.py` with temporary `triton` /
`flash_attn` stubs (`cpu_fallback.py`: SDPA references for both flash-attn
calls, torch `store_kvcache`), sets `TORCH_COMPILE_DISABLE=1`, and installs
the MoE patch. `ensure_process_group()` creates a gloo (CPU) / NCCL (GPU)
world-size-1 group; `build_backbone(cfg, device, dtype)` constructs the model
under `torch.set_default_dtype` + `torch.device(device)` like the runner does.
`allocate_caches()` is a size-explicit replacement for the runner's
`allocate_kv_cache`/`allocate_gdn_state` (used by tests; the runner's version
is fine on the GPU).

Test `tests/test_backbone_fusion.py::test_prefill_then_decode_matches_full_prefill`
shows prefill→decode (recurrent GDN + paged KV) equals a fresh full prefill to
1e-4 on CPU, so the state-slot plumbing is exercised without a GPU.

## 5. Missing / wrong vs Drama 3, and what we did about it

| Gap | Impact | Handling |
|---|---|---|
| **Shared-expert gate shape.** Upstream `_SharedExpertGate` is a scalar `Parameter(ones(1))` applied to the shared-expert *output*. Checkpoint has `shared_expert_gate.weight (1, 2048)`: `sigmoid(x @ Wᵀ)` per token (HF Qwen3-MoE semantics). Upstream `copy_` would fail on shape, and the math is wrong. | Correctness | `backbone/moe_patch.py` rebinds `Qwen3_5MoeBlock` → `Drama3MoeBlock` (`SharedExpertGate = Linear(H→1)`, gate from block input). Tested against a hand computation. |
| **bf16 only; no block-FP8 kernel.** Experts are E4M3 + 128×128 `weight_scale_inv`; upstream loader would copy raw fp8 bits (cast) and drop the scales silently. | Correctness / memory | `weights/fp8.py::dequant_block_fp8` + `weights/loader.py::load_backbone(dequant_experts=True)`; experts become bf16 at load: 40×256×3×(2048×512)×2 B ≈ **64 GB** (+1 GB embed, +1 GB lm_head, ~1 GB rest) — fits the 96 GB card but leaves ~25 GB for KV/GDN state. `dequant_experts=False` raises (nothing to run it on). |
| **`lm_head` slicing** | Memory / speed | Possible: `lm_head` is a plain `F.linear` on `[N, H]`; `load_backbone(lm_head_rows=cfg.semantic_logit_rows())` replaces it with a 4097-row `ParallelLMHead` (4096 semantic + EOS), `model.lm_head_rows` maps logits index → token id. Embedding must stay full (text prompt tokens). Prefill `compute_logits` still only returns the last row per sequence; package 2/4 need all-row hidden states from `model(...)` for the tower (teacher forcing), which `forward` already returns. |
| **Input-embedding injection** for audio frame rows (`[token_id, code0..code9]`, codebook sums from `audio_decoder.codebook_embeddings (40960, 2048)`, `vq_feature_placement=backbone_partial`) | Needed by package 4 | `Qwen3_5MoeModel.forward(input_ids, positions, image_embeds, image_token_mask)` scatters `image_embeds` into masked rows after `embed_tokens`; reuse that hook (tested in `test_embedding_injection_hook`). Note the scatter *replaces* the token embedding; if Drama 3 adds codebook embeddings on top of `embed_tokens(token_id)`, pass the sum. |
| MoE forward = Python loop over up to 256 experts per layer, each a separate `nn.Module` (10,240 expert modules, 20,480 Linear params) | Speed (decode ≫ 46 ms/frame risk) | Accepted for the MVP; a grouped/fused GEMM is the package-4+ optimisation. |
| GDN prefill loops per sequence and per 64-token chunk in Python; decode is a few small matmuls | Speed | Fine at concurrency 1; the FEASIBILITY note on kernels stands. |
| **GDN state slots are never zeroed on (re)allocation.** Prefill with `state_indices` set uses the slot's existing conv/recurrent state as `initial_state`; `StateSlotManager` just recycles ids. After a finished request the stale state leaks into the next sequence in that slot. | Correctness in a multi-request engine | `backbone.reset_state_slot(model, slot)`; package 4 must call it when a slot is (re)assigned, or zero the pools per request. |
| `warmup` branch: when `context.state_indices is None` or pools are empty, GDN uses zero state and does not persist it — fine for one-shot prefill (tests), not for decode. | — | Allocate pools before decode (`allocate_caches` / runner). |
| `norm_topk_prob` absent from Drama 3 `text_config`; fusion always renormalises top-k weights (HF default for Qwen3-Next/3.5 is also True). | Possible logit mismatch | Exposed as `norm_topk_prob` (default True) on the patched block; verify against vLLM-Omni logits in package 4. |
| `mrope_section` default `[11,11,10]` (sum 32 = rotary 64/2) | None for text (1-D positions ⇒ all three axes equal) | Translation sets it explicitly so sizes are consistent for any head_dim. |
| Prefix caching and CUDA graphs disabled for hybrid models; `kvcache_block_size` must be a multiple of 256; `max_num_batched_tokens` default 16384 governs chunked prefill | Perf/behaviour | Known upstream limitations (README). |
| `ParallelLMHead.forward` gathers `cu_seqlens_q[1:] - 1` in prefill — the sampler sees one row per sequence; `Sampler` is temperature-only Gumbel | Package 2 replaces the sampler | Use `model(...)` hidden states + own head call when more rows are needed. |
| Vision path (`model.visual`, MRoPE 3-D positions, `image_token_id` plumbing in the runner) | Unused | Pass `vision_config=None` (default). |
| `transformers.AutoConfig` in `nanovllm/config.py`; tokenizer loaded by `LLMEngine` from the model dir | Package 4 | Drama 3 ships a tokenizer; `AutoConfig` must be bypassed (§2). |

## 6. Dev environment

```
cd nanovllm-port
uv venv --python 3.12 .venv && uv sync          # CPU torch on macOS
uv pip install -e .
.venv/bin/python -m pytest tests/test_backbone* tests/test_weights* -q
```
GPU extras: `uv sync --extra gpu` on Linux (`flash-attn` needs
`--no-build-isolation`), `--extra vision` only if you want `import nanovllm`
to execute its own `__init__`.
