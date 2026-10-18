# Build status (2026-10-18)

CPU packages 1–3 are complete: **95 tests pass, 4 skipped** (torchaudio-only
comparisons) with `.venv/bin/python -m pytest tests/ -q` in ~20 s on an M-series Mac.
Package 4 (GPU integration on the RTX PRO 6000) is queued behind the latency
benchmark that owns the VM's GPU.

| WP | State | Tests | Entry points |
|---|---|---|---|
| 1 backbone | done | 47 | `nanovllm_drama3.config.Drama3Config`, `backbone.import_fusion / build_backbone / allocate_caches / reset_state_slot / ensure_process_group`, `backbone.moe_patch.Drama3MoeBlock`, `weights.loader.load_backbone`, `weights.fp8.dequant_block_fp8`, `weights/param_map.json` |
| 2 tower + sampler + prompt | done | 37 | `tower.CodebookTower` (+`TowerConfig.drama3()`), `sampling.MainTokenSampler`, `prompt.build_prompt_frames / target_text_columns / load_tokenizer / DRAMA3_TOKENS` |
| 3 codec | done | 11 (+4 skipped) | `codec.FishDAC.load / encode_reference / decode` |
| 4 integration | done (MVP: eager, B=1, SDPA attention) | 7 CPU + GPU smoke | `engine.Drama3Engine`, `cli` (`synth` / `batch` / `wer`), `BENCH.md` |

## Facts established by the packages

- Checkpoint: 62,245 tensors in 8 shards. Experts: 30,720 FP8 weights + 30,720
  `weight_scale_inv` (F32, block 128×128, shapes (4,16)/(16,4)), always co-located
  with their weight in the same shard. Tower: 192 tensors under `audio_decoder.*`,
  `audio_decoder_mtp_shared.layers.0-3.*`, `audio_decoder_mtp.layers.0-8.*`,
  `audio_decoder_mtp.heads.0-8.*`. Metadata (config, gzipped index, per-tensor
  dtype/shape from shard headers) is bundled in `nanovllm_drama3/weights/drama3_meta/`.
- Token ids: `<|im_start|>` 248045, `<|im_end|>` = EOS 248046, audio pad 248076,
  `<|voice|>` 248081, `<|speaker:0|>` 248083, semantic range 248099–252194
  (4096 codes), `semantic_logit_rows()` = 4097 rows (EOS + semantic).
- Codec: 44.1 kHz, hop 2048 → 21.53 Hz (46.44 ms/frame), 10 codebooks with sizes
  (4096, 1024×9); `codec.pth` 1.87 GB, 391 M params, loads in 2.4 s on CPU.
  CPU decode ≈ 0.25 s per second of audio in fp32.
- nano-vllm-fusion (`third_party/nano-vllm-fusion` @ `3e44d30`, MIT) gaps found and
  handled: (1) `shared_expert_gate` is a scalar there but a per-token `Linear(2048→1)`
  sigmoid gate in the checkpoint → `backbone/moe_patch.py`; (2) bf16 only, drops
  `weight_scale_inv` → we dequantize experts at load (~64 GB bf16 + ~3 GB rest);
  (3) its config loader uses `transformers.AutoConfig`, which fails on
  `model_type=fish_omni` → feed `FusionTextConfig` directly, bypass `LLMEngine`;
  (4) GDN state slots are never zeroed on reuse → call `reset_state_slot()` per
  request; (5) attention is Triton + flash-attn only, prefix cache and CUDA graphs
  are disabled for hybrid models, and MoE is a Python loop over 256 expert modules
  (expect slow decode vs the 46 ms/frame budget).

## Package 4 wiring checklist (from the three reports)

1. Environment on the VM: separate venv from `~/drama3/.venv` (vLLM must stay
   intact). Code is staged at `~/drama3/nanovllm-port/`. `uv sync --extra gpu`;
   Verify `import_fusion()` picks the real Triton kernels.
   **Toolkit caveat:** the VM has only `/usr/local/cuda-12.9` and `nvcc` is not on
   PATH, while torch is `2.13.0+cu132`. Building flash-attn from source therefore
   needs either `sudo apt-get install cuda-toolkit-13-x` (passwordless sudo works) or
   a prebuilt flash-attn wheel matching torch 2.13 / cu13 — or, simplest for the MVP,
   swap fusion's attention to `torch.nn.functional.scaled_dot_product_attention`
   (eager, B=1) and defer flash-attn to the optimisation pass. Do not spend more than
   ~20 min on flash-attn before falling back.
2. `cfg = Drama3Config.from_dir(model_dir)`; `fcfg = cfg.fusion_text_config()`;
   `model = build_backbone(fcfg, ...)`; `load_backbone(model_dir, device="cuda",
   dtype=torch.bfloat16, dequant_experts=True, lm_head_rows=cfg.semantic_logit_rows())`
   (lm_head slicing saves ~1 GB; memory budget: ~67 GB weights, ~25 GB for KV + GDN
   slots + tower + codec). `allocate_caches()`, `ensure_process_group()`.
3. `tower = CodebookTower(TowerConfig.drama3(), semantic_begin_id=248099).to(bf16)`;
   load the 192 tower tensors (`tower.load_state`, strict by name) straight from the
   shards — `load_backbone` skips them.
4. Prompt: `frames = build_prompt_frames(tok, text, [Reference(ref_text, ref_codes)])`
   → `[T, 11]`. Input embeddings for the backbone: start from `embed_tokens(frames[:,0])`
   and replace semantic rows with `tower.backbone_input_embedding(token_embeds, frames)`
   (`(E_lm + E_cb[codebook1]) / sqrt(2)`); inject through fusion's
   `image_embeds` / `image_token_mask` scatter hook (it *replaces* the token
   embedding, so pass the full sum).
5. Prefill: run the backbone over all T rows; keep all-row final-norm hidden states
   (`model(...)`, not `compute_logits`, which returns only the last row);
   `cache = tower.new_cache(); tower.prefill(h[0:T], frames, cache)`.
6. Decode loop (B=1): logits of the last row → `MainTokenSampler(..., exclude_semantic_zero=True)
   .sample(logits, history, SamplingParams(1.0, 30, 0.9), generator)`; stop on
   token 248046; else `frame = tower.step(h_last, main_token, cache)` → `[11]`;
   next backbone input embedding = `tower.backbone_input_embedding(embed_tokens(frame[:1]), frame[None])`;
   append `frame[1:]` (10 codes) to the output; `max_new_tokens` 500.
7. `wav = FishDAC.load(codec_path, device="cuda", dtype=torch.float32).decode(codes[frames, 10])`
   (column 0 = semantic code 0..4095 — strip the 248099 offset; EOS frame excluded).
8. Validate: 20 Seed-TTS lines from `../results/manifest.jsonl` (pre-encoded
   `ref_audio_codes`); WER with Whisper-large-v3 vs the vLLM-Omni outputs in
   `~/drama3/runs/<best>_c1/wavs/`; check `norm_topk_prob` behaviour (not in Drama 3's
   config; fusion renormalizes top-k like HF default True) by comparing first-step
   logits against vLLM-Omni if WER is off.
9. Measure: ms per decode step (backbone, tower, total), prefill time, time to
   first frame; write `BENCH.md` alongside the vLLM-Omni numbers (c=1: 48 ms TTFA,
   ~43 ms/frame on the same card).
