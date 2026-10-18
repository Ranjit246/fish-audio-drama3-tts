# nano-vllm Drama 3 port — build plan (MVP)

Scope for this pass: the **MVP milestone** from FEASIBILITY.md — non-streaming,
eager, concurrency 1, experts dequantized to bf16 at load, codebook tower as a
plain PyTorch module, DAC codec run on the finished frame sequence. Goal: one
Seed-TTS line in → intelligible cloned audio out, on the RTX PRO 6000, with
per-frame time measured against vLLM-Omni's ~43 ms.

Package: `nanovllm_drama3/`. Base runtime: `xiez-njust-xmu/nano-vllm-fusion`
(already runs Qwen3.5-35B-A3B hybrid GDN+MoE in bf16), vendored under
`third_party/`. Reference implementations to port from (Apache-2.0, keep headers
and record provenance in `VENDORED.md`): vLLM-Omni `fish_omni/` and
`fish_speech/`.

## Work packages (file ownership — do not edit another package's files)

| # | Package | Owns | Needs GPU | Deliverable |
|---|---|---|---|---|
| 1 | backbone | `third_party/`, `nanovllm_drama3/backbone/`, `nanovllm_drama3/config.py`, `nanovllm_drama3/weights/`, `tests/test_backbone*`, `tests/test_weights*` | no (tiny-config CPU tests) | vendored base; Drama 3 `text_config` → fusion config; param-name map (drama-3 safetensors → fusion modules); block-FP8 (128×128 `weight_scale_inv`) → bf16 dequant with test; `load_backbone()` |
| 2 | tower + sampler + prompt | `nanovllm_drama3/tower/`, `nanovllm_drama3/sampling/`, `nanovllm_drama3/prompt/`, `tests/test_tower*`, `tests/test_sampling*`, `tests/test_prompt*` | no | codebook tower (4 shared + 9 residual layers, own causal KV) as eager PyTorch; RAS / top-k / top-p sampler; `reference-bracket-v1` prompt builder; tiny-config tests |
| 3 | codec | `nanovllm_drama3/codec/`, `tests/test_codec*` | no (codec.pth runs on CPU) | vendored Fish DAC; `encode_reference(wav, sr) -> codes[frames,10]`; `decode(codes) -> wav 44.1 kHz`; round-trip test on a real reference clip |
| 4 | integration | `nanovllm_drama3/engine.py`, `nanovllm_drama3/cli.py`, `tests/test_e2e*`, `BENCH.md` | **yes — after the benchmark** | real weights on the VM; prompt → backbone+tower frames → codec → wav; compare vs vLLM-Omni output (WER, listen); per-frame timing |

## Conventions

- Python 3.12, `uv`; one `pyproject.toml` at the root (package 1 creates it; others add deps via `uv add`).
- Local CPU dev env: `nanovllm-port/.venv` (torch CPU wheel on macOS). Triton/flash-attn paths are GPU-only; guard imports so CPU tests run.
- Tests: `pytest tests/ -m "not gpu"` must pass on CPU. Mark GPU tests `@pytest.mark.gpu`.
- Nobody but package 4 touches the GPU box; it is running the latency benchmark.
- Record every vendored file's source repo, commit and path in `VENDORED.md`.

## Known facts (do not re-derive)

- Drama 3 config: `model_type=fish_omni`; `text_config` is Qwen3.5-MoE: 40 layers,
  `layer_types` = 3×linear_attention then full_attention, repeated (10 full); hidden 2048;
  16 heads / 2 KV heads, head_dim 256; `linear_key_head_dim`=`linear_value_head_dim`=128;
  256 experts, top-8, `moe_intermediate_size` 512, shared expert 512 with gate;
  vocab 252,288. Experts FP8 E4M3 with block scales 128×128; everything else bf16.
- Quantization key in config: `quantization_config.quant_method = "bsgemm-moe"`.
- Codebooks: 10 per frame; main token = semantic code 0 (4096 values + EOS) sampled
  from the backbone lm_head restricted to `[semantic_start_token_id, +4096)` ∪ EOS;
  codes 1–9 from the tower (4 shared layers + 9 residual heads, each with its own
  causal KV history over the sequence; teacher-forced on prompt rows).
- Frame rate 44100/2048 = 21.53 Hz (46.4 ms/frame). Codec = Fish Speech S2 Pro DAC
  (`codec.pth` in the checkpoint), 44.1 kHz mono.
- Prompt layout `reference-bracket-v1` (vllm-omni `fish_omni/prompt.py`): segments
  tokenized separately; a frame row is `[token_id, code0..code9]`; reference audio
  rows have `token_id = semantic_begin + code0`.
- Sampling defaults: temperature 1.0, top_p 0.9, top_k 30, repetition-aware re-draw
  (window 16; re-draw at temperature 1.0 / top_k 30 / top_p 0.95).
- The simpler pre-#8933 eager implementation of the tower/sampler/prompt is at
  vllm-omni commit `9b300b983` (`vllm_omni/model_executor/models/fish_omni/`);
  the current main (`c470395`) version is MRv2/paged and harder to lift.
