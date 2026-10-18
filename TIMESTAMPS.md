# Word timestamps from Drama 3's own attention — design

Goal: per-word `{word, start_ms, end_ms}` alongside streamed audio, with **no
forced aligner, no ASR, no second model**, following the method proven on VoxCPM2
in `TTS-duration-predictor/VOXCPM2/nano-vllm/aligner/` (sweep heads → frozen head
mask → in-graph side matmul → softmax / z-score / monotonic DTW → word spans).

## Why Drama 3 suits this better than VoxCPM2

| | VoxCPM2 | Drama 3 |
|---|---|---|
| Text-attention events per second of audio | 6.25 (one LM step = 4 frames × 40 ms; the DiT never sees text) | **21.5** (one frame per step, every step attends to text) |
| Timestamp resolution | 160 ms | **46.4 ms** |
| Heads to sweep | 448 | **160** (10 full-attention layers × 16 heads; the 30 Gated-DeltaNet layers have no attention matrix) |
| Text columns | prompt+target tokenized as one string (BPE merges across the join) | segments tokenized separately → `target_text_columns()` is exact by construction (`nanovllm-port/nanovllm_drama3/prompt/`) |
| Where post-RoPE q/k are available | hand-patched `Attention.forward` | vLLM `Qwen3NextAttention.forward`: `q, k, v, gate = self._project_qkv_gate(qkv, positions); self.attn(q, k, v)` — a forward pre-hook on `self.attn` sees them |

Sequence layout (`reference-bracket-v1`): `system prompt` · `[Reference 1] Text: <ref transcript> Speech: <ref codes rows>` · `<|im_start|>user <|speaker:0|> TARGET TEXT <|im_end|>` · `<|im_start|>assistant <|voice|>` · generated frames. Only the TARGET TEXT rows are alignment columns; the reference transcript rows must be excluded (they would otherwise soak up attention mass, as the VoxCPM2 notes warn).

## Phase 0 — head sweep (offline, ~20 min on the RTX PRO 6000, server stopped)

1. Run ~20 Seed-TTS lines (10 zero-shot = no reference, 10 cloned) through the stock
   server with `enforce_eager` for stage 0 and a hook on the 10 `Qwen3NextAttention`
   layers that records, per decode step, `softmax(q_t · K_textᵀ / √d)` restricted to the
   target-text columns, for all 16 heads → `[steps, 16, n_text]` per layer.
   (Eager is fine offline; it is the 73–85 % throughput loss that rules it out in serving.)
2. Score each (layer, head): `path_mass × monotonicity × min(2·coverage, 1)` with the
   same definitions as `align_head/align_sweep.py`. **Never rank on path_mass alone.**
   Rank by the mean of both conditions. Write `alignment_heads.json` (top 6–8).
3. The DTW / path-mass inner loops are O(steps × text) pure Python in the VoxCPM2 code;
   port them to `numba.njit` (both the sweep and the runtime decoder), since the GIL
   makes the Python version a per-request stall in the API server.
4. Expectation from the literature (ACI, arXiv:2404.19723; Attention Guidance for
   CosyVoice2, arXiv:2509.19852) and the VoxCPM2 result: alignment heads in the
   middle third of the network; here that is full-attention layers 3–7 (indices
   11, 15, 19, 23, 27 of 40).

### Phase 0 result (2026-10-18, `fishaudio/drama-3@b2b5ee9`, 10 cloned + 10 zero-shot Seed-TTS lines)

Tooling: `timestamps/` on the `vllm-omni` branch (hook, sweep client, scorer, decoder;
31 CPU tests). Run time on the RTX PRO 6000: eager server up in 90 s, 20 lines in
~2 min, scoring 160 heads in seconds.

| rank | head | mean score | cloned | zero-shot | target-column mass |
|---:|---|---:|---:|---:|---:|
| 1 | **L27 H9** | 0.826 | 0.859 | 0.793 | 0.73 |
| 2 | L31 H4 | 0.803 | 0.830 | 0.776 | 0.74 |
| 3 | L31 H6 | 0.775 | 0.791 | 0.758 | 0.70 |
| 4 | L31 H0 | 0.755 | 0.786 | 0.724 | 0.69 |
| 5 | L27 H6 | 0.745 | 0.774 | 0.717 | 0.55 |
| 6 | L27 H0 | 0.737 | 0.773 | 0.700 | 0.54 |

- Alignment lives in full-attention layers **27 and 31** (the 7th and 8th of 10),
  the mid-to-late placement the ACI / CosyVoice2 papers report. Median head score
  0.18; best 4.3× median. Monotonicity 0.94–0.97, coverage ≥ 0.98 for the top heads.
- Target-column mass check (the VoxCPM2 trap): uniform would be **0.077** on cloned
  prompts (the reference transcript occupies most of the text axis) and 0.38
  zero-shot; the top head puts **0.73** there — 9.6× uniform cloned, 1.9× zero-shot.
  Real alignment, not degenerate attention.
- The same heads top both conditions, so one mask serves cloned and zero-shot.
- First decode (`timestamps/results/example_req_10.words.json`, "It is situated across
  the northeastern rim of the older and larger crater Danjon.", 5.8 s): 14 words,
  monotonic, contiguous, 46 ms resolution; the last word absorbs the trailing
  silence (expected: the path has nowhere else to go — Phase 2 should cap a word's
  end at the first silent frame of the waveform).
- Files: `timestamps/results/alignment_heads.json`, `sweep_cloned.json`,
  `sweep_zeroshot.json`, `requests.jsonl`, `matched.jsonl`.

### Phase 1 + 2 result (2026-10-18, patch `upstream/patches/0002-…` on the `vllm-omni` branch)

Implemented and **validated on the RTX PRO 6000** (`scripts/validate0002.sh`,
procedure in `upstream/VALIDATION-0002.md`):

- Capture runs inside the FULL decode CUDA graphs: forward pre-hooks on layers 27
  and 31 registered before compile, static stash (96.8 MiB at 128 seqs) + static
  logits buffer; server up in 165–180 s with zero compile/graph-break errors and the
  same 4 capture passes as stock.
- Non-streaming: `X-Word-Timestamps` header, 14/14 words, monotonic, 46 ms grid;
  "Danjon." 3901–4876 ms with `tail_silence_ms` 648.
- Streaming: `speech.alignment.delta` after each `speech.audio.delta`, words settle
  with a 4-frame lag, **final list identical to the non-streaming header for the same
  seed**, no `final` word ever moved across 16 events. (First version re-pushed
  earlier windows because the API accumulator kept the tensor as request-level
  state; fixed by draining it per chunk.)
- Offline check on the Phase 0 capture: the serving reduction (softmax → mean over
  heads, no time z-score) agrees with the Phase 0 decode within 1–2 frames on every
  boundary; `time_norm=True` reproduces Phase 0 exactly.
- Gate: `hf_overrides: fish_omni_word_timestamps: true` (deploy
  `fish_omni_word_timestamps.yaml`); `stream_format=audio` + timestamps → 400.
- 112 CPU tests on the VM against real vLLM 0.31.

Open items: measure the per-step cost at 64–128 streams (expected <1 %), a
preempted-and-resumed request, texts over 512 tokens, and a listening pass on 20
clips against the spans.

## Phase 1 — runtime capture inside the CUDA graph

Stage 0 replays **one FULL CUDA graph per decode step** (`fish_omni.yaml`), so the
capture must be graph-safe: fixed shapes, buffers allocated *before* capture, no
host syncs.

- **Prefill:** a pre-hook on the selected layers' `self.attn` sees `k` for all prompt
  rows; copy the target-text rows (`target_text_columns`) for the selected KV heads
  into a per-request contiguous stash `K_text[req_slot, layer, kv_head, max_text, d]`
  (bf16; `max_text` = 512 tokens is ample). The paged KV cache is never read.
- **Decode:** the same pre-hook sees `q_t` for the batch; compute
  `buf[slot, head_i, :max_text] = (q_t[head] · K_text[slot, layer, kv_head(head)]ᵀ) · d^-½`
  as one `einsum` per selected layer into a pre-allocated fp32 buffer
  `[max_num_seqs, n_sel_heads, max_text]`. Raw logits, no softmax, no masking —
  the server slices `[0, n_text)` and softmaxes, so uninitialised columns never
  leak (the VoxCPM2 lesson). Cost ≈ 6 heads × 512 × 256 MACs ≈ 0.8 MFLOP/step vs
  ~4 GFLOP for the step: negligible.
- **Hand-off:** fish_omni's MRv2 model state (`model_state.py`) already owns per-request
  slots and copies step outputs to the host each step; the alignment rows ride the
  same path (`[steps, n_sel, n_text]` per request, accumulated in the stage-0 worker,
  shipped with each codec window through the shared-memory connector as a small
  extra tensor).
- GQA: 16 query heads / 2 KV heads → `kv_head = head // 8`.

## Phase 2 — decode and API

- Per received window: `reduce_matrix` (slice to target columns → softmax → z-score
  per head over time → mean over heads) then monotonic DTW (numba) → token index per
  frame → word spans via tokenizer offset mapping; frame *k* ⇒ `[k·46.4, (k+1)·46.4) ms`
  (+ the window's left context offset). Words are re-emitted as `provisional` while
  their span can still grow, `final` once the path has moved past them.
- SSE / WebSocket event alongside `speech.audio.delta`:
  `{"type":"alignment","words":[{"word","start_ms","end_ms","final":bool}],"resolution_ms":46.4}`.
  Alignment is computed *after* each audio chunk is yielded, so TTFA and chunk pacing
  are untouched; words trail audio by ≤ 1 window.
- Non-streaming: `X-Word-Timestamps` header, same JSON.

## Validation

- Listen to 20 clips with word highlighting; plus a one-time cross-check against
  Whisper word timestamps (reporting only, never at runtime). Expect boundary error
  ≈ ±1 frame (±46 ms) on clean speech.
- Re-sweep heads only for a new base checkpoint (new voices/prompts don't move heads —
  shown on VoxCPM2 and expected here since reference audio is just prompt rows).

## Effort

Sweep tooling + sweep run: ~half a day. In-graph capture in vllm-omni (`fish_omni`
model state + attention pre-hooks + connector field): ~1–2 days. Decode/API: ~half a
day. Numba kernels: hours. Same design applies 1:1 to the nano-vllm port, where the
hook point is nano-vllm-fusion's attention layer and there is no paged cache to avoid.
