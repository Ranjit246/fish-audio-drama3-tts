# Drama 3 word timestamps — Phase 0 head sweep

Tooling for step 1–3 of [`../TIMESTAMPS.md`](../TIMESTAMPS.md): find which of
Drama 3's 160 full-attention heads (10 layers × 16 heads) track the reading
position, and freeze them in `alignment_heads.json`. Method and scoring are
the ones proven on VoxCPM2 (`TTS-duration-predictor/VOXCPM2/nano-vllm/aligner/`);
the O(steps × text) DP loops are `numba.njit` with a numpy fallback.

Nothing here touches the serving path. The hook is for the offline sweep in
eager mode at concurrency 1; the in-graph runtime capture is Phase 1.

```
timestamps/
├── hook/
│   ├── drama3_align_hook.py   .pth-loaded capture hook (inert unless DRAMA3_ALIGN_SWEEP=1)
│   ├── drama3_align.pth       one line: import drama3_align_hook
│   └── install.sh             copy both into a venv's site-packages (--uninstall to remove)
├── sweep_deploy.yaml          deploy overlay: stage 0 eager, max_num_seqs 1 (edit base_config)
├── sweep/
│   ├── run_sweep.py           client: 10 cloned + 10 zero-shot Seed-TTS lines, c=1, writes requests.jsonl
│   ├── match.py               pair req_<n>.npz captures with requests by prompt_len, skip junk
│   └── score_heads.py         160-head scoring -> sweep_<cond>.json + alignment_heads.json
├── decode/align_decode.py     one request -> word spans (reduce_matrix / dtw_path / words_from_path)
├── d3align/                   shared code: dp.py, scoring.py, decode.py, capture.py,
│                              prompt_layout.py (vendored), tokenizer.py
├── tests/                     pytest, CPU only, synthetic data + the real tokenizer
└── requirements.txt           for timestamps/.venv (torch CPU, numpy, numba, transformers, pytest, aiohttp, soundfile)
```

## What the hook records

`Qwen3NextAttention.forward(positions, hidden_states)` computes
`q, k, v, gate = self._project_qkv_gate(qkv, positions)` and calls
`self.attn(q, k, v)`. `q` is `[num_tokens, num_heads·head_dim]`, `k` is
`[num_tokens, num_kv_heads·head_dim]`, both post q_norm/k_norm and post-RoPE
(the fused Triton path returns the same layout). The hook patches `forward`
to stash `positions` and registers a forward **pre-hook on `self.attn`**, which
receives exactly the `(q, k, v)` flash attention receives — no re-computation,
no change to the model output.

| forward seen | rule | action |
|---|---|---|
| `positions[0] == 0`, > 1 row | new request's prefill | flush the previous request to `req_<n>.npz`; stash this layer's `k[:T]` as `[T, 2, 256]` fp16 |
| 1 row at `prompt_len + steps_so_far` (per layer) | decode step | `q[h] · K_prompt[h // 8]ᵀ · scaling` for all 16 heads → `[16, T]` fp16, appended |
| anything else | mixed / chunked / replayed step | ignored, logged once |

Raw logits over **all** prompt columns are stored; `match.py` / `score_heads.py`
slice the target-text columns **before** the softmax (VoxCPM2 lesson). Tensors
stay on the GPU until flush — one device→host copy per request — at the next
request's prefill, after `DRAMA3_ALIGN_IDLE_S` seconds without a forward
(default 5, background thread), and at interpreter exit. Errors inside the
capture never propagate: the hook logs once and disables itself.

Layer index: read from `module.attn.layer_name` / `qkv_proj.prefix` /
`layer_idx` (`layers.<i>`), else call order mapped onto `3,7,…,39` (logged).
`alignment_heads.json` uses the **model layer index** (3..39), not 0..9.

Environment variables: `DRAMA3_ALIGN_SWEEP=1` (arm), `DRAMA3_ALIGN_OUT` (capture
dir), `DRAMA3_ALIGN_IDLE_S` (5), `DRAMA3_ALIGN_SAVE_K=1` (also store prompt keys),
`DRAMA3_ALIGN_LAYERS` (fallback layer list), `DRAMA3_ALIGN_VERBOSE=1`,
`DRAMA3_ALIGN_TARGET=module:Class` (tests).

## VM procedure (~25 min including the 5 min server start)

The GPU box must be free of the production server (`scripts/serve.sh` kills it).

```bash
# 0. sync this folder to the VM (e.g. ../sync.sh --push, or scp -r timestamps ~/drama3/)
cd ~/drama3 && source .venv/bin/activate              # the vllm / vllm-omni venv
pip install numba                                     # only extra dependency for scoring in this venv

# 1. install the hook into the vllm venv (inert until armed)
bash timestamps/hook/install.sh python
#    -> prints "inert (set DRAMA3_ALIGN_SWEEP=1)" and "armed: target=vllm.model_executor.models.qwen3_next:Qwen3NextAttention ..."

# 2. point the overlay at this venv's stock fish_omni.yaml, then start the server EAGER at c=1
python -c 'import vllm_omni, os; print(os.path.dirname(vllm_omni.__file__))'   # fix base_config in sweep_deploy.yaml if it differs
export DRAMA3_ALIGN_SWEEP=1 DRAMA3_ALIGN_OUT=$HOME/drama3/align_capture DRAMA3_ALIGN_IDLE_S=5
rm -rf $DRAMA3_ALIGN_OUT
SERVE_ENV="DRAMA3_ALIGN_SWEEP=1 DRAMA3_ALIGN_OUT=$DRAMA3_ALIGN_OUT DRAMA3_ALIGN_IDLE_S=5" \
    scripts/serve.sh sweep timestamps/sweep_deploy.yaml
#    (serve.sh passes SERVE_ENV through tmux; the hook must be in the STAGE-0 WORKER's environment)
grep -m1 "drama3-align" logs/server_sweep.log         # "patched vllm.model_executor.models.qwen3_next.Qwen3NextAttention.forward"

# 3. run the sweep client (sequential, 10 cloned + 10 zero-shot; --n-zeroshot 0 to skip)
python timestamps/sweep/run_sweep.py --manifest results/manifest.jsonl --out timestamps/sweep/out
#    waits 8 s at the end for the hook's idle flush of the last request

# 4. stop the server, then pair captures with requests and score
tmux kill-session -t server; pkill -f "vllm serve"
python timestamps/sweep/match.py --capture-dir $DRAMA3_ALIGN_OUT --requests timestamps/sweep/out/requests.jsonl
python timestamps/sweep/score_heads.py --capture-dir $DRAMA3_ALIGN_OUT --requests timestamps/sweep/out/requests.jsonl --topk 6
#    -> timestamps/sweep/out/sweep_cloned.json, sweep_zeroshot.json, alignment_heads.json

# 5. decode one request and listen against wavs/<order>_<utt>.wav
python timestamps/decode/align_decode.py --npz $DRAMA3_ALIGN_OUT/req_3.npz \
    --requests timestamps/sweep/out/requests.jsonl --heads timestamps/sweep/out/alignment_heads.json

# 6. remove the hook again before serving for real
bash timestamps/hook/install.sh --uninstall python
```

Expected runtime: server start ~5 min (eager skips graph capture but still
loads 37 GiB), 20 requests × ~5 s audio at eager RTF ≈ 1–2 → 3–5 min,
scoring 20 × 10 layers × 16 heads with numba under a minute (several minutes
with the numpy fallback). Capture size: ~50 MB per request.

Running `scripts/serve.sh` without `SERVE_ENV` leaves the hook inert; the
server behaves exactly as stock even with the `.pth` installed.

## Sanity checks before trusting `alignment_heads.json`

1. `match.py`: every request matched, `steps` within ±3 of `~audio` (audio
   seconds × 21.53). Junk files (`steps=0`, prompt_len 4096 / tiny) are vLLM's
   profiling and dummy runs — expected, skipped.
2. `score_heads.py` prints per-condition top-20 with `tgt_mass`: the share of
   each head's **full-prompt** softmax on the target columns. Uniform is
   `n_text / prompt_len` (printed). The VoxCPM2 warning applies verbatim: a
   head can look diagonal after slicing while 90 % of its attention reads the
   reference transcript. Good VoxCPM2 heads had 59–88 % target mass (7× uniform);
   the script flags chosen heads below 2× uniform.
3. Never rank on `path_mass` alone — a head parked on one token scores ~1.0
   there with zero coverage (`tests/test_sweep_ranking.py` plants exactly that trap).
4. Best score should be several × the median (3.7× on VoxCPM2) and the winners
   should agree across cloned / zero-shot; the mask is ranked by the mean.
5. Expect winners in the middle third (model layers 11–27) per the literature.
6. `align_decode.py` spans must be monotonic, contiguous and plausible by ear.
   Decode step *s* is timed as frame *s* (`[s·46.44, (s+1)·46.44) ms`); frame 0 is
   sampled from the prefill's last row, so the convention is off by at most one
   frame — `--frame-offset` shifts it if listening says so.

## Caveats

- **Eager mode only.** With CUDA graphs the patched Python forward never runs
  during replay; the overlay sets `enforce_eager: true` and `compilation_config
  mode 0 / cudagraph_mode NONE` for stage 0. Eager was 73–85 % slower on
  VoxCPM2; this is never a serving configuration.
- **Concurrency 1 only.** The hook attributes a 1-row forward to the single
  in-flight request. `max_num_seqs: 1` in the overlay enforces it; the client
  is sequential. Any other row pattern is logged once and dropped.
- **One prefill chunk per request.** `enable_chunked_prefill` stays on
  (stock) with a 4096-row budget; Seed-TTS prompts are a few hundred rows. A
  longer prompt would prefill in two chunks and the second chunk (positions not
  starting at 0) is ignored → that request's steps never match → junk.
- The last request is flushed by the idle thread (`DRAMA3_ALIGN_IDLE_S`), not
  by SIGTERM: do not kill the server within 5 s of the last response.
- `prefix caching` must stay off (stock) so the prefill really has
  `prompt_length()` rows.
- Zero-shot requests (no `ref_audio_codes`/`ref_text`) are accepted by the
  fish_omni adapter (`validate()` requires `ref_text` only alongside
  `ref_audio`; `build_frames` builds a reference-free prompt). If a later
  adapter rejects them, pass `--n-zeroshot 0` and rank on the cloned sweep.
- Verified on CPU with synthetic data and a fake attention module; not yet
  run against a live vLLM. Assumptions to confirm from the first server log:
  the class is reached via `vllm.model_executor.models.qwen3_next` (qwen3_5.py
  imports it from there), `self.attn.layer_name` carries `layers.<i>`, and
  eager decode forwards are unpadded single rows (padded rows are tolerated).

## Tests

```bash
cd timestamps
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest tests -q            # 31 tests; D3ALIGN_NO_NUMBA=1 exercises the numpy fallback
```

Covers: numba vs numpy vs literal-reference DP equality (ties included);
planted-diagonal head ranks first in both conditions while the parked head
loses despite path_mass ≈ 1; match.py skips junk captures; the hook's
prefill/decode/flush logic, padded rows, mixed steps, error isolation and the
post-import patching on a fake `Qwen3NextAttention`; token→column→word
mapping with the real Drama 3 tokenizer (downloaded tokenizer files only) and
the `align_decode.py` CLI end to end.

## Provenance

`d3align/prompt_layout.py` is vendored verbatim from
`Drama3-nano-vllm/nanovllm-port/nanovllm_drama3/prompt/{tokens,reference_bracket_v1}.py`
(a port of vllm-omni `fish_omni/prompt.py` @ 9b300b983 / c470395).
`d3align/dp.py`, `scoring.py`, `decode.py` port
`TTS-duration-predictor/VOXCPM2/nano-vllm/aligner/{decode_runtime.py, align_head/align_sweep.py}`.
The request body in `run_sweep.py` mirrors `Drama3/scripts/bench.py`.
