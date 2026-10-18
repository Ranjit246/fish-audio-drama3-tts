# GPU validation of patch 0002 (fish_omni attention word timestamps)

Patch: `upstream/patches/0002-fish-omni-attention-word-timestamps.patch`, applies on
top of 0001 (load-adaptive codec ramp) over vllm-omni `c470395`, vLLM 0.31.0.
Nothing below needs more than the GPU box that already serves Drama 3; stage 0
must be restarted once with the overlay.

## 0. Put the patched tree on the VM (no GPU)

```bash
# from Drama3-vllm-omni/ on the Mac; only the files of 0001 + 0002, nothing else
# DRAMA3_VM=user@gpu-host (ssh -o BatchMode=yes); the files of 0002 (the
# patch's --stat list), nothing else
rsync -az --files-from=<(grep '^+++ b/' upstream/patches/0002-fish-omni-attention-word-timestamps.patch | cut -c7-) \
    vllm-omni/ $DRAMA3_VM:drama3/vllm-omni-dev/
```

CPU tests (same way as for 0001; `~/drama3/.venv` is not touched):

```bash
ssh -o BatchMode=yes $DRAMA3_VM 'cd ~/drama3/vllm-omni-dev && VIRTUAL_ENV=$HOME/drama3/.venv CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD \
    ~/.local/bin/uv run --active --no-project --with pytest --with pytest-asyncio --with pytest-mock --with pytest-xdist \
    python -m pytest -q -p no:cacheprovider \
    tests/model_executor/models/fish_omni/test_fish_omni_alignment.py \
    tests/model_executor/models/fish_omni/test_fish_omni_prompt.py \
    tests/model_executor/models/fish_omni/test_fish_omni_model_state.py \
    tests/model_executor/stage_input_processors/test_fish_omni_async_chunk.py \
    tests/test_attention_alignment.py \
    tests/entrypoints/openai_api/test_fish_omni_word_timestamps.py \
    tests/entrypoints/openai_api/test_fish_omni_reference_codes.py'
```

(`pytest-xdist` because the repo's `addopts` pass `--dist=loadgroup`.)

Expected: all pass (the two tokenizer-backed tests run, since the VM has the
checkpoint cached). The model-state, adapter and SSE tests import vLLM and
were NOT run on the Mac; the others passed there under a vLLM stub.

## 1. Start stage 0 with the capture (one server restart)

The overlay `vllm_omni/deploy/fish_omni_word_timestamps.yaml` sets
`hf_overrides: {fish_omni_word_timestamps: true}` on stage 0 (default head set,
512 text tokens). It extends `fish_omni.yaml`; stack it on the low-latency
profile by changing its `base_config` if that is the profile under test.

```bash
# the serving venv must import the patched tree: PYTHONPATH through SERVE_ENV
SERVE_ENV="PYTHONPATH=$HOME/drama3/vllm-omni-dev" \
    scripts/serve.sh wts $HOME/drama3/vllm-omni-dev/vllm_omni/deploy/fish_omni_word_timestamps.yaml
grep -m1 "word-timestamp capture" ~/drama3/logs/server_wts.log
#   -> "fish_omni: word-timestamp capture on heads L27H9, L27H6, L27H0, L31H4, L31H6, L31H0 (max 512 text tokens, 96.8 MiB of stash)"
```

What to check in the log before sending anything:

- The line above, once, from the stage-0 worker.
- No torch.compile / Dynamo warning or error naming `register_forward_pre_hook`,
  `_call_impl`, `AlignmentCapture.capture` or a graph break: the pre-hooks on
  `layers.27.self_attn.attn` and `layers.31.self_attn.attn` are traced into
  the compiled backbone and captured into the FULL decode graphs. This is the
  one thing the CPU tests cannot show. If compilation fails here, the fallback
  is `compilation_config.cudagraph_mode: PIECEWISE` for stage 0 (slower) and
  the patch needs the hook replaced by an explicit call in
  `FishOmniForConditionalGeneration.forward` -- report the traceback.
- Graph capture completes for every size in `cudagraph_capture_sizes`
  (same "Graph capturing finished" lines as without the overlay).

## 2. One streaming request (SSE)

Zero-shot first (no reference), the Phase 0 example sentence:

```bash
curl -sN http://127.0.0.1:8091/v1/audio/speech -H 'Content-Type: application/json' -d '{
  "input": "It is situated across the northeastern rim of the older and larger crater Danjon.",
  "stream": true, "response_format": "pcm", "word_timestamps": true, "seed": 7
}' | tee /tmp/wts.sse | grep -c "speech.alignment.delta"
grep -A1 "speech.alignment.delta" /tmp/wts.sse | grep data: | tail -1 | cut -c1-400
```

Check:

- Events in order: `speech.audio.delta` ... with `speech.alignment.delta` only
  after an audio delta, a last `speech.alignment.delta` whose 14 words are all
  `"final": true` (the last with `tail_silence_ms`), then `speech.audio.done`.
- Words are exactly the whitespace split of the text, `start_ms` monotonic,
  each `end_ms` <= the next `start_ms`, `resolution_ms` 46.44, integer ms.
- A word reported `final` in an early event has the same span in the final list.
- The first `speech.audio.delta` arrives as fast as without `word_timestamps`
  (TTFA): compare `[SpeechE2E] ... first_chunk_ms` for the same request with
  and without the flag.
- Play the pcm (44.1 kHz s16le mono: `sox -t raw -r 44100 -e signed -b 16 -c 1`)
  against the spans; boundaries should be within 1-2 frames (46-93 ms).

Negative checks (both must be HTTP 400 with the stated message):

```bash
curl -s http://127.0.0.1:8091/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input":"Hi.","stream_format":"audio","response_format":"pcm","word_timestamps":true}'   # "...streams as speech.alignment.delta SSE events..."
# with the stock deploy (no overlay): "...needs stage 0 launched with hf_overrides fish_omni_word_timestamps: true..."
```

## 3. One non-streaming request (header)

```bash
curl -s -D /tmp/wts.h http://127.0.0.1:8091/v1/audio/speech -H 'Content-Type: application/json' -d '{
  "input": "It is situated across the northeastern rim of the older and larger crater Danjon.",
  "response_format": "wav", "word_timestamps": true, "seed": 7
}' --output /tmp/wts.wav && grep -i x-word-timestamps /tmp/wts.h
```

Check: `X-Word-Timestamps: [{"word": "It", "start_ms": 0, "end_ms": ...}, ...,
{"word": "Danjon.", ..., "tail_silence_ms": ...}]`, 14 words, same spans as the
streaming final list for the same seed (the stage-0 sampling is seeded; the
decode is deterministic).

## 4. Compare with the Phase 0 offline decode

Two conventions differ from Phase 0, on purpose:

1. Frame clock: the row scored in the step that *samples* frame k is frame k's
   row (Phase 0 timed the row whose *input* was frame s as frame s, i.e. one
   row later; it called this off-by-at-most-one). Expect the serving spans about
   one frame (46 ms) later than a Phase 0 decode of the same rows.
2. Reduction: softmax over text tokens -> mean over heads (no z-score over
   time), because the z-score is undefined on a streaming prefix
   (`tests/test_attention_alignment.py` shows the prefix path wandering with
   it). `reduce_matrix(..., time_norm=True)` is the Phase 0 variant.

Apples to apples, offline, deterministic, on the Phase 0 capture of this very
sentence (`req_10.npz`, cloned, 126 steps) if `~/drama3/align_capture/` still
exists:

```bash
cd ~/drama3 && PYTHONPATH=$HOME/drama3/vllm-omni-dev:$HOME/drama3/timestamps .venv/bin/python - <<'EOF'
import json, numpy as np
from d3align.capture import read_capture, read_requests, selected_rows, target_cols_of
from d3align.tokenizer import load_tokenizer
from vllm_omni.utils.attention_alignment import reduce_matrix, dtw_path, words_from_path, WordAlignmentDecoder
from vllm_omni.model_executor.models.fish_omni.prompt import target_text_token_offsets
heads = [(h["layer"], h["head"]) for h in json.load(open("timestamps/sweep/out/alignment_heads.json"))["heads"]]
cap = read_capture("align_capture/req_10.npz")
req = [r for r in read_requests("timestamps/sweep/out/requests.jsonl") if r["prompt_len"] == cap.prompt_len][0]
raw = selected_rows(cap, heads)[:, :, list(target_cols_of(req))]      # [steps, 6, 16] target columns only
tok = load_tokenizer(); offs = target_text_token_offsets(tok, req["text"])
for norm in (True, False):
    words = words_from_path(dtw_path(reduce_matrix(raw, time_norm=norm)), req["text"], offs)
    print("time_norm", norm, [(w["word"], round(w["start_ms"]), round(w["end_ms"])) for w in words])
dec = WordAlignmentDecoder(req["text"], offs); dec.push(raw); print("decoder", [(w["word"], w["start_ms"], w["end_ms"]) for w in dec.finish()])
EOF
```

Expect `time_norm True` to reproduce `timestamps/results/example_req_10.words.json`
exactly, and `time_norm False` / `decoder` to agree with it within a frame on
most boundaries (report any word that moves by more than two frames). Then
send the same sentence with the same reference (`ref_audio_codes` and
`ref_text` of utt `common_voice_en_20080005-common_voice_en_20080006` from
`results/manifest.jsonl`, as `scripts/bench.py` sends them) with
`word_timestamps: true`, and compare the final list against the offline one by
ear: sampling is not the same take, so word order and plausibility, not equal
numbers. `Danjon.` should again end at the last frame with `tail_silence_ms`
near the silence you hear (Phase 0: ~1.3 s span for a ~0.5 s word).

## 5. Cost and no-regression

- Same seed, same request, overlay on vs. stock deploy: identical audio bytes
  without `word_timestamps` (the capture never touches the forward's values),
  and identical audio with it.
- `scripts/bench.py` at c=16 and c=64 with the overlay (requests without
  `word_timestamps`, then with it) vs. the 0001 numbers: audio-s/s and TTFA p50
  within noise (the einsum reads ~100 MiB of stash per step at 128 streams,
  ~0.3 ms, under 1% of a step).
- `nvidia-smi` stage-0 memory: +~100 MiB (stash) + 1.5 MiB (logits) over stock.
- A request with a text over 512 tokens: audio returned, no timestamps, one
  warning "word timestamps cover up to 512" in the stage-0 log.
- A preempted-and-resumed request (c=128 with a tight `gpu_memory_utilization`)
  still produces a complete, monotonic word list: the resumed prefill
  re-stashes its text keys.
