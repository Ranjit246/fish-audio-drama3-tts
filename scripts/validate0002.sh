#!/usr/bin/env bash
# GPU validation of patch 0002 (attention word timestamps), following upstream/VALIDATION-0002.md.
# Assumes the patched files are already rsynced into ~/drama3/vllm-omni-dev.
set -uo pipefail
cd ~/drama3; source .venv/bin/activate; export PATH=$HOME/.local/bin:$PATH
DEV=$HOME/drama3/vllm-omni-dev
TEXT="It is situated across the northeastern rim of the older and larger crater Danjon."

echo "== 0. CPU tests (vLLM-dependent ones included)"
(cd "$DEV" && VIRTUAL_ENV=$HOME/drama3/.venv CUDA_VISIBLE_DEVICES= PYTHONPATH=$DEV \
  uv run --active --no-project --with pytest --with pytest-asyncio --with pytest-mock python -m pytest -q -p no:cacheprovider \
  tests/model_executor/models/fish_omni/test_fish_omni_alignment.py \
  tests/model_executor/models/fish_omni/test_fish_omni_prompt.py \
  tests/model_executor/models/fish_omni/test_fish_omni_model_state.py \
  tests/model_executor/stage_input_processors/test_fish_omni_async_chunk.py \
  tests/test_attention_alignment.py \
  tests/entrypoints/openai_api/test_fish_omni_word_timestamps.py \
  tests/entrypoints/openai_api/test_fish_omni_reference_codes.py -k "not cuda and not gpu" 2>&1 | tail -6)

echo "== 4. offline comparison with the Phase 0 decode (req_10)"
PYTHONPATH=$DEV:$HOME/drama3/timestamps python - <<'EOF' 2>&1 | grep -v Warning
import json
from d3align.capture import read_capture, read_requests, selected_rows, target_cols_of
from d3align.tokenizer import load_tokenizer
from vllm_omni.utils.attention_alignment import reduce_matrix, dtw_path, words_from_path, WordAlignmentDecoder
from vllm_omni.model_executor.models.fish_omni.prompt import target_text_token_offsets
heads = [(h["layer"], h["head"]) for h in json.load(open("timestamps/results/alignment_heads.json"))["heads"]]
cap = read_capture("align_capture/req_10.npz")
req = [r for r in read_requests("timestamps/sweep/out/requests.jsonl") if r["prompt_len"] == cap.prompt_len][0]
raw = selected_rows(cap, heads)[:, :, list(target_cols_of(req))]
tok = load_tokenizer(); offs = target_text_token_offsets(tok, req["text"])
for norm in (True, False):
    words = words_from_path(dtw_path(reduce_matrix(raw, time_norm=norm)), req["text"], offs)
    print("time_norm", norm, [(w["word"], round(w["start_ms"]), round(w["end_ms"])) for w in words])
dec = WordAlignmentDecoder(req["text"], offs); dec.push(raw); print("decoder", [(w["word"], w["start_ms"], w["end_ms"]) for w in dec.finish()])
EOF

echo "== 1. stage 0 with capture"
rm -f GPU_FREE
SERVE_ENV="PYTHONPATH=$DEV" ./serve.sh wts "$DEV/vllm_omni/deploy/fish_omni_word_timestamps.yaml" 2>&1 | tail -3
L=logs/server_wts.log
sed -e "s/\x1b\[[0-9;]*m//g" "$L" | grep -m3 -iE "word-timestamp capture|alignment" | cut -c1-220
echo "-- compile / graph problems (expect none):"
sed -e "s/\x1b\[[0-9;]*m//g" "$L" | grep -iE "graph break|register_forward_pre_hook|AlignmentCapture|Traceback|Error" | grep -v "OOM on device" | head -8
sed -e "s/\x1b\[[0-9;]*m//g" "$L" | grep -c "Capturing CUDA graphs"

echo "== 2. streaming SSE"
curl -sN --max-time 180 http://127.0.0.1:8091/v1/audio/speech -H 'Content-Type: application/json' \
  -d "{\"input\": \"$TEXT\", \"stream\": true, \"response_format\": \"pcm\", \"word_timestamps\": true, \"seed\": 7}" > /tmp/wts.sse
echo "alignment events: $(grep -c 'speech.alignment.delta' /tmp/wts.sse)  audio events: $(grep -c 'speech.audio.delta' /tmp/wts.sse)"
echo "-- event order (run-length):"; grep "^event:" /tmp/wts.sse | uniq -c | head -30
echo "-- final alignment event:"; grep -A1 "speech.alignment.delta" /tmp/wts.sse | grep "^data:" | tail -1 | cut -c1-1200

echo "== 3. non-streaming header"
curl -s -D /tmp/wts.h --max-time 180 http://127.0.0.1:8091/v1/audio/speech -H 'Content-Type: application/json' \
  -d "{\"input\": \"$TEXT\", \"response_format\": \"wav\", \"word_timestamps\": true, \"seed\": 7}" --output /tmp/wts.wav
grep -i x-word-timestamps /tmp/wts.h | cut -c1-1200; ls -la /tmp/wts.wav | awk '{print "wav bytes", $5}'

echo "== negative: stream_format=audio must be 400"
curl -s -o /tmp/neg.txt -w "%{http_code}\n" http://127.0.0.1:8091/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input":"Hi.","stream_format":"audio","response_format":"pcm","word_timestamps":true}'; cut -c1-200 /tmp/neg.txt; echo

echo "== 5. TTFB with / without the flag (3 each, streaming, seed 7)"
for f in false true; do for i in 1 2 3; do
  curl -sN -o /dev/null -w "word_timestamps=$f ttfb=%{time_starttransfer}s\n" --max-time 180 http://127.0.0.1:8091/v1/audio/speech -H 'Content-Type: application/json' \
    -d "{\"input\": \"$TEXT\", \"stream\": true, \"response_format\": \"pcm\", \"word_timestamps\": $f, \"seed\": 7}"; done; done

echo "== cloned voice (manifest ref codes), streaming, final alignment event"
python - <<'EOF' 2>&1 | tail -4
import json, urllib.request
rows = {json.loads(l)["utt"]: json.loads(l) for l in open("manifest.jsonl")}
r = rows.get("common_voice_en_20080005-common_voice_en_20080006") or next(iter(rows.values()))
body = {"input": "It is situated across the northeastern rim of the older and larger crater Danjon.", "ref_text": r["ref_text"],
        "ref_audio_codes": r["codes"], "stream": True, "response_format": "pcm", "word_timestamps": True, "seed": 7}
req = urllib.request.Request("http://127.0.0.1:8091/v1/audio/speech", json.dumps(body).encode(), {"Content-Type": "application/json"})
last = None; n_audio = 0
with urllib.request.urlopen(req, timeout=180) as resp:
    for line in resp:
        s = line.decode().strip()
        if s.startswith("data:") and "speech.alignment.delta" in s: last = s
        if s.startswith("data:") and "speech.audio.delta" in s: n_audio += 1
print("audio events", n_audio); print(last[:1200] if last else "NO ALIGNMENT EVENT")
EOF

tmux kill-session -t server 2>/dev/null; pkill -f "[v]llm serve"; pkill -f "[n]vidia-cuda-mps-server"; pkill -f "[n]vidia-cuda-mps-control"; sleep 5
touch GPU_FREE; echo VALIDATE0002_DONE
