"""Score bench.py runs: Seed-TTS WER (Whisper-large-v3, official normalization) and
speaker similarity (WavLM-base-plus-sv cosine, reference clip vs output).

Run with the TTS server stopped (both models use the GPU).
Usage: score.py RUN_DIR [RUN_DIR ...]
"""

import json
import statistics
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, WavLMForXVector, pipeline

from vllm_omni.benchmarks.data_modules.seed_tts_eval import process_one_official

MANIFEST = Path.home() / "drama3/manifest.jsonl"


def load16k(path):
    wav, _ = librosa.load(path, sr=16000, mono=True)
    return wav


def main() -> None:
    rows = {r["utt"]: r for r in map(json.loads, open(MANIFEST))}
    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        torch_dtype=torch.float16,
        device="cuda",
    )
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").cuda().eval()

    def embed(wav):
        x = fe(wav, sampling_rate=16000, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            e = sv(**x).embeddings[0]
        return torch.nn.functional.normalize(e, dim=-1)

    for run in map(Path, sys.argv[1:]):
        results = [json.loads(line) for line in open(run / "results.jsonl")]
        items = []
        for r in results:
            wav_path = run / "wavs" / f"{r['utt']}.wav"
            if "error" in r or not wav_path.exists():
                continue
            row = rows[r["utt"]]
            wav = load16k(wav_path)
            hyp = asr({"raw": wav, "sampling_rate": 16000}, generate_kwargs={"language": "english", "task": "transcribe"})["text"]
            wer, _, _ = process_one_official(hyp, row["text"], "en")
            sim = float(embed(wav) @ embed(load16k(row["ref_wav"])))
            items.append({"utt": r["utt"], "wer": wer, "sim": sim, "runaway": r["runaway"], "hyp": hyp})
        clean = [i for i in items if not i["runaway"]]
        score = {
            "n": len(items),
            "wer_mean_pct": 100 * statistics.fmean(i["wer"] for i in items),
            "wer_mean_no_runaway_pct": 100 * statistics.fmean(i["wer"] for i in clean) if clean else float("nan"),
            "wer_gt50_count": sum(i["wer"] > 0.5 for i in items),
            "sim_mean": statistics.fmean(i["sim"] for i in items),
            "sim_median": float(np.median([i["sim"] for i in items])),
        }
        (run / "score_items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
        (run / "score.json").write_text(json.dumps(score, indent=2))
        print(run.name, json.dumps(score))


if __name__ == "__main__":
    main()
