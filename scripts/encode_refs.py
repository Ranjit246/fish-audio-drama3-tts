"""Pre-encode Seed-TTS-Eval reference clips into Drama 3 ref_audio_codes.

Writes manifest.jsonl: one row per meta.lst line with utt, ref_text, text,
ref_wav (path) and codes ([codebooks, frames], the /v1/audio/speech layout).
"""

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch

from vllm_omni.model_executor.models.fish_speech.dac_encoder import encode_reference_audio_codes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / "drama3/seed-tts-eval/en"))
    ap.add_argument("--model", default="fishaudio/drama-3")
    ap.add_argument("--out", default=str(Path.home() / "drama3/manifest.jsonl"))
    args = ap.parse_args()

    root = Path(args.root)
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(args.model, allow_patterns=["codec.pth", "config.json"])
    rows = []
    for line in (root / "meta.lst").read_text().splitlines():
        parts = line.strip().split("|")
        if len(parts) < 4 or not parts[3].strip():
            continue
        utt, ref_text, wav_rel, text = (p.strip() for p in parts[:4])
        wav_path = root / wav_rel
        wav, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        codes = encode_reference_audio_codes(model_path, wav.mean(axis=1), sr, device=torch.device("cuda"))
        rows.append(
            {
                "utt": utt,
                "ref_text": ref_text,
                "text": text,
                "ref_wav": str(wav_path),
                "codes": codes.cpu().T.tolist(),  # [frames, Q] -> [Q, frames]
            }
        )
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
