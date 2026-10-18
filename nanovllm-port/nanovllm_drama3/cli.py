"""Command line for the Drama 3 nano-vllm engine.

    # one line, voice cloned from a manifest row's pre-encoded reference codes
    python -m nanovllm_drama3.cli --text "Hello there." --ref-codes-from-manifest <utt> --out x.wav

    # N manifest rows -> out-dir/<utt>.wav + results.jsonl (timings per line)
    python -m nanovllm_drama3.cli batch --manifest ~/drama3/manifest.jsonl --limit 20 --out-dir out/

    # WER (Whisper-large-v3) of out-dir/<utt>.wav, optionally paired with another run's wavs
    python -m nanovllm_drama3.cli wer --manifest ... --hyp-dir out/ --ref-dir ~/drama3/runs/final_c_c1/wavs

Manifest rows carry ``utt``, ``text``, ``ref_text`` and ``codes`` as ``[10, frames]``
(transposed to ``[frames, 10]`` here).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import wave
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

SUBCOMMANDS = ("synth", "batch", "wer")


# ---------------------------------------------------------------------------
# wav / manifest helpers
# ---------------------------------------------------------------------------


def write_wav(path: str | Path, wav: torch.Tensor | np.ndarray, sample_rate: int) -> None:
    """16-bit PCM mono (stdlib only)."""
    data = wav.detach().cpu().numpy() if isinstance(wav, torch.Tensor) else np.asarray(wav)
    pcm = (np.clip(data.astype(np.float32), -1.0, 1.0) * 32767.0).astype("<i2")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """float32 mono ``[samples]`` and its sample rate (soundfile if present, else stdlib PCM)."""
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return data.mean(axis=1), int(sr)
    except ImportError:
        with wave.open(str(path), "rb") as w:
            sr, n, ch, width = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(n)
        if width == 2:
            data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif width == 4:
            data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"unsupported sample width {width} in {path}")
        if ch > 1:
            data = data.reshape(-1, ch).mean(axis=1)
        return data, int(sr)


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def manifest_codes(row: dict[str, Any], num_codebooks: int = 10) -> torch.Tensor:
    """``codes`` as stored (``[Q, frames]``) -> ``LongTensor[frames, Q]``."""
    codes = torch.as_tensor(row["codes"], dtype=torch.long)
    if codes.ndim != 2:
        raise ValueError(f"codes must be 2-D, got {tuple(codes.shape)}")
    if codes.shape[0] == num_codebooks and codes.shape[1] != num_codebooks:
        codes = codes.t()
    return codes.contiguous()


def find_row(rows: Iterable[dict[str, Any]], utt: str) -> dict[str, Any]:
    for row in rows:
        if row.get("utt") == utt:
            return row
    raise KeyError(f"utt {utt!r} not in manifest")


def _print_timings(t: dict[str, Any], prefix: str = "") -> None:
    step = t["step_ms"]
    print(
        f"{prefix}rows={t['prompt_rows']} frames={t['frames']} stop={t['stop_reason']} "
        f"prefill={t['prefill_ms']:.1f}ms (bb {t['backbone_prefill_ms']:.1f} / tower {t['tower_prefill_ms']:.1f}) "
        f"ttff={t['time_to_first_frame_ms']:.1f}ms "
        f"step={step['mean']:.1f}ms (bb {t['backbone_step_ms']['mean']:.1f} / tower {t['tower_step_ms']['mean']:.1f} "
        f"/ sampler {t['sampler_step_ms']['mean']:.1f}) codec={t['codec_ms']:.1f}ms "
        f"audio={t['audio_seconds']:.2f}s total={t['total_ms'] / 1000:.2f}s rtf={t['rtf']:.3f}"
    )


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------


def _load_engine(args: argparse.Namespace):
    from nanovllm_drama3.engine import Drama3Engine

    return Drama3Engine.load(
        args.model_dir, device=args.device, max_seq_len=args.max_seq_len, compile=args.compile,
    )


def cmd_synth(args: argparse.Namespace) -> int:
    ref_text, ref_codes = None, None
    if args.ref_codes_from_manifest:
        row = find_row(read_manifest(args.manifest), args.ref_codes_from_manifest)
        ref_text, ref_codes = row["ref_text"], manifest_codes(row)
        if args.text is None:
            args.text = row["text"]
    if args.text is None:
        print("error: --text (or --ref-codes-from-manifest) is required", file=sys.stderr)
        return 2
    engine = _load_engine(args)
    for i in range(args.repeat):
        wav, frames, timings = engine.synthesize(
            args.text, ref_text=ref_text, ref_codes=ref_codes, max_new_tokens=args.max_new_tokens,
            seed=args.seed, top_k=args.top_k, top_p=args.top_p, temperature=args.temperature,
        )
        _print_timings(timings, prefix=f"[{i}] ")
    if wav is not None:
        write_wav(args.out, wav, engine.codec.sample_rate)
        print(f"wrote {args.out} ({frames.shape[0]} frames, {timings['audio_seconds']:.2f} s)")
    if args.timings_json:
        Path(args.timings_json).write_text(json.dumps({**timings, "load": engine.load_report}, indent=1, default=str))
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    rows = read_manifest(args.manifest)[args.offset : args.offset + args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = _load_engine(args)
    (out_dir / "load.json").write_text(json.dumps(engine.load_report, indent=1, default=str))
    results = []
    with open(out_dir / "results.jsonl", "w") as fh:
        for i, row in enumerate(rows):
            utt = row["utt"]
            t0 = time.perf_counter()
            try:
                wav, frames, timings = engine.synthesize(
                    row["text"], ref_text=row["ref_text"], ref_codes=manifest_codes(row),
                    max_new_tokens=args.max_new_tokens, seed=args.seed, top_k=args.top_k,
                    top_p=args.top_p, temperature=args.temperature,
                )
            except Exception as exc:  # keep going; the WER pass skips the row
                rec = {"utt": utt, "error": repr(exc)}
                print(f"[{i}] {utt}: ERROR {exc!r}")
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                results.append(rec)
                continue
            if wav is not None:
                write_wav(out_dir / f"{utt}.wav", wav, engine.codec.sample_rate)
            timings.pop("per_step", None)
            rec = {
                "utt": utt, "text": row["text"], "frames": int(frames.shape[0]), "codes_min": int(frames.min()) if frames.numel() else None,
                "codes_max": int(frames.max()) if frames.numel() else None, "wall_s": time.perf_counter() - t0, **timings,
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            results.append(rec)
            _print_timings(timings, prefix=f"[{i}] {utt}: ")
    ok = [r for r in results if "error" not in r]
    if ok:
        print(
            f"{len(ok)}/{len(results)} ok; mean step {statistics.fmean(r['step_ms']['mean'] for r in ok):.1f} ms, "
            f"mean ttff {statistics.fmean(r['time_to_first_frame_ms'] for r in ok):.1f} ms, "
            f"mean rtf {statistics.fmean(r['rtf'] for r in ok):.3f}, "
            f"{sum(r['stop_reason'] != 'eos' for r in ok)} hit max_new_tokens"
        )
    return 0


_PUNCT = re.compile(r"[^\w\s']", re.UNICODE)


def normalize_text(text: str, whisper_normalizer=None) -> str:
    """Whisper English normalization when available, then lowercase and strip punctuation."""
    if whisper_normalizer is not None:
        try:
            text = whisper_normalizer(text)
        except Exception:
            pass
    text = text.lower()
    text = _PUNCT.sub(" ", text).replace("'", "")
    return " ".join(text.split())


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    """(edit distance in words, reference word count)."""
    r, h = ref.split(), hyp.split()
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
        prev = cur
    return prev[len(h)], len(r)


def cmd_wer(args: argparse.Namespace) -> int:
    from transformers import pipeline

    from nanovllm_drama3.codec import resample

    rows = read_manifest(args.manifest)
    by_utt = {r["utt"]: r for r in rows}
    hyp_dir = Path(args.hyp_dir)
    utts = sorted(p.stem for p in hyp_dir.glob("*.wav"))
    if args.limit:
        utts = utts[: args.limit]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs: dict[str, Any] = {"device": device}
    if device == "cuda":
        kwargs["dtype"] = torch.float16
    try:
        asr = pipeline("automatic-speech-recognition", model=args.whisper, **kwargs)
    except TypeError:  # older transformers: torch_dtype
        kwargs.pop("dtype", None)
        asr = pipeline("automatic-speech-recognition", model=args.whisper, torch_dtype=torch.float16, **kwargs)
    normalizer = getattr(asr.tokenizer, "normalize", None)

    def transcribe(path: Path) -> str:
        wav, sr = read_wav(path)
        x = torch.from_numpy(np.ascontiguousarray(wav))
        if sr != 16000:
            x = resample(x.unsqueeze(0), sr, 16000).squeeze(0)
        out = asr({"raw": x.numpy(), "sampling_rate": 16000}, generate_kwargs={"language": "english", "task": "transcribe"})
        return out["text"]

    def score(dir_: Path, label: str) -> dict[str, Any]:
        items, err, n_words = [], 0, 0
        for utt in utts:
            path = dir_ / f"{utt}.wav"
            if not path.exists() or utt not in by_utt:
                continue
            hyp_raw = transcribe(path)
            ref = normalize_text(by_utt[utt]["text"], normalizer)
            hyp = normalize_text(hyp_raw, normalizer)
            e, n = word_errors(ref, hyp)
            err += e
            n_words += n
            items.append({"utt": utt, "wer": e / max(n, 1), "ref": ref, "hyp": hyp, "hyp_raw": hyp_raw})
            print(f"[{label}] {utt} wer={e / max(n, 1):.3f} | {hyp}")
        return {
            "label": label,
            "n": len(items),
            "wer_corpus_pct": 100.0 * err / max(n_words, 1),
            "wer_mean_pct": 100.0 * statistics.fmean(i["wer"] for i in items) if items else float("nan"),
            "wer_gt50_count": sum(i["wer"] > 0.5 for i in items),
            "items": items,
        }

    report = {"hyp": score(hyp_dir, "nano-vllm")}
    if args.ref_dir:
        report["ref"] = score(Path(args.ref_dir), "vllm-omni")
    for key in ("hyp", "ref"):
        if key in report:
            r = report[key]
            print(f"{r['label']}: n={r['n']} corpus WER {r['wer_corpus_pct']:.2f}% mean-per-utt WER {r['wer_mean_pct']:.2f}% (>50%: {r['wer_gt50_count']})")
    out = Path(args.out or (hyp_dir / "wer.json"))
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")
    return 0


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model-dir", default=None, help="checkpoint dir (default: $DRAMA3_MODEL_DIR or the HF cache)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--compile", action="store_true", help="keep fusion's @torch.compile decorators")
    p.add_argument("--max-new-tokens", type=int, default=500)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--manifest", default=str(Path.home() / "drama3" / "manifest.jsonl"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m nanovllm_drama3.cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="synthesize one line")
    _common(s)
    s.add_argument("--text", default=None)
    s.add_argument("--ref-codes-from-manifest", metavar="UTT", default=None,
                   help="clone the voice of this manifest row (its ref_text + codes); --text defaults to its text")
    s.add_argument("--out", default="out.wav")
    s.add_argument("--repeat", type=int, default=1, help="synthesize N times (timing warm-up)")
    s.add_argument("--timings-json", default=None)
    s.set_defaults(func=cmd_synth)

    b = sub.add_parser("batch", help="synthesize manifest rows")
    _common(b)
    b.add_argument("--limit", type=int, default=20)
    b.add_argument("--offset", type=int, default=0)
    b.add_argument("--out-dir", default="out")
    b.set_defaults(func=cmd_batch)

    w = sub.add_parser("wer", help="Whisper-large-v3 WER of a wav directory (and optionally a reference run)")
    w.add_argument("--manifest", default=str(Path.home() / "drama3" / "manifest.jsonl"))
    w.add_argument("--hyp-dir", required=True)
    w.add_argument("--ref-dir", default=None)
    w.add_argument("--limit", type=int, default=0)
    w.add_argument("--whisper", default="openai/whisper-large-v3")
    w.add_argument("--out", default=None)
    w.set_defaults(func=cmd_wer)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in SUBCOMMANDS and argv[0] not in ("-h", "--help"):
        argv.insert(0, "synth")
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
