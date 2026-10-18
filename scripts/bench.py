"""Seed-TTS-Eval streaming benchmark for Drama 3 on vLLM-Omni (#8933 methodology).

Each line clones its own reference (pre-encoded ref_audio_codes, or ref_audio
with --mode audio), streams raw PCM from /v1/audio/speech, and records every
chunk's arrival time. Warmup requests use lines outside the measured subset.

Metrics: TTFA (first audio byte), RTF (request -> last chunk / audio duration),
stall (a chunk arrives after the audio already received has played out; strict
and >100 ms), audio s/s, frames/s (21.5 frames = 1 s), runaway takes (>= 20 s).
"""

import argparse
import asyncio
import base64
import json
import random
import statistics
import time
import wave
from pathlib import Path

import aiohttp

FRAMES_PER_S = 44100 / 2048


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def body(row, args):
    b = {
        "model": args.model,
        "input": row["text"],
        "ref_text": row["ref_text"],
        "stream": True,
        "stream_format": "audio",
        "response_format": "pcm",
        "max_new_tokens": args.max_new_tokens,
    }
    if args.mode == "codes":
        b["ref_audio_codes"] = row["codes"]
    else:
        b["ref_audio"] = "data:audio/wav;base64," + base64.b64encode(Path(row["ref_wav"]).read_bytes()).decode()
    if args.top_k is not None:
        b["extra_params"] = {"top_k": args.top_k}
    return b


async def one(session, row, args, wav_dir):
    payload = body(row, args)
    t0 = time.perf_counter()
    arrivals, sizes, pcm = [], [], bytearray()
    sr = 44100
    try:
        async with session.post(args.url, json=payload) as resp:
            if resp.status != 200:
                return {"utt": row["utt"], "error": f"HTTP {resp.status}: {(await resp.text())[:300]}"}
            sr = int(resp.headers.get("X-Audio-Sample-Rate", sr))
            async for chunk in resp.content.iter_any():
                if chunk:
                    arrivals.append(time.perf_counter() - t0)
                    sizes.append(len(chunk))
                    pcm += chunk
    except Exception as e:  # noqa: BLE001
        return {"utt": row["utt"], "error": repr(e)}
    if not arrivals:
        return {"utt": row["utt"], "error": "no audio"}
    bps = 2 * sr
    dur = len(pcm) / bps
    # Player starts at the first chunk and consumes in real time.
    max_under, got = 0.0, 0
    for t, n in zip(arrivals, sizes):
        max_under = max(max_under, (t - arrivals[0]) - got / bps)
        got += n
    if wav_dir is not None:
        with wave.open(str(wav_dir / f"{row['utt']}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(bytes(pcm[: len(pcm) // 2 * 2]))
    return {
        "utt": row["utt"],
        "ttfa_ms": arrivals[0] * 1000,
        "latency_s": arrivals[-1],
        "audio_s": dur,
        "rtf": arrivals[-1] / dur if dur > 0 else float("inf"),
        "max_underrun_ms": max_under * 1000,
        "stall": max_under > 0.001,
        "stall_100ms": max_under > 0.1,
        "runaway": dur >= 20.0,
    }


async def run(rows, args, wav_dir):
    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=600)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:

        async def guarded(r, d):
            async with sem:
                return await one(session, r, args, d)

        start = time.perf_counter()
        res = await asyncio.gather(*(guarded(r, wav_dir) for r in rows))
        return res, time.perf_counter() - start


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(Path.home() / "drama3/manifest.jsonl"))
    ap.add_argument("--url", default="http://127.0.0.1:8091/v1/audio/speech")
    ap.add_argument("--model", default="fishaudio/drama-3")
    ap.add_argument("--n", type=int, default=300, help="measured lines (0 = all)")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--mode", choices=["codes", "audio"], default="codes")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=500)
    ap.add_argument("--save-wav", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.manifest)]
    order = list(range(len(rows)))
    random.Random(0).shuffle(order)
    n = len(rows) if args.n <= 0 else min(args.n, len(rows))
    measured = [rows[i] for i in order[:n]]
    spare = [rows[i] for i in order[n:]] or measured
    warm = spare[: args.warmup]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wav_dir = out / "wavs" if args.save_wav else None
    if wav_dir:
        wav_dir.mkdir(exist_ok=True)

    if warm:
        wres, _ = asyncio.run(run(warm, argparse.Namespace(**{**vars(args), "concurrency": min(args.concurrency, len(warm))}), None))
        werr = [r for r in wres if "error" in r]
        if werr:
            print("warmup errors:", werr[:2])
    res, wall = asyncio.run(run(measured, args, wav_dir))

    ok = [r for r in res if "error" not in r]
    errs = [r for r in res if "error" in r]
    audio = sum(r["audio_s"] for r in ok)
    summary = {
        "concurrency": args.concurrency,
        "mode": args.mode,
        "top_k": args.top_k,
        "n": len(res),
        "errors": len(errs),
        "ttfa_p50_ms": pct([r["ttfa_ms"] for r in ok], 0.5),
        "ttfa_p95_ms": pct([r["ttfa_ms"] for r in ok], 0.95),
        "ttfa_p99_ms": pct([r["ttfa_ms"] for r in ok], 0.99),
        "rtf_mean": statistics.fmean(r["rtf"] for r in ok) if ok else float("nan"),
        "rtf_p95": pct([r["rtf"] for r in ok], 0.95),
        "audio_s_per_s": audio / wall,
        "frames_per_s": audio * FRAMES_PER_S / wall,
        "stall_pct": 100 * sum(r["stall"] for r in ok) / max(len(ok), 1),
        "stall_100ms_pct": 100 * sum(r["stall_100ms"] for r in ok) / max(len(ok), 1),
        "runaway": sum(r["runaway"] for r in ok),
        "wall_s": wall,
    }
    (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in res))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if errs:
        print("errors:", errs[:3])
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
