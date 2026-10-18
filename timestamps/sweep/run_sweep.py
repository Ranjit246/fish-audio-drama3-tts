#!/usr/bin/env python3
"""Phase 0 sweep client: send ~20 Seed-TTS lines through the hooked server.

Picks N cloned lines (``ref_audio_codes`` + ``ref_text`` + ``input``, exactly
the request body of Drama3/scripts/bench.py) and N zero-shot lines (``input``
only) from results/manifest.jsonl, sends them SEQUENTIALLY (concurrency 1 --
the hook attributes decode steps to the single in-flight request), saves the
streamed PCM per utterance, and writes ``requests.jsonl`` with everything
match.py / score_heads.py / align_decode.py need: order, condition, utt, text,
ref_text, prompt_len (``prompt_length``), target columns
(``target_text_columns``), audio seconds and the expected decode-step count.

Zero-shot requests ARE accepted by the fish_omni adapter (its ``validate()``
only demands ``ref_text`` when ``ref_audio`` is present, and ``build_frames``
builds a reference-free prompt when neither codes nor audio arrive). Set
``--n-zeroshot 0`` if a future adapter rejects them.

At the end the client waits for the hook's idle flush (DRAMA3_ALIGN_IDLE_S,
default 5 s) so the LAST request's capture reaches disk before the server is
stopped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
import wave
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from d3align import CODEC_HOP, CODEC_SAMPLE_RATE  # noqa: E402
from d3align.prompt_layout import Reference, prompt_length, target_text_columns  # noqa: E402
from d3align.tokenizer import load_tokenizer  # noqa: E402

DEFAULT_MANIFEST = ROOT.parent / "results" / "manifest.jsonl"


def body(row: dict, args, cloned: bool) -> dict:
    """Request body; same shape as Drama3/scripts/bench.py ``body()`` (mode=codes)."""
    b = {
        "model": args.model,
        "input": row["text"],
        "stream": True,
        "stream_format": "audio",
        "response_format": "pcm",
        "max_new_tokens": args.max_new_tokens,
    }
    if cloned:
        b["ref_text"] = row["ref_text"]
        b["ref_audio_codes"] = row["codes"]
    if args.top_k is not None:
        b["extra_params"] = {"top_k": args.top_k}
    if args.seed is not None:
        b["seed"] = args.seed
    return b


def layout_info(tok, row: dict, cloned: bool) -> dict:
    refs = []
    transcripts, frames = [], []
    if cloned:
        codes = torch.tensor(row["codes"], dtype=torch.long).T.contiguous()  # [frames, codebooks]
        refs.append(Reference(row["ref_text"], codes))
        transcripts.append(row["ref_text"])
        frames.append(int(codes.shape[0]))
    plen = prompt_length(tok, row["text"], transcripts, frames)
    cols = target_text_columns(tok, row["text"], refs)
    return {"prompt_len": plen, "target_cols": [cols.start, cols.stop], "n_text": len(cols), "ref_frames": sum(frames)}


async def send_one(session: aiohttp.ClientSession, url: str, payload: dict, wav_path: Path | None) -> dict:
    t0 = time.perf_counter()
    pcm = bytearray()
    first = None
    sr = CODEC_SAMPLE_RATE
    async with session.post(url, json=payload) as resp:
        if resp.status != 200:
            return {"error": f"HTTP {resp.status}: {(await resp.text())[:300]}"}
        sr = int(resp.headers.get("X-Audio-Sample-Rate", sr))
        async for chunk in resp.content.iter_any():
            if chunk:
                if first is None:
                    first = time.perf_counter() - t0
                pcm += chunk
    dur = len(pcm) / (2 * sr)
    if wav_path is not None:
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(bytes(pcm[: len(pcm) // 2 * 2]))
    return {
        "audio_s": dur,
        "sample_rate": sr,
        "pcm_bytes": len(pcm),
        "ttfa_ms": (first or 0.0) * 1000,
        "latency_s": time.perf_counter() - t0,
        # one decode step per codec frame (frame 0 comes from the prefill's last row,
        # the final step emits end-of-speech), so steps ~= audio frames
        "expected_steps": int(round(dur * CODEC_SAMPLE_RATE / CODEC_HOP)),
    }


def pick_rows(rows: list[dict], n_cloned: int, n_zeroshot: int, seed: int, min_words: int, max_ref_frames: int):
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    ok = [
        rows[i]
        for i in order
        if len(rows[i]["text"].split()) >= min_words and len(rows[i]["codes"][0]) <= max_ref_frames
    ]
    cloned = ok[:n_cloned]
    zeroshot = ok[n_cloned : n_cloned + n_zeroshot]
    return cloned, zeroshot


async def main_async(args) -> int:
    rows = [json.loads(line) for line in open(args.manifest) if line.strip()]
    cloned, zeroshot = pick_rows(rows, args.n_cloned, args.n_zeroshot, args.seed_select, args.min_words, args.max_ref_frames)
    plan = [(r, True) for r in cloned] + [(r, False) for r in zeroshot]
    if not plan:
        print("nothing to send", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wav_dir = out / "wavs"
    wav_dir.mkdir(exist_ok=True)
    req_path = out / "requests.jsonl"
    if req_path.exists() and not args.append:
        req_path.unlink()

    tok = load_tokenizer(args.tokenizer)
    print(f"{len(cloned)} cloned + {len(zeroshot)} zero-shot lines -> {args.url}", flush=True)

    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        if args.warmup:
            spare = [r for r in rows if r not in cloned and r not in zeroshot][: args.warmup]
            for r in spare:
                res = await send_one(session, args.url, body(r, args, True), None)
                print(f"warmup {r['utt']}: {res.get('error') or f'{res['audio_s']:.2f}s'}", flush=True)
        n_ok = 0
        for order, (row, is_cloned) in enumerate(plan):
            cond = "cloned" if is_cloned else "zeroshot"
            info = layout_info(tok, row, is_cloned)
            rec = {
                "order": order,
                "condition": cond,
                "utt": row["utt"],
                "text": row["text"],
                "ref_text": row["ref_text"] if is_cloned else None,
                **info,
                "sent_at": time.time(),
            }
            res = await send_one(session, args.url, body(row, args, is_cloned), wav_dir / f"{order:02d}_{row['utt']}.wav")
            rec.update(res)
            if "error" in res:
                print(f"[{order}] {cond} {row['utt']}: ERROR {res['error']}", flush=True)
            else:
                n_ok += 1
                print(
                    f"[{order}] {cond} {row['utt']}: prompt_len={info['prompt_len']} "
                    f"target_cols={info['target_cols']} audio={res['audio_s']:.2f}s "
                    f"(~{res['expected_steps']} steps) ttfa={res['ttfa_ms']:.0f}ms",
                    flush=True,
                )
            with req_path.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if args.gap_s > 0 and order + 1 < len(plan):
                await asyncio.sleep(args.gap_s)

    wait = args.flush_wait_s
    print(f"done: {n_ok}/{len(plan)} ok -> {req_path}", flush=True)
    print(f"waiting {wait:.0f}s for the hook's idle flush of the last request ...", flush=True)
    time.sleep(wait)
    print("now stop the server and check $DRAMA3_ALIGN_OUT for req_<n>.npz files (python sweep/match.py)")
    return 0 if n_ok == len(plan) else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Drama3/results/manifest.jsonl (utt, ref_text, text, codes)")
    ap.add_argument("--url", default="http://127.0.0.1:8091/v1/audio/speech")
    ap.add_argument("--model", default="fishaudio/drama-3")
    ap.add_argument("--tokenizer", default="fishaudio/drama-3", help="HF id or local dir with tokenizer files")
    ap.add_argument("--n-cloned", type=int, default=10)
    ap.add_argument("--n-zeroshot", type=int, default=10, help="0 disables reference-free requests")
    ap.add_argument("--seed-select", type=int, default=0, help="line selection seed")
    ap.add_argument("--min-words", type=int, default=6, help="skip very short lines (few text columns)")
    ap.add_argument("--max-ref-frames", type=int, default=400, help="skip references longer than this (prompt rows)")
    ap.add_argument("--max-new-tokens", type=int, default=500)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None, help="sampling seed sent with every request")
    ap.add_argument("--warmup", type=int, default=0, help="extra cloned requests before the sweep (their captures become junk)")
    ap.add_argument("--gap-s", type=float, default=0.0, help="pause between requests")
    ap.add_argument("--flush-wait-s", type=float, default=8.0, help="> DRAMA3_ALIGN_IDLE_S so the last capture is flushed")
    ap.add_argument("--out", default=str(ROOT / "sweep" / "out"))
    ap.add_argument("--append", action="store_true", help="append to an existing requests.jsonl")
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
