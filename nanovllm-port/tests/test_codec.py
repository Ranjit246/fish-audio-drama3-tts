"""Work package 3 (codec) tests: vendored Fish Speech S2 Pro DAC on CPU.

Fast tests need no weights. The ``slow`` tests download ``codec.pth`` (1.87 GB)
from ``fishaudio/drama-3`` and one reference clip from the
``zhaochenyang20/seed-tts-eval`` dataset into the Hugging Face cache; they run
by default and skip only when the download fails (offline).
"""

from __future__ import annotations

import json
import math
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # until the package is installed via pyproject
    sys.path.insert(0, str(ROOT))

from nanovllm_drama3.codec import (  # noqa: E402
    DAC_HOP_LENGTH,
    DAC_NUM_CODEBOOKS,
    DAC_SAMPLE_RATE,
    FishDAC,
    build_dac_codec,
    dac_codebook_sizes,
    resample,
    to_mono_tensor,
)

CLIP_NAME = "common_voice_en_10119832.wav"
CLIP_REPO_PATH = f"en/prompt-wavs/{CLIP_NAME}"
# Built by scripts/encode_refs.py (gitignored); the manifest tests skip when it is absent.
MANIFEST = Path(__file__).resolve().parents[2] / "results" / "manifest.jsonl"
OUT_DIR = Path(__file__).resolve().parent / "_out"

torch.manual_seed(0)


# --------------------------------------------------------------------------- helpers
def _hf_download(**kwargs) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:  # pragma: no cover
        pytest.skip("huggingface_hub not installed")
    try:
        return hf_hub_download(**kwargs)
    except Exception as exc:  # offline, auth, ...
        pytest.skip(f"could not download {kwargs.get('filename')}: {exc!r}")


def _read_wav(path: str) -> tuple[np.ndarray, int]:
    """PCM wav -> (float32 [samples] or [channels, samples], sr). Stdlib first, soundfile as fallback."""
    try:
        with wave.open(path, "rb") as w:
            sr, n, ch, width = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(n)
        if width == 2:
            data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif width == 4:
            data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"unsupported sample width {width}")
        if ch > 1:
            data = data.reshape(-1, ch).T
        return data, sr
    except (wave.Error, ValueError):
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return data.T if data.shape[1] > 1 else data[:, 0], sr


def _write_wav(path: Path, wav: np.ndarray, sr: int) -> None:
    pcm = np.clip(wav, -1.0, 1.0)
    pcm = (pcm * 32767.0).round().astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _best_lag(ref: np.ndarray, est: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Lag (samples est is delayed vs ref) maximising normalised cross-correlation, via FFT."""
    n = len(ref) + len(est)
    size = 1 << (n - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(est, size) * np.conj(np.fft.rfft(ref, size)), size)
    lags = np.concatenate([np.arange(0, max_lag + 1), np.arange(-max_lag, 0)])
    vals = corr[lags]
    best = int(lags[np.argmax(vals)])
    denom = np.linalg.norm(ref) * np.linalg.norm(est) + 1e-12
    return best, float(np.max(vals) / denom)


def _align(ref: np.ndarray, est: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag >= 0:
        est = est[lag:]
    else:
        ref = ref[-lag:]
    n = min(len(ref), len(est))
    return ref[:n], est[:n]


def _snr_db(ref: np.ndarray, est: np.ndarray) -> float:
    noise = ref - est
    return float(10 * np.log10((np.sum(ref**2) + 1e-12) / (np.sum(noise**2) + 1e-12)))


def _log_mel(x: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512, n_mels: int = 80) -> np.ndarray:
    """Plain numpy log-mel spectrogram [n_mels, frames] (for a perceptual-ish distance)."""
    win = np.hanning(n_fft + 1)[:-1]
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    n_frames = 1 + (len(xp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    spec = np.abs(np.fft.rfft(xp[idx] * win, axis=1)) ** 2  # [frames, bins]
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mel = lambda f: 2595 * np.log10(1 + f / 700)  # noqa: E731
    mel_pts = np.linspace(mel(0), mel(sr / 2), n_mels + 2)
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    fb = np.zeros((n_mels, len(freqs)))
    for m in range(n_mels):
        lo, c, hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        fb[m] = np.clip(np.minimum((freqs - lo) / (c - lo + 1e-9), (hi - freqs) / (hi - c + 1e-9)), 0, None)
    return np.log10(spec @ fb.T + 1e-8).T


# -------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def codec_path() -> str:
    return _hf_download(repo_id="fishaudio/drama-3", filename="codec.pth")


@pytest.fixture(scope="session")
def dac(codec_path) -> FishDAC:
    return FishDAC.load(codec_path, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="session")
def clip() -> tuple[np.ndarray, int]:
    path = _hf_download(repo_id="zhaochenyang20/seed-tts-eval", repo_type="dataset", filename=CLIP_REPO_PATH)
    return _read_wav(path)


@pytest.fixture(scope="session")
def manifest_row() -> dict:
    if not MANIFEST.exists():
        pytest.skip(f"manifest not found: {MANIFEST}")
    with MANIFEST.open() as f:
        for line in f:
            row = json.loads(line)
            if Path(row["ref_wav"]).name == CLIP_NAME:
                return row
    pytest.skip(f"{CLIP_NAME} not in manifest")


@pytest.fixture(scope="session")
def manifest_codes(manifest_row) -> torch.Tensor:
    codes = torch.tensor(manifest_row["codes"], dtype=torch.long)  # [10, frames]
    assert codes.shape[0] == DAC_NUM_CODEBOOKS
    return codes.transpose(0, 1).contiguous()  # [frames, 10]


# ------------------------------------------------------------------------ fast tests
def test_constants():
    assert DAC_SAMPLE_RATE == 44100
    assert DAC_HOP_LENGTH == 2048
    assert DAC_NUM_CODEBOOKS == 10
    assert dac_codebook_sizes() == (4096,) + (1024,) * 9
    assert FishDAC.codebook_sizes == dac_codebook_sizes()
    assert abs(FishDAC.sample_rate / FishDAC.hop_length - 21.533) < 1e-3


def test_build_codec_structure():
    codec = build_dac_codec()
    assert codec.hop_length == 512  # prod(encoder_rates); quantizer downsamples a further 4x
    assert codec.frame_length == 2048
    assert codec.encoder is not None and codec.decoder is not None
    n_params = sum(p.numel() for p in codec.parameters())
    assert 350e6 < n_params < 450e6, n_params  # 391.4M
    for name in ("pre_module", "post_module"):
        tr = getattr(codec.quantizer, name)
        assert tr.window_size == 128 and len(tr.layers) == 8


def test_num_frames():
    f = FishDAC.__new__(FishDAC)
    assert f.num_frames(2048) == 1
    assert f.num_frames(2049) == 2
    assert f.num_frames(44100) == math.ceil(44100 / 2048)
    # 16 kHz input: samples at 44.1 kHz = ceil(441 * n / 160)
    assert f.num_frames(16000, 16000) == math.ceil(math.ceil(441 * 16000 / 160) / 2048)


def test_to_mono_layouts():
    dev, dt = torch.device("cpu"), torch.float32
    x = torch.randn(2, 1000)
    assert torch.allclose(to_mono_tensor(x, device=dev, dtype=dt), x.mean(0))
    assert torch.allclose(to_mono_tensor(x.T, device=dev, dtype=dt), x.mean(0))
    assert to_mono_tensor(np.zeros(10, dtype=np.int16), device=dev, dtype=dt).shape == (10,)
    assert to_mono_tensor([0.0, 1.0], device=dev, dtype=dt).shape == (2,)


@pytest.mark.parametrize("sr", [16000, 22050, 24000, 48000])
def test_resample_matches_torchaudio(sr):
    torchaudio = pytest.importorskip("torchaudio")
    x = torch.randn(1, sr)  # one second
    ours = resample(x, sr, 44100)
    ref = torchaudio.transforms.Resample(sr, 44100)(x)
    assert ours.shape == ref.shape
    assert torch.allclose(ours, ref, atol=1e-6, rtol=0), (ours - ref).abs().max()


def test_resample_identity_and_length():
    x = torch.randn(3, 1234)
    assert resample(x, 44100, 44100) is x
    y = resample(x, 16000, 44100)
    assert y.shape == (3, math.ceil(1234 * 441 / 160))


# ------------------------------------------------------------------------ slow tests
@pytest.mark.slow
def test_load(dac: FishDAC, codec_path):
    rep = dac.report
    size_gb = Path(codec_path).stat().st_size / 1e9
    print(
        f"\ncodec.pth {size_gb:.2f} GB, {rep.num_state_keys} keys, {rep.num_parameters / 1e6:.1f}M params, "
        f"baked {rep.baked_weight_norms} weight norms, missing={len(rep.missing_keys)} unexpected={len(rep.unexpected_keys)}"
    )
    if rep.missing_keys:
        print("missing:", rep.missing_keys[:10])
    if rep.unexpected_keys:
        print("unexpected:", rep.unexpected_keys[:10])
    # Every weight-bearing key must come from the checkpoint; non-persistent buffers aside.
    assert not rep.missing_keys
    assert not rep.unexpected_keys
    assert dac.hop_length == 2048 and dac.num_codebooks == 10 and dac.sample_rate == 44100
    assert next(dac.codec.parameters()).dtype == torch.float32
    assert not dac.codec.training


@pytest.mark.slow
def test_round_trip_real_clip(dac: FishDAC, clip):
    wav, sr = clip
    ref44 = dac.prepare_reference(wav, sr).numpy()
    t0 = time.perf_counter()
    codes = dac.encode_reference(wav, sr)
    t_enc = time.perf_counter() - t0

    assert codes.dtype == torch.long
    assert codes.ndim == 2 and codes.shape[1] == DAC_NUM_CODEBOOKS
    frames = codes.shape[0]
    assert frames == dac.num_frames(len(ref44)) == math.ceil(len(ref44) / DAC_HOP_LENGTH)
    for i, size in enumerate(dac.codebook_sizes):
        assert 0 <= int(codes[:, i].min()) and int(codes[:, i].max()) < size

    t0 = time.perf_counter()
    out = dac.decode(codes)
    t_dec = time.perf_counter() - t0
    assert out.dtype == torch.float32 and out.ndim == 1
    assert out.shape[0] == frames * DAC_HOP_LENGTH
    assert torch.isfinite(out).all()
    est = out.numpy()

    # Quality: best-lag waveform correlation / SNR plus a log-mel distance.
    n = len(ref44)
    lag, corr = _best_lag(ref44, est[:n], max_lag=DAC_HOP_LENGTH)
    r, e = _align(ref44, est[:n], lag)
    snr = _snr_db(r, e)
    rms_ratio = float(np.sqrt(np.mean(e**2)) / (np.sqrt(np.mean(r**2)) + 1e-12))
    mel_l1 = float(np.mean(np.abs(_log_mel(r, 44100) - _log_mel(e, 44100))))
    dur = n / 44100
    print(
        f"\nclip {CLIP_NAME}: {dur:.2f}s @ {sr} Hz -> {frames} frames; "
        f"encode {t_enc:.2f}s ({t_enc / dur:.2f} s/s), decode {t_dec:.2f}s ({t_dec / dur:.3f} s per s of audio); "
        f"lag {lag} samples, corr {corr:.3f}, SNR {snr:.2f} dB, rms ratio {rms_ratio:.2f}, log-mel L1 {mel_l1:.3f}"
    )
    # A GAN-trained neural codec need not preserve phase, so waveform SNR can be low;
    # these assert sanity (audio present, same loudness, spectrally close), not fidelity.
    assert 0.5 < rms_ratio < 2.0
    assert mel_l1 < 0.6
    assert corr > 0.1

    _write_wav(OUT_DIR / f"{Path(CLIP_NAME).stem}_roundtrip.wav", est, 44100)
    _write_wav(OUT_DIR / f"{Path(CLIP_NAME).stem}_ref44k.wav", ref44, 44100)


def _semantic_distances(dac: FishDAC, wav, sr) -> torch.Tensor:
    """Squared L2 distances [frames, 4096] of the (normalised) semantic latent to every semantic code."""
    import torch.nn.functional as F

    audio = dac.prepare_reference(wav, sr)[None, None]
    pad = math.ceil(audio.shape[-1] / DAC_HOP_LENGTH) * DAC_HOP_LENGTH - audio.shape[-1]
    audio = F.pad(audio, (0, pad))
    q = dac.codec.quantizer
    with torch.inference_mode():
        z = q.pre_module(q.downsample(dac.codec.encoder(audio)))
        vq = q.semantic_quantizer.quantizers[0]
        enc = F.normalize(vq.in_proj(z)[0].T)
        cb = F.normalize(vq.codebook.weight)
        return enc.pow(2).sum(1, keepdim=True) - 2 * enc @ cb.T + cb.pow(2).sum(1, keepdim=True).T


@pytest.mark.slow
def test_codes_match_manifest(dac: FishDAC, clip, manifest_codes):
    """Our CPU codes vs the codes vLLM-Omni produced (on GPU) for the same clip.

    The manifest codes were produced on another machine / device, so exact equality is not
    guaranteed: nearest-neighbour lookups flip when two codebook entries are nearly equidistant.
    We require (a) identical frame count, (b) a low semantic mismatch rate where every mismatch is
    a near-tie (the manifest's code is our 2nd-nearest entry, within a tiny margin), (c) bounded
    residual mismatch, and (d) that both code sets decode to the same audio.
    """
    wav, sr = clip
    codes = dac.encode_reference(wav, sr)
    assert codes.shape == manifest_codes.shape, (codes.shape, manifest_codes.shape)
    mismatch = (codes != manifest_codes).float().mean(dim=0)  # per codebook
    print("\nper-codebook mismatch rate vs manifest:", [f"{m:.4f}" for m in mismatch.tolist()])

    # (b) semantic codebook: rare, and only near-ties.
    assert mismatch[0].item() <= 0.05, mismatch[0]
    dist = _semantic_distances(dac, wav, sr)
    order = dist.argsort(dim=1)
    assert torch.equal(order[:, 0], codes[:, 0])  # our codes are the argmin of these distances
    margins = dist.sort(dim=1).values
    rel_margin = (margins[:, 1] - margins[:, 0]) / margins[:, 0]
    for f in (codes[:, 0] != manifest_codes[:, 0]).nonzero().flatten().tolist():
        rank = int((order[f] == manifest_codes[f, 0]).nonzero())
        print(f"  semantic frame {f}: ours {int(codes[f, 0])} manifest {int(manifest_codes[f, 0])} "
              f"(rank {rank} by our distance, relative margin {rel_margin[f]:.2e}; median margin {rel_margin.median():.2e})")
        assert rank == 1, rank
        assert rel_margin[f] < 1e-2, rel_margin[f]

    # (c) residual codebooks: deeper residuals have smaller margins, so more flips; bound loosely.
    assert mismatch[1:].max().item() < 0.35, mismatch

    # (d) same audio: decode both and compare waveforms directly (both decodes are deterministic here).
    ours = dac.decode(codes).numpy()
    theirs = dac.decode(manifest_codes).numpy()
    snr = _snr_db(theirs, ours)
    print(f"  decode(our codes) vs decode(manifest codes): SNR {snr:.2f} dB")
    assert snr > 10.0


@pytest.mark.slow
def test_decode_manifest_codes_to_wav(dac: FishDAC, manifest_row, manifest_codes):
    frames = manifest_codes.shape[0]
    t0 = time.perf_counter()
    wav = dac.decode(manifest_codes)
    dt = time.perf_counter() - t0
    assert wav.shape[0] == frames * DAC_HOP_LENGTH
    assert torch.isfinite(wav).all() and wav.abs().max() > 0.01
    out = OUT_DIR / f"{Path(CLIP_NAME).stem}_manifest_decode.wav"
    _write_wav(out, wav.numpy(), DAC_SAMPLE_RATE)
    dur = wav.shape[0] / DAC_SAMPLE_RATE
    print(f"\ndecoded {frames} manifest frames -> {dur:.2f}s in {dt:.2f}s ({dt / dur:.3f} s per s); wrote {out}")
    print("ref_text:", manifest_row["ref_text"])


@pytest.mark.slow
def test_decode_batch_matches_single(dac: FishDAC, manifest_codes):
    a = manifest_codes
    b = manifest_codes[: a.shape[0] // 2]
    single = [dac.decode(a), dac.decode(b)]
    batched = dac.decode_batch([a, b])
    for s, bt in zip(single, batched):
        assert s.shape == bt.shape
        # Right zero-padding of the shorter sequence must not leak into its valid samples (causal decoder).
        assert torch.allclose(s, bt, atol=1e-4), (s - bt).abs().max()


@pytest.mark.slow
def test_decode_eos_like_codes_do_not_crash(dac: FishDAC, manifest_codes):
    codes = manifest_codes.clone()
    codes[-1, 0] = 4096  # EOS / out-of-range semantic id gets clamped like the reference
    wav = dac.decode(codes)
    assert wav.shape[0] == codes.shape[0] * DAC_HOP_LENGTH
