"""Drama 3 end-to-end engine (B = 1, eager): prompt -> backbone + codebook tower
-> frames -> DAC codec -> waveform.

    engine = Drama3Engine.load(model_dir, device="cuda")
    wav, frames, timings = engine.synthesize(text, ref_text=..., ref_codes=codes[F, 10])

Wiring (STATUS.md checklist, steps 2-7):

* backbone = fusion ``Qwen3_5MoeForCausalLM`` (bf16, experts dequantized at load,
  ``lm_head`` sliced to the 4096 semantic rows + EOS);
* input embeddings of every row go through
  ``tower.backbone_input_embedding`` and are injected with fusion's
  ``image_embeds`` / ``image_token_mask`` scatter hook (it *replaces* the token
  embedding, so the full mixed embedding is passed for every row);
* prefill runs the backbone over all prompt rows and keeps the all-row
  final-norm hidden states for the tower's teacher-forced prefill;
* decode: main token from the last row's logits (``MainTokenSampler``, RAS),
  codes 1..9 from ``tower.step``, next backbone input = embedding of the frame;
* GDN state slot and paged KV cache are reset / reused per request
  (``reset_state_slot``); the KV block table is the identity (block i holds
  positions [i*block_size, (i+1)*block_size)).

Attention on the GPU: nano-vllm-fusion calls flash-attn, which has no wheel
for torch 2.14 / CUDA 13. When ``flash_attn`` is not importable the engine
imports fusion with the SDPA reference implementations from
``backbone/cpu_fallback.py`` for the two flash-attn entry points (they are plain
torch and run on CUDA); the Triton ``store_kvcache`` kernel is kept.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from nanovllm_drama3.backbone import (
    allocate_caches,
    ensure_process_group,
    import_fusion,
    reset_state_slot,
)
from nanovllm_drama3.backbone.model import default_device_dtype
from nanovllm_drama3.config import HF_REPO_ID, Drama3Config, FusionTextConfig
from nanovllm_drama3.prompt import Reference, build_prompt_frames, load_tokenizer
from nanovllm_drama3.sampling import HEAD_SAMPLING, MainTokenSampler, SamplingParams
from nanovllm_drama3.tower import TOWER_PARAM_PREFIXES, CodebookTower, TowerConfig

Log = Callable[[str], None] | None


# ---------------------------------------------------------------------------
# timings
# ---------------------------------------------------------------------------


class _Clock:
    """perf_counter with a CUDA sync in front, so phase times are real."""

    def __init__(self, device: torch.device) -> None:
        self.cuda = device.type == "cuda"
        self.device = device

    def now(self) -> float:
        if self.cuda:
            torch.cuda.synchronize(self.device)
        return time.perf_counter()


def _ms(a: float, b: float) -> float:
    return (b - a) * 1000.0


def _stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0, "sum": 0.0}
    s = sorted(values)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": s[min(len(s) - 1, int(math.ceil(0.95 * len(s))) - 1)],
        "min": s[0],
        "max": s[-1],
        "sum": sum(values),
    }


@dataclass
class Timings:
    prompt_rows: int = 0
    frames: int = 0
    stop_reason: str = ""
    prefill_ms: float = 0.0            # backbone + tower prefill
    backbone_prefill_ms: float = 0.0
    tower_prefill_ms: float = 0.0
    time_to_first_frame_ms: float = 0.0  # prefill + first sample + first tower step
    codec_ms: float = 0.0
    total_ms: float = 0.0
    audio_seconds: float = 0.0
    backbone_step_ms: list[float] = field(default_factory=list)
    tower_step_ms: list[float] = field(default_factory=list)
    sampler_step_ms: list[float] = field(default_factory=list)

    @property
    def step_ms(self) -> list[float]:
        n = min(len(self.backbone_step_ms), len(self.tower_step_ms), len(self.sampler_step_ms))
        return [self.backbone_step_ms[i] + self.tower_step_ms[i] + self.sampler_step_ms[i] for i in range(n)]

    def as_dict(self) -> dict[str, Any]:
        rtf = (self.total_ms / 1000.0) / self.audio_seconds if self.audio_seconds > 0 else float("nan")
        return {
            "prompt_rows": self.prompt_rows,
            "frames": self.frames,
            "stop_reason": self.stop_reason,
            "prefill_ms": self.prefill_ms,
            "backbone_prefill_ms": self.backbone_prefill_ms,
            "tower_prefill_ms": self.tower_prefill_ms,
            "time_to_first_frame_ms": self.time_to_first_frame_ms,
            "backbone_step_ms": _stats(self.backbone_step_ms),
            "tower_step_ms": _stats(self.tower_step_ms),
            "sampler_step_ms": _stats(self.sampler_step_ms),
            "step_ms": _stats(self.step_ms),
            "codec_ms": self.codec_ms,
            "total_ms": self.total_ms,
            "audio_seconds": self.audio_seconds,
            "rtf": rtf,
            "per_step": {
                "backbone_ms": list(self.backbone_step_ms),
                "tower_ms": list(self.tower_step_ms),
                "sampler_ms": list(self.sampler_step_ms),
            },
        }


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------


def default_model_dir() -> Path:
    """``$DRAMA3_MODEL_DIR`` or the Hub cache snapshot of ``fishaudio/drama-3`` (offline)."""
    env = os.environ.get("DRAMA3_MODEL_DIR")
    if env:
        return Path(env).expanduser()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(HF_REPO_ID, local_files_only=True))


def load_tower_weights(
    tower: CodebookTower,
    model_dir: str | Path,
    device: torch.device,
    dtype: torch.dtype,
    progress: Log = None,
) -> set[str]:
    """Copy the 192 ``audio_decoder*`` tensors straight from the safetensors shards
    (``load_backbone`` skips them). Uses the index to open only the shards that
    hold tower tensors; falls back to scanning every shard without an index."""
    from safetensors import safe_open

    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    by_shard: dict[str, list[str]] = defaultdict(list)
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        for name, shard in weight_map.items():
            if name.startswith(TOWER_PARAM_PREFIXES):
                by_shard[shard].append(name)
    else:
        for shard in sorted(model_dir.glob("*.safetensors")):
            with safe_open(str(shard), "pt", "cpu") as f:
                names = [n for n in f.keys() if n.startswith(TOWER_PARAM_PREFIXES)]
            if names:
                by_shard[shard.name] = names
    loaded: set[str] = set()
    for shard, names in sorted(by_shard.items()):
        if progress:
            progress(f"tower: {len(names)} tensors from {shard}")
        with safe_open(str(model_dir / shard), "pt", "cpu") as f:
            tensors = {n: f.get_tensor(n).to(device=device, dtype=dtype) for n in names}
        loaded |= tower.load_state(tensors, strict=False)
        del tensors
    missing = set(dict(tower.named_parameters())) - loaded
    if missing:
        raise RuntimeError(f"{len(missing)} tower weights missing from {model_dir}, e.g. {sorted(missing)[:4]}")
    return loaded


def sdpa_varlen_prefill(
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    softmax_scale=None, causal=True, block_table=None, **_unused,
):
    """``flash_attn_varlen_func`` replacement on CUDA: per-sequence causal SDPA in
    the weights' dtype (GQA via ``enable_gqa``). q [N, H, D], k/v [N, Hkv, D]."""
    if block_table is not None:
        raise NotImplementedError("prefix cache (block_table) is not used by the Drama 3 engine")
    cq = cu_seqlens_q.tolist()
    ck = cu_seqlens_k.tolist()
    outs = []
    for i in range(len(cq) - 1):
        qi = q[cq[i]:cq[i + 1]].transpose(0, 1).unsqueeze(0)  # [1, H, L, D]
        ki = k[ck[i]:ck[i + 1]].transpose(0, 1).unsqueeze(0)
        vi = v[ck[i]:ck[i + 1]].transpose(0, 1).unsqueeze(0)
        o = F.scaled_dot_product_attention(qi, ki, vi, is_causal=causal, scale=softmax_scale, enable_gqa=True)
        outs.append(o[0].transpose(0, 1))
    return torch.cat(outs, dim=0)


def sdpa_decode_with_kvcache(
    q, k_cache, v_cache, cache_seqlens=None, block_table=None, softmax_scale=None, causal=True, **_unused,
):
    """``flash_attn_with_kvcache`` replacement on CUDA: gather the sequence's blocks
    and attend with SDPA. q [B, 1, H, D]; caches [blocks, block_size, Hkv, D]."""
    B, Lq, H, D = q.shape
    bs = k_cache.shape[1]
    outs = []
    for b in range(B):
        n = int(cache_seqlens[b])
        blocks = block_table[b][: (n + bs - 1) // bs].long()
        kb = k_cache[blocks].reshape(-1, k_cache.shape[-2], D)[:n].transpose(0, 1).unsqueeze(0)  # [1, Hkv, n, D]
        vb = v_cache[blocks].reshape(-1, v_cache.shape[-2], D)[:n].transpose(0, 1).unsqueeze(0)
        qb = q[b].transpose(0, 1)  # [H, Lq, D]
        o = F.scaled_dot_product_attention(qb.unsqueeze(0), kb, vb, is_causal=False, scale=softmax_scale, enable_gqa=True)
        outs.append(o[0].transpose(0, 1))  # [Lq, H, D]
    return torch.stack(outs, dim=0)


def import_fusion_for(device: torch.device, compile: bool = False) -> None:
    """Import fusion for ``device``: on CUDA keep Triton, and substitute SDPA for
    flash-attn when the ``flash_attn`` package is missing (no wheel for this
    torch/CUDA pair). ``compile`` keeps fusion's ``@torch.compile`` decorators."""
    if device.type == "cuda":
        have_fa = importlib.util.find_spec("flash_attn") is not None
        import_fusion(cpu_fallback=not have_fa, compile=compile)
        if not have_fa:
            from nanovllm.layers import attention

            attention.flash_attn_varlen_func = sdpa_varlen_prefill
            attention.flash_attn_with_kvcache = sdpa_decode_with_kvcache
        return
    import_fusion(cpu_fallback=True, compile=False)
    # On a box that has triton installed the Triton store_kvcache kernel survives
    # the stubbing; CPU tensors need the torch version.
    from nanovllm.layers import attention

    from nanovllm_drama3.backbone import cpu_fallback as cpu_mod

    if attention.store_kvcache is not cpu_mod.store_kvcache:
        attention.store_kvcache = cpu_mod.store_kvcache


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------


class Drama3Engine:
    """One-request-at-a-time Drama 3 synthesis on a single device."""

    def __init__(
        self,
        *,
        model: nn.Module,
        text_cfg: FusionTextConfig,
        tower: CodebookTower,
        codec: Any | None,
        tokenizer: Any,
        eos_token_id: int,
        semantic_begin_id: int,
        semantic_end_id: int,
        num_codebooks: int,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_seq_len: int = 8192,
        block_size: int = 256,
        exclude_semantic_zero: bool = True,
        state_slot: int = 0,
        caches: dict[str, Any] | None = None,
        head_sampling: SamplingParams = HEAD_SAMPLING,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = model
        self.text_cfg = text_cfg
        self.tower = tower
        self.codec = codec
        self.tokenizer = tokenizer
        self.eos_token_id = int(eos_token_id)
        self.semantic_begin_id = int(semantic_begin_id)
        self.semantic_end_id = int(semantic_end_id)
        self.num_codebooks = int(num_codebooks)
        self.block_size = int(block_size)
        self.num_blocks = max(1, math.ceil(max_seq_len / self.block_size))
        self.max_seq_len = self.num_blocks * self.block_size
        self.state_slot = int(state_slot)
        self.head_sampling = head_sampling
        self.clock = _Clock(self.device)
        self.vocab_size = int(text_cfg.vocab_size)
        self.lm_head_rows: torch.Tensor | None = getattr(model, "lm_head_rows", None)

        if caches is None:
            caches = allocate_caches(
                model, text_cfg, num_blocks=self.num_blocks, block_size=self.block_size,
                num_state_slots=self.state_slot + 1, device=self.device, dtype=dtype,
            )
        self.caches = caches
        self.block_tables = torch.arange(self.num_blocks, dtype=torch.int32, device=self.device)[None]
        self.state_indices = torch.tensor([self.state_slot], dtype=torch.int32, device=self.device)
        self.sampler = MainTokenSampler(
            eos_token_id=self.eos_token_id,
            semantic_begin_id=self.semantic_begin_id,
            semantic_end_id=self.semantic_end_id,
            exclude_semantic_zero=exclude_semantic_zero,
            device=self.device,
        )
        self.load_report: dict[str, Any] = {}

    # ---- construction -------------------------------------------------------

    @classmethod
    def load(
        cls,
        model_dir: str | Path | None = None,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        *,
        max_seq_len: int = 8192,
        block_size: int = 256,
        codec_dtype: torch.dtype = torch.float32,
        compile: bool = False,
        load_codec: bool = True,
        log: Log = print,
    ) -> "Drama3Engine":
        """Backbone (bf16, experts dequantized, lm_head sliced), tower (192 tensors
        from the shards), DAC codec and tokenizer from one checkpoint directory."""
        from nanovllm_drama3.weights import load_backbone

        model_dir = Path(model_dir) if model_dir is not None else default_model_dir()
        dev = torch.device(device)
        if dev.type == "cuda":
            if dev.index is None:
                dev = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.set_device(dev)
            torch.cuda.reset_peak_memory_stats(dev)
        cfg = Drama3Config.from_dir(model_dir)
        text_cfg = cfg.fusion_text_config(dtype=dtype)
        import_fusion_for(dev, compile=compile)
        ensure_process_group(dev)

        t0 = time.perf_counter()
        model, report = load_backbone(
            model_dir, dev, dtype, config=cfg, lm_head_rows=cfg.semantic_logit_rows(), progress=log,
        )
        model.eval()
        t_backbone = time.perf_counter()
        if log:
            log(report.summary())
            log(f"backbone loaded in {t_backbone - t0:.1f} s")

        tower_cfg = TowerConfig.from_hf_dict(cfg.audio_decoder_config)
        with default_device_dtype(dev, dtype):
            tower = CodebookTower(tower_cfg, semantic_begin_id=cfg.semantic_start_token_id)
        tower.eval()
        load_tower_weights(tower, model_dir, dev, dtype, progress=log)
        t_tower = time.perf_counter()
        if log:
            log(f"tower loaded in {t_tower - t_backbone:.1f} s")

        codec = None
        if load_codec:
            from nanovllm_drama3.codec import FishDAC

            codec = FishDAC.load(model_dir / "codec.pth", dev, codec_dtype)
        t_codec = time.perf_counter()
        tokenizer = load_tokenizer(model_dir)

        engine = cls(
            model=model, text_cfg=text_cfg, tower=tower, codec=codec, tokenizer=tokenizer,
            eos_token_id=cfg.eos_token_id, semantic_begin_id=cfg.semantic_start_token_id,
            semantic_end_id=cfg.semantic_end_token_id, num_codebooks=cfg.num_codebooks,
            device=dev, dtype=dtype, max_seq_len=max_seq_len, block_size=block_size,
        )
        engine.load_report = {
            "model_dir": str(model_dir),
            "backbone_load_s": t_backbone - t0,
            "tower_load_s": t_tower - t_backbone,
            "codec_load_s": t_codec - t_tower,
            "total_load_s": time.perf_counter() - t0,
            "backbone": report.summary(),
        }
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
            engine.load_report["peak_allocated_gb"] = torch.cuda.max_memory_allocated(dev) / 2**30
            engine.load_report["allocated_gb"] = torch.cuda.memory_allocated(dev) / 2**30
            engine.load_report["reserved_gb"] = torch.cuda.memory_reserved(dev) / 2**30
        if log:
            log(f"engine ready in {engine.load_report['total_load_s']:.1f} s"
                + (f", peak {engine.load_report['peak_allocated_gb']:.1f} GB" if dev.type == "cuda" else ""))
        return engine

    # ---- pieces -------------------------------------------------------------

    def build_frames(self, text: str, ref_text: str | None, ref_codes: torch.Tensor | None) -> torch.Tensor:
        """Prompt rows ``[T, 1 + Q]`` (CPU long) in the reference-bracket-v1 layout."""
        refs: list[Reference] = []
        if ref_codes is not None:
            if ref_text is None:
                raise ValueError("ref_codes given without ref_text")
            codes = torch.as_tensor(ref_codes, dtype=torch.long).cpu()
            if codes.ndim != 2:
                raise ValueError(f"ref_codes must be 2-D, got {tuple(codes.shape)}")
            if codes.shape[1] != self.num_codebooks and codes.shape[0] == self.num_codebooks:
                codes = codes.t().contiguous()  # [Q, F] -> [F, Q]
            refs.append(Reference(ref_text, codes))
        return build_prompt_frames(
            self.tokenizer, text, refs, semantic_begin_id=self.semantic_begin_id, num_codebooks=self.num_codebooks,
        )

    def _input_embeddings(self, frames: torch.Tensor) -> torch.Tensor:
        """Backbone input for frame rows: ``(E_lm[token] + E_cb1[code1]) / sqrt(2)`` on
        semantic rows, the token embedding elsewhere."""
        token_embeds = self.model.model.embed_tokens(frames[:, 0])
        return self.tower.backbone_input_embedding(token_embeds, frames)

    @torch.inference_mode()
    def prefill(self, frames: torch.Tensor) -> torch.Tensor:
        """Run the backbone over all ``T`` prompt rows (writes KV cache + GDN state
        of ``state_slot``); returns the final-norm hidden states ``[T, H]``."""
        from nanovllm.utils.context import reset_context, set_context

        frames = frames.to(self.device, dtype=torch.long)
        T = int(frames.shape[0])
        if T > self.max_seq_len:
            raise ValueError(f"prompt has {T} rows > max_seq_len {self.max_seq_len}")
        ids = frames[:, 0]
        positions = torch.arange(T, device=self.device)
        embeds = self._input_embeddings(frames)
        cu = torch.tensor([0, T], dtype=torch.int32, device=self.device)
        slot_mapping = torch.arange(T, dtype=torch.int32, device=self.device)
        set_context(True, cu, cu, T, T, slot_mapping, None, None, state_indices=self.state_indices)
        try:
            hidden = self.model.model(
                ids, positions, image_embeds=embeds,
                image_token_mask=torch.ones(T, dtype=torch.bool, device=self.device),
            )
        finally:
            reset_context()
        return hidden

    @torch.inference_mode()
    def decode_step(self, frame: torch.Tensor, position: int) -> torch.Tensor:
        """Feed one generated frame ``[1 + Q]`` at sequence ``position``; returns the
        final-norm hidden state ``[H]`` of that row."""
        from nanovllm.utils.context import reset_context, set_context

        frames = frame.to(self.device, dtype=torch.long).reshape(1, -1)
        embeds = self._input_embeddings(frames)
        dev = self.device
        set_context(
            False,
            slot_mapping=torch.tensor([position], dtype=torch.int32, device=dev),
            context_lens=torch.tensor([position + 1], dtype=torch.int32, device=dev),
            block_tables=self.block_tables,
            state_indices=self.state_indices,
        )
        try:
            hidden = self.model.model(
                frames[:, 0], torch.tensor([position], device=dev), image_embeds=embeds,
                image_token_mask=torch.ones(1, dtype=torch.bool, device=dev),
            )
        finally:
            reset_context()
        return hidden[0]

    def main_logits(self, hidden_row: torch.Tensor) -> torch.Tensor:
        """Full-vocabulary float32 logits ``[vocab]`` of one hidden row. With a
        sliced ``lm_head`` every row outside ``lm_head_rows`` is ``-inf``."""
        head = self.model.lm_head
        logits = F.linear(hidden_row.reshape(1, -1), head.weight, head.bias).float()[0]
        if self.lm_head_rows is None:
            return logits
        full = torch.full((self.vocab_size,), float("-inf"), dtype=torch.float32, device=logits.device)
        full[self.lm_head_rows] = logits
        return full

    def reset_request_state(self) -> None:
        reset_state_slot(self.model, self.state_slot)

    # ---- synthesis ----------------------------------------------------------

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        ref_text: str | None = None,
        ref_codes: torch.Tensor | None = None,
        max_new_tokens: int = 500,
        seed: int | None = None,
        top_k: int = 100,
        top_p: float = 0.9,
        temperature: float = 1.0,
        *,
        collect_logits: bool = False,
        decode_audio: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor, dict[str, Any]]:
        """Synthesize ``text`` (optionally cloning the voice of ``ref_codes``
        ``[F, 10]`` with transcript ``ref_text``).

        Returns ``(wav, frames, timings)``: ``wav`` float32 mono 44.1 kHz on CPU
        (None without a codec), ``frames`` LongTensor ``[F, Q]`` of generated codes
        (column 0 = semantic code 0..4095, EOS frame excluded), and the timings
        dict (``Timings.as_dict``; plus ``"candidate_logits"`` when
        ``collect_logits``: per-step float32 logits over the sampler's candidates).
        """
        frames = self.build_frames(text, ref_text, ref_codes)
        T = int(frames.shape[0])
        budget = self.max_seq_len - T
        if budget <= 0:
            raise ValueError(f"prompt of {T} rows leaves no room to generate (max_seq_len {self.max_seq_len})")
        max_new_tokens = min(int(max_new_tokens), budget)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
        params = SamplingParams(float(temperature), int(top_k), float(top_p))
        timings = Timings(prompt_rows=T)
        clock = self.clock

        self.reset_request_state()
        frames_dev = frames.to(self.device)
        t_start = clock.now()
        hidden = self.prefill(frames_dev)
        t_bb = clock.now()
        cache = self.tower.new_cache()
        self.tower.prefill(hidden, frames_dev, cache)
        t_prefill = clock.now()
        timings.backbone_prefill_ms = _ms(t_start, t_bb)
        timings.tower_prefill_ms = _ms(t_bb, t_prefill)
        timings.prefill_ms = _ms(t_start, t_prefill)

        h_last = hidden[-1]
        history: list[int] = []
        out_frames: list[torch.Tensor] = []
        logits_log: list[torch.Tensor] = []
        stop_reason = "max_new_tokens"
        for k in range(max_new_tokens):
            t0 = clock.now()
            logits = self.main_logits(h_last)
            draw = self.sampler.sample(logits, history, params, generator)
            t1 = clock.now()
            timings.sampler_step_ms.append(_ms(t0, t1))
            if collect_logits:
                logits_log.append(logits[self.sampler.candidates].cpu())
            if draw.token == self.eos_token_id:
                stop_reason = "eos"
                break
            history.append(draw.token)
            frame = self.tower.step(h_last, draw.token, cache, sampling=self.head_sampling, generator=generator)
            t2 = clock.now()
            timings.tower_step_ms.append(_ms(t1, t2))
            out_frames.append(frame[1:])
            if k == 0:
                timings.time_to_first_frame_ms = _ms(t_start, t2)
            if len(out_frames) >= max_new_tokens:
                break  # the backbone output would never be sampled
            h_last = self.decode_step(frame, T + k)
            t3 = clock.now()
            timings.backbone_step_ms.append(_ms(t2, t3))

        codes = (
            torch.stack(out_frames) if out_frames
            else torch.empty((0, self.num_codebooks), dtype=torch.long, device=self.device)
        )
        wav = None
        t_codec0 = clock.now()
        if decode_audio and self.codec is not None and codes.shape[0] > 0:
            wav = self.codec.decode(codes).detach().float().cpu()
        t_end = clock.now()
        timings.codec_ms = _ms(t_codec0, t_end)
        timings.total_ms = _ms(t_start, t_end)
        timings.frames = int(codes.shape[0])
        timings.stop_reason = stop_reason
        if wav is not None and self.codec is not None:
            timings.audio_seconds = wav.numel() / float(self.codec.sample_rate)
        elif self.codec is not None:
            timings.audio_seconds = codes.shape[0] * self.codec.hop_length / float(self.codec.sample_rate)
        result = timings.as_dict()
        if collect_logits:
            result["candidate_logits"] = logits_log
            result["candidate_ids"] = self.sampler.candidates.cpu()
        return wav, codes.cpu(), result

    # ---- diagnostics --------------------------------------------------------

    @torch.inference_mode()
    def teacher_forced_logits(self, frames: torch.Tensor) -> torch.Tensor:
        """Fresh full prefill over ``frames`` ``[N, 1 + Q]`` (prompt + generated rows);
        returns candidate logits ``[N, C]`` for every row. Used to check that the
        incremental decode path agrees with one prefill (tests / debugging)."""
        self.reset_request_state()
        hidden = self.prefill(frames)
        head = self.model.lm_head
        logits = F.linear(hidden, head.weight, head.bias).float()
        if self.lm_head_rows is not None:
            full = torch.full((hidden.shape[0], self.vocab_size), float("-inf"), dtype=torch.float32, device=logits.device)
            full[:, self.lm_head_rows] = logits
            logits = full
        return logits[:, self.sampler.candidates]


__all__ = ["Drama3Engine", "Timings", "default_model_dir", "import_fusion_for", "load_tower_weights"]
