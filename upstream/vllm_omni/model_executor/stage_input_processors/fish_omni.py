# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage input processor for Fish Audio Drama 3: frame stream -> DAC decoder.

Stage 0 emits one frame ``[main, c0..c9]`` per request per step. Frames whose
main token is not semantic (the end-of-speech frame) carry no audio. Windows
follow a ramp (``codec_chunk_ramp``, default ``6, 8, 12, 16, 24, 32``, then
``codec_chunk_frames``) so the first audio leaves early, and every window
carries up to ``codec_left_context_frames`` earlier frames, which the DAC
decoder trims (or skips, when it keeps the stream's codec state). With
``codec_chunk_ramp_by_load`` a stream's ramp is chosen at its first step from
the number of requests in that step (see ``fish_speech.chunk_ramp``). Windows
travel as ``[codebooks, frames]`` tensors.

A request with word timestamps also emits, per frame, its attention
alignment logits over its text (``align.logits``, ``[1, heads, text]``; see
``fish_omni.alignment``). They travel beside the window's new frames as
``codes.alignment`` ``[frames, heads, text]`` fp16 (the left context is not
repeated), so the server can time words as the audio arrives.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.model_executor.models.fish_speech.chunk_ramp import RampConfig, ramp_config_from_extra

SEMANTIC_BEGIN_ID = 248099
CODEBOOK_SIZE = 4096
NUM_CODEBOOKS = 10
DEFAULT_RAMP = (6, 8, 12, 16, 24, 32)
DEFAULT_CHUNK = 64
DEFAULT_LEFT_CONTEXT = 32
_STATE_KEY = "_fish_omni_stream"


def _config(transfer_manager: Any) -> tuple[RampConfig, int, int]:
    cached = getattr(transfer_manager, "_fish_omni_chunk_config", None)
    if cached is not None:
        return cached
    connector = getattr(transfer_manager, "connector", None)
    raw = getattr(connector, "config", {}) or {}
    cfg = raw.get("extra", raw) if isinstance(raw, dict) else {}
    cfg = cfg if isinstance(cfg, dict) else {}
    chunk = int(cfg.get("codec_chunk_frames", DEFAULT_CHUNK))
    left = int(cfg.get("codec_left_context_frames", DEFAULT_LEFT_CONTEXT))
    if chunk <= 0 or left < 0:
        raise ValueError(f"invalid fish_omni codec chunk config chunk={chunk} left={left}")
    ramps = ramp_config_from_extra(cfg, default=DEFAULT_RAMP)
    transfer_manager._fish_omni_chunk_config = (ramps, chunk, left)
    return transfer_manager._fish_omni_chunk_config


def _state(transfer_manager: Any, request_id: str) -> dict[str, Any]:
    # request_payload entries are reclaimed when a request completes or is cancelled.
    return transfer_manager.request_payload.setdefault(request_id, {}).setdefault(
        # ``codes`` holds frames ``base..``: frames no later window reads are dropped.
        _STATE_KEY,
        # ``align`` holds one [heads, text] row per kept frame for a stream
        # with word timestamps (None otherwise), trimmed with ``codes``.
        {"codes": [], "align": None, "base": 0, "emitted": 0, "windows": 0, "done": False},
    )


def _ramp(transfer_manager: Any, state: dict[str, Any], ramps: RampConfig, running: int | None) -> tuple[int, ...]:
    """The stream's window ramp, chosen at its first step and kept.

    ``running`` is the number of requests in that step, the new one included:
    the streams whose decode steps it shares, so a burst of arrivals that
    starts in one step sees the whole burst. Without it (the single-request
    path) the requests with sender state on this stage stand in:
    ``request_payload`` gains a request's entry at its first step here
    (``_state``) and loses it when the adapter reclaims the request on finish
    or abort. That count lags one step, so every request of a simultaneous
    first wave sees only itself and takes the lightest ramp. ``len`` reads
    the dict atomically while other threads reclaim.
    """
    ramp = state.get("ramp")
    if ramp is None:
        if running is None:
            running = len(transfer_manager.request_payload)
        ramp = state["ramp"] = ramps.select(max(1, running))
    return ramp


def _frames(output: Any) -> torch.Tensor | None:
    if not isinstance(output, Mapping):
        return None
    codes = output.get("codes", {})
    frames = codes.get("audio") if isinstance(codes, Mapping) else None
    if frames is None:
        frames = output.get("codes.audio")
    if not isinstance(frames, torch.Tensor) or frames.numel() == 0:
        return None
    return frames


def _alignment(output: Any) -> torch.Tensor | None:
    """The step's ``[frames, heads, text]`` alignment logits, if the request has word timestamps."""
    if not isinstance(output, Mapping):
        return None
    align = output.get("align", {})
    rows = align.get("logits") if isinstance(align, Mapping) else None
    if rows is None:
        rows = output.get("align.logits")
    if not isinstance(rows, torch.Tensor) or rows.ndim != 3:
        return None
    return rows


def _semantic(main: np.ndarray) -> np.ndarray:
    return (main >= SEMANTIC_BEGIN_ID) & (main < SEMANTIC_BEGIN_ID + CODEBOOK_SIZE)


def _request_id(request: Any) -> str:
    return str(getattr(request, "external_req_id", None) or getattr(request, "request_id", id(request)))


def fish_omni_to_dac_decoder_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    finished = bool(is_finished or (hasattr(request, "is_finished") and request.is_finished()))
    codes = align = None
    frames = _frames(multimodal_output)
    if frames is not None:
        rows = frames.detach().cpu().numpy().reshape(-1, frames.shape[-1])
        keep = _semantic(rows[:, 0])
        if keep.any():
            codes = rows[keep, 1:]
            logits = _alignment(multimodal_output)
            if logits is not None and logits.shape[0] == rows.shape[0]:
                align = logits.detach().cpu().numpy()[keep]
    return _advance(transfer_manager, _request_id(request), codes, finished, align=align)


def fish_omni_to_dac_decoder_async_chunk_batch(
    transfer_manager: Any,
    pooling_outputs: list[Any],
    requests: list[Any],
    is_finished: list[bool],
) -> list[OmniPayloadStruct | None]:
    """One model step of :func:`fish_omni_to_dac_decoder_async_chunk`, in request order.

    The step's frames are validated in one NumPy pass instead of one small
    conversion per request; most requests only append a frame.
    """
    present: list[int] = []
    arrays: list[np.ndarray] = []
    for index, output in enumerate(pooling_outputs):
        frames = _frames(output)
        if frames is not None:
            present.append(index)
            arrays.append(frames.detach().cpu().numpy().reshape(-1, frames.shape[-1]))
    rows: list[np.ndarray | None] = [None] * len(pooling_outputs)
    aligns: list[np.ndarray | None] = [None] * len(pooling_outputs)
    if arrays:
        stacked = np.concatenate(arrays)
        keep = _semantic(stacked[:, 0]).tolist()
        offset = 0
        for index, array in zip(present, arrays, strict=True):
            count = array.shape[0]
            if count == 1:
                rows[index] = stacked[offset : offset + 1, 1:] if keep[offset] else None
                mask = keep[offset : offset + 1]
            else:
                mask = keep[offset : offset + count]
                rows[index] = stacked[offset : offset + count][mask, 1:] if any(mask) else None
            if rows[index] is not None:
                logits = _alignment(pooling_outputs[index])
                if logits is not None and logits.shape[0] == count:
                    aligns[index] = logits.detach().cpu().numpy()[mask]
            offset += count
    # A stream starting in this step sizes its ramp by the step's requests.
    running = len(requests)
    return [
        _advance(
            transfer_manager, _request_id(request), rows[index], bool(is_finished[index]), running, align=aligns[index]
        )
        for index, request in enumerate(requests)
    ]


def _advance(
    transfer_manager: Any,
    request_id: str,
    codes: np.ndarray | None,
    finished: bool,
    running: int | None = None,
    *,
    align: np.ndarray | None = None,
) -> OmniPayloadStruct | None:
    ramps, chunk, left_context = _config(transfer_manager)
    state = _state(transfer_manager, request_id)
    if state["done"]:
        return None
    ramp = _ramp(transfer_manager, state, ramps, running)
    if codes is not None:
        state["codes"].extend(codes)
        if align is not None and state["align"] is None:
            state["align"] = []
        if state["align"] is not None:
            state["align"].extend(_align_rows(align, len(codes), state))

    base = state["base"]
    total = base + len(state["codes"])
    emitted = state["emitted"]
    window = ramp[state["windows"]] if state["windows"] < len(ramp) else chunk
    pending = total - emitted
    if not finished and pending < window:
        return None
    if pending <= 0:
        state["done"] = True
        # Still [codebooks, frames]: the receiver replaces the previous
        # window's codes only with a 2-D tensor, and a kept window would
        # decode again. An aligned stream's alignment is replaced too (the
        # receiver keeps the previous window's otherwise).
        return OmniPayloadStruct(
            codes=CodesStruct(
                audio=torch.empty((NUM_CODEBOOKS, 0), dtype=torch.long),
                alignment=_alignment_window(state, 0, 0),
            ),
            meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
        )
    end = total if finished else emitted + window
    context = min(emitted, left_context)
    # The window's index in its stream: the codec starts a stream's history at
    # chunk 0 (left context alone cannot tell, e.g. with no left context).
    seq = state["windows"]
    frames = np.stack(state["codes"][emitted - context - base : end - base])  # [frames, 10]
    alignment = _alignment_window(state, emitted - base, end - base)
    state["emitted"] = end
    state["windows"] += 1
    done = finished and end == total
    if done:
        state["done"] = True
        state["codes"] = []
        if state["align"] is not None:
            state["align"] = []
    else:
        keep_from = max(end - left_context, base)
        del state["codes"][: keep_from - base]
        if state["align"] is not None:
            del state["align"][: keep_from - base]
        state["base"] = keep_from
    # A [codebooks, frames] tensor: Stage 1 schedules one placeholder token per
    # window and the decoder reads the codes from runtime info, instead of a
    # Python int per code through the scheduler and back.
    # A fresh C-order copy: for a one-frame window ``ascontiguousarray`` keeps the
    # transposed strides, which the shared-memory connector cannot view as bytes.
    codes = torch.from_numpy(np.array(frames.T, dtype=np.int64, order="C", copy=True))
    return OmniPayloadStruct(
        codes=CodesStruct(audio=codes, alignment=alignment),
        meta=MetaStruct(
            left_context_size=context,
            chunk_seq=seq,
            finished=torch.tensor(done, dtype=torch.bool),
        ),
    )


def _align_rows(align: np.ndarray | None, count: int, state: dict[str, Any]) -> list[np.ndarray]:
    """One ``[heads, text]`` fp16 row per kept frame; zero rows when a step brought none."""
    if align is not None and align.shape[0] > 0:
        state["align_shape"] = tuple(align.shape[1:])
        rows = list(np.asarray(align, dtype=np.float16))
        if len(rows) >= count:
            return rows[:count]
    else:
        rows = []
    shape = state.get("align_shape", (1, 1))
    return rows + [np.zeros(shape, dtype=np.float16)] * (count - len(rows))


def _alignment_window(state: dict[str, Any], start: int, end: int) -> torch.Tensor | None:
    """``[end - start, heads, text]`` fp16 alignment rows of the window's new frames (None: no timestamps)."""
    if state["align"] is None:
        return None
    rows = state["align"][start:end]
    if rows:
        return torch.from_numpy(np.stack(rows).astype(np.float16, copy=False))
    return torch.zeros((0, *state.get("align_shape", (1, 1))), dtype=torch.float16)


# Per-step saves (OmniChunkTransferAdapter.save_async_batch) skip the request
# snapshot: this batch form reads only request ids and the finish flags given.
fish_omni_to_dac_decoder_async_chunk_batch.per_step_batch = True  # type: ignore[attr-defined]

__all__ = ["fish_omni_to_dac_decoder_async_chunk", "fish_omni_to_dac_decoder_async_chunk_batch"]
