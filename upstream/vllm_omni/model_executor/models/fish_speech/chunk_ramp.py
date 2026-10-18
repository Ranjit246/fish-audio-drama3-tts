# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Codec window ramps of the Fish Audio frame stream: parsing and selection by load.

``codec_chunk_ramp`` (``"6,8,12,16,24,32"`` or a list) sizes a stream's first
windows, after which every window has ``codec_chunk_frames`` frames. A small
first window gets audio out early, but window ``n + 1`` has to be generated
before window ``n``'s audio has played; how many frames a decode step yields
per stream falls with the number of streams. ``codec_chunk_ramp_by_load``
(``"16:2,4,8,16,32;24:4,6,12,24,32;*:6,8,12,16,24,32"``) therefore keeps one
ramp per load band -- ``<max_running>:<ramp>`` entries separated by ``;``,
``*`` the default -- and a request takes the ramp of the smallest threshold
at or above the number of requests in its first decode step, itself included.

Both the frame-stream processor (Stage 0, which chooses the ramp) and the DAC
decoder (Stage 1, which captures a CUDA graph per window size) read the
connector config through :func:`ramp_config_from_extra`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

STATIC_RAMP_KEY = "codec_chunk_ramp"
BY_LOAD_RAMP_KEY = "codec_chunk_ramp_by_load"

# ``(max_running, ramp)`` entries, thresholds ascending, the default (``None``) last.
RampTable = list[tuple[int | None, tuple[int, ...]]]


def parse_ramp(value: str | Iterable[int], *, name: str = STATIC_RAMP_KEY) -> tuple[int, ...]:
    """``"2,4,8"`` or ``[2, 4, 8]`` -> ``(2, 4, 8)``: one or more windows of at least one frame."""
    if isinstance(value, str):
        parts: list[Any] = [part.strip() for part in value.split(",") if part.strip()]
    else:
        try:
            parts = list(value)
        except TypeError as exc:
            raise ValueError(f"{name}: expected 'a,b,c' or a list of frame counts, got {value!r}") from exc
    try:
        ramp = tuple(int(part) for part in parts)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: windows must be integers, got {value!r}") from exc
    if not ramp:
        raise ValueError(f"{name}: at least one window is required, got {value!r}")
    if any(frames <= 0 for frames in ramp):
        raise ValueError(f"{name}: every window needs at least one frame, got {value!r}")
    return ramp


def parse_ramp_by_load(value: str) -> RampTable:
    """``"8:2,4,8;16:3,6,12;*:6,8,12"`` -> ``[(8, (2, 4, 8)), (16, (3, 6, 12)), (None, (6, 8, 12))]``.

    Entries are ``<max_running>:<ramp>`` separated by ``;``. Thresholds are
    positive and distinct; the ``*`` default is required and sorts last.
    """
    if not isinstance(value, str):
        raise ValueError(f"{BY_LOAD_RAMP_KEY}: expected '<max_running>:<ramp>;...', got {value!r}")
    entries: list[tuple[int, tuple[int, ...]]] = []
    default: tuple[int, ...] | None = None
    for entry in value.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        threshold, separator, ramp = entry.partition(":")
        threshold = threshold.strip()
        if not separator or not threshold:
            raise ValueError(f"{BY_LOAD_RAMP_KEY}: entry {entry!r} is not '<max_running>:<ramp>'")
        windows = parse_ramp(ramp, name=f"{BY_LOAD_RAMP_KEY} entry {entry!r}")
        if threshold == "*":
            if default is not None:
                raise ValueError(f"{BY_LOAD_RAMP_KEY}: more than one '*' default in {value!r}")
            default = windows
            continue
        try:
            limit = int(threshold)
        except ValueError as exc:
            raise ValueError(f"{BY_LOAD_RAMP_KEY}: threshold {threshold!r} must be an integer or '*'") from exc
        if limit <= 0:
            raise ValueError(f"{BY_LOAD_RAMP_KEY}: threshold {threshold!r} must be positive")
        if any(limit == seen for seen, _ in entries):
            raise ValueError(f"{BY_LOAD_RAMP_KEY}: threshold {limit} given twice in {value!r}")
        entries.append((limit, windows))
    if default is None:
        raise ValueError(f"{BY_LOAD_RAMP_KEY}: a '*' default entry is required, got {value!r}")
    entries.sort(key=lambda entry: entry[0])
    return [*entries, (None, default)]


def select_ramp(table: RampTable, running: int) -> tuple[int, ...]:
    """The ramp of the smallest threshold at or above ``running`` streams, else the default."""
    for limit, ramp in table:
        if limit is None or running <= limit:
            return ramp
    raise ValueError("ramp table has no default entry")


def all_window_sizes(table: RampTable, chunk: int) -> set[int]:
    """Every window size the table's ramps can send, ``chunk`` (the steady-state window) included."""
    sizes = {int(chunk)} if chunk > 0 else set()
    for _, ramp in table:
        sizes.update(ramp)
    return sizes


@dataclass
class RampConfig:
    """The window ramps a connector ``extra`` block configures.

    ``static`` is ``codec_chunk_ramp`` (or the caller's default) and
    ``by_load`` the parsed ``codec_chunk_ramp_by_load`` table, which takes
    precedence when present.
    """

    static: tuple[int, ...]
    by_load: RampTable | None = None

    @property
    def ramps(self) -> list[tuple[int, ...]]:
        """Every ramp a stream can follow (one per load band, or the static one)."""
        if self.by_load is not None:
            return [ramp for _, ramp in self.by_load]
        return [self.static]

    def select(self, running: int) -> tuple[int, ...]:
        """The ramp for a stream whose first decode step has ``running`` requests."""
        if self.by_load is not None:
            return select_ramp(self.by_load, running)
        return self.static

    def window_sizes(self, chunk: int) -> set[int]:
        """Every window size any ramp (then ``chunk``) can send."""
        if self.by_load is not None:
            return all_window_sizes(self.by_load, chunk)
        return all_window_sizes([(None, self.static)], chunk)


def ramp_config_from_extra(extra: Mapping[str, Any], *, default: Iterable[int] = ()) -> RampConfig:
    """Read ``codec_chunk_ramp`` / ``codec_chunk_ramp_by_load`` from a connector ``extra`` block.

    A key whose value is ``None`` (``codec_chunk_ramp: null`` in a deploy
    overlay) counts as absent. With both keys set the by-load table wins and a
    warning is logged once.
    """
    static_cfg = extra.get(STATIC_RAMP_KEY)
    by_load_cfg = extra.get(BY_LOAD_RAMP_KEY)
    static = parse_ramp(static_cfg) if static_cfg is not None else tuple(int(frames) for frames in default)
    by_load = parse_ramp_by_load(by_load_cfg) if by_load_cfg is not None else None
    if by_load is not None and static_cfg is not None:
        logger.warning_once(
            "Both %s and %s are set; the by-load table decides every stream's ramp and %s=%r is ignored",
            BY_LOAD_RAMP_KEY,
            STATIC_RAMP_KEY,
            STATIC_RAMP_KEY,
            static_cfg,
        )
    return RampConfig(static=static, by_load=by_load)


__all__ = [
    "BY_LOAD_RAMP_KEY",
    "STATIC_RAMP_KEY",
    "RampConfig",
    "RampTable",
    "all_window_sizes",
    "parse_ramp",
    "parse_ramp_by_load",
    "ramp_config_from_extra",
    "select_ramp",
]
