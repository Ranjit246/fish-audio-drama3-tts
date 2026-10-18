# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Codec window ramps: parsing, selection by load and the DAC graph shapes they need."""

import bisect

import pytest

from vllm_omni.model_executor.models.fish_speech.chunk_ramp import (
    RampConfig,
    all_window_sizes,
    parse_ramp,
    parse_ramp_by_load,
    ramp_config_from_extra,
    select_ramp,
)
from vllm_omni.model_executor.models.fish_speech.dac_channels_last import (
    streaming_decode_shapes,
    streaming_decode_shapes_for_ramps,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

TABLE = "8:2,4,8,16,32;16:3,6,12,24,32;32:4,6,12,24,32;*:6,8,12,16,24,32"


def test_parse_ramp_accepts_strings_and_sequences():
    assert parse_ramp("2, 4,8") == (2, 4, 8)
    assert parse_ramp([2, 4, 8]) == (2, 4, 8)
    assert parse_ramp((6.0, 8.0)) == (6, 8)


@pytest.mark.parametrize("value", ["", ",", [], "2,0,8", [2, -1], "2,x", 4])
def test_parse_ramp_rejects_empty_zero_and_malformed(value):
    with pytest.raises(ValueError, match="codec_chunk_ramp"):
        parse_ramp(value)


def test_parse_ramp_by_load_sorts_thresholds_and_puts_the_default_last():
    assert parse_ramp_by_load("*:6,8; 16 : 3,6 ;8:2,4;") == [(8, (2, 4)), (16, (3, 6)), (None, (6, 8))]


@pytest.mark.parametrize(
    "value",
    [
        "8:2,4;16:3,6",  # no default
        "*:6,8;*:4,8",  # two defaults
        "8:2,0;*:6,8",  # a zero window
        "8:;*:6,8",  # an empty ramp
        "0:2,4;*:6,8",  # a non-positive threshold
        "x:2,4;*:6,8",  # a non-integer threshold
        "8:2,4;8:3,6;*:6,8",  # a threshold twice
        "2,4,8",  # a static ramp
        ["8:2,4", "*:6,8"],
    ],
)
def test_parse_ramp_by_load_rejects_bad_tables(value):
    with pytest.raises(ValueError, match="codec_chunk_ramp_by_load"):
        parse_ramp_by_load(value)


def test_select_ramp_takes_the_smallest_threshold_at_or_above_the_load():
    table = parse_ramp_by_load(TABLE)
    assert select_ramp(table, 0) == (2, 4, 8, 16, 32)
    assert select_ramp(table, 8) == (2, 4, 8, 16, 32)
    assert select_ramp(table, 9) == (3, 6, 12, 24, 32)
    assert select_ramp(table, 20) == (4, 6, 12, 24, 32)
    assert select_ramp(table, 32) == (4, 6, 12, 24, 32)
    assert select_ramp(table, 100) == (6, 8, 12, 16, 24, 32)


def test_all_window_sizes_is_the_union_with_the_chunk():
    assert all_window_sizes(parse_ramp_by_load(TABLE), 64) == {2, 3, 4, 6, 8, 12, 16, 24, 32, 64}


def test_ramp_config_prefers_the_by_load_table_and_warns_once(monkeypatch):
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.fish_speech import chunk_ramp

    warnings = []
    monkeypatch.setattr(chunk_ramp, "logger", SimpleNamespace(warning_once=lambda *args: warnings.append(args)))
    static = ramp_config_from_extra({"codec_chunk_ramp": "2,3"}, default=(6, 8))
    assert static == RampConfig(static=(2, 3)) and static.ramps == [(2, 3)] and static.select(100) == (2, 3)
    assert ramp_config_from_extra({}, default=(6, 8)).static == (6, 8)
    # ``codec_chunk_ramp: null`` (a deploy overlay unsetting the base's ramp) counts as absent.
    assert ramp_config_from_extra({"codec_chunk_ramp": None, "codec_chunk_ramp_by_load": TABLE}).by_load is not None
    assert warnings == []
    both = ramp_config_from_extra({"codec_chunk_ramp": "2,3", "codec_chunk_ramp_by_load": TABLE})
    assert both.select(0) == (2, 4, 8, 16, 32) and both.select(100) == (6, 8, 12, 16, 24, 32)
    assert both.window_sizes(64) == {2, 3, 4, 6, 8, 12, 16, 24, 32, 64}
    assert len(warnings) == 1 and "codec_chunk_ramp_by_load" in warnings[0]


def test_streaming_decode_shapes_matches_the_single_ramp_form():
    # The trims the CUDA test asserts against the real decoder (receptive 10).
    expected = {8: {0}, 10: {0}, 16: {0}, 20: {0}, 24: {10}, 32: {10}, 48: {22}, 64: {22}, 80: {22}, 96: {22}}
    buckets, trims = streaming_decode_shapes((10, 10, 12, 32), 64, 32, 10)
    assert trims == expected
    assert (buckets, trims) == streaming_decode_shapes_for_ramps([(10, 10, 12, 32)], 64, 32, 10)


def _window_pairs(ramp, chunk, left):
    """Every (frames, context) a stream on ``ramp`` can send, as the chunker sizes them."""
    pairs, emitted = set(), 0
    for window in (*ramp, chunk):
        context = min(emitted, left)
        pairs.update((new + context, context) for new in range(1, window + 1))
        emitted += window
    return pairs


@pytest.mark.parametrize("left,receptive", [(32, 10), (0, 0)])
def test_shapes_for_several_ramps_cover_every_ramp_s_windows(left, receptive):
    """A stream follows one of the table's ramps; the union buckets must give each of its
    windows the bucket ``frame_bucket`` picks at run time, with that window's trim captured."""
    ramps = [ramp for _, ramp in parse_ramp_by_load(TABLE)]
    chunk = 64
    buckets, trims = streaming_decode_shapes_for_ramps(ramps, chunk, left, receptive)
    assert buckets == sorted(set(buckets))
    for ramp in ramps:
        own_buckets, _ = streaming_decode_shapes(ramp, chunk, left, receptive)
        assert set(own_buckets) <= set(buckets)
        for frames, context in _window_pairs(ramp, chunk, left):
            bucket = buckets[bisect.bisect_left(buckets, frames)]
            assert context - min(context, receptive) in trims[bucket], (ramp, frames, context)
    if left == 0:
        # Stateful decoding: graphs take new frames only, so buckets are the window sizes.
        assert set(buckets) >= all_window_sizes(parse_ramp_by_load(TABLE), chunk)
        assert all(trims[bucket] == {0} for bucket in buckets)


def test_decoder_graph_planning_covers_the_union_of_ramps():
    from vllm_omni.model_executor.models.fish_speech.fish_speech_dac_decoder import streaming_window_ramps

    extra = {"codec_chunk_ramp_by_load": TABLE, "codec_chunk_frames": 64, "codec_left_context_frames": 32}
    ramps, chunk, left = streaming_window_ramps(extra)
    assert ramps == [(2, 4, 8, 16, 32), (3, 6, 12, 24, 32), (4, 6, 12, 24, 32), (6, 8, 12, 16, 24, 32)]
    assert (chunk, left) == (64, 32)
    # The stateful path (left context and receptive field 0) captures one graph per window size.
    assert max(chunk, *(size for ramp in ramps for size in ramp)) == 64
    buckets, _ = streaming_decode_shapes_for_ramps(ramps, chunk, 0, 0)
    assert set(buckets) >= {2, 3, 4, 6, 8, 12, 16, 24, 32, 64}
    # The static key alone, and the legacy first-window key, keep their plans.
    assert streaming_window_ramps({"codec_chunk_ramp": "6,8,12", "codec_chunk_frames": 16}) == ([(6, 8, 12)], 16, 25)
    assert streaming_window_ramps({"initial_codec_chunk_frames": 4}) == ([(4,)], 25, 25)
    assert streaming_window_ramps({}) == ([()], 25, 25)
