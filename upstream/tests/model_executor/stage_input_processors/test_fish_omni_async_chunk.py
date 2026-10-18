# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drama 3 frame stream -> DAC windows (ramp, left context, final flush)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.data_entry_keys import flatten_payload, to_struct, unflatten_payload
from vllm_omni.model_executor.stage_input_processors.fish_omni import (
    SEMANTIC_BEGIN_ID,
    fish_omni_to_dac_decoder_async_chunk,
    fish_omni_to_dac_decoder_async_chunk_batch,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

EOS = 248046
CODEBOOKS = 10


BY_LOAD = "16:2,4,8,16,32;24:4,6,12,24,32;*:6,8,12,16,24,32"
RAMP_16, RAMP_24, RAMP_DEFAULT = (2, 4, 8, 16, 32), (4, 6, 12, 24, 32), (6, 8, 12, 16, 24, 32)


def _manager(ramp="2,3", chunk=4, left=2, by_load=None, in_flight=0):
    extra = {"codec_chunk_ramp": ramp, "codec_chunk_frames": chunk, "codec_left_context_frames": left}
    if by_load is not None:
        extra["codec_chunk_ramp_by_load"] = by_load
    return SimpleNamespace(
        connector=SimpleNamespace(config={"extra": extra}),
        # Requests already streaming on this stage hold their sender state here.
        request_payload={f"running-{index}": {} for index in range(in_flight)},
    )


def _request(req_id="r"):
    return SimpleNamespace(external_req_id=req_id, is_finished=lambda: False)


HEADS, TEXT = 2, 3


def _frame(index: int, main: int | None = None, align: bool = False) -> dict:
    codes = torch.arange(CODEBOOKS) + 100 * index
    row = torch.cat((torch.tensor([SEMANTIC_BEGIN_ID + index if main is None else main]), codes))
    output = {"codes.audio": row.unsqueeze(0)}
    if align:
        # The step's alignment logits of this frame: [1, heads, text], marked with the frame index.
        output["align.logits"] = torch.full((1, HEADS, TEXT), float(index), dtype=torch.float16)
    return output


def _alignment(payload) -> list[int] | None:
    """Frame index of each alignment row of an emitted window (None: no word timestamps)."""
    rows = payload.codes.alignment
    if rows is None:
        return None
    assert rows.dtype == torch.float16 and rows.ndim == 3 and tuple(rows.shape[1:]) == (HEADS, TEXT)
    return rows[:, 0, 0].int().tolist()


def _window(payload) -> tuple[list[int], int]:
    """(first code 0 of each frame, left context) of an emitted window."""
    codes = payload.codes.audio
    assert codes.dtype == torch.int64 and codes.ndim == 2 and codes.shape[0] == CODEBOOKS
    return (codes[0] // 100).tolist(), int(payload.meta.left_context_size)


def test_windows_follow_the_ramp_with_left_context():
    manager, request = _manager(), _request()
    windows = []
    for index in range(12):
        payload = fish_omni_to_dac_decoder_async_chunk(manager, _frame(index), request)
        if payload is not None:
            windows.append(_window(payload))
    # Ramp 2, 3, then chunks of 4, each with up to 2 earlier frames.
    assert windows == [
        ([0, 1], 0),
        ([0, 1, 2, 3, 4], 2),
        ([3, 4, 5, 6, 7, 8], 2),
    ]
    final = fish_omni_to_dac_decoder_async_chunk(manager, _frame(12, main=EOS), request, is_finished=True)
    assert _window(final) == ([7, 8, 9, 10, 11], 2)
    assert bool(final.meta.finished)
    assert fish_omni_to_dac_decoder_async_chunk(manager, None, request, is_finished=True) is None


def test_end_of_speech_frame_carries_no_audio():
    manager, request = _manager(ramp="1"), _request()
    assert fish_omni_to_dac_decoder_async_chunk(manager, _frame(0, main=EOS), request) is None
    final = fish_omni_to_dac_decoder_async_chunk(manager, None, request, is_finished=True)
    assert final.codes.audio.shape == (CODEBOOKS, 0) and bool(final.meta.finished)


def test_a_one_frame_window_has_byte_viewable_codes():
    # A request that yields a single frame sends a [codebooks, 1] window; the
    # shared-memory connector views the codes as bytes, which needs stride 1.
    manager, request = _manager(ramp="2"), _request()
    assert fish_omni_to_dac_decoder_async_chunk(manager, _frame(0), request) is None
    final = fish_omni_to_dac_decoder_async_chunk(manager, None, request, is_finished=True)
    codes = final.codes.audio
    assert codes.shape == (CODEBOOKS, 1) and codes.stride(-1) == 1
    assert codes.view(torch.uint8).numel() == CODEBOOKS * 8


def test_stream_keeps_only_frames_later_windows_read():
    manager, request = _manager(ramp="2", chunk=2, left=1), _request()
    for index in range(40):
        fish_omni_to_dac_decoder_async_chunk(manager, _frame(index), request)
    state = manager.request_payload["r"]["_fish_omni_stream"]
    assert len(state["codes"]) <= 1 + 2


def test_batch_matches_per_request_calls():
    single, batched = _manager(), _manager()
    requests = [_request("a"), _request("b"), _request("c")]
    for index in range(10):
        outputs = [_frame(index), _frame(index, main=EOS) if index == 7 else _frame(index), {}]
        finished = [False, index == 7, index == 9]
        expected = [
            fish_omni_to_dac_decoder_async_chunk(single, output, request, done)
            for output, request, done in zip(outputs, requests, finished)
        ]
        got = fish_omni_to_dac_decoder_async_chunk_batch(batched, outputs, requests, finished)
        for want, have in zip(expected, got):
            assert (want is None) == (have is None)
            if want is not None:
                assert np.array_equal(want.codes.audio.numpy(), have.codes.audio.numpy())
                assert bool(want.meta.finished) == bool(have.meta.finished)


@pytest.mark.parametrize("left", [0, 2])
def test_windows_carry_their_index_in_the_stream(left):
    """chunk_seq numbers a stream's windows from 0: the codec starts its history at 0 even
    when windows carry no left context."""
    manager, request = _manager(left=left), _request()
    seqs = []
    for index in range(12):
        payload = fish_omni_to_dac_decoder_async_chunk(manager, _frame(index), request)
        if payload is not None:
            seqs.append((payload.meta.chunk_seq, int(payload.meta.left_context_size)))
    final = fish_omni_to_dac_decoder_async_chunk(manager, None, request, is_finished=True)
    seqs.append((final.meta.chunk_seq, int(final.meta.left_context_size)))
    assert [seq for seq, _ in seqs] == list(range(len(seqs)))
    assert all(context == (0 if left == 0 or seq == 0 else left) for seq, context in seqs)


def _first_windows(manager, request, steps=8):
    """Frames of the first windows a stream sends through the single-request path, one frame per step."""
    sizes = []
    for index in range(steps):
        payload = fish_omni_to_dac_decoder_async_chunk(manager, _frame(index), request)
        if payload is not None:
            sizes.append(payload.codes.audio.shape[1] - int(payload.meta.left_context_size))
    return sizes


def _step(manager, requests, index, finished=()):
    """One batched step: every request gets frame ``index``; ``finished`` requests end with it."""
    return fish_omni_to_dac_decoder_async_chunk_batch(
        manager, [_frame(index) for _ in requests], requests, [r in finished for r in requests]
    )


def _ramp_of(manager, request):
    return manager.request_payload[request.external_req_id]["_fish_omni_stream"]["ramp"]


def _by_load_manager():
    return _manager(ramp=None, chunk=64, left=0, by_load=BY_LOAD)


@pytest.mark.parametrize(
    "running,ramp", [(1, RAMP_16), (16, RAMP_16), (17, RAMP_24), (24, RAMP_24), (25, RAMP_DEFAULT)]
)
def test_streams_starting_in_one_step_take_the_ramp_of_that_step_s_size(running, ramp):
    manager = _by_load_manager()
    requests = [_request(f"r{index}") for index in range(running)]
    _step(manager, requests, 0)
    assert all(_ramp_of(manager, request) == ramp for request in requests)
    # Chosen once: the streams keep their ramp while the load around them changes.
    manager.request_payload.update({f"late-{index}": {} for index in range(100)})
    windows = [[] for _ in requests]
    for index in range(1, sum(ramp) + 64):
        for slot, payload in enumerate(_step(manager, requests, index)):
            if payload is not None:
                windows[slot].append(payload.codes.audio.shape[1])
    assert all(sizes == [*ramp, 64] for sizes in windows)


def test_a_burst_starting_in_one_step_sees_itself():
    """64 requests whose first step is the same step all take the default ramp, although none
    of them had sender state before it (a count of streams already running would say 0)."""
    manager = _by_load_manager()
    requests = [_request(f"burst-{index}") for index in range(64)]
    assert manager.request_payload == {}
    payloads = [_step(manager, requests, index) for index in range(6)]
    assert all(_ramp_of(manager, request) == RAMP_DEFAULT for request in requests)
    assert all(payload is None for step in payloads[:5] for payload in step)
    assert [payload.codes.audio.shape[1] for payload in payloads[5]] == [6] * 64


def test_a_stream_joining_running_streams_counts_the_step_it_joins():
    manager = _by_load_manager()
    running = [_request(f"running-{index}") for index in range(15)]
    for index in range(3):
        _step(manager, running, index)
    joiner = _request("joiner")  # a step of 15 + 1 requests
    _step(manager, [*running, joiner], 3)
    assert _ramp_of(manager, joiner) == RAMP_16
    assert all(_ramp_of(manager, request) == RAMP_16 for request in running)
    # Nine more arrive together: 25 in the step, the default ramp; the earlier streams keep theirs.
    wave = [_request(f"wave-{index}") for index in range(9)]
    _step(manager, [*running, joiner, *wave], 4)
    assert all(_ramp_of(manager, request) == RAMP_DEFAULT for request in wave)
    assert _ramp_of(manager, joiner) == RAMP_16


def test_finished_streams_leave_the_step_count():
    manager = _by_load_manager()
    first = [_request(f"first-{index}") for index in range(24)]
    _step(manager, first, 0)
    assert all(_ramp_of(manager, request) == RAMP_24 for request in first)
    _step(manager, first, 1, finished=first[10:])
    survivors = first[:10]
    later = _request("later")  # a step of 10 + 1 requests
    _step(manager, [*survivors, later], 2)
    assert _ramp_of(manager, later) == RAMP_16
    assert all(_ramp_of(manager, request) == RAMP_24 for request in survivors)


@pytest.mark.parametrize("in_flight,ramp", [(0, RAMP_16), (15, RAMP_16), (16, RAMP_24), (24, RAMP_DEFAULT)])
def test_the_single_request_path_falls_back_to_the_streams_with_sender_state(in_flight, ramp):
    """Without a step size the requests holding sender state stand in, the new one included.
    (That count lags one step: a burst through this path would see only itself.)"""
    manager = _manager(ramp=None, chunk=64, left=0, by_load=BY_LOAD, in_flight=in_flight)
    request = _request("new")
    assert _first_windows(manager, request, steps=ramp[0]) == [ramp[0]]
    assert _ramp_of(manager, request) == ramp


def test_the_by_load_table_wins_over_the_static_ramp(monkeypatch):
    from vllm_omni.model_executor.models.fish_speech import chunk_ramp

    warnings = []
    monkeypatch.setattr(chunk_ramp, "logger", SimpleNamespace(warning_once=lambda *args: warnings.append(args)))
    manager = _manager(ramp="1", chunk=64, left=0, by_load=BY_LOAD)
    assert _first_windows(manager, _request("a"), steps=3) == [2]
    assert _first_windows(manager, _request("b"), steps=3) == [2]
    assert len(warnings) == 1  # the config is read once per adapter


def test_the_static_ramp_is_unchanged_without_the_by_load_key():
    manager = _manager(in_flight=50)
    assert _first_windows(manager, _request(), steps=6) == [2, 3]
    assert _ramp_of(manager, _request()) == (2, 3)
    batched = _manager()
    _step(batched, [_request(f"r{index}") for index in range(64)], 0)
    assert all(state["_fish_omni_stream"]["ramp"] == (2, 3) for state in batched.request_payload.values())


@pytest.mark.parametrize("extra", [{"codec_chunk_ramp": "2,0"}, {"codec_chunk_ramp_by_load": "8:2,4"}])
def test_invalid_ramps_are_rejected_at_the_first_step(extra):
    manager = _manager()
    manager.connector.config["extra"].update(extra)
    with pytest.raises(ValueError, match="codec_chunk_ramp"):
        fish_omni_to_dac_decoder_async_chunk(manager, _frame(0), _request())


def test_alignment_rows_ride_with_the_windows_new_frames():
    manager, request = _manager(), _request()
    windows = []
    for index in range(12):
        payload = fish_omni_to_dac_decoder_async_chunk(manager, _frame(index, align=True), request)
        if payload is not None:
            windows.append((_window(payload), _alignment(payload)))
    # The left context is not repeated: a window's rows are exactly its new frames'.
    assert windows == [
        (([0, 1], 0), [0, 1]),
        (([0, 1, 2, 3, 4], 2), [2, 3, 4]),
        (([3, 4, 5, 6, 7, 8], 2), [5, 6, 7, 8]),
    ]
    # The end-of-speech frame carries no audio and no alignment row.
    final = fish_omni_to_dac_decoder_async_chunk(manager, _frame(12, main=EOS, align=True), request, is_finished=True)
    assert _window(final) == ([7, 8, 9, 10, 11], 2) and _alignment(final) == [9, 10, 11]


def test_a_stream_without_word_timestamps_carries_no_alignment():
    manager, request = _manager(ramp="1"), _request()
    payload = fish_omni_to_dac_decoder_async_chunk(manager, _frame(0), request)
    assert payload.codes.alignment is None
    final = fish_omni_to_dac_decoder_async_chunk(manager, None, request, is_finished=True)
    assert final.codes.alignment is None


def test_the_terminal_empty_window_replaces_the_alignment_too():
    # The receiver keeps a window's codes.* sub-keys until the next window
    # replaces them, so the empty terminal window carries an empty [0, heads, text].
    manager, request = _manager(ramp="1"), _request()
    fish_omni_to_dac_decoder_async_chunk(manager, _frame(0, align=True), request)
    final = fish_omni_to_dac_decoder_async_chunk(manager, None, request, is_finished=True)
    assert final.codes.audio.shape == (CODEBOOKS, 0)
    assert tuple(final.codes.alignment.shape) == (0, HEADS, TEXT) and final.codes.alignment.dtype == torch.float16


def test_alignment_rows_are_trimmed_with_the_frames_they_belong_to():
    manager, request = _manager(ramp="2", chunk=2, left=1), _request()
    for index in range(40):
        fish_omni_to_dac_decoder_async_chunk(manager, _frame(index, align=True), request)
    state = manager.request_payload["r"]["_fish_omni_stream"]
    assert len(state["align"]) == len(state["codes"]) <= 1 + 2


def test_batch_form_carries_the_same_alignment_rows():
    single, batched = _manager(), _manager()
    requests = [_request("a"), _request("b")]
    for index in range(10):
        outputs = [_frame(index, align=True), _frame(index, main=EOS) if index == 7 else _frame(index)]
        finished = [False, index == 7]
        expected = [
            fish_omni_to_dac_decoder_async_chunk(single, output, request, done)
            for output, request, done in zip(outputs, requests, finished)
        ]
        got = fish_omni_to_dac_decoder_async_chunk_batch(batched, outputs, requests, finished)
        for want, have in zip(expected, got):
            assert (want is None) == (have is None)
            if want is not None:
                assert _alignment(want) == _alignment(have)


def test_window_payload_round_trips_through_the_payload_schema():
    manager, request = _manager(ramp="2"), _request()
    fish_omni_to_dac_decoder_async_chunk(manager, _frame(0, align=True), request)
    payload = fish_omni_to_dac_decoder_async_chunk(manager, _frame(1, align=True), request)
    nested = {"codes": {"audio": payload.codes.audio, "alignment": payload.codes.alignment}, "meta": {"chunk_seq": 0}}
    flat = flatten_payload(nested)
    assert set(flat) == {"codes.audio", "codes.alignment", "meta.chunk_seq"}
    back = unflatten_payload(flat)
    struct = to_struct(back)
    assert torch.equal(struct.codes.alignment, payload.codes.alignment)
    assert tuple(struct.codes.alignment.shape) == (2, HEADS, TEXT)


def test_alignment_is_a_client_facing_output_key():
    from vllm_omni.utils.mm_outputs import partition_flat_payload

    inter_stage, client = partition_flat_payload({"model_outputs": 1, "alignment": 2, "codes.audio": 3})
    assert client == {"model_outputs": 1, "alignment": 2} and inter_stage == {"codes.audio": 3}
