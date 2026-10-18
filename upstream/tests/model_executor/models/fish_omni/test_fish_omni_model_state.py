# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the Drama 3 MRv2 request state: resuming a preempted request
and per-request seeds. The model is a stand-in that builds fixed prompt rows."""

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.fish_omni import model_state as ms
from vllm_omni.model_executor.models.fish_omni.fish_omni_model import FishOmniBuffers
from vllm_omni.model_executor.models.fish_omni.prompt import PromptLayout
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState
from vllm_omni.worker_v2.output_snapshot import RequestOutputSnapshot

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CODEBOOKS = 3
PROMPT = torch.tensor([[7, 0, 0, 0], [8, 0, 0, 0], [100, 0, 1, 2], [101, 1, 2, 3]])
# Rows 1 and 2 stand for the target text (row 0 for the speaker tag).
TEXT_ROWS = range(1, 3)


class _Model:
    frame_width = 1 + CODEBOOKS
    max_top_k = 64
    alignment = None

    def __init__(self) -> None:
        self.buffers = FishOmniBuffers(
            max_num_tokens=16,
            max_num_seqs=4,
            max_model_len=32,
            num_codebooks=CODEBOOKS,
            hidden_size=4,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        self.buffers.noise = torch.zeros((4, 5, 8))

    def build_layout(self, info, device):
        assert info["text"] == "hello"
        return PromptLayout(PROMPT.clone(), TEXT_ROWS)


@pytest.fixture
def make_state(monkeypatch):
    monkeypatch.setattr(OmniModelState, "add_request", lambda self, index, data: None)
    monkeypatch.setattr(OmniModelState, "on_request_preempted", lambda self, req_id, index: None)
    monkeypatch.setattr(OmniModelState, "on_requests_finished", lambda self, req_ids: None)

    def make() -> ms.FishOmniModelState:
        state = ms.FishOmniModelState.__new__(ms.FishOmniModelState)
        state.model = _Model()
        state.buffers = state.model.buffers
        state.device = torch.device("cpu")
        state.max_num_seqs = 4
        state.frame_width = state.model.frame_width
        state.prompts = {}
        state.prompt_len = np.zeros(4, dtype=np.int64)
        state.base_prompt_len = np.zeros(4, dtype=np.int64)
        state._pending_slots = {}
        state._pending_history = {}
        state._warned_top_k = False
        state._seeds = {}
        state._seed_generator = None
        state._suspended = {}
        state.alignment = None
        state.align_text = np.zeros((4, 2), dtype=np.int64)
        state._align_rows = []
        state._warned_text_len = False
        state._align_pool = None
        state._partitioned = True
        state.intermediate_buffer = SimpleNamespace(buffers={i: {"text": "hello"} for i in range(4)})
        return state

    return make


class _Capture:
    """Stands for fish_omni.alignment.AlignmentCapture: its static index buffers and capacity."""

    def __init__(self, max_text=8, max_tokens=16, max_num_seqs=4) -> None:
        self.max_text = max_text
        self.scratch_slot = max_num_seqs
        self.row_slot = torch.full((max_tokens,), self.scratch_slot, dtype=torch.int64)
        self.row_pos = torch.zeros(max_tokens, dtype=torch.int64)
        self.logits = torch.zeros((max_num_seqs, 2, max_text))

    def extract(self, rows, n_text):
        return self.logits.index_select(0, rows)[:, :, :n_text].to(torch.float16)


class _HostPool:
    """A UVA pool stand-in: the host array itself, as a tensor."""

    def copy_to_uva(self, array):
        return torch.from_numpy(np.ascontiguousarray(array))


def _request(req_id: str, generated: Sequence[int] = (), seed: int | None = None):
    prompt = [1] * PROMPT.shape[0]
    return SimpleNamespace(
        req_id=req_id,
        prompt_token_ids=prompt,
        prefill_token_ids=prompt + list(generated),
        sampling_params=SimpleNamespace(temperature=1.0, top_k=30, top_p=0.9, seed=seed),
    )


def test_new_request_stages_its_prompt_rows(make_state):
    state = make_state()
    state.add_request(1, _request("a"))
    assert np.array_equal(state.prompts[1], PROMPT.numpy())
    assert state.prompt_len[1] == PROMPT.shape[0]
    _, (top_k, generated), ras = state._pending_slots[1]
    assert (top_k, generated) == (30, 0)
    assert ras[-PROMPT.shape[0] :] == PROMPT[:, 0].tolist()


def test_preempted_request_resumes_with_its_generated_frames(make_state):
    state = make_state()
    state.add_request(1, _request("a"))
    codes = torch.tensor([[5, 6, 7], [8, 9, 10], [11, 12, 13]], dtype=torch.int16)
    state.buffers.slot_history[1, :3] = codes
    state.buffers.slot_generated[1] = 3
    state.on_request_preempted("a", 1)
    # Another request takes the slot before "a" comes back.
    state.buffers.slot_history[1].zero_()
    state.buffers.slot_generated[1] = 0

    main = [105, 108, 111]
    state.add_request(2, _request("a", generated=main))
    rows = state.prompts[2]
    assert rows.shape == (PROMPT.shape[0] + 3, 1 + CODEBOOKS)
    assert np.array_equal(rows[: PROMPT.shape[0]], PROMPT.numpy())
    assert rows[PROMPT.shape[0] :, 0].tolist() == main
    assert np.array_equal(rows[PROMPT.shape[0] :, 1:], codes.numpy())
    assert state.prompt_len[2] == PROMPT.shape[0] + 3
    # Its next sampled frame is frame 3, which keys its seeded noise.
    assert state.base_prompt_len[2] == PROMPT.shape[0]
    _, (_, generated), ras = state._pending_slots[2]
    assert generated == 3
    assert ras[-3:] == main
    # The resumed slot gets its code history back, for a later preemption.
    assert list(state._pending_history) == [2] and torch.equal(state._pending_history[2], codes)


def test_slot_reused_before_a_step_keeps_only_the_newest_request(make_state):
    """A request added and finished before any step (e.g. aborted at once) leaves
    nothing for the next request in its slot: one slot write, that request's own
    values, no code history, and the finished request's seed is gone."""
    state = make_state()
    state.add_request(1, _request("a"))
    state.buffers.slot_history[1, :2] = 1
    state.buffers.slot_generated[1] = 2
    state.on_request_preempted("a", 1)
    state.add_request(0, _request("a", generated=[105, 106], seed=7))
    assert 0 in state._pending_history and "a" in state._seeds
    state.on_requests_finished({"a"})

    later = _request("b")
    later.sampling_params = SimpleNamespace(temperature=0.5, top_k=8, top_p=0.7, seed=None)
    state.add_request(0, later)
    (temperature, top_p), (top_k, generated), _ = state._pending_slots[0]
    assert (temperature, top_p, top_k, generated) == (0.5, 0.7, 8, 0)
    assert 0 not in state._pending_history
    assert "a" not in state._seeds and "a" not in state._suspended


def test_resume_without_kept_codes_fails_loudly(make_state):
    state = make_state()
    with pytest.raises(RuntimeError, match="without the codes"):
        state.add_request(0, _request("b", generated=[105]))


def test_resume_needs_every_generated_frame(make_state):
    state = make_state()
    state.add_request(0, _request("c"))
    state.buffers.slot_generated[0] = 1
    state.on_request_preempted("c", 0)
    with pytest.raises(RuntimeError, match="needs 2 generated frames"):
        state.add_request(0, _request("c", generated=[105, 106]))


def test_finished_request_drops_kept_codes_and_generator(make_state):
    state = make_state()
    state.add_request(0, _request("d", seed=3))
    state.on_request_preempted("d", 0)
    state.on_requests_finished({"d"})
    assert "d" not in state._suspended and "d" not in state._seeds


def test_seeded_noise_depends_only_on_seed_and_frame(make_state):
    """The noise of a seeded request's frame k is the same whatever its batch row,
    the other requests, the global RNG, or how many steps it took to get there."""
    first, second = make_state(), make_state()
    first.add_request(0, _request("s", seed=1234))
    second.add_request(3, _request("s", seed=1234))
    second.add_request(1, _request("u"))
    yes = np.array([True, True])

    torch.manual_seed(0)
    first._draw_noise(["s"], np.array([5]), yes)
    torch.manual_seed(1)
    second._draw_noise(["u", "s"], np.array([9, 5]), yes)
    assert torch.equal(first.buffers.noise[0], second.buffers.noise[1])
    assert not torch.equal(second.buffers.noise[0], second.buffers.noise[1])

    # Another frame of the same request, or the same frame of another seed, differs.
    frame_five = first.buffers.noise[0].clone()
    first._draw_noise(["s"], np.array([6]), yes)
    assert not torch.equal(first.buffers.noise[0], frame_five)
    other = make_state()
    other.add_request(0, _request("s", seed=4321))
    torch.manual_seed(0)
    other._draw_noise(["s"], np.array([5]), yes)
    assert not torch.equal(other.buffers.noise[0], frame_five)


def test_seed_is_ignored_on_rows_that_sample_nothing(make_state):
    seeded, plain = make_state(), make_state()
    seeded.add_request(0, _request("s", seed=7))
    plain.add_request(0, _request("p"))
    for state, req_id in ((seeded, "s"), (plain, "p")):
        torch.manual_seed(3)
        state._draw_noise([req_id], np.array([0]), np.array([False]))
    assert torch.equal(seeded.buffers.noise, plain.buffers.noise)


@pytest.mark.parametrize(
    "value, ok",
    [(None, True), (32, True), (64, True), (4096, True), (100, False), (16, False), (1, False), (8192, False)],
)
def test_max_top_k_is_a_power_of_two_from_the_ras_top_k_to_the_codebook(value, ok):
    """The RAS re-draw keeps top-k 30, so fewer than 32 candidates would cut it."""
    from vllm_omni.model_executor.models.fish_omni.fish_omni_model import max_top_k

    config = SimpleNamespace() if value is None else SimpleNamespace(fish_omni_max_top_k=value)
    if ok:
        assert max_top_k(config, 4096) == (64 if value is None else value)
    else:
        with pytest.raises(ValueError, match="power of two"):
            max_top_k(config, 4096)


def test_word_timestamp_requests_record_their_text_rows(make_state):
    state = make_state()
    state.alignment = _Capture()
    state.intermediate_buffer.buffers[1]["word_timestamps"] = True
    state.add_request(1, _request("a"))
    state.add_request(2, _request("b"))
    assert state.align_text[1].tolist() == [TEXT_ROWS.start, len(TEXT_ROWS)]
    assert state.align_text[2].tolist() == [0, 0]
    # A text beyond the capture's capacity gets no timestamps (and the slot is reset).
    state.alignment = _Capture(max_text=1)
    state.add_request(1, _request("c"))
    assert state.align_text[1].tolist() == [0, 0] and state._warned_text_len


def test_mixed_step_stages_the_text_rows_stash_indices(make_state):
    state = make_state()
    state.alignment = _Capture()
    state._token_pool = state._sample_pool = _HostPool()
    state.intermediate_buffer.buffers[1]["word_timestamps"] = True
    state.add_request(1, _request("a"))
    state.add_request(2, _request("b"))
    state._flush_slot_writes = lambda: None
    # Request in slot 1 prefills its whole 4-row prompt, slot 2 decodes one row; 8 padded rows.
    batch = SimpleNamespace(
        num_reqs=2,
        num_tokens=5,
        num_tokens_after_padding=8,
        idx_mapping_np=np.array([1, 2]),
        num_scheduled_tokens=np.array([4, 1]),
        num_computed_tokens_np=np.array([0, 4]),
        query_start_loc_np=np.array([0, 4]),
        req_ids=["a", "b"],
    )
    state.prompt_len[2] = 4
    state.run_preprocess(batch, {})
    scratch = state.alignment.scratch_slot
    assert state.alignment.row_slot[:8].tolist() == [scratch, 1, 1, scratch, scratch, scratch, scratch, scratch]
    assert state.alignment.row_pos[1:3].tolist() == [0, 1]
    # Both sample rows are valid; only slot 1's request wants timestamps (2 text tokens).
    assert state._align_rows == [(0, 2)]


def test_postprocess_ships_alignment_rows_with_the_frames(make_state):
    state = make_state()
    state.alignment = _Capture()
    state._align_pool = _HostPool()
    state._valid_rows = [True, True, True]
    state._align_rows = [(0, 2), (2, 3)]
    state.alignment.logits[0, :, :2] = 1.0
    state.alignment.logits[2, :, :3] = 2.0
    _, snapshot = state.postprocess_model_output(torch.zeros(3, 4), SimpleNamespace(num_reqs=3), None)
    outputs = snapshot.copy_to_cpu(lambda tensor: tensor.clone())
    assert isinstance(outputs, RequestOutputSnapshot) and len(outputs.inter_stage) == 3
    first, second, third = outputs.inter_stage
    assert set(first) == {"codes.audio", "align.logits"} and set(second) == {"codes.audio"}
    assert first["align.logits"].shape == (1, 2, 2) and torch.all(first["align.logits"] == 1)
    assert third["align.logits"].shape == (1, 2, 3) and third["align.logits"].dtype == torch.float16
    # A request whose row sampled nothing still sends no alignment.
    state._valid_rows = [False, True, True]
    state._align_rows = [(1, 2)]
    _, snapshot = state.postprocess_model_output(torch.zeros(3, 4), SimpleNamespace(num_reqs=3), None)
    outputs = snapshot.copy_to_cpu(lambda tensor: tensor.clone())
    assert outputs.inter_stage[0] == {"codes.audio": outputs.inter_stage[0]["codes.audio"]}
    assert outputs.inter_stage[0]["codes.audio"].shape == (0, 4) and "align.logits" in outputs.inter_stage[1]


def test_snapshot_without_partitioning_lists_alignment_per_request():
    frames = torch.zeros((2, 4), dtype=torch.int64)
    align = torch.ones((1, 2, 3), dtype=torch.float16)
    snapshot = ms._FrameRowsSnapshot(frames, [True, True], partitioned=False, align=align, align_rows=[(1, 3)])
    payload = snapshot.copy_to_cpu(lambda tensor: tensor.clone())
    assert list(payload) == ["codes", "align"]
    assert payload["align"]["logits"][0].shape == (0, 2, 0) and payload["align"]["logits"][1].shape == (1, 2, 3)
