# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the Drama 3 word-alignment capture: head grouping, the static
stash / logits buffers and the attention pre-hook, on a small fake backbone."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.fish_omni import alignment as al

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

HEADS, KV_HEADS, HEAD_DIM = 4, 2, 8  # two query heads per KV head
MAX_TEXT = 5
CPU = torch.device("cpu")


class _Attention(nn.Module):
    """Stands for vLLM's ``Attention``: receives (q, k, v) and knows its layer name."""

    def __init__(self, layer: int) -> None:
        super().__init__()
        self.layer_name = f"model.layers.{layer}.self_attn.attn"

    def forward(self, q, k, v):
        return v


class _SelfAttention(nn.Module):
    num_heads, num_kv_heads, head_dim = HEADS, KV_HEADS, HEAD_DIM

    def __init__(self, layer: int) -> None:
        super().__init__()
        self.attn = _Attention(layer)

    def forward(self, q, k, v):
        return self.attn(q, k, v)


class _Layer(nn.Module):
    def __init__(self, layer: int, full: bool) -> None:
        super().__init__()
        if full:
            self.self_attn = _SelfAttention(layer)


def _capture(heads, *, max_num_seqs=3, max_tokens=16) -> al.AlignmentCapture:
    head_set = al.AlignmentHeadSet(heads, num_heads=HEADS, num_kv_heads=KV_HEADS)
    return al.AlignmentCapture(
        head_set,
        max_num_tokens=max_tokens,
        max_num_seqs=max_num_seqs,
        max_text=MAX_TEXT,
        num_heads=HEADS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        scaling=HEAD_DIM**-0.5,
        device=CPU,
    )


def _reference(q_row: torch.Tensor, keys: torch.Tensor, head: int) -> torch.Tensor:
    """``q[head] . K[kv(head)]^T * scale`` in fp32 over the stashed text rows."""
    q = q_row.reshape(HEADS, HEAD_DIM)[head].float()
    k = keys.reshape(-1, KV_HEADS, HEAD_DIM)[:, head // (HEADS // KV_HEADS)].float()
    return (k @ q) * HEAD_DIM**-0.5


def test_head_set_groups_heads_by_layer_in_buffer_order():
    head_set = al.AlignmentHeadSet([(3, 1), (7, 2), (3, 3)], num_heads=HEADS, num_kv_heads=KV_HEADS)
    # Stable re-order by layer: each layer's heads form one block of the logits buffer.
    assert head_set.heads == ((3, 1), (3, 3), (7, 2))
    assert head_set.layers == (3, 7)
    assert head_set.layer_heads == {3: [0, 1], 7: [2]}
    # Head 1 reads KV head 0, head 3 KV head 1, head 2 KV head 1 (only that one is stashed for layer 7).
    assert head_set.layer_kv == {3: [0, 1], 7: [1]}
    assert head_set.layer_kv_pos == {3: [0, 1], 7: [0]}
    with pytest.raises(ValueError, match="out of range"):
        al.AlignmentHeadSet([(3, HEADS)], num_heads=HEADS, num_kv_heads=KV_HEADS)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, al.DEFAULT_ALIGNMENT_HEADS),
        ("27:9,31:4", ((27, 9), (31, 4))),
        ([[27, 9], [31, 4]], ((27, 9), (31, 4))),
        ([{"layer": 27, "head": 9}], ((27, 9),)),
    ],
)
def test_parse_alignment_heads(value, expected):
    assert al.parse_alignment_heads(value) == expected


def test_parse_alignment_heads_rejects_duplicates_and_empty():
    with pytest.raises(ValueError, match="repeats"):
        al.parse_alignment_heads("27:9,27:9")
    with pytest.raises(ValueError, match="no heads"):
        al.parse_alignment_heads([])


def test_alignment_config_is_off_by_default():
    assert al.alignment_config(SimpleNamespace()) == (False, al.DEFAULT_ALIGNMENT_HEADS, al.DEFAULT_MAX_TEXT)
    config = SimpleNamespace(
        fish_omni_word_timestamps=True, fish_omni_alignment_heads="3:1", fish_omni_alignment_max_text=64
    )
    assert al.alignment_config(config) == (True, ((3, 1),), 64)
    with pytest.raises(ValueError, match="positive"):
        al.alignment_config(SimpleNamespace(fish_omni_alignment_max_text=0))


def test_prefill_stashes_text_rows_and_scores_the_sample_row():
    torch.manual_seed(0)
    capture = _capture([(3, 1), (3, 3), (7, 2)])
    # One request in slot 1: a 6-row prompt whose text occupies rows 2, 3, 4;
    # its sample row is the last prompt row (5). Other rows write the scratch slot.
    n_text, slot = 3, 1
    capture.row_slot[2:5] = slot
    capture.row_pos[2:5] = torch.arange(n_text)
    capture.begin_step(1, torch.tensor([5, 0, 0]), torch.tensor([slot, 0, 0]))
    q = torch.randn(6, HEADS * HEAD_DIM, dtype=torch.bfloat16)
    k3 = torch.randn(6, KV_HEADS * HEAD_DIM, dtype=torch.bfloat16)
    k7 = torch.randn(6, KV_HEADS * HEAD_DIM, dtype=torch.bfloat16)
    capture.capture(3, q, k3)
    capture.capture(7, q, k7)

    stash3 = capture.stash[3]
    assert stash3.shape == (4, 2, MAX_TEXT, HEAD_DIM) and stash3.dtype == torch.float16
    keys = k3[2:5].reshape(n_text, KV_HEADS, HEAD_DIM).to(torch.float16)
    assert torch.equal(stash3[slot, :, :n_text], keys.transpose(0, 1))
    assert torch.all(stash3[slot, :, n_text:] == 0)
    # Layer 7 stashes only KV head 1 (the one head 2 reads).
    assert capture.stash[7].shape[1] == 1
    assert torch.equal(capture.stash[7][slot, 0, :n_text], k7[2:5].reshape(n_text, KV_HEADS, HEAD_DIM)[:, 1].half())
    # Untouched slots stay zero; the scratch slot took the other rows.
    assert torch.all(stash3[0] == 0) and torch.all(stash3[2] == 0)
    assert torch.any(stash3[capture.scratch_slot] != 0)

    logits = capture.logits[0]
    assert capture.logits.shape == (3, 3, MAX_TEXT) and logits.dtype == torch.float32
    for index, (layer, head) in enumerate(capture.head_set.heads):
        expected = _reference(q[5].half(), (k3 if layer == 3 else k7)[2:5].half(), head)
        torch.testing.assert_close(logits[index, :n_text], expected, rtol=2e-2, atol=2e-2)
        assert torch.all(logits[index, n_text:] == 0)


def test_decode_rows_score_against_their_slots_and_stash_nothing():
    torch.manual_seed(1)
    capture = _capture([(3, 0)])
    # Slot 2 holds two stashed text keys (KV head 0, the one head 0 reads); slot 0 holds none.
    keys = torch.randn(2, KV_HEADS, HEAD_DIM).half()
    capture.stash[3][2, 0, :2] = keys[:, 0]
    before = capture.stash[3][:3].clone()
    capture.row_slot[:4].fill_(capture.scratch_slot)
    capture.begin_step(2, torch.tensor([0, 1, 0]), torch.tensor([2, 0, 0]))
    q = torch.randn(4, HEADS * HEAD_DIM, dtype=torch.bfloat16)  # padded decode batch of 4 rows
    capture.capture(3, q, torch.randn(4, KV_HEADS * HEAD_DIM, dtype=torch.bfloat16))
    assert torch.equal(capture.stash[3][:3], before)
    torch.testing.assert_close(
        capture.logits[0, 0, :2], _reference(q[0].half(), keys.reshape(2, -1), 0), rtol=2e-2, atol=2e-2
    )
    assert torch.all(capture.logits[1, 0] == 0)


def test_extract_returns_owned_fp16_rows():
    capture = _capture([(3, 0), (3, 2)])
    capture.logits[1, :, :4] = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    capture.logits[2, :, :4] = -1.0
    out = capture.extract(torch.tensor([2, 1]), 4)
    assert out.shape == (2, 2, 4) and out.dtype == torch.float16
    assert out[1].tolist() == [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]
    assert torch.all(out[0] == -1)
    out[1].zero_()
    assert capture.logits[1, 0, 1] == 1.0


def test_hooks_capture_from_the_selected_layers_only():
    layers = [_Layer(index, full=index in (3, 7)) for index in range(8)]
    capture = _capture([(3, 1), (7, 0)])
    hooks = al.install_alignment_hooks(layers, capture)
    assert sorted(hooks) == [3, 7]
    capture.begin_step(1, torch.tensor([0, 0, 0]), torch.tensor([1, 0, 0]))
    capture.stash[3][1, :, :2] = 1.0
    q = torch.ones(1, HEADS * HEAD_DIM)
    k = torch.zeros(1, KV_HEADS * HEAD_DIM)
    layers[3].self_attn(q, k, k)
    assert torch.all(capture.logits[0, 0, :2] == HEAD_DIM * HEAD_DIM**-0.5)
    assert torch.all(capture.logits[0, 1] == 0)
    layers[7].self_attn(q, k, k)
    assert torch.all(capture.logits[0, 1] == 0)  # empty stash for slot 1 at layer 7


def test_hooks_reject_layers_without_full_attention_or_with_another_index():
    layers = [_Layer(index, full=index in (3, 7)) for index in range(8)]
    with pytest.raises(ValueError, match="no full attention"):
        al.install_alignment_hooks(layers, _capture([(2, 0)]))
    with pytest.raises(ValueError, match="names layer 9"):
        al.install_alignment_hooks(layers, _capture([(9, 0)]))
    layers[7].self_attn.attn.layer_name = "model.layers.5.self_attn.attn"
    with pytest.raises(ValueError, match="reports layer index 5"):
        al.install_alignment_hooks(layers, _capture([(7, 0)]))
