# SPDX-License-Identifier: Apache-2.0
"""reference-bracket-v1 prompt layout: segment order vs the vendored reference,
frame rows, prompt_length, speaker tags and target-text columns."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanovllm_drama3.prompt import (  # noqa: E402
    DRAMA3_TOKENS,
    REFERENCE_INSTRUCTION,
    SEG_TARGET_TEXT,
    TTS_INSTRUCTION,
    Reference,
    build_prompt_frames,
    build_prompt_layout,
    prompt_length,
    target_text_columns,
    target_text_columns_of,
    with_speaker_tag,
)

Q = DRAMA3_TOKENS.num_codebooks
BEGIN = DRAMA3_TOKENS.semantic_start_token_id


class StubTokenizer:
    """Deterministic offline tokenizer: ``<|...|>`` control tokens get the released
    ids where known, every other character is one token (its code point)."""

    _CONTROL = re.compile(r"<\|[^|]+\|>")
    _IDS = {
        "<|im_start|>": DRAMA3_TOKENS.im_start_token_id,
        "<|im_end|>": DRAMA3_TOKENS.im_end_token_id,
        "<|voice|>": DRAMA3_TOKENS.voice_token_id,
        **{f"<|speaker:{i}|>": DRAMA3_TOKENS.speaker_token_id(i) for i in range(16)},
    }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        ids: list[int] = []
        cursor = 0
        for match in self._CONTROL.finditer(text):
            ids += [ord(c) for c in text[cursor : match.start()]]
            ids.append(self._IDS[match.group(0)])
            cursor = match.end()
        ids += [ord(c) for c in text[cursor:]]
        return ids

    def decode(self, ids: list[int]) -> str:
        inverse = {v: k for k, v in self._IDS.items()}
        return "".join(inverse.get(i, chr(i)) for i in ids)


def load_reference_prompt_module():
    """The vendored vLLM-Omni prompt.py (uses the `regex` package; stdlib re is equivalent here)."""
    path = ROOT / "nanovllm_drama3" / "tower" / "_reference" / "prompt.py"
    sys.modules.setdefault("regex", re)
    name = "vendored_fish_omni_prompt"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses with postponed annotations look the module up here
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tok() -> StubTokenizer:
    return StubTokenizer()


def ref_codes(frames: int, seed: int = 0) -> torch.Tensor:
    return torch.randint(0, 4096, (frames, Q), generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------------------
# layout vs reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_refs", [0, 1, 2])
def test_frames_equal_vendored_reference(tok: StubTokenizer, n_refs: int):
    reference_module = load_reference_prompt_module()
    refs = [Reference(f"transcript {i}", ref_codes(3 + i, seed=i)) for i in range(n_refs)]
    ref_refs = [reference_module.Reference(r.transcript, r.codes) for r in refs]
    ours = build_prompt_frames(tok, "Hello world.", refs, semantic_begin_id=BEGIN, num_codebooks=Q)
    theirs = reference_module.build_prompt_frames(tok, "Hello world.", ref_refs, semantic_begin_id=BEGIN, num_codebooks=Q)
    assert torch.equal(ours, theirs)
    assert prompt_length(tok, "Hello world.", [r.transcript for r in refs], [r.codes.shape[0] for r in refs]) == ours.shape[0]
    assert reference_module.prompt_length(
        tok, "Hello world.", [r.transcript for r in refs], [r.codes.shape[0] for r in refs]
    ) == ours.shape[0]


def test_segment_order_without_reference(tok: StubTokenizer):
    layout = build_prompt_layout(tok, "Hi there")
    assert [s.name for s in layout.segments] == [
        "system_open",
        "instruction",
        "system_close",
        "user_open",
        "target_text",
        "user_close",
        "assistant_open",
    ]
    texts = [s.text for s in layout.segments]
    assert texts == [
        "<|im_start|>system\n",
        TTS_INSTRUCTION,
        "<|im_end|>\n",
        "<|im_start|>user\n",
        "<|speaker:0|> Hi there",
        "<|im_end|>\n",
        "<|im_start|>assistant\n<|voice|>",
    ]
    # Segments tile the frame matrix.
    assert layout.segments[0].start == 0 and layout.segments[-1].end == len(layout)
    assert all(a.end == b.start for a, b in zip(layout.segments, layout.segments[1:]))
    # Text rows carry zero codes; the last row is <|voice|>.
    assert torch.equal(layout.frames[:, 1:], torch.zeros_like(layout.frames[:, 1:]))
    assert int(layout.frames[-1, 0]) == DRAMA3_TOKENS.voice_token_id
    assert int(layout.frames[0, 0]) == DRAMA3_TOKENS.im_start_token_id


def test_segment_order_with_references(tok: StubTokenizer):
    refs = [Reference("first clip", ref_codes(4, 1)), Reference("<|speaker:2|> second clip", ref_codes(2, 2))]
    layout = build_prompt_layout(tok, "Target.", refs)
    names = [s.name for s in layout.segments]
    assert names == [
        "system_open",
        "instruction",
        "reference_text_1",
        "reference_codes_1",
        "reference_text_2",
        "reference_codes_2",
        "system_close",
        "user_open",
        "target_text",
        "user_close",
        "assistant_open",
    ]
    assert layout.segment("instruction").text == REFERENCE_INSTRUCTION
    assert layout.segment("reference_text_1").text == "\n\n[Reference 1]\nText: <|speaker:0|> first clip\nSpeech: "
    # A transcript's own speaker tag is kept and placed directly after "Text: " (whitespace after it preserved).
    assert layout.segment("reference_text_2").text == "\n\n[Reference 2]\nText: <|speaker:2|> second clip\nSpeech: "
    # Reference audio rows: token id = semantic_begin + code0, codes copied.
    for i, ref in enumerate(refs, start=1):
        seg = layout.segment(f"reference_codes_{i}")
        rows = layout.frames[seg.start : seg.end]
        assert rows.shape == (ref.codes.shape[0], 1 + Q)
        assert torch.equal(rows[:, 1:], ref.codes)
        assert torch.equal(rows[:, 0], ref.codes[:, 0] + BEGIN)


def test_reference_codes_shape_is_checked(tok: StubTokenizer):
    with pytest.raises(ValueError):
        build_prompt_frames(tok, "x", [Reference("t", torch.zeros(3, Q - 1, dtype=torch.long))])


# ---------------------------------------------------------------------------
# speaker tags
# ---------------------------------------------------------------------------


def test_with_speaker_tag():
    assert with_speaker_tag("Hello") == "<|speaker:0|> Hello"
    assert with_speaker_tag("<|speaker:3|> Hello") == "<|speaker:3|> Hello"
    assert with_speaker_tag("  <|speaker:12|>Hello") == "  <|speaker:12|>Hello"  # leading whitespace tolerated
    assert with_speaker_tag("Say <|speaker:1|> later") == "<|speaker:0|> Say <|speaker:1|> later"


def test_target_text_keeps_explicit_speaker(tok: StubTokenizer):
    layout = build_prompt_layout(tok, "<|speaker:5|> Hola")
    seg = layout.target_text
    assert int(layout.frames[seg.start, 0]) == DRAMA3_TOKENS.speaker_token_id(5)
    assert tok.decode(layout.frames[seg.start : seg.end, 0].tolist()) == "<|speaker:5|> Hola"


# ---------------------------------------------------------------------------
# target text columns
# ---------------------------------------------------------------------------


def test_target_text_columns_exclude_speaker_tag_by_default(tok: StubTokenizer):
    text = "Hello world."
    refs = [Reference("clip", ref_codes(5))]
    layout = build_prompt_layout(tok, text, refs)
    cols = target_text_columns(tok, text, refs)
    seg = layout.target_text
    assert cols == range(seg.start + 1, seg.end)
    assert tok.decode(layout.frames[cols.start : cols.stop, 0].tolist()) == " Hello world."
    with_tag = target_text_columns(tok, text, refs, include_speaker_tag=True)
    assert with_tag == range(seg.start, seg.end)
    assert target_text_columns_of(layout) == cols
    # Columns lie after every reference frame and before the assistant turn.
    assert cols.start >= layout.segment("reference_codes_1").end
    assert cols.stop <= layout.segment("assistant_open").start
    # Equivalent segment lookup by constant.
    assert layout.segment(SEG_TARGET_TEXT) == seg


# ---------------------------------------------------------------------------
# real tokenizer (downloads ~20 MB of tokenizer files; skipped offline)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_tok():
    pytest.importorskip("transformers")
    from nanovllm_drama3.prompt import load_tokenizer

    try:
        return load_tokenizer()
    except Exception as exc:  # network / cache problems
        pytest.skip(f"fishaudio/drama-3 tokenizer unavailable: {exc}")


def test_real_tokenizer_control_ids(real_tok):
    assert real_tok.encode("<|im_start|>", add_special_tokens=False) == [DRAMA3_TOKENS.im_start_token_id]
    assert real_tok.encode("<|im_end|>", add_special_tokens=False) == [DRAMA3_TOKENS.im_end_token_id]
    assert real_tok.encode("<|voice|>", add_special_tokens=False) == [DRAMA3_TOKENS.voice_token_id]
    assert real_tok.encode("<|speaker:0|>", add_special_tokens=False) == [DRAMA3_TOKENS.speaker_token_id(0)]
    assert real_tok.encode("<|speaker:15|>", add_special_tokens=False) == [DRAMA3_TOKENS.speaker_token_id(15)]
    assert real_tok.encode("<|semantic:0|>", add_special_tokens=False) == [BEGIN]
    assert real_tok.encode("<|semantic:4095|>", add_special_tokens=False) == [DRAMA3_TOKENS.semantic_end_token_id]
    assert DRAMA3_TOKENS.eos_token_id == DRAMA3_TOKENS.im_end_token_id == real_tok.convert_tokens_to_ids("<|im_end|>")


def test_real_tokenizer_layout_and_target_columns(real_tok):
    text = "The quick brown fox jumps over the lazy dog."
    refs = [Reference("A short reference clip.", ref_codes(7))]
    layout = build_prompt_layout(real_tok, text, refs)
    assert int(layout.frames[0, 0]) == DRAMA3_TOKENS.im_start_token_id
    assert int(layout.frames[-1, 0]) == DRAMA3_TOKENS.voice_token_id
    assert prompt_length(real_tok, text, ["A short reference clip."], [7]) == len(layout)
    seg = layout.target_text
    assert int(layout.frames[seg.start, 0]) == DRAMA3_TOKENS.speaker_token_id(0)
    cols = target_text_columns_of(layout)
    assert cols.start == seg.start + 1 and cols.stop == seg.end
    assert real_tok.decode(layout.frames[cols.start : cols.stop, 0].tolist()) == " " + text
    codes_seg = layout.segment("reference_codes_1")
    assert torch.equal(layout.frames[codes_seg.start : codes_seg.end, 1:], refs[0].codes)
    # Non-reference rows carry zero codes.
    mask = torch.ones(len(layout), dtype=torch.bool)
    mask[codes_seg.start : codes_seg.end] = False
    assert int(layout.frames[mask, 1:].abs().sum()) == 0
