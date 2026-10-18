# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the Drama 3 (fish_omni) prompt frames."""

import pytest
import torch

from vllm_omni.model_executor.models.fish_omni.prompt import (
    REFERENCE_INSTRUCTION,
    TTS_INSTRUCTION,
    Reference,
    build_prompt_frames,
    prompt_length,
    with_speaker_tag,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SEMANTIC_BEGIN = 1000
NUM_CODEBOOKS = 3


class CharTokenizer:
    """One id per character, and a log of every encoded literal."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert add_special_tokens is False
        self.calls.append(text)
        return [ord(c) for c in text]


def _decode(ids: torch.Tensor) -> str:
    return "".join(chr(int(i)) for i in ids)


def _build(tokenizer: CharTokenizer, text: str, references: list[Reference]) -> torch.Tensor:
    return build_prompt_frames(
        tokenizer, text, references, semantic_begin_id=SEMANTIC_BEGIN, num_codebooks=NUM_CODEBOOKS
    )


def test_no_reference_layout():
    tokenizer = CharTokenizer()
    frames = _build(tokenizer, "Hello there.", [])

    assert tokenizer.calls == [
        "<|im_start|>system\n",
        TTS_INSTRUCTION,
        "<|im_end|>\n",
        "<|im_start|>user\n",
        "<|speaker:0|> Hello there.",
        "<|im_end|>\n",
        "<|im_start|>assistant\n<|voice|>",
    ]
    assert frames.dtype == torch.long
    assert frames.shape == (len("".join(tokenizer.calls)), 1 + NUM_CODEBOOKS)
    assert _decode(frames[:, 0]) == "".join(tokenizer.calls)
    assert torch.all(frames[:, 1:] == 0)


def test_one_reference_layout():
    codes = torch.tensor([[5, 1, 2], [7, 3, 4]])
    tokenizer = CharTokenizer()
    frames = _build(tokenizer, "Hello.", [Reference("We asked twenty people.", codes)])

    head = f"<|im_start|>system\n{REFERENCE_INSTRUCTION}\n\n[Reference 1]\nText: <|speaker:0|> We asked twenty people.\nSpeech: "
    tail = "<|im_end|>\n<|im_start|>user\n<|speaker:0|> Hello.<|im_end|>\n<|im_start|>assistant\n<|voice|>"
    assert tokenizer.calls == [
        "<|im_start|>system\n",
        REFERENCE_INSTRUCTION,
        "\n\n[Reference 1]\nText: <|speaker:0|> We asked twenty people.\nSpeech: ",
        "<|im_end|>\n",
        "<|im_start|>user\n",
        "<|speaker:0|> Hello.",
        "<|im_end|>\n",
        "<|im_start|>assistant\n<|voice|>",
    ]
    assert frames.shape == (len(head) + 2 + len(tail), 1 + NUM_CODEBOOKS)
    assert _decode(frames[: len(head), 0]) == head
    assert torch.all(frames[: len(head), 1:] == 0)
    # A reference frame row is [semantic_begin + code 0, code 0, ..., code Q-1].
    audio = frames[len(head) : len(head) + 2]
    assert audio.tolist() == [[SEMANTIC_BEGIN + 5, 5, 1, 2], [SEMANTIC_BEGIN + 7, 7, 3, 4]]
    assert _decode(frames[len(head) + 2 :, 0]) == tail
    assert torch.all(frames[len(head) + 2 :, 1:] == 0)


def test_reference_codes_shape_is_checked():
    with pytest.raises(ValueError, match="reference codes must be"):
        _build(CharTokenizer(), "Hi.", [Reference("ref", torch.zeros((4, NUM_CODEBOOKS + 1), dtype=torch.long))])


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello.", "<|speaker:0|> Hello."),
        ("<|speaker:3|> Hello.", "<|speaker:3|> Hello."),
        ("<|speaker:12|>Hello.", "<|speaker:12|>Hello."),
        ("  <|speaker:1|> Hello.", "  <|speaker:1|> Hello."),
        ("Hi. <|speaker:1|> Hello.", "<|speaker:0|> Hi. <|speaker:1|> Hello."),
        ("[sighs] Fine.", "<|speaker:0|> [sighs] Fine."),
    ],
)
def test_text_speaker_tag(text, expected):
    assert with_speaker_tag(text) == expected


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        ("Some words.", "Text: <|speaker:0|> Some words.\n"),
        ("<|speaker:2|> Some words.", "Text: <|speaker:2|> Some words.\n"),
        # Leading white space before the tag is dropped: the tag follows "Text: ".
        ("  <|speaker:2|> Some words.", "Text: <|speaker:2|> Some words.\n"),
    ],
)
def test_reference_transcript_speaker_tag(transcript, expected):
    tokenizer = CharTokenizer()
    _build(tokenizer, "Hi.", [Reference(transcript, torch.zeros((1, NUM_CODEBOOKS), dtype=torch.long))])
    assert expected in tokenizer.calls[2]


def test_inline_event_tags_pass_through():
    tokenizer = CharTokenizer()
    text = "<|speaker:0|> [sighs] Fine. [laughs] Well."
    frames = _build(tokenizer, text, [])
    assert text in _decode(frames[:, 0])


@pytest.mark.parametrize("frame_counts", [[], [1], [37]])
def test_prompt_length_matches_frames(frame_counts):
    text = "<|speaker:1|> The quarterly report is due on Thursday."
    transcripts = ["We asked over twenty different people."] * len(frame_counts)
    references = [Reference(t, torch.randint(0, 64, (n, NUM_CODEBOOKS))) for t, n in zip(transcripts, frame_counts)]
    rows = int(_build(CharTokenizer(), text, references).shape[0])
    assert prompt_length(CharTokenizer(), text, transcripts, frame_counts) == rows


class TagTokenizer(CharTokenizer):
    """One id per character, and one control id per leading ``<|speaker:N|>`` tag (like the released tokenizer)."""

    TAG_ID = 0x110000

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        self.calls.append(text)
        if text.startswith("<|speaker:") and "|>" in text:
            end = text.index("|>") + 2
            return [self.TAG_ID + int(text[len("<|speaker:") : end - 2])] + [ord(c) for c in text[end:]]
        return [ord(c) for c in text]

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.encode(token)[0]


def test_layout_names_the_target_text_rows_without_the_speaker_tag():
    from vllm_omni.model_executor.models.fish_omni.prompt import build_prompt_layout

    tokenizer = TagTokenizer()
    codes = torch.zeros((5, NUM_CODEBOOKS), dtype=torch.long)
    layout = build_prompt_layout(
        tokenizer,
        "Hello there.",
        [Reference("We asked.", codes)],
        semantic_begin_id=SEMANTIC_BEGIN,
        num_codebooks=NUM_CODEBOOKS,
    )
    assert torch.equal(layout.frames, _build(TagTokenizer(), "Hello there.", [Reference("We asked.", codes)]))
    # The tag is one row, excluded; the columns are " Hello there." (the first carries the space).
    assert _decode(layout.frames[list(layout.target_text), 0]) == " Hello there."
    assert int(layout.frames[layout.target_text.start - 1, 0]) == TagTokenizer.TAG_ID
    # A caller-supplied tag is excluded the same way; a tokenizer that splits
    # the tag (plain characters) keeps its rows among the columns.
    tagged = build_prompt_layout(
        TagTokenizer(), "<|speaker:3|>Hi.", [], semantic_begin_id=SEMANTIC_BEGIN, num_codebooks=NUM_CODEBOOKS
    )
    assert _decode(tagged.frames[list(tagged.target_text), 0]) == "Hi."
    split = build_prompt_layout(CharTokenizer(), "Hi.", [], semantic_begin_id=SEMANTIC_BEGIN, num_codebooks=NUM_CODEBOOKS)
    assert _decode(split.frames[list(split.target_text), 0]) == "<|speaker:0|> Hi."



def _load_drama3_tokenizer():
    """The cached Drama 3 tokenizer (its snapshot directory, so no config.json or network is needed)."""
    import glob
    import os

    from transformers import AutoTokenizer

    root = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )
    snapshots = sorted(glob.glob(os.path.join(root, "models--fishaudio--drama-3", "snapshots", "*", "tokenizer.json")))
    if snapshots:
        return AutoTokenizer.from_pretrained(os.path.dirname(snapshots[-1]))
    return AutoTokenizer.from_pretrained("fishaudio/drama-3", local_files_only=True)


def test_target_text_token_offsets_match_the_layout_columns_with_the_real_tokenizer():
    from vllm_omni.model_executor.models.fish_omni.prompt import build_prompt_layout, target_text_token_offsets

    try:
        tokenizer = _load_drama3_tokenizer()
    except Exception as exc:  # pragma: no cover - offline without the cached tokenizer
        pytest.skip(f"Drama 3 tokenizer not cached: {exc}")
    for text in ("Get the trust fund to the bank early.", "  leading spaces and 3,500 dollars!", "<|speaker:2|> Tagged."):
        layout = build_prompt_layout(tokenizer, text, [], semantic_begin_id=248099, num_codebooks=10)
        offsets = target_text_token_offsets(tokenizer, text)
        assert len(offsets) == len(layout.target_text) > 0
        # Offsets tile the text in order and the tag's own token is gone.
        assert all(0 <= a <= b <= len(text) for a, b in offsets)
        assert all(prev[1] <= cur[0] or cur[0] == 0 for prev, cur in zip(offsets, offsets[1:]))
        assert 248083 <= int(layout.frames[layout.target_text.start - 1, 0]) < 248083 + 16  # the <|speaker:N|> row
