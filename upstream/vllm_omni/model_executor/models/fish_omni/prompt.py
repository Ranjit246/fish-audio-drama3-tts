# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drama 3 prompt frames in the ``reference-bracket-v1`` layout.

A frame row is ``[token id, code 0, ..., code Q-1]``. Text rows carry zero
codes; a reference audio frame's token id is ``semantic_begin + code 0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import regex as re
import torch

TTS_INSTRUCTION = "convert the provided text to speech"
REFERENCE_INSTRUCTION = "convert the provided text to speech reference to the following:"
_SPEAKER_TAG = re.compile(r"\s*<\|speaker:(\d+)\|>")
# The user's text is the only segment whose literal the request supplies alone.
_TARGET_TEXT = "target_text"


@dataclass(frozen=True)
class Reference:
    transcript: str
    codes: torch.Tensor  # [frames, num_codebooks]


@dataclass(frozen=True)
class PromptLayout:
    """Prompt rows and the span of the user's text within them.

    ``target_text`` covers the text tokens of the target segment, without
    the leading ``<|speaker:N|>`` control token (one token in the released
    tokenizer), so its first row is the first word (which carries the space
    after the tag, e.g. ``" Hello"``). These rows are the columns the model's
    attention reads while voicing the text (word timestamps).
    """

    frames: torch.Tensor  # [prompt_len, 1 + num_codebooks] CPU int64
    target_text: range


def with_speaker_tag(text: str) -> str:
    """Prefix ``<|speaker:0|> `` unless the text already starts with a speaker tag."""
    if _SPEAKER_TAG.match(text):
        return text
    return f"<|speaker:0|> {text}"


def speaker_tag_match(text: str):
    """The leading ``<|speaker:N|>`` match (with any leading whitespace), or None."""
    return _SPEAKER_TAG.match(text)


def _reference_transcript(transcript: str) -> str:
    """A reference transcript starts with its speaker tag, directly after "Text: "."""
    match = _SPEAKER_TAG.match(transcript)
    if match is None:
        return f"<|speaker:0|> {transcript}"
    return f"<|speaker:{match.group(1)}|>" + transcript[match.end() :]


def _segments(text: str, references: list[Reference]) -> list[tuple[str | None, str | torch.Tensor]]:
    """``(name, literal or codes)`` in layout order; only the target text is named."""
    segments: list[tuple[str | None, str | torch.Tensor]] = [
        (None, "<|im_start|>system\n"),
        (None, REFERENCE_INSTRUCTION if references else TTS_INSTRUCTION),
    ]
    for index, reference in enumerate(references):
        segments += [
            (None, f"\n\n[Reference {index + 1}]\nText: {_reference_transcript(reference.transcript)}\nSpeech: "),
            (None, reference.codes),
        ]
    segments += [(None, "<|im_end|>\n"), (None, "<|im_start|>user\n"), (_TARGET_TEXT, with_speaker_tag(text))]
    segments += [(None, "<|im_end|>\n"), (None, "<|im_start|>assistant\n<|voice|>")]
    return segments


def _speaker_tag_tokens(tokenizer, literal: str, ids: list[int]) -> int:
    """Leading tokens of a target literal that are its ``<|speaker:N|>`` tag.

    The released tokenizer has one control token per tag, found through
    ``convert_tokens_to_ids``; a tokenizer without that method (or that
    splits the tag) keeps the tag's tokens among the text columns.
    """
    tag = _SPEAKER_TAG.match(literal)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if tag is None or not ids or convert is None:
        return 0
    try:
        tag_id = convert(tag.group(0).strip())
    except Exception:
        return 0
    return 1 if isinstance(tag_id, int) and tag_id == ids[0] else 0


def build_prompt_layout(
    tokenizer,
    text: str,
    references: list[Reference],
    *,
    semantic_begin_id: int,
    num_codebooks: int,
) -> PromptLayout:
    """CPU int64 rows [prompt_len, 1 + num_codebooks] and the target-text span.

    Each literal is tokenized on its own, because tokenizing the joined
    string changes ids at the seams; this also makes the target-text span
    exact by construction.
    """
    rows: list[torch.Tensor] = []
    cursor = 0
    target = range(0, 0)
    for name, segment in _segments(text, references):
        if isinstance(segment, str):
            ids = tokenizer.encode(segment, add_special_tokens=False)
            block = torch.zeros((len(ids), 1 + num_codebooks), dtype=torch.long)
            block[:, 0] = torch.tensor(ids, dtype=torch.long)
            if name == _TARGET_TEXT:
                target = range(cursor + _speaker_tag_tokens(tokenizer, segment, ids), cursor + len(ids))
        else:
            codes = segment.to(device="cpu", dtype=torch.long)
            if codes.ndim != 2 or codes.shape[1] != num_codebooks:
                raise ValueError(f"reference codes must be [frames, {num_codebooks}], got {tuple(codes.shape)}")
            block = torch.cat((codes[:, :1] + semantic_begin_id, codes), dim=1)
        rows.append(block)
        cursor += int(block.shape[0])
    return PromptLayout(frames=torch.cat(rows), target_text=target)


def build_prompt_frames(
    tokenizer,
    text: str,
    references: list[Reference],
    *,
    semantic_begin_id: int,
    num_codebooks: int,
) -> torch.Tensor:
    """CPU int64 rows [prompt_len, 1 + num_codebooks] (see ``build_prompt_layout``)."""
    return build_prompt_layout(
        tokenizer, text, references, semantic_begin_id=semantic_begin_id, num_codebooks=num_codebooks
    ).frames


def prompt_length(tokenizer, text: str, transcripts: list[str], frame_counts: list[int]) -> int:
    """Prompt rows for the given references without encoding their audio."""
    total = 0
    placeholder = [Reference(t, torch.zeros((n, 1), dtype=torch.long)) for t, n in zip(transcripts, frame_counts)]
    for _, segment in _segments(text, placeholder):
        if isinstance(segment, str):
            total += len(tokenizer.encode(segment, add_special_tokens=False))
        else:
            total += int(segment.shape[0])
    return total


def target_text_token_offsets(tokenizer, text: str) -> list[tuple[int, int]]:
    """Character offsets, relative to ``text``, of each target-text column's token.

    The target segment is tokenized the way ``build_prompt_layout`` tokenizes
    it (alone, with its speaker tag), so the list has exactly
    ``len(layout.target_text)`` entries for the same tokenizer. Needs a fast
    tokenizer (``return_offsets_mapping``). The first token's offsets include
    the space after the tag and are clamped to the text, so it still overlaps
    the first word.
    """
    literal = with_speaker_tag(text)
    shift = len(literal) - len(text)
    encoded = tokenizer(literal, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    offsets = [tuple(int(v) for v in pair) for pair in encoded["offset_mapping"]]
    if ids != list(tokenizer.encode(literal, add_special_tokens=False)):
        raise RuntimeError("tokenizer(...) and tokenizer.encode(...) disagree on the target text")
    offsets = offsets[_speaker_tag_tokens(tokenizer, literal, ids) :]
    return [(max(0, a - shift), max(0, b - shift)) for a, b in offsets]
