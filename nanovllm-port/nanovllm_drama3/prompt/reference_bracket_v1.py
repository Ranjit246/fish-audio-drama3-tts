# SPDX-License-Identifier: Apache-2.0
# Ported from vLLM-Omni (Apache-2.0), vllm_omni/model_executor/models/fish_omni/
# prompt.py at commit 9b300b983 (identical on current main c470395); see
# nanovllm_drama3/tower/_reference/prompt.py and VENDORED-wp2.md.
"""Drama 3 prompt frames in the ``reference-bracket-v1`` layout.

A frame row is ``[token id, code 0, ..., code Q-1]``. Text rows carry zero
codes; a reference audio frame's token id is ``semantic_begin + code 0``.

Segment order (each literal is tokenized on its own):

    <|im_start|>system\\n
    <instruction>                      "convert the provided text to speech"
                                       (+ " reference to the following:" with references)
    for each reference i (1-based):
        \\n\\n[Reference i]\\nText: <|speaker:N|><transcript>\\nSpeech:    (text rows)
        <reference codes>                                         (frame rows)
    <|im_end|>\\n
    <|im_start|>user\\n
    <|speaker:0|> <text>               (the target text; see target_text_columns)
    <|im_end|>\\n
    <|im_start|>assistant\\n<|voice|>

Generation starts after the last row; the backbone's hidden state of that row
produces the first frame's main token.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import re  # the reference uses `regex`; this pattern is identical under stdlib re
import torch

from .tokens import DRAMA3_TOKENS

TTS_INSTRUCTION = "convert the provided text to speech"
REFERENCE_INSTRUCTION = "convert the provided text to speech reference to the following:"
_SPEAKER_TAG = re.compile(r"\s*<\|speaker:(\d+)\|>")

# Segment names, in layout order (references expand to reference_text_i / reference_codes_i).
SEG_SYSTEM_OPEN = "system_open"
SEG_INSTRUCTION = "instruction"
SEG_REFERENCE_TEXT = "reference_text"
SEG_REFERENCE_CODES = "reference_codes"
SEG_SYSTEM_CLOSE = "system_close"
SEG_USER_OPEN = "user_open"
SEG_TARGET_TEXT = "target_text"
SEG_USER_CLOSE = "user_close"
SEG_ASSISTANT_OPEN = "assistant_open"


@dataclass(frozen=True)
class Reference:
    transcript: str
    codes: torch.Tensor  # [frames, num_codebooks]


@dataclass(frozen=True)
class Segment:
    """One prompt segment and the rows it occupies in the frame matrix."""

    name: str
    start: int
    end: int
    text: str | None = None  # None for reference-code segments

    @property
    def rows(self) -> range:
        return range(self.start, self.end)

    def __len__(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class PromptLayout:
    frames: torch.Tensor  # [prompt_len, 1 + num_codebooks] CPU int64
    segments: tuple[Segment, ...]

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    def segment(self, name: str) -> Segment:
        for segment in self.segments:
            if segment.name == name:
                return segment
        raise KeyError(name)

    @property
    def target_text(self) -> Segment:
        return self.segment(SEG_TARGET_TEXT)


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


def _segments(text: str, references: Sequence[Reference]) -> list[tuple[str, str | torch.Tensor]]:
    """(name, literal-or-codes) in layout order. Mirrors the reference ``_segments``."""
    segments: list[tuple[str, str | torch.Tensor]] = [
        (SEG_SYSTEM_OPEN, "<|im_start|>system\n"),
        (SEG_INSTRUCTION, REFERENCE_INSTRUCTION if references else TTS_INSTRUCTION),
    ]
    for index, reference in enumerate(references):
        segments += [
            (
                f"{SEG_REFERENCE_TEXT}_{index + 1}",
                f"\n\n[Reference {index + 1}]\nText: {_reference_transcript(reference.transcript)}\nSpeech: ",
            ),
            (f"{SEG_REFERENCE_CODES}_{index + 1}", reference.codes),
        ]
    segments += [
        (SEG_SYSTEM_CLOSE, "<|im_end|>\n"),
        (SEG_USER_OPEN, "<|im_start|>user\n"),
        (SEG_TARGET_TEXT, with_speaker_tag(text)),
    ]
    segments += [
        (SEG_USER_CLOSE, "<|im_end|>\n"),
        (SEG_ASSISTANT_OPEN, "<|im_start|>assistant\n<|voice|>"),
    ]
    return segments


def _encode(tokenizer: Any, literal: str) -> list[int]:
    return list(tokenizer.encode(literal, add_special_tokens=False))


def build_prompt_layout(
    tokenizer: Any,
    text: str,
    references: Sequence[Reference] = (),
    *,
    semantic_begin_id: int = DRAMA3_TOKENS.semantic_start_token_id,
    num_codebooks: int = DRAMA3_TOKENS.num_codebooks,
) -> PromptLayout:
    """``build_prompt_frames`` plus the row span of every segment."""
    rows: list[torch.Tensor] = []
    segments: list[Segment] = []
    cursor = 0
    for name, segment in _segments(text, references):
        if isinstance(segment, str):
            ids = torch.tensor(_encode(tokenizer, segment), dtype=torch.long)
            block = torch.zeros((ids.numel(), 1 + num_codebooks), dtype=torch.long)
            block[:, 0] = ids
            literal: str | None = segment
        else:
            codes = segment.to(device="cpu", dtype=torch.long)
            if codes.ndim != 2 or codes.shape[1] != num_codebooks:
                raise ValueError(f"reference codes must be [frames, {num_codebooks}], got {tuple(codes.shape)}")
            block = torch.cat((codes[:, :1] + semantic_begin_id, codes), dim=1)
            literal = None
        rows.append(block)
        segments.append(Segment(name, cursor, cursor + int(block.shape[0]), literal))
        cursor += int(block.shape[0])
    return PromptLayout(frames=torch.cat(rows), segments=tuple(segments))


def build_prompt_frames(
    tokenizer: Any,
    text: str,
    references: Sequence[Reference] = (),
    *,
    semantic_begin_id: int = DRAMA3_TOKENS.semantic_start_token_id,
    num_codebooks: int = DRAMA3_TOKENS.num_codebooks,
) -> torch.Tensor:
    """CPU int64 rows [prompt_len, 1 + num_codebooks]. Each literal is tokenized on
    its own, because tokenizing the joined string changes ids at the seams."""
    return build_prompt_layout(
        tokenizer, text, references, semantic_begin_id=semantic_begin_id, num_codebooks=num_codebooks
    ).frames


def prompt_length(tokenizer: Any, text: str, transcripts: Sequence[str], frame_counts: Sequence[int]) -> int:
    """Prompt rows for the given references without encoding their audio."""
    total = 0
    placeholder = [Reference(t, torch.zeros((n, 1), dtype=torch.long)) for t, n in zip(transcripts, frame_counts)]
    for _, segment in _segments(text, placeholder):
        if isinstance(segment, str):
            total += len(_encode(tokenizer, segment))
        else:
            total += int(segment.shape[0])
    return total


def target_text_columns(
    tokenizer: Any,
    text: str,
    references: Sequence[Reference] = (),
    *,
    include_speaker_tag: bool = False,
    speaker_token_ids: frozenset[int] | set[int] | None = None,
    semantic_begin_id: int = DRAMA3_TOKENS.semantic_start_token_id,
    num_codebooks: int = DRAMA3_TOKENS.num_codebooks,
) -> range:
    """Row indices (prompt columns, for attention maps) of the user's target text.

    By default the leading ``<|speaker:N|>`` control token is excluded, so the
    range covers exactly the text tokens (the first of which carries the space
    after the tag, e.g. ``" Hello"``). ``speaker_token_ids`` defaults to the
    released tokenizer's 16 speaker tokens; pass your own for another tokenizer.
    """
    layout = build_prompt_layout(
        tokenizer, text, references, semantic_begin_id=semantic_begin_id, num_codebooks=num_codebooks
    )
    return target_text_columns_of(
        layout, include_speaker_tag=include_speaker_tag, speaker_token_ids=speaker_token_ids
    )


def target_text_columns_of(
    layout: PromptLayout,
    *,
    include_speaker_tag: bool = False,
    speaker_token_ids: frozenset[int] | set[int] | None = None,
) -> range:
    """``target_text_columns`` for an already built layout."""
    segment = layout.target_text
    start, end = segment.start, segment.end
    if not include_speaker_tag and end > start:
        ids = speaker_token_ids if speaker_token_ids is not None else DRAMA3_TOKENS.speaker_token_ids
        if int(layout.frames[start, 0]) in ids:
            start += 1
    return range(start, end)


__all__ = [
    "REFERENCE_INSTRUCTION",
    "SEG_ASSISTANT_OPEN",
    "SEG_INSTRUCTION",
    "SEG_REFERENCE_CODES",
    "SEG_REFERENCE_TEXT",
    "SEG_SYSTEM_CLOSE",
    "SEG_SYSTEM_OPEN",
    "SEG_TARGET_TEXT",
    "SEG_USER_CLOSE",
    "SEG_USER_OPEN",
    "TTS_INSTRUCTION",
    "PromptLayout",
    "Reference",
    "Segment",
    "build_prompt_frames",
    "build_prompt_layout",
    "prompt_length",
    "speaker_tag_match",
    "target_text_columns",
    "target_text_columns_of",
    "with_speaker_tag",
]
