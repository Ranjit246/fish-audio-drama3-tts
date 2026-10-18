# SPDX-License-Identifier: Apache-2.0
"""Special token ids of the released ``fishaudio/drama-3`` checkpoint.

Values come from ``config.json`` (semantic band, eos, pad) and
``tokenizer.json`` (control tokens). They are defaults only; the prompt and
sampling code take ids as parameters so a different checkpoint can override.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Drama3Tokens:
    # config.json
    semantic_start_token_id: int = 248099  # <|semantic:0|>
    semantic_end_token_id: int = 252194  # <|semantic:4095|>
    eos_token_id: int = 248046  # <|im_end|> (end of speech for the main token)
    audio_pad_token_id: int = 248076  # <|audio_pad|>
    pad_token_id: int = 248044  # <|endoftext|>
    # tokenizer.json control tokens used by the prompt layout
    im_start_token_id: int = 248045  # <|im_start|>
    im_end_token_id: int = 248046  # <|im_end|>
    voice_token_id: int = 248081  # <|voice|>
    speaker0_token_id: int = 248083  # <|speaker:0|> ... <|speaker:15|> are consecutive
    num_speakers: int = 16
    num_codebooks: int = 10

    @property
    def codebook_size(self) -> int:
        return self.semantic_end_token_id - self.semantic_start_token_id + 1

    def speaker_token_id(self, index: int) -> int:
        if not 0 <= index < self.num_speakers:
            raise ValueError(f"speaker index {index} out of range 0..{self.num_speakers - 1}")
        return self.speaker0_token_id + index

    @property
    def speaker_token_ids(self) -> frozenset[int]:
        return frozenset(range(self.speaker0_token_id, self.speaker0_token_id + self.num_speakers))

    def is_semantic(self, token_id: int) -> bool:
        return self.semantic_start_token_id <= token_id <= self.semantic_end_token_id


DRAMA3_TOKENS = Drama3Tokens()

__all__ = ["DRAMA3_TOKENS", "Drama3Tokens"]
