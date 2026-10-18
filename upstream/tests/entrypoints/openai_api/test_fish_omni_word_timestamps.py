# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Word timestamps from Drama 3's own attention at the speech endpoint: the
adapter's capability and request parameters, and the SSE event order
(``speech.alignment.delta`` after the audio delta its frames arrived with)."""

import json
import types

import numpy as np
import pytest
import torch

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.tts_adapters import resolve_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.fish_omni import FishOmniAdapter
from vllm_omni.utils.attention_alignment import FRAME_MS

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

TEXT = "one two three four"


TAG = "<|speaker:0|>"


class _Tokenizer:
    """One token per character and one control token for the speaker tag, with character offsets."""

    def _pieces(self, text):
        if text.startswith(TAG):
            yield 999, (0, len(TAG))
            text, base = text[len(TAG) :], len(TAG)
        else:
            base = 0
        for i, c in enumerate(text):
            yield ord(c) % 900, (base + i, base + i + 1)

    def encode(self, text, add_special_tokens=False):
        return [token for token, _ in self._pieces(text)]

    def convert_tokens_to_ids(self, token):
        return 999 if token == TAG else ord(token[0]) % 900

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        pieces = list(self._pieces(text))
        return {"input_ids": [t for t, _ in pieces], "offset_mapping": [o for _, o in pieces]}


class _Server:
    _tts_executor = None
    uploaded_speakers: dict = {}
    forced_aligner_enabled = False

    def _apply_uploaded_speaker(self, request):
        return None

    def _validate_ref_audio_format(self, ref_audio):
        return None


def _engine(enabled: bool, max_text: int | None = None):
    hf_config = types.SimpleNamespace(audio_decoder_config=types.SimpleNamespace(num_codebooks=10, vocab_size=4096))
    if enabled:
        hf_config.fish_omni_word_timestamps = True
    if max_text is not None:
        hf_config.fish_omni_alignment_max_text = max_text
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(hf_config=hf_config, max_model_len=8192, model="fishaudio/drama-3")
    )


def _adapter(enabled: bool, max_text: int | None = None, server=None) -> FishOmniAdapter:
    adapter = FishOmniAdapter(SpeechServingContext(server=server or _Server(), engine_client=_engine(enabled, max_text)))
    adapter._tokenizer = _Tokenizer()
    return adapter


def _request(**kw) -> OpenAICreateSpeechRequest:
    body = {"input": TEXT, "word_timestamps": True}
    body.update(kw)
    return OpenAICreateSpeechRequest(**body)


def test_capability_follows_the_stage_0_override():
    assert resolve_adapter("fish_omni_tts") is FishOmniAdapter  # the registry is keyed by adapter name
    assert not _adapter(False).supports_attention_alignment()
    assert _adapter(True).supports_attention_alignment()


def test_without_capture_or_aligner_the_request_is_rejected_with_a_hint():
    error = _adapter(False).validate(_request())
    assert error is not None and "fish_omni_word_timestamps" in error and "--forced-aligner" in error
    assert _adapter(True).validate(_request()) is None


@pytest.mark.asyncio
async def test_build_marks_the_request_and_prepares_the_word_decoder_parameters():
    adapter = _adapter(True)
    prepared = await adapter.build(_request(), [], False)
    assert prepared.prompt["additional_information"]["word_timestamps"] is True
    params = prepared.tts_params["word_alignment"]
    assert params["text"] == TEXT
    # The character tokenizer keeps the space after the speaker tag as the first column.
    assert len(params["offsets"]) == len(TEXT) + 1 and params["offsets"][0] == [0, 0]
    assert params["offsets"][1:] == [[i, i + 1] for i in range(len(TEXT))]
    # Without the flag nothing is added.
    plain = await adapter.build(_request(word_timestamps=False), [], False)
    assert "word_timestamps" not in plain.prompt["additional_information"] and "word_alignment" not in plain.tts_params


@pytest.mark.asyncio
async def test_texts_beyond_the_capture_capacity_are_rejected():
    with pytest.raises(ValueError, match="up to 4 tokens"):
        await _adapter(True, max_text=4).build(_request(), [], False)


def _serving(enabled: bool = True) -> OmniOpenAIServingSpeech:
    serving = OmniOpenAIServingSpeech.__new__(OmniOpenAIServingSpeech)
    serving._tts_model_type = FishOmniAdapter.name
    serving._diffusion_mode = False
    serving.forced_aligner_enabled = False
    serving.engine_client = _engine(enabled)
    serving._adapter = FishOmniAdapter(SpeechServingContext(server=serving, engine_client=serving.engine_client))
    serving._build_speech_usage = lambda *args, **kwargs: types.SimpleNamespace(model_dump=lambda: {"total_tokens": 0})
    return serving


def test_the_request_gate_knows_which_models_align_from_attention():
    assert _serving(True)._uses_attention_alignment(_request())
    assert not _serving(True)._uses_attention_alignment(_request(word_timestamps=False))
    assert not _serving(False)._uses_attention_alignment(_request())


def _planted_rows(path: np.ndarray, n_text: int, heads: int = 2) -> torch.Tensor:
    rows = np.zeros((len(path), heads, n_text), dtype=np.float32)
    rows[np.arange(len(path)), :, path] = 8.0
    return torch.from_numpy(rows).to(torch.float16)


@pytest.mark.asyncio
async def test_alignment_events_follow_the_audio_delta_that_brought_their_frames():
    serving = _serving()
    words = TEXT.split()
    offsets = [[0, 0]] + [[TEXT.index(w), TEXT.index(w) + len(w)] for w in words]  # space column + one per word
    n_text = len(offsets)
    # 30 frames read the four words in turn (column 0 is the space after the tag).
    path = 1 + np.minimum(np.arange(30) * 4 // 30, 3)
    chunks = [path[:8], path[8:18], path[18:30]]

    async def results():
        for rows in chunks:
            yield types.SimpleNamespace(
                multimodal_output={
                    "model_outputs": torch.zeros(len(rows) * 2048),
                    "sr": 44100,
                    "alignment": _planted_rows(rows, n_text),
                }
            )
        yield types.SimpleNamespace(multimodal_output={"model_outputs": torch.empty(0), "sr": 44100})

    request = _request(stream=True, response_format="pcm")
    tts_params = {"word_alignment": {"text": TEXT, "offsets": offsets}}
    events = []
    async for event in serving._generate_audio_sse_events(results(), "req", "pcm", request=request, tts_params=tts_params):
        kind, data = event.split("\n")[0].removeprefix("event: "), json.loads(event.split("\ndata: ")[1])
        events.append((kind, data))
    kinds = [kind for kind, _ in events]
    assert kinds[-1] == "speech.audio.done"
    assert kinds.count("speech.audio.delta") == 3
    alignment = [data for kind, data in events if kind == "speech.alignment.delta"]
    assert alignment, "no speech.alignment.delta events"
    # Every alignment event follows an audio delta (never precedes the first one),
    # and the last one, with every word final, comes right before speech.audio.done.
    for index, kind in enumerate(kinds):
        if kind == "speech.alignment.delta":
            assert "speech.audio.delta" in kinds[:index]
    assert kinds[-2] == "speech.alignment.delta"
    final = alignment[-1]
    assert [w["word"] for w in final["words"]] == words and all(w["final"] for w in final["words"])
    assert final["resolution_ms"] == round(FRAME_MS, 2) and "tail_silence_ms" in final["words"][-1]
    assert final["words"][0]["start_ms"] == 0 and final["words"][-1]["end_ms"] == round(30 * FRAME_MS)
    # Earlier events settle words in order: a word reported final keeps its span in the final list.
    settled = {w["word"]: (w["start_ms"], w["end_ms"]) for event in alignment[:-1] for w in event["words"] if w["final"]}
    assert settled and all((w["start_ms"], w["end_ms"]) == settled[w["word"]] for w in final["words"] if w["word"] in settled)
    # Within an event, final words come first; the provisional ones (the words the
    # path has reached but not settled, up to four frames behind) follow.
    for event in alignment[:-1]:
        flags = [w["final"] for w in event["words"]]
        assert flags == sorted(flags, reverse=True)
