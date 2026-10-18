# Vendored files — work package 3 (codec)

Source repo: vLLM-Omni (`vllm-project/vllm-omni`), commit `c470395bde2f8de4f6f29b54adf970f885afa32f`,
directory `vllm_omni/model_executor/models/fish_speech/`. All files are Apache-2.0 (headers kept);
`dac_layers.py` is itself adapted upstream from descript-audio-codec (MIT) and the DAC files from the
fish-speech 0.1.0 PyPI release (Apache-2.0), as their headers say.

| File here (`nanovllm_drama3/codec/`) | Source path (at `c470395`) | Edits |
|---|---|---|
| `dac_conv.py` | `fish_speech/dac_modules/conv.py` | provenance header only |
| `dac_layers.py` | `fish_speech/dac_modules/layers.py` | provenance header; `snake` = `torch.jit.script` with eager fallback (TorchScript is deprecated in torch ≥ 2.6 and may be unavailable); legacy `weight_norm` FutureWarning filtered (the legacy `weight_g`/`weight_v` layout is what `codec.pth` stores) |
| `dac_rvq.py` | `fish_speech/dac_modules/rvq.py` | provenance header; imports → relative |
| `dac_transformer.py` | `fish_speech/dac_modules/transformer.py` | provenance header only |
| `dac_codec.py` | `fish_speech/dac_modules/codec.py` | provenance header; imports → relative |
| `dac_utils.py` | `fish_speech/dac_utils.py` | imports → relative (module-level instead of lazy); otherwise unchanged (`build_dac_codec`, `bake_weight_norm`, constants) |
| `fish_dac.py` | new; logic lifted from `fish_speech/dac_encoder.py` (`_load_dac_codec`, `_prepare_reference_audio_tensor`, `encode_reference_audio_codes`) and the module (non-streaming) path of `fish_speech/fish_speech_dac_decoder.py` (`_ensure_codec_loaded`, `_decode_codes`) | `FishDAC` class; no vllm/vllm_omni/transformers deps |
| `resample.py` | new; port of `torchaudio.functional.resample` (`sinc_interp_hann`, lowpass_filter_width 6, rolloff 0.99), BSD-2-Clause | replaces `torchaudio.transforms.Resample` used by `dac_encoder.py`; `tests/test_codec.py::test_resample_matches_torchaudio` checks equality (atol 1e-6) |

Not vendored (GPU-only / vllm-runtime glue): `dac_channels_last.py` (channels-last + CUDA-graph +
stateful streaming decoder; its window/receptive-field logic is summarised in `streaming_notes.md`),
`fish_speech_dac_decoder.py` (vllm model-runner wrapper), `dac_encoder.py` (codec cache + vllm logger).

Weights: `codec.pth` from `fishaudio/drama-3` (1,871,099,728 bytes; plain state dict, loads with
`missing_keys == unexpected_keys == []` against `build_dac_codec()`; 391.4M parameters, fp32).
