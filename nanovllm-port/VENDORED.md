# Vendored sources

Record every third-party file or tree copied into this repo: source repo, commit,
path in that repo, path here, license. Other packages append their own sections.

## Package 1 (backbone)

### nano-vllm-fusion (whole repo, shallow clone)

| Field | Value |
|---|---|
| Repo | https://github.com/xiez-njust-xmu/nano-vllm-fusion |
| Commit | `3e44d3008de0a42bc655f3523f324c6804623cf5` (2026-09-03, "更新readme文档") |
| Path here | `third_party/nano-vllm-fusion/` (unmodified; patched at import time by `nanovllm_drama3/backbone/`) |
| License | MIT, `Copyright (c) 2025 Xingkai Yu` (upstream nano-vllm author; fork keeps the MIT LICENSE file) |
| Used files | `nanovllm/models/qwen3_5_moe.py`, `nanovllm/models/qwen3_5.py`, `nanovllm/layers/{gated_delta_net,attention,linear,layernorm,embed_head,rotary_embedding,activation}.py`, `nanovllm/utils/{loader,context}.py`, engine files for package 4 |

Runtime patches (no files edited in `third_party/`):

* `nanovllm_drama3/backbone/moe_patch.py` rebinds `qwen3_5_moe.Qwen3_5MoeBlock`
  to a subclass with a per-token `Linear(hidden, 1)` shared-expert gate (the
  checkpoint shape); upstream used a scalar.
* `nanovllm_drama3/backbone/cpu_fallback.py` supplies CPU `flash_attn_*` and
  `store_kvcache` references when Triton / flash-attn are not installed.

### Drama 3 metadata (no weights)

| Field | Value |
|---|---|
| Repo | https://huggingface.co/fishaudio/drama-3 (revision `main`, fetched 2026-10-18) |
| Files | `config.json` -> `nanovllm_drama3/weights/drama3_meta/config.json`; `model.safetensors.index.json` -> `.../model.safetensors.index.json.gz`; safetensors *headers* of the 8 shards (HTTP range requests, dtype+shape only) -> `.../tensor_shapes.json.gz` |
| License | Fish Audio model license applies to the weights; the metadata here is used for name/shape mapping only |

## Work package 2 (tower / sampler / prompt)


To be merged into `VENDORED.md` by package 1.

All files below are Apache-2.0, `SPDX-FileCopyrightText: Copyright contributors
to the vLLM-Omni project`, from https://github.com/vllm-project/vllm-omni.

## Verbatim copies (provenance only, not imported at runtime)

Directory: `nanovllm_drama3/tower/_reference/` — fetched with
`gh api repos/vllm-project/vllm-omni/contents/<path>?ref=9b300b983` (raw).

| file | repo path | commit |
|---|---|---|
| `codebook_tower.py` | `vllm_omni/model_executor/models/fish_omni/codebook_tower.py` | `9b300b98338daf1ce3366b6598b870248b5c32f3` |
| `fish_omni_model.py` | `vllm_omni/model_executor/models/fish_omni/fish_omni_model.py` | `9b300b983…` |
| `prompt.py` | `vllm_omni/model_executor/models/fish_omni/prompt.py` | `9b300b983…` |
| `configuration_fish_omni.py` | `vllm_omni/model_executor/models/fish_omni/configuration_fish_omni.py` | `9b300b983…` |
| `__init__.py` | `vllm_omni/model_executor/models/fish_omni/__init__.py` (empty) | `9b300b983…` |

Commit `9b300b983` = "[CI/Build] Run Omni-DuplexEval nightly through bench serve (#8871)",
the last eager (pre-#8933 MRv2/paged) version of the Drama 3 tower.
`tests/test_prompt.py` imports `_reference/prompt.py` to cross-check the layout.

## Ported (rewritten, same semantics)

| ours | derived from | notes |
|---|---|---|
| `nanovllm_drama3/tower/tower.py` | `codebook_tower.py` (GemmaRMSNorm, interleaved_rope, TowerAttention/FeedForward/Block/Layer, CodebookTower.run) | paged cache + flash-attn + CUDA graphs replaced by a growing per-layer KV and SDPA; `prefill`/`step` API from `fish_omni_model.py` (`_prefill_tower`, `_run_pending_prefills`, `_tower_step`) |
| `nanovllm_drama3/tower/config.py` | `configuration_fish_omni.py` (`FishOmniAudioDecoderConfig`) | dataclass, no transformers |
| `nanovllm_drama3/sampling/sampler.py` | `codebook_tower.py` (`sample_rows`, `sample_top_k_top_p`), `fish_omni_model.py` (`sample`, `_RAS_*`, `_HEAD_SAMPLING`) | plus `main_candidate_ids` from PR #9044 ("never sample semantic code 0", current main) |
| `nanovllm_drama3/prompt/reference_bracket_v1.py` | `prompt.py` | identical on main `c470395`; adds `PromptLayout`/`Segment` and `target_text_columns`; uses stdlib `re` instead of `regex` |
| `nanovllm_drama3/prompt/tokens.py` | ids from `fishaudio/drama-3` `config.json` + `tokenizer.json` | — |

## Downloaded data (not committed)

* `fishaudio/drama-3` `model.safetensors.index.json` → tower parameter names in
  `tests/test_tower_param_names.txt` (192 names, prefixes `audio_decoder.`,
  `audio_decoder_mtp_shared.`, `audio_decoder_mtp.`).
* `fishaudio/drama-3` tokenizer files (`tokenizer.json`, `tokenizer_config.json`,
  `special_tokens_map.json`, `chat_template.jinja`) via
  `huggingface_hub.snapshot_download` into the HF cache at test time
  (`nanovllm_drama3.prompt.load_tokenizer`).

## Work package 3 (codec)

See `nanovllm_drama3/codec/VENDORED-wp3.md`.
