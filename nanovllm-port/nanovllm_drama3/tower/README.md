# Drama 3 codebook tower (`nanovllm_drama3.tower`)

Eager PyTorch port of the vLLM-Omni `fish_omni/codebook_tower.py` (commit
`9b300b983`, see `_reference/` and `VENDORED-wp2.md`). No vLLM, no flash
attention, no CUDA graphs; one request at a time (B = 1).

## What the tower is

Each Drama 3 frame has Q = 10 codes. The backbone (Qwen3.5 MoE) samples only
**code 0** — the "main token", one of 4096 semantic token ids
`[semantic_begin, semantic_begin + 4096)` or end-of-speech. The tower predicts
**codes 1..9** of the same frame from

* the backbone's final-norm hidden state of the row that produced the main token, and
* the main token itself (its semantic code is embedded in every tower layer).

It has 13 layers, all with the *same* block structure (`TowerLayer`):

| group | checkpoint prefix | count | embedding table | input code |
|---|---|---|---|---|
| frame embeddings | `audio_decoder.codebook_embeddings` | 1 table `[10*4096, 2048]` | codebook k code c → row `k*4096 + c` | — |
| shared layers | `audio_decoder_mtp_shared.layers.{0..3}` | 4 | `semantic_embedding` | code 0 of the *next* frame |
| residual layers | `audio_decoder_mtp.layers.{0..8}` | 9 | `codebook_embedding` | code r of the next frame (r = layer index) |
| heads | `audio_decoder_mtp.heads.{0..8}` | 9 | `Linear(2048, 4096)` | head r predicts code r+1 |

Residual layer r is fed code r and head r predicts code r+1, so **code k
conditions on codes < k** (sequential within a frame, 9 sampling steps).

Every layer attends **causally over the whole sequence** — prompt rows and
generated frames alike — with its own key/value history (`LayerKV`), GQA
(16 query heads / 2 KV heads, head_dim 256), GemmaRMSNorm q/k norms, GPT-J
interleaved rope on the first 25% of each head (base 1e7), and a sigmoid output
gate (`wqkv` rows are `[query | gate | key | value]`).

### One layer

```
embedded  = code_is_semantic ? table[code] : 0        (+ extra_input in shared layer 0)
mixed     = eh_proj( cat( enorm(embedded), hnorm(hidden) ) )
attended  = attention( attention_norm(mixed) )        # appends to this layer's KV
ffn_in, residual = ffn_norm(attended, residual=mixed) # residual = attended + mixed
out       = final_layernorm( feed_forward(ffn_in), residual=residual )  # norm(ffn + residual)
```

All norms are zero-centered-gain RMSNorm (`scale = 1 + weight`).

## Dataflow over the sequence

Rows are indexed like the backbone's tokens. A prompt built by
`nanovllm_drama3.prompt` has `T` rows `[token_id, code0..code9]` (text rows have
zero codes; reference-audio rows have `token_id = semantic_begin + code0`).

```
tower position p   reads   hidden[p]      backbone hidden state at row p
                           frame[p]       input frame: codes 2..9 summed via
                                          codebook_embeddings -> extra input of shared layer 0
                                          (only when frame[p] is a semantic row)
                           main[p+1]      main token of row p+1 -> code 0 embedded in every layer
                   writes  KV at p in all 13 layers
                   emits   codes 1..9 of frame p+1
```

* **Prefill** (`prefill(hidden[T, dim], frames[T, 11], cache)`): rows `0..T-2`
  are teacher-forced in one pass — the label of row p is prompt frame p+1, so
  residual layer r is fed the *true* code r. Row `T-1` is not run; its frame is
  stored as `cache.input_frame` for the first step.
* **Step k** (`step(hidden_of_last_row, main_token, cache)`): the backbone has
  just produced hidden state `h[T-1+k]` and the engine sampled `main_token`
  from its logits (see `nanovllm_drama3.sampling.MainTokenSampler`). The tower
  runs position `T-1+k` with `input_frame` (last prompt frame for k = 0, then the
  previously generated frame), embeds `code0 = main_token - semantic_begin`, and
  samples codes 1..9 head by head (default temperature 1.0 / top_k 30 / top_p 0.9,
  optional seeded `torch.Generator`). Returns the frame row
  `[main_token, code0..code9]` (codes are zero when the main token is
  end-of-speech) and stores it as the next `input_frame`.

Timing for a prompt of `T` rows (same as the reference docstring):

```
prefill   backbone rows 0..T-1 -> h[0..T-1];  tower teacher-forced on rows 0..T-2
decode k  main[T+k] sampled from h[T-1+k]  -> tower at position T-1+k with h[T-1+k] -> frame T+k
          backbone consumes embed(frame T+k) -> h[T+k]
```

### Backbone input embedding

The tower also owns the codebook table the backbone input needs
(`backbone_input_embedding(token_embeds, frames)`): for semantic rows
`(E_lm[token] + E_cb[codebook 1, code1]) / sqrt(2)` (`vq_backbone_codebooks = 1`),
text rows pass through unchanged. Package 4 must apply this to every prompt row
and to every generated frame before feeding the backbone.

## API

```python
from nanovllm_drama3.tower import CodebookTower, TowerConfig

tower = CodebookTower(TowerConfig.drama3(), semantic_begin_id=248099).to(torch.bfloat16)
tower.load_state(safetensors_dict)            # keys audio_decoder*.*, strict by default

cache = tower.new_cache()                      # per-request KV + input frame
tower.prefill(hidden_T, frames_T, cache)       # teacher-force prompt rows
frame = tower.step(h_last, main_token, cache)  # -> LongTensor[11]
emb   = tower.backbone_input_embedding(embed_tokens(frame[:1]), frame[None])  # next backbone input
```

Lower level: `tower.run(hidden, input_codes, input_is_semantic, next_main_tokens,
next_codes | None, cache, sampling=..., generator=..., return_head_logits=...)`
runs any number of consecutive positions (teacher-forced when `next_codes` is
given), appending to the cache — used by the tests to show that incremental and
batched execution agree.

Parameter names equal the checkpoint's 192 `audio_decoder*` tensors
(`tests/test_tower_param_names.txt`, extracted from
`fishaudio/drama-3/model.safetensors.index.json`).

## Notes / differences from the reference

* The reference keeps a paged `TowerKVCache` with slots and uses flash attention
  on CUDA; this port keeps a growing `[len, kv_heads, head_dim]` tensor per layer
  and plain `scaled_dot_product_attention` (the reference's CPU fallback path).
  Numerics follow the reference's eager path (fp32 norms, dtype of the weights elsewhere).
* Codes of an end-of-speech frame are zeroed (as on current vLLM-Omni main; the
  pre-#8933 code returned whatever the heads sampled, which was never used).
* `rope_base=1e7`, `rotary_percent=0.25`, `norm_eps=1e-6` come from
  `config.json["audio_decoder_config"]`; `TowerConfig.from_hf_dict` reads them.
