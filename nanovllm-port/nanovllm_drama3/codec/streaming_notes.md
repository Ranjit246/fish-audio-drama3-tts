# Streaming the Fish DAC decoder — notes (no implementation)

Read from vllm-omni `c470395` `fish_speech/dac_channels_last.py` (`ChannelsLastDACDecoder.receptive_frames`,
`_receptive_field`, `streaming_decode_shapes`, `DACStreamState`) and `fish_speech_dac_decoder.py`
(`_capture_graphs`, `forward` context trimming), plus the module shapes of `build_dac_codec()`.

## Decode graph and rates

```
codes [B, 10, F]  (F frames @ 21.53 Hz, 2048 samples/frame)
  └ quantizer.decode:
      embed + out_proj per codebook, sum (semantic + 9 residual)           [B, 1024, F]
      post_module  = WindowLimitedTransformer(8 layers, dim 1024, causal, window_size=128)
      upsample     = 2 × (CausalTransConvNet k=2,s=2 → ConvNeXtBlock dwconv k=7 causal)  → [B, 1024, 4F]
  └ decoder.model (all causal convs; the decoder "transformer layers" are never registered upstream):
      CausalWNConv1d k=7
      4 × DecoderBlock: Snake → CausalWNConvTranspose1d(k=2s, s) → 3 × ResidualUnit(k=7, dil 1/3/9 + k=1)
         strides 8, 8, 4, 2  → total ×512
      Snake → CausalWNConv1d k=7 → Tanh                                     [B, 1, 2048·F]
```

Everything is strictly causal (left padding only, transposed convs trimmed on the right), so a frame's
samples depend only on frames ≤ itself. Output length is exactly `F × 2048`; there is no look-ahead and
no output delay to compensate.

## State a streaming decoder needs

1. **Transformer context (`post_module`)** — causal attention with `window_size=128`: frame *t* attends to
   frames `[t-127, t]`. Exact streaming needs per-layer K/V of the last **127 frames** (8 layers,
   16 heads × 64 dim, i.e. 1024 floats × 2 (K,V) × 8 layers per frame ≈ 64 KiB/frame in fp32) plus the
   absolute position for RoPE (`freqs_cis[input_pos]`, `block_size=4096` → max 4096 frames ≈ 190 s
   per stream before positions run out; the upstream ring uses `freqs_cis.shape[0]` as the limit).
   The reference, when *not* keeping K/V state, instead re-decodes a left context of
   `codec_left_context_frames` = **25 frames** (default; `chunk` also 25 frames ≈ 1.16 s) and treats the
   transformer's attention as approximately converged after 25 frames — i.e. it accepts a small
   approximation beyond the conv receptive field. The transformer always sees the full window including
   context; only the convs are trimmed.

2. **Convolution history after the transformer** — the left receptive field of everything downstream
   of `post_module`, measured in codec frames (`ChannelsLastDACDecoder.receptive_frames`):
   - `upsample` ConvNeXt dwconvs: `(7-1)/2 + (7-1)/4 = 4.5` frames
   - `decoder.model`: first k=7 conv at 4 latents/frame (1.5), then each DecoderBlock's transposed conv
     (`ceil(k/s)-1 = 1` input step) and its three ResidualUnits (`6·(1+3+9) = 78` taps at the block's rate)
     → `4.6455` frames
   - total **9.15 → `receptive_frames = 10`** codec frames (= 20,480 samples of left context).
   With ≥ 10 frames of left context re-decoded (or the equivalent per-layer conv tails kept), the convs'
   samples equal the untrimmed decode bit-for-bit; the reference therefore decodes a window with
   `trim = context - min(context, receptive_frames)` frames dropped from the conv stage, then cuts
   `context × 2048` samples from the waveform.

   The stateful path (`DACStreamState`, `fish_speech_dac_stateful`) keeps instead, per request slot:
   a per-layer left-padding history for every causal conv / dwconv (`history(key, pad)` of size
   `kernel_size - stride` at that layer's rate), and a K/V ring of 128 frames per transformer layer.
   Then a window carries *only* its new frames.

3. **Chunking policy** used upstream: optional ramp (`codec_chunk_ramp`, e.g. small first chunks for low
   first-audio latency), then `codec_chunk_frames` = 25 new frames per window, with
   `context = min(frames_emitted_so_far, 25)`. Windows are bucketed to `{8,16,24,32,48,64,…}` frames
   for CUDA graphs; irrelevant for an eager CPU/GPU module decode.

4. **Encoder side** (if streaming the *reference* encode were ever needed): also fully causal.
   Conv receptive field 7,977 samples (encoder) + downsample path 19,968 samples; encoder transformer
   in block 4 has `window_size=512` latents (= 128 frames), `pre_module` window 128 frames. Not needed
   for the MVP (reference audio is encoded once, offline).

## Minimal non-stateful streaming recipe (what package 4 could do later)

```
emitted = 0
for each new chunk of frames (≥1):
    ctx   = min(emitted, 25)              # ≥ 10 keeps convs exact; 25 is the upstream default
    window = codes[emitted-ctx : emitted+len(chunk)]
    wav    = dac.decode(window)[ctx*2048:] # drop left-context samples
    emitted += len(chunk)
```

Cost: each window re-decodes `ctx` extra frames (×2 work at chunk = ctx = 25). Numerical note: with
ctx ≥ 10 the conv stage is exact; the transformer output differs slightly from a full-sequence decode
for frames whose true 128-frame attention window extends beyond `ctx` — upstream accepts this.
Exact streaming requires the per-layer conv tails + 127-frame K/V described above.
