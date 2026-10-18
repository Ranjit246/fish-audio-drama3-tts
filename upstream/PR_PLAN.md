# Upstream PRs to vllm-project/vllm-omni

Pinned base: vllm-omni `c470395` (submodule `vllm-omni/`), vLLM 0.31.0.
Files under `upstream/` mirror the upstream tree and are the PR payloads.

| # | PR | Payload | Evidence | State |
|---|---|---|---|---|
| 1 | `[Deploy][Drama 3] Low-latency profile + RTX PRO 6000 recipe section` | `upstream/vllm_omni/deploy/fish_omni_low_latency.yaml`; a "Low-latency profile" section for `recipes/fishaudio/Drama-3.md` (`upstream/recipes/...`, to write from `results/report.md`) | 1088-line Seed-TTS en runs at c=1/16/32 + WER/SIM (`../results/final_c_*`) | profile written; recipe text pending the final numbers |
| 2 | `[Bugfix][Drama 3] bsgemm-moe must honour vLLM's Blackwell DeepGEMM auto-disable` | patch to `vllm_omni/model_executor/models/fish_omni/quantization.py` | startup traceback on sm_120 with CUDA 12.9 toolkit: `DeepGEMM SM120 kernels require CUDA 13.0 or newer`; vLLM logs "Auto-disabled DeepGemm ... Falling back to CUTLASS" at the API-server level but stage 0 still logs "Using DEEPGEMM Fp8 MoE backend" | to write + test on the VM once the GPU is free |
| 3 | `[Build] opencv-python -> opencv-python-headless` | requirements change | `ImportError: libGL.so.1` in stage 0 on a headless VM | to write |
| 4 | `[Feature][Drama 3] Word timestamps from the model's own attention` | `upstream/patches/0002-fish-omni-attention-word-timestamps.patch` (on top of 0001): `fish_omni/alignment.py` capture + model-state staging, `codes.alignment` payload field, DAC pass-through, `speech.alignment.delta` SSE events / `X-Word-Timestamps`, `deploy/fish_omni_word_timestamps.yaml`, docs, CPU tests | design in `../TIMESTAMPS.md`; head sweep in `../timestamps/results/alignment_heads.json`; GPU procedure in `VALIDATION-0002.md` | patch written, CPU tests pass; GPU validation pending |

Process notes (from #8900 / #8933 reviews): fill the four "Contributor intent"
questions; Test Plan must state vLLM version + vllm-omni commit; `pre-commit` clean;
leave a self-review comment before a reviewer is assigned; the review bot reproduces
perf claims, so include the exact benchmark commands.
