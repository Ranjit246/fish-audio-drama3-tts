"""nano-vllm runtime for Fish Audio Drama 3.

Package layout (see PLAN.md for ownership):
  config.py   - Drama 3 config.json -> fusion text config + token-id constants
  backbone/   - vendored nano-vllm-fusion Qwen3.5-MoE hybrid backbone (CPU-importable)
  weights/    - param-name map, block-FP8 dequant, safetensors loader
  tower/, sampling/, prompt/, codec/, engine.py - other work packages
"""

__version__ = "0.0.1"

from nanovllm_drama3.config import Drama3Config, FusionTextConfig  # noqa: E402,F401

__all__ = ["Drama3Config", "FusionTextConfig", "__version__"]
