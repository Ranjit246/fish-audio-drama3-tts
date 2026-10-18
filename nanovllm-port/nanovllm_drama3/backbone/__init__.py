"""Drama 3 backbone = vendored nano-vllm-fusion Qwen3.5-MoE hybrid model.

    from nanovllm_drama3.backbone import import_fusion, build_backbone
    pkg = import_fusion()                      # CPU-safe import of third_party/nano-vllm-fusion
    model = build_backbone(text_cfg, "cuda")   # Qwen3_5MoeForCausalLM with Drama 3 MoE patch

See NOTES.md in this directory for what the fusion code does and does not support.
"""

from nanovllm_drama3.backbone.vendor import (  # noqa: F401
    ensure_process_group,
    fusion_root,
    fusion_state,
    import_fusion,
)
from nanovllm_drama3.backbone.model import (  # noqa: F401
    allocate_caches,
    attention_layers,
    build_backbone,
    gdn_layers,
    reset_state_slot,
)

__all__ = [
    "import_fusion",
    "ensure_process_group",
    "fusion_root",
    "fusion_state",
    "build_backbone",
    "allocate_caches",
    "attention_layers",
    "gdn_layers",
    "reset_state_slot",
]
