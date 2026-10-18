"""Construct the Drama 3 backbone (fusion `Qwen3_5MoeForCausalLM`) and its caches."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn

from nanovllm_drama3.config import FusionTextConfig
from nanovllm_drama3.backbone.vendor import ensure_process_group, import_fusion


@contextmanager
def default_device_dtype(device: torch.device, dtype: torch.dtype) -> Iterator[None]:
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            yield
    finally:
        torch.set_default_dtype(prev_dtype)


def build_backbone(
    text_cfg: FusionTextConfig,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Instantiate `Qwen3_5MoeForCausalLM(text_cfg)` on `device` with uninitialised
    weights (same construction path as fusion's ModelRunner, minus CUDA setup).

    `text_cfg.dtype` is overwritten with `dtype` so fusion's runner, if used later,
    agrees with the parameters.
    """
    dev = torch.device(device)
    import_fusion()
    ensure_process_group(dev)
    from nanovllm.models.qwen3_5_moe import Qwen3_5MoeForCausalLM

    text_cfg.dtype = dtype
    with default_device_dtype(dev, dtype):
        model = Qwen3_5MoeForCausalLM(text_cfg)
    return model


def gdn_layers(model: nn.Module) -> list[nn.Module]:
    from nanovllm.layers.gated_delta_net import GatedDeltaNet

    return [m for m in model.modules() if isinstance(m, GatedDeltaNet)]


def attention_layers(model: nn.Module) -> list[nn.Module]:
    return [m for m in model.modules() if hasattr(m, "k_cache") and hasattr(m, "v_cache")]


def allocate_caches(
    model: nn.Module,
    text_cfg: FusionTextConfig,
    *,
    num_blocks: int,
    block_size: int,
    num_state_slots: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor | int]:
    """Allocate paged KV cache for full-attention layers and GDN conv/recurrent
    state slots, mirroring `ModelRunner.allocate_kv_cache` / `allocate_gdn_state`
    but with explicit sizes (no GPU-memory probing). Returns the tensors so a
    caller can keep them alive / inspect them.
    """
    dev = torch.device(device)
    attn = attention_layers(model)
    kv = torch.zeros(
        2, len(attn), num_blocks, block_size, text_cfg.num_key_value_heads, text_cfg.head_dim,
        dtype=dtype, device=dev,
    )
    for i, layer in enumerate(attn):
        layer.k_cache = kv[0, i]
        layer.v_cache = kv[1, i]

    gdn = gdn_layers(model)
    conv_states = []
    rec_states = []
    for layer in gdn:
        cs = torch.zeros(num_state_slots, layer.conv_dim, layer.conv_kernel_size - 1, dtype=dtype, device=dev)
        rs = torch.zeros(num_state_slots, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim, dtype=torch.float32, device=dev)
        layer.conv_states = cs
        layer.recurrent_states = rs
        conv_states.append(cs)
        rec_states.append(rs)
    return {
        "kv_cache": kv,
        "conv_states": conv_states,
        "recurrent_states": rec_states,
        "num_blocks": num_blocks,
        "block_size": block_size,
        "num_state_slots": num_state_slots,
    }


def reset_state_slot(model: nn.Module, slot: int) -> None:
    """Zero a GDN state slot (fusion never clears slots on reuse; prefill writes a
    fresh state only when `warmup` is False, so stale conv state would leak)."""
    for layer in gdn_layers(model):
        if layer.conv_states.numel():
            layer.conv_states[slot].zero_()
            layer.recurrent_states[slot].zero_()
