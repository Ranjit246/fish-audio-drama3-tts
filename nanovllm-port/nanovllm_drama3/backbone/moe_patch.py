"""Corrected MoE block for the fusion backbone.

nano-vllm-fusion's `Qwen3_5MoeBlock` models `shared_expert_gate` as a scalar
`nn.Parameter(torch.ones(1))` applied to the shared-expert *output*. The Drama 3
checkpoint (like HF Qwen3-MoE / Qwen3.5-MoE) stores
`shared_expert_gate.weight` with shape `(1, hidden_size)`: it is a
`Linear(hidden_size -> 1, bias=False)` whose sigmoid gates the shared expert
*per token*, computed from the block *input*:

    out = sum_k w_k * expert_k(x) + sigmoid(x @ W_gate^T) * shared_expert(x)

Loading the (1, 2048) tensor into fusion's (1,) parameter would fail, and the
math is different, so we swap the class before the model is built.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SharedExpertGate(nn.Module):
    """`sigmoid(Linear(hidden_size -> 1))`, weight named `.weight` with shape (1, hidden)."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(F.linear(hidden_states, self.weight))  # [N, 1]


def make_drama3_moe_block(fusion_moe_module):
    """Build a subclass of fusion's Qwen3_5MoeBlock with the corrected shared gate."""
    Base = fusion_moe_module.Qwen3_5MoeBlock

    class Drama3MoeBlock(Base):
        def __init__(self, config) -> None:
            super().__init__(config)
            self.norm_topk_prob = bool(getattr(config, "norm_topk_prob", True))
            if self.shared_expert is not None:
                self.shared_expert_gate = SharedExpertGate(config.hidden_size)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            num_tokens, hidden_dim = hidden_states.shape
            router_logits = self.gate(hidden_states)
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            if self.norm_topk_prob:
                routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
            routing_weights = routing_weights.to(hidden_states.dtype)

            final = torch.zeros_like(hidden_states)
            # [E, top_k, N] one-hot, same scheme as upstream (python loop over hit experts).
            expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
            hit = expert_mask.sum(dim=(-1, -2)).nonzero(as_tuple=False).flatten().tolist()
            for expert_idx in hit:
                idx, top_x = torch.where(expert_mask[expert_idx])
                current = hidden_states[top_x]
                out = self.experts[expert_idx](current) * routing_weights[top_x, idx, None]
                final.index_add_(0, top_x, out.to(hidden_states.dtype))

            if self.shared_expert is not None:
                shared_out = self.shared_expert(hidden_states)
                final = final + self.shared_expert_gate(hidden_states) * shared_out
            return final

    Drama3MoeBlock.__name__ = "Drama3MoeBlock"
    Drama3MoeBlock.__qualname__ = "Drama3MoeBlock"
    return Drama3MoeBlock


def install(fusion_moe_module) -> type:
    """Replace `Qwen3_5MoeBlock` in the fusion module namespace (idempotent).

    `Qwen3_5MoeDecoderLayer.__init__` resolves `Qwen3_5MoeBlock` from module
    globals at call time, so rebinding the name is enough.
    """
    current = fusion_moe_module.Qwen3_5MoeBlock
    if getattr(current, "__name__", "") == "Drama3MoeBlock":
        return current
    patched = make_drama3_moe_block(fusion_moe_module)
    fusion_moe_module._UpstreamQwen3_5MoeBlock = current
    fusion_moe_module.Qwen3_5MoeBlock = patched
    return patched
