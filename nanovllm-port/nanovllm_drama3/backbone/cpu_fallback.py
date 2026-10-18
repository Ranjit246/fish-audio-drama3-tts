"""CPU stand-ins for the GPU-only bits of nano-vllm-fusion.

The only kernels in the fusion backbone that are not plain PyTorch live in
`nanovllm/layers/attention.py`: a Triton `store_kvcache` kernel and the two
flash-attn entry points. GatedDeltaNet, MoE routing, norms and rotary are pure
torch. These reference implementations let the whole backbone run (slowly) on
CPU for tests; they are NOT used when real `triton`/`flash_attn` are importable.
"""

from __future__ import annotations

import sys
import types

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# flash-attn reference
# --------------------------------------------------------------------------
def _sdpa_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, causal: bool) -> torch.Tensor:
    """q: [Lq, H, D], k/v: [Lk, Hkv, D] -> [Lq, H, D]. Keys may be longer than queries
    (decode / prefix); causal alignment is bottom-right like flash-attn."""
    Lq, H, D = q.shape
    Lk, Hkv, _ = k.shape
    rep = H // Hkv
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    qf = q.float().transpose(0, 1)  # [H, Lq, D]
    kf = k.float().transpose(0, 1)
    vf = v.float().transpose(0, 1)
    scores = (qf @ kf.transpose(-1, -2)) * scale  # [H, Lq, Lk]
    if causal:
        i = torch.arange(Lq, device=q.device).unsqueeze(1)
        j = torch.arange(Lk, device=q.device).unsqueeze(0)
        mask = j > (i + (Lk - Lq))
        scores = scores.masked_fill(mask, float("-inf"))
    probs = scores.softmax(dim=-1)
    out = probs @ vf  # [H, Lq, D]
    return out.transpose(0, 1).to(q.dtype)


def flash_attn_varlen_func(
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    softmax_scale=None, causal=True, block_table=None, **_unused,
):
    """Prefill: q/k/v are packed [total, heads, D]. Prefix-cache (`block_table`) is
    not supported here; fusion disables prefix caching for hybrid models anyway."""
    if block_table is not None:
        raise NotImplementedError("cpu flash_attn_varlen_func: block_table (prefix cache) unsupported")
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    outs = []
    cq = cu_seqlens_q.tolist()
    ck = cu_seqlens_k.tolist()
    for i in range(len(cq) - 1):
        qi = q[cq[i]:cq[i + 1]]
        ki = k[ck[i]:ck[i + 1]]
        vi = v[ck[i]:ck[i + 1]]
        outs.append(_sdpa_gqa(qi, ki, vi, scale, causal))
    return torch.cat(outs, dim=0)


def flash_attn_with_kvcache(
    q, k_cache, v_cache, cache_seqlens=None, block_table=None, softmax_scale=None, causal=True, **_unused,
):
    """Decode: q [B, 1, H, D]; caches [num_blocks, block_size, Hkv, D]; block_table [B, nb]."""
    B, Lq, H, D = q.shape
    scale = softmax_scale if softmax_scale is not None else D ** -0.5
    outs = []
    for b in range(B):
        n = int(cache_seqlens[b])
        blocks = block_table[b]
        blocks = blocks[blocks >= 0]
        kb = k_cache[blocks.long()].reshape(-1, k_cache.shape[-2], D)[:n]
        vb = v_cache[blocks.long()].reshape(-1, v_cache.shape[-2], D)[:n]
        outs.append(_sdpa_gqa(q[b], kb, vb, scale, causal))
    return torch.stack(outs, dim=0)


# --------------------------------------------------------------------------
# store_kvcache reference (replaces the Triton kernel)
# --------------------------------------------------------------------------
def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    valid = slot_mapping >= 0
    slots = slot_mapping[valid].long()
    k_cache.view(-1, D)[slots] = key.reshape(N, D)[valid].to(k_cache.dtype)
    v_cache.view(-1, D)[slots] = value.reshape(N, D)[valid].to(v_cache.dtype)


# --------------------------------------------------------------------------
# module stubs
# --------------------------------------------------------------------------
class _JitStub:
    """Mimics a `@triton.jit` function object; launching it is an error on CPU."""

    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, grid):
        def _launch(*args, **kwargs):
            raise RuntimeError("Triton kernel launched on CPU; cpu_fallback.store_kvcache should have replaced it")
        return _launch

    def __call__(self, *args, **kwargs):
        raise RuntimeError("Triton kernel called on CPU")


def make_triton_stub() -> types.ModuleType:
    triton = types.ModuleType("triton")
    triton.__nanovllm_drama3_stub__ = True
    triton.jit = lambda fn=None, **kw: (_JitStub(fn) if fn is not None else (lambda f: _JitStub(f)))
    tl = types.ModuleType("triton.language")
    tl.constexpr = int
    for name in ("program_id", "load", "store", "arange"):
        setattr(tl, name, lambda *a, **k: None)
    triton.language = tl
    return triton


def make_flash_attn_stub() -> types.ModuleType:
    fa = types.ModuleType("flash_attn")
    fa.__nanovllm_drama3_stub__ = True
    fa.flash_attn_varlen_func = flash_attn_varlen_func
    fa.flash_attn_with_kvcache = flash_attn_with_kvcache
    return fa


def _real_module_available(name: str) -> bool:
    import importlib.util

    mod = sys.modules.get(name)
    if mod is not None:
        return not getattr(mod, "__nanovllm_drama3_stub__", False)
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


class stubbed_imports:
    """Context manager: expose `triton` / `flash_attn` stubs in sys.modules only
    while a module that imports them is being executed, then remove them again.

    Leaving the stubs registered would break torch itself (inductor does
    `from triton import Config` when it probes for Triton), so the importing
    module keeps references to the stub objects in its own globals and
    `sys.modules` is restored on exit. `installed` records what was stubbed.
    """

    def __init__(self, force: bool = False) -> None:
        self.force = force
        self.installed: dict[str, bool] = {}
        self._saved: dict[str, object] = {}

    def __enter__(self) -> "stubbed_imports":
        for name, maker in (("triton", make_triton_stub), ("flash_attn", make_flash_attn_stub)):
            if self.force or not _real_module_available(name):
                mod = maker()
                for key in ((name, mod), ("triton.language", mod.language)) if name == "triton" else ((name, mod),):
                    self._saved[key[0]] = sys.modules.get(key[0])
                    sys.modules[key[0]] = key[1]
                self.installed[name] = True
            else:
                self.installed[name] = False
        return self

    def __exit__(self, *exc) -> None:
        for key, prev in self._saved.items():
            if prev is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = prev

