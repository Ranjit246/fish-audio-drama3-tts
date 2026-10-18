"""The .pth hook against a CPU fake of vLLM's Qwen3NextAttention.

The fake mirrors what the hook relies on: ``forward(positions, hidden_states)``
calling ``self.attn(q, k, v)`` with q ``[rows, num_heads*head_dim]`` and k
``[rows, num_kv_heads*head_dim]`` (post-norm, post-RoPE in the real thing), the
attributes ``num_heads``, ``num_kv_heads``, ``head_dim``, ``scaling`` and a layer
index readable from ``self.attn.layer_name``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

H, KV, D, HID = 4, 2, 8, 16


class FakeInnerAttention(nn.Module):
    def __init__(self, layer_name: str):
        super().__init__()
        self.layer_name = layer_name

    def forward(self, q, k, v):
        return q


class FakeQwen3NextAttention(nn.Module):
    def __init__(self, layer_idx: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.num_heads, self.num_kv_heads, self.head_dim = H, KV, D
        self.scaling = D**-0.5
        self.qkv_proj = nn.Linear(HID, (H + 2 * KV) * D, bias=False)
        self.attn = FakeInnerAttention(f"model.language_model.layers.{layer_idx}.self_attn.attn")

    def project(self, positions, hidden_states):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([H * D, KV * D, KV * D], dim=-1)
        # a stand-in for RoPE so q/k depend on the position
        rot = (1.0 + 0.05 * positions.to(q.dtype)).unsqueeze(-1)
        return q * rot, k * rot, v

    def forward(self, positions, hidden_states):
        q, k, v = self.project(positions, hidden_states)
        return self.attn(q, k, v)


def _run_request(layers, hook, T: int, n_steps: int, seed: int):
    """Drive a prefill + decode steps through the patched fakes; return expected logits per layer."""
    g = torch.Generator().manual_seed(seed)
    prompt_hidden = torch.randn(T, HID, generator=g)
    positions = torch.arange(T)
    expected = {}
    for mod in layers:
        with torch.no_grad():
            out = mod(positions, prompt_hidden)
        assert out.shape == (T, H * D)
        _, k, _ = mod.project(positions, prompt_hidden)
        expected[mod] = {"k": k.reshape(T, KV, D), "steps": []}
    for s in range(n_steps):
        hidden = torch.randn(1, HID, generator=g)
        pos = torch.tensor([T + s])
        for mod in layers:
            with torch.no_grad():
                mod(pos, hidden)
            q, _, _ = mod.project(pos, hidden)
            q0 = q[0].reshape(H, D)
            kexp = expected[mod]["k"].repeat_interleave(H // KV, dim=1)  # [T, H, D]
            expected[mod]["steps"].append(torch.einsum("hd,thd->ht", q0, kexp) * mod.scaling)
    return expected


def test_prefill_decode_flush_roundtrip(hook_module, tmp_path):
    hook = hook_module
    layers = [FakeQwen3NextAttention(3, 1), FakeQwen3NextAttention(7, 2)]
    assert hook.patch_class(FakeQwen3NextAttention)
    assert not hook.patch_class(FakeQwen3NextAttention)  # idempotent

    exp1 = _run_request(layers, hook, T=9, n_steps=5, seed=10)
    assert hook.STATE.cur is not None and hook.STATE.cur.prompt_len == 9
    assert not hook.STATE.flushed  # nothing flushed until the next request arrives
    exp2 = _run_request(layers, hook, T=12, n_steps=3, seed=11)  # its prefill flushes request 1
    assert len(hook.STATE.flushed) == 1
    hook.flush_now()  # the idle timer / atexit path for the last request
    assert len(hook.STATE.flushed) == 2
    assert hook.flush_now() is None  # nothing left

    for path, exp, T, n in ((hook.STATE.flushed[0], exp1, 9, 5), (hook.STATE.flushed[1], exp2, 12, 3)):
        z = np.load(path, allow_pickle=False)
        assert int(z["prompt_len"]) == T and int(z["steps"]) == n
        np.testing.assert_array_equal(z["layers"], [3, 7])
        for mod, li in zip(layers, (3, 7)):
            got = z[f"logits_L{li}"]
            assert got.dtype == np.float16 and got.shape == (n, H, T)
            want = torch.stack(exp[mod]["steps"]).detach().numpy()
            np.testing.assert_allclose(got.astype(np.float32), want, rtol=2e-3, atol=2e-3)
    assert Path(hook.STATE.flushed[0]).name == "req_0.npz"
    assert Path(hook.STATE.flushed[1]).name == "req_1.npz"
    assert not hook.STATE.disabled


def test_layer_index_sources(hook_module):
    hook = hook_module

    class ByAttr:
        layer_idx = 11

    class ByQkv:
        class qkv_proj:
            prefix = "model.language_model.layers.15.self_attn.qkv_proj"

    class Nothing:
        pass

    assert hook.layer_index(ByAttr()) == 11
    assert hook.layer_index(ByQkv()) == 15
    a, b = Nothing(), Nothing()
    assert hook.layer_index(a) == hook.FULL_ATTENTION_LAYERS[0]
    assert hook.layer_index(b) == hook.FULL_ATTENTION_LAYERS[1]
    assert hook.layer_index(a) == hook.FULL_ATTENTION_LAYERS[0]  # cached per module
    assert "fallback-layer" in hook.STATE.logged


def test_mixed_and_padded_steps_are_ignored_and_logged_once(hook_module, capsys):
    hook = hook_module
    layers = [FakeQwen3NextAttention(3, 1)]
    hook.patch_class(FakeQwen3NextAttention)
    mod = layers[0]
    with torch.no_grad():
        mod(torch.arange(6), torch.randn(6, HID))  # prefill T=6
        # a chunked / mixed step: several rows not starting at 0 -> ignored
        mod(torch.tensor([6, 7, 8]), torch.randn(3, HID))
        mod(torch.tensor([6, 7, 8]), torch.randn(3, HID))
        # a padded decode batch: row 0 at the expected position, pad rows after it -> row 0 used
        mod(torch.tensor([6, 0]), torch.randn(2, HID))
        mod(torch.tensor([7]), torch.randn(1, HID))
        # decode at a stale position (e.g. a replayed step) -> ignored
        mod(torch.tensor([7]), torch.randn(1, HID))
    err = capsys.readouterr().err
    assert err.count("ignoring forward with 3 rows") == 1
    assert err.count("using row 0 only") == 1
    assert len(hook.STATE.cur.steps[3]) == 2
    path = hook.flush_now()
    z = np.load(path, allow_pickle=False)
    assert int(z["steps"]) == 2 and z["logits_L3"].shape == (2, H, 6)


def test_prefill_prefix_len_ignores_padded_tail(hook_module):
    hook = hook_module
    hook.patch_class(FakeQwen3NextAttention)
    mod = FakeQwen3NextAttention(3, 1)
    with torch.no_grad():
        mod(torch.tensor([0, 1, 2, 3, 0, 0]), torch.randn(6, HID))
    assert hook.STATE.cur.prompt_len == 4
    assert tuple(hook.STATE.cur.k[3].shape) == (4, KV, D)


def test_capture_error_disables_but_never_raises(hook_module, monkeypatch, capsys):
    hook = hook_module
    hook.patch_class(FakeQwen3NextAttention)
    mod = FakeQwen3NextAttention(7, 3)

    def boom(*a, **k):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(hook, "_handle", boom)
    with torch.no_grad():
        out = mod(torch.arange(5), torch.randn(5, HID))  # must still return the real output
        out2 = mod(torch.arange(5), torch.randn(5, HID))
    assert out.shape == (5, H * D) and out2.shape == (5, H * D)
    assert hook.STATE.disabled
    assert capsys.readouterr().err.count("CAPTURE ERROR") == 1


def test_junk_prefill_without_decode_is_flushed_with_zero_steps(hook_module):
    hook = hook_module
    hook.patch_class(FakeQwen3NextAttention)
    mod = FakeQwen3NextAttention(3, 1)
    with torch.no_grad():
        mod(torch.arange(50), torch.randn(50, HID))  # profiling-style run
        mod(torch.arange(7), torch.randn(7, HID))  # next request flushes it
    z = np.load(hook.STATE.flushed[0], allow_pickle=False)
    assert int(z["prompt_len"]) == 50 and int(z["steps"]) == 0
    assert z["logits_L3"].shape == (0, 0, 50)


def test_import_hook_patches_target_module_on_import(hook_module, tmp_path, monkeypatch):
    """The meta_path finder patches the class right after the target module executes."""
    hook = hook_module
    pkg = tmp_path / "fakevllm"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "qwen3_next.py").write_text(
        textwrap.dedent(
            """
            import torch
            from torch import nn

            class Qwen3NextAttention(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.num_heads, self.num_kv_heads, self.head_dim, self.scaling = 2, 1, 4, 0.5
                    self.attn = nn.Identity()
                    self.attn.layer_name = "model.layers.3.self_attn.attn"
                def forward(self, positions, hidden_states):
                    return self.attn(hidden_states)
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    hook.install("fakevllm.qwen3_next:Qwen3NextAttention")
    import importlib

    mod = importlib.import_module("fakevllm.qwen3_next")
    assert mod.Qwen3NextAttention in hook.STATE.patched_classes
    assert hasattr(mod.Qwen3NextAttention.forward, "_drama3_orig")
    m = mod.Qwen3NextAttention()
    with torch.no_grad():
        m(torch.arange(3), torch.zeros(3, 4))  # Identity has no q/k args -> hook ignores it, no raise
    assert not hook.STATE.disabled
    monkeypatch.delitem(__import__("sys").modules, "fakevllm.qwen3_next", raising=False)
    monkeypatch.delitem(__import__("sys").modules, "fakevllm", raising=False)


def test_pth_file_imports_the_hook():
    pth = Path(__file__).resolve().parents[1] / "hook" / "drama3_align.pth"
    assert pth.read_text().strip() == "import drama3_align_hook"
