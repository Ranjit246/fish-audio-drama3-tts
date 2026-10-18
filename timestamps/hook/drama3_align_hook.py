"""Phase 0 attention capture for Drama 3 on vLLM-Omni (head sweep only).

Installed as a .pth-loaded module (hook/install.sh) so it is imported in EVERY
Python process of the vllm venv, including the stage-0 worker where the model
runs. It is INERT unless ``DRAMA3_ALIGN_SWEEP=1``. When armed it installs an
import hook that patches ``Qwen3NextAttention.forward`` the moment
``vllm.model_executor.models.qwen3_next`` is imported (vLLM is never imported
by this module itself, so the .pth costs nothing in unrelated processes).

What it records (eager mode, concurrency 1):

* Prefill: a forward whose ``positions[0] == 0`` with more than one row is a
  new request. The previous request's capture is flushed to
  ``$DRAMA3_ALIGN_OUT/req_<counter>.npz`` and this layer's post-norm, post-RoPE
  ``k`` for ALL prompt rows is stashed as ``[T, num_kv_heads, head_dim]`` fp16.
* Decode: a forward with one row at the position this layer expects next
  (``prompt_len + steps_so_far``) yields raw logits
  ``q[h] . K_prompt[h // (num_heads // num_kv_heads)]^T * scaling`` for all
  heads -> ``[num_heads, T]`` fp16, appended to this layer's step list.
* Anything else (mixed / chunked steps with ``positions[0] != 0`` and several
  rows, padded or out-of-order decode rows) is ignored and logged once.

Everything stays on the GPU until flush: one device->host copy per layer per
request. Flush happens at the next request's prefill, after ``DRAMA3_ALIGN_IDLE_S``
seconds without any forward (default 5 s; a background thread), and at
interpreter exit.

Where q / k come from. ``Qwen3NextAttention.forward`` does
``qkv, _ = self.qkv_proj(hidden_states); q, k, v, gate = self._project_qkv_gate(qkv, positions);
attn_output = self.attn(q, k, v)``. ``_project_qkv_gate`` returns q as
``[num_tokens, num_heads * head_dim]`` and k as ``[num_tokens, num_kv_heads * head_dim]``,
both after q_norm / k_norm and RoPE (the fused Triton path returns the same
layout). We do not duplicate that body: the patched forward stashes
``positions`` and a one-time forward PRE-hook on ``self.attn`` (vLLM's
``Attention`` nn.Module) receives exactly the ``(q, k, v)`` that flash
attention receives. Shapes are normalised with ``reshape(rows, heads, head_dim)``
so a ``[rows, heads, head_dim]`` layout works too.

Environment
-----------
DRAMA3_ALIGN_SWEEP=1          arm the hook (anything else: inert)
DRAMA3_ALIGN_OUT=<dir>        where req_<n>.npz go (default ./align_capture)
DRAMA3_ALIGN_IDLE_S=5         idle seconds before the current capture is flushed
DRAMA3_ALIGN_SAVE_K=0         also store the stashed prompt keys (big; debugging)
DRAMA3_ALIGN_LAYERS=3,7,...   model layer indices assumed for call-order fallback
DRAMA3_ALIGN_TARGET=<module>:<class>  override of the patched class (tests)
DRAMA3_ALIGN_VERBOSE=1        log every prefill / flush

Never raises inside the patched forward: any error disables the capture for
the rest of the process and is logged once.
"""

from __future__ import annotations

import atexit
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import re
import sys
import threading
import time
import traceback

ENABLED = os.environ.get("DRAMA3_ALIGN_SWEEP") == "1"
OUT_DIR = os.environ.get("DRAMA3_ALIGN_OUT", os.path.abspath("align_capture"))
IDLE_S = float(os.environ.get("DRAMA3_ALIGN_IDLE_S", "5"))
SAVE_K = os.environ.get("DRAMA3_ALIGN_SAVE_K") == "1"
VERBOSE = os.environ.get("DRAMA3_ALIGN_VERBOSE") == "1"
DEFAULT_TARGET = "vllm.model_executor.models.qwen3_next:Qwen3NextAttention"
TARGET = os.environ.get("DRAMA3_ALIGN_TARGET", DEFAULT_TARGET)
FULL_ATTENTION_LAYERS = tuple(
    int(v) for v in os.environ.get("DRAMA3_ALIGN_LAYERS", ",".join(str(i) for i in range(3, 40, 4))).split(",") if v
)

_LAYER_RE = re.compile(r"layers\.(\d+)(?:\.|$)")


def _log(msg: str) -> None:
    print(f"[drama3-align pid={os.getpid()}] {msg}", file=sys.stderr, flush=True)


class _Request:
    """Capture of one request, GPU-resident until flush."""

    __slots__ = ("prompt_len", "k", "steps", "created")

    def __init__(self, prompt_len: int):
        self.prompt_len = prompt_len
        self.k: dict[int, object] = {}  # layer -> [T, kv_heads, head_dim] fp16
        self.steps: dict[int, list] = {}  # layer -> [[heads, T] fp16, ...]
        self.created = time.time()


class _State:
    def __init__(self):
        self.lock = threading.RLock()
        self.out_dir = OUT_DIR
        self.idle_s = IDLE_S
        self.save_k = SAVE_K
        self.counter = 0
        self.cur: _Request | None = None
        self.disabled = False
        self.logged: set[str] = set()
        self.layer_of: dict[int, int] = {}  # id(module) -> model layer index
        self.next_fallback = 0
        self.last_seen = 0.0
        self.flushed: list[str] = []
        self.patched_classes: set[type] = set()
        self.idle_thread: threading.Thread | None = None
        self.stop_idle = threading.Event()

    def log_once(self, key: str, msg: str) -> None:
        if key not in self.logged:
            self.logged.add(key)
            _log(msg)


STATE = _State()


# --------------------------------------------------------------------------
# layer index of a Qwen3NextAttention module
# --------------------------------------------------------------------------

def _parse_layer(*names) -> int | None:
    for n in names:
        if isinstance(n, str):
            m = _LAYER_RE.search(n)
            if m:
                return int(m.group(1))
    return None


def layer_index(module) -> int:
    """Model layer index (0..39) of an attention module.

    Tries, in order: ``module.attn.layer_name`` (vLLM's Attention stores its
    prefix, e.g. ``model.language_model.layers.3.self_attn.attn``),
    ``module.qkv_proj.prefix``, ``module.prefix``, ``module.layer_idx`` /
    ``module._align_layer_idx``; else the order of first appearance mapped onto
    ``FULL_ATTENTION_LAYERS`` (logged, since it assumes layers are called in order).
    """
    idx = getattr(module, "_drama3_layer", None)
    if idx is None:
        idx = STATE.layer_of.get(id(module))
    if idx is not None:
        return idx
    for attr in ("_align_layer_idx", "layer_idx"):
        v = getattr(module, attr, None)
        if isinstance(v, int):
            idx = v
            break
    if idx is None:
        attn = getattr(module, "attn", None)
        idx = _parse_layer(
            getattr(attn, "layer_name", None),
            getattr(attn, "prefix", None),
            getattr(getattr(module, "qkv_proj", None), "prefix", None),
            getattr(module, "prefix", None),
        )
    if idx is None:
        n = STATE.next_fallback
        STATE.next_fallback += 1
        idx = FULL_ATTENTION_LAYERS[n % len(FULL_ATTENTION_LAYERS)]
        STATE.log_once(
            "fallback-layer",
            f"could not read a layer index from the module; assuming call order -> {FULL_ATTENTION_LAYERS}",
        )
    try:
        module._drama3_layer = idx  # cached on the module itself (ids can be reused after GC)
    except Exception:
        STATE.layer_of[id(module)] = idx
    return idx


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------

def _prefix_len(positions) -> int:
    """Rows at the front with positions 0, 1, 2, ... (robust to padded tails)."""
    import torch

    rows = positions.shape[0]
    if int(positions[rows - 1]) == rows - 1:
        return rows
    ar = torch.arange(rows, device=positions.device, dtype=positions.dtype)
    ok = (positions == ar).to(torch.int64).cumprod(0)
    return int(ok.sum())


def _handle(module, positions, q, k) -> None:
    import torch

    st = STATE
    if positions.ndim == 2:  # mrope-style [3, T]: all rows equal for text-only
        positions = positions[0]
    rows = int(positions.shape[0])
    if rows == 0:
        return
    layer = layer_index(module)
    heads = int(module.num_heads)
    kv_heads = int(module.num_kv_heads)
    head_dim = int(module.head_dim)
    scaling = float(module.scaling)
    p0 = int(positions[0])

    with st.lock:
        st.last_seen = time.monotonic()
        cur = st.cur
        if rows > 1 and p0 == 0:
            # a new request's prefill (or a warmup / profiling run that looks like one)
            if cur is None or layer in cur.k:
                _flush_locked()
                T = _prefix_len(positions)
                cur = st.cur = _Request(T)
                if VERBOSE:
                    _log(f"prefill: {rows} rows, prompt_len={T}")
            T = cur.prompt_len
            kk = k[:T].reshape(T, kv_heads, head_dim)
            cur.k[layer] = kk.to(torch.float16).clone()
            return
        if cur is None:
            st.log_once("decode-without-prefill", f"decode-like forward ({rows} rows, pos {p0}) before any prefill; ignored")
            return
        kk = cur.k.get(layer)
        steps = cur.steps.setdefault(layer, [])
        expected = cur.prompt_len + len(steps)
        if kk is None or p0 != expected:
            st.log_once(
                "unexpected-step",
                f"ignoring forward with {rows} rows at position {p0} (layer {layer} expected {expected}); "
                "mixed/chunked step or padded batch -- the sweep needs concurrency 1 and one prefill chunk",
            )
            return
        if rows > 1:
            st.log_once("padded-decode", f"decode forward has {rows} rows; using row 0 only")
        q0 = q[0].reshape(heads, head_dim).to(torch.float32)  # [H, d]
        rep = heads // kv_heads
        kexp = kk.to(torch.float32)  # [T, kv, d]
        if rep > 1:
            kexp = kexp.repeat_interleave(rep, dim=1)  # [T, H, d]; head h -> kv head h // rep
        logits = torch.einsum("hd,thd->ht", q0, kexp) * scaling  # [H, T]
        steps.append(logits.to(torch.float16))


def _flush_locked() -> str | None:
    """Write the current request (if it has anything) and reset. Caller holds the lock."""
    import numpy as np
    import torch

    st = STATE
    cur = st.cur
    st.cur = None
    if cur is None or not cur.k:
        return None
    layers = sorted(cur.k)
    n_steps = min((len(cur.steps.get(l, [])) for l in layers), default=0)
    os.makedirs(st.out_dir, exist_ok=True)
    name = f"req_{st.counter}.npz"
    path = os.path.join(st.out_dir, name)
    if os.path.exists(path):
        path = os.path.join(st.out_dir, f"req_{st.counter}_{os.getpid()}.npz")
    st.counter += 1
    arrays = {
        "prompt_len": np.int64(cur.prompt_len),
        "layers": np.asarray(layers, dtype=np.int64),
        "steps": np.int64(n_steps),
        "counter": np.int64(st.counter - 1),
        "pid": np.int64(os.getpid()),
        "created": np.float64(cur.created),
    }
    if n_steps > 0:
        # [L, steps, heads, T] stacked on the device -> ONE device->host copy per request
        big = torch.stack([torch.stack(cur.steps[l][:n_steps], 0) for l in layers], 0).cpu().numpy()
        for i, l in enumerate(layers):
            arrays[f"logits_L{l}"] = big[i]
    else:
        for l in layers:
            arrays[f"logits_L{l}"] = np.zeros((0, 0, cur.prompt_len), dtype=np.float16)
    if st.save_k:
        kbig = torch.stack([cur.k[l] for l in layers], 0).cpu().numpy()
        for i, l in enumerate(layers):
            arrays[f"k_L{l}"] = kbig[i]
    np.savez(path, **arrays)
    st.flushed.append(path)
    _log(f"flushed {path}: prompt_len={cur.prompt_len} steps={n_steps} layers={layers}")
    del cur
    return path


def flush_now() -> str | None:
    """Flush the in-progress capture (idle timer / atexit / tests)."""
    with STATE.lock:
        try:
            return _flush_locked()
        except Exception as e:  # pragma: no cover - defensive
            STATE.log_once("flush-error", f"flush failed: {type(e).__name__}: {e}")
            return None


def _idle_loop() -> None:
    while not STATE.stop_idle.wait(0.5):
        with STATE.lock:
            if STATE.cur is not None and STATE.last_seen and time.monotonic() - STATE.last_seen > STATE.idle_s:
                try:
                    _flush_locked()
                except Exception as e:  # pragma: no cover - defensive
                    STATE.log_once("idle-flush-error", f"idle flush failed: {type(e).__name__}: {e}")


def _ensure_idle_thread() -> None:
    if STATE.idle_thread is None and STATE.idle_s > 0:
        t = threading.Thread(target=_idle_loop, name="drama3-align-idle", daemon=True)
        STATE.idle_thread = t
        t.start()


# --------------------------------------------------------------------------
# patching
# --------------------------------------------------------------------------

def _attn_pre_hook(attn_module, args):
    """Forward pre-hook on ``Qwen3NextAttention.attn``: args are (q, k, v)."""
    if STATE.disabled:
        return None
    owner = getattr(attn_module, "_drama3_owner", None)
    positions = getattr(attn_module, "_drama3_positions", None)
    if owner is None or positions is None or len(args) < 2:
        return None
    try:
        _handle(owner, positions, args[0], args[1])
    except Exception as e:
        STATE.disabled = True
        _log(f"CAPTURE ERROR, disabling: {type(e).__name__}: {e}\n{traceback.format_exc()}")
    finally:
        attn_module._drama3_positions = None
    return None


def patch_class(cls) -> bool:
    """Monkeypatch ``cls.forward(self, positions, hidden_states)``. Idempotent."""
    if cls in STATE.patched_classes:
        return False
    orig_forward = cls.forward

    def forward(self, positions, hidden_states, *args, **kwargs):
        if not STATE.disabled:
            try:
                attn = getattr(self, "attn", None)
                if attn is not None and hasattr(attn, "register_forward_pre_hook"):
                    if not getattr(attn, "_drama3_hooked", False):
                        attn.register_forward_pre_hook(_attn_pre_hook)
                        attn._drama3_hooked = True
                    attn._drama3_owner = self
                    attn._drama3_positions = positions
                    _ensure_idle_thread()
                else:
                    STATE.log_once("no-attn", "module has no .attn nn.Module to hook; capture disabled")
                    STATE.disabled = True
            except Exception as e:
                STATE.disabled = True
                _log(f"HOOK SETUP ERROR, disabling: {type(e).__name__}: {e}")
        return orig_forward(self, positions, hidden_states, *args, **kwargs)

    forward._drama3_orig = orig_forward
    cls.forward = forward
    STATE.patched_classes.add(cls)
    _log(f"patched {cls.__module__}.{cls.__qualname__}.forward; out={STATE.out_dir}")
    return True


def unpatch_class(cls) -> None:
    f = getattr(cls, "forward", None)
    orig = getattr(f, "_drama3_orig", None)
    if orig is not None:
        cls.forward = orig
    STATE.patched_classes.discard(cls)


def _patch_module(module, class_name: str) -> None:
    cls = getattr(module, class_name, None)
    if cls is None:
        _log(f"{module.__name__} has no class {class_name}; nothing patched")
        return
    patch_class(cls)


class _PostImportFinder(importlib.abc.MetaPathFinder):
    """Runs ``_patch_module`` right after the target module is first executed."""

    def __init__(self, module_name: str, class_name: str):
        self.module_name = module_name
        self.class_name = class_name
        self._busy = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.module_name or self._busy:
            return None
        self._busy = True
        try:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None:
                for finder in sys.meta_path:
                    if finder is self or not hasattr(finder, "find_spec"):
                        continue
                    spec = finder.find_spec(fullname, path, target)
                    if spec is not None:
                        break
        finally:
            self._busy = False
        if spec is None or spec.loader is None:
            return None
        spec.loader = _WrappedLoader(spec.loader, self.class_name)
        return spec


class _WrappedLoader(importlib.abc.Loader):
    def __init__(self, inner, class_name: str):
        self._inner = inner
        self._class_name = class_name

    def create_module(self, spec):
        create = getattr(self._inner, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        self._inner.exec_module(module)
        try:
            _patch_module(module, self._class_name)
        except Exception as e:
            _log(f"post-import patch failed: {type(e).__name__}: {e}")

    def __getattr__(self, name):  # delegate get_source / is_package / ...
        return getattr(self._inner, name)


def install(target: str = TARGET) -> None:
    """Arm the import hook for ``module:Class`` (patch immediately if already imported)."""
    module_name, _, class_name = target.partition(":")
    class_name = class_name or "Qwen3NextAttention"
    if module_name in sys.modules:
        _patch_module(sys.modules[module_name], class_name)
        return
    if not any(isinstance(f, _PostImportFinder) and f.module_name == module_name for f in sys.meta_path):
        sys.meta_path.insert(0, _PostImportFinder(module_name, class_name))


def _atexit() -> None:
    STATE.stop_idle.set()
    if STATE.cur is None:  # nothing captured: do not import numpy/torch during shutdown
        return
    try:
        flush_now()
    except Exception:
        pass


STATUS = "inert (set DRAMA3_ALIGN_SWEEP=1)"
if ENABLED:
    try:
        install()
        atexit.register(_atexit)
        STATUS = f"armed: target={TARGET} out={OUT_DIR} idle={IDLE_S}s"
        if VERBOSE:
            _log(STATUS)
    except Exception as e:  # pragma: no cover
        STATUS = f"failed to arm: {type(e).__name__}: {e}"
        _log(STATUS)
