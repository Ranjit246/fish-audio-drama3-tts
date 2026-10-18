"""Import the vendored nano-vllm-fusion package so it works on CPU.

Problems with a plain `import nanovllm`:
  * `nanovllm/__init__.py` imports the engine, which pulls transformers, tqdm,
    PIL and torchvision (image_processing.py) - heavy and irrelevant for Drama 3.
  * `nanovllm/layers/attention.py` imports `triton` and `flash_attn` at module
    level and defines a `@triton.jit` kernel.
  * `@torch.compile` decorators run at import time; inductor on macOS CPU is slow
    and fragile for tests.
  * `LinearBase` calls `torch.distributed.get_rank()` -> a process group must
    exist even for a single process.

`import_fusion()` fixes all of that: it puts `third_party/nano-vllm-fusion` on
sys.path, registers a lightweight `nanovllm` package object that skips the
package `__init__` (submodules import normally, and `nanovllm.LLM` /
`SamplingParams` are provided lazily), installs CPU stubs only when the real
GPU packages are missing, and applies the Drama 3 MoE patch.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import types
from pathlib import Path

import torch
import torch.distributed as dist

_PKG = "nanovllm"
_STATE: dict[str, object] = {}


def fusion_root() -> Path:
    env = os.environ.get("NANOVLLM_FUSION_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "third_party" / "nano-vllm-fusion"


def _register_light_package(root: Path) -> types.ModuleType:
    pkg_dir = root / _PKG
    pkg = types.ModuleType(_PKG)
    pkg.__path__ = [str(pkg_dir)]
    pkg.__file__ = str(pkg_dir / "__init__.py")
    pkg.__package__ = _PKG
    pkg.__nanovllm_drama3_light__ = True

    def __getattr__(name: str):  # PEP 562 lazy engine exports
        if name == "LLM":
            return importlib.import_module(f"{_PKG}.llm").LLM
        if name == "SamplingParams":
            return importlib.import_module(f"{_PKG}.sampling_params").SamplingParams
        if name == "LLMEngine":
            return importlib.import_module(f"{_PKG}.engine.llm_engine").LLMEngine
        raise AttributeError(name)

    pkg.__getattr__ = __getattr__
    sys.modules[_PKG] = pkg
    return pkg


def disable_torch_compile() -> None:
    """Make `@torch.compile` decorators evaluated later return the plain function."""
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    try:
        torch._dynamo.config.disable = True  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - very old torch
        pass


def import_fusion(cpu_fallback: bool | None = None, compile: bool | None = None) -> types.ModuleType:
    """Return the vendored `nanovllm` package (model submodules imported, patched).

    cpu_fallback: install triton/flash_attn stubs + torch store_kvcache. Default:
                  True when CUDA is unavailable.
    compile:      keep `@torch.compile`. Default: only when CUDA is available.
    Idempotent; subsequent calls return the cached package.
    """
    if "pkg" in _STATE:
        return _STATE["pkg"]  # type: ignore[return-value]

    root = fusion_root()
    if not (root / _PKG / "models" / "qwen3_5_moe.py").exists():
        raise ImportError(
            f"nano-vllm-fusion not found at {root}; run "
            "`git clone --depth 1 https://github.com/xiez-njust-xmu/nano-vllm-fusion third_party/nano-vllm-fusion` "
            "or set NANOVLLM_FUSION_ROOT"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    cuda = torch.cuda.is_available()
    if cpu_fallback is None:
        cpu_fallback = not cuda
    if compile is None:
        compile = cuda
    if not compile:
        disable_torch_compile()

    from nanovllm_drama3.backbone import cpu_fallback as cpu_mod

    if _PKG in sys.modules and not getattr(sys.modules[_PKG], "__nanovllm_drama3_light__", False):
        pkg = sys.modules[_PKG]  # a real import already happened (GPU box with deps)
    else:
        pkg = _register_light_package(root)

    stubs = {"triton": False, "flash_attn": False}
    if cpu_fallback:
        with cpu_mod.stubbed_imports() as ctx:
            attention = importlib.import_module(f"{_PKG}.layers.attention")
        stubs = ctx.installed
        if stubs["triton"]:
            attention.store_kvcache = cpu_mod.store_kvcache
    else:
        attention = importlib.import_module(f"{_PKG}.layers.attention")
    moe = importlib.import_module(f"{_PKG}.models.qwen3_5_moe")
    importlib.import_module(f"{_PKG}.utils.loader")
    importlib.import_module(f"{_PKG}.utils.context")

    from nanovllm_drama3.backbone import moe_patch

    moe_patch.install(moe)

    _STATE.update(pkg=pkg, root=root, cpu_fallback=cpu_fallback, stubs=stubs, compile=compile)
    return pkg


def fusion_state() -> dict[str, object]:
    return dict(_STATE)


def ensure_process_group(device: str | torch.device | None = None) -> bool:
    """Create a world-size-1 process group if none exists (fusion's TP layers need it).

    Returns True if this call initialised it. No-op (False) if already initialised,
    e.g. when fusion's ModelRunner did `dist.init_process_group` itself.
    """
    if dist.is_available() and dist.is_initialized():
        return False
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type == "cuda" and dist.is_nccl_available():
        backend = "nccl"
        init_method = "tcp://127.0.0.1:2333"
    else:
        backend = "gloo"
        init_method = "file://" + os.path.join(tempfile.mkdtemp(prefix="nanovllm_drama3_pg_"), "pg")
        # Without an explicit interface gloo resolves the hostname, which takes
        # ~35 s to time out on a Mac off the corporate DNS. Loopback is all a
        # world-size-1 group needs.
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group(backend, init_method=init_method, world_size=1, rank=0)
    return True
