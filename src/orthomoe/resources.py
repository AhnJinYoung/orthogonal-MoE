"""CPU/RAM workload constraints for memory-fragile pods.

The experiments run on temporary kubectl pods that get OOM-killed (exit code
137) when host RAM or CPU is oversubscribed. The most common triggers are:

1. Loading a 26B/35B checkpoint with ``device_map="auto"`` while host RAM is
   smaller than the checkpoint. ``low_cpu_mem_usage`` helps, but without a
   ``max_memory`` budget and a disk ``offload_folder`` the loader can still
   spike past the pod limit.
2. CPU thread oversubscription from BLAS/OpenMP/tokenizers spawning one thread
   per visible core (pods often advertise the whole node), each with its own
   working set.
3. ``datasets`` map/tokenization buffering large Arrow batches in RAM.

This module centralises the guards. Everything is driven by an optional
``resources:`` block in the YAML config so the same code path works on a laptop
and on a constrained pod.
"""
from __future__ import annotations

import gc
import os
import resource as _resource
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "RAYON_NUM_THREADS",  # tokenizers/safetensors rust threads
)


def _read_meminfo_gb() -> Dict[str, float]:
    """Return host memory figures in GiB parsed from /proc/meminfo."""
    info: Dict[str, float] = {}
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return info
    for line in meminfo.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].endswith(":"):
            try:
                info[parts[0][:-1]] = float(parts[1]) / (1024.0 * 1024.0)
            except ValueError:
                continue
    return info


def _cgroup_memory_limit_gb() -> Optional[float]:
    """Best-effort container memory limit (cgroup v2 then v1) in GiB."""
    candidates = (
        "/sys/fs/cgroup/memory.max",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
    )
    for path in candidates:
        p = Path(path)
        if not p.exists():
            continue
        raw = p.read_text().strip()
        if raw in {"max", ""}:
            continue
        try:
            limit = float(raw) / 1e9
        except ValueError:
            continue
        # cgroup v1 reports a huge sentinel value when unlimited.
        if limit > 0 and limit < 1e6:
            return limit
    return None


def process_rss_gb() -> float:
    """Resident set size of the current process in GiB.

    Prefers the live value from ``/proc/self/status`` (VmRSS, in KiB) and falls
    back to ``getrusage`` peak RSS. ``ru_maxrss`` is KiB on Linux but bytes on
    macOS, so the unit is chosen by platform rather than by magnitude.
    """
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                try:
                    return float(line.split()[1]) / (1024.0 * 1024.0)  # KiB -> GiB
                except (IndexError, ValueError):
                    break
    try:
        usage = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return 0.0
    divisor = 1024.0 ** 3 if sys.platform == "darwin" else 1024.0 ** 2  # bytes vs KiB
    return usage / divisor


def host_memory_summary() -> Dict[str, float]:
    """Snapshot of host/container memory in GiB for logging."""
    mem = _read_meminfo_gb()
    summary = {
        "rss_gb": round(process_rss_gb(), 2),
        "mem_total_gb": round(mem.get("MemTotal", 0.0), 2),
        "mem_available_gb": round(mem.get("MemAvailable", 0.0), 2),
    }
    cg = _cgroup_memory_limit_gb()
    if cg is not None:
        summary["cgroup_limit_gb"] = round(cg, 2)
    return summary


def log_memory(tag: str) -> Dict[str, float]:
    """Print and return a memory snapshot (host RAM + CUDA)."""
    summary = host_memory_summary()
    msg = (
        f"[mem:{tag}] rss={summary['rss_gb']:.1f}GiB "
        f"avail={summary.get('mem_available_gb', 0.0):.1f}GiB "
        f"total={summary.get('mem_total_gb', 0.0):.1f}GiB"
    )
    if "cgroup_limit_gb" in summary:
        msg += f" cgroup_limit={summary['cgroup_limit_gb']:.1f}GiB"
    if torch.cuda.is_available():
        msg += f" cuda_alloc={torch.cuda.memory_allocated() / 1e9:.1f}GiB"
    print(msg, flush=True)
    return summary


def effective_cpu_budget_gb(res_cfg: Mapping[str, Any]) -> Optional[float]:
    """Resolve the CPU RAM budget from config or the detected cgroup limit."""
    explicit = res_cfg.get("max_cpu_ram_gb")
    if explicit is not None:
        return float(explicit)
    cg = _cgroup_memory_limit_gb()
    if cg is not None:
        # Leave headroom so the guard trips before the kernel OOM-killer does.
        return round(cg * float(res_cfg.get("cgroup_headroom_fraction", 0.9)), 1)
    return None


def apply_resource_limits(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply thread caps and an optional soft address-space rlimit.

    Args:
        cfg: full experiment config. Reads the optional ``resources`` block.

    Returns:
        The resolved resource settings (also useful for logging).
    """
    res_cfg = dict(cfg.get("resources", {}) or {})

    # 1) Thread caps. Pods often expose every node core; without a cap each
    #    BLAS/OpenMP pool sizes to that and multiplies the resident footprint.
    num_threads = res_cfg.get("num_threads")
    if num_threads is None:
        # Conservative default: never grab more than 4 threads on a shared pod.
        detected = os.cpu_count() or 1
        num_threads = max(1, min(4, detected))
    num_threads = int(num_threads)
    for var in _THREAD_ENV_VARS:
        os.environ.setdefault(var, str(num_threads))
    try:
        torch.set_num_threads(num_threads)
    except Exception:
        pass
    # tokenizers parallelism is a frequent RAM/instability source on pods.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # 2) Optional soft RLIMIT_DATA guard. We deliberately avoid RLIMIT_AS:
    #    CUDA/torch reserve enormous virtual address space, so capping AS kills
    #    legitimate runs. RLIMIT_DATA caps the heap (brk/sbrk) and lets us fail
    #    with a Python MemoryError instead of a silent SIGKILL in many cases.
    budget_gb = effective_cpu_budget_gb(res_cfg)
    if budget_gb is not None and bool(res_cfg.get("set_rlimit_data", False)):
        try:
            soft_bytes = int(budget_gb * 1e9)
            _, hard = _resource.getrlimit(_resource.RLIMIT_DATA)
            new_hard = hard if hard != _resource.RLIM_INFINITY else soft_bytes
            _resource.setrlimit(_resource.RLIMIT_DATA, (soft_bytes, new_hard))
        except (ValueError, OSError) as exc:
            print(f"[resources] could not set RLIMIT_DATA: {exc}", flush=True)

    resolved = {
        "num_threads": num_threads,
        "max_cpu_ram_gb": budget_gb,
        "set_rlimit_data": bool(res_cfg.get("set_rlimit_data", False)),
    }
    print(f"[resources] {resolved}", flush=True)
    log_memory("startup")
    return resolved


def build_load_memory_kwargs(model_cfg: Mapping[str, Any], res_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Build ``from_pretrained`` kwargs that bound host RAM during loading.

    Produces a ``max_memory`` map (per-GPU + cpu) and a disk ``offload_folder``
    so the loader streams shards instead of materialising the whole checkpoint
    in host RAM. Only emitted when the config asks for it, to avoid changing
    behaviour for users who do not set a budget.
    """
    kwargs: Dict[str, Any] = {}

    cpu_gb = res_cfg.get("max_cpu_ram_gb")
    if cpu_gb is None:
        cpu_gb = effective_cpu_budget_gb(res_cfg)

    # We only emit a max_memory map when there is a CPU budget to enforce. When
    # we do, we MUST also list every GPU, otherwise accelerate concludes no GPU
    # is usable and places the whole model on CPU/disk.
    if cpu_gb is not None and torch.cuda.is_available():
        max_memory: Dict[Any, str] = {}
        gpu_gb = res_cfg.get("max_gpu_mem_gb")
        for idx in range(torch.cuda.device_count()):
            if gpu_gb is not None:
                max_memory[idx] = f"{int(gpu_gb)}GiB"
            else:
                total_gib = torch.cuda.get_device_properties(idx).total_memory / (1024.0 ** 3)
                # Leave a little headroom for activations/CUDA context.
                max_memory[idx] = f"{int(total_gib * 0.92)}GiB"
        # Reserve a slice of the CPU budget for everything that is not weights.
        weight_cpu = max(1.0, float(cpu_gb) * float(res_cfg.get("cpu_weight_fraction", 0.7)))
        max_memory["cpu"] = f"{weight_cpu:.0f}GiB"
        kwargs["max_memory"] = max_memory
    elif cpu_gb is not None:
        # CPU-only host: cap the CPU weight budget so loading still spills to disk.
        weight_cpu = max(1.0, float(cpu_gb) * float(res_cfg.get("cpu_weight_fraction", 0.7)))
        kwargs["max_memory"] = {"cpu": f"{weight_cpu:.0f}GiB"}

    offload_folder = res_cfg.get("offload_folder")
    if offload_folder is None and "max_memory" in kwargs:
        # Prefer the PVC offload dir exported by scripts/env.sh so it persists.
        offload_folder = os.environ.get("ORTHOMOE_OFFLOAD") or str(
            Path(os.environ.get("ORTHOMOE_PVC", ".")) / "offload"
        )
    if offload_folder:
        Path(offload_folder).mkdir(parents=True, exist_ok=True)
        kwargs["offload_folder"] = offload_folder
        if bool(res_cfg.get("offload_state_dict", True)):
            kwargs["offload_state_dict"] = True

    return kwargs


class MemoryGuard:
    """Trip a clean error before the kernel OOM-killer sends SIGKILL.

    Call :meth:`check` inside hot loops. When resident memory crosses the
    configured fraction of the budget it frees caches once; if still over, it
    raises ``MemoryError`` so the run dies with a traceback we can read instead
    of an opaque exit code 137.
    """

    def __init__(self, budget_gb: Optional[float], *, warn_fraction: float = 0.85, abort_fraction: float = 0.95):
        self.budget_gb = budget_gb
        self.warn_fraction = warn_fraction
        self.abort_fraction = abort_fraction
        self._warned = False

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "MemoryGuard":
        res_cfg = dict(cfg.get("resources", {}) or {})
        return cls(
            effective_cpu_budget_gb(res_cfg),
            warn_fraction=float(res_cfg.get("warn_fraction", 0.85)),
            abort_fraction=float(res_cfg.get("abort_fraction", 0.95)),
        )

    def check(self, tag: str = "") -> None:
        if not self.budget_gb:
            return
        rss = process_rss_gb()
        # Prefer the host's own availability when it is the tighter bound.
        avail = _read_meminfo_gb().get("MemAvailable")
        over_rss = rss >= self.budget_gb * self.abort_fraction
        over_avail = avail is not None and avail <= self.budget_gb * (1.0 - self.abort_fraction)
        if over_rss or over_avail:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            rss = process_rss_gb()
            if rss >= self.budget_gb * self.abort_fraction:
                raise MemoryError(
                    f"RAM guard tripped at {tag or 'check'}: rss={rss:.1f}GiB >= "
                    f"{self.abort_fraction:.0%} of budget {self.budget_gb:.1f}GiB. "
                    "Lower data.max_eval_samples / eval.max_batches / block_size, "
                    "or raise resources.max_cpu_ram_gb."
                )
        elif not self._warned and rss >= self.budget_gb * self.warn_fraction:
            self._warned = True
            print(
                f"[mem-guard] WARNING {tag}: rss={rss:.1f}GiB past "
                f"{self.warn_fraction:.0%} of {self.budget_gb:.1f}GiB budget.",
                flush=True,
            )
