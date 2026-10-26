"""PyTorch device probe + light benchmark.

Usage:
  python pytorch_bench.py
  python pytorch_bench.py --bench bandwidth,compile,export
  python pytorch_bench.py --device cuda --compile-mode default,reduce-overhead
  python pytorch_bench.py --sequential --dtype float16 --json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from typing import Any, Callable

try:
    import torch
    import torch.nn as nn
except ImportError:
    sys.stderr.write("PyTorch is not installed. Try: pip install torch\n")
    sys.exit(1)


# ---------------------------------------------------------------------------
# ANSI color (no extra deps)
# ---------------------------------------------------------------------------

class Color:
    reset = bold = dim = red = green = yellow = cyan = magenta = ""

    @classmethod
    def enable(cls, on: bool) -> None:
        if not on:
            return
        cls.reset = "\033[0m"
        cls.bold = "\033[1m"
        cls.dim = "\033[2m"
        cls.red = "\033[31m"
        cls.green = "\033[32m"
        cls.yellow = "\033[33m"
        cls.cyan = "\033[36m"
        cls.magenta = "\033[35m"


def c(style: str, text: str) -> str:
    s = getattr(Color, style, "")
    return f"{s}{text}{Color.reset}" if s else text


def enable_windows_ansi() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


PRINT_LOCK = threading.Lock()
COMPILE_LOCK = threading.Lock()  # dynamo/inductor is not happy compiling on two threads


def log(msg: str = "") -> None:
    with PRINT_LOCK:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------

def fmt_bytes(n: int | None) -> str:
    if n is None:
        return "n/a"
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{n} B"


def cpu_name() -> str:
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        elif system == "Windows":
            return platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or "Unknown CPU"
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def ram_bytes() -> int | None:
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        elif system == "Darwin":
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        elif system == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullTotalPhys)
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Devices
# ---------------------------------------------------------------------------

def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize(device)


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.empty_cache()


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple MPS (unified memory)"
    if device.type == "xpu" and hasattr(torch, "xpu"):
        try:
            return torch.xpu.get_device_name(device)
        except Exception:
            return "Intel XPU"
    if device.type == "cpu":
        return cpu_name()
    return str(device)


def device_vram(device: torch.device) -> int | None:
    if device.type == "cuda":
        return int(torch.cuda.get_device_properties(device).total_memory)
    if device.type == "xpu" and hasattr(torch, "xpu"):
        try:
            return int(torch.xpu.get_device_properties(device).total_memory)
        except Exception:
            return None
    if device.type == "mps":
        return ram_bytes()
    return None


def extra_device_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        info["capability"] = f"{p.major}.{p.minor}"
        info["multi_processor_count"] = getattr(p, "multi_processor_count", None)
    return info


def discover_devices() -> list[torch.device]:
    found: list[torch.device] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            found.append(torch.device(f"cuda:{i}"))
    if bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available():
        found.append(torch.device("mps"))
    if hasattr(torch, "xpu"):
        try:
            if torch.xpu.is_available():
                for i in range(torch.xpu.device_count()):
                    found.append(torch.device(f"xpu:{i}"))
        except Exception:
            pass
    found.append(torch.device("cpu"))
    return found


def filter_devices(all_devs: list[torch.device], spec: str) -> list[torch.device]:
    spec = spec.lower().strip()
    if spec in ("all", ""):
        return all_devs
    if spec == "auto":
        acc = [d for d in all_devs if d.type != "cpu"]
        return acc[:1] if acc else all_devs[-1:]

    wanted: list[torch.device] = []
    for d in all_devs:
        key = str(d)
        if spec in ("cuda", "rocm", "gpu") and d.type == "cuda":
            wanted.append(d)
        elif spec == "mps" and d.type == "mps":
            wanted.append(d)
        elif spec == "xpu" and d.type == "xpu":
            wanted.append(d)
        elif spec == "cpu" and d.type == "cpu":
            wanted.append(d)
        elif spec in (key, f"{d.type}:{d.index}"):
            wanted.append(d)
    return wanted


def backend_label() -> str:
    if torch.cuda.is_available() and getattr(torch.version, "hip", None):
        return f"ROCm/HIP {torch.version.hip}"
    if torch.cuda.is_available() and torch.version.cuda:
        return f"CUDA {torch.version.cuda}"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "MPS"
    if hasattr(torch, "xpu"):
        try:
            if torch.xpu.is_available():
                return "XPU"
        except Exception:
            pass
    return "CPU"


@contextmanager
def on_device(device: torch.device):
    if device.type == "cuda":
        with torch.cuda.device(device):
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream):
                yield
                stream.synchronize()
        return
    if device.type == "xpu" and hasattr(torch, "xpu"):
        cm = torch.xpu.device(device) if hasattr(torch.xpu, "device") else nullcontext()
        with cm:
            yield
        return
    yield


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def timed_runs(
    fn: Callable[[], Any],
    device: torch.device,
    warmup: int = 1,
    runs: int = 3,
) -> tuple[float, float, list[float]]:
    for _ in range(warmup):
        fn()
        sync(device)

    times: list[float] = []
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(runs):
            start.record()
            fn()
            end.record()
            sync(device)
            times.append(start.elapsed_time(end) / 1000.0)
    else:
        for _ in range(runs):
            sync(device)
            t0 = time.perf_counter()
            fn()
            sync(device)
            times.append(time.perf_counter() - t0)
    return min(times), sum(times) / len(times), times


def tflops(flops: float, seconds: float) -> float:
    return 0.0 if seconds <= 0 else flops / seconds / 1e12


def gbs(nbytes: float, seconds: float) -> float:
    return 0.0 if seconds <= 0 else nbytes / seconds / 1e9


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = name.lower()
    if key not in mapping:
        raise argparse.ArgumentTypeError(f"unsupported dtype: {name}")
    return mapping[key]


def dtype_supported(device: torch.device, dtype: torch.dtype) -> bool:
    if dtype == torch.float32:
        return True
    if device.type == "cpu":
        return dtype == torch.bfloat16
    if device.type == "mps":
        return dtype in (torch.float16, torch.bfloat16)
    return True


# ---------------------------------------------------------------------------
# Model + FLOP estimates
# ---------------------------------------------------------------------------

class TinyNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def conv_flops(batch: int, cin: int, cout: int, h: int, w: int, k: int) -> int:
    return 2 * batch * cout * h * w * (k * k * cin)


def tiny_forward_flops(batch: int, image: int) -> int:
    h = w = image
    return (
        conv_flops(batch, 3, 32, h, w, 3)
        + conv_flops(batch, 32, 64, h, w, 3)
        + conv_flops(batch, 64, 64, h, w, 3)
        + 2 * batch * 64 * 10
    )


def new_eval_model(device: torch.device, dtype: torch.dtype) -> TinyNet:
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return TinyNet().to(device=device, dtype=dtype).eval()


def run_steps(model: nn.Module, x: torch.Tensor, steps: int) -> torch.Tensor:
    with torch.inference_mode():
        y = x
        for _ in range(steps):
            y = model(x)
        if not torch.isfinite(y.float().sum()):
            raise RuntimeError("infer produced non-finite values")
        return y


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_matmul(device: torch.device, dtype: torch.dtype, n: int, inner: int) -> dict[str, Any]:
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    out = torch.empty(n, n, device=device, dtype=dtype)

    def run() -> None:
        nonlocal out
        with torch.inference_mode():
            for _ in range(inner):
                out = a @ b

    flops = inner * (2 * n * n * n)
    fastest, avg, times = timed_runs(run, device)
    if not torch.isfinite(out.float().sum()):
        raise RuntimeError("matmul produced non-finite values")
    return {
        "bench": "matmul",
        "detail": f"{n}x{n} gemm x{inner}",
        "flops": flops,
        "fastest_s": fastest,
        "average_s": avg,
        "times_s": times,
        "tflops_best": tflops(flops, fastest),
        "tflops_avg": tflops(flops, avg),
    }


def bench_train(
    device: torch.device, dtype: torch.dtype, batch: int, image: int, steps: int
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = TinyNet().to(device=device, dtype=dtype)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(batch, 3, image, image, device=device, dtype=dtype)
    y = torch.randint(0, 10, (batch,), device=device)

    def run() -> None:
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

    flops = steps * 3 * tiny_forward_flops(batch, image)
    fastest, avg, times = timed_runs(run, device)
    return {
        "bench": "train",
        "detail": f"TinyNet {batch}x3x{image}x{image}, {steps} steps",
        "flops": flops,
        "flops_note": "estimated as 3x forward (fwd+bwd)",
        "fastest_s": fastest,
        "average_s": avg,
        "times_s": times,
        "tflops_best": tflops(flops, fastest),
        "tflops_avg": tflops(flops, avg),
    }


def bench_infer(
    device: torch.device, dtype: torch.dtype, batch: int, image: int, steps: int
) -> dict[str, Any]:
    model = new_eval_model(device, dtype)
    x = torch.randn(batch, 3, image, image, device=device, dtype=dtype)
    fastest, avg, times = timed_runs(lambda: run_steps(model, x, steps), device)
    flops = steps * tiny_forward_flops(batch, image)
    return {
        "bench": "infer",
        "detail": f"TinyNet {batch}x3x{image}x{image}, {steps} fwd",
        "flops": flops,
        "fastest_s": fastest,
        "average_s": avg,
        "times_s": times,
        "tflops_best": tflops(flops, fastest),
        "tflops_avg": tflops(flops, avg),
    }


def _host_tensor(n: int, device: torch.device) -> torch.Tensor:
    pin = device.type == "cuda"
    try:
        return torch.empty(n, dtype=torch.float32, pin_memory=pin)
    except Exception:
        return torch.empty(n, dtype=torch.float32)


def bench_bandwidth(device: torch.device, size_mb: int, inner: int) -> dict[str, Any]:
    nbytes = max(1, size_mb) * 1024 * 1024
    n = nbytes // 4
    nbytes = n * 4
    non_blocking = device.type in ("cuda", "xpu")

    src = torch.empty(n, device=device, dtype=torch.float32)
    dst = torch.empty_like(src)
    src.fill_(1)
    copies = {"d2d": (src, dst)}
    if device.type != "cpu":
        host = _host_tensor(n, device)
        host.fill_(1)
        copies["h2d"] = (host, dst)
        copies["d2h"] = (src, host)

    metrics: dict[str, Any] = {}
    for name, (a, b) in copies.items():
        def run(a=a, b=b) -> None:
            for _ in range(inner):
                b.copy_(a, non_blocking=non_blocking)

        fastest, avg, times = timed_runs(run, device)
        moved = nbytes * inner
        metrics[name] = {
            "fastest_s": fastest,
            "average_s": avg,
            "times_s": times,
            "gbs_best": gbs(moved, fastest),
            "gbs_avg": gbs(moved, avg),
        }

    return {
        "bench": "bandwidth",
        "detail": f"{size_mb} MiB x{inner} copies, float32",
        "bytes": nbytes,
        "inner": inner,
        "copies": metrics,
        "fastest_s": metrics["d2d"]["fastest_s"],
        "average_s": metrics["d2d"]["average_s"],
        "gbs_best": metrics["d2d"]["gbs_best"],
        "gbs_avg": metrics["d2d"]["gbs_avg"],
    }


def dynamo_reset() -> None:
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "reset"):
        dynamo.reset()


def bench_compile(
    device: torch.device,
    dtype: torch.dtype,
    batch: int,
    image: int,
    steps: int,
    mode: str,
) -> dict[str, Any]:
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile requires PyTorch 2.0+")

    example = torch.randn(batch, 3, image, image, device=device, dtype=dtype)
    eager = new_eval_model(device, dtype)
    eager_fastest, eager_avg, eager_times = timed_runs(
        lambda: run_steps(eager, example, steps), device
    )

    model = new_eval_model(device, dtype)
    with COMPILE_LOCK:
        dynamo_reset()
        compiled = torch.compile(model, mode=mode)
        t0 = time.perf_counter()
        run_steps(compiled, example, 1)
        sync(device)
        compile_s = time.perf_counter() - t0

    # extra warmup: reduce-overhead may capture CUDA graphs after the first run
    fastest, avg, times = timed_runs(
        lambda: run_steps(compiled, example, steps), device, warmup=2, runs=3
    )
    flops = steps * tiny_forward_flops(batch, image)
    speedup = (eager_fastest / fastest) if fastest > 0 else 0.0
    return {
        "bench": "compile",
        "mode": mode,
        "detail": f"torch.compile({mode}) TinyNet {batch}x3x{image}x{image}, {steps} fwd",
        "compile_s": compile_s,
        "flops": flops,
        "fastest_s": fastest,
        "average_s": avg,
        "times_s": times,
        "tflops_best": tflops(flops, fastest),
        "tflops_avg": tflops(flops, avg),
        "eager_fastest_s": eager_fastest,
        "eager_average_s": eager_avg,
        "eager_times_s": eager_times,
        "eager_tflops_best": tflops(flops, eager_fastest),
        "speedup_best": speedup,
    }


def bench_export(
    device: torch.device,
    dtype: torch.dtype,
    batch: int,
    image: int,
    steps: int,
) -> dict[str, Any]:
    try:
        from torch.export import export as torch_export
    except ImportError as e:
        raise RuntimeError("torch.export requires PyTorch 2.1+") from e

    example = torch.randn(batch, 3, image, image, device=device, dtype=dtype)
    eager = new_eval_model(device, dtype)
    eager_fastest, eager_avg, eager_times = timed_runs(
        lambda: run_steps(eager, example, steps), device
    )

    model = new_eval_model(device, dtype)
    with COMPILE_LOCK:
        t0 = time.perf_counter()
        try:
            exported = torch_export(model, (example,))
        except Exception:
            exported = torch_export(model, (example,), strict=False)
        compile_s = time.perf_counter() - t0
        run_mod = exported.module()
        run_steps(run_mod, example, 1)
        sync(device)

    fastest, avg, times = timed_runs(
        lambda: run_steps(run_mod, example, steps), device, warmup=2, runs=3
    )
    flops = steps * tiny_forward_flops(batch, image)
    speedup = (eager_fastest / fastest) if fastest > 0 else 0.0
    return {
        "bench": "export",
        "mode": "export",
        "detail": f"torch.export TinyNet {batch}x3x{image}x{image}, {steps} fwd",
        "compile_s": compile_s,
        "flops": flops,
        "fastest_s": fastest,
        "average_s": avg,
        "times_s": times,
        "tflops_best": tflops(flops, fastest),
        "tflops_avg": tflops(flops, avg),
        "eager_fastest_s": eager_fastest,
        "eager_average_s": eager_avg,
        "eager_times_s": eager_times,
        "eager_tflops_best": tflops(flops, eager_fastest),
        "speedup_best": speedup,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def collect_system() -> dict[str, Any]:
    cudnn = None
    try:
        if torch.backends.cudnn.is_available():
            cudnn = torch.backends.cudnn.version()
    except Exception:
        pass
    return {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu": cpu_name(),
        "cpu_cores": os.cpu_count(),
        "ram": ram_bytes(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "hip": getattr(torch.version, "hip", None),
        "cudnn": cudnn,
        "backend": backend_label(),
    }


def print_header(sysinfo: dict[str, Any], devices: list[torch.device]) -> None:
    log(c("bold", c("cyan", "PyTorch device probe + light benchmark")))
    log(c("dim", "1 warmup + 3 runs  →  fastest + average  |  multi-GPU in parallel"))
    log()
    log(c("bold", "System"))
    rows = [
        ("Python", sysinfo["python"]),
        ("OS", sysinfo["platform"]),
        ("CPU", sysinfo["cpu"]),
        ("CPU cores", str(sysinfo["cpu_cores"])),
        ("RAM", fmt_bytes(sysinfo["ram"])),
        ("PyTorch", sysinfo["pytorch"]),
        ("CUDA", sysinfo["cuda"] or "n/a"),
        ("HIP/ROCm", sysinfo["hip"] or "n/a"),
        ("cuDNN", str(sysinfo["cudnn"] if sysinfo["cudnn"] is not None else "n/a")),
        ("Primary backend", sysinfo["backend"]),
    ]
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        log(f"  {c('dim', k.ljust(width))}  {v}")

    log()
    log(c("bold", "Devices"))
    for d in devices:
        vram = device_vram(d)
        extra = extra_device_info(d)
        cap = f"  cc {extra['capability']}" if extra.get("capability") else ""
        if d.type == "cpu":
            mem = f"  RAM {fmt_bytes(ram_bytes())}"
        elif d.type == "mps":
            mem = f"  unified {fmt_bytes(vram)}"
        else:
            mem = f"  VRAM {fmt_bytes(vram)}" if vram is not None else ""
        log(f"  {c('yellow', str(d)):<22} {device_name(d)}{cap}{mem}")
    log()


def print_result(device: torch.device, result: dict[str, Any] | None, err: str | None) -> None:
    tag = f"{device}"
    if err:
        name = (result or {}).get("bench", "")
        mode = (result or {}).get("mode")
        label = f"{name}/{mode}" if mode else name
        log(f"  {c('red', 'FAIL')}  {tag:<10} {label:<22} {err}")
        return
    assert result is not None
    bench = result["bench"]
    if bench == "bandwidth":
        parts = []
        for key, label in (("d2d", "D2D"), ("h2d", "H2D"), ("d2h", "D2H")):
            block = result.get("copies", {}).get(key)
            if not block:
                continue
            parts.append(f"{label} {block['gbs_best']:7.1f} GB/s")
        log(
            f"  {c('green', 'PASS')}  {tag:<10} {'bandwidth':<22} "
            + "  ".join(parts)
            + f"  {c('dim', result['detail'])}"
        )
        return
    best_ms = result["fastest_s"] * 1000
    avg_ms = result["average_s"] * 1000
    label = f"{bench}/{result['mode']}" if result.get("mode") else bench
    extra = ""
    if bench in ("compile", "export"):
        extra = (
            f"  {c('magenta', f'{result['speedup_best']:.2f}x eager')}  "
            f"compile {result['compile_s']*1000:.0f} ms"
        )
    log(
        f"  {c('green', 'PASS')}  {tag:<10} {label:<22} "
        f"best {c('bold', f'{best_ms:8.2f} ms')}  "
        f"avg {avg_ms:8.2f} ms  "
        f"{c('cyan', f'{result['tflops_best']:7.2f} TFLOPS')}"
        f"{extra}  {c('dim', result['detail'])}"
    )


def json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_ready(v) for v in obj]
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj).replace("torch.", "")
    return obj


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_BENCHES = ["matmul", "bandwidth", "train", "infer", "compile", "export"]
COMPILE_MODES = ["default", "reduce-overhead", "max-autotune"]


def parse_benches(spec: str) -> list[str]:
    spec = spec.lower().strip()
    if spec == "all":
        return list(ALL_BENCHES)
    items = [x.strip() for x in spec.split(",") if x.strip()]
    mapped: list[str] = []
    for x in items:
        if x == "jit":
            mapped.append("compile")
        else:
            mapped.append(x)
    unknown = [x for x in mapped if x not in ALL_BENCHES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown bench {unknown}; use {','.join(ALL_BENCHES)},all"
        )
    return mapped


def parse_compile_modes(spec: str) -> list[str]:
    spec = spec.lower().strip()
    if spec in ("all", "both"):
        return ["default", "reduce-overhead"]
    items = [x.strip() for x in spec.split(",") if x.strip()]
    unknown = [x for x in items if x not in COMPILE_MODES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown compile mode {unknown}; use {','.join(COMPILE_MODES)},all"
        )
    return items


def auto_bw_mb(device: torch.device, requested: int) -> int:
    if requested > 0:
        return requested
    vram = device_vram(device)
    if vram:
        return max(64, min(1024, int(vram * 0.08 / (1024 * 1024))))
    return 256 if device.type != "cpu" else 128


def run_one_bench(
    device: torch.device,
    bench: str,
    args: argparse.Namespace,
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if bench == "matmul":
        n = args.size if args.size > 0 else (2048 if device.type == "cpu" else 4096)
        rows.append(bench_matmul(device, dtype, n, max(1, args.matmul_inner)))
    elif bench == "train":
        rows.append(
            bench_train(device, dtype, args.batch, args.image_size, max(1, args.train_steps))
        )
    elif bench == "infer":
        rows.append(
            bench_infer(device, dtype, args.batch, args.image_size, max(1, args.infer_steps))
        )
    elif bench == "bandwidth":
        rows.append(
            bench_bandwidth(device, auto_bw_mb(device, args.bw_mb), max(1, args.bw_inner))
        )
    elif bench == "compile":
        for mode in parse_compile_modes(args.compile_mode):
            rows.append(
                bench_compile(
                    device, dtype, args.batch, args.image_size, max(1, args.infer_steps), mode
                )
            )
    elif bench == "export":
        rows.append(
            bench_export(device, dtype, args.batch, args.image_size, max(1, args.infer_steps))
        )
    else:
        raise ValueError(bench)
    return rows


def run_device(
    device: torch.device,
    benches: list[str],
    args: argparse.Namespace,
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    with on_device(device):
        use_dtype = dtype if dtype_supported(device, dtype) else torch.float32
        if use_dtype != dtype:
            log(c("yellow", "SKIP") + f"  {device} {args.dtype} unsupported; using float32")

        for bench in benches:
            meta = {
                "device": str(device),
                "device_name": device_name(device),
                "vram": device_vram(device),
                "dtype": str(use_dtype).replace("torch.", ""),
            }
            try:
                for result in run_one_bench(device, bench, args, use_dtype):
                    print_result(device, result, None)
                    report.append({**meta, **result, "ok": True})
            except Exception as e:
                stub = {"bench": bench}
                print_result(device, stub, f"{type(e).__name__}: {e}")
                report.append({**meta, "bench": bench, "ok": False, "error": str(e)})
            finally:
                empty_cache(device)
    return report


def group_devices(devices: list[torch.device]) -> list[list[torch.device]]:
    cuda = [d for d in devices if d.type == "cuda"]
    xpu = [d for d in devices if d.type == "xpu"]
    rest = [d for d in devices if d.type not in ("cuda", "xpu", "cpu")]
    cpu = [d for d in devices if d.type == "cpu"]
    groups: list[list[torch.device]] = []
    if cuda:
        groups.append(cuda)
    if xpu:
        groups.append(xpu)
    for d in rest:
        groups.append([d])
    if cpu:
        groups.append(cpu)
    return groups


def run_group(
    group: list[torch.device],
    benches: list[str],
    args: argparse.Namespace,
    dtype: torch.dtype,
    parallel: bool,
) -> list[dict[str, Any]]:
    if len(group) == 1 or not parallel:
        out: list[dict[str, Any]] = []
        for d in group:
            out.extend(run_device(d, benches, args, dtype))
        return out

    jobs = len(group) if args.jobs <= 0 else max(1, min(args.jobs, len(group)))
    log(
        c("dim", f"Running {len(group)} devices in parallel (jobs={jobs}): ")
        + ", ".join(str(d) for d in group)
    )
    out: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="pt-bench") as pool:
        futs = [pool.submit(run_device, d, benches, args, dtype) for d in group]
        for fut in as_completed(futs):
            out.extend(fut.result())
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Probe PyTorch devices and run a light benchmark")
    p.add_argument("--device", default="all", help="all | auto | cuda | mps | xpu | cpu | cuda:0")
    p.add_argument(
        "--bench",
        default="all",
        help="all | comma list: matmul,bandwidth,train,infer,compile,export",
    )
    p.add_argument("--size", type=int, default=0, help="matmul N (0=auto)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--train-steps", type=int, default=10)
    p.add_argument("--infer-steps", type=int, default=20)
    p.add_argument("--matmul-inner", type=int, default=1)
    p.add_argument("--bw-mb", type=int, default=0, help="copy size in MiB (0=auto ~8% VRAM)")
    p.add_argument("--bw-inner", type=int, default=20, help="copies per timed bandwidth run")
    p.add_argument(
        "--compile-mode",
        default="default,reduce-overhead",
        help="default | reduce-overhead | max-autotune | all (all = first two)",
    )
    p.add_argument("--dtype", default="float32", help="float32 | float16 | bfloat16")
    p.add_argument("--jobs", type=int, default=0, help="parallel GPU workers (0=one per GPU)")
    p.add_argument("--sequential", action="store_true", help="disable multi-GPU parallel")
    p.add_argument("--json", nargs="?", const="pytorch_bench.json")
    p.add_argument("--no-color", action="store_true")
    args = p.parse_args()

    use_color = (not args.no_color) and sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    if use_color:
        enable_windows_ansi()
        Color.enable(True)

    try:
        benches = parse_benches(args.bench)
        dtype = parse_dtype(args.dtype)
        parse_compile_modes(args.compile_mode)
    except argparse.ArgumentTypeError as e:
        p.error(str(e))

    all_devs = discover_devices()
    devices = filter_devices(all_devs, args.device)
    if not devices:
        log(c("red", f"No devices matched --device {args.device}"))
        log("Available: " + (", ".join(str(d) for d in all_devs) or "(none)"))
        return 2

    sysinfo = collect_system()
    print_header(sysinfo, all_devs)
    if args.device != "all":
        log(c("dim", "Running on: " + ", ".join(str(d) for d in devices)))
        log()

    parallel = not args.sequential
    log(
        c("bold", "Benchmark")
        + c(
            "dim",
            f"  dtype={str(dtype).replace('torch.', '')}  benches={','.join(benches)}"
            f"  parallel={'on' if parallel else 'off'}  compile={args.compile_mode}",
        )
    )

    report: list[dict[str, Any]] = []
    for group in group_devices(devices):
        report.extend(run_group(group, benches, args, dtype, parallel))

    failures = sum(1 for r in report if not r.get("ok", False))
    log()
    if failures:
        log(c("yellow", f"Done with {failures} failure(s)."))
    else:
        log(c("green", "All selected benches passed."))

    if args.json:
        payload = {
            "system": sysinfo,
            "devices": [
                {
                    "id": str(d),
                    "name": device_name(d),
                    "vram": device_vram(d),
                    **extra_device_info(d),
                }
                for d in all_devs
            ],
            "results": report,
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(json_ready(payload), f, indent=2)
        log(c("dim", f"Wrote {args.json}"))

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
