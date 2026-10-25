#!/usr/bin/env python3
"""PyTorch device probe + light benchmark.

Usage:
  python pytorch_bench.py
  python pytorch_bench.py --bench matmul
  python pytorch_bench.py --bench train,infer --device cuda
  python pytorch_bench.py --device auto --json results.json
  python pytorch_bench.py --size 4096 --dtype float16
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# ANSI color (no extra deps)
# ---------------------------------------------------------------------------

class Color:
    off = ""
    reset = ""
    bold = ""
    dim = ""
    red = ""
    green = ""
    yellow = ""
    cyan = ""
    magenta = ""

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
    if not s:
        return text
    return f"{s}{text}{Color.reset}"


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


# ---------------------------------------------------------------------------
# System info (stdlib only)
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
            return (
                platform.processor()
                or os.environ.get("PROCESSOR_IDENTIFIER")
                or "Unknown CPU"
            )
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
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out.strip())
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
# PyTorch import + devices
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
except ImportError:
    sys.stderr.write("PyTorch is not installed. Try: pip install torch\n")
    sys.exit(1)


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
        return ram_bytes()  # unified
    return None


def extra_device_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        info["capability"] = f"{p.major}.{p.minor}"
        info["multi_processor_count"] = getattr(p, "multi_processor_count", None)
    return info


def discover_devices() -> list[torch.device]:
    """Accelerators first (CUDA/ROCm, MPS, XPU), CPU last."""
    found: list[torch.device] = []

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            found.append(torch.device(f"cuda:{i}"))

    mps_ok = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    if mps_ok:
        found.append(torch.device("mps"))

    if hasattr(torch, "xpu"):
        try:
            if torch.xpu.is_available():
                n = torch.xpu.device_count()
                for i in range(n):
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
        elif spec == key or spec == f"{d.type}:{d.index}":
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
    for _ in range(runs):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - t0)

    return min(times), sum(times) / len(times), times


def tflops(flops: float, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return flops / seconds / 1e12


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
        return dtype == torch.bfloat16  # fp16 CPU matmul is often missing/slow-path
    if device.type == "mps":
        return dtype in (torch.float16, torch.bfloat16)
    return True  # CUDA / XPU


# ---------------------------------------------------------------------------
# Model + FLOP estimates
# ---------------------------------------------------------------------------

class TinyNet(nn.Module):
    """Small conv net for a light train/infer smoke test."""

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
        )
        self.head = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.head(x)


def conv_flops(batch: int, cin: int, cout: int, h: int, w: int, k: int) -> int:
    # multiply-add counted as 2 FLOPs
    return 2 * batch * cout * h * w * (k * k * cin)


def tiny_forward_flops(batch: int, image: int) -> int:
    h = w = image
    return (
        conv_flops(batch, 3, 32, h, w, 3)
        + conv_flops(batch, 32, 64, h, w, 3)
        + conv_flops(batch, 64, 64, h, w, 3)
        + 2 * batch * 64 * 10
    )


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_matmul(
    device: torch.device,
    dtype: torch.dtype,
    n: int,
    inner: int,
) -> dict[str, Any]:
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    out = torch.empty(n, n, device=device, dtype=dtype)

    def run() -> None:
        nonlocal out
        for _ in range(inner):
            out = a @ b

    flops = inner * (2 * n * n * n)
    fastest, avg, times = timed_runs(run, device)
    # keep a live reference so the last result is not optimized away
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
    device: torch.device,
    dtype: torch.dtype,
    batch: int,
    image: int,
    steps: int,
) -> dict[str, Any]:
    model = TinyNet().to(device=device, dtype=dtype)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(batch, 3, image, image, device=device, dtype=dtype)
    y = torch.randint(0, 10, (batch,), device=device)

    def run() -> None:
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

    # ~1x forward + ~2x backward
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
    device: torch.device,
    dtype: torch.dtype,
    batch: int,
    image: int,
    steps: int,
) -> dict[str, Any]:
    model = TinyNet().to(device=device, dtype=dtype)
    model.eval()
    x = torch.randn(batch, 3, image, image, device=device, dtype=dtype)

    def run() -> None:
        with torch.no_grad():
            y = x
            for _ in range(steps):
                y = model(x)
            if not torch.isfinite(y.float().sum()):
                raise RuntimeError("infer produced non-finite values")

    flops = steps * tiny_forward_flops(batch, image)
    fastest, avg, times = timed_runs(run, device)
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


BENCH_FNS = {
    "matmul": bench_matmul,
    "train": bench_train,
    "infer": bench_infer,
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
    print(c("bold", c("cyan", "PyTorch device probe + light benchmark")))
    print(c("dim", "1 warmup + 3 runs  →  fastest + average  →  estimated TFLOPS"))
    print()
    print(c("bold", "System"))
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
        print(f"  {c('dim', k.ljust(width))}  {v}")

    print()
    print(c("bold", "Devices"))
    for d in devices:
        vram = device_vram(d)
        extra = extra_device_info(d)
        cap = f"  cc {extra['capability']}" if extra.get("capability") else ""
        mem = f"  VRAM {fmt_bytes(vram)}" if vram is not None else ""
        if d.type == "mps":
            mem = f"  unified {fmt_bytes(vram)}"
        if d.type == "cpu":
            mem = f"  RAM {fmt_bytes(ram_bytes())}"
        print(f"  {c('yellow', str(d)):<22} {device_name(d)}{cap}{mem}")
    print()


def print_result(device: torch.device, result: dict[str, Any] | None, err: str | None) -> None:
    tag = f"{device}"
    if err:
        print(f"  {c('red', 'FAIL')}  {tag:<10} {result and result.get('bench') or '':<8} {err}")
        return
    assert result is not None
    best_ms = result["fastest_s"] * 1000
    avg_ms = result["average_s"] * 1000
    print(
        f"  {c('green', 'PASS')}  {tag:<10} {result['bench']:<8} "
        f"best {c('bold', f'{best_ms:8.2f} ms')}  "
        f"avg {avg_ms:8.2f} ms  "
        f"{c('cyan', f'{result['tflops_best']:7.2f} TFLOPS')}  "
        f"{c('dim', result['detail'])}"
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
# Main
# ---------------------------------------------------------------------------

def parse_benches(spec: str) -> list[str]:
    spec = spec.lower().strip()
    if spec == "all":
        return ["matmul", "train", "infer"]
    items = [x.strip() for x in spec.split(",") if x.strip()]
    unknown = [x for x in items if x not in BENCH_FNS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown bench {unknown}; use matmul,train,infer,all"
        )
    return items


def main() -> int:
    p = argparse.ArgumentParser(description="Probe PyTorch devices and run a light benchmark")
    p.add_argument(
        "--device",
        default="all",
        help="all | auto | cuda | mps | xpu | cpu | cuda:0 (default: all)",
    )
    p.add_argument(
        "--bench",
        default="all",
        help="all | comma list: matmul,train,infer (default: all)",
    )
    p.add_argument("--size", type=int, default=0, help="matmul N for NxN (0 = auto: 4096 accel / 2048 CPU)")
    p.add_argument("--batch", type=int, default=32, help="train/infer batch size")
    p.add_argument("--image-size", type=int, default=64, help="train/infer spatial size")
    p.add_argument("--train-steps", type=int, default=10, help="optimizer steps per timed run")
    p.add_argument("--infer-steps", type=int, default=20, help="forward passes per timed run")
    p.add_argument("--matmul-inner", type=int, default=1, help="gemm repeats per timed run")
    p.add_argument("--dtype", default="float32", help="float32 | float16 | bfloat16")
    p.add_argument("--json", nargs="?", const="pytorch_bench.json", help="write JSON report")
    p.add_argument("--no-color", action="store_true")
    args = p.parse_args()

    use_color = (not args.no_color) and sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    if use_color:
        enable_windows_ansi()
        Color.enable(True)

    try:
        benches = parse_benches(args.bench)
        dtype = parse_dtype(args.dtype)
    except argparse.ArgumentTypeError as e:
        p.error(str(e))

    all_devs = discover_devices()
    devices = filter_devices(all_devs, args.device)
    if not devices:
        print(c("red", f"No devices matched --device {args.device}"))
        print("Available:", ", ".join(str(d) for d in all_devs) or "(none)")
        return 2

    sysinfo = collect_system()
    print_header(sysinfo, all_devs)
    if args.device != "all":
        print(c("dim", f"Running on: {', '.join(str(d) for d in devices)}"))
        print()

    print(c("bold", "Benchmark") + c("dim", f"  dtype={str(dtype).replace('torch.', '')}  benches={','.join(benches)}"))
    report: list[dict[str, Any]] = []
    failures = 0

    for device in devices:
        use_dtype = dtype
        if not dtype_supported(device, dtype):
            print(
                c("yellow", "SKIP")
                + f"  {device} does not support {args.dtype} well; using float32"
            )
            use_dtype = torch.float32

        for bench in benches:
            meta = {
                "device": str(device),
                "device_name": device_name(device),
                "vram": device_vram(device),
                "dtype": str(use_dtype).replace("torch.", ""),
            }
            try:
                if bench == "matmul":
                    n = args.size if args.size > 0 else (2048 if device.type == "cpu" else 4096)
                    result = bench_matmul(device, use_dtype, n, max(1, args.matmul_inner))
                elif bench == "train":
                    result = bench_train(
                        device, use_dtype, args.batch, args.image_size, max(1, args.train_steps)
                    )
                else:
                    result = bench_infer(
                        device, use_dtype, args.batch, args.image_size, max(1, args.infer_steps)
                    )
                print_result(device, result, None)
                report.append({**meta, **result, "ok": True})
            except Exception as e:
                failures += 1
                print_result(device, {"bench": bench}, f"{type(e).__name__}: {e}")
                report.append({**meta, "bench": bench, "ok": False, "error": str(e)})
            finally:
                empty_cache(device)

    print()
    if failures:
        print(c("yellow", f"Done with {failures} failure(s)."))
    else:
        print(c("green", "All selected benches passed."))

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
        print(c("dim", f"Wrote {args.json}"))

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
