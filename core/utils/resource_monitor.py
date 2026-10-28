"""크래시/무응답 대응 로깅 — 문제가 생겼을 때만 VRAM/RAM 상태와 스택을 남긴다 (상시 모니터링 X).

외부 의존성 없이 torch.cuda + nvidia-smi + /proc 만 사용 (psutil/pynvml 미설치 환경).

- ``snapshot()`` / ``format_snapshot``: 프로세스(torch) VRAM + 디바이스 VRAM + RSS / 시스템 RAM
  → 로그 한 줄. 예외 발생 시 호출.
- ``gpu_processes``: GPU 점유 프로세스 목록 (다른 모듈이 VRAM 을 먹는지 확인용).
- ``HangWatchdog``: 요청 시작 시 arm, 끝나면 disarm. 타임아웃 시에만 스택 + 상태 기록.
- ``enable_fault_dump``: faulthandler 로 SIGSEGV/SIGABRT/SIGFPE 시 전체 스레드 파이썬
  스택을 파일에 덤프 (CUDA abort / open3d segfault 대응).

torch 는 ``sys.modules`` 에 이미 올라와 있을 때만 조회한다 — 진단 코드가 torch import 나
CUDA context 생성(수백 MB VRAM)을 유발하지 않도록.
"""

from __future__ import annotations

import faulthandler
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_GIB = 1024 ** 3
_MIB = 1024 ** 2

_NVSMI_TIMEOUT_SEC = 3.0


# ----------------------------------------------------------------------
# 수집
# ----------------------------------------------------------------------

def _torch_cuda():
    """이미 import 되고 CUDA context 가 초기화된 torch.cuda 만 반환. 아니면 None."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            return torch.cuda
    except Exception:
        pass
    return None


def _torch_stats() -> Optional[Dict[str, float]]:
    cuda = _torch_cuda()
    if cuda is None:
        return None
    try:
        dev = cuda.current_device()
        return {
            "device": dev,
            "allocated": cuda.memory_allocated(dev),
            "reserved": cuda.memory_reserved(dev),
            "peak_allocated": cuda.max_memory_allocated(dev),
            "peak_reserved": cuda.max_memory_reserved(dev),
        }
    except Exception:
        return None


def _nvidia_smi(args: List[str]) -> Optional[List[List[str]]]:
    try:
        out = subprocess.run(
            ["nvidia-smi", *args, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=_NVSMI_TIMEOUT_SEC,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    return [
        [c.strip() for c in line.split(",")]
        for line in out.stdout.strip().splitlines() if line.strip()
    ]


def _gpu_devices() -> Optional[List[Dict[str, Any]]]:
    """디바이스 전체 VRAM (모든 프로세스 합). nvidia-smi 실패 시 torch.mem_get_info fallback."""
    rows = _nvidia_smi([
        "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu",
    ])
    if rows:
        devs = []
        for r in rows:
            try:
                devs.append({
                    "index": int(r[0]),
                    "used": float(r[1]) * _MIB,
                    "total": float(r[2]) * _MIB,
                    "util": _to_float(r[3]),
                    "temp": _to_float(r[4]),
                })
            except (ValueError, IndexError):
                continue
        if devs:
            return devs

    cuda = _torch_cuda()
    if cuda is None:
        return None
    try:
        dev = cuda.current_device()
        free, total = cuda.mem_get_info(dev)
        return [{"index": dev, "used": total - free, "total": total, "util": None, "temp": None}]
    except Exception:
        return None


def gpu_processes() -> Optional[List[Dict[str, Any]]]:
    """GPU 를 점유 중인 프로세스 목록 (pid, 이름, 사용 VRAM). 다른 모듈이 VRAM 을 먹는지 확인용."""
    rows = _nvidia_smi(["--query-compute-apps=pid,process_name,used_memory"])
    if rows is None:
        return None
    procs = []
    for r in rows:
        try:
            procs.append({"pid": int(r[0]), "name": r[1], "used": float(r[2]) * _MIB})
        except (ValueError, IndexError):
            continue
    return sorted(procs, key=lambda p: p["used"], reverse=True)


def _read_kv(path: str) -> Dict[str, int]:
    """/proc/meminfo, /proc/self/status 형식 ("Key:   123 kB") → bytes dict."""
    out: Dict[str, int] = {}
    try:
        with open(path, "r") as f:
            for line in f:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if len(parts) >= 2 and parts[1] == "kB" and parts[0].isdigit():
                    out[key] = int(parts[0]) * 1024
    except OSError:
        pass
    return out


def _ram_stats() -> Dict[str, Optional[int]]:
    proc = _read_kv("/proc/self/status")
    mem = _read_kv("/proc/meminfo")
    swap_total = mem.get("SwapTotal")
    swap_free = mem.get("SwapFree")
    return {
        "rss": proc.get("VmRSS"),
        "rss_peak": proc.get("VmHWM"),
        "mem_total": mem.get("MemTotal"),
        "mem_available": mem.get("MemAvailable"),
        "swap_used": (swap_total - swap_free) if swap_total is not None and swap_free is not None else None,
    }


def snapshot() -> Dict[str, Any]:
    return {
        "torch": _torch_stats(),
        "gpus": _gpu_devices(),
        "ram": _ram_stats(),
    }


def reset_peak() -> None:
    """torch peak 통계 리셋 — 요청 단위 peak VRAM 을 보기 위해 요청 시작 시 호출."""
    cuda = _torch_cuda()
    if cuda is None:
        return
    try:
        cuda.reset_peak_memory_stats()
    except Exception:
        pass


def empty_cache() -> None:
    cuda = _torch_cuda()
    if cuda is None:
        return
    try:
        cuda.empty_cache()
    except Exception:
        pass


def is_cuda_oom(e: BaseException) -> bool:
    """torch.cuda.OutOfMemoryError 또는 CUDA OOM RuntimeError 여부 (torch import 없이 판별)."""
    if type(e).__name__ == "OutOfMemoryError":
        return True
    return "CUDA out of memory" in str(e) or "CUBLAS_STATUS_ALLOC_FAILED" in str(e)


# ----------------------------------------------------------------------
# 포맷
# ----------------------------------------------------------------------

def _to_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except ValueError:
        return None


def _g(v: Optional[float]) -> str:
    return "?" if v is None else f"{v / _GIB:.2f}G"


def format_snapshot(snap: Dict[str, Any], tag: str = "") -> str:
    """예: ``[RES] req_match | torch alloc 2.31G rsv 3.50G peak 6.12G | gpu0 7.80G/31.84G
    util 12% 45C | rss 4.21G (peak 5.02G) | ram avail 40.1G/62.7G swap 0.00G``"""
    parts = [f"[RES] {tag}".rstrip()]

    t = snap.get("torch")
    if t:
        parts.append(
            f"torch alloc {_g(t['allocated'])} rsv {_g(t['reserved'])} "
            f"peak {_g(t['peak_allocated'])} (rsv {_g(t['peak_reserved'])})"
        )
    else:
        parts.append("torch cuda n/a")

    for g in snap.get("gpus") or []:
        s = f"gpu{g['index']} {_g(g['used'])}/{_g(g['total'])}"
        if g.get("util") is not None:
            s += f" util {g['util']:.0f}%"
        if g.get("temp") is not None:
            s += f" {g['temp']:.0f}C"
        parts.append(s)
    if not snap.get("gpus"):
        parts.append("gpu n/a")

    r = snap.get("ram") or {}
    parts.append(f"rss {_g(r.get('rss'))} (peak {_g(r.get('rss_peak'))})")
    parts.append(
        f"ram avail {_g(r.get('mem_available'))}/{_g(r.get('mem_total'))} "
        f"swap {_g(r.get('swap_used'))}"
    )
    return " | ".join(parts)


def format_gpu_processes(procs: Optional[List[Dict[str, Any]]]) -> str:
    if procs is None:
        return "[RES] gpu procs: n/a"
    if not procs:
        return "[RES] gpu procs: (none)"
    me = os.getpid()
    items = [
        f"{p['pid']}{'*' if p['pid'] == me else ''}:{Path(p['name']).name}={_g(p['used'])}"
        for p in procs
    ]
    return "[RES] gpu procs: " + ", ".join(items)


# ----------------------------------------------------------------------
# hang watchdog / 크래시 덤프
# ----------------------------------------------------------------------

class HangWatchdog:
    """요청 단위 무응답 감시 — 평소엔 아무것도 안 하고, 타임아웃 때만 기록.

    ``arm()`` 후 ``timeout_sec`` 내 ``disarm()`` 이 안 되면 ``on_timeout(tag, elapsed, stack)``
    을 호출한다 (stack = arm 한 스레드의 현재 파이썬 스택). 막힌 C 확장이 GIL 을 쥐고 있으면
    파이썬 Timer 가 못 도므로, ``fault_file`` 이 있으면 faulthandler 의 C 레벨 타이머로
    전체 스레드 스택도 덤프한다 (백업, timeout + 5s).
    """

    def __init__(
        self,
        timeout_sec: float,
        on_timeout: Callable[[str, float, str], None],
        fault_file=None,
    ):
        self.timeout_sec = float(timeout_sec)
        self.fault_file = fault_file
        self._on_timeout = on_timeout
        self._timer: Optional[threading.Timer] = None

    def arm(self, tag: str) -> None:
        self.disarm()
        if self.timeout_sec <= 0:
            return
        t0 = time.monotonic()
        ident = threading.get_ident()
        self._timer = threading.Timer(self.timeout_sec, self._fire, args=(tag, t0, ident))
        self._timer.daemon = True
        self._timer.start()
        if self.fault_file is not None:
            faulthandler.dump_traceback_later(self.timeout_sec + 5.0, file=self.fault_file)

    def disarm(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self.fault_file is not None:
            faulthandler.cancel_dump_traceback_later()

    def _fire(self, tag: str, t0: float, ident: int) -> None:
        frame = sys._current_frames().get(ident)
        stack = "".join(traceback.format_stack(frame)) if frame is not None else "(thread gone)"
        try:
            self._on_timeout(tag, time.monotonic() - t0, stack)
        except Exception:
            pass

def enable_fault_dump(path: Path):
    """faulthandler 를 ``path`` 에 append 로 연결. 반환된 파일 객체를 살아있게 유지해야 함."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a", buffering=1, encoding="utf-8")
    faulthandler.enable(file=f, all_threads=True)
    return f
