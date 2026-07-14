#!/usr/bin/env python3
"""
scripts/hardware_check.py

Этап 1. Анализ окружения.

Определяет CPU, RAM, GPU (модель, число устройств, VRAM), версию CUDA,
версию драйвера NVIDIA и свободное место на диске. Печатает красивую
таблицу через `rich` и возвращает словарь с данными, который используется
scripts/optimize.py для автоподбора параметров запуска.

Запуск:
    python scripts/hardware_check.py
"""

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict

try:
    import psutil
except ImportError:
    psutil = None

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
except ImportError:
    Console = None


@dataclass
class GPUInfo:
    index: int
    name: str
    total_memory_mb: int
    free_memory_mb: int
    driver_version: str = ""
    compute_cap: str = ""


@dataclass
class HardwareReport:
    cpu_model: str = "unknown"
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    disk_free_gb: float = 0.0
    cuda_version: str = "not found"
    driver_version: str = "not found"
    gpu_count: int = 0
    gpus: list = field(default_factory=list)
    is_kaggle: bool = False


def _run(cmd: str) -> str:
    try:
        out = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=15
        )
        return out.stdout.strip()
    except Exception:
        return ""


def detect_cpu(report: HardwareReport) -> None:
    report.cpu_model = platform.processor() or "unknown"
    # На Kaggle platform.processor() часто пустой — читаем /proc/cpuinfo
    if not report.cpu_model or report.cpu_model == "unknown":
        cpuinfo = _run("grep -m1 'model name' /proc/cpuinfo")
        if cpuinfo:
            report.cpu_model = cpuinfo.split(":", 1)[-1].strip()
    if psutil:
        report.cpu_cores_physical = psutil.cpu_count(logical=False) or 0
        report.cpu_cores_logical = psutil.cpu_count(logical=True) or 0
    else:
        report.cpu_cores_logical = os.cpu_count() or 0
        report.cpu_cores_physical = report.cpu_cores_logical


def detect_ram(report: HardwareReport) -> None:
    if psutil:
        vm = psutil.virtual_memory()
        report.ram_total_gb = round(vm.total / (1024 ** 3), 2)
        report.ram_available_gb = round(vm.available / (1024 ** 3), 2)


def detect_disk(report: HardwareReport, path: str = "/kaggle/working") -> None:
    target = path if os.path.exists(path) else "."
    total, used, free = shutil.disk_usage(target)
    report.disk_free_gb = round(free / (1024 ** 3), 2)


def detect_gpus(report: HardwareReport) -> None:
    smi = _run(
        "nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap "
        "--format=csv,noheader,nounits"
    )
    if not smi:
        return
    for line in smi.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        idx, name, total_mem, free_mem, driver, cc = parts
        report.gpus.append(
            GPUInfo(
                index=int(idx),
                name=name,
                total_memory_mb=int(float(total_mem)),
                free_memory_mb=int(float(free_mem)),
                driver_version=driver,
                compute_cap=cc,
            )
        )
        report.driver_version = driver
    report.gpu_count = len(report.gpus)

    nvcc = _run("nvcc --version")
    if nvcc:
        for line in nvcc.splitlines():
            if "release" in line.lower():
                # пример: "Cuda compilation tools, release 12.5, V12.5.82"
                try:
                    report.cuda_version = line.split("release")[1].split(",")[0].strip()
                except Exception:
                    pass
    if report.cuda_version == "not found":
        # fallback: смотрим версию CUDA runtime, которую видит сам драйвер
        cuda_from_smi = _run("nvidia-smi | grep -o 'CUDA Version: [0-9.]*'")
        if cuda_from_smi:
            report.cuda_version = cuda_from_smi.replace("CUDA Version:", "").strip()


def detect_kaggle(report: HardwareReport) -> None:
    report.is_kaggle = bool(
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.exists("/kaggle")
    )


def collect() -> HardwareReport:
    report = HardwareReport()
    detect_cpu(report)
    detect_ram(report)
    detect_disk(report)
    detect_gpus(report)
    detect_kaggle(report)
    return report


def print_report(report: HardwareReport) -> None:
    if Console is None:
        print(json.dumps(asdict(report), indent=2, ensure_ascii=False))
        return

    console = Console()
    console.print(
        Panel.fit(
            "[bold cyan]kaggle-llm-server[/bold cyan] — анализ окружения (Этап 1)",
            border_style="cyan",
        )
    )

    sys_table = Table(title="Система", show_header=True, header_style="bold magenta")
    sys_table.add_column("Параметр")
    sys_table.add_column("Значение")
    sys_table.add_row("Kaggle-окружение", "да" if report.is_kaggle else "нет (локально)")
    sys_table.add_row("CPU", report.cpu_model)
    sys_table.add_row("Физических ядер", str(report.cpu_cores_physical))
    sys_table.add_row("Логических ядер (threads)", str(report.cpu_cores_logical))
    sys_table.add_row("RAM всего, GB", str(report.ram_total_gb))
    sys_table.add_row("RAM доступно, GB", str(report.ram_available_gb))
    sys_table.add_row("Свободно на диске, GB", str(report.disk_free_gb))
    sys_table.add_row("CUDA Toolkit", report.cuda_version)
    sys_table.add_row("Драйвер NVIDIA", report.driver_version)
    console.print(sys_table)

    gpu_table = Table(title="GPU", show_header=True, header_style="bold green")
    gpu_table.add_column("#")
    gpu_table.add_column("Модель")
    gpu_table.add_column("VRAM всего, MB")
    gpu_table.add_column("VRAM свободно, MB")
    gpu_table.add_column("Compute Cap.")
    if report.gpus:
        for g in report.gpus:
            gpu_table.add_row(
                str(g.index), g.name, str(g.total_memory_mb), str(g.free_memory_mb), g.compute_cap
            )
    else:
        gpu_table.add_row("-", "GPU не обнаружены", "-", "-", "-")
    console.print(gpu_table)

    if report.gpu_count == 0:
        console.print(
            "[bold yellow]Внимание:[/bold yellow] GPU не найдены. Проверьте, что в Kaggle "
            "Notebook включён accelerator 'GPU T4 x2' (Settings → Accelerator)."
        )


def save_json(report: HardwareReport, path: str = "./logs/hardware_report.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    rep = collect()
    print_report(rep)
    save_json(rep)
    if rep.gpu_count == 0:
        sys.exit(1)
