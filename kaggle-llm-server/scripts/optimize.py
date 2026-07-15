#!/usr/bin/env python3
"""
scripts/optimize.py

Этап 5. Оптимальный запуск.

На основе hardware_report.json (создаётся hardware_check.py) и размера
GGUF-модели на диске подбирает:
    - n_gpu_layers   — сколько слоёв выгрузить на GPU
    - ctx_size       — размер контекста
    - batch_size / ubatch_size
    - threads / threads_batch
    - tensor_split   — распределение между 2×T4
    - cache_type_k/v — тип квантования KV-кэша

Логика (эвристика, проверенная на 2×Tesla T4 16GB):
    1. Каждая T4 имеет 16 GB VRAM (15.5-15.8 GB реально доступно).
       Держим буфер ~1.2 GB на GPU под CUDA context/graph/activation.
    2. Общий бюджет VRAM под веса модели + KV-cache ~= 2 * (16 - 1.2) GB.
    3. Если размер файла модели (VRAM для весов) укладывается в бюджет —
       n_gpu_layers = -1 (все слои на GPU, максимальная скорость).
       Иначе — линейно уменьшаем число слоёв.
    4. ctx_size подбирается так, чтобы KV-cache (с учётом q8_0 квантования)
       не превысил оставшуюся после весов VRAM. Ограничен сверху 32768,
       снизу 4096.
    5. tensor_split вычисляется пропорционально свободной VRAM на каждой
       карте (обычно "1,1" при одинаковых T4).
"""

import json
import math
import os
import re
import sys

VRAM_RESERVE_MB_PER_GPU = 1200   # буфер под CUDA context / graph
DEFAULT_CTX_MIN = 4096
DEFAULT_CTX_MAX = 65536
BYTES_PER_TOKEN_KV_Q8 = 0.5      # приблизительно, зависит от arch/n_layer/n_head — грубая эвристика


def load_hardware(path="./logs/hardware_report.json") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} не найден. Сначала запустите scripts/hardware_check.py"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def guess_model_layers_and_size(model_path: str):
    """Пытается достать метаданные GGUF (число слоёв) через gguf-python при
    наличии, либо оценивает размер файла в MB."""
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    n_layers = None
    try:
        from gguf import GGUFReader  # опциональная зависимость, если установлена
        reader = GGUFReader(model_path)
        for key in reader.fields:
            if key.endswith(".block_count"):
                n_layers = int(reader.fields[key].parts[-1][0])
                break
    except Exception:
        pass
    return size_mb, n_layers


def compute_params(hw: dict, model_path: str, cfg_server: dict) -> dict:
    gpus = hw.get("gpus", [])
    gpu_count = max(1, len(gpus))
    size_mb, n_layers = guess_model_layers_and_size(model_path)

    # Если модель мультимодальная и mmproj был скачан — его тоже грузят в
    # VRAM (обычно на первый GPU), резервируем под него место заранее.
    mmproj_mb = 0
    mmproj_path_file = "./logs/mmproj_path.txt"
    if os.path.exists(mmproj_path_file):
        with open(mmproj_path_file, "r", encoding="utf-8") as f:
            mmproj_path = f.read().strip()
        if mmproj_path and os.path.exists(mmproj_path):
            mmproj_mb = os.path.getsize(mmproj_path) / (1024 * 1024)

    # Эвристика: определяем, поместится ли модель целиком на один GPU (GPU 0)
    # Резервируем VRAM под саму модель, mmproj и гипотетический максимальный KV cache (~2.5 GB)
    estimated_kv_mb = 2560
    use_single_gpu = False
    if gpus and len(gpus) > 1:
        single_gpu_usable = gpus[0]["free_memory_mb"] - VRAM_RESERVE_MB_PER_GPU
        if size_mb + mmproj_mb + estimated_kv_mb < single_gpu_usable:
            use_single_gpu = True

    if use_single_gpu:
        gpu_count = 1
        total_free_mb = gpus[0]["free_memory_mb"]
    else:
        total_free_mb = sum(g["free_memory_mb"] for g in gpus) if gpus else 0

    usable_vram_mb = max(0, total_free_mb - VRAM_RESERVE_MB_PER_GPU * gpu_count - mmproj_mb)

    # --- n_gpu_layers ---
    if n_layers and size_mb > 0:
        mb_per_layer = size_mb / n_layers
        # оставляем ~25% бюджета под KV-cache
        weight_budget_mb = usable_vram_mb * 0.75
        fit_layers = int(weight_budget_mb / mb_per_layer) if mb_per_layer > 0 else n_layers
        if fit_layers >= n_layers:
            n_gpu_layers = -1  # все слои на GPU
        else:
            n_gpu_layers = max(1, fit_layers)
    else:
        # нет метаданных — если модель заметно меньше бюджета, выгружаем всё
        n_gpu_layers = -1 if size_mb < usable_vram_mb * 0.75 else 999

    # --- ctx_size ---
    kv_budget_mb = usable_vram_mb * 0.25 if n_gpu_layers == -1 else usable_vram_mb * 0.4
    approx_ctx = int((kv_budget_mb * 1024) / BYTES_PER_TOKEN_KV_Q8)
    ctx_size = max(DEFAULT_CTX_MIN, min(DEFAULT_CTX_MAX, approx_ctx))
    # округляем до кратного 1024
    ctx_size = (ctx_size // 1024) * 1024

    ctx_size_min = cfg_server.get("ctx_size_min") or 0
    ctx_size_capped_by_vram = ctx_size
    if ctx_size_min and ctx_size < ctx_size_min:
        # Модель хочет больше контекста (напр. Qwen3.6 рекомендует 131072 для
        # thinking mode), чем реально влезает в VRAM с учётом весов. Уважаем
        # желание пользователя, но честно предупреждаем — не понижаем то,
        # что физически посчитано, а поднимаем до ctx_size_min: это может
        # привести к OOM при реальной длинной генерации, пользователь должен
        # это осознанно принять (либо использовать более лёгкий квант).
        ctx_size = ctx_size_min

    # --- batch / ubatch ---
    batch_size = 2048 if ctx_size >= 8192 else 1024
    ubatch_size = 512

    # --- threads ---
    logical = hw.get("cpu_cores_logical", 4) or 4
    threads = max(1, logical - 2)  # оставляем ядра под I/O/tunnel/python
    threads_batch = logical

    # --- tensor_split ---
    if use_single_gpu:
        tensor_split = "1"
    elif gpus and len(gpus) > 1:
        frees = [g["free_memory_mb"] for g in gpus]
        total = sum(frees) or 1
        ratios = [round(f / total, 3) for f in frees]
        tensor_split = ",".join(str(r) for r in ratios)
    else:
        tensor_split = "1"

    return {
        "n_gpu_layers": n_gpu_layers,
        "ctx_size": ctx_size,
        "ctx_size_capped_by_vram": ctx_size_capped_by_vram,
        "ctx_size_min_requested": ctx_size_min,
        "batch_size": batch_size,
        "ubatch_size": ubatch_size,
        "threads": threads,
        "threads_batch": threads_batch,
        "tensor_split": tensor_split,
        "split_mode": cfg_server.get("split_mode", "layer"),
        "flash_attn": cfg_server.get("flash_attn", True),
        "cache_type_k": cfg_server.get("cache_type_k", "q8_0"),
        "cache_type_v": cfg_server.get("cache_type_v", "q8_0"),
        "parallel": cfg_server.get("parallel", 4),
        "model_size_mb": round(size_mb, 1),
        "detected_layers": n_layers,
        "gpu_count": gpu_count,
        "usable_vram_mb": round(usable_vram_mb, 1),
    }


def main():
    import yaml

    config_path = os.environ.get("CONFIG_FILE", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    hw = load_hardware()

    model_dir = cfg["model"]["local_dir"]
    candidates = [
        os.path.join(model_dir, f) for f in os.listdir(model_dir)
        if f.endswith(".gguf")
    ] if os.path.isdir(model_dir) else []

    if not candidates:
        print("Не найден .gguf файл в models/. Сначала запустите download_model.py", file=sys.stderr)
        sys.exit(1)

    model_path = max(candidates, key=os.path.getsize)
    params = compute_params(hw, model_path, cfg["server"])

    if params["ctx_size_min_requested"] and params["ctx_size"] > params["ctx_size_capped_by_vram"]:
        print(
            f"[optimize][warn] server.ctx_size_min={params['ctx_size_min_requested']} превышает "
            f"расчётный безопасный контекст по VRAM ({params['ctx_size_capped_by_vram']}). "
            f"Используем {params['ctx_size']}, но при длинных запросах возможен CUDA OOM. "
            f"Снизьте ctx_size_min или используйте более лёгкий квант модели.",
            file=sys.stderr,
        )

    os.makedirs("./logs", exist_ok=True)
    port = cfg["server"].get("port", 8080)
    output_json = f"./logs/optimized_params_{port}.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"model_path": model_path, **params}, f, indent=2, ensure_ascii=False)

    print(json.dumps({"model_path": model_path, **params}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
