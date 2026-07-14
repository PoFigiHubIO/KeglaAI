#!/usr/bin/env bash
# =============================================================================
# install.sh — Этап 2. Установка зависимостей.
#
# Устанавливает всё необходимое для сборки llama.cpp с CUDA на Kaggle Notebook:
# cmake, ninja, gcc/g++, git, python-зависимости, huggingface_hub.
# CUDA Toolkit на образах Kaggle GPU обычно уже установлен (проверяем и
# ставим при необходимости).
#
# Особенности Kaggle:
#   - образ Docker уже содержит nvidia-driver и CUDA runtime;
#   - sudo доступен, apt работает, но сеть иногда медленная — используем
#     retries;
#   - /kaggle/working — единственная директория с сохраняемым содержимым,
#     всё остальное временное.
# =============================================================================
set -euo pipefail

log()  { echo -e "\033[1;36m[install]\033[0m $*"; }
warn() { echo -e "\033[1;33m[install][warn]\033[0m $*"; }
err()  { echo -e "\033[1;31m[install][error]\033[0m $*" >&2; }

APT_RETRY=3

apt_install() {
    local pkgs="$*"
    for i in $(seq 1 $APT_RETRY); do
        if sudo apt-get install -y --no-install-recommends $pkgs; then
            return 0
        fi
        warn "apt-get install не удался (попытка $i/$APT_RETRY), повтор через 5с..."
        sleep 5
    done
    err "Не удалось установить пакеты: $pkgs"
    return 1
}

log "Обновление списка пакетов..."
sudo apt-get update -y -qq || warn "apt-get update завершился с предупреждениями, продолжаем"

log "Установка базовых инструментов сборки (build-essential, git, curl, wget)..."
apt_install build-essential git curl wget unzip pkg-config software-properties-common

log "Установка cmake..."
if ! command -v cmake &>/dev/null || [[ "$(cmake --version | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)" < "3.21" ]]; then
    apt_install cmake
fi
cmake --version | head -1

log "Установка ninja-build..."
apt_install ninja-build
ninja --version

log "Проверка gcc/g++..."
apt_install gcc g++
gcc --version | head -1

log "Проверка CUDA Toolkit (nvcc)..."
if command -v nvcc &>/dev/null; then
    nvcc --version | tail -1
else
    warn "nvcc не найден в PATH. Проверяем стандартные пути установки CUDA..."
    for CUDA_HOME_CANDIDATE in /usr/local/cuda /usr/local/cuda-12.* /usr/local/cuda-11.*; do
        if [[ -x "$CUDA_HOME_CANDIDATE/bin/nvcc" ]]; then
            export PATH="$CUDA_HOME_CANDIDATE/bin:$PATH"
            export LD_LIBRARY_PATH="$CUDA_HOME_CANDIDATE/lib64:${LD_LIBRARY_PATH:-}"
            log "Найден CUDA Toolkit в $CUDA_HOME_CANDIDATE"
            break
        fi
    done
    if ! command -v nvcc &>/dev/null; then
        warn "CUDA Toolkit не обнаружен. На стандартном образе Kaggle GPU nvcc обычно уже есть."
        warn "Сборка llama.cpp с GGML_CUDA всё равно попробует найти CUDA через CMake FindCUDAToolkit."
    fi
fi

# Исправление для CMake CUDAToolkit: на Kaggle часто отсутствует dev-симлинк libcuda.so
log "Проверка наличия dev-симлинка libcuda.so для CMake..."
if ! find /usr/lib /usr/local/lib -name "libcuda.so" 2>/dev/null | grep -q "libcuda.so"; then
    log "libcuda.so не найден в стандартных путях поиска линковщика."
    # Ищем libcuda.so.1
    LIBCUDA_1=$(find /usr/lib -name "libcuda.so.1" 2>/dev/null | head -n 1)
    if [[ -n "$LIBCUDA_1" ]]; then
        DIR=$(dirname "$LIBCUDA_1")
        log "Найден libcuda.so.1 в $DIR. Создаём симлинк libcuda.so..."
        sudo ln -sf libcuda.so.1 "$DIR/libcuda.so"
    else
        # fallback: пробуем найти stubs в CUDA
        STUB_CUDA=$(find /usr/local/cuda* -name "libcuda.so" 2>/dev/null | head -n 1)
        if [[ -n "$STUB_CUDA" ]]; then
            log "Найден stub libcuda.so в $STUB_CUDA. Создаём симлинк в /usr/lib/x86_64-linux-gnu/..."
            sudo mkdir -p /usr/lib/x86_64-linux-gnu
            sudo ln -sf "$STUB_CUDA" /usr/lib/x86_64-linux-gnu/libcuda.so
        else
            warn "Ни libcuda.so.1, ни stub libcuda.so не найдены. Сборка llama.cpp может дать сбой."
        fi
    fi
else
    log "libcuda.so найден, всё в порядке."
fi

log "Проверка драйвера NVIDIA (nvidia-smi)..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    err "nvidia-smi не найден. Убедитесь, что в Kaggle Notebook включён GPU-аксселератор:"
    err "  Settings -> Accelerator -> GPU T4 x2"
    exit 1
fi

log "Установка Python-зависимостей из requirements.txt..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

log "Проверка/установка Node.js (нужен только для localtunnel, опционально)..."
if ! command -v node &>/dev/null; then
    warn "Node.js не найден, localtunnel будет недоступен (это опционально, по умолчанию используется cloudflared)."
fi

mkdir -p models logs bin mcp

log "Установка завершена успешно."
log "Далее выполните: bash build.sh"
