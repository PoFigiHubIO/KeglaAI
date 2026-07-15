#!/usr/bin/env bash
# =============================================================================
# build.sh — Этап 3. Сборка последней версии llama.cpp с поддержкой CUDA.
#
# Собирает llama-server (включает встроенный OpenAI-compatible API и
# современный Web UI), llama-cli, llama-bench.
#
# УСКОРЕНИЕ ПОВТОРНЫХ ЗАПУСКОВ (prebuilt):
#   Сборка на Kaggle (обычно 2 физических ядра CPU) занимает 20-60+ минут.
#   Чтобы не ждать это каждый раз, скрипт:
#     1) при старте ищет готовый архив llama-cpp-prebuilt-*.tar.gz
#        (в /kaggle/input/*/  — т.е. в подключённых Kaggle Dataset — или в
#        ./prebuilt/), пробует его распаковать и ЗАПУСТИТЬ (--version).
#        Если бинарник реально запускается — полная сборка пропускается.
#     2) после УСПЕШНОЙ полной сборки автоматически упаковывает результат в
#        /kaggle/working/prebuilt/llama-cpp-prebuilt-<arch>-cuda<ver>.tar.gz
#        — этот файл виден в панели Output ноутбука Kaggle, его можно
#        скачать и загрузить как новый Kaggle Dataset для следующих запусков.
#
#   Важно: у Kaggle включена опция "Always use latest environment" — базовый
#   Docker-образ (версии CUDA/драйвера/glibc) может со временем измениться.
#   Поэтому восстановленный бинарник ВСЕГДА проверяется реальным запуском
#   (--version), а не только сверкой версий в manifest.json. Если проверка
#   не прошла — скрипт просто пересоберёт всё с нуля, без ручного вмешательства.
#
# Ключевые флаги CMake:
#   GGML_CUDA=ON          — включает бэкенд CUDA (cuBLAS используется внутри
#                            автоматически для GEMM-операций)
#   GGML_CUDA_F16=ON      — использовать fp16 для промежуточных вычислений
#                            на GPU (быстрее на Tensor Cores T4)
#   GGML_NATIVE=OFF       — не завязываемся на -march=native (Kaggle CPU
#                            в контейнере может отличаться от хоста сборки)
#   CMAKE_CUDA_ARCHITECTURES=75  — Tesla T4 = архитектура Turing = sm_75.
#                            Сборка только под нужную архитектуру ускоряет
#                            компиляцию в разы по сравнению с "all".
#   LLAMA_CURL=ON         — llama-server сможет сам скачивать модели по URL
#   LLAMA_BUILD_SERVER=ON — собрать llama-server (Web UI + OpenAI API)
#
# Flash Attention реализована в ggml-cuda и включается не флагом сборки,
# а рантайм-параметром `--flash-attn` при запуске llama-server (см.
# start_server.sh) — она доступна автоматически, если бинарник собран с CUDA.
# =============================================================================
set -euo pipefail

log()  { echo -e "\033[1;36m[build]\033[0m $*"; }
warn() { echo -e "\033[1;33m[build][warn]\033[0m $*"; }
err()  { echo -e "\033[1;31m[build][error]\033[0m $*" >&2; }

REPO_URL="${LLAMA_CPP_REPO_URL:-https://github.com/ggml-org/llama.cpp.git}"
BRANCH="${LLAMA_CPP_BRANCH:-master}"
BUILD_DIR="${LLAMA_CPP_DIR:-./llama.cpp}"
CUDA_ARCH="${CUDA_ARCH:-75}"     # Tesla T4 (Turing, sm_75)
JOBS="${BUILD_JOBS:-$(nproc)}"
PREBUILT_OUT_DIR="/kaggle/working/prebuilt"
[[ -d "/kaggle/working" ]] || PREBUILT_OUT_DIR="./prebuilt"  # локальный запуск вне Kaggle

# =============================================================================
# Шаг 0: попытка восстановить готовую сборку вместо компиляции с нуля
# =============================================================================
read_config_yaml() {
    python3 -c "import yaml, os; d=yaml.safe_load(open(os.environ.get('CONFIG_FILE', 'config.yaml'))); print(d.get('llama_cpp',{}).get('$1','') or '')" 2>/dev/null || true
}

maybe_download_from_gdrive() {
    local file_id
    file_id=$(read_config_yaml "prebuilt_gdrive_file_id")
    if [[ -z "$file_id" ]]; then
        return 1
    fi
    if compgen -G "./prebuilt/llama-cpp-prebuilt*.tar.gz" > /dev/null; then
        log "В ./prebuilt уже есть архив, скачивание из Google Drive пропущено."
        return 0
    fi

    log "config.yaml: llama_cpp.prebuilt_gdrive_file_id задан — качаем готовую сборку из Google Drive..."
    mkdir -p ./prebuilt
    python3 -c "
import gdown, sys
try:
    gdown.download(id='${file_id}', output='./prebuilt/llama-cpp-prebuilt-gdrive.tar.gz', quiet=False)
except Exception as e:
    print(f'gdown error: {e}', file=sys.stderr)
    sys.exit(1)
" || { warn "Не удалось скачать prebuilt-архив из Google Drive (проверьте, что доступ по ссылке 'Anyone with the link' включён) — соберём с нуля."; return 1; }

    if [[ ! -s "./prebuilt/llama-cpp-prebuilt-gdrive.tar.gz" ]]; then
        warn "Скачанный файл из Google Drive пустой/повреждён — соберём с нуля."
        rm -f "./prebuilt/llama-cpp-prebuilt-gdrive.tar.gz"
        return 1
    fi
    log "Скачано: ./prebuilt/llama-cpp-prebuilt-gdrive.tar.gz ($(du -h ./prebuilt/llama-cpp-prebuilt-gdrive.tar.gz | cut -f1))"
    return 0
}

maybe_download_from_gdrive || true

try_restore_prebuilt() {
    local candidates=()
    while IFS= read -r -d '' f; do candidates+=("$f"); done < <(
        find /kaggle/input ./prebuilt -maxdepth 3 -iname "llama-cpp-prebuilt*.tar.gz" -print0 2>/dev/null
    )

    if [[ ${#candidates[@]} -eq 0 ]]; then
        log "Готовый архив llama-cpp-prebuilt*.tar.gz не найден (ни в /kaggle/input, ни в ./prebuilt) — соберём с нуля."
        return 1
    fi

    local archive="${candidates[0]}"
    log "Найден готовый архив: $archive — пробуем восстановить сборку..."

    local tmp_extract
    tmp_extract=$(mktemp -d)
    if ! tar -xzf "$archive" -C "$tmp_extract"; then
        warn "Не удалось распаковать $archive — соберём с нуля."
        rm -rf "$tmp_extract"
        return 1
    fi

    if [[ -f "$tmp_extract/manifest.json" ]]; then
        log "manifest.json восстановленной сборки:"
        cat "$tmp_extract/manifest.json"
    fi

    if [[ ! -x "$tmp_extract/bin/llama-server" ]]; then
        warn "В архиве нет исполняемого bin/llama-server — соберём с нуля."
        rm -rf "$tmp_extract"
        return 1
    fi

    mkdir -p "$BUILD_DIR/build"
    rm -rf "$BUILD_DIR/build/bin"
    cp -r "$tmp_extract/bin" "$BUILD_DIR/build/bin"
    chmod -R u+x "$BUILD_DIR/build/bin"
    rm -rf "$tmp_extract"

    log "Проверяем, что восстановленный бинарник реально запускается на этом железе/CUDA..."
    if "$BUILD_DIR/build/bin/llama-server" --version &>/tmp/prebuilt_version_check.log; then
        log "Восстановленная сборка рабочая! Полная компиляция пропущена."
        cat /tmp/prebuilt_version_check.log
        return 0
    else
        warn "Восстановленный бинарник не запустился (несовместимая версия CUDA/драйвера/glibc после обновления Kaggle-образа)."
        warn "Лог проверки:"
        cat /tmp/prebuilt_version_check.log || true
        warn "Соберём с нуля — это надёжный запасной путь."
        rm -rf "$BUILD_DIR/build/bin"
        return 1
    fi
}

if try_restore_prebuilt; then
    log "Готово (восстановлено из prebuilt-архива). Далее: python download_model.py && bash start_server.sh"
    exit 0
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

# =============================================================================
# Шаг 1-4: полная сборка с нуля (обычный путь)
# =============================================================================
if [[ -d "$BUILD_DIR/.git" ]]; then
    log "Репозиторий llama.cpp уже существует, обновляем..."
    git -C "$BUILD_DIR" fetch --depth 1 origin "$BRANCH"
    git -C "$BUILD_DIR" reset --hard "origin/$BRANCH"
else
    log "Клонирование llama.cpp (branch=$BRANCH)..."
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$BUILD_DIR"
fi

cd "$BUILD_DIR"
COMMIT=$(git rev-parse --short HEAD)
log "Собираем llama.cpp @ $COMMIT"

log "Конфигурация CMake (GGML_CUDA=ON, arch=sm_${CUDA_ARCH}, Flash Attention доступна в рантайме)..."
cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_F16=ON \
    -DGGML_NATIVE=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DLLAMA_CURL=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DGGML_OPENMP=ON

log "Компиляция (jobs=$JOBS, на 2-ядерном Kaggle CPU это обычно 20-60+ минут)..."
cmake --build build --config Release -j "$JOBS"

log "Проверка бинарников..."
BIN_DIR="build/bin"
BUILD_OK=true
for bin in llama-server llama-cli llama-bench llama-quantize; do
    if [[ -x "$BIN_DIR/$bin" ]]; then
        log "  OK: $BIN_DIR/$bin"
    else
        err "  Отсутствует ожидаемый бинарник: $BIN_DIR/$bin"
        BUILD_OK=false
    fi
done

log "Быстрая проверка сборки CUDA..."
./build/bin/llama-cli --version || true

cd ..
log "Сборка завершена. Бинарники находятся в ${BUILD_DIR}/build/bin/"

# =============================================================================
# Шаг 5: упаковка результата для повторного использования в след. раз
# =============================================================================
if [[ "$BUILD_OK" == "true" ]]; then
    log "Упаковываем сборку для повторного использования (чтобы не ждать компиляцию в следующий раз)..."
    mkdir -p "$PREBUILT_OUT_DIR"

    CUDA_VER=$(nvcc --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' || echo "unknown")
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    ARCHIVE_NAME="llama-cpp-prebuilt-sm${CUDA_ARCH}-cuda${CUDA_VER}.tar.gz"

    TMP_PACK=$(mktemp -d)
    mkdir -p "$TMP_PACK/bin"
    cp -r "${BUILD_DIR}/build/bin/." "$TMP_PACK/bin/"

    cat > "$TMP_PACK/manifest.json" << EOF
{
  "llama_cpp_commit": "${COMMIT}",
  "llama_cpp_branch": "${BRANCH}",
  "cuda_arch": "${CUDA_ARCH}",
  "cuda_version_build": "${CUDA_VER}",
  "driver_version_build": "${DRIVER_VER}",
  "built_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

    tar -czf "${PREBUILT_OUT_DIR}/${ARCHIVE_NAME}" -C "$TMP_PACK" bin manifest.json
    rm -rf "$TMP_PACK"

    log "Готовая сборка сохранена: ${PREBUILT_OUT_DIR}/${ARCHIVE_NAME}"
    log ""
    log "Чтобы не ждать компиляцию в следующий раз, есть два варианта:"
    log ""
    log "  Вариант А (Google Drive, рекомендуется):"
    log "    1) Скачайте этот файл из панели Output ноутбука Kaggle на свой ПК."
    log "    2) Загрузите его в папку KeglyaAI (или любую) в вашем Google Drive."
    log "    3) ПКМ -> Share -> Anyone with the link -> скопируйте ссылку,"
    log "       достаньте ID файла (часть между /file/d/ и /view)."
    log "    4) Впишите его в config.yaml -> llama_cpp.prebuilt_gdrive_file_id."
    log "    5) В следующий раз просто Run All — build.sh сам скачает архив"
    log "       из Drive и восстановит сборку за секунды."
    log ""
    log "  Вариант Б (Kaggle Dataset):"
    log "    1) Тот же файл загрузите как новый Kaggle Dataset."
    log "    2) Подключите датасет к ноутбуку (Add Input)."
    log "    3) build.sh сам найдёт архив в /kaggle/input/."
    log ""
    log "  В обоих случаях: если Kaggle обновит образ и старый бинарник"
    log "  перестанет запускаться — скрипт это обнаружит и молча пересоберёт всё заново."
fi

log "Далее: python download_model.py && bash start_server.sh"
