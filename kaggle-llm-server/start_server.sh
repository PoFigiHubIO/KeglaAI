#!/usr/bin/env bash
# =============================================================================
# start_server.sh — Этап 6-8. Запуск llama-server:
#   - OpenAI-compatible API (/v1/chat/completions, /v1/completions, /v1/models)
#   - streaming, function calling, JSON mode — встроены в llama-server
#   - современный Web UI (встроен в llama-server, флаг --webui включён по
#     умолчанию с llama.cpp #7935+; явно не отключаем)
#
# Параметры n_gpu_layers/ctx_size/batch_size/... берутся из
# ./logs/optimized_params.json, который генерирует scripts/optimize.py
# (Этап 5), поэтому этот скрипт нужно запускать ПОСЛЕ optimize.py.
# =============================================================================
set -euo pipefail

log()  { echo -e "\033[1;36m[server]\033[0m $*"; }
err()  { echo -e "\033[1;31m[server][error]\033[0m $*" >&2; }

PARAMS_JSON="./logs/optimized_params.json"
CONFIG_YAML="./config.yaml"
LLAMA_BIN="./llama.cpp/build/bin/llama-server"

if [[ ! -f "$PARAMS_JSON" ]]; then
    err "$PARAMS_JSON не найден. Сначала выполните: python scripts/optimize.py"
    exit 1
fi

if [[ ! -x "$LLAMA_BIN" ]]; then
    # Файл мог остаться от предыдущей сессии (Kaggle Persistence) или быть
    # восстановлен из prebuilt-архива без сохранения бита исполняемости.
    if [[ -f "$LLAMA_BIN" ]]; then
        chmod +x "$LLAMA_BIN" 2>/dev/null || true
    fi
fi

if [[ ! -x "$LLAMA_BIN" ]]; then
    err "$LLAMA_BIN не найден. Сначала выполните: bash build.sh"
    exit 1
fi

# --- Читаем JSON через python (без внешних зависимостей типа jq) ---
read_json() {
    python3 -c "import json,sys; d=json.load(open('$PARAMS_JSON')); print(d.get('$1',''))"
}
read_yaml() {
    python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d['server'].get('$1',''))"
}

MODEL_PATH=$(read_json model_path)
N_GPU_LAYERS=$(read_json n_gpu_layers)
CTX_SIZE=$(read_json ctx_size)
BATCH_SIZE=$(read_json batch_size)
UBATCH_SIZE=$(read_json ubatch_size)
THREADS=$(read_json threads)
THREADS_BATCH=$(read_json threads_batch)
TENSOR_SPLIT=$(read_json tensor_split)
CACHE_TYPE_K=$(read_json cache_type_k)
CACHE_TYPE_V=$(read_json cache_type_v)
PARALLEL=$(read_json parallel)

HOST=$(read_yaml host)
PORT=$(read_yaml port)
API_KEY=$(read_yaml api_key)
JINJA=$(read_yaml jinja)
WEBUI_MCP_PROXY=$(read_yaml webui_mcp_proxy)

mkdir -p logs

# --- Проверка поддерживаемых флагов через --help собранного бинарника ---
# llama.cpp — быстро развивающийся проект, CLI-флаги переименовываются/
# убираются между версиями (build.sh всегда собирает актуальный master).
# Чтобы не падать на "invalid argument" из-за такого дрейфа API, читаем
# --help один раз и решаем, что добавлять в команду запуска.
HELP_OUTPUT=$("$LLAMA_BIN" --help 2>&1 || true)

flag_supported() {
    echo "$HELP_OUTPUT" | grep -q -- "$1"
}

# --- Определяем реальное имя флага CORS-прокси для MCP Servers в WebUI ---
# Флаг менял имя между версиями llama.cpp (--webui-mcp-proxy / --ui-mcp-proxy).
MCP_PROXY_FLAG=""
if [[ "$WEBUI_MCP_PROXY" == "True" || "$WEBUI_MCP_PROXY" == "true" ]]; then
    if flag_supported "--webui-mcp-proxy"; then
        MCP_PROXY_FLAG="--webui-mcp-proxy"
    elif flag_supported "--ui-mcp-proxy"; then
        MCP_PROXY_FLAG="--ui-mcp-proxy"
    else
        log "server.webui_mcp_proxy включён, но собранный llama-server не поддерживает ни --webui-mcp-proxy, ни --ui-mcp-proxy (обновите llama.cpp через bash build.sh). Пропускаем флаг."
    fi
fi

MMPROJ_PATH=""
if [[ -f "./logs/mmproj_path.txt" ]]; then
    MMPROJ_PATH=$(cat ./logs/mmproj_path.txt)
    if [[ ! -f "$MMPROJ_PATH" ]]; then
        err "mmproj указан в logs/mmproj_path.txt, но файл не найден: $MMPROJ_PATH — vision будет отключён"
        MMPROJ_PATH=""
    fi
fi

log "Модель:        $MODEL_PATH"
log "GPU layers:    $N_GPU_LAYERS"
log "Context size:  $CTX_SIZE"
log "Batch/UBatch:  $BATCH_SIZE / $UBATCH_SIZE"
log "Threads:       $THREADS (batch: $THREADS_BATCH)"
log "Tensor split:  $TENSOR_SPLIT (2x Tesla T4)"
log "KV cache type: K=$CACHE_TYPE_K V=$CACHE_TYPE_V"
log "Parallel slots (continuous batching): $PARALLEL"
log "Jinja chat template: $JINJA"
if [[ -n "$MMPROJ_PATH" ]]; then
    log "Vision (mmproj):    $MMPROJ_PATH"
else
    log "Vision (mmproj):    выключен"
fi
if [[ -n "$MCP_PROXY_FLAG" ]]; then
    log "MCP CORS proxy:      включён ($MCP_PROXY_FLAG) — MCP Servers в WebUI смогут подключаться к серверам на других доменах"
else
    log "MCP CORS proxy:      выключен"
fi

ARGS=(
    --model "$MODEL_PATH"
    --host "$HOST"
    --port "$PORT"
    --n-gpu-layers "$N_GPU_LAYERS"
    --ctx-size "$CTX_SIZE"
    --batch-size "$BATCH_SIZE"
    --ubatch-size "$UBATCH_SIZE"
    --threads "$THREADS"
    --threads-batch "$THREADS_BATCH"
    --mmap
    --cache-type-k "$CACHE_TYPE_K"
    --cache-type-v "$CACHE_TYPE_V"
    --parallel "$PARALLEL"
    --cont-batching
    --split-mode layer
    --tensor-split "$TENSOR_SPLIT"
)

# --- Flash Attention: в разных версиях llama.cpp это то булев флаг
# (--flash-attn), то флаг со значением (--flash-attn on|off|auto).
# Определяем по тексту --help, что ожидает конкретно эта сборка.
if echo "$HELP_OUTPUT" | grep -A 2 -i "flash-attn" | grep -qE "(on|off|auto|FA_TYPE)"; then
    ARGS+=(--flash-attn on)
elif flag_supported "--flash-attn"; then
    ARGS+=(--flash-attn)
else
    log "Собранный llama-server не поддерживает --flash-attn (неожиданно для сборки с GGML_CUDA) — пропускаем флаг, будет использовано поведение по умолчанию."
fi

if flag_supported "--metrics"; then
    ARGS+=(--metrics)
fi

if flag_supported "--log-format"; then
    ARGS+=(--log-format text)
fi

if [[ "$JINJA" == "True" || "$JINJA" == "true" ]]; then
    if flag_supported "--jinja"; then
        ARGS+=(--jinja)
    else
        log "server.jinja включён, но --jinja не поддерживается собранной версией llama-server (возможно, jinja-шаблон уже используется по умолчанию) — пропускаем флаг."
    fi
fi

if [[ -n "$MMPROJ_PATH" ]]; then
    ARGS+=(--mmproj "$MMPROJ_PATH")
fi

if [[ -n "$MCP_PROXY_FLAG" ]]; then
    ARGS+=("$MCP_PROXY_FLAG")
fi

if [[ -n "$API_KEY" ]]; then
    ARGS+=(--api-key "$API_KEY")
    log "API-key защита: включена"
else
    log "API-key защита: выключена (server.api_key пуст в config.yaml)"
fi

log "Запуск llama-server..."
log "Команда: $LLAMA_BIN ${ARGS[*]}"

nohup "$LLAMA_BIN" "${ARGS[@]}" > logs/llama-server.log 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > logs/llama-server.pid

log "llama-server запущен, PID=$SERVER_PID"
log "Ожидание готовности сервера..."

for i in $(seq 1 60); do
    if curl -s "http://127.0.0.1:${PORT}/health" | grep -q "ok"; then
        log "Сервер готов! http://127.0.0.1:${PORT}"
        log "Web UI:  http://127.0.0.1:${PORT}/"
        log "API:     http://127.0.0.1:${PORT}/v1/chat/completions"
        exit 0
    fi
    sleep 2
done

err "Сервер не ответил на /health за 120 секунд. Смотрите logs/llama-server.log"
tail -n 50 logs/llama-server.log
exit 1
