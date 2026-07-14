# kaggle-llm-server

Production-ready решение для запуска любой GGUF-модели на **Kaggle Notebook
(2× Tesla T4)** с OpenAI-совместимым API, современным Web UI, поддержкой
MCP и публичным доступом через туннель. Работает по принципу **Run All**:
никаких ручных действий после запуска ноутбука.

## Содержание

- [Быстрый старт](#быстрый-старт)
- [Структура проекта](#структура-проекта)
- [Этапы работы](#этапы-работы)
- [Оптимизация под 2×T4 — объяснение параметров](#оптимизация-под-2t4)
- [OpenAI API](#openai-api)
- [Web UI](#web-ui)
- [MCP](#mcp)
- [Подключение из VS Code](#vs-code)
- [Тестирование](#тестирование)
- [Ограничения Kaggle](#ограничения-kaggle)
- [Troubleshooting](#troubleshooting)

---

## Ускорение повторных запусков: кэшируем сборку llama.cpp

Долгий этап при **первом** запуске — это компиляция `llama.cpp` в `build.sh`
(Этап 3): на Kaggle обычно только 2 физических ядра CPU, поэтому сборка с
CUDA занимает 20-60+ минут. Чтобы не ждать это каждый раз, есть два
равнозначных варианта — Google Drive (проще всего для одного человека) или
Kaggle Dataset.

### Вариант А — Google Drive (рекомендуется)

**После первого успешного `Run All`:**

1. `build.sh` сам упаковывает готовые бинарники в
   `/kaggle/working/prebuilt/llama-cpp-prebuilt-sm75-cudaXX.X.tar.gz`
   (виден в панели **Output** ноутбука Kaggle, справа) — скачайте его на ПК.
2. Создайте папку (например `KeglyaAI`) в своём Google Drive и загрузите
   туда этот `.tar.gz`.
3. ПКМ по файлу → **Share** → **General access → Anyone with the link**.
4. Скопируйте ссылку вида
   `https://drive.google.com/file/d/1AbCDeFGhIJKLmnoPQRstuVWxyz1234/view`
   и возьмите из неё ID файла — часть между `/file/d/` и `/view`.
5. Впишите его в `config.yaml`:
   ```yaml
   llama_cpp:
     prebuilt_gdrive_file_id: "1AbCDeFGhIJKLmnoPQRstuVWxyz1234"
   ```

**В следующих сессиях** — просто **Run All**. `build.sh` сам скачает архив
по этому ID (через `gdown`) в `./prebuilt/`, распакует и **реальным
запуском** (`llama-server --version`) проверит совместимость с текущим
образом Kaggle. Если да — сборка занимает секунды вместо часа. Ничего
руками пересобирать/переподключать не нужно — только один раз обновить
`prebuilt_gdrive_file_id`, если вы захотите заменить файл на более свежую
сборку.

### Вариант Б — Kaggle Dataset

1. Тот же `.tar.gz` из `/kaggle/working/prebuilt/` загрузите на
   [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**.
2. В следующих сессиях подключите датасет через **Add Input**.
3. `build.sh` сам найдёт архив в `/kaggle/input/*/` — приоритет отдаётся
   именно Dataset-у, если он подключён (Google Drive используется только
   если локальной/датасетной копии не нашлось).

### В обоих случаях

**Важный нюанс:** в Session options у вас включено *"Always use latest
environment"* — базовый Docker-образ Kaggle (CUDA Toolkit, драйвер, glibc)
может со временем измениться. Именно поэтому проверка prebuilt-бинарника —
не сверка версий в файле, а **реальный запуск** `--version`: если старый
бинарник несовместим с обновлённым образом, скрипт сам, без вашего
участия, откатится на полную пересборку с нуля (и снова упакует свежий
`.tar.gz` для следующего раза). То же самое произойдёт, если вы просто
**возобновите ту же сессию** через Kaggle Persistence ("Variables and
Files") — `start.py` перед пропуском сборки тоже проверяет старый бинарник
реальным запуском (и на всякий случай восстанавливает `chmod +x`, если
права слетели при сохранении/восстановлении сессии).

> Модель (`.gguf`) кэшировать таким же способом не обязательно —
> `download_model.py` и так пропускает повторное скачивание, если файл уже
> есть в `models/`. Но если хотите не тратить трафик/время на скачивание с
> HuggingFace при каждой новой сессии — модель тоже можно положить в ту же
> папку Google Drive и указать `model.source: gdrive` +
> `model.gdrive_file_id` в `config.yaml` (это уже поддерживается
> `download_model.py` из коробки).

---

## Быстрый старт

### В Kaggle Notebook

1. Создайте новый Kaggle Notebook, **Settings → Accelerator → GPU T4 x2**,
   **Internet → On**.
2. Загрузите этот проект (см. `notebooks/kaggle_run_all.ipynb` — готовый
   ноутбук) или склонируйте его в первой ячейке.
3. При необходимости отредактируйте `config.yaml` (раздел `model:`) —
   укажите нужный `repo_id`/`filename` GGUF-модели с HuggingFace.
4. **Run All**. Через 10-20 минут (в основном время сборки и скачивания
   модели) в выводе появятся публичные ссылки на API и Web UI.

### Локально / вручную (по шагам)

```bash
cd kaggle-llm-server
bash install.sh
bash build.sh
python download_model.py
python scripts/optimize.py
bash start_server.sh
python scripts/tunnel.py --provider cloudflared --port 8080
```

Либо всё сразу:

```bash
python start.py
```

---

## Структура проекта

```
kaggle-llm-server/
│
├── install.sh              # Этап 2: установка системных и Python зависимостей
├── build.sh                 # Этап 3: сборка llama.cpp с CUDA
├── start.py                  # Оркестратор: запускает всё по порядку (Run All)
├── start_server.sh            # Этап 6-8: запуск llama-server (API + Web UI)
├── download_model.py           # Этап 4: загрузка GGUF модели
├── config.yaml                  # Центральный конфиг всего проекта
├── requirements.txt              # Python-зависимости
│
├── scripts/
│   ├── hardware_check.py          # Этап 1: анализ CPU/RAM/GPU/CUDA/диска
│   ├── optimize.py                 # Этап 5: автоподбор параметров под 2xT4
│   ├── tunnel.py                    # Этап 7: Cloudflare/Pinggy/LT/ngrok
│   └── mcp_agent.py                  # Этап 9: пример agent loop с MCP tools
│
├── vscode/
│   ├── continue_config.json           # Этап 10: конфиг для Continue
│   ├── cline_config.json               # Этап 10: конфиг для Cline
│   ├── roo_code_config.json             # Этап 10: конфиг для Roo Code
│   ├── copilot_notes.md                  # Этап 10: заметки по Copilot
│   ├── example_openai_python.py           # Пример OpenAI SDK (Python)
│   ├── example_openai_js.js                # Пример OpenAI SDK (JS)
│   └── generated/                            # Автогенерируются start.py с реальным URL
│
├── mcp/
│   └── mcp_servers.json                        # Этап 9: конфиг MCP-серверов
│
├── notebooks/
│   └── kaggle_run_all.ipynb                       # Этап 14: готовый Kaggle Notebook
│
├── models/                                          # Кэш GGUF-моделей (генерируется)
├── logs/                                              # Логи и артефакты (генерируется)
└── README.md
```

---

## Этапы работы

| # | Этап | Файл | Что делает |
|---|------|------|------------|
| 1 | Анализ окружения | `scripts/hardware_check.py` | CPU, RAM, GPU x2, CUDA, драйвер, диск → таблица + `logs/hardware_report.json` |
| 2 | Установка | `install.sh` | cmake, ninja, gcc/g++, git, python-зависимости |
| 3 | Сборка | `build.sh` | Клонирует и собирает последний llama.cpp с `GGML_CUDA=ON`, `sm_75` |
| 4 | Загрузка модели | `download_model.py` | HuggingFace / Google Drive / прямая ссылка, кэш, sha256 |
| 5 | Оптимизация | `scripts/optimize.py` | Подбирает `n_gpu_layers`, `ctx_size`, `batch`, `threads`, `tensor_split` |
| 6-8 | Запуск сервера | `start_server.sh` | `llama-server` с OpenAI API + Web UI |
| 7 | Туннель | `scripts/tunnel.py` | cloudflared / pinggy / localtunnel / ngrok |
| 9 | MCP | `mcp/mcp_servers.json`, `scripts/mcp_agent.py` | Конфиг MCP-серверов + пример agent loop |
| 10 | VS Code | `vscode/*.json` | Готовые конфиги Continue/Cline/Roo Code |

---

## Оптимизация под 2×T4

`scripts/optimize.py` вычисляет параметры на основе реальных данных
`nvidia-smi` (а не хардкода), но логика опирается на характеристики
Tesla T4 (16 GB VRAM, Turing, без поддержки bf16, хорошая fp16/int8
Tensor Core производительность):

- **`n_gpu_layers`** — сколько слоёв модели выгружается на GPU. Если
  суммарная VRAM обеих карт (минус резерв ~1.2 GB/карту под CUDA context)
  вмещает все веса модели с запасом под KV-cache — используется `-1`
  (все слои на GPU, максимальная скорость). Для больших моделей (30B+ в
  высокой битности) число слоёв уменьшается пропорционально.

- **`ctx_size`** — размер контекста подбирается так, чтобы KV-cache (при
  квантовании `q8_0`) не вытеснил веса модели из VRAM. Диапазон
  4096–32768, с округлением до кратного 1024.

- **`tensor_split`** — распределение слоёв между GPU 0 и GPU 1
  пропорционально их **свободной** памяти (обычно `"1,1"` — карты
  одинаковые). Используется `--split-mode layer`: каждый GPU считает
  свой диапазон слоёв последовательно (pipeline), что для 2×T4 обычно
  эффективнее, чем `row`-split, из-за отсутствия NVLink между T4 в
  Kaggle (связь только через PCIe).

- **`batch_size` / `ubatch_size`** — 2048/512 при большом контексте,
  иначе 1024/512. Больший batch увеличивает throughput при параллельных
  запросах (continuous batching) ценой VRAM.

- **`threads` / `threads_batch`** — `логические_ядра - 2` (резерв под
  Python/tunnel/ОС) для decode, все ядра для prompt-processing.

- **Flash Attention** (`--flash-attn`) — включена всегда: собранный с
  `GGML_CUDA=ON` бинарник поддерживает её нативно, она снижает расход
  VRAM на KV-cache и ускоряет prompt processing без потери качества.

- **`--cont-batching` + `--parallel N`** — continuous batching позволяет
  обрабатывать несколько одновременных запросов (VS Code + Web UI +
  API-клиенты) без блокировки друг друга, эффективно используя простаивающие
  такты GPU между шагами декодирования разных запросов.

- **`mmap` включён** (быстрая загрузка модели с диска, ОС сама управляет
  страницами), **`mlock` выключен** по умолчанию (на Kaggle обычно нет
  смысла блокировать страницы в RAM — веса и так почти полностью на GPU).

---

## OpenAI API

`llama-server` предоставляет полностью совместимый с OpenAI SDK API:

- `POST /v1/chat/completions` — чат, включая `stream=true`, `tools`
  (function calling), `response_format={"type":"json_object"}`
- `POST /v1/completions` — legacy completion API
- `GET  /v1/models` — список загруженных моделей
- `GET  /health`, `GET  /metrics` — служебные эндпоинты

Примеры — `vscode/example_openai_python.py`, `vscode/example_openai_js.js`.

```bash
curl https://<PUBLIC_URL>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local-model","messages":[{"role":"user","content":"Привет!"}],"stream":false}'
```

---

## Vision / мультимодальные модели (mmproj)

Если GGUF-модель мультимодальная (есть отдельный projector-файл на странице
репозитория, обычно `mmproj-*.gguf`), включите это в `config.yaml`:

```yaml
model:
  mmproj_enabled: true
  mmproj_filename: "mmproj-<имя-модели>-f16.gguf"
```

`download_model.py` скачает его вторым файлом тем же источником, что и
основную модель, `scripts/optimize.py` заранее зарезервирует под него VRAM,
а `start_server.sh` автоматически подключит его через `--mmproj`. Если
vision не нужен — оставьте `mmproj_enabled: false`.

> На данный момент автоскачивание mmproj поддержано только для
> `source: huggingface`. Для `gdrive`/`direct_url` укажите
> `mmproj_gdrive_file_id` / `mmproj_direct_url` соответственно — либо
> скачайте mmproj-файл вручную в `models/`.

## Chat-шаблон модели (--jinja)

Некоторые модели (в т.ч. Qwen3.6 и его файнтюны) требуют флаг `--jinja`,
чтобы `llama-server` использовал встроенный в GGUF jinja-шаблон чата вместо
встроенного упрощённого — это важно для корректной работы tool-calling и
"thinking mode". Управляется полем `server.jinja` в `config.yaml`
(включено по умолчанию).

Если карточка модели рекомендует большой контекст (например 128K для
сохранения thinking-режима), можно задать нижнюю границу:

```yaml
server:
  ctx_size_min: 131072
```

`scripts/optimize.py` попытается её удовлетворить, но **не может обойти
физический лимит VRAM** — если весов модели + запрошенный контекст не
помещаются в 2×T4 (32 GB), скрипт всё равно выставит запрошенное значение
и выведет предупреждение о риске `CUDA out of memory` при длинной генерации.
В таком случае либо используйте более лёгкий квант, либо оставьте
`ctx_size_min: 0` (авто) и полагайтесь на автоподбор.

---

## Web UI

Встроенный Web UI `llama-server` (доступен по адресу `/`) включает: чат,
историю сообщений, настройку параметров генерации (temperature, top_p,
top_k, repeat_penalty...), переключение system-промпта, потоковую
генерацию, рендеринг Markdown и подсветку синтаксиса кода, а для
мультимодальных GGUF (LLaVA-подобные) — загрузку изображений в чат.

---

## MCP

Начиная с недавних версий llama.cpp, встроенный **WebUI сам является MCP-клиентом**
(панель "MCP Servers" в сайдбаре / Settings → MCP). Это работает так:

- MCP-клиент выполняется **в браузере** (не на стороне C++ `llama-server`):
  подключается к MCP-серверу (WebSocket → StreamableHTTP → SSE), передаёт
  список инструментов модели и сам гоняет агентный цикл tool-call → результат
  → продолжение.
- `llama-server` в этом участвует только как **CORS-прокси**, когда MCP-сервер
  находится на другом домене (а на Kaggle через туннель это практически
  всегда так). Прокси включается флагом `--webui-mcp-proxy` (в новых сборках
  — `--ui-mcp-proxy`).

**В этом проекте это уже настроено:** `server.webui_mcp_proxy: true` в
`config.yaml` включён по умолчанию, `start_server.sh` сам определяет, какое
имя флага поддерживает собранная версия `llama-server` (через `--help`), и
подставляет нужное. Достаточно открыть публичный Web UI → Settings → MCP
(или "MCP Servers" в сайдбаре) и вставить URL нужного MCP-сервера — как на
скриншотах официального демо-Space.

Дополнительно в проекте есть:
- `mcp/mcp_servers.json` — конфиг в стандартном формате `mcpServers` **для
  внешних клиентов** (Continue, Cline, кастомные скрипты), а не для панели
  в браузере.
- `scripts/mcp_agent.py` — рабочий пример agent loop поверх
  `/v1/chat/completions` с `tools`/`tool_calls`, показывающий тот же
  протокол «модель просит вызвать инструмент → мы выполняем → возвращаем
  результат → модель продолжает», но со стороны Python, а не браузера.

---

## VS Code

1. Установите одно из расширений: **Continue**, **Cline** или **Roo Code**.
2. После `python start.py` возьмите готовые конфиги из
   `vscode/generated/*.json` (URL уже подставлен) и вставьте в настройки
   расширения (для Continue — `~/.continue/config.json`).
3. Про GitHub Copilot — см. `vscode/copilot_notes.md` (нативный custom
   endpoint не поддерживается, только Continue/Cline/Roo Code).
4. Для чистого API — `vscode/example_openai_python.py` /
   `example_openai_js.js`.

---

## Тестирование

```bash
# health-check
curl https://<PUBLIC_URL>/health

# список моделей
curl https://<PUBLIC_URL>/v1/models

# бенчмарк производительности (после сборки)
./llama.cpp/build/bin/llama-bench -m ./models/<model>.gguf -ngl 999 -fa 1
```

---

## Ограничения Kaggle

- **Нет входящих соединений** — поэтому обязателен туннель (cloudflared
  по умолчанию, без регистрации).
- **Временное хранилище** — при остановке ноутбука всё, что не в
  `/kaggle/working`, теряется; модель скачивается заново при следующем
  запуске (либо сохраните её как Kaggle Dataset для переиспользования).
- **Лимит сессии** (обычно 9-12 часов на GPU-сессию, лимит квоты GPU в
  неделю) — долгие эксперименты планируйте заранее.
- **Две отдельные T4 без NVLink** — связь только через PCIe, поэтому
  `--split-mode layer` обычно эффективнее `row`.

---

## Troubleshooting

| Симптом | Причина / решение |
|---|---|
| `nvidia-smi: command not found` | В Settings ноутбука не включён GPU-аксселератор |
| Сборка падает на `GGML_CUDA` | Проверьте `nvcc --version`; при отсутствии CUDA Toolkit — используйте образ Kaggle с предустановленным CUDA (стандартный) |
| `CUDA out of memory` при старте | Уменьшите `server.ctx_size` в `config.yaml` или используйте более квантованную модель (Q4_K_M вместо Q5/Q6) |
| Туннель не даёт URL | Смотрите `logs/tunnel.log`; попробуйте `tunnel.provider: pinggy` как альтернативу |
| Модель скачивается, но 401/403 | Приватный репозиторий — задайте `HF_TOKEN` через Kaggle Secrets и `model.hf_token_env` |


## Запуск из GitHub в Kaggle

Ноутбук `notebooks/kaggle_run_all.ipynb` автоматически клонирует публичный репозиторий `PoFigiHubIO/KeglaAI` в `/kaggle/working/KeglaAI`, а при повторном запуске получает свежий коммит ветки `main`. Рабочая папка проекта — `/kaggle/working/KeglaAI/kaggle-llm-server`.

Для запуска загрузите **только этот ноутбук** в Kaggle, включите `GPU T4 x2` и Internet, затем выберите **Run All**. Не подключайте архив проекта как Kaggle Dataset: код берётся из GitHub. Модели и каталог `llama.cpp` не входят в GitHub и остаются в Kaggle Persistence либо подключаются отдельными Dataset/архивами.
