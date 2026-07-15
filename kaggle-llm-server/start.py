#!/usr/bin/env python3
"""
start.py

Главный оркестратор проекта kaggle-llm-server. Предназначен для запуска
из Kaggle Notebook одной ячейкой (`!python start.py`) в режиме Run All.

Последовательность (соответствует этапам 1-9 из ТЗ):
    1. hardware_check.py   — анализ окружения
    2. install.sh           — установка зависимостей
    3. build.sh              — сборка llama.cpp с CUDA
    4. download_model.py     — загрузка GGUF-модели
    5. optimize.py            — подбор оптимальных параметров под 2xT4
    6-8. start_server.sh      — запуск llama-server (OpenAI API + Web UI)
    7. tunnel.py               — публикация через Cloudflare Tunnel
    9. генерация VS Code / MCP конфигов с подставленным публичным URL

По завершении печатает итоговую сводку со всеми ссылками.
"""

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "scripts")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)


def run(cmd: list, check=True):
    print(f"\n$ {' '.join(cmd)}\n" + "-" * 70)
    result = subprocess.run(cmd)
    if check and result.returncode != 0:
        print(f"[start.py] Команда завершилась с ошибкой (code={result.returncode}): {cmd}")
        sys.exit(result.returncode)
    return result.returncode


def step(title: str):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def main():
    import yaml

    config_path = os.environ.get("CONFIG_FILE", "config.yaml")
    # Пробрасываем в окружение для всех дочерних процессов
    os.environ["CONFIG_FILE"] = config_path

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # --- Этап 1: анализ окружения ---
    step("ЭТАП 1/9 — Анализ окружения")
    run([sys.executable, "scripts/hardware_check.py"], check=False)

    # --- Этап 2: установка зависимостей ---
    step("ЭТАП 2/9 — Установка зависимостей")
    run(["bash", "install.sh"])

    # --- Этап 3: сборка llama.cpp ---
    step("ЭТАП 3/9 — Сборка llama.cpp (CUDA)")
    existing_binary = "./llama.cpp/build/bin/llama-server"
    needs_build = True
    if os.path.exists(existing_binary):
        # Файлы могли остаться от предыдущей сессии (Kaggle Persistence:
        # Variables and Files), но за это время Kaggle мог обновить образ
        # (CUDA/драйвер/glibc), а бит исполняемости мог "слететь" при
        # восстановлении файлов между сессиями — восстанавливаем его перед
        # проверкой. Проверяем реальным запуском, а не просто наличием файла,
        # чтобы не упасть позже на start_server.sh.
        try:
            os.chmod(existing_binary, 0o755)
        except OSError as e:
            print(f"[start.py][warn] Не удалось выставить права на {existing_binary}: {e}")

        try:
            check = subprocess.run(
                [existing_binary, "--version"], capture_output=True, timeout=30
            )
            binary_ok = check.returncode == 0
        except (PermissionError, OSError) as e:
            print(f"[start.py] {existing_binary} не запускается ({e}) — пересобираем.")
            binary_ok = False

        if binary_ok:
            needs_build = False
            print(f"[start.py] {existing_binary} уже собран и рабочий, пропускаем build.sh")
        else:
            print(f"[start.py] {existing_binary} найден, но не запускается "
                  f"(вероятно, Kaggle обновил образ, либо потеряны права доступа) — пересобираем.")
    if needs_build:
        run(["bash", "build.sh"])

    # --- Этап 4: загрузка модели ---
    step("ЭТАП 4/9 — Загрузка GGUF-модели")
    run([sys.executable, "download_model.py"])

    # --- Этап 5: оптимальные параметры под 2xT4 ---
    step("ЭТАП 5/9 — Автоподбор параметров запуска")
    run([sys.executable, "scripts/optimize.py"])

    # --- Этап 6-8: запуск сервера (API + Web UI) ---
    step("ЭТАП 6-8/9 — Запуск llama-server (OpenAI API + Web UI)")
    run(["bash", "start_server.sh"])

    # --- Этап 7: публичный туннель ---
    step("ЭТАП 7/9 — Публикация через туннель")
    from tunnel import start_tunnel

    provider = cfg["tunnel"]["provider"]
    port = cfg["server"]["port"]
    
    # Очищаем старый туннель на этом порту, если он остался
    tunnel_pid_file = f"logs/tunnel_{port}.pid"
    if os.path.exists(tunnel_pid_file):
        try:
            with open(tunnel_pid_file, "r") as f:
                old_pid = int(f.read().strip())
            os.kill(old_pid, 15)  # SIGTERM
            print(f"[start.py] Остановлен старый туннель (PID={old_pid}) на порту {port}")
            time.sleep(1)
        except Exception:
            pass

    # Извлекаем параметры для перманентных туннелей
    cloudflare_token = cfg["tunnel"].get("cloudflare_token") or os.environ.get("CLOUDFLARE_TUNNEL_TOKEN", "")
    cloudflare_domain = cfg["tunnel"].get("cloudflare_domain", "")
    ngrok_domain = cfg["tunnel"].get("ngrok_domain", "")

    proc, public_url = start_tunnel(
        provider,
        port,
        cloudflare_token=cloudflare_token,
        cloudflare_domain=cloudflare_domain,
        ngrok_domain=ngrok_domain
    )

    # Сохраняем PID нового туннеля
    with open(tunnel_pid_file, "w") as f:
        f.write(str(proc.pid))

    if not public_url:
        print("[start.py][warn] Не удалось автоматически определить публичный URL. "
              "Проверьте ./logs/tunnel_{}.log".format(port))
        public_url = "http://127.0.0.1:{}".format(port)
    else:
        print(f"[start.py] Публичный URL: {public_url}")

    # --- Этап 9: генерация конфигов VS Code / MCP с подставленным URL ---
    step("ЭТАП 9/9 — Генерация конфигов для VS Code")
    generate_vscode_configs(public_url, cfg)

    # --- Итоговая сводка ---
    print_summary(public_url, cfg)


def generate_vscode_configs(public_url: str, cfg: dict):
    os.makedirs("vscode/generated", exist_ok=True)
    api_base = f"{public_url}/v1"
    api_key = cfg["server"].get("api_key") or "sk-no-key-required"
    port = cfg["server"]["port"]

    # Читаем реальное имя модели и контекст из optimized_params_${port}.json
    model_name = "local-model"
    ctx_size = 32768
    params_json = f"./logs/optimized_params_{port}.json"
    if os.path.exists(params_json):
        try:
            with open(params_json, "r", encoding="utf-8") as f:
                params = json.load(f)
                if "model_path" in params:
                    model_name = params["model_path"]
                if "ctx_size" in params:
                    ctx_size = params["ctx_size"]
        except Exception:
            pass

    templates = {
        "continue_config.json": ("vscode/continue_config.json", f"vscode/generated/continue_config_{port}.json"),
        "cline_config.json": ("vscode/cline_config.json", f"vscode/generated/cline_config_{port}.json"),
        "roo_code_config.json": ("vscode/roo_code_config.json", f"vscode/generated/roo_code_config_{port}.json"),
    }
    for name, (src, dst) in templates.items():
        if not os.path.exists(src):
            continue
        with open(src, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace("https://PUBLIC_URL/v1", api_base).replace(
            "sk-no-key-required" if api_key == "sk-no-key-required" else "__API_KEY__",
            api_key,
        ).replace("local-model", model_name).replace("32768", str(ctx_size))
        with open(dst, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[start.py] Сгенерирован {dst}")


def print_summary(public_url: str, cfg: dict):
    port = cfg["server"]["port"]
    print("\n" + "#" * 78)
    print("#  kaggle-llm-server — ГОТОВО")
    print("#" * 78)
    print(f"""
  Web UI (публично):      {public_url}/
  OpenAI API (публично):  {public_url}/v1
  Локально (внутри Kaggle): http://127.0.0.1:{port}/

  Проверка:
    curl {public_url}/v1/models

  Пример запроса:
    curl {public_url}/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{{"model":"local-model","messages":[{{"role":"user","content":"Привет!"}}]}}'

  Конфиги для VS Code сгенерированы в ./vscode/generated/
  (Continue, Cline, Roo Code) — скопируйте нужный в соответствующее
  расширение (см. README.md, раздел "VS Code").

  MCP: ./mcp/mcp_servers.json, агентный цикл: scripts/mcp_agent.py

  Логи сервера:   ./logs/llama-server_{port}.log
  Логи туннеля:   ./logs/tunnel_{port}.log
  Параметры запуска: ./logs/optimized_params_{port}.json
""")


if __name__ == "__main__":
    main()
