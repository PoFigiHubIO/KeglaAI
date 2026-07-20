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


def register_with_cloudflare_worker(public_url: str, port: int, cfg: dict):
    cf_url = cfg["tunnel"].get("cloudflare_worker_url") or os.environ.get("CF_WORKER_URL", "")
    secret = os.environ.get("HANDOVER_SECRET", "default_secret")
    if not cf_url:
        print("[start.py] cloudflare_worker_url не настроен. Пропускаем регистрацию в KV.")
        return

    import requests
    try:
        payload = {}
        if port == 8080:
            payload["llm_url"] = public_url
        elif port == 8081:
            payload["media_url"] = public_url

        print(f"[start.py] Регистрация URL ({public_url}) в Cloudflare Worker...")
        res = requests.post(
            f"{cf_url}/register",
            json=payload,
            headers={"Authorization": f"Bearer {secret}"},
            timeout=15
        )
        if res.status_code == 200:
            print("[start.py] ✅ URL успешно зарегистрирован в Cloudflare KV.")
        else:
            print(f"[start.py][error] Ошибка регистрации URL: {res.status_code} - {res.text}")
    except Exception as e:
        print(f"[start.py][error] Ошибка подключения к Cloudflare Worker: {e}")


def trigger_handover(cfg: dict):
    cf_url = cfg["tunnel"].get("cloudflare_worker_url") or os.environ.get("CF_WORKER_URL", "")
    secret = os.environ.get("HANDOVER_SECRET", "default_secret")
    
    # 1. Если настроен Cloudflare Worker, отправляем сигнал через него
    if cf_url:
        import requests
        try:
            print("[start.py] Запрос активных URL для передачи управления...")
            res = requests.get(f"{cf_url}/active", timeout=10)
            if res.status_code == 200:
                data = res.json()
                old_media_url = data.get("media")
                if old_media_url:
                    print(f"[start.py] Отправка сигнала передачи управления старой ноде: {old_media_url}/v1/handover/complete ...")
                    h_res = requests.post(
                        f"{old_media_url}/v1/handover/complete",
                        json={"secret": secret},
                        timeout=15
                    )
                    if h_res.status_code == 200:
                        print("[start.py] ✅ Сигнал передачи управления успешно отправлен старой ноде.")
                        return
                    else:
                        print(f"[start.py][warn] Старая нода отклонила запрос: {h_res.status_code}")
            else:
                print(f"[start.py][warn] Не удалось получить активные URL: {res.status_code}")
        except Exception as e:
            print(f"[start.py][warn] Не удалось связаться со старой нодой через Worker: {e}")

    # 2. Резервный канал: сигнализация через файл на облачном диске
    print("[start.py] Отправка сигнала передачи управления через облачный диск (Rclone)...")
    try:
        import subprocess
        # Запускаем скрипт Rclone для выгрузки файла-сигнала
        res = subprocess.run(
            ["bash", "scripts/rclone_sync.sh", "upload_signal"],
            capture_output=True, text=True, timeout=60
        )
        if res.returncode == 0:
            print("[start.py] ✅ Сигнал передачи управления успешно записан в облако.")
        else:
            print(f"[start.py][error] Не удалось записать сигнал в облако (code={res.returncode}): {res.stderr}")
    except Exception as e:
        print(f"[start.py][error] Ошибка при отправке облачного сигнала: {e}")


def load_kaggle_secrets():
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        for key in [
            "HF_TOKEN",
            "TELEGRAM_BOT_TOKEN",
            "CLOUDFLARE_TUNNEL_TOKEN",
            "YANDEX_TOKEN",
            "RCLONE_PROVIDER",
            "RCLONE_USER",
            "RCLONE_PASS",
            "HANDOVER_SECRET",
            "NEXT_KAGGLE_USERNAME",
            "NEXT_KAGGLE_KEY",
            "NEXT_KAGGLE_SLUG",
            "ROTATION_TIME_SECONDS",
            "NGROK_AUTHTOKEN",
            "NGROK_AUTHTOKEN_2",
            "CLOUDFLARE_TUNNEL_DOMAIN_8083",
            "CLOUDFLARE_TUNNEL_DOMAIN_8084"
        ]:
            try:
                val = user_secrets.get_secret(key)
                if val:
                    os.environ[key] = val
            except Exception:
                pass
    except Exception:
        pass


def main():
    load_kaggle_secrets()
    import yaml

    config_path = os.environ.get("CONFIG_FILE", "config.yaml")
    os.environ["CONFIG_FILE"] = config_path

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    gpu = cfg.get("server", {}).get("gpu")
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        print(f"[start.py] Изолируем процесс на GPU: {gpu}")

    port = cfg["server"].get("port", 8080)

    # --- Этап 1: анализ окружения ---
    step("ЭТАП 1/9 — Анализ окружения")
    run([sys.executable, "scripts/hardware_check.py"], check=False)

    # --- Этап 2: установка зависимостей ---
    step("ЭТАП 2/9 — Установка зависимостей")
    run(["bash", "install.sh"])

    # --- Этап 2.5: Скачивание БД из облака (failover) ---
    step("ЭТАП 2.5/9 — Восстановление базы данных из облака")
    run(["bash", "scripts/rclone_sync.sh", "download"], check=False)

    # Isolate Hugging Face cache on temporary directory (to avoid 20GB disk quota limit for FLUX/Wan)
    os.environ["HF_HOME"] = "/tmp/.cache"

    # --- Этап 3: сборка llama.cpp ---
    step("ЭТАП 3/9 — Сборка llama.cpp (CUDA)")
    existing_binary = "./llama.cpp/build/bin/llama-server"
    needs_build = True
    if os.path.exists(existing_binary):
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
            binary_ok = False

        if binary_ok:
            print("[start.py] Восстановленная сборка рабочая! Полная компиляция пропущена.")
            needs_build = False

    if needs_build:
        run(["bash", "build.sh"])

    # --- Этап 4: загрузка GGUF-модели ---
    step(f"ЭТАП 4/9 — Загрузка GGUF-модели ({config_path})")
    run([sys.executable, "download_model.py"])

    # --- Этап 5: автоподбор параметров ---
    step("ЭТАП 5/9 — Автоподбор параметров запуска")
    run([sys.executable, "scripts/optimize.py"])

    # --- Этап 6: запуск сервера llama-server ---
    step("ЭТАП 6/9 — Запуск llama-server")
    run(["bash", "start_server.sh"])

    # --- Этап 7: запуск API Gateway ---
    step("ЭТАП 7/9 — Запуск API Gateway и Web UI (порт 8080)")
    os.system("pkill -f gateway.py")
    gateway_log = open("logs/gateway.log", "w")
    gateway_proc = subprocess.Popen(
        [sys.executable, "scripts/gateway.py"],
        stdout=gateway_log,
        stderr=subprocess.STDOUT,
        start_new_session=True
    )
    with open("logs/gateway.pid", "w") as f:
        f.write(str(gateway_proc.pid))
    print(f"[start.py] API Gateway запущен (PID={gateway_proc.pid})")
    
    # Ожидание готовности API Gateway
    ready = False
    for _ in range(30):
        if gateway_proc.poll() is not None:
            print("[start.py][error] Процесс API Gateway аварийно завершился!")
            break
        try:
            import urllib.request
            with urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=2) as response:
                if response.status == 200:
                    ready = True
                    print("[start.py] ✅ API Gateway успешно запущен и слушает порт 8080!")
                    break
        except Exception:
            pass
        time.sleep(2)

    if not ready:
        print("[start.py][error] API Gateway не ответил на /v1/models. Логи:")
        if os.path.exists("logs/gateway.log"):
            with open("logs/gateway.log", "r") as lf:
                for line in lf.readlines()[-40:]:
                    print(line, end="")
        sys.exit(1)

    # --- Этап 8: публичный туннель ---
    step("ЭТАП 8/9 — Публикация через туннель")
    from tunnel import start_tunnel

    provider = cfg["tunnel"]["provider"]
    tunnel_pid_file = f"logs/tunnel_{port}.pid"
    if os.path.exists(tunnel_pid_file):
        try:
            with open(tunnel_pid_file, "r") as f:
                old_pid = int(f.read().strip())
            os.kill(old_pid, 15)
            time.sleep(1)
        except Exception:
            pass

    if provider == "none" or not provider:
        print(f"[start.py] Публичный туннель отключен для порта {port} (provider = none).")
        public_url = f"http://127.0.0.1:{port}"
    else:
        cloudflare_token = cfg["tunnel"].get("cloudflare_token") or os.environ.get("CLOUDFLARE_TUNNEL_TOKEN", "")
        # Dynamically load the domain from secret like CLOUDFLARE_TUNNEL_DOMAIN_8083
        domain_secret_key = f"CLOUDFLARE_TUNNEL_DOMAIN_{port}"
        cloudflare_domain = cfg["tunnel"].get("cloudflare_domain") or os.environ.get(domain_secret_key, "")
        
        ngrok_domain = cfg["tunnel"].get("ngrok_domain", "")
        ngrok_token_env = cfg["tunnel"].get("ngrok_token_env", "NGROK_AUTHTOKEN")
        ngrok_token = os.environ.get(ngrok_token_env, "")

        # Проксируем порт модели (8083 или 8084) напрямую для встроенного llama.cpp Web UI
        proc, public_url = start_tunnel(
            provider,
            port,
            cloudflare_token=cloudflare_token,
            cloudflare_domain=cloudflare_domain,
            ngrok_domain=ngrok_domain,
            ngrok_token=ngrok_token
        )
        with open(tunnel_pid_file, "w") as f:
            f.write(str(proc.pid))

        if not public_url:
            public_url = f"http://127.0.0.1:{port}"
        print(f"[start.py] Публичный URL модели (порт {port}): {public_url}")

        # Регистрация в Cloudflare KV
        register_with_cloudflare_worker(public_url, port, cfg)

    # --- Запуск Telegram Bot ---
    step("ЭТАП 9/9 — Запуск Telegram Bot Agent Loop")
    # Clean up old bot instance
    os.system("pkill -f telegram_bot.py")
    bot_log = open("logs/telegram_bot.log", "w")
    bot_proc = subprocess.Popen(
        [sys.executable, "scripts/telegram_bot.py"],
        stdout=bot_log,
        stderr=subprocess.STDOUT,
        start_new_session=True
    )
    with open("logs/telegram_bot.pid", "w") as f:
        f.write(str(bot_proc.pid))
    print(f"[start.py] Telegram Bot запущен (PID={bot_proc.pid})")
    
    time.sleep(5)
    if bot_proc.poll() is not None:
        print("[start.py][error] Telegram Bot завершился с ошибкой!")
        if os.path.exists("logs/telegram_bot.log"):
            with open("logs/telegram_bot.log", "r") as lf:
                for line in lf.readlines()[-40:]:
                    print(f"[bot] {line.strip()}")
        sys.exit(1)
    else:
        print("[start.py] ✅ Telegram Bot успешно работает в фоне.")

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
