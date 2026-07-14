#!/usr/bin/env python3
"""
scripts/tunnel.py

Этап 7. Публичная ссылка.

Kaggle Notebook не принимает входящие соединения напрямую, поэтому нужен
туннель наружу. Поддерживаются:
    - cloudflared   (рекомендуется, без регистрации, стабильный, TryCloudflare)
    - pinggy        (без установки, через ssh)
    - localtunnel   (npm, требует Node.js)
    - ngrok         (требует NGROK_AUTHTOKEN)

Скрипт запускает выбранный туннель в фоне, парсит его stdout/stderr в
поисках публичного URL и возвращает его вызывающему коду (start.py).
"""

import os
import re
import subprocess
import sys
import time

CLOUDFLARED_URL_RE = re.compile(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com")
PINGGY_URL_RE = re.compile(r"https://[a-zA-Z0-9\-]+\.a\.pinggy\.link")
LOCALTUNNEL_URL_RE = re.compile(r"https://[a-zA-Z0-9\-]+\.loca\.lt")
NGROK_API = "http://127.0.0.1:4040/api/tunnels"


def _ensure_cloudflared() -> str:
    """Скачивает статический бинарник cloudflared, если его ещё нет."""
    binary = "./bin/cloudflared"
    if os.path.exists(binary):
        return binary
    os.makedirs("./bin", exist_ok=True)
    url = (
        "https://github.com/cloudflare/cloudflared/releases/latest/download/"
        "cloudflared-linux-amd64"
    )
    subprocess.run(["wget", "-q", "-O", binary, url], check=True)
    os.chmod(binary, 0o755)
    return binary


def start_cloudflared(port: int, log_path: str = "./logs/tunnel.log"):
    binary = _ensure_cloudflared()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        [binary, "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )
    url = _wait_for_url(log_path, CLOUDFLARED_URL_RE)
    return proc, url


def start_pinggy(port: int, log_path: str = "./logs/tunnel.log"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        [
            "ssh", "-p", "443", "-o", "StrictHostKeyChecking=no",
            "-o", "ServerAliveInterval=30",
            "-R", f"0:localhost:{port}", "a.pinggy.io",
        ],
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )
    url = _wait_for_url(log_path, PINGGY_URL_RE)
    return proc, url


def start_localtunnel(port: int, log_path: str = "./logs/tunnel.log"):
    subprocess.run(["npm", "install", "-g", "localtunnel"], check=False)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        ["lt", "--port", str(port)],
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )
    url = _wait_for_url(log_path, LOCALTUNNEL_URL_RE)
    return proc, url


def start_ngrok(port: int, log_path: str = "./logs/tunnel.log"):
    import json as _json
    import urllib.request

    token = os.environ.get("NGROK_AUTHTOKEN", "")
    if not token:
        raise RuntimeError("Переменная окружения NGROK_AUTHTOKEN не задана")
    subprocess.run(["ngrok", "config", "add-authtoken", token], check=False)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        ["ngrok", "http", str(port), "--log=stdout"],
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )
    for _ in range(30):
        time.sleep(1)
        try:
            with urllib.request.urlopen(NGROK_API, timeout=2) as resp:
                data = _json.loads(resp.read())
                tunnels = data.get("tunnels", [])
                for t in tunnels:
                    if t.get("public_url", "").startswith("https"):
                        return proc, t["public_url"]
        except Exception:
            continue
    return proc, None


def _wait_for_url(log_path: str, pattern: re.Pattern, timeout: int = 45):
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(log_path):
            with open(log_path, "r", errors="ignore") as f:
                content = f.read()
            match = pattern.search(content)
            if match:
                return match.group(0)
        time.sleep(1)
    return None


def start_tunnel(provider: str, port: int):
    provider = provider.lower()
    if provider == "cloudflared":
        return start_cloudflared(port)
    if provider == "pinggy":
        return start_pinggy(port)
    if provider == "localtunnel":
        return start_localtunnel(port)
    if provider == "ngrok":
        return start_ngrok(port)
    raise ValueError(f"Неизвестный провайдер туннеля: {provider}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="cloudflared")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    proc, url = start_tunnel(args.provider, args.port)
    if url:
        print(f"PUBLIC_URL={url}")
    else:
        print("Не удалось получить публичный URL, см. ./logs/tunnel.log", file=sys.stderr)
        sys.exit(1)

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
