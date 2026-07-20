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


def _download_file(url: str, output_path: str):
    import urllib.request
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
    )
    with urllib.request.urlopen(req) as response, open(output_path, "wb") as out_file:
        out_file.write(response.read())

def _ensure_cloudflared() -> str:
    """Скачивает статический бинарник cloudflared, если его ещё нет."""
    binary = "./bin/cloudflared"
    if not os.path.exists(binary):
        os.makedirs("./bin", exist_ok=True)
        url = (
            "https://github.com/cloudflare/cloudflared/releases/latest/download/"
            "cloudflared-linux-amd64"
        )
        print(f"[tunnel] Downloading cloudflared from {url}...")
        _download_file(url, binary)
    os.chmod(binary, 0o755)
    return binary


def _ensure_ngrok() -> str:
    """Скачивает и распаковывает статический бинарник ngrok, если его ещё нет."""
    binary = "./bin/ngrok"
    if not os.path.exists(binary):
        os.makedirs("./bin", exist_ok=True)
        url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz"
        tar_path = "./bin/ngrok.tgz"
        print(f"[tunnel] Downloading ngrok from {url}...")
        _download_file(url, tar_path)
        subprocess.run(["tar", "-xzf", tar_path, "-C", "./bin"], check=True)
        if os.path.exists(tar_path):
            os.remove(tar_path)
    os.chmod(binary, 0o755)
    return binary


def start_cloudflared(port: int, log_path: str = "./logs/tunnel.log", token: str = "", domain: str = ""):
    binary = _ensure_cloudflared()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_f = open(log_path, "w")
    if token:
        cmd = [binary, "tunnel", "--no-autoupdate", "run", "--token", token]
        url = domain if domain else "https://your-custom-cloudflare-domain.com"
    else:
        cmd = [binary, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"]
        
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    if not token:
        url = _wait_for_url(log_path, CLOUDFLARED_URL_RE)
    return proc, url


def start_pinggy(port: int, log_path: str = "./logs/tunnel.log"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        [
            "ssh", "-p", "443", "-o", "StrictHostKeyChecking=no",
            "-o", "ServerAliveInterval=30",
            "-R", f"0:127.0.0.1:{port}", "a.pinggy.io",
        ],
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
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
        start_new_session=True,
    )
    url = _wait_for_url(log_path, LOCALTUNNEL_URL_RE)
    return proc, url


def start_ngrok(port: int, log_path: str = "./logs/tunnel.log", domain: str = "", token: str = ""):
    import json as _json
    import urllib.request

    binary = _ensure_ngrok()

    if not token:
        token = os.environ.get("NGROK_AUTHTOKEN", "")
    if not token:
        raise RuntimeError("Токен Ngrok не задан")
        
    subprocess.run([binary, "config", "add-authtoken", token], check=False)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_f = open(log_path, "w")
    
    cmd = [binary, "http", str(port), "--log=stdout"]
    if domain:
        cmd.extend(["--domain", domain])
        
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
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


def start_tunnel(provider: str, port: int, cloudflare_token: str = "", cloudflare_domain: str = "", ngrok_domain: str = "", ngrok_token: str = ""):
    provider = provider.lower()
    log_path = f"./logs/tunnel_{port}.log"
    if provider == "cloudflared" or provider == "cloudflare":
        return start_cloudflared(port, log_path, cloudflare_token, cloudflare_domain)
    if provider == "pinggy":
        return start_pinggy(port, log_path)
    if provider == "localtunnel":
        return start_localtunnel(port, log_path)
    if provider == "ngrok":
        return start_ngrok(port, log_path, ngrok_domain, ngrok_token)
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
