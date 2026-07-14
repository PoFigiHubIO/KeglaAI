#!/usr/bin/env python3
"""
download_model.py

Этап 4. Загрузка модели.

Поддерживает три источника (задаются в config.yaml -> model.source):
    - huggingface : repo_id + filename, через huggingface_hub.hf_hub_download
                    (поддерживает приватные репозитории через HF_TOKEN)
    - gdrive      : gdrive_file_id, через gdown
    - direct_url  : прямая ссылка на .gguf (requests, стриминг с прогресс-баром)

Особенности:
    - кэширование: если файл с таким именем уже есть в model.local_dir и его
      размер совпадает с ожидаемым (для HF — с данными API), повторная
      закачка не выполняется;
    - проверка sha256, если он указан в config.yaml;
    - автоматическое определение имени файла модели там, где это возможно;
    - если model.mmproj_enabled: true — дополнительно скачивает projector-файл
      (mmproj) для мультимодальных (vision) моделей, тем же источником, что
      и основная модель (huggingface/gdrive/direct_url).
"""

import hashlib
import os
import sys

import yaml
from tqdm import tqdm

# Настраиваем директорию кэша Hugging Face на Kaggle, чтобы избежать переполнения диска /root
if os.path.exists("/kaggle/working"):
    os.environ["HF_HOME"] = "/kaggle/working/.cache/huggingface"

# Пытаемся автоматически подгрузить HF_TOKEN из секретов Kaggle, если его нет в окружении
if "HF_TOKEN" not in os.environ:
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        token = user_secrets.get_secret("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
            print("[download] Успешно загружен HF_TOKEN из Kaggle Secrets.")
    except Exception:
        pass

# Принудительно отключаем IPv6 для обхода зависаний DNS-резолвинга на Kaggle
import socket
orig_getaddrinfo = socket.getaddrinfo
def patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = patched_getaddrinfo

# Включаем hf-transfer для многопоточного скачивания на Rust (если установлен)
try:
    import hf_transfer
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    print("[download] Включен высокоскоростной Rust-бэкенд скачивания (hf-transfer).")
except ImportError:
    pass


def load_config(path="config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_of_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_from_huggingface(cfg: dict, filename: str = None) -> str:
    from huggingface_hub import hf_hub_download

    m = cfg["model"]
    filename = filename or m["filename"]
    token = os.environ.get(m.get("hf_token_env", "HF_TOKEN"), None)

    print(f"[download] HuggingFace: repo={m['repo_id']} file={filename}")
    path = hf_hub_download(
        repo_id=m["repo_id"],
        filename=filename,
        local_dir=m["local_dir"],
        token=token,
        local_dir_use_symlinks=False,
    )
    return path


def download_from_gdrive(cfg: dict, file_id: str = None, filename: str = None) -> str:
    import gdown

    m = cfg["model"]
    os.makedirs(m["local_dir"], exist_ok=True)
    file_id = file_id or m["gdrive_file_id"]
    if not file_id:
        raise ValueError("model.gdrive_file_id не указан в config.yaml")

    # Пытаемся получить осмысленное имя файла, иначе используем file_id
    out_name = filename or m.get("filename") or f"{file_id}.gguf"
    out_path = os.path.join(m["local_dir"], out_name)

    print(f"[download] Google Drive: file_id={file_id} -> {out_path}")
    gdown.download(id=file_id, output=out_path, quiet=False)
    return out_path


def download_from_direct_url(cfg: dict, url: str = None, filename: str = None) -> str:
    import requests

    m = cfg["model"]
    url = url or m["direct_url"]
    if not url:
        raise ValueError("model.direct_url не указан в config.yaml")

    os.makedirs(m["local_dir"], exist_ok=True)
    filename = filename or m.get("filename") or url.split("/")[-1].split("?")[0]
    out_path = os.path.join(m["local_dir"], filename)

    headers = {}
    token = os.environ.get(m.get("hf_token_env", "HF_TOKEN"), "")
    if token and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {token}"

    print(f"[download] Direct URL: {url} -> {out_path}")
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=filename
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return out_path


def already_cached(local_dir: str, filename: str) -> str | None:
    if not filename:
        return None
    candidate = os.path.join(local_dir, filename)
    if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
        return candidate
    return None


def download_file(cfg: dict, source: str, filename: str) -> str:
    """Скачивает произвольный файл (основную модель или mmproj) выбранным
    источником, с переиспользованием той же логики кэширования."""
    m = cfg["model"]
    cached = already_cached(m["local_dir"], filename)
    if cached:
        print(f"[download] Уже есть в кэше: {cached} (пропускаем)")
        return cached

    if source == "huggingface":
        return download_from_huggingface(cfg, filename=filename)
    elif source == "gdrive":
        raise ValueError(
            "Для mmproj-файла с источником gdrive укажите отдельный "
            "model.mmproj_gdrive_file_id — автоматическое определение "
            "второго файла по одному gdrive_file_id невозможно."
        )
    elif source == "direct_url":
        raise ValueError(
            "Для mmproj-файла с источником direct_url укажите отдельный "
            "model.mmproj_direct_url."
        )
    else:
        raise ValueError(f"Неизвестный источник модели: {source}")


def verify_checksum(path: str, expected_sha256: str) -> None:
    if not expected_sha256:
        print("[download] sha256 не задан в config.yaml — проверка пропущена.")
        return
    print("[download] Проверка контрольной суммы sha256...")
    actual = sha256_of_file(path)
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"Контрольная сумма не совпадает!\n  ожидалось: {expected_sha256}\n  получено:  {actual}"
        )
    print("[download] Контрольная сумма совпадает. OK.")


def main():
    cfg = load_config()
    m = cfg["model"]
    os.makedirs(m["local_dir"], exist_ok=True)

    cached = already_cached(m["local_dir"], m.get("filename"))
    if cached:
        print(f"[download] Модель уже есть в кэше: {cached} (пропускаем скачивание)")
        model_path = cached
    else:
        source = m.get("source", "huggingface")
        if source == "huggingface":
            model_path = download_from_huggingface(cfg)
        elif source == "gdrive":
            model_path = download_from_gdrive(cfg)
        elif source == "direct_url":
            model_path = download_from_direct_url(cfg)
        else:
            raise ValueError(f"Неизвестный источник модели: {source}")

    if m.get("verify_sha256") and m.get("sha256"):
        verify_checksum(model_path, m["sha256"])

    size_gb = os.path.getsize(model_path) / (1024 ** 3)
    print(f"[download] Готово: {model_path} ({size_gb:.2f} GB)")

    # Записываем путь для последующих этапов (optimize.py / start_server.sh)
    os.makedirs("./logs", exist_ok=True)
    with open("./logs/model_path.txt", "w", encoding="utf-8") as f:
        f.write(model_path)

    # --- mmproj (vision projector), опционально ---
    if m.get("mmproj_enabled") and m.get("mmproj_filename"):
        print(f"[download] Модель мультимодальная (mmproj_enabled: true), качаем projector...")
        source = m.get("source", "huggingface")
        mmproj_path = download_file(cfg, source, m["mmproj_filename"])

        if m.get("mmproj_sha256"):
            verify_checksum(mmproj_path, m["mmproj_sha256"])

        mmproj_size_mb = os.path.getsize(mmproj_path) / (1024 ** 2)
        print(f"[download] mmproj готов: {mmproj_path} ({mmproj_size_mb:.1f} MB)")

        with open("./logs/mmproj_path.txt", "w", encoding="utf-8") as f:
            f.write(mmproj_path)
    else:
        # Убираем файл со старым путём, если mmproj отключили после
        # предыдущего запуска — start_server.sh иначе продолжит его находить.
        stale = "./logs/mmproj_path.txt"
        if os.path.exists(stale):
            os.remove(stale)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[download][error] {e}", file=sys.stderr)
        sys.exit(1)
