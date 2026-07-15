#!/usr/bin/env python3
import argparse
import json
import sys

try:
    import requests
except ImportError:
    print("[error] Библиотека 'requests' не найдена. Установите её: pip install requests")
    sys.exit(1)


def test_api(api_url: str):
    if api_url.endswith("/"):
        api_url = api_url[:-1]
    if not api_url.endswith("/v1"):
        api_url += "/v1"

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-no-key-required"
    }

    payload = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Привет! Ответь одним словом 'Тест'."}
        ],
        "temperature": 0.3,
        "max_tokens": 50,
        "stream": False
    }

    print("=" * 80)
    print(f" Запуск диагностики API для адреса: {api_url}")
    print("=" * 80)

    print("\n--- ТЕСТ 1: Обычный запрос (stream=False) ---")
    try:
        url = f"{api_url}/chat/completions"
        print(f"[POST] {url}")
        print(f"[Body] {json.dumps(payload, ensure_ascii=False)}")
        
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        print(f"[Status] {response.status_code}")
        print("[Headers]")
        for k, v in response.headers.items():
            print(f"  {k}: {v}")
        
        print("\n[Raw Response Text]")
        print(response.text)
        
        if response.status_code == 200:
            data = response.json()
            print("\n[Parsed JSON Response]")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            choices = data.get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content", "")
                print(f"\n[!] Извлеченный текст ответа: '{text}'")
                if not text:
                    print("[WARNING] Текст ответа пустой!")
            else:
                print("[WARNING] В ответе нет поля 'choices'!")
        else:
            print(f"[ERROR] Запрос завершился с кодом ошибки {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Ошибка выполнения запроса ТЕСТ 1: {e}")

    print("\n" + "=" * 80)
    print("\n--- ТЕСТ 2: Потоковый запрос (stream=True) ---")
    payload["stream"] = True
    try:
        url = f"{api_url}/chat/completions"
        print(f"[POST] {url}")
        print(f"[Body] {json.dumps(payload, ensure_ascii=False)}")
        
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=15)
        print(f"[Status] {response.status_code}")
        print("[Headers]")
        for k, v in response.headers.items():
            print(f"  {k}: {v}")
            
        print("\n[Stream Chunks]")
        has_content = False
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                print(f"  {decoded_line}")
                if decoded_line.startswith("data:"):
                    data_str = decoded_line[5:].strip()
                    if data_str == "[DONE]":
                        print("  [Конец потока]")
                        break
                    try:
                        data_json = json.loads(data_str)
                        delta = data_json.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            has_content = True
                    except Exception:
                        pass
        if not has_content and response.status_code == 200:
            print("[WARNING] Ни один чанк не содержал текста (content)!")
    except Exception as e:
        print(f"[ERROR] Ошибка выполнения запроса ТЕСТ 2: {e}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Диагностический скрипт для API kaggle-llm-server")
    parser.add_argument("--url", required=True, help="Публичный Cloudflare URL (например, https://xxxx.trycloudflare.com)")
    args = parser.parse_args()
    test_api(args.url)
