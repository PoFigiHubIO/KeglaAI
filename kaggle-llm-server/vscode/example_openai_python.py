"""
Этап 10. Пример подключения к серверу через OpenAI SDK (Python).

pip install openai

Замените PUBLIC_URL на ссылку, которую выводит start.py после запуска
Cloudflare Tunnel (например https://random-words.trycloudflare.com).
"""

from openai import OpenAI

client = OpenAI(
    base_url="https://PUBLIC_URL/v1",
    api_key="sk-no-key-required",   # или значение server.api_key из config.yaml
)

# --- Простой запрос ---
resp = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Привет! Напиши функцию сортировки на Python."}],
    temperature=0.7,
    max_tokens=512,
)
print(resp.choices[0].message.content)

# --- Streaming ---
stream = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Расскажи короткую историю про робота."}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()

# --- JSON mode ---
resp_json = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Верни JSON с полями name и age для пользователя John, 30 лет."}],
    response_format={"type": "json_object"},
)
print(resp_json.choices[0].message.content)

# --- Список доступных моделей ---
print(client.models.list())
