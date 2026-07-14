#!/usr/bin/env python3
"""
scripts/mcp_agent.py

Этап 9 (практическая часть). Минимальный рабочий agent loop, который:
    1. Отправляет сообщения в локальный llama-server (OpenAI-compatible
       /v1/chat/completions) с полем `tools`, собранным из mcp/mcp_servers.json
       описаний (здесь — упрощённо, как локальные python-функции-примеры,
       т.к. полноценный MCP client требует stdio-транспорт до Node-серверов).
    2. Если модель возвращает tool_calls — выполняет соответствующую функцию.
    3. Отправляет результат обратно как role="tool" и повторяет цикл.
    4. Останавливается, когда модель отвечает обычным текстом (без tool_calls)
       либо когда достигнут mcp.agent_loop.max_iterations.

Это референсная реализация, показывающая протокол взаимодействия. Для
подключения реальных MCP-серверов (filesystem, fetch, memory, ...) из
mcp/mcp_servers.json используйте полноценный MCP-клиент, например пакет
`mcp` (pip install mcp) с stdio_client, либо интеграцию MCP в Continue/Cline
(см. vscode/*.json) — они уже умеют разговаривать по MCP из коробки и
использовать этот сервер как OpenAI-compatible backend.

Запуск:
    python scripts/mcp_agent.py "Найди файл config.yaml и скажи, какой backend туннеля выбран"
"""

import json
import os
import sys

from openai import OpenAI

BASE_URL = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080/v1")
API_KEY = os.environ.get("LLAMA_API_KEY", "sk-no-key-required")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


# --- Примеры локальных "инструментов", эмулирующих filesystem MCP-сервер ---
def tool_read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()[:4000]
    except Exception as e:
        return f"ERROR: {e}"


def tool_list_dir(path: str) -> str:
    try:
        return json.dumps(os.listdir(path))
    except Exception as e:
        return f"ERROR: {e}"


TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Прочитать текстовый файл проекта и вернуть его содержимое.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Путь к файлу"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "Показать список файлов в директории.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Путь к директории"}},
                "required": ["path"],
            },
        },
    },
]

TOOL_IMPL = {"read_file": tool_read_file, "list_dir": tool_list_dir}


def run_agent(user_prompt: str, model: str = "local-model", max_iterations: int = 8):
    messages = [
        {
            "role": "system",
            "content": (
                "Ты — полезный ассистент с доступом к инструментам файловой системы "
                "проекта kaggle-llm-server. Используй их, когда нужно посмотреть содержимое "
                "конфигов или логов, прежде чем отвечать."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    for step in range(max_iterations):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SPEC,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            print("\n=== Финальный ответ ===")
            print(msg.content)
            return msg.content

        for call in msg.tool_calls:
            fn_name = call.function.name
            args = json.loads(call.function.arguments or "{}")
            print(f"[agent] шаг {step+1}: вызов инструмента {fn_name}({args})")
            impl = TOOL_IMPL.get(fn_name)
            result = impl(**args) if impl else f"ERROR: unknown tool {fn_name}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(result),
                }
            )

    print("[agent] достигнут лимит итераций max_iterations")
    return None


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "Опиши структуру проекта в models/ и logs/"
    run_agent(prompt)
