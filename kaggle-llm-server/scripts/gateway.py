import asyncio
import json
import logging
import os
import sys
import time
import uuid
import httpx
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gateway")

app = FastAPI(title="KeglaAI Gateway", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration & Backend servers
CONFIG_PATH = os.environ.get("CONFIG_FILE", "config.yaml")
PORT_12B = 8083
PORT_E2B = 8084

# MCP Client Session class
class StdioMcpClient:
    def __init__(self, name: str, command: str, args: List[str]):
        self.name = name
        self.command = command
        self.args = args
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.pending_requests: Dict[int, asyncio.Future] = {}
        self.req_id_counter = 1
        self.tools: List[Dict[str, Any]] = []
        self._reader_task: Optional[asyncio.Task] = None

    async def start(self):
        log.info(f"Starting MCP server: {self.name} ({self.command} {' '.join(self.args)})")
        try:
            self.proc = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL
            )
            self._reader_task = asyncio.create_task(self._read_loop())
            
            # Initialize MCP handshake
            init_res = await self.call_method("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "gateway-client", "version": "1.0.0"}
            })
            log.info(f"MCP server {self.name} initialized: {init_res}")
            
            # Fetch tools list
            tools_res = await self.call_method("tools/list")
            self.tools = tools_res.get("tools", [])
            log.info(f"MCP server {self.name} tools: {[t['name'] for t in self.tools]}")
        except Exception as e:
            log.error(f"Failed to start MCP server {self.name}: {e}")

    async def _read_loop(self):
        while self.proc and self.proc.stdout:
            line = await self.proc.stdout.readline()
            if not line:
                break
            try:
                data = json.loads(line.decode("utf-8"))
                req_id = data.get("id")
                if req_id in self.pending_requests:
                    fut = self.pending_requests.pop(req_id)
                    if "error" in data:
                        fut.set_exception(RuntimeError(data["error"].get("message", "Unknown error")))
                    else:
                        fut.set_result(data.get("result", {}))
            except Exception as e:
                log.error(f"Error reading from MCP {self.name}: {e}")

    async def call_method(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.proc or not self.proc.stdin:
            raise RuntimeError(f"MCP server {self.name} is not running")
        
        req_id = self.req_id_counter
        self.req_id_counter += 1
        
        req = {
            "jsonrpc": "2.0",
            "method": method,
            "id": req_id
        }
        if params is not None:
            req["params"] = params
            
        fut = asyncio.get_event_loop().create_future()
        self.pending_requests[req_id] = fut
        
        self.proc.stdin.write((json.dumps(req) + "\n").encode("utf-8"))
        await self.proc.stdin.drain()
        
        return await fut

    async def stop(self):
        if self._reader_task:
            self._reader_task.cancel()
        if self.proc:
            try:
                self.proc.terminate()
                await self.proc.wait()
            except Exception:
                pass
            self.proc = None

# Global list of MCP clients
mcp_clients: Dict[str, StdioMcpClient] = {}

async def load_mcp_servers():
    mcp_config_path = "mcp/mcp_servers.json"
    if not os.path.exists(mcp_config_path):
        log.warning(f"MCP config not found at {mcp_config_path}")
        return
        
    try:
        with open(mcp_config_path, "r", encoding="utf-8") as f:
            # Strip comments
            lines = [l for l in f.readlines() if not l.strip().startswith("//")]
            config = json.loads("".join(lines))
            
        servers = config.get("mcpServers", {})
        for name, cfg in servers.items():
            cmd = cfg.get("command")
            args = cfg.get("args", [])
            client = StdioMcpClient(name, cmd, args)
            await client.start()
            mcp_clients[name] = client
    except Exception as e:
        log.error(f"Error loading MCP servers: {e}", exc_info=True)

# ---------------------------------------------------------------------------
# API Gateway Routing & Agent Loop
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    await load_mcp_servers()

@app.on_event("shutdown")
async def shutdown_event():
    for client in mcp_clients.values():
        await client.stop()

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "gemma-4-12b", "object": "model", "owned_by": "llmfan46"},
            {"id": "gemma-4-e2b", "object": "model", "owned_by": "HauhauCS"}
        ]
    }

def get_backend_url(model: str) -> str:
    m = model.lower()
    if "e2b" in m:
        return f"http://127.0.0.1:{PORT_E2B}/v1/chat/completions"
    return f"http://127.0.0.1:{PORT_12B}/v1/chat/completions"

async def manage_mcp_server_tool(arguments: Dict[str, Any]) -> str:
    action = arguments.get("action")
    name = arguments.get("name")
    command = arguments.get("command")
    args = arguments.get("args", [])
    description = arguments.get("description", "")
    
    mcp_config_path = "mcp/mcp_servers.json"
    
    if action == "reload":
        for client in list(mcp_clients.values()):
            await client.stop()
        mcp_clients.clear()
        await load_mcp_servers()
        return "Успешно перезапущены все MCP-серверы из конфига."
        
    if action == "list":
        status_list = []
        for c_name, client in mcp_clients.items():
            status_list.append({
                "name": c_name,
                "status": "running" if client.proc and client.proc.returncode is None else "stopped",
                "tools": [t["name"] for t in client.tools]
            })
        return json.dumps(status_list, ensure_ascii=False, indent=2)

    if not name:
        return "Ошибка: Не указано имя MCP-сервера (параметр 'name')."
        
    config = {"mcpServers": {}}
    if os.path.exists(mcp_config_path):
        try:
            with open(mcp_config_path, "r", encoding="utf-8") as f:
                lines = [l for l in f.readlines() if not l.strip().startswith("//")]
                config = json.loads("".join(lines))
        except Exception as e:
            log.error(f"Error parsing mcp_servers.json: {e}")
            
    if "mcpServers" not in config:
        config["mcpServers"] = {}
        
    if action == "add":
        if not command:
            return "Ошибка: Не указана команда запуска (параметр 'command') для добавления сервера."
        
        config["mcpServers"][name] = {
            "command": command,
            "args": args,
            "description": description
        }
        
        try:
            with open(mcp_config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            return f"Ошибка записи конфигурации на диск: {e}"
            
        if name in mcp_clients:
            await mcp_clients[name].stop()
            mcp_clients.pop(name)
            
        client = StdioMcpClient(name, command, args)
        await client.start()
        mcp_clients[name] = client
        
        tools_added = [t["name"] for t in client.tools]
        return f"Успешно добавлен и запущен MCP-сервер '{name}'. Подключенные инструменты: {tools_added}"
        
    elif action == "remove":
        if name not in config["mcpServers"]:
            return f"Сервер '{name}' не найден в конфигурации."
            
        config["mcpServers"].pop(name)
        
        try:
            with open(mcp_config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            return f"Ошибка записи конфигурации на диск: {e}"
            
        if name in mcp_clients:
            await mcp_clients[name].stop()
            mcp_clients.pop(name)
            
        return f"Успешно удален и остановлен MCP-сервер '{name}'."
        
    return f"Неизвестное действие: {action}"

# Expose local tools for llama-server function calling
def get_mcp_tools_list() -> List[Dict[str, Any]]:
    tools = []
    
    # Add gateway native tools
    tools.append({
        "type": "function",
        "function": {
            "name": "gateway__manage_mcp_server",
            "description": "Управление MCP-серверами: добавление (add), удаление (remove), список (list) или перезапуск (reload). Позволяет динамически подключать новые API и инструменты.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "remove", "list", "reload"], "description": "Действие для выполнения"},
                    "name": {"type": "string", "description": "Имя MCP-сервера (для add/remove)"},
                    "command": {"type": "string", "description": "Команда запуска (например, 'npx' или 'python')"},
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Аргументы запуска"
                    },
                    "description": {"type": "string", "description": "Описание сервера"}
                },
                "required": ["action"]
            }
        }
    })
    
    for client in mcp_clients.values():
        for t in client.tools:
            # Format tool description according to OpenAI spec
            tools.append({
                "type": "function",
                "function": {
                    "name": f"{client.name}__{t['name']}",
                    "description": t.get("description", ""),
                    "parameters": t.get("inputSchema", {"type": "object", "properties": {}})
                }
            })
    return tools

async def execute_tool_call(tool_name: str, arguments: Dict[str, Any]) -> str:
    if tool_name == "gateway__manage_mcp_server":
        return await manage_mcp_server_tool(arguments)
        
    if "__" not in tool_name:
        return f"Error: Invalid tool name format {tool_name}"
        
    server_name, actual_tool_name = tool_name.split("__", 1)
    if server_name not in mcp_clients:
        return f"Error: MCP Server {server_name} not found"
        
    client = mcp_clients[server_name]
    log.info(f"Executing tool {actual_tool_name} on {server_name} with args: {arguments}")
    
    try:
        res = await client.call_method("tools/call", {
            "name": actual_tool_name,
            "arguments": arguments
        })
        content = res.get("content", [])
        text_outputs = [c["text"] for c in content if c.get("type") == "text"]
        return "\n".join(text_outputs) if text_outputs else str(res)
    except Exception as e:
        log.error(f"Error executing tool: {e}")
        return f"Error executing tool: {e}"

# OpenAI Chat completions with embedded Agent Loop
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "gemma-4-12b")
    messages = body.get("messages", [])
    
    backend_url = get_backend_url(model)
    tools = get_mcp_tools_list()
    
    # We execute the agent loop
    max_iterations = 8
    current_messages = list(messages)
    
    # Inject tools to the request body if model supports it
    body_with_tools = dict(body)
    if tools:
        body_with_tools["tools"] = tools
        body_with_tools["tool_choice"] = "auto"

    async def event_generator():
        # Yield streaming response or run full agent loop
        # For simplicity, we execute the agent loop non-streaming, but yield tokens
        # If stream is True, we yield JSON-chunks, otherwise we return normal response
        nonlocal current_messages
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            for iteration in range(max_iterations):
                log.info(f"Agent Loop iteration {iteration + 1}/{max_iterations}")
                
                # Update messages in payload
                body_with_tools["messages"] = current_messages
                
                # Call backend LLM
                response = await client.post(backend_url, json=body_with_tools)
                if response.status_code != 200:
                    yield f"data: {json.dumps({'error': response.text})}\n\n"
                    return
                    
                res_data = response.json()
                choice = res_data["choices"][0]
                message = choice.get("message", {})
                
                tool_calls = message.get("tool_calls", [])
                content = message.get("content", "")
                
                if not tool_calls:
                    # Final response reached, yield text
                    if body.get("stream"):
                        # Format as stream chunk
                        chunk = {
                            "id": f"chatcmpl-{uuid.uuid4()}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": content},
                                "finish_reason": "stop"
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                    else:
                        yield json.dumps(res_data)
                    return
                
                # Yield a status update to the client that tools are being run
                if body.get("stream"):
                    status_text = f"\n*[Запуск инструментов... Попытка {iteration + 1}]*\n"
                    chunk = {
                        "choices": [{
                            "index": 0,
                            "delta": {"content": status_text},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                # Append assistant message with tool calls to history
                current_messages.append(message)
                
                # Execute all tool calls in parallel
                tool_tasks = []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    t_name = func.get("name")
                    try:
                        t_args = json.loads(func.get("arguments", "{}"))
                    except Exception:
                        t_args = {}
                    tool_tasks.append((tc.get("id"), t_name, t_args))
                    
                # Run tool execution
                for tc_id, t_name, t_args in tool_tasks:
                    tool_result = await execute_tool_call(t_name, t_args)
                    current_messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": t_name,
                        "content": tool_result
                    })
                    
                    if body.get("stream"):
                        # Log tool result into stream
                        clean_result = tool_result[:400] + "..." if len(tool_result) > 400 else tool_result
                        tool_log = f"\n* инструмент `{t_name}` вернул:\n```\n{clean_result}\n```\n"
                        chunk = {
                            "choices": [{
                                "index": 0,
                                "delta": {"content": tool_log},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
            
            # Loop ended without final text
            err_msg = "Error: Agent loop reached maximum iterations."
            if body.get("stream"):
                yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': err_msg}, 'finish_reason': 'length'}]})}\n\n"
            else:
                yield json.dumps({
                    "choices": [{
                        "message": {"role": "assistant", "content": err_msg},
                        "finish_reason": "length"
                    }]
                })

    # Wrap the generator
    if body.get("stream"):
        import time
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        # Non-streaming call, wait for the first item from generator
        async for item in event_generator():
            return JSONResponse(content=json.loads(item))

# ---------------------------------------------------------------------------
# HTML Web UI serving
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_webui():
    html_content = """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>KeglaAI — Панель Управления ИИ</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-primary: #0a0516;
                --bg-secondary: #120b24;
                --bg-tertiary: #191030;
                --accent-primary: #8a2be2;
                --accent-secondary: #4b0082;
                --text-primary: #f3effa;
                --text-secondary: #b5a9c6;
                --border-color: #2b1f48;
                --glass-bg: rgba(18, 11, 36, 0.65);
                --glass-border: rgba(138, 43, 226, 0.25);
            }
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
                font-family: 'Outfit', sans-serif;
            }
            body {
                background: linear-gradient(135deg, var(--bg-primary) 0%, #15092a 100%);
                color: var(--text-primary);
                height: 100vh;
                display: flex;
                overflow: hidden;
            }
            /* Sidebar */
            .sidebar {
                width: 280px;
                background-color: var(--bg-secondary);
                border-right: 1px solid var(--border-color);
                display: flex;
                flex-direction: column;
                padding: 20px;
                gap: 20px;
            }
            .brand {
                font-size: 1.5rem;
                font-weight: 700;
                background: linear-gradient(45deg, #d87093, #8a2be2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .sidebar-title {
                font-size: 0.8rem;
                text-transform: uppercase;
                color: var(--text-secondary);
                letter-spacing: 1px;
            }
            .btn-new-chat {
                background: linear-gradient(90deg, var(--accent-primary), #bf55ec);
                color: white;
                border: none;
                padding: 12px;
                border-radius: 8px;
                font-weight: 600;
                cursor: pointer;
                transition: opacity 0.2s;
            }
            .btn-new-chat:hover {
                opacity: 0.9;
            }
            .settings-panel {
                display: flex;
                flex-direction: column;
                gap: 15px;
                margin-top: auto;
                background: var(--bg-tertiary);
                padding: 15px;
                border-radius: 10px;
                border: 1px solid var(--border-color);
            }
            .settings-group {
                display: flex;
                flex-direction: column;
                gap: 5px;
            }
            label {
                font-size: 0.8rem;
                color: var(--text-secondary);
            }
            select, textarea, input {
                background-color: var(--bg-primary);
                color: var(--text-primary);
                border: 1px solid var(--border-color);
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 0.9rem;
                outline: none;
                width: 100%;
            }
            select:focus, textarea:focus {
                border-color: var(--accent-primary);
            }
            /* Main Chat Area */
            .chat-container {
                flex: 1;
                display: flex;
                flex-direction: column;
                background-color: transparent;
                position: relative;
            }
            .chat-header {
                height: 70px;
                border-bottom: 1px solid var(--border-color);
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0 30px;
                background: var(--glass-bg);
                backdrop-filter: blur(10px);
                z-index: 10;
            }
            .header-info {
                display: flex;
                flex-direction: column;
            }
            .active-model-title {
                font-weight: 700;
                font-size: 1.1rem;
            }
            .status-badge {
                font-size: 0.8rem;
                color: #26c281;
                display: flex;
                align-items: center;
                gap: 5px;
            }
            .status-dot {
                width: 8px;
                height: 8px;
                background-color: #26c281;
                border-radius: 50%;
            }
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 30px;
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            .message {
                max-width: 80%;
                padding: 16px 20px;
                border-radius: 12px;
                line-height: 1.6;
                position: relative;
            }
            .message.user {
                align-self: flex-end;
                background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
                border-bottom-right-radius: 2px;
            }
            .message.assistant {
                align-self: flex-start;
                background-color: var(--bg-secondary);
                border: 1px solid var(--border-color);
                border-bottom-left-radius: 2px;
            }
            /* Code Block Styling */
            pre {
                background-color: var(--bg-primary);
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
                overflow-x: auto;
                border: 1px solid var(--border-color);
            }
            code {
                font-family: 'Fira Code', monospace;
                font-size: 0.85rem;
            }
            /* Tool execution block */
            .tool-execution {
                color: #f39c12;
                font-size: 0.85rem;
                background-color: rgba(243, 156, 18, 0.1);
                border: 1px dashed rgba(243, 156, 18, 0.3);
                padding: 10px 15px;
                border-radius: 6px;
                margin: 5px 0;
                font-family: 'Fira Code', monospace;
            }
            /* Input Area */
            .chat-input-area {
                padding: 20px 30px;
                background: var(--glass-bg);
                backdrop-filter: blur(10px);
                border-top: 1px solid var(--border-color);
            }
            .input-wrapper {
                position: relative;
                display: flex;
                gap: 12px;
            }
            .chat-input {
                flex: 1;
                background-color: var(--bg-primary);
                color: var(--text-primary);
                border: 1px solid var(--border-color);
                border-radius: 10px;
                padding: 15px 20px;
                font-size: 1rem;
                outline: none;
                resize: none;
                height: 56px;
                max-height: 200px;
                transition: border-color 0.2s;
            }
            .chat-input:focus {
                border-color: var(--accent-primary);
            }
            .btn-send {
                background: linear-gradient(90deg, var(--accent-primary), #bf55ec);
                color: white;
                border: none;
                width: 56px;
                height: 56px;
                border-radius: 10px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: opacity 0.2s;
            }
            .btn-send:hover {
                opacity: 0.9;
            }
            /* Typing indicator */
            .typing-indicator {
                display: flex;
                align-items: center;
                gap: 5px;
                margin-left: 10px;
            }
            .typing-dot {
                width: 6px;
                height: 6px;
                background-color: var(--text-secondary);
                border-radius: 50%;
                animation: typing 1.4s infinite;
            }
            .typing-dot:nth-child(2) { animation-delay: 0.2s; }
            .typing-dot:nth-child(3) { animation-delay: 0.4s; }
            @keyframes typing {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-6px); }
            }
        </style>
    </head>
    <body>
        <div class="sidebar">
            <div class="brand">
                <span>🛸</span> KeglaAI
            </div>
            <button class="btn-new-chat" onclick="newChat()">+ Новый диалог</button>
            <div class="sidebar-title">История</div>
            <div id="chat-history-list" style="overflow-y:auto; flex:1;"></div>
            
            <div class="settings-panel">
                <div class="settings-group">
                    <label for="model-select">Модель</label>
                    <select id="model-select" onchange="updateActiveModel()">
                        <option value="gemma-4-12b">Gemma-4-12B (GPU 0)</option>
                        <option value="gemma-4-e2b">Gemma-4-E2B (GPU 1)</option>
                    </select>
                </div>
                <div class="settings-group">
                    <label for="system-prompt">Системный промпт</label>
                    <textarea id="system-prompt" rows="4">Ты — продвинутый ИИ-ассистент. Ты можешь запускать bash-команды, писать и изменять файлы проекта, а также администрировать этот сервер. Используй инструменты автономно.</textarea>
                </div>
            </div>
        </div>
        <div class="chat-container">
            <div class="chat-header">
                <div class="header-info">
                    <div class="active-model-title" id="active-model-display">Gemma-4-12B</div>
                    <div class="status-badge"><div class="status-dot"></div> Готов (Dual-GPU)</div>
                </div>
            </div>
            <div class="chat-messages" id="chat-messages">
                <div class="message assistant">
                    Привет! Я ИИ-ассистент KeglaAI. Я полностью автономен и могу выполнять ваши задачи напрямую на сервере (запускать bash-команды, изменять файлы проекта, настраивать MCP). Чем могу помочь?
                </div>
            </div>
            <div class="chat-input-area">
                <div class="input-wrapper">
                    <textarea class="chat-input" id="chat-input" placeholder="Введите ваш запрос... (Shift + Enter для новой строки)" onkeydown="handleKeyDown(event)"></textarea>
                    <button class="btn-send" onclick="sendMessage()">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="22" y1="2" x2="11" y2="13"></line>
                            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                        </svg>
                    </button>
                </div>
            </div>
        </div>

        <script>
            let messages = [];

            function updateActiveModel() {
                const model = document.getElementById("model-select").value;
                document.getElementById("active-model-display").innerText = model === "gemma-4-12b" ? "Gemma-4-12B" : "Gemma-4-E2B";
            }

            function handleKeyDown(event) {
                if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    sendMessage();
                }
            }

            function newChat() {
                document.getElementById("chat-messages").innerHTML = `
                    <div class="message assistant">
                        Диалог очищен. Начните новую беседу!
                    </div>
                `;
                messages = [];
            }

            async function sendMessage() {
                const inputEl = document.getElementById("chat-input");
                const text = inputEl.value.trim();
                if (!text) return;

                inputEl.value = "";
                
                // Add user message to UI
                const chatMessages = document.getElementById("chat-messages");
                chatMessages.innerHTML += `<div class="message user">${escapeHtml(text)}</div>`;
                chatMessages.scrollTop = chatMessages.scrollHeight;

                messages.push({"role": "user", "content": text});

                // Add assistant typing indicator
                const typingId = "typing-" + Date.now();
                chatMessages.innerHTML += `
                    <div class="message assistant" id="${typingId}">
                        <div class="typing-indicator">
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                        </div>
                    </div>
                `;
                chatMessages.scrollTop = chatMessages.scrollHeight;

                const model = document.getElementById("model-select").value;
                const sysPrompt = document.getElementById("system-prompt").value;

                const payloadMessages = [
                    {"role": "system", "content": sysPrompt},
                    ...messages
                ];

                try {
                    const response = await fetch("/v1/chat/completions", {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify({
                            model: model,
                            messages: payloadMessages,
                            stream: true
                        })
                    });

                    const typingEl = document.getElementById(typingId);
                    typingEl.innerHTML = ""; // Clear typing indicator

                    const reader = response.body.getReader();
                    const decoder = new TextDecoder("utf-8");
                    let done = false;
                    let accumulatedText = "";

                    while (!done) {
                        const {value, done: doneReading} = await reader.read();
                        done = doneReading;
                        const chunk = decoder.decode(value);
                        
                        const lines = chunk.split("\\n");
                        for (const line of lines) {
                            if (line.startsWith("data: ")) {
                                const dataStr = line.slice(6).trim();
                                if (dataStr === "[DONE]") continue;
                                try {
                                    const parsed = JSON.parse(dataStr);
                                    const delta = parsed.choices[0].delta;
                                    if (delta && delta.content) {
                                        accumulatedText += delta.content;
                                        typingEl.innerHTML = formatMarkdown(accumulatedText);
                                        chatMessages.scrollTop = chatMessages.scrollHeight;
                                    }
                                } catch (e) {}
                            }
                        }
                    }
                    messages.push({"role": "assistant", "content": accumulatedText});
                } catch (e) {
                    console.error(e);
                    document.getElementById(typingId).innerText = "Ошибка соединения с сервером.";
                }
            }

            function escapeHtml(text) {
                return text
                    .replace(/&/g, "&amp;")
                    .replace(/</g, "&lt;")
                    .replace(/>/g, "&gt;")
                    .replace(/"/g, "&quot;")
                    .replace(/'/g, "&#039;");
            }

            function formatMarkdown(text) {
                // Quick and dirty formatter for preview / logs / code
                let html = escapeHtml(text);
                
                // Format code blocks
                html = html.replace(/```([\\s\\S]*?)```/g, function(match, code) {
                    return `<pre><code>${code.trim()}</code></pre>`;
                });
                
                // Format inline code
                html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
                
                // Format tool calls status
                html = html.replace(/\\n\\*\\[Запуск инструментов[^\\]]+\\]\\*\\n/g, '<div class="tool-execution">⚙️ Запуск инструментов...</div>');
                html = html.replace(/\\n\\* инструмент `([^`]+)` вернул:\\n```\\n([\\s\\S]*?)```\\n/g, '<div class="tool-execution">🛠️ Инструмент <code>$1</code> выполнен.<br>Вывод:<br><pre><code>$2</code></pre></div>');

                // Newlines
                html = html.replace(/\\n/g, "<br>");
                return html;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
