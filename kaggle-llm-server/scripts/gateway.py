import asyncio
import json
import logging
import os
import sys
import time
import uuid
import httpx
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request, HTTPException, Response
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
PORT_12B = 8084
PORT_E2B = 8083

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
    asyncio.create_task(load_mcp_servers())

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

# Proxy catch-all route to llama-server to serve the built-in Web UI
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def catch_all_proxy(request: Request, path: str):
    # Default backend is Gemma-4-12B on port 8083, which contains all static UI assets
    target_url = f"http://127.0.0.1:8083/{path}"
    
    query_params = request.url.query
    if query_params:
        target_url += f"?{query_params}"
        
    body = await request.body()
    
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
            )
            
            # Remove content-encoding and other transit-sensitive headers
            exclude_headers = {"content-encoding", "content-length", "transfer-encoding", "connection"}
            resp_headers = {k: v for k, v in resp.headers.items() if k.lower() not in exclude_headers}
            
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=resp_headers,
            )
    except Exception as e:
        log.error(f"Error in catch_all_proxy for {path}: {e}")
        raise HTTPException(status_code=502, detail=f"Bad Gateway proxying to llama-server: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
