#!/usr/bin/env python3
"""
scripts/telegram_bot.py

Telegram Bot Agent Loop with MCP orchestration.
Designed to run inside a Kaggle Notebook alongside llama-server (GPU 0)
and media_server (GPU 1).

Features:
  - MCPOrchestrator: manages all enabled stdio MCP processes (filesystem, fetch, memory)
    and remote/local SSE MCP endpoints (media_server) using the official `mcp` SDK.
  - Agent loop with tool-calling: sends chat history to llama-server,
    intercepts tool_calls, runs them via MCPOrchestrator, and feeds results back.
  - Dynamic Installer system tools: LLM can search NPM Registry for MCP servers,
    install/register them dynamically in SQLite, and start them on the fly.
  - Premium Media Delivery: uploads generated images/videos directly to the Telegram chat
    as Photo/Video messages and feeds back metadata to LLM context.
  - Whitelist security, chat logs auditing, and full Telegram commands menu.
"""

import asyncio
import base64
import io
import json
import logging
import os
import sys
import time
import uuid
from typing import Optional

# Ensure project modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot_db import BotDatabase

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("telegram_bot")

# ---------------------------------------------------------------------------
# Configuration (env vars / Kaggle Secrets)
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
LLM_API_URL = os.environ.get("LLM_API_URL", "http://127.0.0.1:8080/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "sk-no-key-required")
MEDIA_SERVER_URL = ""
MAX_AGENT_ITERATIONS = int(os.environ.get("MAX_AGENT_ITERATIONS", "10"))
MAX_HISTORY_MESSAGES = int(os.environ.get("MAX_HISTORY_MESSAGES", "40"))

SYSTEM_PROMPT = os.environ.get("BOT_SYSTEM_PROMPT", (
    "Ты — продвинутый AI-ассистент с доступом к инструментам.\n"
    "Ты умеешь читать файлы проекта, делать веб-запросы и управлять MCP-инструментами.\n"
    "Отвечай на русском языке. Используй инструменты, когда это уместно, не спрашивая разрешения."
))


# ---------------------------------------------------------------------------
# MCP Orchestrator — manages multiple stdio & SSE Client Sessions
# ---------------------------------------------------------------------------
class MCPOrchestrator:
    """
    Manages active MCP server connections. Spawns stdio processes (via npx, python, etc.)
    and connects to SSE streams, maintaining persistent ClientSessions.
    """

    def __init__(self, db: BotDatabase):
        from mcp import ClientSession
        self.db = db
        self.sessions: dict[str, ClientSession] = {}
        self.cached_tools: dict[str, list] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def start_all(self):
        """Start all enabled auto-start servers from the database."""
        servers = self.db.list_mcp_servers(enabled_only=True)
        for s in servers:
            if s.get("auto_start"):
                asyncio.create_task(self.start_server(s["name"]))

    async def start_server(self, name: str) -> bool:
        """Start a single MCP server connection in the background."""
        async with self._lock:
            if name in self.sessions:
                return True

            cfg = self.db.get_mcp_server(name)
            if not cfg or not cfg.get("enabled"):
                log.warning(f"Server '{name}' is not registered or is disabled.")
                return False

            log.info(f"Starting MCP server '{name}'...")
            task = asyncio.create_task(self._run_server_loop(name, cfg))
            self.tasks[name] = task

            # Wait for initialization (up to 5 seconds)
            for _ in range(50):
                if name in self.sessions:
                    return True
                await asyncio.sleep(0.1)

            log.warning(f"MCP Server '{name}' failed to initialize in time.")
            return False

    async def _run_server_loop(self, name: str, cfg: dict):
        from mcp import ClientSession
        
        max_restarts = 3
        backoff = [5, 15, 30]
        restart_count = 0

        while True:
            try:
                if cfg["server_type"] == "stdio":
                    from mcp import stdio_client, StdioServerParameters

                    cmd = cfg["command"]
                    if os.name == "nt":
                        if cmd == "npx":
                            cmd = "npx.cmd"
                        elif cmd == "npm":
                            cmd = "npm.cmd"

                    params = StdioServerParameters(
                        command=cmd,
                        args=cfg["args"],
                        env={**os.environ, **cfg["env"]}
                    )
                    log.info(f"Spawning stdio process for '{name}': {cmd} {' '.join(cfg['args'])}")
                    async with stdio_client(params) as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            tools_res = await session.list_tools()
                            self.sessions[name] = session
                            self.cached_tools[name] = tools_res.tools
                            log.info(f"MCP Server '{name}' (stdio) active with {len(tools_res.tools)} tools.")
                            
                            # Reset restart counter upon successful connection
                            restart_count = 0
                            
                            while True:
                                await asyncio.sleep(1)

                elif cfg["server_type"] == "sse":
                    from mcp.client.sse import sse_client
                    log.info(f"Connecting to SSE endpoint for '{name}': {cfg['url']}")
                    async with sse_client(cfg["url"]) as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            tools_res = await session.list_tools()
                            self.sessions[name] = session
                            self.cached_tools[name] = tools_res.tools
                            log.info(f"MCP Server '{name}' (SSE) active with {len(tools_res.tools)} tools.")
                            
                            restart_count = 0
                            
                            while True:
                                await asyncio.sleep(1)

            except asyncio.CancelledError:
                log.info(f"MCP Server '{name}' connection loop cancelled.")
                break
            except Exception as e:
                log.error(f"Error in MCP Server '{name}' connection loop: {e}", exc_info=True)
                self.sessions.pop(name, None)
                self.cached_tools.pop(name, None)

                # Process Warden recovery logic
                if restart_count < max_restarts:
                    sleep_time = backoff[min(restart_count, len(backoff) - 1)]
                    restart_count += 1
                    log.warning(
                        f"[Process Warden] MCP Server '{name}' connection dropped/crashed. "
                        f"Restarting in {sleep_time}s... (attempt {restart_count}/{max_restarts})"
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    log.error(
                        f"[Process Warden] MCP Server '{name}' failed repeatedly after "
                        f"{max_restarts} attempts. Giving up."
                    )
                    break
        
        # Cleanup
        self.sessions.pop(name, None)
        self.cached_tools.pop(name, None)
        self.tasks.pop(name, None)

    async def stop_server(self, name: str):
        """Stop and clean up a server connection."""
        async with self._lock:
            task = self.tasks.get(name)
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            self.sessions.pop(name, None)
            self.cached_tools.pop(name, None)
            self.tasks.pop(name, None)
            log.info(f"MCP Server '{name}' connection stopped.")

    async def stop_all(self):
        """Stop all active MCP server sessions."""
        names = list(self.tasks.keys())
        for name in names:
            await self.stop_server(name)

    def get_openai_tools(self) -> list[dict]:
        """Convert cached tools to OpenAI function calling format."""
        openai_tools = []
        for server_name, tools in self.cached_tools.items():
            for t in tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.inputSchema,
                    }
                })
        return openai_tools

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Find the server hosting the tool and execute it."""
        target_server = None
        for server_name, tools in self.cached_tools.items():
            if any(t.name == tool_name for t in tools):
                target_server = server_name
                break

        if not target_server:
            return {
                "text": f"Error: Tool '{tool_name}' not found on any active MCP server.",
                "images": [],
                "files": [],
                "success": False
            }

        session = self.sessions.get(target_server)
        if not session:
            return {
                "text": f"Error: MCP Server '{target_server}' is offline.",
                "images": [],
                "files": [],
                "success": False
            }

        try:
            result = await session.call_tool(tool_name, arguments)
            texts = []
            images = []
            files = []

            for block in result.content:
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    texts.append(block.text)
                    # Detect files in JSON metadata output
                    try:
                        data = json.loads(block.text)
                        if isinstance(data, dict):
                            fpath = data.get("file")
                            if fpath and os.path.exists(fpath):
                                files.append(fpath)
                    except Exception:
                        pass
                elif block_type == "image":
                    images.append(block.data)

            return {
                "text": "\n".join(texts) if texts else "Executed successfully.",
                "images": images,
                "files": files,
                "success": not getattr(result, "isError", False)
            }
        except Exception as e:
            return {
                "text": f"Error during tool call '{tool_name}': {str(e)}",
                "images": [],
                "files": [],
                "success": False
            }


# ---------------------------------------------------------------------------
# LLM Client — llama-server API wrapper
# ---------------------------------------------------------------------------
class LLMClient:
    """Client for llama-server OpenAI-compatible API."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> dict:
        import httpx

        payload = {
            "model": "local-model",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=150.0) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )

        if resp.status_code != 200:
            raise RuntimeError(f"LLM API returned code {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        return data["choices"][0]["message"]


# ---------------------------------------------------------------------------
# NPM Registry Search — helper for dynamic MCP discovery
# ---------------------------------------------------------------------------
async def search_mcp_registry(query: str) -> str:
    """Query NPM registry for packages containing 'mcp-server' or query terms."""
    import httpx
    url = f"https://registry.npmjs.org/-/v1/search?text=mcp-server {query}&size=8"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return f"NPM Search failed (status: {resp.status_code})"
            data = resp.json()
            packages = data.get("objects", [])
            if not packages:
                return f"No NPM packages matching '{query}' were found."

            results = ["NPM MCP Search results:\n"]
            for obj in packages:
                pkg = obj.get("package", {})
                results.append(
                    f"📦 *{pkg.get('name')}* (v{pkg.get('version')})\n"
                    f"   _{pkg.get('description', 'No description')}_\n"
                    f"   Command: `npx -y {pkg.get('name')}`\n"
                )
            return "\n".join(results)
    except Exception as e:
        return f"Error connecting to NPM Registry: {str(e)}"


# ---------------------------------------------------------------------------
# System Tools Schema (Dynamic Installer)
# ---------------------------------------------------------------------------
SYSTEM_TOOLS_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_mcp_registry",
            "description": "Поиск MCP-серверов в публичном NPM реестре по ключевым словам.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос (например: 'postgres', 'calculator')"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "install_mcp_server",
            "description": "Загрузка, регистрация в SQLite и немедленный запуск нового stdio MCP сервера.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Короткое имя сервера латиницей (например: 'calculator')"
                    },
                    "command": {
                        "type": "string",
                        "description": "Команда запуска (обычно 'npx')"
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Аргументы запуска (например: ['-y', '@modelcontextprotocol/server-postgres'])"
                    },
                    "env": {
                        "type": "object",
                        "description": "Дополнительные env-переменные в виде JSON-словаря (опционально)",
                        "additionalProperties": {"type": "string"}
                    },
                    "description": {
                        "type": "string",
                        "description": "Краткое описание назначения сервера"
                    }
                },
                "required": ["name", "command", "args"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "uninstall_mcp_server",
            "description": "Остановка и полное удаление MCP сервера из системы.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Имя удаляемого сервера"
                    }
                },
                "required": ["name"]
            }
        }
    }
]


# ---------------------------------------------------------------------------
# Core Agent Loop
# ---------------------------------------------------------------------------
async def agent_loop(
    user_message: str,
    chat_id: int,
    db: BotDatabase,
    llm: LLMClient,
    orchestrator: MCPOrchestrator,
    progress_callback=None,
    media_callback=None,
) -> str:
    """Runs the LLM function-calling loop, executing local tools or installing new ones."""
    db.add_message(chat_id, "user", user_message)

    history = db.get_history(chat_id, limit=MAX_HISTORY_MESSAGES)
    if not history or history[0]["role"] != "system":
        system_prompt = db.get_setting("system_prompt") or SYSTEM_PROMPT
        messages = [{"role": "system", "content": system_prompt}] + history
    else:
        messages = history

    # Merge system tools with dynamic MCP tools
    tools = SYSTEM_TOOLS_SCHEMAS + orchestrator.get_openai_tools()

    for iteration in range(MAX_AGENT_ITERATIONS):
        log.info(f"Agent loop iteration {iteration + 1}/{MAX_AGENT_ITERATIONS}")

        try:
            response_msg = await llm.chat_completion(messages, tools=tools or None)
        except Exception as e:
            err_msg = f"LLM error: {str(e)}"
            log.error(err_msg)
            db.add_message(chat_id, "assistant", err_msg)
            return err_msg

        tool_calls = response_msg.get("tool_calls")
        content = response_msg.get("content", "")

        if not tool_calls:
            if content:
                db.add_message(chat_id, "assistant", content)
            return content or "(empty response)"

        # Save assistant message with tool calls
        db.add_message(
            chat_id, "assistant",
            content=content,
            tool_calls=[
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
                for tc in tool_calls
            ]
        )
        messages.append(response_msg)

        # Process each tool call
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            try:
                fn_args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                fn_args = {}

            log.info(f"  Tool call: {fn_name}({json.dumps(fn_args, ensure_ascii=False)[:100]})")
            if progress_callback:
                await progress_callback(f"Calling tool: {fn_name}...")

            start_time = time.time()
            tool_text = ""
            success = True
            files = []
            images = []

            # 1. System/Meta tools
            if fn_name == "search_mcp_registry":
                tool_text = await search_mcp_registry(fn_args.get("query", ""))
            elif fn_name == "install_mcp_server":
                name = fn_args.get("name")
                cmd = fn_args.get("command")
                args = fn_args.get("args", [])
                env = fn_args.get("env", {})
                desc = fn_args.get("description", "")
                db.add_mcp_server(
                    name=name, server_type="stdio", command=cmd,
                    args=args, env=env, description=desc
                )
                success = await orchestrator.start_server(name)
                tool_text = (
                    f"Server '{name}' registered and started successfully."
                    if success else f"Server '{name}' registered, but failed to initialize."
                )
            elif fn_name == "uninstall_mcp_server":
                name = fn_args.get("name")
                await orchestrator.stop_server(name)
                removed = db.remove_mcp_server(name)
                tool_text = (
                    f"Server '{name}' uninstalled successfully."
                    if removed else f"Server '{name}' stopped but was not in database."
                )
            # 2. Dynamic MCP tools
            else:
                tool_res = await orchestrator.call_tool(fn_name, fn_args)
                tool_text = tool_res["text"]
                success = tool_res["success"]
                files = tool_res["files"]
                images = tool_res["images"]

            duration_ms = int((time.time() - start_time) * 1000)

            # Audit logging
            db.log_tool_call(
                tool_name=fn_name,
                arguments=json.dumps(fn_args, ensure_ascii=False),
                result=tool_text,
                duration_ms=duration_ms,
                success=success,
                chat_id=chat_id,
                server_name="system" if fn_name in ["search_mcp_registry", "install_mcp_server", "uninstall_mcp_server"] else "mcp-server"
            )

            # Send files/images directly to user if requested
            if (files or images) and media_callback:
                await media_callback(files, images)
                # Modify tool result to tell LLM that file was uploaded directly
                tool_text += f"\n[System notification: files/images were successfully uploaded directly to the user's chat.]"

            # Feed result back to conversation context
            tool_msg = {
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": tool_text,
            }
            messages.append(tool_msg)
            db.add_message(
                chat_id, "tool",
                content=tool_text,
                tool_call_id=tc["id"],
                name=fn_name
            )

            log.info(f"  Tool result: {tool_text[:120]}... ({duration_ms}ms)")

    final_msg = "Agent loop: maximum iterations reached."
    db.add_message(chat_id, "assistant", final_msg)
    return final_msg


# ---------------------------------------------------------------------------
# Telegram Bot Core
# ---------------------------------------------------------------------------
async def run_bot():
    from telegram import Update, BotCommand
    from telegram.ext import (
        ApplicationBuilder,
        CommandHandler,
        MessageHandler,
        ContextTypes,
        filters,
    )

    # Get Token
    token = TELEGRAM_BOT_TOKEN
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("TELEGRAM_BOT_TOKEN")
            log.info("Loaded Telegram Token from Kaggle Secrets.")
        except Exception:
            pass

    if not token:
        log.error("TELEGRAM_BOT_TOKEN is missing. Exiting.")
        sys.exit(1)

    # Initialize DB & Orchestrator
    db = BotDatabase()
    db.seed_from_mcp_json()
    log.info(f"Database ready: {db.db_path}")

    orchestrator = MCPOrchestrator(db)
    # Start auto-start servers
    await orchestrator.start_all()

    llm = LLMClient(LLM_API_URL, LLM_API_KEY)

    # --- Command Handlers ---

    async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not db.is_user_allowed(user.id):
            await update.message.reply_text("Access denied.")
            return
        await update.message.reply_text(
            f"Привет, {user.first_name}!\n\n"
            "Я умный AI-ассистент на базе llama.cpp.\n"
            "Я поддерживаю динамическую установку инструментов (MCP) через чат.\n\n"
            "Доступные команды:\n"
            "/clear — очистить историю диалога\n"
            "/mcp_list — список MCP серверов и инструментов\n"
            "/mcp_search <запрос> — поиск новых MCP в реестре\n"
            "/mcp_install <npm-пакет> — быстрая установка MCP\n"
            "/status — проверить состояние кластера\n"
            "/help — показать эту справку"
        )

    async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not db.is_user_allowed(update.effective_user.id):
            return
        db.clear_history(update.effective_chat.id)
        await update.message.reply_text("История переписки стёрта.")

    async def cmd_mcp_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not db.is_user_allowed(update.effective_user.id):
            return
        servers = db.list_mcp_servers()
        if not servers:
            await update.message.reply_text("Нет зарегистрированных MCP серверов.")
            return

        lines = ["📋 *Зарегистрированные MCP серверы:*\n"]
        for s in servers:
            status = "🟢 ON" if s["name"] in orchestrator.sessions else "🔴 OFF"
            lines.append(f"{status} *{s['name']}* ({s['server_type']})")
            if s.get("description"):
                lines.append(f"   _{s['description'][:80]}_\n")

        # List all cached tools
        lines.append("\n🛠️ *Доступные инструменты:*")
        has_tools = False
        for server_name, tools in orchestrator.cached_tools.items():
            for t in tools:
                has_tools = True
                lines.append(f"  • `{t.name}` (из {server_name})")
        if not has_tools:
            lines.append("  (нет активных инструментов)")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    async def cmd_mcp_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not db.is_user_allowed(update.effective_user.id):
            return
        if not context.args:
            await update.message.reply_text("Использование: /mcp_search <запрос>")
            return
        query = " ".join(context.args)
        await update.message.reply_text(f"Поиск MCP-серверов для '{query}'...")
        results = await search_mcp_registry(query)
        await update.message.reply_text(results, parse_mode="Markdown")

    async def cmd_mcp_install(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not db.is_user_allowed(update.effective_user.id):
            return
        if not context.args:
            await update.message.reply_text("Использование: /mcp_install <npm_package>")
            return
        pkg = context.args[0]
        name = pkg.split("/")[-1].replace("mcp-server-", "").replace("server-", "")
        await update.message.reply_text(f"Установка и запуск MCP-сервера '{name}'...")

        db.add_mcp_server(
            name=name,
            server_type="stdio",
            command="npx",
            args=["-y", pkg],
            description=f"Manual install: {pkg}"
        )
        success = await orchestrator.start_server(name)
        if success:
            await update.message.reply_text(f"✅ MCP-сервер '{name}' успешно запущен!")
        else:
            await update.message.reply_text(f"❌ Ошибка инициализации сервера '{name}'.")

    async def cmd_mcp_enable(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not db.is_user_allowed(update.effective_user.id):
            return
        if not context.args:
            await update.message.reply_text("Использование: /mcp_enable <name>")
            return
        name = context.args[0]
        db.set_mcp_enabled(name, True)
        success = await orchestrator.start_server(name)
        if success:
            await update.message.reply_text(f"✅ Сервер '{name}' включен и запущен.")
        else:
            await update.message.reply_text(f"❌ Ошибка запуска сервера '{name}'.")

    async def cmd_mcp_disable(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not db.is_user_allowed(update.effective_user.id):
            return
        if not context.args:
            await update.message.reply_text("Использование: /mcp_disable <name>")
            return
        name = context.args[0]
        db.set_mcp_enabled(name, False)
        await orchestrator.stop_server(name)
        await update.message.reply_text(f"🔴 Сервер '{name}' выключен и остановлен.")

    async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not db.is_user_allowed(update.effective_user.id):
            return
        stats = db.get_history_stats(update.effective_chat.id)
        lines = [
            "📊 *Состояние системы:*",
            f"  • LLM API: `{LLM_API_URL}`",
            f"  • Media Server: `{MEDIA_SERVER_URL}`",
            f"  • Подключено серверов: `{len(orchestrator.sessions)}`",
            f"  • Сообщений в чате: `{stats['message_count']}`",
            f"  • Использовано токенов: `{stats['total_tokens']}`",
        ]
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    # --- Message Handler (Agent Loop) ---

    async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not db.is_user_allowed(update.effective_user.id):
            await update.message.reply_text("Access denied.")
            return

        user_text = update.message.text
        if not user_text:
            return

        chat_id = update.effective_chat.id
        await update.effective_chat.send_action("typing")

        async def progress_callback(text: str):
            try:
                await update.effective_chat.send_action("typing")
            except Exception:
                pass

        async def media_callback(files: list[str], images: list[str]):
            # 1. Send files (video/images)
            for fpath in files:
                try:
                    if fpath.endswith(".mp4"):
                        log.info(f"Sending video file: {fpath}")
                        await update.message.reply_video(
                            video=open(fpath, "rb"),
                            caption=f"🎥 Сгенерированное видео\nПромпт: {user_text[:100]}"
                        )
                    elif fpath.endswith((".png", ".jpg", ".jpeg")):
                        log.info(f"Sending image file: {fpath}")
                        await update.message.reply_photo(
                            photo=open(fpath, "rb"),
                            caption=f"🎨 Сгенерированное изображение\nПромпт: {user_text[:100]}"
                        )
                except Exception as e:
                    log.error(f"Error sending file {fpath}: {e}")
                    await update.message.reply_text(f"[Ошибка при отправке файла: {str(e)[:150]}]")

            # 2. Send base64 fallback
            for img_b64 in images:
                try:
                    img_bytes = base64.b64decode(img_b64)
                    await update.message.reply_photo(
                        photo=io.BytesIO(img_bytes),
                        caption=f"🎨 Сгенерированное изображение"
                    )
                except Exception as e:
                    log.error(f"Error sending base64 image: {e}")

        try:
            response = await agent_loop(
                user_message=user_text,
                chat_id=chat_id,
                db=db,
                llm=llm,
                orchestrator=orchestrator,
                progress_callback=progress_callback,
                media_callback=media_callback,
            )

            # Telegram max limit is 4096 characters
            if len(response) > 4000:
                chunks = [response[i:i+4000] for i in range(0, len(response), 4000)]
                for chunk in chunks:
                    await update.message.reply_text(chunk)
            else:
                await update.message.reply_text(response or "(пустой ответ)")

        except Exception as e:
            log.error(f"Error in handle_message: {e}", exc_info=True)
            await update.message.reply_text(f"Произошла ошибка: {str(e)[:200]}")

    # --- Setup Application ---
    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("mcp_list", cmd_mcp_list))
    app.add_handler(CommandHandler("mcp_search", cmd_mcp_search))
    app.add_handler(CommandHandler("mcp_install", cmd_mcp_install))
    app.add_handler(CommandHandler("mcp_enable", cmd_mcp_enable))
    app.add_handler(CommandHandler("mcp_disable", cmd_mcp_disable))
    app.add_handler(CommandHandler("status", cmd_status))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Set command menu
    try:
        await app.bot.set_my_commands([
            BotCommand("start", "Показать приветствие"),
            BotCommand("clear", "Очистить историю диалога"),
            BotCommand("mcp_list", "Список MCP инструментов"),
            BotCommand("mcp_search", "Найти MCP в NPM"),
            BotCommand("mcp_install", "Установить MCP сервер"),
            BotCommand("status", "Состояние кластера"),
        ])
    except Exception as e:
        log.warning(f"Failed to set command menu: {e}")

    log.info("Bot is polling for updates...")
    async with app:
        await app.start()
        await app.updater.start_polling(
            drop_pending_updates=True,
            allowed_updates=["message"],
        )
        try:
            while True:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
            pass
        finally:
            log.info("Shutting down bot. Stopping all MCP servers...")
            try:
                await app.updater.stop()
            except Exception:
                pass
            try:
                await app.stop()
            except Exception:
                pass
            await orchestrator.stop_all()
            db.close()


if __name__ == "__main__":
    asyncio.run(run_bot())
