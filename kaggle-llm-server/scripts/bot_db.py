#!/usr/bin/env python3
"""
scripts/bot_db.py

SQLite database layer for the Telegram Bot Agent.
Provides persistent storage for:
  - Chat/conversation history (role, content, tool_calls)
  - MCP server configurations (name, command, args, env, status)
  - User settings and allowed user IDs

Kaggle-specific design:
  - DB lives at /kaggle/working/data/agent.db (persistent across cells)
  - Before session handover, the orchestrator calls:
      rclone copy ./data/agent.db gdrive:/sync/
  - On new session boot:
      rclone copy gdrive:/sync/agent.db ./data/
  - All paths are relative to the project root so the same code
    works both locally (dev) and on Kaggle (prod).
"""

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# On Kaggle: /kaggle/working/KeglaAI/kaggle-llm-server/data/agent.db
# Locally:   ./data/agent.db
DEFAULT_DB_DIR = os.environ.get("BOT_DB_DIR", "./data")
DEFAULT_DB_PATH = os.path.join(DEFAULT_DB_DIR, "agent.db")


class BotDatabase:
    """
    Async-safe SQLite database for the Telegram Bot.

    Thread-safety: SQLite in WAL mode + check_same_thread=False allows
    concurrent reads from the uvicorn/asyncio event loop. Writes are
    serialized internally by SQLite.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            timeout=10.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        """Create tables if they don't exist."""
        self._conn.executescript("""
            -- ---------------------------------------------------------------
            -- Allowed Telegram user IDs (whitelist)
            -- ---------------------------------------------------------------
            CREATE TABLE IF NOT EXISTS allowed_users (
                user_id     INTEGER PRIMARY KEY,
                username    TEXT,
                added_at    REAL DEFAULT (strftime('%s', 'now')),
                is_admin    INTEGER DEFAULT 0
            );

            -- ---------------------------------------------------------------
            -- Chat history (per Telegram chat_id)
            -- ---------------------------------------------------------------
            CREATE TABLE IF NOT EXISTS chat_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id     INTEGER NOT NULL,
                role        TEXT NOT NULL,           -- 'system', 'user', 'assistant', 'tool'
                content     TEXT,                    -- message text (nullable for tool_calls-only messages)
                tool_calls  TEXT,                    -- JSON array of tool_call objects (nullable)
                tool_call_id TEXT,                   -- for role='tool' responses
                name        TEXT,                    -- tool name for role='tool'
                timestamp   REAL DEFAULT (strftime('%s', 'now')),
                token_count INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_chat_history_chat
                ON chat_history(chat_id, timestamp);

            -- ---------------------------------------------------------------
            -- MCP server configurations
            -- ---------------------------------------------------------------
            CREATE TABLE IF NOT EXISTS mcp_servers (
                name        TEXT PRIMARY KEY,
                server_type TEXT NOT NULL DEFAULT 'stdio',  -- 'stdio' | 'sse'
                command     TEXT,                    -- for stdio: 'npx', 'python', etc.
                args        TEXT DEFAULT '[]',       -- JSON array of arguments
                env         TEXT DEFAULT '{}',       -- JSON dict of environment vars
                url         TEXT,                    -- for sse: the SSE endpoint URL
                description TEXT DEFAULT '',
                enabled     INTEGER DEFAULT 1,
                auto_start  INTEGER DEFAULT 1,       -- start on bot boot
                installed_at REAL DEFAULT (strftime('%s', 'now')),
                pid         INTEGER                  -- current process PID (NULL if not running)
            );

            -- ---------------------------------------------------------------
            -- Tool execution log (audit trail)
            -- ---------------------------------------------------------------
            CREATE TABLE IF NOT EXISTS tool_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id     INTEGER,
                server_name TEXT,
                tool_name   TEXT NOT NULL,
                arguments   TEXT,                    -- JSON
                result      TEXT,                    -- JSON (truncated for large outputs)
                duration_ms INTEGER,
                success     INTEGER DEFAULT 1,
                timestamp   REAL DEFAULT (strftime('%s', 'now'))
            );
            CREATE INDEX IF NOT EXISTS idx_tool_log_ts
                ON tool_log(timestamp);

            -- ---------------------------------------------------------------
            -- Key-value settings store
            -- ---------------------------------------------------------------
            CREATE TABLE IF NOT EXISTS settings (
                key         TEXT PRIMARY KEY,
                value       TEXT,
                updated_at  REAL DEFAULT (strftime('%s', 'now'))
            );
        """)
        self._conn.commit()

    def close(self):
        self._conn.close()

    # -- Settings -----------------------------------------------------------

    def get_setting(self, key: str, default: str = None) -> Optional[str]:
        row = self._conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str):
        self._conn.execute(
            "INSERT OR REPLACE INTO settings(key, value, updated_at) VALUES(?, ?, ?)",
            (key, value, time.time()),
        )
        self._conn.commit()

    # -- Allowed Users ------------------------------------------------------

    def is_user_allowed(self, user_id: int) -> bool:
        """Check if user_id is in the whitelist. Empty table = allow all."""
        count = self._conn.execute(
            "SELECT COUNT(*) FROM allowed_users"
        ).fetchone()[0]
        if count == 0:
            return True  # No whitelist configured = open access
        row = self._conn.execute(
            "SELECT 1 FROM allowed_users WHERE user_id = ?", (user_id,)
        ).fetchone()
        return row is not None

    def add_allowed_user(self, user_id: int, username: str = "", is_admin: bool = False):
        self._conn.execute(
            "INSERT OR REPLACE INTO allowed_users(user_id, username, is_admin) VALUES(?, ?, ?)",
            (user_id, username, 1 if is_admin else 0),
        )
        self._conn.commit()

    def list_allowed_users(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT user_id, username, is_admin FROM allowed_users ORDER BY added_at"
        ).fetchall()
        return [dict(r) for r in rows]

    # -- Chat History -------------------------------------------------------

    def add_message(
        self,
        chat_id: int,
        role: str,
        content: Optional[str] = None,
        tool_calls: Optional[list] = None,
        tool_call_id: Optional[str] = None,
        name: Optional[str] = None,
        token_count: int = 0,
    ):
        """Append a message to the chat history."""
        self._conn.execute(
            """INSERT INTO chat_history
               (chat_id, role, content, tool_calls, tool_call_id, name, token_count)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                chat_id,
                role,
                content,
                json.dumps(tool_calls) if tool_calls else None,
                tool_call_id,
                name,
                token_count,
            ),
        )
        self._conn.commit()

    def get_history(self, chat_id: int, limit: int = 50) -> list[dict]:
        """
        Retrieve the last `limit` messages for a chat, ordered oldest-first.
        Returns dicts ready for the OpenAI messages format.
        """
        rows = self._conn.execute(
            """SELECT role, content, tool_calls, tool_call_id, name
               FROM chat_history
               WHERE chat_id = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (chat_id, limit),
        ).fetchall()

        messages = []
        for row in reversed(rows):  # Reverse to get chronological order
            msg = {"role": row["role"]}
            if row["content"] is not None:
                msg["content"] = row["content"]
            if row["tool_calls"]:
                msg["tool_calls"] = json.loads(row["tool_calls"])
            if row["tool_call_id"]:
                msg["tool_call_id"] = row["tool_call_id"]
            if row["name"]:
                msg["name"] = row["name"]
            messages.append(msg)
        return messages

    def clear_history(self, chat_id: int):
        """Delete all messages for a chat."""
        self._conn.execute(
            "DELETE FROM chat_history WHERE chat_id = ?", (chat_id,)
        )
        self._conn.commit()

    def get_history_stats(self, chat_id: int) -> dict:
        """Get message count and total tokens for a chat."""
        row = self._conn.execute(
            """SELECT COUNT(*) as count, COALESCE(SUM(token_count), 0) as tokens
               FROM chat_history WHERE chat_id = ?""",
            (chat_id,),
        ).fetchone()
        return {"message_count": row["count"], "total_tokens": row["tokens"]}

    # -- MCP Servers --------------------------------------------------------

    def add_mcp_server(
        self,
        name: str,
        server_type: str = "stdio",
        command: str = "",
        args: list = None,
        env: dict = None,
        url: str = "",
        description: str = "",
        enabled: bool = True,
        auto_start: bool = True,
    ):
        """Register or update an MCP server configuration."""
        self._conn.execute(
            """INSERT OR REPLACE INTO mcp_servers
               (name, server_type, command, args, env, url, description, enabled, auto_start)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                name,
                server_type,
                command,
                json.dumps(args or []),
                json.dumps(env or {}),
                url,
                description,
                1 if enabled else 0,
                1 if auto_start else 0,
            ),
        )
        self._conn.commit()

    def get_mcp_server(self, name: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM mcp_servers WHERE name = ?", (name,)
        ).fetchone()
        if row:
            d = dict(row)
            d["args"] = json.loads(d["args"])
            d["env"] = json.loads(d["env"])
            return d
        return None

    def list_mcp_servers(self, enabled_only: bool = False) -> list[dict]:
        query = "SELECT * FROM mcp_servers"
        if enabled_only:
            query += " WHERE enabled = 1"
        query += " ORDER BY name"
        rows = self._conn.execute(query).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["args"] = json.loads(d["args"])
            d["env"] = json.loads(d["env"])
            result.append(d)
        return result

    def set_mcp_enabled(self, name: str, enabled: bool):
        self._conn.execute(
            "UPDATE mcp_servers SET enabled = ? WHERE name = ?",
            (1 if enabled else 0, name),
        )
        self._conn.commit()

    def set_mcp_pid(self, name: str, pid: Optional[int]):
        """Update the running PID for an MCP server process."""
        self._conn.execute(
            "UPDATE mcp_servers SET pid = ? WHERE name = ?",
            (pid, name),
        )
        self._conn.commit()

    def remove_mcp_server(self, name: str) -> bool:
        cursor = self._conn.execute(
            "DELETE FROM mcp_servers WHERE name = ?", (name,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    # -- Tool Log -----------------------------------------------------------

    def log_tool_call(
        self,
        tool_name: str,
        arguments: str = "",
        result: str = "",
        duration_ms: int = 0,
        success: bool = True,
        chat_id: int = 0,
        server_name: str = "",
    ):
        """Record a tool execution for auditing."""
        # Truncate large results to save space
        if len(result) > 4096:
            result = result[:4000] + f"\n... (truncated, {len(result)} bytes total)"
        self._conn.execute(
            """INSERT INTO tool_log
               (chat_id, server_name, tool_name, arguments, result, duration_ms, success)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (chat_id, server_name, tool_name, arguments, result, duration_ms,
             1 if success else 0),
        )
        self._conn.commit()

    def get_recent_tool_calls(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            """SELECT tool_name, server_name, success, duration_ms, timestamp
               FROM tool_log ORDER BY timestamp DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- Seed data: import from mcp_servers.json ----------------------------

    def seed_from_mcp_json(self, json_path: str = "./mcp/mcp_servers.json"):
        """
        Import MCP server configs from the project's mcp_servers.json
        if they haven't been added yet. This is called on first boot
        to pre-populate the DB with the default MCP toolkit.
        """
        if not os.path.exists(json_path):
            return

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        servers = data.get("mcpServers", {})
        imported = 0
        for name, cfg in servers.items():
            existing = self.get_mcp_server(name)
            if existing is None:
                self.add_mcp_server(
                    name=name,
                    server_type="stdio",
                    command=cfg.get("command", ""),
                    args=cfg.get("args", []),
                    env=cfg.get("env", {}),
                    description=cfg.get("description", ""),
                    enabled=True,
                    auto_start=True,
                )
                imported += 1

        # Also add the built-in media server (SSE)
        if self.get_mcp_server("media-server") is None:
            self.add_mcp_server(
                name="media-server",
                server_type="sse",
                url="http://127.0.0.1:8081/sse",
                description="GPU 1 media server: generate_image (FLUX.1 Dev), generate_video (Wan 2.1)",
                enabled=True,
                auto_start=False,  # Started separately by start.py
            )
            imported += 1

        return imported


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    # Test with a temp DB
    test_dir = tempfile.mkdtemp()
    db = BotDatabase(os.path.join(test_dir, "test_agent.db"))

    # Test settings
    db.set_setting("system_prompt", "Ты полезный AI-ассистент.")
    assert db.get_setting("system_prompt") == "Ты полезный AI-ассистент."

    # Test users
    db.add_allowed_user(123456, "testuser", is_admin=True)
    assert db.is_user_allowed(123456) is True
    assert db.is_user_allowed(999999) is False

    # Test chat history
    db.add_message(100, "user", "Привет!")
    db.add_message(100, "assistant", "Здравствуйте!")
    history = db.get_history(100)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["content"] == "Здравствуйте!"

    # Test MCP servers
    db.add_mcp_server("test-server", command="npx", args=["-y", "test"])
    servers = db.list_mcp_servers()
    assert len(servers) >= 1

    # Test tool log
    db.log_tool_call("read_file", '{"path": "/tmp"}', "file contents", 150)
    calls = db.get_recent_tool_calls()
    assert len(calls) == 1

    db.close()
    print("[OK] All BotDatabase tests passed!")
