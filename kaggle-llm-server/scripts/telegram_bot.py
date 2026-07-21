#!/usr/bin/env python3
"""
scripts/telegram_bot.py

Lightweight Telegram Bot client for KeglaAI.
Delegates all agent loop execution and tool calls to the API Gateway on port 8080.
Exposes a simple chat interface, handles message streaming, and uploads output files.
"""

import asyncio
import base64
import io
import json
import logging
import os
import re
import sys
import time
import httpx
from pathlib import Path
from typing import List

from telegram import Update, BotCommand
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Ensure project modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bot_db import BotDatabase

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("telegram_bot")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
GATEWAY_URL = "http://127.0.0.1:8080/v1"
MAX_HISTORY_MESSAGES = 40

db = BotDatabase()

async def upload_generated_files(update: Update, text: str):
    """
    Scans the response text for paths like 'output/file.png' or '/v1/outputs/file.mp4'
    and uploads them directly to the Telegram chat.
    """
    # Regex to find output filenames
    matches = re.findall(r"(?:output/|/v1/outputs/)([a-zA-Z0-9_\-\.\s]+)", text)
    if not matches:
        return

    # Remove duplicates
    unique_files = list(set(matches))
    for filename in unique_files:
        filename = filename.strip()
        filepath = Path("./output") / filename
        if not filepath.exists():
            continue
            
        log.info(f"Detected output file in text: {filepath}. Preparing upload...")
        try:
            ext = filepath.suffix.lower()
            if ext == ".mp4":
                await update.message.reply_video(
                    video=open(filepath, "rb"),
                    caption=f"🎥 Сгенерированное видео: {filename}"
                )
            elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
                await update.message.reply_photo(
                    photo=open(filepath, "rb"),
                    caption=f"🎨 Сгенерированное изображение: {filename}"
                )
            else:
                await update.message.reply_document(
                    document=open(filepath, "rb"),
                    caption=f"📄 Файл: {filename}"
                )
        except Exception as e:
            log.error(f"Failed to upload file {filename}: {e}")
            await update.message.reply_text(f"[Ошибка отправки файла {filename}: {e}]")

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not db.is_user_allowed(user.id):
        await update.message.reply_text("Access denied.")
        return
        
    chat_id = update.effective_chat.id
    current_model = db.get_setting(f"model_{chat_id}", "gemma-4-12b")
    
    await update.message.reply_text(
        f"Привет, {user.first_name}!\n\n"
        f"Я умный AI-ассистент KeglaAI. Я запущен в параллельном режиме на двух GPU!\n\n"
        f"Текущая модель для этого чата: *{current_model.upper()}*\n\n"
        "Доступные команды:\n"
        "/model 12b — переключить на Gemma-4-12B (GPU 0)\n"
        "/model e2b — переключить на Gemma-4-E2B (GPU 1)\n"
        "/clear — очистить историю диалога\n"
        "/status — проверить состояние бэкенда\n"
        "/help — показать эту справку",
        parse_mode="Markdown"
    )

async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not db.is_user_allowed(update.effective_user.id):
        return
    if not context.args:
        await update.message.reply_text("Использование: /model <12b|e2b>")
        return
        
    val = context.args[0].lower()
    chat_id = update.effective_chat.id
    
    if val == "12b":
        db.set_setting(f"model_{chat_id}", "gemma-4-12b")
        await update.message.reply_text("✅ Модель переключена на *Gemma-4-12B* (GPU 0)", parse_mode="Markdown")
    elif val == "e2b":
        db.set_setting(f"model_{chat_id}", "gemma-4-e2b")
        await update.message.reply_text("✅ Модель переключена на *Gemma-4-E2B* (GPU 1)", parse_mode="Markdown")
    else:
        await update.message.reply_text("Неизвестная модель. Выберите '12b' или 'e2b'.")

async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not db.is_user_allowed(update.effective_user.id):
        return
    db.clear_history(update.effective_chat.id)
    await update.message.reply_text("История переписки стёрта.")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not db.is_user_allowed(update.effective_user.id):
        return
    chat_id = update.effective_chat.id
    current_model = db.get_setting(f"model_{chat_id}", "gemma-4-12b")
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://127.0.0.1:8080/v1/models")
            models_data = resp.json()
            models_list = [m["id"] for m in models_data.get("data", [])]
            
        lines = [
            "📊 *Состояние системы:*",
            f"  • Активная модель чата: `{current_model.upper()}`",
            f"  • Доступно моделей на шлюзе: `{', '.join(models_list)}`",
            "  • Статус шлюза: `🟢 Работает`"
        ]
    except Exception as e:
        lines = [
            "📊 *Состояние системы:*",
            f"  • Активная модель чата: `{current_model.upper()}`",
            f"  • Статус шлюза: `🔴 Ошибка подключения к API шлюзу ({e})`"
        ]
        
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not db.is_user_allowed(update.effective_user.id):
        await update.message.reply_text("Access denied.")
        return

    user_text = update.message.text
    if not user_text:
        return

    chat_id = update.effective_chat.id
    await update.effective_chat.send_action("typing")

    # Get active model
    model = db.get_setting(f"model_{chat_id}", "gemma-4-12b")
    
    # Store user message
    db.add_message(chat_id, "user", user_text)
    
    # Retrieve history
    history = db.get_history(chat_id, limit=MAX_HISTORY_MESSAGES)
    
    # Check if system prompt is present
    messages = []
    if not history or history[0]["role"] != "system":
        sys_prompt = "Ты — продвинутый ИИ-ассистент. Ты можешь запускать bash-команды, писать и изменять файлы проекта, а также администрировать этот сервер. Используй инструменты автономно."
        messages.append({"role": "system", "content": sys_prompt})
        
    for h in history:
        # Ignore tool messages in base history if we restart loop cleanly
        if h["role"] in ["user", "assistant", "system"]:
            messages.append({"role": h["role"], "content": h["content"]})

    # Try using the new Telegram Bot API sendMessageDraft method (added March 2026)
    # If not supported, we fall back to standard message editing
    draft_id = int(time.time() * 1000) % 2147483647
    use_drafts = True
    status_msg = None
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessageDraft",
                json={
                    "chat_id": chat_id,
                    "draft_id": draft_id,
                    "text": "Thinking..."
                },
                timeout=5.0
            )
            if resp.status_code != 200:
                use_drafts = False
    except Exception:
        use_drafts = False

    if not use_drafts:
        # Fallback to creating a status message
        status_msg = await update.message.reply_text("Thinking...")
    
    accumulated_text = ""
    last_update_time = time.time()
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient(timeout=300.0, headers=headers) as client:
            async with client.stream(
                "POST", 
                f"{GATEWAY_URL}/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True
                }
            ) as response:
                if response.status_code != 200:
                    err_body = await response.aread()
                    err_msg_text = f"Ошибка API шлюза: {err_body.decode('utf-8')[:200]}"
                    if use_drafts:
                        await update.message.reply_text(err_msg_text)
                    else:
                        await status_msg.edit_text(err_msg_text)
                    return
                    
                last_sent_text = ""
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            continue
                        try:
                            parsed = json.loads(data_str)
                            delta = parsed["choices"][0]["delta"]
                            if "content" in delta and delta["content"]:
                                accumulated_text += delta["content"]
                                
                                if use_drafts:
                                    # Draft updates have no rate limit in Telegram (updates every 100ms for smoothness)
                                    if time.time() - last_update_time > 0.1:
                                        try:
                                            async with httpx.AsyncClient() as client:
                                                await client.post(
                                                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessageDraft",
                                                    json={
                                                        "chat_id": chat_id,
                                                        "draft_id": draft_id,
                                                        "text": accumulated_text[:4000]
                                                    },
                                                    timeout=5.0
                                                )
                                            last_update_time = time.time()
                                        except Exception:
                                            pass
                                else:
                                    # Fallback: Throttle Telegram message updates to avoid rate limits
                                    if time.time() - last_update_time > 1.5:
                                        preview = accumulated_text[-4000:] if len(accumulated_text) > 4000 else accumulated_text
                                        if preview != last_sent_text:
                                            try:
                                                await status_msg.edit_text(preview)
                                                last_sent_text = preview
                                            except Exception as telegram_err:
                                                if "Message is not modified" not in str(telegram_err):
                                                    raise telegram_err
                                            last_update_time = time.time()
                        except Exception:
                            pass

        # Final edit with complete response
        final_text = accumulated_text if accumulated_text else "(пустой ответ)"
        if use_drafts:
            # 1. Update final draft
            try:
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessageDraft",
                        json={
                            "chat_id": chat_id,
                            "draft_id": draft_id,
                            "text": final_text[:4000]
                        },
                        timeout=5.0
                    )
            except Exception:
                pass
            
            # 2. Publish final message (clears the draft automatically in Telegram UI)
            if len(final_text) > 4000:
                chunks = [final_text[i:i+4000] for i in range(0, len(final_text), 4000)]
                for chunk in chunks:
                    await update.message.reply_text(chunk)
            else:
                await update.message.reply_text(final_text)
        else:
            # Fallback final message edit
            if len(final_text) > 4000:
                chunks = [final_text[i:i+4000] for i in range(0, len(final_text), 4000)]
                if chunks[0] != last_sent_text:
                    try:
                        await status_msg.edit_text(chunks[0])
                    except Exception as telegram_err:
                        if "Message is not modified" not in str(telegram_err):
                            raise telegram_err
                for chunk in chunks[1:]:
                    await update.message.reply_text(chunk)
            else:
                if final_text != last_sent_text:
                    try:
                        await status_msg.edit_text(final_text)
                    except Exception as telegram_err:
                        if "Message is not modified" not in str(telegram_err):
                            raise telegram_err
            
        # Store assistant response in DB
        db.add_message(chat_id, "assistant", final_text)
        
        # Upload any files mentioned/created in response
        await upload_generated_files(update, final_text)
        
    except Exception as e:
        log.error(f"Error in handle_message: {e}", exc_info=True)
        err_msg_text = f"Произошла ошибка: {str(e)[:200]}"
        if use_drafts:
            await update.message.reply_text(err_msg_text)
        else:
            await status_msg.edit_text(err_msg_text)

async def run_bot():
    token = TELEGRAM_BOT_TOKEN
    if not token:
        log.error("TELEGRAM_BOT_TOKEN environment variable not set.")
        return

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("status", cmd_status))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    try:
        await app.bot.set_my_commands([
            BotCommand("start", "Показать приветствие"),
            BotCommand("model", "Выбрать модель (12b или e2b)"),
            BotCommand("clear", "Очистить историю диалога"),
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
            log.info("Shutting down bot...")
            try:
                await app.updater.stop()
            except Exception:
                pass
            try:
                await app.stop()
            except Exception:
                pass
            db.close()

if __name__ == "__main__":
    asyncio.run(run_bot())
