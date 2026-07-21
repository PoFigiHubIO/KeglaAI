#!/usr/bin/env python3
"""
scripts/telegram_bot.py

Lightweight Telegram Bot client for KeglaAI.
Delegates all agent loop execution and tool calls to the API Gateway on port 8080.
Exposes a simple chat interface, handles message streaming, and uploads output files.
"""

import asyncio
import base64
import html
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

from telegram import Update, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

def format_telegram_html(text: str, reasoning: str, show_thinking: bool) -> str:
    parts = []
    if show_thinking and reasoning:
        # Escape HTML in reasoning content
        escaped_reasoning = html.escape(reasoning.strip())
        if escaped_reasoning:
            # Wrap in collapsible blockquote
            parts.append(f"<blockquote expandable>{escaped_reasoning}</blockquote>")
            
    # Escape HTML in main text
    escaped_text = html.escape(text)
    
    # 1. Convert code blocks: ```lang ... ```
    escaped_text = re.sub(
        r'```(\w*)\n(.*?)```',
        lambda m: f'<pre><code class="language-{m.group(1)}">{m.group(2)}</code></pre>',
        escaped_text,
        flags=re.DOTALL
    )
    
    # 2. Convert inline code: `code`
    escaped_text = re.sub(
        r'`([^`\n]+)`',
        r'<code>\1</code>',
        escaped_text
    )
    
    # 3. Convert bold: **text**
    escaped_text = re.sub(
        r'\*\*([^*]+)\*\*',
        r'<b>\1</b>',
        escaped_text
    )
    
    # 4. Convert italics: *text* (avoiding bold fragments)
    escaped_text = re.sub(
        r'\*([^*]+)\*',
        r'<i>\1</i>',
        escaped_text
    )
    
    # 5. Convert links: [text](url)
    escaped_text = re.sub(
        r'\[([^\]]+)\]\((https?://[^\)]+)\)',
        r'<a href="\2">\1</a>',
        escaped_text
    )

    # 6. Convert blockquotes: > text
    escaped_text = re.sub(
        r'(?:^|\n)&gt;\s*([^\n]+)',
        r'\n<blockquote>\1</blockquote>',
        escaped_text
    )
    
    parts.append(escaped_text)
    return "\n\n".join(parts)

def convert_ogg_to_wav(ogg_path: str, wav_path: str) -> bool:
    try:
        import subprocess
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", ogg_path, "-ac", "1", "-ar", "16000", wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return result.returncode == 0
    except Exception as e:
        log.error(f"Error converting OGG to WAV: {e}")
        return False

def transcribe_audio_file(wav_path: str) -> str:
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = r.record(source)
        return r.recognize_google(audio, language="ru-RU")
    except Exception as e:
        log.error(f"Error transcribing audio: {e}")
        return f"[Ошибка распознавания речи: {e}]"

async def cmd_mcp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not db.is_user_allowed(update.effective_user.id):
        return
    await send_mcp_dashboard(update.message.reply_text)

async def send_mcp_dashboard(reply_func, message_to_edit=None):
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{GATEWAY_URL}/mcp/status")
            if resp.status_code != 200:
                await reply_func("Не удалось загрузить статус MCP-серверов со шлюза.")
                return
            servers = resp.json()
    except Exception as e:
        await reply_func(f"Ошибка подключения к шлюзу: {e}")
        return

    if not servers:
        await reply_func("Нет настроенных MCP-серверов в системе.")
        return

    text = "🖥️ *Панель управления MCP-серверами:*\n\n"
    keyboard = []
    
    for s in servers:
        name = s["name"]
        running = s["running"]
        enabled = s["enabled"]
        
        status_emoji = "🟢" if (running and enabled) else "🔴"
        status_text = "Активен" if (running and enabled) else "Выключен"
        
        text += f"{status_emoji} *{name}* — {status_text}\n"
        if s["description"]:
            text += f"   _Описание:_ {s['description']}\n"
        if s["tools"]:
            text += f"   _Инструменты:_ {', '.join(s['tools'])}\n"
        text += "\n"
        
        btn_action = "Выключить" if enabled else "Включить"
        keyboard.append([InlineKeyboardButton(f"{btn_action} {name}", callback_data=f"mcp_toggle:{name}")])
        
    keyboard.append([InlineKeyboardButton("🔄 Обновить список", callback_data="mcp_refresh")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if message_to_edit:
        await message_to_edit.edit_text(text, reply_markup=reply_markup, parse_mode="Markdown")
    else:
        await reply_func(text, reply_markup=reply_markup, parse_mode="Markdown")

async def handle_mcp_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if not db.is_user_allowed(update.effective_user.id):
        return
        
    data = query.data
    if data == "mcp_refresh":
        await send_mcp_dashboard(query.edit_message_text, query.message)
        return
        
    if data.startswith("mcp_toggle:"):
        server_name = data.split(":", 1)[1]
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{GATEWAY_URL}/mcp/toggle",
                    json={"name": server_name}
                )
                if resp.status_code == 200:
                    res_body = resp.json()
                    alert_text = res_body.get("message", f"Переключен статус {server_name}")
                else:
                    alert_text = f"Ошибка шлюза: {resp.text}"
        except Exception as e:
            alert_text = f"Ошибка сети: {e}"
            
        await query.answer(text=alert_text, show_alert=True)
        await send_mcp_dashboard(query.edit_message_text, query.message)


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
        "/show_thinking — включить/выключить отображение хода рассуждений (Thinking)\n"
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

async def cmd_show_thinking(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not db.is_user_allowed(update.effective_user.id):
        return
    chat_id = update.effective_chat.id
    current = db.get_setting(f"show_thinking_{chat_id}", "true")
    new_value = "false" if current == "true" else "true"
    db.set_setting(f"show_thinking_{chat_id}", new_value)
    
    status = "включено" if new_value == "true" else "выключено"
    await update.message.reply_text(
        f"🧠 Отображение хода рассуждений модели (Thinking) теперь *{status}* для этого чата.",
        parse_mode="Markdown"
    )

async def run_chat_stream(update: Update, chat_id: int, model: str, messages: List[dict], show_thinking: bool):
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
    accumulated_reasoning = ""
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
                            
                            has_new_content = False
                            if "reasoning_content" in delta and delta["reasoning_content"]:
                                accumulated_reasoning += delta["reasoning_content"]
                                has_new_content = True
                            if "content" in delta and delta["content"]:
                                accumulated_text += delta["content"]
                                has_new_content = True
                                
                            if has_new_content:
                                preview = format_telegram_html(accumulated_text, accumulated_reasoning, show_thinking)
                                
                                if use_drafts:
                                    # Draft updates have no rate limit in Telegram (updates every 150ms for smoothness)
                                    if time.time() - last_update_time > 0.15:
                                        try:
                                            async with httpx.AsyncClient() as client:
                                                await client.post(
                                                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessageDraft",
                                                    json={
                                                        "chat_id": chat_id,
                                                        "draft_id": draft_id,
                                                        "text": preview[:4000],
                                                        "parse_mode": "HTML"
                                                    },
                                                    timeout=5.0
                                                )
                                            last_update_time = time.time()
                                        except Exception:
                                            pass
                                else:
                                    # Fallback: Throttle Telegram message updates to avoid rate limits
                                    if time.time() - last_update_time > 1.5:
                                        if preview != last_sent_text:
                                            try:
                                                await status_msg.edit_text(preview, parse_mode="HTML")
                                                last_sent_text = preview
                                            except Exception as telegram_err:
                                                if "Message is not modified" not in str(telegram_err):
                                                    raise telegram_err
                                            last_update_time = time.time()
                        except Exception:
                            pass

        # Final edit with complete response
        final_text = accumulated_text if accumulated_text else "(пустой ответ)"
        final_html = format_telegram_html(final_text, accumulated_reasoning, show_thinking)
        
        if use_drafts:
            # 1. Update final draft
            try:
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessageDraft",
                        json={
                            "chat_id": chat_id,
                            "draft_id": draft_id,
                            "text": final_html[:4000],
                            "parse_mode": "HTML"
                        },
                        timeout=5.0
                    )
            except Exception:
                pass
            
            # 2. Publish final message (clears the draft automatically in Telegram UI)
            if len(final_html) > 4000:
                chunks = [final_html[i:i+4000] for i in range(0, len(final_html), 4000)]
                for chunk in chunks:
                    await update.message.reply_text(chunk, parse_mode="HTML")
            else:
                await update.message.reply_text(final_html, parse_mode="HTML")
        else:
            # Fallback final message edit
            if len(final_html) > 4000:
                chunks = [final_html[i:i+4000] for i in range(0, len(final_html), 4000)]
                if chunks[0] != last_sent_text:
                    try:
                        await status_msg.edit_text(chunks[0], parse_mode="HTML")
                    except Exception as telegram_err:
                        if "Message is not modified" not in str(telegram_err):
                            raise telegram_err
                for chunk in chunks[1:]:
                    await update.message.reply_text(chunk, parse_mode="HTML")
            else:
                if final_html != last_sent_text:
                    try:
                        await status_msg.edit_text(final_html, parse_mode="HTML")
                    except Exception as telegram_err:
                        if "Message is not modified" not in str(telegram_err):
                            raise telegram_err
            
        # Store assistant response in DB
        db.add_message(chat_id, "assistant", final_text)
        
        # Upload any files mentioned/created in response
        await upload_generated_files(update, final_text)
        
    except Exception as e:
        log.error(f"Error in run_chat_stream: {e}", exc_info=True)
        err_msg_text = f"Произошла ошибка: {str(e)[:200]}"
        if use_drafts:
            await update.message.reply_text(err_msg_text)
        else:
            await status_msg.edit_text(err_msg_text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not db.is_user_allowed(update.effective_user.id):
        await update.message.reply_text("Access denied.")
        return

    user_text = update.message.text
    if not user_text:
        return

    chat_id = update.effective_chat.id
    await update.effective_chat.send_action("typing")

    # Get active model and settings
    model = db.get_setting(f"model_{chat_id}", "gemma-4-12b")
    show_thinking_str = db.get_setting(f"show_thinking_{chat_id}", "true")
    show_thinking = (show_thinking_str == "true")
    
    # Store user message
    db.add_message(chat_id, "user", user_text)
    
    # Retrieve history
    history = db.get_history(chat_id, limit=MAX_HISTORY_MESSAGES)
    
    # Prepend specialized Telegram formatting system instructions
    messages = []
    sys_prompt = (
        "Ты — продвинутый ИИ-ассистент KeglaAI. Ты можешь запускать bash-команды, писать и изменять файлы проекта, "
        "а также выполнять администрирование этого сервера. Используй инструменты автономно.\n\n"
        "ВАЖНОЕ ТРЕБОВАНИЕ К ОФОРМЛЕНИЮ:\n"
        "Поскольку ты общаешься через Telegram, форматируй свои ответы красиво и читаемо с помощью Markdown:\n"
        "- Используй жирный шрифт (**текст**) для заголовков разделов, важных терминов и ключевых мыслей.\n"
        "- Оформляй код в блоки кода с указанием языка (например, ```python ... ```).\n"
        "- Размечай списки, важные перечисления и таблицы.\n"
        "- Для обычных цитат используй блок цитирования (> текст).\n"
        "- Пиши структурированно, разбивай текст на небольшие логичные абзацы."
    )
    messages.append({"role": "system", "content": sys_prompt})
    
    for h in history:
        if h["role"] in ["user", "assistant"]:
            messages.append({"role": h["role"], "content": h["content"]})

    await run_chat_stream(update, chat_id, model, messages, show_thinking)

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not db.is_user_allowed(update.effective_user.id):
        await update.message.reply_text("Access denied.")
        return

    chat_id = update.effective_chat.id
    await update.effective_chat.send_action("typing")

    caption = update.message.caption if update.message.caption else "Опиши это изображение"
    
    try:
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        base64_image = base64.b64encode(photo_bytes).decode("utf-8")
    except Exception as e:
        await update.message.reply_text(f"Не удалось загрузить изображение: {e}")
        return

    model = db.get_setting(f"model_{chat_id}", "gemma-4-12b")
    show_thinking_str = db.get_setting(f"show_thinking_{chat_id}", "true")
    show_thinking = (show_thinking_str == "true")
    
    db.add_message(chat_id, "user", f"[Изображение] {caption}")
    history = db.get_history(chat_id, limit=MAX_HISTORY_MESSAGES)
    
    messages = []
    sys_prompt = (
        "Ты — продвинутый ИИ-ассистент KeglaAI. Ты можешь запускать bash-команды, писать и изменять файлы проекта, "
        "а также выполнять администрирование этого сервера. Используй инструменты автономно.\n\n"
        "ВАЖНОЕ ТРЕБОВАНИЕ К ОФОРМЛЕНИЮ:\n"
        "Поскольку ты общаешься через Telegram, форматируй свои ответы красиво и читаемо с помощью Markdown:\n"
        "- Используй жирный шрифт (**текст**) для заголовков разделов, важных терминов и ключевых мыслей.\n"
        "- Оформляй код в блоки кода с указанием языка (например, ```python ... ```).\n"
        "- Размечай списки, важные перечисления и таблицы.\n"
        "- Для обычных цитат используй блок цитирования (> текст).\n"
        "- Пиши структурированно, разбивай текст на небольшие логичные абзацы."
    )
    messages.append({"role": "system", "content": sys_prompt})
    
    for h in history[:-1]:
        if h["role"] in ["user", "assistant"]:
            messages.append({"role": h["role"], "content": h["content"]})
            
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": caption},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
    })
    
    await run_chat_stream(update, chat_id, model, messages, show_thinking)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not db.is_user_allowed(update.effective_user.id):
        await update.message.reply_text("Access denied.")
        return

    chat_id = update.effective_chat.id
    await update.effective_chat.send_action("typing")

    voice = update.message.voice
    if not voice:
        return

    os.makedirs("./logs", exist_ok=True)
    ogg_path = "./logs/voice_temp.ogg"
    wav_path = "./logs/voice_temp.wav"
    
    try:
        voice_file = await voice.get_file()
        await voice_file.download_to_drive(ogg_path)
        
        if not convert_ogg_to_wav(ogg_path, wav_path):
            await update.message.reply_text("Не удалось декодировать аудио-формат голосового сообщения.")
            return
            
        transcribed_text = transcribe_audio_file(wav_path)
        
        # Cleanup
        for p in [ogg_path, wav_path]:
            if os.path.exists(p):
                os.remove(p)
                
        if not transcribed_text or transcribed_text.startswith("["):
            await update.message.reply_text(f"Голос не распознан: {transcribed_text}")
            return
            
        await update.message.reply_text(f"🎤 *Вы сказали:* {transcribed_text}", parse_mode="Markdown")
        
        model = db.get_setting(f"model_{chat_id}", "gemma-4-12b")
        show_thinking_str = db.get_setting(f"show_thinking_{chat_id}", "true")
        show_thinking = (show_thinking_str == "true")
        
        db.add_message(chat_id, "user", transcribed_text)
        history = db.get_history(chat_id, limit=MAX_HISTORY_MESSAGES)
        
        messages = []
        sys_prompt = (
            "Ты — продвинутый ИИ-ассистент KeglaAI. Ты можешь запускать bash-команды, писать и изменять файлы проекта, "
            "а также выполнять администрирование этого сервера. Используй инструменты автономно.\n\n"
            "ВАЖНОЕ ТРЕБОВАНИЕ К ОФОРМЛЕНИЮ:\n"
            "Поскольку ты общаешься через Telegram, форматируй свои ответы красиво и читаемо с помощью Markdown:\n"
            "- Используй жирный шрифт (**текст**) для заголовков разделов, важных терминов и ключевых мыслей.\n"
            "- Оформляй код в блоки кода с указанием языка (например, ```python ... ```).\n"
            "- Размечай списки, важные перечисления и таблицы.\n"
            "- Для обычных цитат используй блок цитирования (> текст).\n"
            "- Пиши структурированно, разбивай текст на небольшие логичные абзацы."
        )
        messages.append({"role": "system", "content": sys_prompt})
        
        for h in history:
            if h["role"] in ["user", "assistant"]:
                messages.append({"role": h["role"], "content": h["content"]})
                
        await run_chat_stream(update, chat_id, model, messages, show_thinking)
        
    except Exception as e:
        log.error(f"Error handling voice message: {e}", exc_info=True)
        await update.message.reply_text(f"Ошибка обработки голосового сообщения: {e}")

async def run_bot():
    token = TELEGRAM_BOT_TOKEN
    if not token:
        log.error("TELEGRAM_BOT_TOKEN environment variable not set.")
        return

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("show_thinking", cmd_show_thinking))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("mcp", cmd_mcp))

    app.add_handler(CallbackQueryHandler(handle_mcp_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    try:
        await app.bot.set_my_commands([
            BotCommand("start", "Показать приветствие"),
            BotCommand("model", "Выбрать модель (12b или e2b)"),
            BotCommand("show_thinking", "Вкл/выкл отображение рассуждений"),
            BotCommand("mcp", "Панель управления MCP-серверами"),
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
            allowed_updates=["message", "callback_query"],
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
