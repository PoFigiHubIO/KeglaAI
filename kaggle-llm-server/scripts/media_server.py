#!/usr/bin/env python3
"""
scripts/media_server.py

SSE MCP-совместимый медиа-сервер для GPU 1 (порт 8081).
Предоставляет инструменты generate_image и generate_video через
Server-Sent Events (SSE) протокол MCP для интеграции с:
  - Встроенным Web UI llama.cpp (панель MCP Servers)
  - Telegram Bot Agent Loop

Запуск:
    CUDA_VISIBLE_DEVICES=1 python scripts/media_server.py
    # или через start.py с CONFIG_FILE=config_gpu1.yaml

Архитектура памяти:
    На одной Tesla T4 (16 GB VRAM) одновременно помещается только ОДНА
    модель генерации. При переключении между image и video пайплайнами
    сервер полностью выгружает текущую модель из VRAM перед загрузкой
    следующей (swap-режим).
"""

import asyncio
import base64
import gc
import io
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# PyTorch memory fragmentation mitigation (must be set BEFORE import torch)
# ---------------------------------------------------------------------------
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "garbage_collection_threshold:0.6,max_split_size_mb:128",
)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("media_server")

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------
HOST = os.environ.get("MEDIA_HOST", "0.0.0.0")
PORT = int(os.environ.get("MEDIA_PORT", "8081"))
OUTPUT_DIR = Path(os.environ.get("MEDIA_OUTPUT_DIR", "./output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# FLUX.1 Dev model configuration
FLUX_MODEL_ID = os.environ.get(
    "FLUX_MODEL_ID", "black-forest-labs/FLUX.1-dev"
)
FLUX_LORA_ID = os.environ.get(
    "FLUX_LORA_ID", "aiMaDi/aidmaNSFWunlock"
)
FLUX_LORA_WEIGHT_NAME = os.environ.get(
    "FLUX_LORA_WEIGHT_NAME", "aidmaNSFWunlock_flux_lora.safetensors"
)
FLUX_LORA_SCALE = float(os.environ.get("FLUX_LORA_SCALE", "0.8"))

# Defaults for image generation
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 20
DEFAULT_GUIDANCE = 3.5


# ---------------------------------------------------------------------------
# VRAM Manager — ensures only one heavy model is loaded at a time
# ---------------------------------------------------------------------------
class ActiveModel(str, Enum):
    NONE = "none"
    IMAGE = "image"       # FLUX.1 Dev pipeline
    VIDEO = "video"       # Wan 2.2 Remix pipeline


class VRAMManager:
    """
    Manages GPU memory by ensuring mutual exclusion between the image
    and video generation pipelines. Only one pipeline can be resident
    in VRAM at any given time on a single T4 (16 GB).
    """

    def __init__(self):
        self.active: ActiveModel = ActiveModel.NONE
        self.image_pipe = None
        self.video_pipe = None
        self._lock = asyncio.Lock()

    async def get_image_pipe(self):
        """Return the image pipeline, loading it if necessary."""
        async with self._lock:
            if self.active != ActiveModel.IMAGE:
                await self._unload_current()
                self.image_pipe = await self._load_image_pipeline()
                self.active = ActiveModel.IMAGE
            return self.image_pipe

    async def get_video_pipe(self):
        """Return the video pipeline, loading it if necessary."""
        async with self._lock:
            if self.active != ActiveModel.VIDEO:
                await self._unload_current()
                self.video_pipe = await self._load_video_pipeline()
                self.active = ActiveModel.VIDEO
            return self.video_pipe

    async def _unload_current(self):
        """Completely free VRAM occupied by the current pipeline."""
        if self.active == ActiveModel.IMAGE and self.image_pipe is not None:
            log.info("Выгрузка image pipeline из VRAM...")
            del self.image_pipe
            self.image_pipe = None
        elif self.active == ActiveModel.VIDEO and self.video_pipe is not None:
            log.info("Выгрузка video pipeline из VRAM...")
            del self.video_pipe
            self.video_pipe = None

        self.active = ActiveModel.NONE
        self._force_gc()

    @staticmethod
    def _force_gc():
        """Aggressive garbage collection + CUDA cache flush."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                log.info(
                    f"VRAM после очистки: "
                    f"allocated={allocated:.2f} GB, reserved={reserved:.2f} GB"
                )
        except ImportError:
            pass

    async def _load_image_pipeline(self):
        """
        Load the FLUX.1 Dev image generation pipeline with NF4
        quantization (fits in ~8 GB VRAM on T4) and inject the
        aidmaNSFWunlock LoRA for uncensored generation.

        Memory layout on T4 16 GB:
          - Transformer (NF4): ~6 GB
          - VAE (FP16):        ~0.2 GB
          - LoRA weights:      ~0.1 GB
          - KV / workspace:    ~2 GB
          Total:               ~8.3 GB  (leaves ~7.7 GB headroom)
        """
        log.info(f"Загрузка image pipeline: {FLUX_MODEL_ID}")
        log.info(f"  Квантование: NF4 (bitsandbytes)")
        log.info(f"  LoRA: {FLUX_LORA_ID} (scale={FLUX_LORA_SCALE})")

        import torch
        from diffusers import FluxPipeline
        from transformers import BitsAndBytesConfig

        # --- NF4 quantization config for the transformer ---
        # This reduces the 12B-param FLUX transformer from ~24 GB (FP16)
        # down to ~6 GB, making it fit on a single T4.
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        def _load_sync():
            pipe = FluxPipeline.from_pretrained(
                FLUX_MODEL_ID,
                transformer_kwargs={"quantization_config": nf4_config},
                torch_dtype=torch.float16,
            )

            # CPU offload: moves each component to GPU only when needed,
            # then back to CPU. This dramatically reduces peak VRAM usage
            # because the text encoders (CLIP + T5-XXL) are not held in
            # VRAM simultaneously with the transformer.
            pipe.enable_model_cpu_offload()

            # VAE memory optimizations
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()

            # --- Load NSFW-unlock LoRA ---
            try:
                pipe.load_lora_weights(
                    FLUX_LORA_ID,
                    weight_name=FLUX_LORA_WEIGHT_NAME,
                )
                pipe.fuse_lora(lora_scale=FLUX_LORA_SCALE)
                log.info(f"  LoRA '{FLUX_LORA_ID}' загружена и слита (fused)")
            except Exception as e:
                log.warning(
                    f"  Не удалось загрузить LoRA '{FLUX_LORA_ID}': {e}. "
                    f"Генерация будет работать без неё."
                )

            return pipe

        # Run the blocking model load in a thread to keep the event loop free
        loop = asyncio.get_event_loop()
        pipe = await loop.run_in_executor(None, _load_sync)

        self._force_gc()
        log.info("Image pipeline загружен и готов к генерации.")
        return pipe

    async def _load_video_pipeline(self):
        """
        Load the Wan 2.2 Remix video generation pipeline.

        TODO (Stage 2, subtask 3): Replace this stub with actual
        Wan 2.2 Remix loading with FP8/GGUF quantization and
        VAE tiling/slicing configuration.
        """
        log.info("Загрузка video pipeline (Wan 2.2 Remix) — STUB...")
        # Placeholder — will be replaced with real Diffusers code
        return {"name": "wan22-remix-stub", "status": "loaded"}

    def status(self) -> dict:
        """Return current VRAM status for health checks."""
        info = {
            "active_model": self.active.value,
            "image_loaded": self.image_pipe is not None,
            "video_loaded": self.video_pipe is not None,
        }
        try:
            import torch
            if torch.cuda.is_available():
                info["vram_allocated_gb"] = round(
                    torch.cuda.memory_allocated() / 1024**3, 2
                )
                info["vram_reserved_gb"] = round(
                    torch.cuda.memory_reserved() / 1024**3, 2
                )
                info["vram_total_gb"] = round(
                    torch.cuda.get_device_properties(0).total_mem / 1024**3, 2
                )
                info["gpu_name"] = torch.cuda.get_device_name(0)
        except (ImportError, RuntimeError):
            info["vram_allocated_gb"] = None
        return info


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
vram = VRAMManager()


# ---------------------------------------------------------------------------
# FastAPI lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(f"Media Server запускается на {HOST}:{PORT}")
    log.info(f"Выходная папка для файлов: {OUTPUT_DIR.resolve()}")
    log.info(f"PYTORCH_CUDA_ALLOC_CONF = {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")

    # Pre-check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
            log.info(f"GPU обнаружен: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            log.warning("CUDA недоступен — сервер запустится в CPU-режиме (медленно)")
    except ImportError:
        log.warning("PyTorch не установлен — генерация недоступна")

    yield  # App is running

    # Shutdown: release all VRAM
    log.info("Завершение: выгрузка всех моделей...")
    await vram._unload_current()
    log.info("Media Server остановлен.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Kaggle Media Generation Server",
    description=(
        "SSE MCP-совместимый сервер генерации изображений и видео. "
        "Предоставляет инструменты generate_image и generate_video "
        "через протокол Model Context Protocol (SSE transport)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health & Status endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Эндпоинт проверки здоровья для Ngrok / Cloudflare healthcheck."""
    return {"status": "ok", "service": "media-server", "port": PORT}


@app.get("/status")
async def status():
    """Детальная информация о состоянии VRAM и загруженных моделях."""
    return vram.status()


# ---------------------------------------------------------------------------
# Direct REST API endpoints (for testing without MCP)
# ---------------------------------------------------------------------------
@app.post("/api/generate_image")
async def api_generate_image(request: Request):
    """
    REST-эндпоинт для генерации изображений через FLUX.1 Dev.
    Принимает JSON:
        {
            "prompt": "описание изображения",
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "guidance_scale": 3.5,
            "seed": -1
        }
    Возвращает JSON с base64-encoded PNG и метаданными.
    """
    body = await request.json()
    prompt = body.get("prompt", "")
    width = body.get("width", DEFAULT_WIDTH)
    height = body.get("height", DEFAULT_HEIGHT)
    steps = body.get("steps", DEFAULT_STEPS)
    guidance = body.get("guidance_scale", DEFAULT_GUIDANCE)
    seed = body.get("seed", -1)

    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    # Clamp dimensions to multiples of 8 (required by VAE)
    width = max(256, min(2048, (width // 8) * 8))
    height = max(256, min(2048, (height // 8) * 8))

    log.info(
        f"generate_image: prompt='{prompt[:80]}...', "
        f"{width}x{height}, steps={steps}, guidance={guidance}"
    )

    pipe = await vram.get_image_pipe()
    image_id = str(uuid.uuid4())[:8]

    try:
        import torch

        # Reproducible seed
        generator = None
        if seed >= 0:
            generator = torch.Generator(device="cpu").manual_seed(seed)
        else:
            seed = torch.randint(0, 2**32, (1,)).item()
            generator = torch.Generator(device="cpu").manual_seed(seed)

        # Run inference in a thread to avoid blocking the event loop
        def _generate():
            result = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )
            return result.images[0]

        loop = asyncio.get_event_loop()
        image = await loop.run_in_executor(None, _generate)

        # Save to disk
        filename = f"{image_id}.png"
        filepath = OUTPUT_DIR / filename
        image.save(filepath, format="PNG")
        log.info(f"generate_image: сохранено в {filepath}")

        # Encode to base64 for API response
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")

        result = {
            "id": image_id,
            "status": "success",
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance,
            "seed": seed,
            "image_base64": b64_data,
            "file": str(filepath),
        }
        log.info(f"generate_image: завершено, id={image_id}, seed={seed}")
        return JSONResponse(result)

    except Exception as e:
        log.error(f"generate_image: ошибка генерации: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Image generation failed: {str(e)}",
        )


@app.post("/api/generate_video")
async def api_generate_video(request: Request):
    """
    REST-эндпоинт для генерации видео.
    Принимает JSON: {"prompt": "...", "image_url": "...", "seconds": 3}

    TODO (Stage 2, subtask 3): Wire up real Wan 2.2 pipeline here.
    """
    body = await request.json()
    prompt = body.get("prompt", "")
    image_url = body.get("image_url", "")
    seconds = body.get("seconds", 3)

    if not prompt and not image_url:
        raise HTTPException(
            status_code=400, detail="prompt or image_url is required"
        )

    log.info(f"generate_video: prompt='{prompt[:80]}', seconds={seconds}")

    pipe = await vram.get_video_pipe()

    # --- STUB: Return a placeholder response ---
    video_id = str(uuid.uuid4())[:8]
    result = {
        "id": video_id,
        "status": "stub",
        "message": f"Video pipeline '{pipe['name']}' received prompt. "
                   f"Real generation will be implemented in the next step.",
        "prompt": prompt,
        "seconds": seconds,
    }

    log.info(f"generate_video: завершено (stub), id={video_id}")
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# SSE MCP Protocol endpoints — STUBS
# ---------------------------------------------------------------------------
# These will be implemented in Stage 2, subtask 5.
# The MCP protocol requires:
#   GET  /sse       — SSE stream for server-initiated messages
#   POST /messages  — JSON-RPC endpoint for client requests

@app.get("/sse")
async def sse_endpoint():
    """
    SSE MCP endpoint stub.

    TODO (Stage 2, subtask 5): Implement full MCP SSE transport using
    the official `mcp` Python SDK. This endpoint will:
    1. Open an SSE stream
    2. Send an 'endpoint' event with the /messages URL
    3. Stream tool results back to the MCP client
    """
    async def event_stream():
        # Send initial endpoint event (MCP protocol requirement)
        messages_url = f"http://127.0.0.1:{PORT}/messages"
        yield f"event: endpoint\ndata: {messages_url}\n\n"

        # Keep the connection alive with periodic heartbeats
        while True:
            yield f"event: heartbeat\ndata: {json.dumps({'time': time.time()})}\n\n"
            await asyncio.sleep(15)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/messages")
async def mcp_messages(request: Request):
    """
    MCP JSON-RPC message handler stub.

    TODO (Stage 2, subtask 5): Implement full JSON-RPC 2.0 handling for:
    - initialize
    - tools/list  (return generate_image and generate_video schemas)
    - tools/call  (dispatch to the appropriate pipeline)
    """
    body = await request.json()
    method = body.get("method", "")
    req_id = body.get("id", None)

    log.info(f"MCP message: method={method}, id={req_id}")

    # Minimal stub responses for MCP protocol handshake
    if method == "initialize":
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "kaggle-media-server",
                    "version": "0.1.0",
                },
            },
        })

    elif method == "tools/list":
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "generate_image",
                        "description": (
                            "Генерирует изображение по текстовому описанию. "
                            "Возвращает base64-encoded PNG."
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "Описание изображения для генерации",
                                },
                                "width": {
                                    "type": "integer",
                                    "description": "Ширина в пикселях (по умолчанию 1024)",
                                    "default": 1024,
                                },
                                "height": {
                                    "type": "integer",
                                    "description": "Высота в пикселях (по умолчанию 1024)",
                                    "default": 1024,
                                },
                            },
                            "required": ["prompt"],
                        },
                    },
                    {
                        "name": "generate_video",
                        "description": (
                            "Генерирует короткое видео по текстовому описанию "
                            "или на основе входного изображения (Image-to-Video)."
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "Описание видео для генерации",
                                },
                                "image_base64": {
                                    "type": "string",
                                    "description": "Base64-encoded исходное изображение (для I2V)",
                                },
                                "seconds": {
                                    "type": "integer",
                                    "description": "Длительность видео в секундах (по умолчанию 3)",
                                    "default": 3,
                                },
                            },
                            "required": ["prompt"],
                        },
                    },
                ]
            },
        })

    elif method == "tools/call":
        tool_name = body.get("params", {}).get("name", "")
        tool_args = body.get("params", {}).get("arguments", {})
        log.info(f"MCP tools/call: {tool_name}({json.dumps(tool_args, ensure_ascii=False)[:100]})")

        # Dispatch to the appropriate handler
        if tool_name == "generate_image":
            result = await api_generate_image(
                type("FakeRequest", (), {"json": lambda self: tool_args})()
            )
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result.body.decode()
                                               if hasattr(result, 'body') else str(result)),
                        }
                    ]
                },
            })

        elif tool_name == "generate_video":
            result = await api_generate_video(
                type("FakeRequest", (), {"json": lambda self: tool_args})()
            )
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result.body.decode()
                                               if hasattr(result, 'body') else str(result)),
                        }
                    ]
                },
            })

        else:
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown tool: {tool_name}",
                },
            })

    # Unknown method
    return JSONResponse({
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {
            "code": -32601,
            "message": f"Method not found: {method}",
        },
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Force GPU 1 if not already set
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        log.info("CUDA_VISIBLE_DEVICES установлен в '1' (GPU 1)")

    uvicorn.run(
        "media_server:app",
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=True,
    )
