"""FastAPI serving layer for the fine-tuned LLM.

Endpoint: POST /generate
Features:
  - Request/Response schema validation (Pydantic)
  - Structured JSON logging (latency, token count, model version)
  - Basic rate limiting (in-memory, per-IP)
  - Safe mode toggle via request field
  - Health check at GET /health

Usage:
    uvicorn serve.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
import pathlib
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from serve.guardrails import MAX_OUTPUT_TOKENS, GuardrailEngine

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR: str = os.environ.get("MODEL_DIR", "train/outputs/final")
MANIFEST_PATH: str = os.environ.get("MANIFEST_PATH", "data/processed/manifest.json")
RATE_LIMIT_RPM: int = int(os.environ.get("RATE_LIMIT_RPM", "30"))


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Load model at startup."""
    _load_model()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="LLM Fine-tune Pipeline API",
    version="1.0.0",
    description="Serves the fine-tuned causal language model with safety guardrails.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    """Request schema for /generate endpoint."""

    prompt: str = Field(..., min_length=1, max_length=2000, description="User prompt")
    max_tokens: int = Field(
        default=128,
        ge=1,
        le=MAX_OUTPUT_TOKENS,
        description="Maximum tokens to generate",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    safe_mode: bool = Field(default=True, description="Enable safety guardrails")


class GenerateResponse(BaseModel):
    """Response schema for /generate endpoint."""

    generated_text: str
    prompt: str
    model_version: str
    dataset_hash: str
    token_count: int
    latency_ms: float
    safe_mode: bool


# ---------------------------------------------------------------------------
# State (loaded at startup)
# ---------------------------------------------------------------------------
class AppState:
    """Holds loaded model, tokenizer, and metadata."""

    model: Any = None
    tokenizer: Any = None
    pipe: Any = None
    model_version: str = "unknown"
    dataset_hash: str = "unknown"
    loaded: bool = False

    # Simple in-memory rate limiter: ip -> list of timestamps
    request_log: dict[str, list[float]] = defaultdict(list)


state = AppState()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load_model() -> None:
    """Load model and metadata."""
    model_path = pathlib.Path(MODEL_DIR)

    if not model_path.exists():
        logger.warning("model_not_found", path=MODEL_DIR)
        # Allow startup without model for testing
        return

    logger.info("loading_model", path=MODEL_DIR)
    state.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if state.tokenizer.pad_token is None:
        state.tokenizer.pad_token = state.tokenizer.eos_token

    state.model = AutoModelForCausalLM.from_pretrained(str(model_path))
    state.model.eval()

    state.pipe = pipeline(
        "text-generation",
        model=state.model,
        tokenizer=state.tokenizer,
        device="cpu",
    )

    # Load manifest metadata
    manifest_path = pathlib.Path(MANIFEST_PATH)
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        state.dataset_hash = manifest.get("dataset_hash", "unknown")

    state.model_version = model_path.name
    state.loaded = True
    logger.info("model_loaded", model_version=state.model_version)


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
def _check_rate_limit(client_ip: str) -> bool:
    """Return True if request is within rate limit."""
    now = time.time()
    window = 60.0  # 1 minute
    timestamps = state.request_log[client_ip]

    # Prune old entries
    state.request_log[client_ip] = [t for t in timestamps if now - t < window]

    if len(state.request_log[client_ip]) >= RATE_LIMIT_RPM:
        return False

    state.request_log[client_ip].append(now)
    return True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy" if state.loaded else "model_not_loaded",
        "model_version": state.model_version,
        "dataset_hash": state.dataset_hash,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, request: Request) -> GenerateResponse:
    """Generate text from a prompt with optional safety guardrails."""
    start = time.time()
    client_ip = request.client.host if request.client else "unknown"

    # Rate limit
    if not _check_rate_limit(client_ip):
        logger.warning("rate_limited", client_ip=client_ip)
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")

    # Model must be loaded
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Guardrails
    engine = GuardrailEngine(safe_mode=req.safe_mode)
    input_check = engine.check_input(req.prompt)

    if not input_check.allowed:
        logger.info(
            "input_blocked",
            reason=input_check.reason,
            category=input_check.category,
            client_ip=client_ip,
        )
        latency = (time.time() - start) * 1000
        return GenerateResponse(
            generated_text=engine.get_refusal_response(),
            prompt=req.prompt,
            model_version=state.model_version,
            dataset_hash=state.dataset_hash,
            token_count=0,
            latency_ms=round(latency, 2),
            safe_mode=req.safe_mode,
        )

    # Generate
    # NOTE: This model was fine-tuned on plain text rather than
    # conversational User/Assistant data, so we use the raw prompt
    # instead of the system-prompt-wrapped version produced by the
    # guardrails engine.
    prompt_text = req.prompt

    outputs = state.pipe(
        prompt_text,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature if req.temperature > 0 else None,
        do_sample=req.temperature > 0,
        pad_token_id=state.tokenizer.pad_token_id,
    )
    generated = outputs[0]["generated_text"]

    # Strip the prompt from the output
    if generated.startswith(prompt_text):
        generated = generated[len(prompt_text) :]
    generated = generated.strip()

    # Output guardrails
    output_check = engine.check_output(generated)
    if not output_check.allowed:
        logger.info("output_blocked", reason=output_check.reason)
        generated = engine.get_refusal_response()

    # Count tokens
    token_count = len(state.tokenizer.encode(generated))
    latency = (time.time() - start) * 1000

    logger.info(
        "generate_request",
        client_ip=client_ip,
        token_count=token_count,
        latency_ms=round(latency, 2),
        model_version=state.model_version,
        dataset_hash=state.dataset_hash,
        safe_mode=req.safe_mode,
    )

    return GenerateResponse(
        generated_text=generated,
        prompt=req.prompt,
        model_version=state.model_version,
        dataset_hash=state.dataset_hash,
        token_count=token_count,
        latency_ms=round(latency, 2),
        safe_mode=req.safe_mode,
    )
