"""FastAPI entry point for the moderated chat."""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter_ns

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .models import MODEL_NAMES, ModelBank, init_cpu
from .text import normalize, rule_label

WEB = Path(__file__).with_name("web")
MASK = "[Message hidden by moderation]"


class Message(BaseModel):
    text: str = Field(min_length=1, max_length=600)
    model: str = "int8"


def create_app(artifact_path: Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        init_cpu(int(os.environ.get("CHATGUARD_THREADS", "2")))
        app.state.bank = ModelBank(artifact_path)
        yield

    app = FastAPI(title="ChatGuard", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(WEB / "index.html")

    @app.get("/api/health")
    def health():
        return {"status": "ready", "default_model": "int8", "models": MODEL_NAMES}

    @app.post("/api/moderate")
    def moderate(message: Message):
        if message.model not in MODEL_NAMES:
            raise HTTPException(status_code=422, detail="Choose one of the four models.")
        if not message.text.strip():
            raise HTTPException(status_code=422, detail="Enter a message first.")
        return moderate_text(app.state.bank, message)
    return app


def moderate_text(bank: ModelBank, message: Message) -> dict:
    start = perf_counter_ns()
    text = normalize(message.text)
    probabilities = bank.probabilities(message.model, [text])[0]
    label = int(probabilities.argmax())
    return {
        "model": message.model,
        "model_name": MODEL_NAMES[message.model],
        "label": "offensive" if label else "clean",
        "confidence": round(float(probabilities[label]), 4),
        "display_text": MASK if label else message.text.strip(),
        "rule_flag": bool(rule_label(text)),
        "latency_ms": round((perf_counter_ns() - start) / 1_000_000, 3),
    }


app = create_app(Path(os.environ.get("CHATGUARD_ARTIFACTS", "artifacts")))
