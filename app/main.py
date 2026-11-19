from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import scrape
from .agent import answer
from .llm import OpenRouter
from .retrieval import HybridIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("app")

ROOT = Path(__file__).resolve().parent
PAGES = scrape.OUT_FILE
STALE_AFTER_H = float(os.getenv("REFRESH_IF_OLDER_THAN_HOURS", "24"))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

app = FastAPI(title="Dar Global & Wasalt Assistant")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

index = HybridIndex()
llm = OpenRouter()
state = {"status": "starting", "refreshing": False, "last_refresh": None, "error": None}
_lock = threading.Lock()


def _rebuild() -> None:
    index.build(PAGES, use_embeddings=os.getenv("DISABLE_EMBEDDINGS") != "1")
    state["status"] = "ready" if index.chunks else "no data"


def refresh_data() -> None:
    if not _lock.acquire(blocking=False):
        return
    state["refreshing"] = True
    try:
        rc = scrape.main()
        if rc == 0:
            state["last_refresh"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
            state["error"] = None
        else:
            state["error"] = "Scrape returned no pages; serving previous data"
        _rebuild()
    except Exception as e:
        log.exception("refresh failed")
        state["error"] = str(e)
    finally:
        state["refreshing"] = False
        _lock.release()


def startup_job() -> None:
    llm.discover_free_models()
    if PAGES.exists():  # serve baked-in data right away
        state["last_refresh"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(PAGES.stat().st_mtime))
        _rebuild()
    age_h = (time.time() - PAGES.stat().st_mtime) / 3600 if PAGES.exists() else 1e9
    if age_h > STALE_AFTER_H:
        refresh_data()


@app.on_event("startup")
def on_startup() -> None:
    threading.Thread(target=startup_job, daemon=True).start()


# --- simple per-IP rate limit so a public link can't burn the API key ---
_hits: dict[str, deque] = defaultdict(deque)
RATE = int(os.getenv("RATE_LIMIT_PER_10MIN", "25"))


def _rate_ok(ip: str) -> bool:
    now, q = time.time(), _hits[ip]
    while q and now - q[0] > 600:
        q.popleft()
    if len(q) >= RATE:
        return False
    q.append(now)
    return True


class Msg(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    history: list[Msg] = []


@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/status")
def status():
    return {
        **state,
        **index.stats(),
        "llm_configured": llm.configured,
        "models": llm.models[:5],
        "last_model": llm.last_model,
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/chat")
def chat(body: ChatIn, request: Request):
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0]
    if not _rate_ok(ip):
        raise HTTPException(429, "Too many messages. Please wait a few minutes.")
    if not llm.configured:
        raise HTTPException(503, "The server has no OpenRouter API key configured.")
    try:
        return answer(body.message, [m.model_dump() for m in body.history], index, llm)
    except RuntimeError as e:
        log.warning("chat failed: %s", e)
        raise HTTPException(503, "The free AI models are busy right now. Please try again in a minute.")


@app.post("/api/refresh")
def refresh(x_admin_token: str = Header(default="")):
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(403, "Forbidden")
    if state["refreshing"]:
        return {"started": False, "reason": "already running"}
    threading.Thread(target=refresh_data, daemon=True).start()
    return {"started": True}
