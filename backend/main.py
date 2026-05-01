import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from db.database import init_db
from routers import analyze, patients, scores, ws

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("neoguard")

app = FastAPI(title="NeoGuard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(patients.router)
app.include_router(scores.router)
app.include_router(ws.router)
app.include_router(analyze.router)


@app.on_event("startup")
async def _startup() -> None:
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    await init_db()
    log.info("neoguard up: models_dir=%s", settings.models_dir)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/")
async def root() -> dict:
    return {"name": settings.app_name, "version": "1.0.0", "docs": "/docs"}
