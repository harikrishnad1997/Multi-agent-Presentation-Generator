"""FastAPI interface for the presentation generator.

Run with ``uvicorn presentation_generator.api:app --reload`` after adding
``src`` to ``PYTHONPATH`` (or use ``python serve.py`` from the repository).
"""
from __future__ import annotations

import os
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .comments import edit_deck_file
from .graph import run
from .llm import use_api_key
from .tracing import use_opik_api_key
from .sources import load_source

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
UPLOAD_EXTENSIONS = {".pdf", ".txt", ".md"}


class GenerationRequest(BaseModel):
    topic: str | None = Field(default=None, max_length=500)
    source: str | None = Field(default=None, description="Local file/folder path or public GitHub URL")
    source_text: str = Field(default="", max_length=12_000, description="Source content supplied directly by the UI")
    references: list[str] = Field(default_factory=list, description="Source citations to include on the references slide")
    author: str = "Harikrishna Dev"
    institute: str = "University of Texas at Dallas"
    slides: int = Field(default=30, ge=10, le=40)
    model: str | None = None
    charts: bool = True
    images: bool = False
    search: bool = True
    shallow: bool = False
    comments: str = ""
    api_key: str | None = Field(default=None, max_length=500, repr=False)
    opik_api_key: str | None = Field(default=None, max_length=500, repr=False)


class EditRequest(BaseModel):
    comments: str = Field(min_length=1, description="Instructions such as 'Slide 3: revise using the supplied source'")
    source: str | None = Field(default=None, description="Local file/folder path or public GitHub URL")
    source_text: str = Field(default="", max_length=12_000, description="Source content to ground this revision")
    model: str | None = None
    author: str = "Harikrishna Dev"
    institute: str = "University of Texas at Dallas"
    conversation_context: str = Field(default="", max_length=12_000, description="Prior chat context for interpreting this edit")
    api_key: str | None = Field(default=None, max_length=500, repr=False)


app = FastAPI(title="Multi-agent Presentation Generator", version="1.0.0")
origins = [value.strip() for value in os.getenv(
    "PRESENTATION_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
).split(",") if value.strip()]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True,
    allow_methods=["GET", "POST"], allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=max(1, int(os.getenv("PRESENTATION_API_WORKERS", "2"))))
_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()
_output_root = Path(os.getenv("PRESENTATION_OUTPUT_DIR", "out/api")).resolve()


def _restore_completed_jobs() -> None:
    """Restore completed artifact-backed jobs after an API restart."""
    if not _output_root.is_dir():
        return
    for folder in _output_root.iterdir():
        deck_path = folder / "deck.json"
        if not folder.is_dir() or not deck_path.is_file():
            continue
        try:
            import json

            deck = json.loads(deck_path.read_text(encoding="utf-8"))
            outline_path, review_path = folder / "outline.json", folder / "review.json"
            outline = json.loads(outline_path.read_text(encoding="utf-8")) if outline_path.exists() else None
            review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else None
            chart_dir = folder / "charts"
            chart_count = sum(1 for path in chart_dir.glob("*.png")) if chart_dir.is_dir() else 0
            pdf_path = str(folder / "presentation.pdf") if (folder / "presentation.pdf").exists() else None
            result = {
                "topic": deck.get("title"), "deck": deck, "outline": outline,
                "review": review, "chart_count": chart_count, "pdf_path": pdf_path,
                "attempts": 1,
            }
            _jobs[folder.name] = {
                "id": folder.name, "status": "completed", "operation": "restored",
                "created_at": datetime.fromtimestamp(folder.stat().st_mtime, timezone.utc).isoformat(),
                "result": result,
            }
        except (OSError, ValueError, TypeError):
            continue


_restore_completed_jobs()


async def _read_uploaded_source(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Upload a PDF, TXT, or Markdown file.")
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Source file must be 20 MB or smaller.")
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="mapg-source-", suffix=suffix, delete=False) as temp:
            temp.write(data)
            temp_path = temp.name
        return load_source(temp_path).text
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


def _run_job(job_id: str, request: GenerationRequest, output_dir: Path) -> None:
    with _lock:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()
    try:
        topic = request.topic.strip() if request.topic else ""
        source_text = request.source_text.strip()
        if request.source:
            source = load_source(request.source)
            source_text = (source_text + "\n\n" + source.text).strip()
            request.references.append(source.ref)
            topic = topic or source.title
        if not topic:
            raise ValueError("Provide a topic or a source.")
        with use_api_key(request.api_key), use_opik_api_key(request.opik_api_key):
            result = run(
                topic=topic, author=request.author, institute=request.institute,
                num_slides=request.slides, model=request.model, output_dir=str(output_dir),
                charts_enabled=request.charts, source_text=source_text[:12_000],
                comments=request.comments, search=request.search, shallow=request.shallow,
                images_enabled=request.images,
                references=request.references,
            )
        with _lock:
            _jobs[job_id].update(status="completed", result=result)
    except Exception as exc:  # report a useful error without exposing a traceback
        with _lock:
            _jobs[job_id].update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        with _lock:
            _jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()


def _run_edit_job(job_id: str, parent_id: str, request: EditRequest, output_dir: Path) -> None:
    with _lock:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()
    try:
        parent_dir = _output_root / parent_id
        source_text = request.source_text.strip()
        if request.source:
            source_text = (source_text + "\n\n" + load_source(request.source).text).strip()
        with use_api_key(request.api_key):
            result = edit_deck_file(
                deck_path=str(parent_dir / "deck.json"),
                outline_path=str(parent_dir / "outline.json"),
                comments=(
                    f"Conversation context for interpreting the latest request (do not reapply prior requests):\n"
                    f"{request.conversation_context.strip()}\n\nLatest requested edit:\n{request.comments}"
                    if request.conversation_context.strip() else request.comments
                ), model=request.model,
                author=request.author, institute=request.institute,
                output_dir=str(output_dir), source_material=source_text[:12_000],
            )
        with _lock:
            _jobs[job_id].update(status="completed", result=result)
    except Exception as exc:  # report a useful error without exposing a traceback
        with _lock:
            _jobs[job_id].update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        with _lock:
            _jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()


def _public_job(job_id: str) -> dict[str, Any]:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Generation job not found")
        payload = {key: value for key, value in job.items() if key != "result"}
        payload["status_url"] = f"/api/generations/{job_id}"
        if job["status"] == "completed":
            payload["result_url"] = f"/api/generations/{job_id}/result"
        return payload


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/generations", status_code=202)
def create_generation(request: GenerationRequest) -> dict[str, Any]:
    if not request.topic and not request.source and not request.source_text.strip():
        raise HTTPException(status_code=422, detail="Provide a topic or a source.")
    job_id = uuid.uuid4().hex
    output_dir = _output_root / job_id
    output_dir.mkdir(parents=True, exist_ok=False)
    with _lock:
        _jobs[job_id] = {
            "id": job_id, "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    _executor.submit(_run_job, job_id, request, output_dir)
    return _public_job(job_id)


@app.post("/api/generations/upload", status_code=202)
async def create_generation_upload(
    file: UploadFile = File(...), topic: str | None = Form(default=None),
    author: str = Form(default="Harikrishna Dev"),
    institute: str = Form(default="University of Texas at Dallas"),
    slides: int = Form(default=30, ge=10, le=40), model: str | None = Form(default=None),
    charts: bool = Form(default=True), search: bool = Form(default=True),
    shallow: bool = Form(default=False), images: bool = Form(default=False), comments: str = Form(default=""),
    source_name: str | None = Form(default=None),
    api_key: str | None = Form(default=None), opik_api_key: str | None = Form(default=None),
) -> dict[str, Any]:
    source_text = await _read_uploaded_source(file)
    await file.close()
    request = GenerationRequest(
        topic=topic, source_text=source_text, author=author, institute=institute,
        slides=slides, model=model, charts=charts, search=search,
        shallow=shallow, images=images, comments=comments, api_key=api_key,
        opik_api_key=opik_api_key,
        references=[f"Uploaded source: {source_name}"] if source_name else [],
    )
    return create_generation(request)


@app.get("/api/generations/{job_id}")
def generation_status(job_id: str) -> dict[str, Any]:
    return _public_job(job_id)


@app.post("/api/generations/{job_id}/edits", status_code=202)
def create_edit(job_id: str, request: EditRequest) -> dict[str, Any]:
    with _lock:
        parent = _jobs.get(job_id)
        if not parent:
            raise HTTPException(status_code=404, detail="Generation job not found")
        if parent["status"] != "completed":
            raise HTTPException(status_code=409, detail="The source generation must be completed before editing")
    revision_id = uuid.uuid4().hex
    output_dir = _output_root / revision_id
    output_dir.mkdir(parents=True, exist_ok=False)
    with _lock:
        _jobs[revision_id] = {
            "id": revision_id, "status": "queued", "operation": "edit",
            "parent_id": job_id, "created_at": datetime.now(timezone.utc).isoformat(),
        }
    _executor.submit(_run_edit_job, revision_id, job_id, request, output_dir)
    return _public_job(revision_id)


@app.post("/api/generations/{job_id}/edits/upload", status_code=202)
async def create_edit_upload(
    job_id: str, file: UploadFile = File(...),
    comments: str = Form(..., min_length=1), model: str | None = Form(default=None),
    author: str = Form(default="Harikrishna Dev"),
    institute: str = Form(default="University of Texas at Dallas"),
    conversation_context: str = Form(default=""),
) -> dict[str, Any]:
    source_text = await _read_uploaded_source(file)
    await file.close()
    return create_edit(job_id, EditRequest(
        comments=comments, source_text=source_text, model=model,
        author=author, institute=institute, conversation_context=conversation_context,
    ))


@app.get("/api/generations/{job_id}/result")
def generation_result(job_id: str) -> dict[str, Any]:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Generation job not found")
        if job["status"] != "completed":
            raise HTTPException(status_code=409, detail=f"Generation is {job['status']}")
        result = job["result"]
    output = _output_root / job_id
    artifacts = [p.name for p in output.iterdir() if p.is_file()]
    charts_dir = output / "charts"
    if charts_dir.is_dir():
        artifacts.extend(f"charts/{p.name}" for p in charts_dir.iterdir() if p.is_file())
    images_dir = output / "images"
    if images_dir.is_dir():
        artifacts.extend(f"images/{p.name}" for p in images_dir.iterdir() if p.is_file())
    return {
        "id": job_id, "topic": result.get("topic") or (result.get("deck") or {}).get("title"),
        "deck": result.get("deck"),
        "outline": result.get("outline"), "review": result.get("review"),
        "grounding": result.get("grounding"), "attempts": result.get("attempts"),
        "chart_count": result.get("chart_count", 0), "pdf_path": result.get("pdf_path"),
        "image_count": result.get("image_count", 0),
        "compile_log": result.get("compile_log"), "artifacts": artifacts,
        "artifact_base_url": f"/api/generations/{job_id}/artifacts/",
    }


@app.get("/api/generations/{job_id}/artifacts/{artifact_path:path}")
def generation_artifact(job_id: str, artifact_path: str) -> FileResponse:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Generation job not found")
        if job["status"] != "completed":
            raise HTTPException(status_code=409, detail=f"Generation is {job['status']}")
    root = (_output_root / job_id).resolve()
    target = (root / artifact_path).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(target)


@app.get("/")
def root() -> dict[str, str]:
    return {"name": app.title, "docs": "/docs", "health": "/api/health"}
