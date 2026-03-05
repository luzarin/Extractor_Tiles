from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_PATH = PROJECT_ROOT / "ui" / "workflow.html"
DOWNLOAD_SCRIPT = PROJECT_ROOT / "scripts" / "download_tiles_from_aoi.py"
RUNTIME_DIR = PROJECT_ROOT / "data" / "ui_runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env", override=False)

PROVIDER_TO_BACKEND = {
    "apple": "jimutmap_apple",
    "esri": "xyz_template",
    "google": "xyz_template",
    "custom_xyz": "xyz_template",
}

DEFAULT_TILE_TEMPLATES = {
    "apple": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",  # unused by apple backend, kept for logs/compat
    "esri": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "google": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    "custom_xyz": "",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobState:
    job_id: str
    kind: str
    status: str = "queued"
    stage: str = "queued"
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    logs: List[str] = field(default_factory=list)
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    cancel_requested: bool = False


class JobCancelledError(RuntimeError):
    pass


class JobManager:
    def __init__(self):
        self._jobs: Dict[str, JobState] = {}
        self._procs: Dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def create(self, kind: str) -> JobState:
        job = JobState(job_id=str(uuid.uuid4()), kind=kind)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> JobState:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def patch(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        stage: Optional[str] = None,
        log: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if status is not None:
                job.status = status
            if stage is not None:
                job.stage = stage
            if log is not None:
                job.logs.append(log.rstrip("\n"))
                if len(job.logs) > 2000:
                    job.logs = job.logs[-2000:]
            if result is not None:
                job.result.update(result)
            if error is not None:
                job.error = error
            job.updated_at = utc_now_iso()

    def attach_process(self, job_id: str, proc: subprocess.Popen) -> None:
        with self._lock:
            self._procs[job_id] = proc
            if job_id in self._jobs:
                self._jobs[job_id].updated_at = utc_now_iso()

    def detach_process(self, job_id: str) -> None:
        with self._lock:
            self._procs.pop(job_id, None)
            if job_id in self._jobs:
                self._jobs[job_id].updated_at = utc_now_iso()

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            return bool(job.cancel_requested)

    def request_cancel(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status in {"completed", "failed", "cancelled"}:
                return {"accepted": False, "message": f"Job already {job.status}.", "had_process": False}
            job.cancel_requested = True
            job.updated_at = utc_now_iso()
            proc = self._procs.get(job_id)

        had_process = proc is not None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        return {"accepted": True, "message": "Cancellation requested.", "had_process": had_process}

    def to_dict(self, job_id: str) -> Dict[str, Any]:
        job = self.get(job_id)
        return {
            "job_id": job.job_id,
            "kind": job.kind,
            "status": job.status,
            "stage": job.stage,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "logs": job.logs,
            "result": job.result,
            "error": job.error,
            "cancel_requested": job.cancel_requested,
        }

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs_sorted = sorted(jobs, key=lambda j: j.created_at, reverse=True)
        return [
            {
                "job_id": job.job_id,
                "kind": job.kind,
                "status": job.status,
                "stage": job.stage,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
                "error": job.error,
                "cancel_requested": job.cancel_requested,
            }
            for job in jobs_sorted
        ]


manager = JobManager()


def _safe_name(name: str) -> str:
    return Path(name).name


def _clean_user_path(path_value: str) -> str:
    raw = (path_value or "").strip()
    raw = raw.replace('\\"', '"').replace("\\'", "'")
    quoted_abs = re.search(r'"([A-Za-z]:\\[^"]+)"', raw)
    if quoted_abs:
        raw = quoted_abs.group(1).strip()
    for _ in range(3):
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
            raw = raw[1:-1].strip()
        else:
            break
    return raw


def _resolve_user_path(path_value: str) -> Path:
    cleaned = _clean_user_path(path_value)
    if not cleaned:
        raise ValueError("Path is empty.")
    p = Path(cleaned)
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def _save_uploads(files: List[UploadFile], out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for f in files:
        filename = _safe_name(f.filename or "")
        if not filename:
            continue
        target = out_dir / filename
        content = f.file.read()
        target.write_bytes(content)
        saved.append(target)
    return saved


def _resolve_aoi_from_uploads(saved_files: List[Path], work_dir: Path) -> Path:
    if not saved_files:
        raise ValueError("No files uploaded.")

    if len(saved_files) == 1 and saved_files[0].suffix.lower() == ".zip":
        zip_path = saved_files[0]
        extract_dir = work_dir / "unzipped_aoi"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        def _single_or_error(candidates: List[Path], label: str) -> Optional[Path]:
            if not candidates:
                return None
            if len(candidates) == 1:
                return candidates[0]
            names = ", ".join(str(p.relative_to(extract_dir)) for p in candidates[:6])
            raise ValueError(
                f"ZIP has multiple {label} files ({len(candidates)}). "
                f"Keep a single AOI dataset in the ZIP. Found: {names}"
            )

        shp_files = sorted(extract_dir.rglob("*.shp"))
        p = _single_or_error(shp_files, "SHP")
        if p is not None:
            return p
        gpkg_files = sorted(extract_dir.rglob("*.gpkg"))
        p = _single_or_error(gpkg_files, "GPKG")
        if p is not None:
            return p
        geojson_files = sorted(extract_dir.rglob("*.geojson"))
        p = _single_or_error(geojson_files, "GeoJSON")
        if p is not None:
            return p
        raise ValueError("ZIP uploaded but no .shp/.gpkg/.geojson found inside.")

    preferred_single = [p for p in saved_files if p.suffix.lower() in {".gpkg", ".geojson", ".json"}]
    if preferred_single:
        return preferred_single[0]

    shp_files = [p for p in saved_files if p.suffix.lower() == ".shp"]
    if shp_files:
        shp = shp_files[0]
        stem = shp.with_suffix("")
        missing = []
        for ext in (".dbf", ".shx"):
            if not (stem.with_suffix(ext)).exists():
                missing.append(ext)
        if missing:
            raise ValueError(
                "Shapefile incomplete. Upload .shp + .dbf + .shx (and .prj recommended), or upload a .zip."
            )
        return shp

    raise ValueError("Unsupported AOI format. Use .zip, .gpkg, .geojson, or shapefile set (.shp/.dbf/.shx).")


def _run_command(job_id: str, cmd: List[str], cwd: Path) -> None:
    if manager.is_cancel_requested(job_id):
        raise JobCancelledError("Cancelled before command start.")

    manager.patch(job_id, log=f"$ {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    manager.attach_process(job_id, proc)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            manager.patch(job_id, log=line.rstrip("\n"))
            if manager.is_cancel_requested(job_id) and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
        proc.wait()
    finally:
        manager.detach_process(job_id)

    if manager.is_cancel_requested(job_id):
        raise JobCancelledError("Cancelled by user.")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def _provider_to_tile_template(provider: str, tile_url_template: str) -> str:
    if provider in {"esri", "google"}:
        return DEFAULT_TILE_TEMPLATES[provider]
    if provider == "custom_xyz":
        tmpl = (tile_url_template or "").strip()
        if not tmpl:
            raise ValueError("Custom XYZ provider requires tile_url_template.")
        return tmpl
    return DEFAULT_TILE_TEMPLATES["apple"]


def _worker_download(
    job_id: str,
    *,
    aoi_file: Path,
    aoi_layer: str,
    aoi_where: str,
    zoom: int,
    provider: str,
    tile_url_template: str,
    output_tif: str,
    jimutmap_threads: int,
    jimutmap_v: int,
    jimutmap_access_key: str,
    jimutmap_container_dir: str,
    max_tiles: int,
) -> None:
    try:
        manager.patch(job_id, status="running", stage="downloading_tiles")
        output_tif_path = _resolve_user_path(output_tif)
        output_tif_path.parent.mkdir(parents=True, exist_ok=True)

        backend_provider = PROVIDER_TO_BACKEND[provider]
        resolved_template = _provider_to_tile_template(provider, tile_url_template)

        cmd = [
            sys.executable,
            "-u",
            str(DOWNLOAD_SCRIPT),
            "--aoi-file",
            str(aoi_file),
            "--zoom",
            str(zoom),
            "--provider",
            backend_provider,
            "--tile-url-template",
            resolved_template,
            "--output-tif",
            str(output_tif_path),
            "--jimutmap-threads",
            str(jimutmap_threads),
            "--jimutmap-v",
            str(jimutmap_v),
            "--max-tiles",
            str(max_tiles),
        ]
        if aoi_layer:
            cmd.extend(["--aoi-layer", aoi_layer])
        if aoi_where:
            cmd.extend(["--aoi-where", aoi_where])
        if jimutmap_access_key:
            cmd.extend(["--jimutmap-access-key", jimutmap_access_key])
        if jimutmap_container_dir:
            cmd.extend(["--jimutmap-container-dir", str(_resolve_user_path(jimutmap_container_dir))])

        _run_command(job_id, cmd, PROJECT_ROOT)
        manager.patch(
            job_id,
            status="completed",
            stage="completed",
            result={
                "aoi_file": str(aoi_file),
                "output_tif": str(output_tif_path),
                "provider": provider,
                "backend_provider": backend_provider,
                "tile_url_template": resolved_template,
            },
        )
    except JobCancelledError as exc:
        manager.patch(job_id, status="cancelled", stage="cancelled", error=None, log=f"[INFO] {exc}")
    except Exception as exc:
        if manager.is_cancel_requested(job_id):
            manager.patch(job_id, status="cancelled", stage="cancelled", error=None, log="[INFO] Cancelled by user.")
        else:
            manager.patch(job_id, status="failed", stage="failed", error=str(exc), log=f"[ERROR] {exc}")


def _start_thread(target, *args, **kwargs) -> None:
    thread = threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True)
    thread.start()


app = FastAPI(title="Tiles Downloader UI API", version="1.0.0")


@app.get("/")
def workflow_ui() -> FileResponse:
    if not UI_PATH.exists():
        raise HTTPException(status_code=404, detail="workflow.html not found")
    return FileResponse(UI_PATH)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/tiles/start")
async def start_download(
    aoi_files: List[UploadFile] = File(...),
    provider: str = Form("apple"),
    zoom: int = Form(18),
    tile_url_template: str = Form(""),
    output_tif: str = Form("data/aoi_tiles.tif"),
    aoi_layer: str = Form(""),
    aoi_where: str = Form(""),
    jimutmap_threads: int = Form(20),
    jimutmap_v: int = Form(10221),
    max_tiles: int = Form(50000),
    jimutmap_access_key: str = Form(""),
    jimutmap_container_dir: str = Form(""),
) -> Dict[str, Any]:
    if provider not in PROVIDER_TO_BACKEND:
        raise HTTPException(status_code=400, detail="provider must be apple|esri|google|custom_xyz")

    if provider == "custom_xyz" and not tile_url_template.strip():
        raise HTTPException(status_code=400, detail="custom_xyz requires tile_url_template.")

    if provider == "apple":
        fallback_key = os.environ.get("JIMUTMAP_ACCESS_KEY", "").strip()
        if not (jimutmap_access_key.strip() or fallback_key):
            raise HTTPException(
                status_code=400,
                detail="Apple provider requires access key. Fill 'jimutmap access key' or set JIMUTMAP_ACCESS_KEY.",
            )

    job = manager.create("download_tiles")
    job_dir = UPLOAD_DIR / job.job_id
    saved_files = _save_uploads(aoi_files, job_dir)

    try:
        aoi_file = _resolve_aoi_from_uploads(saved_files, job_dir)
    except Exception as exc:
        manager.patch(job.job_id, status="failed", stage="failed", error=str(exc), log=f"[ERROR] {exc}")
        raise HTTPException(status_code=400, detail=str(exc))

    manager.patch(
        job.job_id,
        stage="queued",
        result={"aoi_file": str(aoi_file), "output_tif_requested": output_tif, "provider": provider},
    )
    _start_thread(
        _worker_download,
        job.job_id,
        aoi_file=aoi_file,
        aoi_layer=aoi_layer,
        aoi_where=aoi_where,
        zoom=zoom,
        provider=provider,
        tile_url_template=tile_url_template,
        output_tif=output_tif,
        jimutmap_threads=jimutmap_threads,
        jimutmap_v=jimutmap_v,
        max_tiles=max_tiles,
        jimutmap_access_key=jimutmap_access_key.strip() or os.environ.get("JIMUTMAP_ACCESS_KEY", "").strip(),
        jimutmap_container_dir=jimutmap_container_dir,
    )
    return {"job_id": job.job_id}


@app.get("/tiles/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    try:
        return manager.to_dict(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")


@app.post("/tiles/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> Dict[str, Any]:
    try:
        result = manager.request_cancel(job_id)
        if result.get("accepted"):
            manager.patch(job_id, stage="cancelling", log="[INFO] Cancellation requested by user.")
        return {"job_id": job_id, **result}
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")


@app.get("/tiles/jobs")
def list_jobs() -> Dict[str, Any]:
    return {"jobs": manager.list_jobs()}


@app.get("/tiles/jobs/{job_id}/logs")
def get_job_logs(job_id: str) -> Dict[str, Any]:
    try:
        data = manager.to_dict(job_id)
        return {"job_id": job_id, "logs": data["logs"]}
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")


@app.get("/tiles/examples")
def examples() -> Dict[str, Any]:
    return {
        "providers": list(PROVIDER_TO_BACKEND.keys()),
        "example_esri_template": DEFAULT_TILE_TEMPLATES["esri"],
        "example_google_template": DEFAULT_TILE_TEMPLATES["google"],
        "example_custom_template": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    }
