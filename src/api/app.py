"""
FastAPI application for building use classification.

Start:
    source .venv/bin/activate
    python src/api/app.py

Then open http://localhost:7860
"""

import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.base import BaseHTTPMiddleware

from src.persistence.database import init_db, get_db_session
from src.persistence.repository import PredictionRepository
from src.adapters.foundry_adapter import FoundryVlmAdapter
from src.adapters.mock_adapter import MockVlmAdapter
from src.application.classify_service import ClassifyService
from src.application.review_service import ReviewService

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def _cleanup_stale_sdk_subprocesses() -> int:
    """Kill any leftover Azure ML SDK subprocesses from previous crashes.

    The SDK's begin_create_or_update / begin_delete use multiprocessing.spawn
    to run the poller in a separate process. If the parent process crashes
    or is killed, the poller subprocesses can survive as zombies, holding
    file descriptors and eventually hanging the asyncio event loop.

    Returns the number of subprocesses killed.
    """
    import os, signal, subprocess
    try:
        result = subprocess.run(
            ["pgrep", "-f", "multiprocessing.spawn.*spawn_main"],
            capture_output=True, text=True, timeout=5,
        )
        killed = 0
        for pid_str in result.stdout.strip().split("\n"):
            if not pid_str:
                continue
            try:
                pid = int(pid_str)
                if pid == os.getpid():
                    continue
                os.kill(pid, signal.SIGTERM)
                killed += 1
            except (ProcessLookupError, ValueError):
                pass
        if killed:
            logger.info(f"Cleaned up {killed} stale SDK subprocess(es) from previous run")
        return killed
    except Exception as e:
        logger.debug(f"Subprocess cleanup skipped: {e}")
        return 0


classify_service: ClassifyService = None
review_service: ReviewService = None
use_mock: bool = False
traffic_pct: int = 0
_traffic_client = None
_traffic_endpoint_name: str = ""
_traffic_deployment_name: str = ""

_traffic_endpoint_name = os.getenv("ENDPOINT_NAME", "")
_traffic_deployment_name = os.getenv("DEPLOYMENT_NAME", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classify_service, review_service, use_mock, traffic_pct
    global _traffic_client, _traffic_endpoint_name, _traffic_deployment_name

    _cleanup_stale_sdk_subprocesses()

    await init_db()
    logger.info("Database initialized")

    use_mock = False
    force_mock = os.getenv("FORCE_MOCK", "").lower() in ("1", "true", "yes")

    if force_mock:
        vlm_client = MockVlmAdapter(latency_ms=200)
        use_mock = True
        logger.info("Forced mock mode via FORCE_MOCK env var")
    else:
        try:
            vlm_client = FoundryVlmAdapter(prompt_version="p004")
            if vlm_client.health_check():
                logger.info("Using Foundry VLM adapter (p004)")
            else:
                raise Exception("Health check failed")
        except Exception as e:
            logger.warning(f"Foundry adapter unavailable: {e}. Falling back to mock.")
            vlm_client = MockVlmAdapter(latency_ms=200)
            use_mock = True

    classify_service = ClassifyService(vlm_client=vlm_client)
    logger.info(f"Classification service ready (mock={use_mock})")
    review_service = ReviewService()
    logger.info("Review service ready")


    if _traffic_endpoint_name and not use_mock:
        try:
            credential = DefaultAzureCredential()
            _traffic_client = MLClient(
                credential=credential,
                subscription_id=os.getenv("SUBSCRIPTION_ID"),
                resource_group_name=os.getenv("RESOURCE_GROUP"),
                workspace_name=os.getenv("WORKSPACE_NAME"),
            )
            ep = _traffic_client.online_endpoints.get(_traffic_endpoint_name)
            if _traffic_deployment_name in ep.traffic:
                traffic_pct = ep.traffic[_traffic_deployment_name]
            else:
                available = [d for d, t in ep.traffic.items() if t > 0]
                if available:
                    _traffic_deployment_name = available[0]
                    traffic_pct = ep.traffic[_traffic_deployment_name]
                    logger.warning(f"Configured deployment not found, falling back to {available[0]}")
                else:
                    traffic_pct = 0
                    logger.warning(f"No deployment with traffic, using 0%")
            logger.info(f"Endpoint traffic: {traffic_pct}% (deployment={_traffic_deployment_name})")
        except Exception as e:
            logger.warning(f"Cannot read endpoint traffic: {e}")

    yield

    logger.info("Shutting down")


app = FastAPI(
    title="Bogota Building Use Classification",
    version="0.1.0",
    lifespan=lifespan,
)


class AdminKeyMiddleware(BaseHTTPMiddleware):
    """Require ADMIN_API_KEY for /admin/* endpoints when configured.

    If ADMIN_API_KEY is not set, the middleware is transparent.
    This is a lightweight guard for the deployment-control endpoints;
    it does not replace full authentication (deferred to Phase 10).
    """

    async def dispatch(self, request: Request, call_next):
        admin_key = os.getenv("ADMIN_API_KEY", "")
        if admin_key and request.url.path.startswith("/admin"):
            provided = request.headers.get("X-Admin-Key") or request.query_params.get("admin_key")
            if provided != admin_key:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing admin key"},
                )
        return await call_next(request)


app.add_middleware(AdminKeyMiddleware)


TEST_IMAGES_DIR = Path(__file__).resolve().parents[2] / "experiments" / "test_images"
if TEST_IMAGES_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(TEST_IMAGES_DIR)), name="images")


class ClassifyRequest(BaseModel):
    building_id: str = Field(..., min_length=1, description="Building identifier")
    image_urls: list[str] = Field(
        ..., min_length=1, max_length=8, description="Image URLs or file paths"
    )
    job_id: Optional[str] = Field(None, description="Optional batch job identifier")


class TrafficRequest(BaseModel):
    traffic: int = Field(..., ge=0, le=100, description="Traffic percentage (0 or 100)")


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=_render_classify_ui())


def _render_classify_ui() -> str:
    """Render the Classify page with current server state injected.

    Uses .replace() (not .format()) because the HTML contains literal
    { and } in JavaScript that would confuse str.format().
    """
    return (
        CLASSIFY_UI_HTML
        .replace("{mock_badge}", _mock_badge())
        .replace("{use_mock_banner}", _use_mock_banner())
    )


def _mock_badge() -> str:
    """Small ⚠ MOCK badge for the nav bar (all tabs)."""
    if not use_mock:
        return ''
    return (
        '<span style="background:#f59e0b;color:white;padding:2px 8px;'
        'border-radius:4px;font-size:11px;font-weight:bold;margin-left:8px;">'
        '⚠ MOCK</span>'
    )


def _use_mock_banner() -> str:
    """Yellow warning banner at the top of the Classify page."""
    if not use_mock:
        return ''
    return (
        '<div style="background:#fef3c7;border:2px solid #f59e0b;border-radius:6px;'
        'padding:12px 16px;margin:0 0 16px 0;color:#92400e;font-size:14px;font-weight:500;">'
        '⚠️ <strong>MOCK MODE</strong> — Results are simulated by <code>MockVlmAdapter</code> for UI testing. '
        'Every classify call returns the same hardcoded response. '
        'The real VLM will be used once the online deployment is running.'
        '</div>'
    )


def _render_nav() -> str:
    """Render the nav bar with the {mock_badge} placeholder filled in.

    Uses .replace() (not .format()) because the HTML contains literal
    { and } in JavaScript that would confuse str.format().
    """
    return NAV_HTML.replace("{mock_badge}", _mock_badge())


@app.get("/health")
async def health():
    if use_mock:
        _try_promote_to_real_vlm()
    return {
        "status": "healthy",
        "mock_mode": use_mock,
        "vlm_available": classify_service.vlm_client.health_check() if classify_service else False,
    }


def _try_promote_to_real_vlm() -> bool:
    """Self-heal: if currently in mock mode, try to swap to the real VLM.

    Called from /health and /classify when use_mock is True. If the Foundry
    adapter's health check passes, replaces classify_service.vlm_client with
    the real adapter and flips use_mock to False.

    This lets the server recover without a restart after the user creates
    the deployment via the Start button (5-10 min Azure provisioning).

    Returns True if promoted to real VLM, False if still in mock mode.
    """
    global use_mock
    if not use_mock or classify_service is None:
        return False
    try:
        from src.adapters.foundry_adapter import FoundryVlmAdapter
        candidate = FoundryVlmAdapter(prompt_version="p004")
        if candidate.health_check():
            classify_service.vlm_client = candidate
            use_mock = False
            logger.info("Self-heal: promoted to Foundry VLM adapter (mock mode off)")
            return True
    except Exception as e:
        logger.debug(f"Self-heal not yet ready: {e}")
    return False


@app.post("/classify")
async def classify(req: ClassifyRequest, session: AsyncSession = Depends(get_db_session)):
    if use_mock:
        _try_promote_to_real_vlm()

    if not use_mock and classify_service is not None:
        from src.infrastructure.aml_online import get_online_manager, load_state, OnlineState
        from src.infrastructure.aml_online import STATE_FILE as _ONLINE_STATE_FILE
        try:
            status = get_online_manager().get_status()
        except Exception:
            status = load_state()
        state = status.get("state", "not_found")
        if state != OnlineState.RUNNING.value:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "Online deployment is not running. Click 'Start Deployment' on the Classify page and wait 1-2 minutes.",
                    "state": state,
                },
            )

    logger.info(f"Classifying building {req.building_id} with {len(req.image_urls)} images")

    prediction = await classify_service.classify_building(
        building_id=req.building_id,
        image_urls=req.image_urls,
        job_id=req.job_id,
    )

    repo = PredictionRepository(session)
    prediction = await repo.save(prediction)

    return JSONResponse(content=prediction.to_dict())


@app.get("/classifications/{prediction_id}")
async def get_classification(prediction_id: str, session: AsyncSession = Depends(get_db_session)):
    repo = PredictionRepository(session)
    pred = await repo.get_by_id(prediction_id)
    if not pred:
        raise HTTPException(status_code=404, detail="Classification not found")
    return JSONResponse(content=pred.to_dict())



@app.get("/admin/traffic")
async def get_traffic():
    global traffic_pct
    if not _traffic_client:
        return {"traffic_pct": traffic_pct, "available": False, "note": "Azure client not initialized (mock mode?)"}
    try:
        ep = _traffic_client.online_endpoints.get(_traffic_endpoint_name)
        if _traffic_deployment_name in ep.traffic:
            traffic_pct = ep.traffic[_traffic_deployment_name]
        else:
            available = [d for d, t in ep.traffic.items() if t > 0]
            if available:
                return {"traffic_pct": ep.traffic[available[0]], "available": True, "note": f"Using deployment {available[0]} (configured {_traffic_deployment_name} not found)"}
            traffic_pct = 0
    except Exception as e:
        return {"traffic_pct": traffic_pct, "available": False, "error": str(e)}
    return {"traffic_pct": traffic_pct, "available": True}


@app.post("/admin/traffic")
async def set_traffic(req: TrafficRequest):
    global traffic_pct
    if not _traffic_client:
        raise HTTPException(status_code=503, detail="Azure client not available (mock mode?)")
    ep = _traffic_client.online_endpoints.get(_traffic_endpoint_name)
    ep.traffic = {_traffic_deployment_name: req.traffic}
    _traffic_client.begin_create_or_update(ep).wait(timeout=120)
    traffic_pct = req.traffic
    return {"traffic_pct": traffic_pct, "status": "updated"}


@app.get("/admin/online/status")
async def get_online_status():
    """Return the current online deployment state for the UI.

    Reads Azure (with state file cache fallback) so the UI gets a fast response
    on page load. Used by /admin/online/start and /admin/online/stop as a
    guard against no-op transitions.

    The Azure SDK call is wrapped in asyncio.to_thread so it doesn't block
    the event loop while waiting for Azure to respond.
    """
    from src.infrastructure.aml_online import load_state
    if not os.getenv("ENDPOINT_NAME"):
        return await asyncio.to_thread(load_state)
    return await asyncio.to_thread(_get_online_status_sync)


def _get_online_status_sync():
    from src.infrastructure.aml_online import get_online_manager, load_state
    try:
        return get_online_manager().get_status()
    except Exception as e:
        logger.warning(f"online status error: {e}")
        cached = load_state()
        cached["error"] = str(e)
        return cached


@app.post("/admin/online/start")
async def start_online_deployment():
    """Start the online deployment. Fire-and-forget: returns in <100ms with
    state=creating; the actual provisioning runs on Azure (5-10 min)."""
    if not os.getenv("ENDPOINT_NAME"):
        raise HTTPException(status_code=400, detail="ENDPOINT_NAME not configured in .env")
    return JSONResponse(content=await asyncio.to_thread(_start_online_sync), status_code=202)


def _start_online_sync():
    from src.infrastructure.aml_online import get_online_manager
    try:
        return get_online_manager().start()
    except Exception as e:
        logger.error(f"start failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start: {e}")


@app.post("/admin/online/stop")
async def stop_online_deployment():
    """Stop the online deployment (delete). Fire-and-forget: returns in <100ms
    with state=deleting; the actual teardown runs on Azure (~30 sec)."""
    if not os.getenv("ENDPOINT_NAME"):
        raise HTTPException(status_code=400, detail="ENDPOINT_NAME not configured in .env")
    return JSONResponse(content=await asyncio.to_thread(_stop_online_sync), status_code=202)


def _stop_online_sync():
    from src.infrastructure.aml_online import get_online_manager
    try:
        return get_online_manager().stop()
    except Exception as e:
        logger.error(f"stop failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop: {e}")


# --- Review Endpoints ---

@app.get("/review/items")
async def list_review_items(limit: int = 50, session: AsyncSession = Depends(get_db_session)):
    """List predictions that need human review."""
    repo = PredictionRepository(session)
    items = await repo.get_review_queue(limit)
    return JSONResponse(content=[item.to_dict() for item in items])


@app.post("/review/items/{prediction_id}/decision")
async def submit_review_decision(
    prediction_id: str,
    request: dict,
    session: AsyncSession = Depends(get_db_session),
):
    """Submit a human review decision for a prediction.

    Request body:
    {
        "decision": "accept" | "change_label" | "mark_unknown" | "request_images" | "field_visit",
        "corrected_label": "string (required if decision=change_label)",
        "correction_reason": "string",
        "reviewer_id": "string",
        "reviewer_notes": "string"
    }
    """
    decision = request.get("decision")
    if decision not in ("accept", "change_label", "mark_unknown", "request_images", "field_visit"):
        raise HTTPException(status_code=400, detail=f"Invalid decision: {decision}")

    repo = PredictionRepository(session)
    pred = await repo.get_by_id(prediction_id)
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found")

    review_service.process_decision(
        prediction=pred,
        decision=decision,
        reviewer_id=request.get("reviewer_id", "anonymous"),
        corrected_label=request.get("corrected_label"),
        correction_reason=request.get("correction_reason", ""),
        reviewer_notes=request.get("reviewer_notes", ""),
    )

    pred = await repo.save(pred)
    return JSONResponse(content=pred.to_dict())


@app.get("/review/stats")
async def review_stats(session: AsyncSession = Depends(get_db_session)):
    """Get review queue statistics."""
    stats = await review_service.get_review_stats(session)
    return JSONResponse(content=stats)


@app.get("/review", response_class=HTMLResponse)
async def review_page():
    """Serve the Review workspace UI."""
    return HTMLResponse(content=REVIEW_UI_HTML.replace("{mock_badge}", _mock_badge()))




# --- Batch Job Endpoints ---
@app.post("/batch/upload")
async def upload_batch_dataset(file: UploadFile = File(...)):
    """Upload a ZIP file of building images and register as Azure ML Data Asset.

    Expected ZIP structure:
        building_id_1/
            building_id_01.jpg
            building_id_02.jpg
        building_id_2/
            building_id_01.jpg

    Generates _task.json per building and registers with Azure ML.
    """
    from src.infrastructure.aml_batch import get_aml_batch_client

    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are accepted")

    job_id = f"batch-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    tmp_dir = Path(tempfile.mkdtemp())
    extract_dir = tmp_dir / job_id
    extract_dir.mkdir()
    zip_path = tmp_dir / file.filename
    valid_buildings = []
    total_images = 0

    try:
        with open(zip_path, "wb") as f:
            content = await file.read()
            f.write(content)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        # Validate structure: each immediate subdirectory must have .jpg files
        building_dirs = [d for d in extract_dir.iterdir() if d.is_dir()]
        if not building_dirs:
            raise HTTPException(status_code=400, detail="ZIP must contain building subdirectories")

        total_images = 0
        valid_buildings = []
        for d in sorted(building_dirs):
            images = sorted(d.glob("*.jpg"))
            if not images:
                shutil.rmtree(d)
                continue
            total_images += len(images)
            valid_buildings.append(d.name)

            task = {
                "building_id": d.name,
                "num_images": len(images),
                "image_files": [img.name for img in images],
            }
            with open(d / "_task.json", "w") as f:
                json.dump(task, f)

        if not valid_buildings:
            raise HTTPException(status_code=400, detail="No valid building directories with .jpg images found")

        # Register as Azure ML Data Asset
        aml_client = get_aml_batch_client()
        data_asset_name = f"bogota-batch-{job_id}"
        try:
            storage_path = aml_client.register_data_asset(str(extract_dir), data_asset_name)
        except Exception as e:
            logger.error(f"Data Asset registration failed: {e}")
            storage_path = f"local://{extract_dir}"

        logger.info(f"Uploaded {len(valid_buildings)} buildings, {total_images} images → {data_asset_name}")

        return JSONResponse(content={
            "job_id": job_id,
            "buildings": len(valid_buildings),
            "total_images": total_images,
            "avg_images": round(total_images / len(valid_buildings), 1) if valid_buildings else 0,
            "storage_path": storage_path,
            "data_asset_name": data_asset_name,
            "ready": True,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if zip_path.exists():
            zip_path.unlink()
        # Clean up temp dir on failure
        if shutil.os.path.exists(str(tmp_dir)) and not valid_buildings:
            shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/jobs")
async def create_job(request: dict, session: AsyncSession = Depends(get_db_session)):
    """Create a new Azure ML batch classification job.

    Request body:
    {
        "job_id": "optional",
        "data_asset_path": "azureml://...",
        "data_asset_name": "string",
        "total_buildings": 100
    }
    """
    from src.infrastructure.aml_batch import get_aml_batch_client
    from src.persistence.repository import JobRepository
    from src.persistence.models import Job

    data_asset_path = request.get("data_asset_path")
    total_buildings = request.get("total_buildings", 0)

    if not data_asset_path:
        raise HTTPException(status_code=400, detail="data_asset_path is required")

    job_id = request.get("job_id") or f"aml-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    aml_client = get_aml_batch_client()

    try:
        endpoint_name = aml_client.ensure_endpoint()
        aml_client.ensure_deployment(endpoint_name)
        aml_job_name = aml_client.submit_job(data_asset_path, endpoint_name)
    except Exception as e:
        logger.error(f"AML job submission failed: {e}")
        raise HTTPException(status_code=500, detail=f"Azure ML job submission failed: {e}")

    job = Job(
        id=job_id,
        status="submitted",
        total_buildings=total_buildings or 0,
        created_by=request.get("submitted_by", "api"),
        aml_job_name=aml_job_name,
        data_asset_path=data_asset_path,
    )
    repo = JobRepository(session)
    job = await repo.save(job)

    logger.info(f"AML job created: {job_id} (AML: {aml_job_name})")

    return JSONResponse(content={
        "job_id": job_id,
        "status": "submitted",
        "aml_job_name": aml_job_name,
        "total_buildings": total_buildings or 0,
        "message": f"Azure ML job submitted: {aml_job_name}",
    })


@app.get("/jobs")
async def list_jobs(limit: int = 20, session: AsyncSession = Depends(get_db_session)):
    """List all batch jobs."""
    from src.persistence.repository import JobRepository
    repo = JobRepository(session)
    jobs = await repo.list_jobs(limit)
    return JSONResponse(content=[{
        "job_id": j.id,
        "status": j.status,
        "aml_job_name": j.aml_job_name,
        "total_buildings": j.total_buildings,
        "completed_buildings": j.completed_buildings,
        "failed_buildings": j.failed_buildings,
        "review_count": j.review_count,
        "created_by": j.created_by,
        "created_at": j.created_at.isoformat(),
    } for j in jobs])


@app.get("/jobs/{job_id}")
async def get_job(job_id: str, session: AsyncSession = Depends(get_db_session)):
    """Get job details with predictions summary and AML live progress."""
    from src.persistence.repository import JobRepository, PredictionRepository
    job_repo = JobRepository(session)
    job = await job_repo.get_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    pred_repo = PredictionRepository(session)
    predictions = await pred_repo.get_by_job(job_id)

    label_counts = {}
    for p in predictions:
        lbl = p.primary_label or "PARSE_ERROR"
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    aml_progress = None
    if job.aml_job_name and job.status in ("submitted", "running", "Starting", "Provisioning"):
        try:
            from src.infrastructure.aml_batch import get_aml_batch_client
            aml_client = get_aml_batch_client()
            aml_progress = aml_client.get_job_status(job.aml_job_name)
        except Exception as e:
            logger.warning(f"Could not fetch AML progress for {job.aml_job_name}: {e}")
            aml_progress = {"error": str(e)}

    return JSONResponse(content={
        "job_id": job.id,
        "status": job.status,
        "aml_job_name": job.aml_job_name,
        "total_buildings": job.total_buildings,
        "completed_buildings": job.completed_buildings or 0,
        "failed_buildings": job.failed_buildings or 0,
        "review_count": job.review_count or 0,
        "label_distribution": label_counts,
        "aml_progress": aml_progress,
        "predictions": [p.to_dict() for p in predictions],
        "created_by": job.created_by,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
    })


@app.post("/jobs/{job_id}/stop")
async def stop_job(job_id: str, session: AsyncSession = Depends(get_db_session)):
    """Cancel a running Azure ML batch job."""
    from src.persistence.repository import JobRepository
    job_repo = JobRepository(session)
    job = await job_repo.get_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not job.aml_job_name:
        raise HTTPException(status_code=400, detail="No AML job associated")

    from src.infrastructure.aml_batch import get_aml_batch_client
    aml_client = get_aml_batch_client()
    cancelled = aml_client.cancel_job(job.aml_job_name)

    if cancelled:
        job.status = "cancelled"
        await job_repo.save(job)
        return JSONResponse(content={
            "job_id": job_id,
            "status": "cancelled",
            "message": "Azure ML job cancelled.",
        })
    else:
        return JSONResponse(content={
            "job_id": job_id,
            "status": job.status,
            "message": "Failed to cancel AML job.",
        }, status_code=500)


@app.post("/jobs/{job_id}/import")
async def import_job_results(job_id: str, session: AsyncSession = Depends(get_db_session)):
    """Import predictions from a completed Azure ML batch job.

    Downloads predictions.csv from AML output, parses rows into Prediction
    records, and persists them to SQLite. Idempotent: clears existing
    predictions for this job before re-importing.
    """
    from sqlalchemy.exc import IntegrityError
    from src.infrastructure.aml_batch import get_aml_batch_client
    from src.persistence.repository import JobRepository, PredictionRepository
    from src.persistence.models import Prediction

    job_repo = JobRepository(session)
    job = await job_repo.get_by_id(job_id)
    if not job or not job.aml_job_name:
        raise HTTPException(status_code=404, detail="Job has no AML job associated")

    aml_client = get_aml_batch_client()
    csv_content = aml_client.download_output(job.aml_job_name)

    if csv_content is None:
        raise HTTPException(status_code=400, detail="Job output not available. Job may not be completed.")

    pred_repo = PredictionRepository(session)

    # Idempotency: clear any prior predictions for this job so a re-import
    # does not duplicate rows. The legacy csv.DictReader left 199 ghost rows
    # for aml-20260615-085139 from earlier failed runs.
    deleted = await pred_repo.delete_by_job(job_id)
    if deleted:
        logger.info(f"Cleared {deleted} existing predictions for job {job_id} before re-import")

    # Parse CSV — batch scoring output is whitespace-separated with NO header
    # Columns: building_id label confidence images_used status
    imported = 0
    failed = 0
    review_count = 0
    staged_preds: list[Prediction] = []

    try:
        for raw_line in csv_content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 5:
                logger.warning(f"Skipping malformed CSV line: {line[:80]}")
                failed += 1
                continue

            building_id = parts[0]
            label = parts[1]
            try:
                confidence = float(parts[2])
            except (ValueError, TypeError):
                confidence = 0
            try:
                images_used = int(parts[3])
            except (ValueError, TypeError):
                images_used = 0
            status = parts[4] if len(parts) > 4 else "unknown"

            if status != "success" or label in ("ERROR", "PARSE_ERROR"):
                failed += 1
                pred = Prediction(
                    building_id=building_id,
                    job_id=job_id,
                    model_id="Qwen2.5-VL-32B-Instruct",
                    perfil_modelo="batch-aml",
                    prompt_version="p004-batch",
                    schema_version="1.0.0",
                    taxonomy_version="1.0.0",
                    image_uris=[],
                    classification={"label": label, "confidence": confidence, "status": status},
                    primary_label=None,
                    max_confidence=None,
                    requires_review=True,
                    review_status="pending",
                    latency_ms=0,
                )
            else:
                imported += 1
                from src.application.review_service import ReviewService
                parsed_minimal = {
                    "clases": [{"label": label, "confidence": confidence, "is_primary": True}],
                }
                needs_review, _ = ReviewService.apply_routing_rules(parsed_minimal)
                if needs_review:
                    review_count += 1

                pred = Prediction(
                    building_id=building_id,
                    job_id=job_id,
                    model_id="Qwen2.5-VL-32B-Instruct",
                    perfil_modelo="batch-aml",
                    prompt_version="p004-batch",
                    schema_version="1.0.0",
                    taxonomy_version="1.0.0",
                    image_uris=[],
                    classification={
                        "clases": [{"label": label, "confidence": confidence, "is_primary": True}],
                    },
                    primary_label=label,
                    max_confidence=confidence,
                    requires_review=needs_review,
                    review_status="pending" if needs_review else "accepted",
                    latency_ms=0,
                )

            pred_repo.session.add(pred)
            staged_preds.append(pred)

        # Single commit for the whole batch — atomic, no partial imports.
        await pred_repo.session.commit()

    except IntegrityError as e:
        await pred_repo.session.rollback()
        logger.error(f"Integrity error during import of job {job_id}: {e}")
        raise HTTPException(
            status_code=409,
            detail=f"Duplicate (job_id, building_id) detected. Use /jobs/{job_id}/purge to clear, then retry.",
        )
    except Exception as e:
        await pred_repo.session.rollback()
        logger.error(f"Import failed for job {job_id}: {e}")
        raise

    # Update job stats
    job.completed_buildings = imported
    job.failed_buildings = failed
    job.review_count = review_count
    job.status = "completed"
    await job_repo.save(job)

    logger.info(f"Imported {imported} predictions, {failed} errors, {review_count} in review for job {job_id}")

    return JSONResponse(content={
        "job_id": job_id,
        "imported": imported,
        "failed": failed,
        "review_count": review_count,
        "status": "completed",
    })


@app.post("/jobs/{job_id}/purge")
async def purge_job_predictions(job_id: str, session: AsyncSession = Depends(get_db_session)):
    """Delete all predictions for a job. Does NOT re-import from AML.

    Useful when:
    - Predictions CSV is corrupt and re-import will not help
    - User wants to clear the DB without touching Azure ML
    - Cleaning up ghost rows from a botched prior import

    Resets job.completed_buildings/failed_buildings/review_count to 0
    and sets job.status to "completed" (since AML itself is still done).
    """
    from src.persistence.repository import JobRepository, PredictionRepository

    job_repo = JobRepository(session)
    job = await job_repo.get_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    pred_repo = PredictionRepository(session)
    deleted = await pred_repo.delete_by_job(job_id)

    job.completed_buildings = 0
    job.failed_buildings = 0
    job.review_count = 0
    await job_repo.save(job)

    logger.info(f"Purged {deleted} predictions for job {job_id}")
    return JSONResponse(content={
        "job_id": job_id,
        "deleted": deleted,
        "status": "completed",
        "message": f"Cleared {deleted} predictions. Use /jobs/{job_id}/import to re-import from AML output.",
    })


@app.get("/jobs/{job_id}/export")
async def export_job(
    job_id: str,
    format: str = "json",
    session: AsyncSession = Depends(get_db_session),
):
    """Export job predictions as CSV or JSON."""
    from src.persistence.repository import PredictionRepository, JobRepository
    from fastapi.responses import StreamingResponse
    import io, csv as _csv

    job_repo = JobRepository(session)
    job = await job_repo.get_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    pred_repo = PredictionRepository(session)
    predictions = await pred_repo.get_by_job(job_id)

    if format == "csv":
        output = io.StringIO()
        writer = _csv.writer(output)
        writer.writerow([
            "prediction_id", "building_id", "primary_label", "max_confidence",
            "requires_review", "review_status", "reviewed_by",
            "model_id", "prompt_version", "latency_ms", "created_at",
        ])
        for p in predictions:
            writer.writerow([
                p.id, p.building_id, p.primary_label, p.max_confidence,
                p.requires_review, p.review_status, p.reviewed_by,
                p.model_id, p.prompt_version, p.latency_ms, p.created_at.isoformat(),
            ])
        output.seek(0)
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=job_{job_id}.csv"},
        )
    else:
        return JSONResponse(content=[p.to_dict() for p in predictions])


@app.get("/batch", response_class=HTMLResponse)
async def batch_page():
    """Serve the Batch Processing UI."""
    return HTMLResponse(
        content=BATCH_UI_HTML.replace("{mock_badge}", _mock_badge()),
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


NAV_HTML = """
<nav style="display:flex;align-items:center;gap:0;margin:0 0 12px 0;border-bottom:2px solid #e2e8f0;">
    <a href="/" style="padding:8px 20px;text-decoration:none;color:#475569;font-weight:500;border-radius:6px 6px 0 0;" class="nav-tab">Classify</a>
    <a href="/review" style="padding:8px 20px;text-decoration:none;color:#475569;font-weight:500;border-radius:6px 6px 0 0;" class="nav-tab">Review</a>
    <a href="/batch" style="padding:8px 20px;text-decoration:none;color:#475569;font-weight:500;border-radius:6px 6px 0 0;" class="nav-tab">Batch</a>
    {mock_badge}
    <div id="onlineNavIndicator" style="margin-left:auto;padding:6px 14px;font-size:13px;font-weight:500;border-radius:6px;cursor:pointer;text-decoration:none;display:inline-flex;align-items:center;gap:6px;" onclick="window.location.href='/';" title="Go to Classify to manage deployment">
        <span id="onlineNavDot" style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#94a3b8;"></span>
        <span id="onlineNavLabel">Online: ?</span>
    </div>
</nav>
<script>
(function(){
    var tabs=document.querySelectorAll(".nav-tab");
    var path=window.location.pathname;
    tabs.forEach(function(t){
        if(path===t.getAttribute("href")){
            t.style.color="#2563eb";
            t.style.borderBottom="2px solid #2563eb";
            t.style.marginBottom="-2px";
        }
    });

    var pollMs = 30000;
    var transitionStates = ['creating','deleting'];

    function setIndicator(state, label) {
        var dot = document.getElementById('onlineNavDot');
        var lbl = document.getElementById('onlineNavLabel');
        if (!dot || !lbl) return;
        var color = '#94a3b8';
        if (state === 'running') color = '#16a34a';
        else if (state === 'creating') color = '#eab308';
        else if (state === 'deleting') color = '#f97316';
        else if (state === 'failed') color = '#dc2626';
        else if (state === 'not_found') color = '#dc2626';
        dot.style.background = color;
        lbl.textContent = 'Online: ' + label;
    }

    async function tick() {
        try {
            var r = await fetch('/admin/online/status');
            var d = await r.json();
            var state = d.state || 'not_found';
            var lbl;
            if (state === 'running') lbl = 'Running';
            else if (state === 'creating') lbl = 'Creating...';
            else if (state === 'deleting') lbl = 'Stopping...';
            else if (state === 'not_found') lbl = 'Not deployed';
            else if (state === 'failed') lbl = 'Failed';
            else lbl = state;
            setIndicator(state, lbl);
            pollMs = (transitionStates.indexOf(state) >= 0) ? 3000 : 30000;
        } catch (e) {
            setIndicator('not_found', '?');
        }
        setTimeout(tick, pollMs);
    }
    tick();
})();
</script>
"""

CLASSIFY_UI_HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Classify — Bogota Building Use</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 960px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        .card { background: white; border-radius: 8px; padding: 20px; margin: 16px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        input, button { padding: 8px 12px; margin: 4px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #2563eb; color: white; border: none; cursor: pointer; }
        button:disabled { background: #94a3b8; }
        .label { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
        .blue { background: #dbeafe; color: #1e40af; }
        .amber { background: #fef3c7; color: #92400e; }
        .purple { background: #ede9fe; color: #6b21a8; }
        .green { background: #dcfce7; color: #166534; }
        .review-badge { background: #fef2f2; color: #991b1b; padding: 2px 8px; border-radius: 4px; font-size: 12px; }
        .confidence-bar { height: 8px; background: #e2e8f0; border-radius: 4px; margin: 4px 0; }
        .confidence-fill { height: 100%; border-radius: 4px; background: #2563eb; }
        .evidence { font-size: 14px; color: #475569; margin: 8px 0; }
        .evidence li { margin: 2px 0; }
        #dropzone { border: 2px dashed #94a3b8; border-radius: 8px; padding: 32px; text-align: center; color: #64748b; cursor: pointer; transition: all 0.2s; }
        #dropzone.drag-over { border-color: #2563eb; background: #eff6ff; color: #2563eb; }
        #dropzone.has-images { border-style: solid; border-color: #2563eb; padding: 12px; }
        #images-preview { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0; }
        #images-preview img { max-height: 120px; border-radius: 4px; object-fit: cover; }
        #classifyBtn:disabled { background: #94a3b8; cursor: not-allowed; }
        .traffic-warning { color: #dc2626; font-size: 13px; margin-top: 4px; display: none; }
    </style>
</head>
<body>
    """ + _render_nav() + """
    {use_mock_banner}
    <h1>Classify Building</h1>
    <div style="margin-bottom:-8px;"></div>

    <div class="card" id="onlineControlCard" style="background:#f8fafc; border-left:4px solid #6366f1;">
        <h3 style="margin-top:0;">Online Deployment</h3>
        <div id="onlineStatusBanner" style="padding:12px 14px;border-radius:6px;margin-bottom:12px;font-size:14px;display:flex;align-items:center;gap:10px;">
            <span id="onlineStatusDot" style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#94a3b8;"></span>
            <span id="onlineStatusText" style="flex:1;">Checking deployment state...</span>
        </div>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
            <button id="onlineStartBtn" onclick="onOnlineStart()" style="background:#059669;color:white;border:none;padding:8px 16px;border-radius:4px;cursor:pointer;font-weight:500;display:none;">▶ Start Deployment</button>
            <button id="onlineStopBtn" onclick="onOnlineStopAsk()" style="background:#dc2626;color:white;border:none;padding:8px 16px;border-radius:4px;cursor:pointer;font-weight:500;display:none;">⏸ Stop Deployment</button>
            <button id="onlineRetryBtn" onclick="onOnlineStart()" style="background:#dc2626;color:white;border:none;padding:8px 16px;border-radius:4px;cursor:pointer;font-weight:500;display:none;">🔄 Retry</button>
            <span id="onlineCostNote" style="font-size:12px;color:#64748b;"></span>
        </div>
    </div>

    <div class="card">
        <h3>Input</h3>
        <input id="buildingId" placeholder="Building ID" value="bogota-001" style="width:200px;">
        <div id="dropzone">
            <p>Drop building images here (drag from your computer)</p>
            <p style="font-size:12px;color:#94a3b8;">JPG or PNG — up to 8 images</p>
        </div>
        <div id="images-preview"></div>
        <div class="traffic-warning" id="trafficWarning">Start the online deployment first (see control panel above)</div>
        <button id="classifyBtn" onclick="classify()" disabled>Classify Building</button>
        <span id="status"></span>
    </div>

    <div class="card" id="resultCard" style="display:none;">
        <h3>Result</h3>
        <div id="resultContent"></div>
    </div>

    <script>
        let imageUrls = [];
        let currentTraffic = 0;
        let onlineState = 'not_found';
        let onlineStartTime = null;
        const dropzone = document.getElementById('dropzone');
        const preview = document.getElementById('images-preview');
        const classifyBtn = document.getElementById('classifyBtn');
        const trafficWarning = document.getElementById('trafficWarning');

        // --- Drag and drop ---
        dropzone.addEventListener('dragover', e => {
            e.preventDefault();
            dropzone.classList.add('drag-over');
        });
        dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
        dropzone.addEventListener('drop', e => {
            e.preventDefault();
            dropzone.classList.remove('drag-over');
            const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
            if (imageUrls.length + files.length > 8) { alert('Maximum 8 images'); return; }
            files.forEach(file => {
                const reader = new FileReader();
                reader.onload = ev => {
                    imageUrls.push(ev.target.result);
                    renderImages();
                };
                reader.readAsDataURL(file);
            });
        });

        dropzone.addEventListener('click', () => {
            const input = document.createElement('input');
            input.type = 'file'; input.accept = 'image/*'; input.multiple = true;
            input.onchange = () => {
                const files = Array.from(input.files);
                if (imageUrls.length + files.length > 8) { alert('Maximum 8 images'); return; }
                files.forEach(file => {
                    const reader = new FileReader();
                    reader.onload = ev => {
                        imageUrls.push(ev.target.result);
                        renderImages();
                    };
                    reader.readAsDataURL(file);
                });
            };
            input.click();
        });

        function renderImages() {
            if (imageUrls.length > 0) dropzone.classList.add('has-images');
            else dropzone.classList.remove('has-images');
            dropzone.querySelector('p').textContent = imageUrls.length
                ? `${imageUrls.length} image(s) loaded`
                : 'Drop building images here (drag from your computer)';
            preview.innerHTML = imageUrls.map((url, i) =>
                `<div style="position:relative;">
                    <img src="${url}">
                    <button onclick="imageUrls.splice(${i},1);renderImages()" style="position:absolute;top:4px;right:4px;background:#ef4444;color:white;border:none;font-size:12px;padding:2px 6px;border-radius:4px;cursor:pointer;">x</button>
                </div>`
            ).join('');
            updateClassifyBtn();
        }

        // --- Online deployment control ---
        const transitionStates = ['creating', 'deleting'];
        function formatElapsed() {
            if (!onlineStartTime) return '';
            const s = Math.floor((Date.now() - onlineStartTime) / 1000);
            const m = Math.floor(s / 60);
            const sec = s % 60;
            return m + ':' + String(sec).padStart(2, '0');
        }

        function renderOnlineState(d) {
            onlineState = d.state || 'not_found';
            const banner = document.getElementById('onlineStatusBanner');
            const dot = document.getElementById('onlineStatusDot');
            const txt = document.getElementById('onlineStatusText');
            const startBtn = document.getElementById('onlineStartBtn');
            const stopBtn = document.getElementById('onlineStopBtn');
            const retryBtn = document.getElementById('onlineRetryBtn');
            const costNote = document.getElementById('onlineCostNote');

            startBtn.style.display = 'none';
            stopBtn.style.display = 'none';
            retryBtn.style.display = 'none';

            if (onlineState === 'not_found') {
                dot.style.background = '#dc2626';
                banner.style.background = '#fef2f2';
                txt.innerHTML = '<strong>🔴 No deployment.</strong> Click below to create one. Takes 5-10 minutes (model download + VM provision).';
                startBtn.textContent = '▶ Create Deployment';
                startBtn.style.display = 'inline-block';
                costNote.textContent = 'Cost: $0 when stopped · ~$3.67/hr when running';
            } else if (onlineState === 'creating') {
                dot.style.background = '#eab308';
                banner.style.background = '#fef9c3';
                if (!onlineStartTime) onlineStartTime = Date.now();
                txt.innerHTML = '<strong>🟡 Creating deployment...</strong> Please wait. ⏱ ' + formatElapsed() + ' elapsed';
                costNote.textContent = 'Usually 5-10 minutes';
            } else if (onlineState === 'running') {
                dot.style.background = '#16a34a';
                banner.style.background = '#dcfce7';
                onlineStartTime = null;
                txt.innerHTML = '<strong>🟢 Online deployment running.</strong> Ready to classify.';
                stopBtn.style.display = 'inline-block';
                costNote.textContent = 'Cost: ~$3.67/hr while running · $0 when stopped';
            } else if (onlineState === 'deleting') {
                dot.style.background = '#f97316';
                banner.style.background = '#fff7ed';
                if (!onlineStartTime) onlineStartTime = Date.now();
                txt.innerHTML = '<strong>🟡 Stopping deployment...</strong> Removing compute. ⏱ ' + formatElapsed() + ' elapsed';
                costNote.textContent = 'Usually ~30 seconds';
            } else if (onlineState === 'failed') {
                dot.style.background = '#dc2626';
                banner.style.background = '#fef2f2';
                onlineStartTime = null;
                const errMsg = d.error ? ' — ' + d.error : '';
                txt.innerHTML = '<strong>🔴 Deployment failed</strong>' + errMsg;
                retryBtn.style.display = 'inline-block';
                costNote.textContent = '';
            } else {
                dot.style.background = '#94a3b8';
                banner.style.background = '#f1f5f9';
                txt.innerHTML = 'Unknown state: ' + onlineState;
                costNote.textContent = '';
            }
            updateClassifyBtn();
        }

        async function pollOnlineStatus() {
            try {
                const r = await fetch('/admin/online/status');
                const d = await r.json();
                renderOnlineState(d);
            } catch (e) {
                document.getElementById('onlineStatusText').textContent = 'Error checking deployment state: ' + e.message;
            }
            const inTransition = transitionStates.indexOf(onlineState) >= 0;
            setTimeout(pollOnlineStatus, inTransition ? 3000 : 30000);
            if (inTransition) {
                if (!onlineStartTime) onlineStartTime = Date.now();
                const txt = document.getElementById('onlineStatusText');
                const elapsed = formatElapsed();
                if (onlineState === 'creating') {
                    txt.innerHTML = '<strong>🟡 Creating deployment...</strong> Please wait. ⏱ ' + elapsed + ' elapsed';
                } else if (onlineState === 'deleting') {
                    txt.innerHTML = '<strong>🟡 Stopping deployment...</strong> Removing compute. ⏱ ' + elapsed + ' elapsed';
                }
            }
        }

        async function onOnlineStart() {
            const startBtn = document.getElementById('onlineStartBtn');
            const retryBtn = document.getElementById('onlineRetryBtn');
            startBtn.disabled = true;
            retryBtn.disabled = true;
            onlineStartTime = Date.now();
            try {
                const r = await fetch('/admin/online/start', { method: 'POST' });
                if (!r.ok) {
                    const err = await r.json().catch(() => ({detail: 'Unknown error'}));
                    alert('Failed to start: ' + (err.detail || r.statusText));
                }
            } catch (e) {
                alert('Network error: ' + e.message);
            }
            startBtn.disabled = false;
            retryBtn.disabled = false;
            pollOnlineStatus();
        }

        function onOnlineStopAsk() {
            if (confirm('Stop the online deployment?\\n\\nThe deployment will be DELETED (Azure ML does not allow instance_count=0). The next Start will recreate it from scratch (5-10 minutes).\\n\\nCost: $0/hr while stopped.')) {
                onOnlineStop();
            }
        }

        async function onOnlineStop() {
            const stopBtn = document.getElementById('onlineStopBtn');
            stopBtn.disabled = true;
            onlineStartTime = Date.now();
            try {
                const r = await fetch('/admin/online/stop', { method: 'POST' });
                if (!r.ok) {
                    const err = await r.json().catch(() => ({detail: 'Unknown error'}));
                    alert('Failed to stop: ' + (err.detail || r.statusText));
                }
            } catch (e) {
                alert('Network error: ' + e.message);
            }
            stopBtn.disabled = false;
            pollOnlineStatus();
        }

        // --- Traffic guard ---
        function updateClassifyBtn() {
            const canClassify = onlineState === 'running' && imageUrls.length > 0;
            classifyBtn.disabled = !canClassify;
            trafficWarning.style.display = (onlineState !== 'running' && imageUrls.length > 0) ? 'block' : 'none';
        }

        // --- Classify ---
        async function classify() {
            const buildingId = document.getElementById('buildingId').value.trim();
            if (!buildingId || imageUrls.length === 0) {
                alert('Please enter a Building ID and add at least one image');
                return;
            }
            classifyBtn.disabled = true;
            document.getElementById('status').textContent = 'Classifying...';

            try {
                const resp = await fetch('/classify', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({building_id: buildingId, image_urls: imageUrls}),
                });
                const data = await resp.json();
                if (resp.status === 503 && data.detail && data.detail.message) {
                    document.getElementById('status').textContent = '⚠ ' + data.detail.message;
                    pollOnlineStatus();
                    return;
                }
                renderResult(data);
                document.getElementById('status').textContent = 'Done!';
            } catch (err) {
                document.getElementById('status').textContent = 'Error: ' + err.message;
            }
            updateClassifyBtn();
        }

        function renderResult(data) {
            document.getElementById('resultCard').style.display = 'block';
            const classification = data.classification;
            const clases = classification.clases || [];
            const primary = clases.find(c => c.is_primary) || clases[0];
            const labelColors = {RESIDENCIAL:'blue', COMERCIAL:'amber', MIXTO:'purple', RURAL:'green', DOTACIONAL:'blue', MOLES:'purple'};
            let lc = labelColors[primary?.label?.split('_')[0]] || 'blue';

            let html = `<p><strong>Building:</strong> ${data.building_id}</p>`;
            if (primary) {
                html += `<p><span class="label ${lc}">${primary.label}</span></p>`;
                html += `<p>Confidence: ${(primary.confidence * 100).toFixed(0)}%</p>`;
                html += `<div class="confidence-bar"><div class="confidence-fill" style="width:${primary.confidence * 100}%"></div></div>`;
                html += `<p><strong>Evidence:</strong></p><ul class="evidence">`;
                (primary.evidencia_visual || []).forEach(e => html += `<li>${e}</li>`);
                html += `</ul>`;
            }
            if (data.requires_review) {
                html += `<span class="review-badge">NEEDS REVIEW</span>`;
                html += ` <a href="/review#${data.id}" style="color:#2563eb;font-size:12px;margin-left:8px;">Review Now →</a>`;
            }
            html += `<p><strong>All classes:</strong> ${clases.map(c => `${c.label} (${(c.confidence*100).toFixed(0)}%)`).join(', ')}</p>`;
            html += `<p><strong>Visible features:</strong> ${(classification.caracteristicas_visibles || []).join(', ')}</p>`;
            html += `<p><small>Model: ${data.model_id} | Latency: ${data.latency_ms?.toFixed(0) || '?'}ms</small></p>`;
            if (data.model_id && data.model_id.startsWith && data.model_id.startsWith('mock')) {
                html += '<p style="background:#fef3c7;padding:8px;border-radius:4px;margin-top:8px;font-size:12px;color:#92400e;">'
                      + '⚠️ <strong>Mock result</strong> — simulated, not a real VLM prediction.'
                      + '</p>';
            }
            document.getElementById('resultContent').innerHTML = html;
        }

        pollOnlineStatus();
    </script>
</body>
</html>
"""

REVIEW_UI_HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Review — Bogota Building Use</title>
    <style>
        * { box-sizing: border-box; }
        body, h3, h4, p, ul { margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 960px; margin: 0 auto; padding: 20px 20px 0 20px; background: #f8f9fa; color: #1a1a2e; }
        .layout { display: flex; height: calc(100vh - 72px); }
        .sidebar { width: 200px; background: #1e293b; color: white; padding: 12px; overflow-y: auto; }
        .sidebar h3 { margin-bottom: 12px; font-size: 14px; }
        .main { flex: 1; display: flex; flex-direction: column; }
        .toolbar { background: white; padding: 12px 20px; border-bottom: 1px solid #e2e8f0; display: flex; gap: 12px; align-items: center; }
        .content { flex: 1; display: flex; overflow: hidden; }
        .images-panel { flex: 1; padding: 12px; overflow-y: auto; background: #0f172a; display: flex; flex-direction: column; gap: 8px; align-items: center; }
        .images-panel img { max-width: 100%; max-height: 500px; object-fit: contain; border-radius: 4px; }
        .decision-panel { width: 280px; background: white; padding: 16px; overflow-y: auto; border-left: 1px solid #e2e8f0; }
        .prediction-card { background: #f1f5f9; padding: 12px; border-radius: 6px; margin: 8px 0; }
        .label-tag { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 600; margin: 2px; }
        .label-residencial { background: #dbeafe; color: #1e40af; }
        .label-comercial { background: #fef3c7; color: #92400e; }
        .label-dotacional { background: #dcfce7; color: #166534; }
        .label-mixed { background: #ede9fe; color: #6b21a8; }
        .label-rural { background: #f0fdf4; color: #14532d; }
        .label-moles { background: #fef2f2; color: #991b1b; }
        .label-unknown { background: #f1f5f9; color: #475569; }
        .confidence-bar { height: 6px; background: #e2e8f0; border-radius: 3px; margin: 4px 0; }
        .confidence-fill { height: 100%; border-radius: 3px; }
        .evidence-list { font-size: 11px; color: #475569; margin: 4px 0; list-style: inside; }
        button { padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-size: 13px; margin: 4px 0; }
        .btn-accept { background: #16a34a; color: white; width: 100%; }
        .btn-change { background: #ea580c; color: white; width: 100%; }
        .btn-unknown { background: #94a3b8; color: white; width: 100%; }
        .btn-field { background: #dc2626; color: white; width: 100%; }
        .queue-item { padding: 8px; border-bottom: 1px solid #334155; cursor: pointer; font-size: 12px; }
        .queue-item:hover { background: #334155; }
        .queue-item.active { background: #2563eb; }
        select, textarea { width: 100%; padding: 8px; margin: 4px 0; border: 1px solid #ddd; border-radius: 4px; font-family: inherit; }
        .review-badge { background: #fef2f2; color: #991b1b; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }
        .nav-link { color: #93c5fd; font-size: 12px; margin-top: 16px; display: block; }
    </style>
</head>
<body>
    """ + NAV_HTML + """
    <div class="layout">
        <div class="sidebar" id="queue">
            <h3>Review Queue</h3>
            <div id="queueList">Loading...</div>
        </div>
        <div class="main">
            <div class="toolbar">
                <span id="itemCounter">No item selected</span>
                <span style="flex:1"></span>
                <span id="reviewStats" style="font-size:12px;color:#64748b;"></span>
            </div>
            <div class="content">
                <div class="images-panel" id="imagePanel">
                    <p style="color:#94a3b8;">Select a building from the queue to review</p>
                </div>
                <div class="decision-panel" id="decisionPanel">
                    <p style="color:#94a3b8;">No item selected</p>
                </div>
            </div>
        </div>
    </div>
    <script>
        let queue = [];
        let currentIndex = -1;

        async function loadQueue() {
            const resp = await fetch('/review/items?limit=50');
            queue = await resp.json();
            document.getElementById('queueList').innerHTML = queue.map((item, i) => {
                const p = item.classification || {};
                const primary = (p.clases || []).find(c => c.is_primary) || (p.clases || [])[0];
                const label = primary ? primary.label : '?';
                let lblClass = 'label-unknown';
                if (label.startsWith('RESIDENCIAL')) lblClass = 'label-residencial';
                if (label.startsWith('COMERCIAL')) lblClass = 'label-comercial';
                if (label.startsWith('DOTACIONAL')) lblClass = 'label-dotacional';
                if (label.startsWith('MIXTO')) lblClass = 'label-mixed';
                if (label.startsWith('RURAL')) lblClass = 'label-rural';
                if (label.startsWith('MOLES')) lblClass = 'label-moles';
                return `<div class="queue-item" onclick="selectItem(${i})" id="q-${i}">
                    <strong>${item.building_id}</strong><br>
                    <span class="label-tag ${lblClass}">${label}</span>
                    ${item.requires_review ? '<span class="review-badge">REVIEW</span>' : ''}
                </div>`;
            }).join('');
            document.getElementById('itemCounter').textContent = queue.length ? `${queue.length} items in queue` : 'Queue empty';
            loadStats();
        }

        async function loadStats() {
            const resp = await fetch('/review/stats');
            const stats = await resp.json();
            document.getElementById('reviewStats').textContent =
                `Accepted: ${stats.total_accepted} | Corrected: ${stats.total_corrected} | Override: ${(stats.override_rate*100).toFixed(0)}%`;
        }

        function selectItem(i) {
            currentIndex = i;
            document.querySelectorAll('.queue-item').forEach(el => el.classList.remove('active'));
            const el = document.getElementById(`q-${i}`);
            if (el) el.classList.add('active');
            renderItem(queue[i]);
        }

        function renderItem(item) {
            const p = item.classification || {};
            const clases = p.clases || [];
            const primary = clases.find(c => c.is_primary) || clases[0];

            const imagePanel = document.getElementById('imagePanel');
            const uris = item.image_uris || [];
            imagePanel.innerHTML = uris.map(url =>
                `<img src="${url}" onerror="this.alt='Image unavailable'" alt="Building view" loading="lazy">`
            ).join('') || '<p style="color:#94a3b8;">No images available</p>';

            let html = `<h3>${item.building_id}</h3>`;

            if (primary) {
                let lblClass = 'label-unknown';
                const label = primary.label;
                if (label.startsWith('RESIDENCIAL')) lblClass = 'label-residencial';
                if (label.startsWith('COMERCIAL')) lblClass = 'label-comercial';
                if (label.startsWith('DOTACIONAL')) lblClass = 'label-dotacional';
                if (label.startsWith('MIXTO')) lblClass = 'label-mixed';
                if (label.startsWith('RURAL')) lblClass = 'label-rural';
                if (label.startsWith('MOLES')) lblClass = 'label-moles';

                html += `<div class="prediction-card">
                    <strong>Model Prediction</strong><br>
                    <span class="label-tag ${lblClass}">${primary.label}</span>
                    <div class="confidence-bar"><div class="confidence-fill" style="width:${primary.confidence*100}%;background:#2563eb;"></div></div>
                    <span style="font-size:12px;color:#64748b;">Confidence: ${(primary.confidence*100).toFixed(0)}%</span>
                    <br><strong>Evidence:</strong>
                    <ul class="evidence-list">${(primary.evidencia_visual||[]).map(e => `<li>${e}</li>`).join('')}</ul>
                </div>`;

                if (clases.length > 1) {
                    html += `<p><strong>Other classes:</strong></p>`;
                    clases.filter(c => !c.is_primary).forEach(c => {
                        html += `<span class="label-tag label-unknown">${c.label} (${(c.confidence*100).toFixed(0)}%)</span> `;
                    });
                }
            }

            html += `<p style="font-size:13px;color:#475569;"><strong>Visible features:</strong> ${(p.caracteristicas_visibles||[]).join(', ')}</p>`;

            html += `<hr style="margin:16px 0;">
                <h4>Your Decision</h4>
                <div style="display:flex;flex-direction:column;gap:4px;">
                    <button class="btn-accept" onclick="submitDecision('accept')">Accept Prediction</button>
                    <button class="btn-change" onclick="showChangeForm()">Change Label</button>
                    <div id="changeForm" style="display:none;margin:4px 0;">
                        <select id="newLabel">
                            <option value="">Select correct label...</option>
                            <option value="RESIDENCIAL_1">RESIDENCIAL_1 — Residential</option>
                            <option value="COMERCIAL_1">COMERCIAL_1 — Retail</option>
                            <option value="COMERCIAL_2">COMERCIAL_2 — Office</option>
                            <option value="COMERCIAL_3">COMERCIAL_3 — Services</option>
                            <option value="DOTACIONAL_1">DOTACIONAL_1 — Community</option>
                            <option value="DOTACIONAL_2">DOTACIONAL_2 — Education</option>
                            <option value="MOLES_1">MOLES_1 — Construction</option>
                            <option value="RURAL_1">RURAL_1 — Rural</option>
                            <option value="MIXTO_1">MIXTO_1 — Residential+Retail</option>
                            <option value="MIXTO_2">MIXTO_2 — Residential+Office</option>
                            <option value="UNKNOWN_OR_INSUFFICIENT_EVIDENCE">UNKNOWN — Insufficient evidence</option>
                        </select>
                        <textarea id="correctionReason" placeholder="Why are you changing the label? (required)" rows="2"></textarea>
                        <button class="btn-change" onclick="submitDecision('change_label')">Confirm Change</button>
                    </div>
                    <button class="btn-unknown" onclick="submitDecision('mark_unknown')">Mark Unknown</button>
                    <button class="btn-field" onclick="submitDecision('field_visit')">Require Field Visit</button>
                </div>
                <textarea id="reviewerNotes" placeholder="Reviewer notes (optional)" rows="2" style="margin:8px 0;width:100%;padding:8px;border:1px solid #ddd;border-radius:4px;font-family:inherit;"></textarea>
                <p style="font-size:11px;color:#94a3b8;margin-top:4px;">Reviewer: <input id="reviewerId" value="reviewer-1" style="width:120px;padding:4px;border:1px solid #ddd;border-radius:4px;"></p>`;
            html += `<p style="font-size:11px;color:#94a3b8;margin-top:8px;">Model: ${item.model_id} | ${item.perfil_modelo} | ver: ${item.prompt_version} | ${item.latency_ms?.toFixed(0) || '?'}ms</p>`;

            document.getElementById('decisionPanel').innerHTML = html;
        }

        function showChangeForm() {
            document.getElementById('changeForm').style.display = 'block';
        }

        async function submitDecision(decision) {
            if (currentIndex < 0) return;
            const item = queue[currentIndex];
            const reviewerId = document.getElementById('reviewerId').value || 'anonymous';
            const notes = document.getElementById('reviewerNotes').value;
            const reason = document.getElementById('correctionReason')?.value || '';

            const body = {
                decision: decision,
                reviewer_id: reviewerId,
                reviewer_notes: notes,
                correction_reason: reason || `Decision: ${decision}`,
            };

            if (decision === 'change_label') {
                const newLabel = document.getElementById('newLabel').value;
                if (!newLabel) { alert('Please select a label'); return; }
                if (!reason) { alert('Please provide a reason for changing the label'); return; }
                body.corrected_label = newLabel;
            }

            const resp = await fetch(`/review/items/${item.id}/decision`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(body),
            });
            if (resp.ok) {
                queue.splice(currentIndex, 1);
                currentIndex = -1;
                document.getElementById('imagePanel').innerHTML = '<p style="color:#94a3b8;">Decision submitted!</p>';
                document.getElementById('decisionPanel').innerHTML = '<p style="color:#94a3b8;">Select another item</p>';
                loadQueue();
            } else {
                const err = await resp.json();
                alert('Error: ' + (err.detail || 'Failed to submit'));
            }
        }

        loadQueue().then(() => {
            const hashId = window.location.hash.slice(1);
            if (hashId) {
                const idx = queue.findIndex(item => item.id === hashId);
                if (idx >= 0) selectItem(idx);
            }
        });
    </script>
</body>
</html>
"""

BATCH_UI_HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Batch — Bogota Building Use</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 960px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        .card { background: white; border-radius: 8px; padding: 20px; margin: 16px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        button { padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }
        .btn-primary { background: #2563eb; color: white; }
        .btn-primary:disabled { background: #94a3b8; cursor: not-allowed; }
        .btn-danger { background: #dc2626; color: white; }
        .btn-export { background: #16a34a; color: white; font-size: 12px; padding: 4px 10px; margin: 2px; }
        .btn-aml { background: #7c3aed; color: white; }
        .btn-aml:disabled { background: #c4b5fd; cursor: not-allowed; }
        .job-card { background: white; border-radius: 8px; padding: 16px; margin: 12px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .job-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.12); }
        .progress-bar { height: 12px; background: #e2e8f0; border-radius: 6px; margin: 8px 0; overflow: hidden; }
        .progress-fill { height: 100%; border-radius: 6px; background: #2563eb; transition: width 0.5s; }
        .progress-fill.done { background: #16a34a; }
        .progress-fill.cancelled { background: #94a3b8; }
        .status-badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }
        .status-running { background: #dbeafe; color: #1e40af; }
        .status-completed { background: #dcfce7; color: #166534; }
        .status-cancelled { background: #f1f5f9; color: #64748b; }
        .status-failed { background: #fef2f2; color: #991b1b; }
        .label-tag { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; margin: 2px; }
        .label-residencial { background: #dbeafe; color: #1e40af; }
        .label-comercial { background: #fef3c7; color: #92400e; }
        .label-dotacional { background: #dcfce7; color: #166534; }
        .label-mixto { background: #ede9fe; color: #6b21a8; }
        .label-moles { background: #fef2f2; color: #991b1b; }
        .label-rural { background: #f0fdf4; color: #14532d; }
        input { padding: 8px 12px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; }
        #dropzone { border: 2px dashed #94a3b8; border-radius: 8px; padding: 24px; text-align: center; color: #64748b; cursor: pointer; transition: all 0.2s; }
        #dropzone.drag-over { border-color: #7c3aed; background: #f5f3ff; color: #7c3aed; }
        #dropzone.selected { border-color: #7c3aed; background: #f5f3ff; }
        #uploadStatus { margin-top: 8px; font-size: 13px; }
    </style>
</head>
<body>
    <noscript><div style="background:red;color:white;padding:20px;font-size:20px;font-weight:bold;">JAVASCRIPT IS DISABLED IN YOUR BROWSER</div></noscript>
    """ + NAV_HTML + """
    <h1>Batch Processing <span style="font-size:11px;color:#94a3b8;font-weight:normal;">v3-diagnostics</span></h1>

    <div class="card">
        <h3>Upload Dataset (ZIP)</h3>
        <div id="dropzone" style="border:2px dashed #94a3b8;border-radius:8px;padding:32px 20px;text-align:center;cursor:pointer;transition:all 0.2s;">
            <p style="margin:0 0 6px;">Drop a ZIP file here, or click to browse</p>
            <p style="font-size:12px;color:#94a3b8;margin:0;">building_id/01.jpg, building_id/02.jpg, ...</p>
        </div>
        <div id="dropStatus" style="font-size:12px;color:#94a3b8;margin-top:6px;">Initializing dropzone…</div>
        <div id="fileSelected" style="font-size:13px;color:#475569;margin-top:6px;"></div>
        <div id="fileError" style="font-size:13px;color:#dc2626;margin-top:6px;"></div>
        <div style="margin-top:8px;display:flex;gap:8px;align-items:center;">
            <button class="btn-aml" id="uploadBtn" onclick="uploadZip()" disabled>Upload to Azure ML</button>
            <span id="uploadStatus" style="font-size:13px;"></span>
        </div>
    </div>

    <div class="card">
        <h3>Create Job</h3>
        <button class="btn-aml" id="createAmlBtn" onclick="createAmlJob()" disabled>Submit to Azure ML</button>
        <span id="createStatus" style="margin-left:12px;color:#64748b;"></span>
    </div>

    <div id="jobsList">
        <p style="color:#64748b;">Loading jobs...</p>
    </div>

    <script>
        // Show any JS error in the visible status line
        window.addEventListener('error', function(ev) {
            var st = document.getElementById('dropStatus');
            if (st) st.innerHTML = '<span style="color:#dc2626;">JS Error: ' + (ev.message || ev.error || 'unknown') + '</span>';
            console.error('[Batch] window error:', ev);
        });

        // Mark script started
        (function(){
            var st = document.getElementById('dropStatus');
            if (st) st.textContent = 'Script started, looking for dropzone…';
        })();

        var JOBS = [];
        var POLL_INTERVAL = null;
        var selectedFile = null;
        var uploadResult = null;

        // ── ZIP Upload ────────────────────────────────────────
        function selectFile(file) {
            if (!file) {
                document.getElementById('fileError').textContent = 'No file provided.';
                return;
            }
            selectedFile = file;
            var mb = (file.size / 1024 / 1024).toFixed(1);
            document.getElementById('fileSelected').textContent = 'Selected: ' + file.name + ' (' + mb + ' MB)';
            document.getElementById('fileError').textContent = '';
            document.getElementById('uploadBtn').disabled = false;
            var dz = document.getElementById('dropzone');
            dz.style.borderColor = '#7c3aed';
            dz.style.background = '#f5f3ff';
            var st = document.getElementById('dropStatus');
            if (st) st.textContent = 'File ready: ' + file.name + ' (' + mb + ' MB) — click Upload.';
            console.log('[Batch] File selected:', file.name, '(' + mb + ' MB)');
        }

        function clearSelection() {
            selectedFile = null;
            document.getElementById('fileSelected').textContent = '';
            document.getElementById('fileError').textContent = '';
            document.getElementById('uploadBtn').disabled = true;
            var dz = document.getElementById('dropzone');
            dz.style.borderColor = '#94a3b8';
            dz.style.background = '';
        }

        // ── Dropzone wiring (mirrors working Classify page) ────
        var dz = document.getElementById('dropzone');
        var dropStatus = document.getElementById('dropStatus');
        console.log('[Batch] dz element:', dz);

        if (!dz) {
            if (dropStatus) dropStatus.textContent = 'ERROR: #dropzone not found in DOM';
            console.error('[Batch] #dropzone not found');
        } else {
            if (dropStatus) dropStatus.textContent = 'Dropzone ready — drag a ZIP or click to browse.';
        }

        dz.addEventListener('dragover', function(e) {
            e.preventDefault();
            dz.style.borderColor = '#7c3aed';
            dz.style.background = '#f5f3ff';
            if (dropStatus) dropStatus.textContent = 'Release to upload ZIP…';
            document.getElementById('fileError').textContent = '';
        });

        dz.addEventListener('dragleave', function() {
            dz.style.borderColor = '#94a3b8';
            dz.style.background = '';
            if (dropStatus) dropStatus.textContent = 'Dropzone ready — drag a ZIP or click to browse.';
        });

        dz.addEventListener('drop', function(e) {
            e.preventDefault();
            dz.style.borderColor = '#94a3b8';
            dz.style.background = '';
            var files = e.dataTransfer && e.dataTransfer.files;
            console.log('[Batch] drop fired, files:', files ? files.length : 'null');
            if (dropStatus) dropStatus.textContent = 'Drop event received, ' + (files ? files.length : 0) + ' file(s).';
            if (files && files.length > 0) {
                selectFile(files[0]);
            } else {
                document.getElementById('fileError').textContent = 'No file found in drop event.';
            }
        });

        dz.addEventListener('click', function() {
            console.log('[Batch] click, creating dynamic input');
            if (dropStatus) dropStatus.textContent = 'Opening file dialog…';
            var input = document.createElement('input');
            input.type = 'file';
            input.accept = '.zip';
            input.onchange = function() {
                console.log('[Batch] dynamic change, files:', input.files.length);
                if (input.files.length > 0) selectFile(input.files[0]);
            };
            input.click();
        });

        async function uploadZip() {
            if (!selectedFile) return;
            document.getElementById('uploadBtn').disabled = true;
            document.getElementById('uploadStatus').textContent = 'Uploading...';

            var formData = new FormData();
            formData.append('file', selectedFile);

            try {
                var r = await fetch('/batch/upload', { method: 'POST', body: formData });
                if (!r.ok) {
                    var errText = await r.text();
                    throw new Error(errText || ('HTTP ' + r.status));
                }
                uploadResult = await r.json();
                document.getElementById('uploadStatus').innerHTML =
                    '<span style="color:#16a34a;">Uploaded: ' + uploadResult.buildings + ' buildings, ' + uploadResult.total_images + ' images</span>';
                document.getElementById('createAmlBtn').disabled = false;
            } catch(e) {
                document.getElementById('uploadStatus').innerHTML =
                    '<span style="color:#dc2626;">Upload failed: ' + e.message + '</span>';
                document.getElementById('uploadBtn').disabled = false;
            }
        }

        // ── Job Creation ───────────────────────────────────────
        async function createAmlJob() {
            if (!uploadResult || !uploadResult.ready) {
                alert('Upload a dataset first');
                return;
            }
            document.getElementById('createAmlBtn').disabled = true;
            document.getElementById('createStatus').innerHTML =
                'Submitting to Azure ML… <span style="color:#94a3b8;font-size:12px;">(first run may take 5-20 min for endpoint provisioning)</span>';

            var body = {
                data_asset_path: uploadResult.storage_path,
                total_buildings: uploadResult.buildings,
            };

            console.log('[Batch] POST /jobs body:', body);

            try {
                var r = await fetch('/jobs', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(body),
                });
                console.log('[Batch] POST /jobs response status:', r.status);
                if (!r.ok) {
                    var errText = await r.text();
                    throw new Error('HTTP ' + r.status + ': ' + errText);
                }
                var d = await r.json();
                document.getElementById('createStatus').innerHTML =
                    '<span style="color:#16a34a;">Job ' + d.job_id + ' submitted! AML: ' + (d.aml_job_name || '?') + '</span>';
                uploadResult = null;
                clearSelection();
                document.getElementById('createAmlBtn').disabled = true;
                loadJobs();
                startPolling();
            } catch(e) {
                console.error('[Batch] createAmlJob error:', e);
                document.getElementById('createStatus').innerHTML =
                    '<span style="color:#dc2626;">Error: ' + e.message + '</span>';
                document.getElementById('createAmlBtn').disabled = false;
            }
        }

        // ── Job List ───────────────────────────────────────────
        async function loadJobs() {
            try {
                var r = await fetch('/jobs?limit=20');
                JOBS = await r.json();
                renderJobs();
            } catch(e) {}
        }

        function getLabelClass(label) {
            if (!label) return 'label-comercial';
            if (label.startsWith('RESIDENCIAL')) return 'label-residencial';
            if (label.startsWith('COMERCIAL')) return 'label-comercial';
            if (label.startsWith('DOTACIONAL')) return 'label-dotacional';
            if (label.startsWith('MIXTO')) return 'label-mixto';
            if (label.startsWith('MOLES')) return 'label-moles';
            if (label.startsWith('RURAL')) return 'label-rural';
            return 'label-comercial';
        }

        async function loadJobDetail(jobId) {
            try {
                var r = await fetch('/jobs/' + jobId);
                return await r.json();
            } catch(e) { return null; }
        }

        async function renderJobs() {
            var el = document.getElementById('jobsList');
            if (JOBS.length === 0) {
                el.innerHTML = '<p style="color:#64748b;">No jobs yet. Upload a dataset or create one from pre-loaded packages.</p>';
                return;
            }

            var detailPromises = JOBS.map(function(j) { return loadJobDetail(j.job_id); });
            var details = await Promise.all(detailPromises);

            var html = '';
            for (var i = 0; i < JOBS.length; i++) {
                var j = JOBS[i];
                var d = details[i] || {};
                var pct = j.total_buildings > 0 ? ((j.completed_buildings || 0) / j.total_buildings * 100).toFixed(0) : 0;
                var running = j.status === 'running' || j.status === 'pending' || j.status === 'submitted' || j.status === 'Starting' || j.status === 'Provisioning';

                var statusClass = 'status-' + (j.status === 'cancelled' ? 'cancelled' : j.status === 'completed' || j.status === 'Completed' ? 'completed' : j.status === 'failed' || j.status === 'Failed' ? 'failed' : 'running');
                var fillClass = j.status === 'completed' || j.status === 'Completed' ? 'done' : j.status === 'cancelled' ? 'cancelled' : '';

                html += '<div class="job-card">';
                html += '<strong>' + j.job_id + '</strong> ';
                html += '<span class="status-badge ' + statusClass + '">' + (j.status || '?') + '</span>';
                html += '<span style="font-size:11px;color:#7c3aed;margin-left:6px;">Azure ML</span>';

                var aml = d.aml_progress;
                if (aml) {
                    var amlPct = aml.total_tasks > 0 ? ((aml.completed_tasks || 0) / aml.total_tasks * 100).toFixed(0) : 0;
                    html += '<div style="font-size:12px;color:#7c3aed;margin-top:4px;">☁️ Azure ML: <strong>' + (aml.status || '?') + '</strong>';
                    if (aml.total_tasks) html += ' — ' + (aml.completed_tasks || 0) + '/' + aml.total_tasks + ' tasks';
                    if (aml.failed_tasks) html += ' (' + aml.failed_tasks + ' failed)';
                    html += '</div>';
                    if (aml.total_tasks) {
                        html += '<div class="progress-bar" style="background:#ede9fe;"><div class="progress-fill" style="width:' + amlPct + '%;background:#7c3aed;"></div></div>';
                    }
                    if (aml.studio_url) html += '<a href="' + aml.studio_url + '" target="_blank" style="font-size:11px;color:#7c3aed;">View in Azure ML Studio ↗</a>';
                }

                html += '<div class="progress-bar"><div class="progress-fill ' + fillClass + '" style="width:' + pct + '%"></div></div>';
                html += '<span style="font-size:13px;color:#64748b;">Imported: ' + (j.completed_buildings || 0) + '/' + j.total_buildings + ' | ' + (j.failed_buildings || 0) + ' failed | ' + (j.review_count || 0) + ' in review</span>';

                if (d.label_distribution && Object.keys(d.label_distribution).length > 0) {
                    html += '<div style="margin-top:6px;">';
                    var dist = d.label_distribution;
                    Object.keys(dist).sort().forEach(function(lbl) {
                        html += '<span class="label-tag ' + getLabelClass(lbl) + '">' + lbl + ': ' + dist[lbl] + '</span>';
                    });
                    html += '</div>';
                }

                html += '<div style="margin-top:8px;">';
                var amlStatus = aml ? aml.status : null;
                var amlDone = amlStatus === 'Completed' || amlStatus === 'Failed' || amlStatus === 'Cancelled';
                var localDone = j.status === 'completed' || j.status === 'Completed' || j.status === 'cancelled';
                var hasData = (j.completed_buildings || 0) > 0;

                if (running && !amlDone) {
                    html += '<button class="btn-danger" onclick="stopJob(\\\'' + j.job_id + '\\\')" style="font-size:12px;padding:4px 10px;">Stop</button> ';
                }
                if (hasData) {
                    html += '<button class="btn-export" onclick="exportJob(\\\'' + j.job_id + '\\\',\\\'csv\\\')">CSV</button>';
                    html += '<button class="btn-export" onclick="exportJob(\\\'' + j.job_id + '\\\',\\\'json\\\')">JSON</button>';
                }
                if (amlDone && !hasData) {
                    html += '<button onclick="importResults(\\\'' + j.job_id + '\\\')" style="background:#16a34a;color:white;border:none;border-radius:4px;padding:6px 14px;cursor:pointer;font-size:13px;font-weight:600;">⬇ Import Results from Azure ML</button>';
                }
                if (amlStatus === 'Failed' || amlStatus === 'Cancelled') {
                    html += '<span style="color:#dc2626;font-size:12px;margin-left:8px;">Azure ML: ' + amlStatus + '</span>';
                }
                html += '</div></div>';
            }
            el.innerHTML = html;

            var anyRunning = JOBS.some(function(j) { return j.status === 'running' || j.status === 'pending' || j.status === 'submitted' || j.status === 'Starting' || j.status === 'Provisioning'; });
            var anyAmlRunning = details.some(function(d) {
                var a = d && d.aml_progress;
                return a && a.status && a.status !== 'Completed' && a.status !== 'Failed' && a.status !== 'Cancelled';
            });
            if (anyRunning || anyAmlRunning) startPolling();
            else stopPolling();
        }

        async function stopJob(jobId) {
            try {
                var r = await fetch('/jobs/' + jobId + '/stop', {method: 'POST'});
                var d = await r.json();
                loadJobs();
            } catch(e) { alert('Error stopping job: ' + e.message); }
        }

        async function importResults(jobId) {
            try {
                var r = await fetch('/jobs/' + jobId + '/import', {method: 'POST'});
                var d = await r.json();
                if (d.imported > 0) {
                    alert('Imported ' + d.imported + ' predictions from Azure ML output.');
                }
                loadJobs();
            } catch(e) { alert('Error importing: ' + e.message); }
        }

        function exportJob(jobId, format) {
            window.open('/jobs/' + jobId + '/export?format=' + format, '_blank');
        }

        function startPolling() {
            if (POLL_INTERVAL) return;
            POLL_INTERVAL = setInterval(loadJobs, 3000);
        }

        function stopPolling() {
            if (POLL_INTERVAL) { clearInterval(POLL_INTERVAL); POLL_INTERVAL = null; }
        }

        loadJobs().then(function() {
            var anyRunning = JOBS.some(function(j) { return j.status === 'running' || j.status === 'pending' || j.status === 'submitted'; });
            if (anyRunning) startPolling();
        });
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=7860, reload=False)
