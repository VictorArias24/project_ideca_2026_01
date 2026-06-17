"""
Azure ML Batch Endpoint client wrapper.

Wraps Azure ML SDK calls for batch endpoint lifecycle:
data asset registration, endpoint/deployment creation, job submission,
status polling, cancellation, and output download.
"""

import logging
import os
import time
from pathlib import Path

from azure.ai.ml import MLClient, Input
from azure.ai.ml.entities import (
    BatchEndpoint,
    ModelBatchDeployment,
    ModelBatchDeploymentSettings,
    Model,
    AmlCompute,
    CodeConfiguration,
    Environment,
    BatchRetrySettings,
)
from azure.ai.ml.constants import AssetTypes, BatchDeploymentOutputAction
from azure.identity import DefaultAzureCredential

logger = logging.getLogger(__name__)

BATCH_SCORING_DIR = Path(__file__).resolve().parents[2] / "batch_scoring"
CODE_DIR = BATCH_SCORING_DIR / "code"
CONDA_FILE = BATCH_SCORING_DIR / "environment" / "conda.yaml"

DEFAULT_ENDPOINT_NAME = "bogota-batch"
DEFAULT_DEPLOYMENT_NAME = "qwen-vl-batch"
DEFAULT_COMPUTE = "gpu-a100-cluster"
DEFAULT_MODEL_NAME = "bogota-vlm-batch-router"
DEFAULT_ENV_NAME = "batch-vlm-env"


class AmlBatchClient:
    """Client for Azure ML Batch Endpoint operations."""

    def __init__(self):
        credential = DefaultAzureCredential()
        self._client = MLClient(
            credential=credential,
            subscription_id=os.getenv("SUBSCRIPTION_ID"),
            resource_group_name=os.getenv("RESOURCE_GROUP"),
            workspace_name=os.getenv("WORKSPACE_NAME"),
        )
        self._endpoint_name = None
        self._deployment_name = None
        self._online_endpoint = os.getenv("ENDPOINT_NAME", "")
        self._online_deployment = os.getenv("DEPLOYMENT_NAME", "")
        self._online_api_url = os.getenv("API_URL", "")
        self._online_api_key = os.getenv("API_KEY", "")

    # ── Data Asset ──────────────────────────────────────────────────

    def register_data_asset(self, local_path: str, name: str) -> str:
        """Register a local folder as an Azure ML Data Asset.

        Returns the azureml:// URI path to the data.
        """
        from azure.ai.ml.entities import Data

        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"Data path not found: {local_path}")

        logger.info(f"Registering data asset '{name}' from {local_path}")

        data_asset = self._client.data.create_or_update(
            Data(
                name=name,
                path=str(path),
                type=AssetTypes.URI_FOLDER,
                description=f"Batch input: {name}",
            )
        )

        logger.info(f"Data asset registered: {data_asset.path}")
        return data_asset.path

    # ── Endpoint ────────────────────────────────────────────────────

    def ensure_endpoint(
        self,
        endpoint_name: str | None = None,
    ) -> str:
        """Create or get the batch endpoint. Returns endpoint name."""
        if endpoint_name is None:
            endpoint_name = DEFAULT_ENDPOINT_NAME

        try:
            ep = self._client.batch_endpoints.get(endpoint_name)
            logger.info(f"Reusing existing batch endpoint: {ep.name}")
        except Exception:
            logger.info(f"Creating batch endpoint: {endpoint_name}")
            ep = BatchEndpoint(
                name=endpoint_name,
                description="Bogota land-use VLM — batch endpoint",
            )
            self._client.batch_endpoints.begin_create_or_update(ep).wait(timeout=600)
            logger.info(f"Batch endpoint created: {endpoint_name}")

        self._endpoint_name = endpoint_name
        self._ensure_model()
        return endpoint_name

    # ── Model ──────────────────────────────────────────────────────

    def _ensure_model(self) -> None:
        """Register the placeholder model if it doesn't exist."""
        import json as _json
        try:
            self._client.models.get(DEFAULT_MODEL_NAME, version="1")
            logger.info(f"Model {DEFAULT_MODEL_NAME}:1 already registered")
            return
        except Exception:
            pass

        model_dir = BATCH_SCORING_DIR / "model"
        model_dir.mkdir(exist_ok=True)
        model_info = model_dir / "model_info.json"
        if not model_info.exists():
            model_info.write_text(_json.dumps({
                "model_type": "vlm-batch-router",
                "description": "Routes batch requests to online VLM endpoint",
            }, indent=2))

        model = Model(
            name=DEFAULT_MODEL_NAME,
            path=str(model_dir),
            type=AssetTypes.CUSTOM_MODEL,
            description="Batch router to online VLM endpoint",
        )
        self._client.models.create_or_update(model)
        logger.info(f"Model registered: {DEFAULT_MODEL_NAME}:1")

    # ── Deployment ──────────────────────────────────────────────────

    def ensure_deployment(
        self,
        endpoint_name: str | None = None,
        deployment_name: str | None = None,
    ) -> str:
        """Create or update a batch deployment. Returns deployment name."""
        if endpoint_name is None:
            endpoint_name = self._endpoint_name
        if endpoint_name is None:
            raise ValueError("No endpoint name provided or cached")

        if deployment_name is None:
            deployment_name = DEFAULT_DEPLOYMENT_NAME

        # Ensure environment is registered
        env = Environment(
            name=DEFAULT_ENV_NAME,
            conda_file=str(CONDA_FILE),
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
        )
        try:
            self._client.environments.create_or_update(env)
        except Exception:
            pass  # Already exists or will use inline

        deployment = ModelBatchDeployment(
            name=deployment_name,
            endpoint_name=endpoint_name,
            model=f"{DEFAULT_MODEL_NAME}:1",
            code_configuration=CodeConfiguration(
                code=str(CODE_DIR),
                scoring_script="batch_driver.py",
            ),
            environment=env,
            compute=DEFAULT_COMPUTE,
            instance_count=1,
            settings=ModelBatchDeploymentSettings(
                max_concurrency_per_instance=1,
                mini_batch_size=10,
                output_action=BatchDeploymentOutputAction.APPEND_ROW,
                output_file_name="predictions.csv",
                retry_settings=BatchRetrySettings(max_retries=2, timeout=300),
                logging_level="info",
                environment_variables={
                    "ONLINE_API_URL": self._online_api_url,
                    "ONLINE_API_KEY": self._online_api_key,
                    "ONLINE_DEPLOYMENT": self._online_deployment,
                    "MODEL_NAME": "Qwen/Qwen2.5-VL-32B-Instruct",
                },
            ),
        )

        logger.info(f"Creating/updating deployment: {deployment_name} on {endpoint_name}")
        poller = self._client.batch_deployments.begin_create_or_update(deployment)
        poller.wait(timeout=900)
        logger.info(f"Deployment provisioned: {deployment_name}")

        # Set as default
        ep = self._client.batch_endpoints.get(endpoint_name)
        ep.defaults.deployment_name = deployment_name
        self._client.batch_endpoints.begin_create_or_update(ep).wait(timeout=300)

        self._deployment_name = deployment_name
        return deployment_name

    # ── Job ─────────────────────────────────────────────────────────

    def submit_job(
        self,
        data_asset_path: str,
        endpoint_name: str | None = None,
        deployment_name: str | None = None,
    ) -> str:
        """Submit a batch scoring job. Returns job name."""
        if endpoint_name is None:
            endpoint_name = self._endpoint_name
        if endpoint_name is None:
            raise ValueError("No endpoint configured")

        logger.info(f"Submitting batch job to {endpoint_name}")

        job = self._client.batch_endpoints.invoke(
            endpoint_name=endpoint_name,
            deployment_name=deployment_name,
            input=Input(
                path=data_asset_path,
                type=AssetTypes.URI_FOLDER,
            ),
        )

        logger.info(f"Job submitted: {job.name}")
        return job.name

    def get_job_status(self, job_name: str) -> dict:
        """Get job status and progress from Azure ML.

        Returns dict with:
        - job_name, status, description
        - completed_tasks, total_tasks, failed_tasks (from job properties / services)
        - studio_url (link to Azure ML Studio)
        """
        try:
            job = self._client.jobs.get(job_name)
        except Exception as e:
            logger.error(f"Failed to get job {job_name}: {e}")
            return {"job_name": job_name, "status": "NotFound", "error": str(e)}

        props = getattr(job, "properties", {}) or {}

        def _first(*names: str) -> int:
            for n in names:
                v = props.get(n)
                if v is None and hasattr(job, n):
                    v = getattr(job, n, None)
                if v is not None:
                    try:
                        return int(v)
                    except (ValueError, TypeError):
                        continue
            return 0

        completed = _first("TasksCompleted", "CompletedTasks", "tasks_completed", "node_completed")
        total = _first("TasksTotal", "TotalTasks", "tasks_total", "node_total", "mini_batches_total")
        failed = _first("TasksFailed", "FailedTasks", "tasks_failed", "node_failed")

        if not total:
            try:
                tags = getattr(job, "tags", {}) or {}
                total = int(tags.get("total_buildings", 0) or 0)
            except (ValueError, TypeError):
                pass

        studio_url = ""
        try:
            services = job.services or {}
            for key in ("studio", "tracking", "default"):
                svc = services.get(key, {})
                if isinstance(svc, dict) and svc.get("url"):
                    studio_url = svc["url"]
                    break
        except Exception:
            pass

        if not studio_url:
            try:
                studio_url = getattr(job, "studio_url", "") or ""
            except Exception:
                pass

        return {
            "job_name": job.name,
            "status": job.status,
            "description": getattr(job, "description", ""),
            "completed_tasks": completed,
            "total_tasks": total,
            "failed_tasks": failed,
            "studio_url": studio_url,
        }

    def cancel_job(self, job_name: str) -> bool:
        """Cancel a running batch job. Returns True if cancelled."""
        try:
            self._client.jobs.begin_cancel(job_name).wait(timeout=120)
            logger.info(f"Job cancelled: {job_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel job {job_name}: {e}")
            return False

    def download_output(self, job_name: str) -> str | None:
        """Download predictions.csv from a completed batch endpoint job.

        Returns CSV content as string, or None if not available.
        """
        import tempfile
        import shutil

        try:
            job = self._client.jobs.get(job_name)
            logger.info(f"Job {job_name} status: {job.status}")
            if job.status != "Completed":
                logger.warning(f"Job {job_name} is {job.status}, not Completed")
                return None

            tmp_dir = Path(tempfile.mkdtemp(prefix="aml_output_"))
            try:
                try:
                    self._client.jobs.download(
                        name=job_name,
                        download_path=str(tmp_dir),
                        output_name="score",
                    )
                except Exception as e1:
                    logger.warning(f"download with output_name='score' failed: {e1}; retrying with no output_name")
                    self._client.jobs.download(
                        name=job_name,
                        download_path=str(tmp_dir),
                    )

                candidates = list(tmp_dir.rglob("predictions.csv"))
                if not candidates:
                    candidates = list(tmp_dir.rglob("*.csv"))
                if not candidates:
                    logger.error(f"No CSV found under {tmp_dir}. Files: {list(tmp_dir.rglob('*'))}")
                    return None

                csv_path = candidates[0]
                logger.info(f"Found predictions CSV at {csv_path}")
                return csv_path.read_text()
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        except Exception as e:
            logger.error(f"Failed to download output for {job_name}: {e}")
            return None

    # ── Cleanup ─────────────────────────────────────────────────────

    def delete_endpoint(self, endpoint_name: str | None = None) -> None:
        """Delete a batch endpoint."""
        if endpoint_name is None:
            endpoint_name = self._endpoint_name
        if endpoint_name is None:
            return

        try:
            self._client.batch_endpoints.begin_delete(name=endpoint_name).wait(timeout=300)
            logger.info(f"Deleted endpoint: {endpoint_name}")
        except Exception as e:
            logger.error(f"Failed to delete endpoint {endpoint_name}: {e}")


# ── Singleton ───────────────────────────────────────────────────────

_aml_batch_client: AmlBatchClient | None = None


def get_aml_batch_client() -> AmlBatchClient:
    """Get or create the singleton AML Batch Client."""
    global _aml_batch_client
    if _aml_batch_client is None:
        _aml_batch_client = AmlBatchClient()
    return _aml_batch_client
