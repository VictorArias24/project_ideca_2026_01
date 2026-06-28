"""
Online deployment lifecycle manager for Azure ML Managed Online Endpoints.

Provides manual start/stop control over a single online deployment to control
compute costs. Azure ML does not support instance_count=0, so "Stop" is
implemented as a full delete (compute deallocated, model artifacts removed).
The next "Start" recreates the deployment from scratch (5-10 min).

State machine:
    not_found        No deployment registered (initial state or after Stop)
    creating         Create in progress (5-10 min first time, similar after Stop)
    running          instance_count=1, accepting traffic (cost: ~$3.67/hr)
    deleting         Delete in progress (~30 sec)
    failed           Last transition failed; retry required

State is persisted to data/online_deployment_state.json so the UI can show
the last-known state on page load without a blocking Azure API call.
"""

import json
import logging
import os
import time
from enum import Enum
from pathlib import Path
from typing import Optional

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineDeployment, OnlineRequestSettings
from azure.core.exceptions import ResourceNotFoundError

logger = logging.getLogger(__name__)


class OnlineState(str, Enum):
    NOT_FOUND = "not_found"
    CREATING = "creating"
    RUNNING = "running"
    DELETING = "deleting"
    FAILED = "failed"


# Default deployment config (mirrors scripts/deploy_foundry_endpoint.py).
DEFAULT_MODEL_URI = (
    "azureml://registries/HuggingFace/models/"
    "qwen-qwen2.5-vl-32b-instruct/labels/latest"
)
DEFAULT_INSTANCE_TYPE = "Standard_NC24ads_A100_v4"
DEFAULT_REQUEST_TIMEOUT_MS = 120000
DEFAULT_MAX_CONCURRENT_PER_INSTANCE = 1

STATE_FILE = Path("data/online_deployment_state.json")


class OnlineDeploymentManager:
    """Manages a single online deployment: create, delete, status."""

    def __init__(
        self,
        endpoint_name: str,
        deployment_name: str,
        subscription_id: str,
        resource_group: str,
        workspace_name: str,
        model_uri: str = DEFAULT_MODEL_URI,
        instance_type: str = DEFAULT_INSTANCE_TYPE,
    ):
        self.endpoint_name = endpoint_name
        self.deployment_name = deployment_name
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        self.model_uri = model_uri
        self.instance_type = instance_type

        self._client: Optional[MLClient] = None

    @property
    def client(self) -> MLClient:
        if self._client is None:
            credential = DefaultAzureCredential()
            self._client = MLClient(
                credential=credential,
                subscription_id=self.subscription_id,
                resource_group_name=self.resource_group,
                workspace_name=self.workspace_name,
            )
        return self._client

    def get_status(self) -> dict:
        """Returns current state by querying Azure.

        Returns dict with: state, instances, traffic_pct, last_updated, error.
        """
        try:
            endpoint = self.client.online_endpoints.get(self.endpoint_name)
        except ResourceNotFoundError:
            return self._save_and_return({
                "state": OnlineState.NOT_FOUND.value,
                "instances": 0,
                "traffic_pct": 0,
                "last_updated": _now(),
                "error": None,
            })

        try:
            deployment = self.client.online_deployments.get(
                endpoint_name=self.endpoint_name,
                name=self.deployment_name,
            )
        except ResourceNotFoundError:
            return self._save_and_return({
                "state": OnlineState.NOT_FOUND.value,
                "instances": 0,
                "traffic_pct": 0,
                "last_updated": _now(),
                "error": None,
            })

        instances = getattr(deployment, "instance_count", 0) or 0
        provisioning = getattr(deployment, "provisioning_state", "Unknown")
        traffic_pct = int(endpoint.traffic.get(self.deployment_name, 0))

        state = self._map_state(provisioning)

        # Auto-route traffic to 100% when deployment becomes Succeeded.
        # Azure ML does NOT auto-route traffic on create — it must be set
        # explicitly. This fires on the first poll that sees Succeeded with
        # traffic < 100, so the user never has to manually route traffic.
        if state == OnlineState.RUNNING and traffic_pct < 100:
            traffic_pct = self._ensure_traffic_100()

        return self._save_and_return({
            "state": state.value,
            "instances": instances,
            "traffic_pct": traffic_pct,
            "last_updated": _now(),
            "error": None if provisioning != "Failed" else "Deployment in Failed state",
        })

    def _map_state(self, provisioning: str) -> OnlineState:
        if provisioning == "Creating":
            return OnlineState.CREATING
        if provisioning == "Deleting":
            return OnlineState.DELETING
        if provisioning == "Failed":
            return OnlineState.FAILED
        if provisioning == "Succeeded":
            return OnlineState.RUNNING
        return OnlineState.FAILED

    def _ensure_traffic_100(self) -> int:
        """Set endpoint traffic to 100% for this deployment.

        Azure ML does NOT auto-route traffic when a deployment is created —
        it must be set explicitly. This is called from get_status() the
        moment the deployment transitions to Succeeded with traffic < 100.

        Fire-and-forget: returns immediately after Azure accepts the endpoint
        update. The traffic change propagates within seconds.

        Returns the traffic percentage (100 if just set, existing if already
        at 100, 0 on error).
        """
        try:
            endpoint = self.client.online_endpoints.get(self.endpoint_name)
            current = int(endpoint.traffic.get(self.deployment_name, 0))
            if current >= 100:
                return current
            endpoint.traffic = {self.deployment_name: 100}
            poller = self.client.online_endpoints.begin_create_or_update(endpoint)
            logger.info(f"Traffic set to 100% for {self.deployment_name}")
            del poller
            return 100
        except Exception as e:
            logger.warning(f"Could not set traffic to 100%: {e}")
            return 0

    def start(self) -> dict:
        """Create the deployment (or no-op if already running).

        Azure ML doesn't support instance_count=0, so each Start is a full
        create (5-10 min the first time, similar after a Stop).

        Returns the resulting status dict.
        """
        status = self.get_status()
        state = OnlineState(status["state"])

        if state == OnlineState.RUNNING:
            return status
        if state in (OnlineState.CREATING, OnlineState.DELETING):
            return status

        if state in (OnlineState.NOT_FOUND, OnlineState.FAILED):
            return self._create_deployment()

        return status

    def stop(self) -> dict:
        """Delete the deployment (Stop = delete, since Azure ML does not allow
        instance_count=0). Compute is deallocated, model artifacts removed.

        Returns the resulting status dict.
        """
        status = self.get_status()
        state = OnlineState(status["state"])

        if state in (OnlineState.NOT_FOUND, OnlineState.DELETING):
            return status
        if state == OnlineState.CREATING:
            return status

        if state in (OnlineState.RUNNING, OnlineState.FAILED):
            return self._delete_deployment()

        return status

    def _create_deployment(self) -> dict:
        """Initiate deployment create (5-10 min on Azure).

        Fire-and-forget: returns immediately after Azure accepts the operation.
        The state is set to 'creating'. Status updates come from get_status()
        polling Azure directly. We never call poller.wait() because the SDK's
        poller uses multiprocessing, which leaves zombie subprocesses that
        eventually hang the asyncio event loop.
        """
        try:
            deployment = ManagedOnlineDeployment(
                name=self.deployment_name,
                endpoint_name=self.endpoint_name,
                model=self.model_uri,
                instance_type=self.instance_type,
                instance_count=1,
                request_settings=OnlineRequestSettings(
                    max_concurrent_requests_per_instance=DEFAULT_MAX_CONCURRENT_PER_INSTANCE,
                    request_timeout_ms=DEFAULT_REQUEST_TIMEOUT_MS,
                ),
            )
            poller = self.client.online_deployments.begin_create_or_update(deployment)
            logger.info(f"Create initiated for {self.deployment_name} (5-10 min on Azure)")
            del poller
        except Exception as e:
            logger.error(f"Failed to initiate deployment create: {e}")
            return self._save_and_return({
                "state": OnlineState.FAILED.value,
                "instances": 0,
                "traffic_pct": 0,
                "last_updated": _now(),
                "error": str(e),
            })

        return self._save_and_return({
            "state": OnlineState.CREATING.value,
            "instances": 0,
            "traffic_pct": 0,
            "last_updated": _now(),
            "error": None,
        })

    def _delete_deployment(self) -> dict:
        """Initiate deployment delete (~30 sec on Azure).

        Fire-and-forget: returns immediately after Azure accepts the operation.
        The state is set to 'deleting'. Status updates come from get_status()
        polling Azure directly. We never call poller.wait() (see _create_deployment).

        Azure ML rejects begin_delete() if the deployment has traffic > 0,
        so we MUST clear traffic on the endpoint before calling delete.
        """
        # Step 1: Clear traffic BEFORE delete (Azure rejects delete with traffic > 0)
        try:
            endpoint = self.client.online_endpoints.get(self.endpoint_name)
            current_traffic = int(endpoint.traffic.get(self.deployment_name, 0))
            if current_traffic > 0:
                endpoint.traffic = {}
                poller_ep = self.client.online_endpoints.begin_create_or_update(endpoint)
                logger.info(f"Traffic clear initiated for {self.deployment_name}")
                del poller_ep

                # Wait for Azure to process the traffic change (poll, no poller.wait)
                # Endpoint updates typically take 2-5 sec; timeout after 30 sec.
                for i in range(15):
                    time.sleep(2)
                    try:
                        ep_check = self.client.online_endpoints.get(self.endpoint_name)
                        if int(ep_check.traffic.get(self.deployment_name, 0)) == 0:
                            logger.info(f"Traffic confirmed at 0% (after {(i+1)*2}s)")
                            break
                    except Exception:
                        pass
                else:
                    logger.warning("Traffic did not reach 0% within 30s — trying delete anyway")
        except Exception as e:
            logger.warning(f"Could not clear traffic before delete: {e}")

        # Step 2: Delete the deployment (traffic is now 0, Azure accepts)
        try:
            poller = self.client.online_deployments.begin_delete(
                endpoint_name=self.endpoint_name,
                name=self.deployment_name,
            )
            logger.info(f"Delete initiated for {self.deployment_name} (~30 sec on Azure)")
            del poller
        except Exception as e:
            err_str = str(e)
            if "ResourceNotFound" in err_str or "NotFound" in err_str:
                logger.info(f"Deployment {self.deployment_name} already gone")
                return self._save_and_return({
                    "state": OnlineState.NOT_FOUND.value,
                    "instances": 0,
                    "traffic_pct": 0,
                    "last_updated": _now(),
                    "error": None,
                })
            logger.error(f"Failed to initiate deployment delete: {e}")
            return self._save_and_return({
                "state": OnlineState.FAILED.value,
                "instances": 0,
                "traffic_pct": 0,
                "last_updated": _now(),
                "error": str(e),
            })

        return self._save_and_return({
            "state": OnlineState.DELETING.value,
            "instances": 0,
            "traffic_pct": 0,
            "last_updated": _now(),
            "error": None,
        })

    def _save_and_return(self, status: dict) -> dict:
        status = {**status, "deployment_name": self.deployment_name, "endpoint_name": self.endpoint_name}
        _save_state(status)
        return status


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _save_state(status: dict) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(status, indent=2))
    except Exception as e:
        logger.warning(f"Could not save state file: {e}")


def load_state() -> dict:
    """Load last-known state from disk. Used for fast page-load rendering."""
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text())
    except Exception as e:
        logger.warning(f"Could not load state file: {e}")
    return {
        "state": OnlineState.NOT_FOUND.value,
        "instances": 0,
        "traffic_pct": 0,
        "last_updated": None,
        "error": None,
    }


_singleton: Optional[OnlineDeploymentManager] = None


def get_online_manager() -> OnlineDeploymentManager:
    """Get or create the singleton OnlineDeploymentManager.

    Reads config from environment:
      ENDPOINT_NAME, DEPLOYMENT_NAME, SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME
    """
    global _singleton
    if _singleton is None:
        _singleton = OnlineDeploymentManager(
            endpoint_name=os.getenv("ENDPOINT_NAME", ""),
            deployment_name=os.getenv("DEPLOYMENT_NAME", "qwen-vl-16k"),
            subscription_id=os.getenv("SUBSCRIPTION_ID", ""),
            resource_group=os.getenv("RESOURCE_GROUP", ""),
            workspace_name=os.getenv("WORKSPACE_NAME", ""),
        )
    return _singleton
