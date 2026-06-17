"""
Foundry VLM adapter: Clean wrapper around the Azure ML Managed Online Endpoint
via the OpenAI-compatible API. Same client pattern proven in Phase 02.

Implements VlmClientPort for hexagonal architecture (Phase 04).

Usage:
    from src.adapters.foundry_adapter import FoundryVlmAdapter

    adapter = FoundryVlmAdapter()
    if adapter.health_check():
        # Low-level: pass pre-built messages
        result = adapter.classify_raw(messages)
        # High-level (VlmClientPort): images + building_id
        result = adapter.classify_building(data_urls, "b-001")
        print(result["text"])
"""

import os
import time
import logging
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv
from openai import OpenAI

from src.ports.vlm_client_port import VlmClientPort

_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env")

logger = logging.getLogger(__name__)


def _get_build_messages(prompt_version: str = "p001"):
    """Import the right prompt builder based on version."""
    if prompt_version == "p004":
        from src.prompts.classification_p004 import build_messages as bm
        return bm
    elif prompt_version == "p003":
        from src.prompts.classification_p003 import build_messages as bm
        return bm
    elif prompt_version == "p002":
        from src.prompts.classification_p002 import build_messages as bm
        return bm
    else:
        from src.prompts.classification_p001 import build_messages as bm
        return bm


class FoundryVlmAdapter(VlmClientPort):
    """Adapter for Azure ML Managed Online Endpoint via OpenAI-compatible API."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        deployment_name: Optional[str] = None,
        model_name: str = "Qwen/Qwen2.5-VL-32B-Instruct",
        max_tokens: int = 2048,
        temperature: float = 0.0,
        prompt_version: str = "p001",
    ):
        self.api_url = api_url or os.getenv("API_URL")
        self.api_key = api_key or os.getenv("API_KEY")
        self.deployment_name = deployment_name or os.getenv("DEPLOYMENT_NAME")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.prompt_version = prompt_version
        self._build_messages = _get_build_messages(prompt_version)

        if not all([self.api_url, self.api_key, self.deployment_name]):
            raise ValueError(
                "Missing configuration. Set API_URL, API_KEY, DEPLOYMENT_NAME "
                "in .env or pass to constructor."
            )

        self.client = OpenAI(
            base_url=f"{self.api_url}/v1",
            api_key=self.api_key,
            default_headers={"azureml-model-deployment": self.deployment_name},
        )

    def classify_raw(
        self,
        messages: list[dict],
    ) -> dict[str, Any]:
        """Low-level: send pre-built messages to the VLM and return raw response.

        Args:
            messages: List of message dicts (system + user with text/images).

        Returns:
            Dict with keys: text, latency_ms, usage, finish_reason.
        """
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        elapsed_ms = (time.time() - start) * 1000

        usage = response.usage
        return {
            "text": response.choices[0].message.content,
            "latency_ms": elapsed_ms,
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            "finish_reason": response.choices[0].finish_reason,
        }

    def classify_building(
        self,
        image_data_urls: list[str],
        building_id: str = "unknown",
    ) -> dict[str, Any]:
        """Classify a building from image data URLs (VlmClientPort interface).

        Builds the full prompt via the configured prompt version ({self.prompt_version})
        and sends it to the endpoint. Returns the same dict as classify_raw().
        """
        messages = self._build_messages(image_data_urls, building_id)
        return self.classify_raw(messages)

    def health_check(self) -> bool:
        """Verify the endpoint is reachable and accepting requests."""
        try:
            self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
