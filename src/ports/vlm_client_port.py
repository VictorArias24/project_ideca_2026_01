"""
VlmClientPort: Abstract interface for VLM inference.

All domain logic depends on this port, never on concrete adapters.
Follow hexagonal (ports & adapters) architecture.
"""

from abc import ABC, abstractmethod
from typing import Any


class VlmClientPort(ABC):
    """Port for Vision Language Model inference."""

    @abstractmethod
    def classify_building(
        self,
        image_data_urls: list[str],
        building_id: str = "unknown",
    ) -> dict[str, Any]:
        """
        Send images + prompt to the VLM and return raw response.

        Args:
            image_data_urls: List of base64 data: URLs (one per view).
            building_id: Identifier for the building being classified.

        Returns:
            Dict with keys: text, latency_ms, usage, finish_reason.
        """
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the VLM endpoint is reachable and accepting requests."""
        ...
