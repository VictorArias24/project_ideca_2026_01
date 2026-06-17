"""
Classification service: orchestrates VLM call + parse + persist.
"""

import asyncio
import logging
from typing import Optional
import asyncio

from src.ports.vlm_client_port import VlmClientPort
from src.parsers.output_parser import parse_and_validate
from src.persistence.models import Prediction
from src.utils.image_encoding import image_to_data_url
from src.application.review_service import ReviewService

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen2.5-VL-32B-Instruct"
PROFILE = "80gb_high_accuracy"
SCHEMA_VERSION = "1.0.0"
TAXONOMY_VERSION = "1.0.0"
PROMPT_VERSION = "p004"


class ClassifyService:
    """Application service for building classification.

    Depends on VlmClientPort (not on concrete adapters).
    """

    def __init__(
        self,
        vlm_client: VlmClientPort,
        model_id: str = MODEL_ID,
        profile: str = PROFILE,
        schema_version: str = SCHEMA_VERSION,
        taxonomy_version: str = TAXONOMY_VERSION,
    ):
        self.vlm_client = vlm_client
        self.model_id = model_id
        self.profile = profile
        self.schema_version = schema_version
        self.taxonomy_version = taxonomy_version

    async def classify_building(
        self,
        building_id: str,
        image_urls: list[str],
        job_id: Optional[str] = None,
    ) -> Prediction:
        """Classify one building and return a Prediction record.

        Args:
            building_id: Identifier for the building.
            image_urls: File paths, HTTP URLs, or data: URIs.
            job_id: Optional batch job identifier.

        Returns:
            Prediction ORM record. Caller persists it via PredictionRepository.
        """

        data_urls = [image_to_data_url(u, max_dim=0) for u in image_urls]

        try:
            vlm_response = await asyncio.to_thread(
                self.vlm_client.classify_building, data_urls, building_id
            )
        except Exception as exc:
            logger.error(f"VLM call failed for {building_id}: {exc}")
            return Prediction(
                building_id=building_id,
                job_id=job_id,
                model_id=self.model_id,
                perfil_modelo=self.profile,
                prompt_version=PROMPT_VERSION,
                schema_version=self.schema_version,
                taxonomy_version=self.taxonomy_version,
                image_uris=image_urls,
                classification={"error": str(exc)[:2000]},
                requires_review=True,
                review_status="pending",
                latency_ms=0,
            )

        parsed, error = parse_and_validate(vlm_response["text"])

        if parsed is None:
            parsed, error = await self._retry_with_correction(
                data_urls, building_id, error
            )
            if parsed is not None:
                vlm_response = parsed.pop("_vlm_response", vlm_response)

        if parsed is None:
            logger.warning(f"Classification failed for {building_id}: {error}")
            return Prediction(
                building_id=building_id,
                job_id=job_id,
                model_id=self.model_id,
                perfil_modelo=self.profile,
                prompt_version=PROMPT_VERSION,
                schema_version=self.schema_version,
                taxonomy_version=self.taxonomy_version,
                image_uris=image_urls,
                classification={"error": error, "raw_response": vlm_response["text"][:5000]},
                requires_review=True,
                review_status="pending",
                latency_ms=vlm_response.get("latency_ms", 0),
            )

        needs_review, routing_reasons = ReviewService.apply_routing_rules(parsed)
        if needs_review:
            existing = parsed.setdefault("razon_revision", [])
            for reason in routing_reasons:
                if reason not in existing:
                    existing.append(reason)
            parsed["requiere_revision"] = True

        primary = next(
            (c for c in parsed.get("clases", []) if c.get("is_primary")),
            parsed["clases"][0] if parsed.get("clases") else None,
        )

        return Prediction(
            building_id=building_id,
            job_id=job_id,
            model_id=self.model_id,
            perfil_modelo=self.profile,
            prompt_version=PROMPT_VERSION,
            schema_version=self.schema_version,
            taxonomy_version=self.taxonomy_version,
            image_uris=image_urls,
            classification=parsed,
            primary_label=primary["label"] if primary else None,
            max_confidence=primary["confidence"] if primary else None,
            requires_review=parsed.get("requiere_revision", True),
            review_status="pending" if parsed.get("requiere_revision", True) else "accepted",
            latency_ms=vlm_response.get("latency_ms", 0),
        )

    async def _retry_with_correction(
        self,
        data_urls: list[str],
        building_id: str,
        error: str,
    ):
        corrected_id = f"{building_id}-retry"
        vlm_response = await asyncio.to_thread(
            self.vlm_client.classify_building, data_urls, corrected_id
        )

        parsed, error2 = parse_and_validate(vlm_response["text"])
        if parsed is not None:
            parsed["building_id"] = building_id
            parsed["_vlm_response"] = vlm_response

        return parsed, error2
