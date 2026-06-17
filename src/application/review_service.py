"""
Review service: routing rules, decision processing, and queue statistics.

Kept separate from classify_service so classification orchestration
does not depend on review logic. Both are application services that
depend only on domain objects (ports, models, parsers).
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from src.persistence.models import Prediction
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

logger = logging.getLogger(__name__)

VALID_DECISIONS = frozenset({"accept", "change_label", "mark_unknown", "request_images", "field_visit"})

DECISION_STATUS_MAP = {
    "accept": "accepted",
    "change_label": "corrected",
    "mark_unknown": "unknown",
    "request_images": "pending",
    "field_visit": "field_visit",
}


class ReviewService:
    """Stateless service for review routing, decision processing, and stats."""

    # ── Routing ────────────────────────────────────────────────────

    @staticmethod
    def apply_routing_rules(parsed: dict) -> tuple[bool, list[str]]:
        """Apply server-side review routing rules to parsed VLM output.

        These rules complement the model's own requiere_revision flag.
        They catch cases the model might miss or where policy mandates review.

        Returns:
            (needs_review: bool, reasons: list[str])
        """
        reasons = []

        primary = next(
            (c for c in parsed.get("clases", []) if c.get("is_primary")),
            parsed["clases"][0] if parsed.get("clases") else None,
        )

        if not primary:
            return True, ["no_primary_class_found"]

        conf = primary.get("confidence", 0)
        label = primary.get("label", "")

        if conf < 0.65:
            reasons.append("top_class_confidence_below_threshold")

        clases = parsed.get("clases", [])
        if len(clases) > 1:
            confs = sorted([c.get("confidence", 0) for c in clases], reverse=True)
            if len(confs) >= 2 and (confs[0] - confs[1]) < 0.10:
                reasons.append("ambiguous_top_classes")

        if label in ("COMERCIAL_3", "DOTACIONAL_2", "MOLES_1"):
            reasons.append("suggested_class_requires_validation")

        if label == "UNKNOWN_OR_INSUFFICIENT_EVIDENCE":
            reasons.append("model_cannot_determine")

        return bool(reasons), reasons

    # ── Decision processing ────────────────────────────────────────

    @staticmethod
    def process_decision(
        prediction: Prediction,
        decision: str,
        reviewer_id: str = "anonymous",
        corrected_label: Optional[str] = None,
        correction_reason: str = "",
        reviewer_notes: str = "",
    ) -> Prediction:
        """Apply a human review decision to a Prediction record.

        Mutates *prediction* in place and returns it.
        The caller is responsible for persisting via PredictionRepository.save().

        Args:
            prediction: The Prediction record to update.
            decision: One of accept, change_label, mark_unknown, request_images, field_visit.
            reviewer_id: Identifier for the reviewer.
            corrected_label: Required if decision=change_label.
            correction_reason: Explanation for the decision.
            reviewer_notes: Free-form reviewer notes (stored in correction_reason for now).
        """
        if decision not in VALID_DECISIONS:
            raise ValueError(f"Invalid decision: {decision}. Must be one of {sorted(VALID_DECISIONS)}")

        if decision == "change_label" and not corrected_label:
            raise ValueError("corrected_label is required when decision=change_label")

        prediction.review_status = DECISION_STATUS_MAP[decision]
        prediction.reviewed_by = reviewer_id
        prediction.review_timestamp = datetime.now(timezone.utc)
        prediction.correction_reason = correction_reason or reviewer_notes or f"Decision: {decision}"
        prediction.updated_at = datetime.now(timezone.utc)

        if decision == "change_label":
            classification = prediction.classification or {}
            clases = classification.get("clases", [])
            for c in clases:
                if c.get("is_primary"):
                    c["label"] = corrected_label
                    c["is_corrected"] = True
            classification["clases"] = clases
            prediction.classification = classification
            prediction.primary_label = corrected_label

        if decision == "accept":
            prediction.requires_review = False

        logger.info(
            "Review decision for %s: %s → %s by %s",
            prediction.building_id,
            decision,
            prediction.review_status,
            reviewer_id,
        )

        return prediction

    # ── Statistics ─────────────────────────────────────────────────

    @staticmethod
    async def get_review_stats(session: AsyncSession) -> dict:
        """Get review queue statistics.

        Returns dict with keys: queue_size, total_accepted, total_corrected,
        total_field_visit, total_unknown, override_rate.
        """
        from src.persistence.models import Prediction

        total_in_queue = await session.scalar(
            select(func.count()).where(
                Prediction.requires_review == True,
                Prediction.review_status == "pending",
            )
        ) or 0

        total_accepted = await session.scalar(
            select(func.count()).where(Prediction.review_status == "accepted")
        ) or 0

        total_corrected = await session.scalar(
            select(func.count()).where(Prediction.review_status == "corrected")
        ) or 0

        total_field_visit = await session.scalar(
            select(func.count()).where(Prediction.review_status == "field_visit")
        ) or 0

        total_unknown = await session.scalar(
            select(func.count()).where(Prediction.review_status == "unknown")
        ) or 0

        reviewed = total_accepted + total_corrected
        override_rate = total_corrected / reviewed if reviewed > 0 else 0.0

        return {
            "queue_size": total_in_queue,
            "total_accepted": total_accepted,
            "total_corrected": total_corrected,
            "total_field_visit": total_field_visit,
            "total_unknown": total_unknown,
            "override_rate": round(override_rate, 4),
        }
