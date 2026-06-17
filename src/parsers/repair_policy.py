"""
JSON repair policy: What to do when the VLM returns invalid JSON.

Strategy (ordered):
1. Attempt direct parse_and_validate
2. If no JSON found: apply basic repairs (strip markdown, clip braces)
3. If parse error: apply basic_json_repair and retry parse
4. If validation error: retry with correction prompt (1 retry)
5. If retry fails: return error — caller routes to review queue

Usage:
    from src.parsers.repair_policy import RepairPolicy
    repair = RepairPolicy(vlm_call_fn)
    result, error, attempts = repair.classify_with_repair(messages)
"""

import logging
from typing import Any, Callable, Optional
from src.parsers.output_parser import parse_and_validate, extract_json

logger = logging.getLogger(__name__)

MAX_RETRIES = 1


def basic_json_repair(json_str: str) -> str:
    """Attempt basic fixes on malformed JSON strings.

    - Remove trailing commas before } and ]
    - Strip markdown code block markers
    - Clip to outermost { ... }
    """
    import re

    # Remove markdown code block markers
    json_str = re.sub(r"^```(?:json)?\s*\n?", "", json_str)
    json_str = re.sub(r"\n?```$", "", json_str)

    # Remove trailing commas before closing brackets
    json_str = re.sub(r",\s*}", "}", json_str)
    json_str = re.sub(r",\s*]", "]", json_str)

    # Clip to outermost {} pair
    brace_start = json_str.find("{")
    brace_end = json_str.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        json_str = json_str[brace_start : brace_end + 1]

    return json_str.strip()


class RepairPolicy:
    """Handles parsing failures with retry and fallback logic."""

    def __init__(
        self,
        vlm_call: Callable[[list[dict]], str],
        max_retries: int = MAX_RETRIES,
    ):
        self.vlm_call = vlm_call
        self.max_retries = max_retries

    def classify_with_repair(
        self,
        messages: list[dict],
    ) -> tuple[Optional[dict[str, Any]], Optional[str], int]:
        """Attempt classification with repair on failure.

        Args:
            messages: List of message dicts for the VLM API.

        Returns:
            (parsed_result_or_None, error_message_or_None, attempt_count)
        """

        for attempt in range(self.max_retries + 1):
            try:
                response_text = self.vlm_call(messages)
            except Exception as e:
                return None, f"VLM call failed: {e}", attempt + 1

            # Try direct parse
            result, error = parse_and_validate(response_text)

            if result is not None:
                return result, None, attempt + 1

            # Try basic repair, then re-parse
            json_str = extract_json(response_text)
            if json_str is not None:
                repaired = basic_json_repair(json_str)
                try:
                    import json
                    data = json.loads(repaired)
                    from src.parsers.output_parser import validate_output
                    is_valid, val_error = validate_output(data)
                    if is_valid:
                        logger.info(f"JSON repaired successfully on attempt {attempt + 1}")
                        return data, None, attempt + 1
                    else:
                        error = val_error
                except json.JSONDecodeError:
                    pass

            # If more retries available, retry with correction prompt
            if attempt < self.max_retries:
                logger.warning(
                    f"Parse attempt {attempt + 1} failed: {error}. Retrying..."
                )
                messages = self._build_retry_messages(messages, error)
            else:
                logger.error(
                    f"All {self.max_retries + 1} attempts failed. "
                    f"Last error: {error}"
                )

        return None, error, self.max_retries + 1

    def _build_retry_messages(
        self, original_messages: list[dict], error: str
    ) -> list[dict]:
        """Add a correction instruction to the messages."""
        correction = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Your previous response was invalid. Error: {error}\n\n"
                        "Please return ONLY the corrected JSON. No other text."
                    ),
                }
            ],
        }
        return original_messages + [correction]
