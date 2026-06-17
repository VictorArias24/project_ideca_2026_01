"""
Mock VLM adapter for testing without Azure GPU costs.
Returns deterministic, schema-valid responses.
"""

import json
import time
from typing import Any

from src.ports.vlm_client_port import VlmClientPort

MOCK_RESPONSES = {
    "residential": {
        "building_id": "mock",
        "caracteristicas_visibles": [
            "ventanas residenciales", "balcones", "cortinas visibles",
            "entrada residencial", "ladrillo visto",
        ],
        "clases": [
            {
                "label": "RESIDENCIAL_1",
                "confidence": 0.88,
                "evidencia_visual": [
                    "balcones con baranda metalica",
                    "cortinas en ventanas",
                    "puerta de entrada residencial",
                ],
                "is_primary": True,
            }
        ],
        "requiere_revision": False,
        "razon_revision": [],
        "model_id": "mock-adapter",
        "perfil_modelo": "80gb_high_accuracy",
        "schema_version": "1.0.0",
        "taxonomy_version": "1.0.0",
    },
    "commercial": {
        "building_id": "mock",
        "caracteristicas_visibles": [
            "vitrina comercial", "letrero de negocio", "entrada de clientes",
            "fachada de local", "rejas metalicas",
        ],
        "clases": [
            {
                "label": "COMERCIAL_1",
                "confidence": 0.82,
                "evidencia_visual": [
                    "vitrina con productos exhibidos",
                    "letrero comercial visible",
                    "entrada amplia para clientes",
                ],
                "is_primary": True,
            }
        ],
        "requiere_revision": False,
        "razon_revision": [],
        "model_id": "mock-adapter",
        "perfil_modelo": "80gb_high_accuracy",
        "schema_version": "1.0.0",
        "taxonomy_version": "1.0.0",
    },
    "rural": {
        "building_id": "mock",
        "caracteristicas_visibles": [
            "estructura de madera y zinc", "cercado de alambre",
            "terreno con pasto", "animales visibles",
            "camino sin pavimentar", "contexto no urbano",
        ],
        "clases": [
            {
                "label": "RURAL_1",
                "confidence": 0.91,
                "evidencia_visual": [
                    "techo de zinc ondulado",
                    "cerca de alambre de puas",
                    "gallinas en el terreno",
                    "vegetacion de pastizal alrededor",
                ],
                "is_primary": True,
            }
        ],
        "requiere_revision": False,
        "razon_revision": [],
        "model_id": "mock-adapter",
        "perfil_modelo": "80gb_high_accuracy",
        "schema_version": "1.0.0",
        "taxonomy_version": "1.0.0",
    },
    "review": {
        "building_id": "mock",
        "caracteristicas_visibles": [
            "construccion nueva", "malla de construccion verde",
            "andamios visibles", "concreto expuesto",
        ],
        "clases": [
            {
                "label": "MOLES_1",
                "confidence": 0.55,
                "evidencia_visual": [
                    "malla plastica verde cubriendo la fachada",
                    "andamios metalicos en varios pisos",
                ],
                "is_primary": True,
            }
        ],
        "requiere_revision": True,
        "razon_revision": [
            "max_confidence < 0.70",
            "MOLES_1 is a suggested class",
        ],
        "model_id": "mock-adapter",
        "perfil_modelo": "80gb_high_accuracy",
        "schema_version": "1.0.0",
        "taxonomy_version": "1.0.0",
    },
}


class MockVlmAdapter(VlmClientPort):
    """Mock VLM that returns pre-built valid responses.

    Use for local development and automated testing without Azure GPU costs.

    Args:
        mode: Which mock response to return ("residential", "commercial",
              "rural", "review").
        latency_ms: Simulated inference latency in milliseconds.
    """

    def __init__(self, mode: str = "residential", latency_ms: float = 100):
        self.mode = mode
        self.latency_ms = latency_ms
        self.call_count = 0

    def classify_building(
        self,
        image_data_urls: list[str],
        building_id: str = "unknown",
    ) -> dict[str, Any]:
        self.call_count += 1
        time.sleep(self.latency_ms / 1000)

        response = dict(MOCK_RESPONSES.get(self.mode, MOCK_RESPONSES["residential"]))
        response["building_id"] = building_id

        n_images = len(image_data_urls)
        return {
            "text": json.dumps(response, ensure_ascii=False),
            "latency_ms": self.latency_ms,
            "usage": {
                "prompt_tokens": 500 + n_images * 200,
                "completion_tokens": 200,
                "total_tokens": 700 + n_images * 200,
            },
            "finish_reason": "stop",
        }

    def health_check(self) -> bool:
        return True
