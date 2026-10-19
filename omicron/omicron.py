"""Omicron: die KI, die aus Produktdetails Produktbeschreibungen erstellt."""

from __future__ import annotations

import anthropic

from ._llm import generate_structured
from .mira import MIRA, MiraModel
from .schemas import ProductDescription, ProductDetails


class Omicron:
    """Erstellt Produktbeschreibungen mit dem Mira-Modell."""

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        *,
        model: MiraModel = MIRA,
    ) -> None:
        self.client = client or anthropic.Anthropic()
        self.model = model

    def describe(
        self,
        details: ProductDetails,
        *,
        language: str = "Deutsch",
        tone: str = "professionell und freundlich",
    ) -> ProductDescription:
        prompt = (
            f"Sprache: {language}\n"
            f"Tonalität: {tone}\n\n"
            "Produktdetails (JSON):\n"
            f"{details.model_dump_json(indent=2)}"
        )
        return generate_structured(
            self.client,
            model=self.model.base_model,
            system=self.model.system_prompt,
            prompt=prompt,
            output_type=ProductDescription,
            effort=self.model.effort,
            max_tokens=self.model.max_tokens,
        )
