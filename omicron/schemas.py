"""Datenmodelle für Produktdetails und Produktbeschreibungen."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ProductDetails(BaseModel):
    """Die Produktdetails, die die Detail-KI liefert und Omicron als Eingabe nutzt."""

    model_config = ConfigDict(extra="forbid")

    name: str
    category: str
    features: list[str]
    material: str
    price: float
    currency: str
    target_audience: str
    dimensions: str
    colors: list[str]
    additional_info: str


class ProductDescription(BaseModel):
    """Die fertige Produktbeschreibung, die Omicron (Mira) erstellt."""

    model_config = ConfigDict(extra="forbid")

    headline: str
    short_description: str
    long_description: str
    bullet_points: list[str]
    call_to_action: str
    seo_keywords: list[str]

    def to_markdown(self) -> str:
        bullets = "\n".join(f"- {point}" for point in self.bullet_points)
        keywords = ", ".join(self.seo_keywords)
        return (
            f"# {self.headline}\n\n"
            f"*{self.short_description}*\n\n"
            f"{self.long_description}\n\n"
            f"{bullets}\n\n"
            f"**{self.call_to_action}**\n\n"
            f"_SEO-Keywords: {keywords}_\n"
        )


def output_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON-Schema für Structured Outputs (alle Felder Pflicht, keine Zusatzfelder)."""
    return model.model_json_schema()
