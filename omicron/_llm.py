"""Gemeinsamer Aufruf der Claude-API mit strukturierter JSON-Ausgabe."""

from __future__ import annotations

from typing import TypeVar

import anthropic
from pydantic import BaseModel

from .schemas import output_schema

T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class OmicronError(RuntimeError):
    """Fehler, wenn eine KI keine verwertbare Antwort liefert."""


def generate_structured(
    client: anthropic.Anthropic,
    *,
    model: str,
    system: str,
    prompt: str,
    output_type: type[T],
    effort: str,
    max_tokens: int,
) -> T:
    """Fragt das Modell an und validiert die Antwort gegen ``output_type``."""
    response = client.beta.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        output_config={
            "effort": effort,
            "format": {"type": "json_schema", "schema": output_schema(output_type)},
        },
        # Lehnt das Modell eine Anfrage ab, übernimmt serverseitig ein Ersatzmodell.
        betas=[FALLBACK_BETA],
        fallbacks="default",
    )

    if response.stop_reason == "refusal":
        explanation = response.stop_details.explanation if response.stop_details else None
        raise OmicronError(f"Die Anfrage wurde abgelehnt. {explanation or ''}".strip())
    if response.stop_reason == "max_tokens":
        raise OmicronError("Die Antwort wurde abgeschnitten (max_tokens erreicht).")

    text = "".join(block.text for block in response.content if block.type == "text")
    try:
        return output_type.model_validate_json(text)
    except ValueError as exc:
        raise OmicronError(f"Ungültige Antwort des Modells: {exc}") from exc
