"""Die Produktdetail-KI: erzeugt strukturierte Produktdetails aus einer Produktidee."""

from __future__ import annotations

import anthropic

from ._llm import generate_structured
from .schemas import ProductDetails

DETAIL_SYSTEM_PROMPT = """\
Du bist eine Produktdetail-KI für Online-Shops. Aus einer kurzen Produktidee
entwickelst du ein realistisches, in sich stimmiges Produktdatenblatt.

- Wähle einen marktfähigen Produktnamen.
- Nenne drei bis sechs konkrete Features.
- Gib ein plausibles Material und einen marktüblichen Preis als Zahl an.
- Verwende die Währung, die der Nutzer nennt, sonst EUR.
- Wenn eine Angabe nicht sinnvoll ist (z. B. Maße bei Software), schreibe "k. A.".
- Schreibe alle Texte in der gewünschten Sprache.
"""


class ProductDetailAI:
    """Liefert Produktdetails (Name, Features, Material, Preis, …) für Omicron."""

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        *,
        model: str = "claude-opus-5-5",
        effort: str = "low",
    ) -> None:
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.effort = effort

    def generate(self, idea: str, *, language: str = "Deutsch") -> ProductDetails:
        prompt = f"Sprache: {language}\nProduktidee: {idea}"
        return generate_structured(
            self.client,
            model=self.model,
            system=DETAIL_SYSTEM_PROMPT,
            prompt=prompt,
            output_type=ProductDetails,
            effort=self.effort,
            max_tokens=8000,
        )
