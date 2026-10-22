import json
from types import SimpleNamespace

import pytest

from omicron import MIRA, Omicron, OmicronError, ProductDetailAI, ProductDetails
from omicron.__main__ import main

DETAILS = {
    "name": "PendlerPack Pro",
    "category": "Rucksack",
    "features": ["Wasserdicht", "Laptopfach 16 Zoll", "Reflektoren"],
    "material": "Recyceltes Polyester",
    "price": 89.9,
    "currency": "EUR",
    "target_audience": "Fahrradpendler",
    "dimensions": "45 x 30 x 15 cm",
    "colors": ["Schwarz", "Olivgrün"],
    "additional_info": "Volumen 22 Liter",
}

DESCRIPTION = {
    "headline": "Trocken ins Büro – bei jedem Wetter",
    "short_description": "Der wasserdichte Rucksack für Fahrradpendler.",
    "long_description": "Absatz eins.\n\nAbsatz zwei.",
    "bullet_points": ["Wasserdicht", "Platz für 16-Zoll-Laptop", "Gut sichtbar"],
    "call_to_action": "Jetzt entdecken.",
    "seo_keywords": ["Pendlerrucksack", "wasserdichter Rucksack"],
}


class FakeClient:
    """Imitiert ``anthropic.Anthropic`` und gibt vorbereitete Antworten zurück."""

    def __init__(self, *payloads, stop_reason="end_turn"):
        self.payloads = list(payloads)
        self.stop_reason = stop_reason
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        text = json.dumps(self.payloads.pop(0)) if self.payloads else ""
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            stop_details=SimpleNamespace(explanation="abgelehnt"),
            content=[SimpleNamespace(type="text", text=text)],
        )


def test_omicron_uses_mira_and_returns_description():
    client = FakeClient(DESCRIPTION)
    description = Omicron(client).describe(ProductDetails(**DETAILS), tone="locker")

    assert description.headline == DESCRIPTION["headline"]
    call = client.calls[0]
    assert call["model"] == MIRA.base_model
    assert call["system"] == MIRA.system_prompt
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["fallbacks"] == "default"
    assert "PendlerPack Pro" in call["messages"][0]["content"]
    assert "Tonalität: locker" in call["messages"][0]["content"]


def test_detail_ai_returns_product_details():
    client = FakeClient(DETAILS)
    details = ProductDetailAI(client).generate("Rucksack für Pendler")

    assert details.price == 89.9
    assert "Produktidee: Rucksack für Pendler" in client.calls[0]["messages"][0]["content"]


def test_refusal_raises():
    client = FakeClient(DESCRIPTION, stop_reason="refusal")
    with pytest.raises(OmicronError, match="abgelehnt"):
        Omicron(client).describe(ProductDetails(**DETAILS))


def test_invalid_json_raises():
    client = FakeClient({"headline": "nur ein Feld"})
    with pytest.raises(OmicronError, match="Ungültige Antwort"):
        Omicron(client).describe(ProductDetails(**DETAILS))


def test_markdown_contains_all_parts():
    from omicron import ProductDescription

    markdown = ProductDescription(**DESCRIPTION).to_markdown()
    assert markdown.startswith("# Trocken ins Büro")
    assert "- Wasserdicht" in markdown
    assert "Pendlerrucksack" in markdown


def test_cli_pipeline(monkeypatch, capsys):
    client = FakeClient(DETAILS, DESCRIPTION)
    monkeypatch.setattr("omicron.__main__.anthropic.Anthropic", lambda: client)

    assert main(["Rucksack für Pendler"]) == 0
    assert "# Trocken ins Büro" in capsys.readouterr().out
    assert len(client.calls) == 2
