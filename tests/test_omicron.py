import csv
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


def test_csv_writes_description_per_row_without_inventing_features(tmp_path):
    from omicron.csv_batch import process_csv

    source = tmp_path / "produkte.csv"
    source.write_text(
        "name;kategorie;material;features;preis;zielgruppe;output\n"
        'PendlerPack Pro;Rucksack;Recyceltes Polyester;"Wasserdicht, Reflektoren";89,90 €;Pendler;\n'
        "Kaffeebecher;Küche;Edelstahl;Doppelwandig;24,95;Berufstätige;\n",
        encoding="utf-8",
    )
    target = tmp_path / "produkte_fertig.csv"
    client = FakeClient(DESCRIPTION, DESCRIPTION)

    assert process_csv(source, target, Omicron(client)) == 0

    with target.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file, delimiter=";"))
    assert [row["name"] for row in rows] == ["PendlerPack Pro", "Kaffeebecher"]
    assert all(row["output"].startswith("# Trocken ins Büro") for row in rows)

    # Mira bekommt genau die Features aus der Tabelle und keine erfundenen Zusatzangaben.
    sent = json.loads(client.calls[0]["messages"][0]["content"].split("(JSON):\n", 1)[1])
    assert sent["features"] == ["Wasserdicht", "Reflektoren"]
    assert sent["price"] == 89.9
    assert sent["colors"] == []
    assert sent["dimensions"] == "k. A."
    assert "Nenne nur Features" in client.calls[0]["system"]


def test_csv_marks_failed_rows_and_continues(tmp_path):
    from omicron.csv_batch import process_csv

    source = tmp_path / "produkte.csv"
    source.write_text(
        "name,kategorie,material,features,preis,zielgruppe,output\n"
        "Ohne Preis,Test,Holz,Stabil,,Alle,\n"
        "Becher,Küche,Edelstahl,Doppelwandig,24.95,Alle,\n",
        encoding="utf-8",
    )
    target = tmp_path / "produkte_fertig.csv"

    assert process_csv(source, target, Omicron(FakeClient(DESCRIPTION))) == 1

    with target.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["output"].startswith("FEHLER: Ungültiger Preis")
    assert rows[1]["output"].startswith("# Trocken ins Büro")


def test_csv_missing_columns_raises(tmp_path):
    from omicron.csv_batch import process_csv

    source = tmp_path / "produkte.csv"
    source.write_text("name,preis\nBecher,5\n", encoding="utf-8")
    with pytest.raises(OmicronError, match="kategorie"):
        process_csv(source, tmp_path / "out.csv", Omicron(FakeClient()))
