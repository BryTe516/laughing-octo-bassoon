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


CSV_HEADER = "name,kategorie,material,features,preis,zielgruppe,output\n"
CSV_ROW = (
    'Edelstahl-Trinkflasche Aqua 750 ml,Sport & Outdoor,Edelstahl,'
    '"750 ml, hält 24 h kalt, BPA-frei",24.90 €,Sportler und Wanderer,\n'
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("39.90 €", (39.9, "EUR")),
        ("1.299,00 EUR", (1299.0, "EUR")),
        ("5,99 €", (5.99, "EUR")),
        ("1,299.50 $", (1299.5, "USD")),
        ("12", (12.0, "")),
    ],
)
def test_parse_price(value, expected):
    from omicron.csv_batch import parse_price

    assert parse_price(value) == expected


def test_row_to_details_maps_german_columns():
    import csv
    import io

    from omicron.csv_batch import row_to_details

    row = next(csv.DictReader(io.StringIO(CSV_HEADER + CSV_ROW)))
    details = row_to_details(row)

    assert details.name == "Edelstahl-Trinkflasche Aqua 750 ml"
    assert details.category == "Sport & Outdoor"
    assert details.features == ["750 ml", "hält 24 h kalt", "BPA-frei"]
    assert (details.price, details.currency) == (24.9, "EUR")
    assert details.target_audience == "Sportler und Wanderer"
    # Nicht vorhandene Angaben werden leer gelassen, nicht erfunden.
    assert details.dimensions == "" and details.colors == [] and details.additional_info == ""


def test_cli_csv_fills_output_column(monkeypatch, tmp_path):
    import csv

    path = tmp_path / "produkte.csv"
    done_row = "Fertig,Test,Holz,a,1 €,alle,schon da\n"
    path.write_text(CSV_HEADER + CSV_ROW + done_row, encoding="utf-8")
    client = FakeClient(DESCRIPTION)
    monkeypatch.setattr("omicron.__main__.anthropic.Anthropic", lambda: client)

    assert main(["--csv", str(path)]) == 0

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows[0]["output"].startswith("# Trocken ins Büro")
    assert rows[0]["preis"] == "24.90 €"
    # Zeilen mit vorhandener Ausgabe werden nicht erneut angefragt.
    assert rows[1]["output"] == "schon da"
    assert len(client.calls) == 1
    assert "Edelstahl-Trinkflasche" in client.calls[0]["messages"][0]["content"]


def test_cli_csv_json_to_separate_output(monkeypatch, tmp_path):
    import csv

    source = tmp_path / "in.csv"
    target = tmp_path / "out.csv"
    source.write_text(CSV_HEADER + CSV_ROW, encoding="utf-8")
    monkeypatch.setattr("omicron.__main__.anthropic.Anthropic", lambda: FakeClient(DESCRIPTION))

    assert main(["--csv", str(source), "--output", str(target), "--json"]) == 0

    assert source.read_text(encoding="utf-8") == CSV_HEADER + CSV_ROW
    row = next(csv.DictReader(target.open(encoding="utf-8")))
    assert json.loads(row["output"]) == DESCRIPTION


def test_cli_csv_reports_failed_rows(monkeypatch, tmp_path):
    import csv

    path = tmp_path / "produkte.csv"
    path.write_text(CSV_HEADER + CSV_ROW + CSV_ROW.replace("Aqua", "Beta"), encoding="utf-8")
    client = FakeClient({"headline": "kaputt"}, DESCRIPTION)
    monkeypatch.setattr("omicron.__main__.anthropic.Anthropic", lambda: client)

    assert main(["--csv", str(path)]) == 1

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows[0]["output"] == ""
    assert rows[1]["output"].startswith("# Trocken ins Büro")
