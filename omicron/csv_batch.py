"""Stapelverarbeitung: Produkte aus einer CSV-Datei lesen und Beschreibungen schreiben."""

from __future__ import annotations

import csv
import io
import re
import sys
from pathlib import Path

import anthropic

from ._llm import OmicronError
from .omicron import Omicron
from .schemas import ProductDetails

INPUT_COLUMNS = ["name", "kategorie", "material", "features", "preis", "zielgruppe"]
OUTPUT_COLUMN = "output"
UNKNOWN = "k. A."


def parse_features(value: str) -> list[str]:
    """Trennt die Features einer Zelle an ';' oder '|', sonst an ','."""
    separator = ";" if ";" in value else "|" if "|" in value else ","
    return [feature.strip() for feature in value.split(separator) if feature.strip()]


def parse_price(value: str) -> tuple[float, str]:
    """Liest Preise wie '89,90', '89.90 €' oder '1.299,00 EUR' als (Betrag, Währung)."""
    currency = "USD" if "$" in value or "USD" in value.upper() else "EUR"
    number = re.sub(r"[^\d,.]", "", value)
    if not number:
        raise ValueError(f"Ungültiger Preis: {value!r}")
    if "," in number:
        number = number.replace(".", "").replace(",", ".")
    return float(number), currency


def row_to_details(row: dict[str, str]) -> ProductDetails:
    """Übernimmt genau die Werte aus der Tabelle; alles andere bleibt unbekannt."""
    price, currency = parse_price(row["preis"])
    return ProductDetails(
        name=row["name"].strip(),
        category=row["kategorie"].strip() or UNKNOWN,
        features=parse_features(row["features"]),
        material=row["material"].strip() or UNKNOWN,
        price=price,
        currency=currency,
        target_audience=row["zielgruppe"].strip() or UNKNOWN,
        dimensions=UNKNOWN,
        colors=[],
        additional_info="",
    )


def process_csv(
    input_path: Path,
    output_path: Path,
    omicron: Omicron,
    *,
    language: str = "Deutsch",
    tone: str = "professionell und freundlich",
) -> int:
    """Schreibt für jede Zeile eine Beschreibung in die Spalte ``output``.

    Gibt die Anzahl der fehlgeschlagenen Zeilen zurück. Fehler stehen in der
    Spalte ``output`` der betroffenen Zeile; die übrigen Zeilen laufen weiter.
    """
    # utf-8-sig liest auch Dateien mit BOM, wie Excel sie speichert.
    text = input_path.read_text(encoding="utf-8-sig")
    try:
        dialect = csv.Sniffer().sniff(text.splitlines()[0], delimiters=",;\t")
    except (csv.Error, IndexError):
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text, newline=""), dialect=dialect)
    missing = [column for column in INPUT_COLUMNS if column not in (reader.fieldnames or [])]
    if missing:
        raise OmicronError(f"Fehlende Spalten in {input_path}: {', '.join(missing)}")
    rows = list(reader)

    failures = 0
    for number, row in enumerate(rows, start=1):
        print(f"[{number}/{len(rows)}] {row['name']}", file=sys.stderr)
        try:
            details = row_to_details(row)
            description = omicron.describe(details, language=language, tone=tone)
            row[OUTPUT_COLUMN] = description.to_markdown()
        except (OmicronError, anthropic.APIError, ValueError) as exc:
            failures += 1
            row[OUTPUT_COLUMN] = f"FEHLER: {exc}"

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[*INPUT_COLUMNS, OUTPUT_COLUMN],
            delimiter=dialect.delimiter,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    return failures
