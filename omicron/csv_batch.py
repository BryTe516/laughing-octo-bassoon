"""CSV-Modus: Produktdetails zeilenweise aus einer CSV lesen und Beschreibungen zurückschreiben."""

from __future__ import annotations

import csv
import re
import sys
from collections.abc import Callable
from pathlib import Path

from ._llm import OmicronError
from .omicron import Omicron
from .schemas import ProductDescription, ProductDetails

OUTPUT_COLUMN = "output"

# Spaltennamen in der CSV (deutsch oder englisch) -> Feld in ProductDetails.
COLUMN_ALIASES = {
    "name": "name",
    "kategorie": "category",
    "category": "category",
    "material": "material",
    "features": "features",
    "merkmale": "features",
    "preis": "price",
    "price": "price",
    "währung": "currency",
    "waehrung": "currency",
    "currency": "currency",
    "zielgruppe": "target_audience",
    "target_audience": "target_audience",
    "maße": "dimensions",
    "masse": "dimensions",
    "dimensions": "dimensions",
    "farben": "colors",
    "colors": "colors",
    "zusatzinfo": "additional_info",
    "additional_info": "additional_info",
}

LIST_FIELDS = {"features", "colors"}

CURRENCY_SYMBOLS = {"€": "EUR", "$": "USD", "£": "GBP"}
CURRENCY_CODES = {"EUR", "USD", "GBP", "CHF"}


def parse_price(value: str) -> tuple[float, str]:
    """Zerlegt z. B. ``"39.90 €"`` oder ``"1.299,00 EUR"`` in Betrag und Währungscode."""
    currency = ""
    for symbol, code in CURRENCY_SYMBOLS.items():
        if symbol in value:
            currency = code
    for code in CURRENCY_CODES:
        if re.search(rf"\b{code}\b", value, re.IGNORECASE):
            currency = code

    number = re.sub(r"[^0-9.,]", "", value)
    if not number:
        raise ValueError(f"Kein Preis erkennbar: {value!r}")
    if "," in number and "." in number:
        # Das zuletzt stehende Zeichen ist das Dezimaltrennzeichen.
        thousands = "." if number.rfind(",") > number.rfind(".") else ","
        number = number.replace(thousands, "")
    return float(number.replace(",", ".")), currency


def _split_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def row_to_details(row: dict[str, str]) -> ProductDetails:
    """Wandelt eine CSV-Zeile in ProductDetails um. Fehlende Angaben bleiben leer."""
    fields: dict[str, str] = {}
    for column, value in row.items():
        field = COLUMN_ALIASES.get((column or "").strip().lower())
        if field:
            fields[field] = (value or "").strip()

    if not fields.get("name"):
        raise ValueError("Spalte 'name' fehlt oder ist leer.")

    price, currency = parse_price(fields.get("price", ""))
    return ProductDetails(
        name=fields["name"],
        category=fields.get("category", ""),
        features=_split_list(fields.get("features", "")),
        material=fields.get("material", ""),
        price=price,
        currency=fields.get("currency") or currency,
        target_audience=fields.get("target_audience", ""),
        dimensions=fields.get("dimensions", ""),
        colors=_split_list(fields.get("colors", "")),
        additional_info=fields.get("additional_info", ""),
    )


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]], str]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        delimiter = csv.Sniffer().sniff(text.splitlines()[0] if text else "", delimiters=",;").delimiter
    except csv.Error:
        delimiter = ","
    reader = csv.DictReader(text.splitlines(keepends=True), delimiter=delimiter)
    fieldnames = list(reader.fieldnames or [])
    if "name" not in [name.strip().lower() for name in fieldnames]:
        raise ValueError(f"{path}: Spalte 'name' fehlt.")
    rows = list(reader)
    if OUTPUT_COLUMN not in fieldnames:
        fieldnames.append(OUTPUT_COLUMN)
    return fieldnames, rows, delimiter


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]], delimiter: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def describe_csv(
    input_path: Path,
    output_path: Path,
    omicron: Omicron,
    *,
    language: str = "Deutsch",
    tone: str = "professionell und freundlich",
    render: Callable[[ProductDescription], str] = ProductDescription.to_markdown,
) -> int:
    """Schreibt für jede Zeile ohne Ausgabe eine Beschreibung in die Spalte ``output``.

    Zeilen mit vorhandener Ausgabe werden übersprungen, sodass ein abgebrochener Lauf
    einfach erneut gestartet werden kann. Die Datei wird nach jeder Zeile gespeichert.
    Gibt die Anzahl der fehlgeschlagenen Zeilen zurück.
    """
    fieldnames, rows, delimiter = _read_csv(input_path)
    failures = 0
    for number, row in enumerate(rows, start=2):
        if (row.get(OUTPUT_COLUMN) or "").strip():
            continue
        label = row.get("name") or f"Zeile {number}"
        print(f"[{number - 1}/{len(rows)}] {label} …", file=sys.stderr)
        try:
            details = row_to_details(row)
            row[OUTPUT_COLUMN] = render(omicron.describe(details, language=language, tone=tone))
        except (OmicronError, ValueError) as exc:
            failures += 1
            print(f"Fehler in Zeile {number}: {exc}", file=sys.stderr)
            continue
        _write_csv(output_path, fieldnames, rows, delimiter)

    if failures or not output_path.exists():
        _write_csv(output_path, fieldnames, rows, delimiter)
    return failures
