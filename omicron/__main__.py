"""Kommandozeile: Produktidee -> Produktdetails -> Produktbeschreibung."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anthropic

from . import MIRA, Omicron, OmicronError, ProductDetailAI, ProductDetails
from .csv_batch import process_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="omicron",
        description=f"Omicron ({MIRA.display_name}) – erstellt Produktbeschreibungen.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("idea", nargs="?", help="Produktidee, z. B. 'wasserdichter Pendler-Rucksack'")
    source.add_argument("--details", type=Path, help="JSON-Datei mit fertigen Produktdetails")
    source.add_argument(
        "--csv",
        type=Path,
        nargs="?",
        const=Path("produkte.csv"),
        help="CSV-Datei mit Produkten (Standard: produkte.csv); schreibt eine Beschreibung pro Zeile",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("produkte_fertig.csv"),
        help="Zieldatei für --csv (Standard: produkte_fertig.csv)",
    )
    parser.add_argument("--language", default="Deutsch", help="Sprache der Texte (Standard: Deutsch)")
    parser.add_argument("--tone", default="professionell und freundlich", help="Tonalität der Beschreibung")
    parser.add_argument("--json", action="store_true", help="Ausgabe als JSON statt Markdown")
    args = parser.parse_args(argv)

    try:
        client = anthropic.Anthropic()
        if args.csv:
            failures = process_csv(
                args.csv, args.output, Omicron(client), language=args.language, tone=args.tone
            )
            print(f"Gespeichert in {args.output}", file=sys.stderr)
            if failures:
                print(f"{failures} Zeile(n) fehlgeschlagen, siehe Spalte 'output'.", file=sys.stderr)
            return 1 if failures else 0
        if args.details:
            details = ProductDetails.model_validate_json(args.details.read_text(encoding="utf-8"))
        else:
            print("Produktdetail-KI erstellt die Produktdetails …", file=sys.stderr)
            details = ProductDetailAI(client).generate(args.idea, language=args.language)
            print(details.model_dump_json(indent=2), file=sys.stderr)

        print(f"Omicron ({MIRA.display_name}) schreibt die Beschreibung …", file=sys.stderr)
        description = Omicron(client).describe(details, language=args.language, tone=args.tone)
    except (OmicronError, anthropic.APIError, ValueError, OSError) as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1

    print(description.model_dump_json(indent=2) if args.json else description.to_markdown())
    return 0


if __name__ == "__main__":
    sys.exit(main())
