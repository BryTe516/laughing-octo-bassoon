"""Kommandozeile: Produktidee -> Produktdetails -> Produktbeschreibung."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anthropic

from . import MIRA, Omicron, OmicronError, ProductDetailAI, ProductDetails


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="omicron",
        description=f"Omicron ({MIRA.display_name}) – erstellt Produktbeschreibungen.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("idea", nargs="?", help="Produktidee, z. B. 'wasserdichter Pendler-Rucksack'")
    source.add_argument("--details", type=Path, help="JSON-Datei mit fertigen Produktdetails")
    parser.add_argument("--language", default="Deutsch", help="Sprache der Texte (Standard: Deutsch)")
    parser.add_argument("--tone", default="professionell und freundlich", help="Tonalität der Beschreibung")
    parser.add_argument("--json", action="store_true", help="Ausgabe als JSON statt Markdown")
    args = parser.parse_args(argv)

    try:
        client = anthropic.Anthropic()
        if args.details:
            details = ProductDetails.model_validate_json(args.details.read_text(encoding="utf-8"))
        else:
            print("Produktdetail-KI erstellt die Produktdetails …", file=sys.stderr)
            details = ProductDetailAI(client).generate(args.idea, language=args.language)
            print(details.model_dump_json(indent=2), file=sys.stderr)

        print(f"Omicron ({MIRA.display_name}) schreibt die Beschreibung …", file=sys.stderr)
        description = Omicron(client).describe(details, language=args.language, tone=args.tone)
    except (OmicronError, anthropic.APIError, ValueError) as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1

    print(description.model_dump_json(indent=2) if args.json else description.to_markdown())
    return 0


if __name__ == "__main__":
    sys.exit(main())
