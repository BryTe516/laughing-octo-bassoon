# Omicron – KI für Produktbeschreibungen (Mira-Modell)

Omicron erstellt aus Produktdetails fertige Produktbeschreibungen für Online-Shops.
Dafür arbeiten zwei spezialisierte KIs zusammen:

```
Produktidee ──► Produktdetail-KI ──► ProductDetails ──► Omicron (Mira) ──► Produktbeschreibung
               (Name, Features,       (strukturiert)     (Überschrift, Text,
                Material, Preis …)                         Bullet Points, SEO)
```

| Komponente | Datei | Aufgabe |
|---|---|---|
| **Produktdetail-KI** | `omicron/detail_ai.py` | Entwickelt aus einer Idee ein Produktdatenblatt (Name, Kategorie, Features, Material, Preis, Zielgruppe, Maße, Farben) |
| **Omicron** | `omicron/omicron.py` | Schreibt aus den Produktdetails die Produktbeschreibung |
| **Mira-Modell** | `omicron/mira.py` | Das Modellprofil hinter Omicron: Persönlichkeit, Schreibregeln, Basismodell und Denk-Einstellungen |

Mira ist ein spezialisiertes Modellprofil auf Basis von Claude (`claude-opus-5-5`).
Beide KIs liefern ihre Antworten als validiertes JSON (Structured Outputs). So hält
sich jede Ausgabe exakt an das Datenmodell in `omicron/schemas.py`. Mira bekommt die
Anweisung, keine Eigenschaften zu erfinden, die nicht in den Produktdetails stehen.

## Installation

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."   # API-Schlüssel von console.anthropic.com
```

## Verwendung

**Komplette Pipeline: Idee → Details → Beschreibung**

```bash
python -m omicron "wasserdichter Rucksack für Fahrradpendler"
```

**Eigene Produktdetails verwenden (nur Omicron)**

```bash
python -m omicron --details examples/rucksack.json
python -m omicron --details examples/rucksack.json --tone "locker und jung" --language Englisch --json
```

**In Python**

```python
from omicron import Omicron, ProductDetailAI

details = ProductDetailAI().generate("Keramik-Kaffeetasse mit Deckel")
beschreibung = Omicron().describe(details, tone="warm und hochwertig")
print(beschreibung.to_markdown())
```

## Mira anpassen

Alle Einstellungen des Modells stehen in `omicron/mira.py`. Eine eigene Variante
kannst du so erstellen:

```python
from omicron import Omicron, MiraModel

mira_pro = MiraModel(version="1.1-pro", effort="high")
Omicron(model=mira_pro).describe(details)
```

- `effort`: wie gründlich Mira nachdenkt (`low`, `medium`, `high`, `xhigh`, `max`)
- `system_prompt`: Schreibregeln und Persönlichkeit
- `base_model`: das zugrunde liegende Claude-Modell

## Tests

```bash
pip install pytest
python -m pytest
```

Die Tests ersetzen die API durch einen Fake-Client. Sie brauchen daher keinen
API-Schlüssel und kosten nichts.
