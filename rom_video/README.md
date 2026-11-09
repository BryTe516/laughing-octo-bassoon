# Rom & Griechenland – Kartenvideo (erster Versuch)

`rom_griechenland.mp4`: etwa 20 s, 1280×720, 30 fps. Chronologie 753 v. Chr. bis 476 n. Chr.

- **Rot** = römisch, **dunkelrot** = römisch, aber vorher griechisch/hellenistisch,
  **hellblau** = griechisch/hellenistisch, noch nicht römisch
- **Weiße Kreuze / goldene Städte** = Christianisierung (Paulus, um 300, 380 Staatsreligion)
- Zeitleiste unten, Jahreszahl oben rechts, kurze Anmerkung mit Piktogramm unten links

Neu erzeugen: `pip install matplotlib shapely numpy` (ffmpeg wird benötigt), dann `python make_video.py`.
Die Provinzgrenzen sind bewusst stark vereinfacht. Ablauf, Texte und Dauer stehen in `PHASES`
in `make_video.py`.
Kartendaten: Natural Earth (gemeinfrei).
