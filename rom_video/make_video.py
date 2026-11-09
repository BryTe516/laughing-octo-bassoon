"""Erzeugt ein ~20-sekündiges Video über die Expansion Roms
mit Fokus auf die Eroberung der griechischen / hellenistischen Welt
und Hinweisen zur Christianisierung.

Aufruf:  python make_video.py            -> rom_griechenland.mp4
Benötigt: matplotlib, shapely, numpy, ffmpeg
"""
import json
import math
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle, Polygon as MplPolygon, FancyBboxPatch
from matplotlib.path import Path as MplPath
import numpy as np
from shapely.geometry import shape, Polygon, Point
from shapely.prepared import prep

HERE = Path(__file__).parent
OUT = HERE / "rom_griechenland.mp4"
FPS = 30
W, H = 1280, 720
DPI = 100

# ---------------------------------------------------------------- Farben
SEA = "#1f3b52"
LAND = "#e9dfc6"
LAND_EDGE = "#b9a985"
ROME = "#c0392b"          # römisch
ROME_GREEK = "#6e1810"    # römisch, ehemals griechisch/hellenistisch (dunkler)
GREEK_FREE = "#86b2d6"    # griechisch/hellenistisch, noch nicht römisch
LOST_WEST = "#8c8579"     # nach 476: Germanenreiche
GOLD = "#f4c542"
CHRIST = "#ffffff"
TEXT = "#f7f1e3"

# ---------------------------------------------------------------- Kartenausschnitt
LON0, LON1 = -14.0, 54.0
LAT0 = 25.5
LAT1 = LAT0 + (LON1 - LON0) * (H / W) / (1 / math.cos(math.radians(40)))

land = shape(json.load(open(HERE / "data/land_mittelmeer.geojson"))["geometry"])

# ---------------------------------------------------------------- Regionen
# (stark vereinfachte Umrisse, lon/lat) -> werden mit der Landmasse verschnitten
# greek=True: ehemals griechisch/hellenistisches Gebiet
R = {
    "latium":      ([(11.6, 42.5), (13.6, 42.5), (13.9, 41.1), (12.2, 41.1)], False, "Latium"),
    "italia":      ([(9.6, 44.3), (12.4, 44.6), (14.2, 42.4), (16.3, 41.9), (15.5, 40.8), (14.8, 40.3),
                     (13.0, 40.8), (11.0, 42.3), (9.8, 43.8)], False, "Italien"),
    "magna":       ([(14.8, 40.3), (15.5, 40.8), (16.3, 41.9), (18.8, 40.6), (18.8, 39.5), (17.3, 37.7),
                     (15.4, 37.7), (15.4, 38.9)], True, "Magna Graecia"),
    "sicilia":     ([(12.0, 36.4), (15.9, 36.4), (15.9, 38.5), (12.0, 38.5)], True, "Sizilien"),
    "sardinia":    ([(7.8, 38.7), (10.2, 38.7), (10.2, 43.2), (7.8, 43.2)], False, "Sardinien"),
    "cisalpina":   ([(6.6, 44.0), (7.0, 46.2), (10.0, 46.6), (13.8, 46.4), (13.6, 45.5), (12.4, 44.6),
                     (9.6, 44.3), (8.0, 43.7)], False, "Gallia Cisalpina"),
    "hispania1":   ([(-9.0, 35.9), (-5.0, 35.9), (0.0, 38.0), (3.6, 41.8), (3.4, 42.6), (0.5, 42.0),
                     (-2.5, 40.0), (-5.0, 38.5), (-7.5, 37.8)], False, "Hispania"),
    "hispania2":   ([(-10.0, 35.9), (3.6, 35.9), (3.6, 44.0), (-10.0, 44.0)], False, "Hispania"),
    "illyria":     ([(13.6, 45.5), (15.0, 44.0), (19.6, 41.6), (20.6, 41.8), (19.6, 43.5), (17.0, 45.0),
                     (16.0, 46.4), (13.8, 46.4)], False, "Illyricum"),
    "macedonia":   ([(19.0, 41.9), (19.2, 40.1), (21.0, 39.8), (23.0, 39.9), (24.2, 40.4), (26.6, 40.6),
                     (26.6, 41.6), (23.5, 42.2), (21.0, 42.4)], True, "Macedonia"),
    "achaia":      ([(19.2, 40.1), (21.0, 39.8), (23.0, 39.9), (24.6, 38.6), (26.6, 37.6), (26.6, 36.2),
                     (21.0, 36.2), (19.2, 38.6)], True, "Achaia"),
    "africa":      ([(8.0, 37.6), (11.6, 37.6), (11.4, 35.0), (10.4, 33.6), (8.0, 33.8)], False, "Africa"),
    "asia":        ([(25.8, 41.5), (25.8, 36.2), (29.6, 36.0), (30.4, 38.0), (31.2, 39.6), (29.6, 40.6),
                     (28.0, 40.5)], True, "Asia (Pergamon)"),
    "narbonensis": ([(3.0, 42.3), (3.0, 43.1), (1.5, 43.3), (1.8, 44.3), (4.5, 45.5), (6.0, 46.3),
                     (7.1, 45.9), (7.6, 44.0), (6.0, 43.0)], False, "Gallia Narbonensis"),
    "cyrenaica":   ([(19.5, 30.2), (19.5, 31.5), (20.5, 33.2), (23.5, 33.2), (25.2, 32.2), (25.2, 30.2)],
                    True, "Kyrene"),
    "creta":       ([(23.3, 34.7), (26.5, 34.7), (26.5, 35.8), (23.3, 35.8)], True, "Kreta"),
    "anatolia":    ([(29.6, 40.6), (31.2, 39.6), (30.4, 38.0), (29.6, 36.0), (36.4, 36.0), (36.4, 37.2),
                     (41.0, 40.5), (41.6, 41.6), (36.0, 42.2), (31.0, 41.5)], True, "Kleinasien"),
    "syria":       ([(35.6, 36.9), (36.4, 36.0), (36.4, 37.2), (38.2, 37.2), (40.2, 36.6), (38.6, 34.0),
                     (36.6, 33.0), (35.0, 33.0)], True, "Syria"),
    "judaea":      ([(34.0, 31.4), (35.0, 33.0), (36.6, 33.0), (36.2, 31.0), (35.0, 29.4), (34.3, 31.0)],
                    False, "Judäa"),
    "gallia":      ([(-5.5, 48.5), (-1.6, 43.3), (1.8, 44.3), (4.5, 45.5), (6.0, 46.3), (8.5, 47.6),
                     (7.6, 49.5), (6.4, 51.7), (4.0, 52.0), (2.0, 51.3), (-2.0, 49.0)], False, "Gallien"),
    "africa_nova": ([(1.5, 36.9), (8.0, 37.6), (8.0, 33.8), (1.5, 34.2)], False, "Numidien"),
    "aegyptus":    ([(24.8, 31.8), (34.2, 31.5), (32.7, 29.5), (33.4, 27.0), (34.6, 24.0), (31.5, 24.0),
                     (30.6, 27.0), (29.4, 29.0), (24.8, 30.0)], True, "Ägypten"),
    "danube":      ([(7.0, 46.2), (7.5, 47.7), (10.0, 47.9), (13.0, 48.8), (17.0, 48.3), (19.0, 47.9),
                     (20.0, 45.2), (23.0, 44.1), (28.0, 44.4), (29.8, 45.3), (28.6, 43.0), (26.6, 42.3),
                     (23.5, 42.2), (21.0, 42.4), (19.6, 43.5), (17.0, 45.0), (16.0, 46.4), (13.8, 46.4),
                     (10.0, 46.6)], False, "Donauprovinzen"),
    "thracia":     ([(23.5, 42.2), (26.6, 42.3), (28.6, 43.0), (29.4, 41.2), (26.6, 40.6), (26.6, 41.6)],
                    False, "Thrakien"),
    "mauretania":  ([(-9.8, 35.9), (-5.8, 35.9), (-2.0, 35.1), (1.5, 36.9), (1.5, 34.2), (-6.0, 33.0),
                     (-9.9, 32.5)], False, "Mauretanien"),
    "britannia":   ([(-6.0, 49.8), (2.0, 49.8), (2.0, 55.2), (-6.0, 55.2)], False, "Britannien"),
    "dacia":       ([(21.4, 45.5), (22.0, 47.6), (24.6, 47.9), (26.2, 46.5), (26.0, 45.2), (23.0, 44.3)],
                    False, "Dakien"),
    "arabia":      ([(35.0, 29.4), (36.2, 31.0), (37.6, 32.6), (38.6, 30.4), (36.8, 27.8), (35.0, 27.8)],
                    False, "Arabia"),
    "mesopot":     ([(38.2, 37.2), (42.0, 37.8), (46.0, 39.6), (48.2, 38.4), (47.2, 31.5), (44.0, 30.8),
                     (40.2, 36.6)], False, "Mesopotamien"),
}

GEOM = {}
for k, (coords, greek, label) in R.items():
    GEOM[k] = Polygon(coords).intersection(land)

GREEK = {k for k, v in R.items() if v[1]}
WEST = {"latium", "italia", "magna", "sicilia", "sardinia", "cisalpina", "hispania1", "hispania2",
        "africa", "narbonensis", "gallia", "africa_nova", "mauretania", "britannia", "illyria"}
# 476 fällt der Westen; der Osten (griechisch) bleibt
FALLEN_476 = WEST - {"illyria"}
EAST_CHRIST_300 = {"asia", "anatolia", "syria", "aegyptus", "africa", "achaia", "macedonia", "latium",
                   "judaea", "cyrenaica", "thracia"}

# ---------------------------------------------------------------- Orte
CITY = {
    "Rom": (12.5, 41.9), "Athen": (23.73, 37.98), "Korinth": (22.88, 37.9), "Karthago": (10.32, 36.85),
    "Syrakus": (15.29, 37.07), "Tarent": (17.24, 40.47), "Alexandria": (29.9, 31.2),
    "Pergamon": (27.18, 39.13), "Antiochia": (36.16, 36.2), "Jerusalem": (35.23, 31.78),
    "Konstantinopel": (28.98, 41.01), "Ephesos": (27.35, 37.94), "Thessaloniki": (22.95, 40.64),
    "Philippi": (24.29, 41.01), "Tarsos": (34.9, 36.92), "Nikaia": (29.72, 40.43), "Mailand": (9.19, 45.46),
    "Kynoskephalai": (22.6, 39.4), "Pydna": (22.6, 40.4), "Aktium": (20.75, 38.95), "Ravenna": (12.2, 44.4),
}

LABEL_POS = {"achaia": (21.4, 38.9), "macedonia": (21.6, 41.4), "asia": (29.0, 39.0),
             "anatolia": (34.5, 39.2), "aegyptus": (30.5, 27.5), "magna": (16.4, 39.2)}
LABEL_OFF = {"Korinth": (-0.35, -0.55, "right"), "Thessaloniki": (-0.3, 0.35, "right"),
             "Jerusalem": (-0.35, 0.0, "right"), "Kynoskephalai": (-0.35, 0.3, "right")}

# ---------------------------------------------------------------- Ablauf
# year: Anzeige, title, note, add/remove: Regionen, icon, battle: Orte (⚔),
# cities: Orte, die hervorgehoben werden, christ: Christianisierungs-Stufe (0..4)
PHASES = [
    dict(year="753 v. Chr.", title="Gründung Roms", note="Der Sage nach durch Romulus.\nDie Griechen siedeln schon in Süditalien.",
         add=["latium"], icon="wolf", cities=["Rom", "Tarent", "Syrakus"], t=-753),
    dict(year="272 v. Chr.", title="Italien wird römisch", note="Tarent fällt: Rom erobert die\ngriechischen Städte Süditaliens.",
         add=["italia", "magna"], icon="temple", cities=["Tarent"], battle=["Tarent"], t=-272),
    dict(year="241–212 v. Chr.", title="Sizilien & Syrakus", note="1. Punischer Krieg; 212 fällt Syrakus –\nArchimedes stirbt.",
         add=["sicilia", "sardinia"], icon="temple", cities=["Syrakus"], battle=["Syrakus"], t=-241),
    dict(year="197 v. Chr.", title="Kynoskephalai", note="Rom schlägt Makedonien.\n196: »Freiheit der Griechen« verkündet.",
         add=["cisalpina", "hispania1"], icon="swords", battle=["Kynoskephalai"], t=-197),
    dict(year="168 v. Chr.", title="Schlacht bei Pydna", note="Ende des makedonischen Königreichs.",
         add=["illyria"], icon="swords", battle=["Pydna"], t=-168),
    dict(year="146 v. Chr.", title="Korinth zerstört", note="Makedonien & Griechenland (Achaia)\nwerden römisch – Karthago fällt.",
         add=["macedonia", "achaia", "africa"], icon="temple", cities=["Korinth", "Athen", "Karthago"],
         battle=["Korinth", "Karthago"], t=-146, hold=1.6),
    dict(year="133 v. Chr.", title="Pergamon vererbt", note="Attalos III. vermacht sein Reich Rom\n→ Provinz Asia.",
         add=["asia", "narbonensis"], icon="temple", cities=["Pergamon"], t=-133),
    dict(year="86–67 v. Chr.", title="Sulla erobert Athen", note="»Graecia capta ferum victorem cepit« –\ndas besiegte Griechenland eroberte Rom kulturell.",
         add=["cyrenaica", "creta"], icon="temple", cities=["Athen"], battle=["Athen"], t=-86, hold=1.5),
    dict(year="64–63 v. Chr.", title="Pompeius im Osten", note="Ende des Seleukidenreichs:\nSyrien, Pontos, Kilikien.",
         add=["anatolia", "syria", "judaea"], icon="swords", cities=["Antiochia"], t=-64),
    dict(year="58–50 v. Chr.", title="Caesar erobert Gallien", note="Rom greift nach Nordwesten aus.",
         add=["gallia", "africa_nova"], icon="swords", t=-55),
    dict(year="31–30 v. Chr.", title="Aktium & Ägypten", note="Tod Kleopatras: das letzte\nhellenistische Reich wird römisch.",
         add=["aegyptus", "hispania2"], icon="temple", cities=["Alexandria"], battle=["Aktium"], t=-30, hold=1.4),
    dict(year="~30–60 n. Chr.", title="Jesus & Paulus", note="Paulus missioniert in griechischen Städten –\ndas Neue Testament wird griechisch geschrieben.",
         add=["danube"], icon="chirho", cities=["Jerusalem", "Antiochia", "Ephesos", "Korinth", "Philippi", "Thessaloniki", "Athen", "Rom"],
         christ=1, t=45, hold=1.7),
    dict(year="117 n. Chr.", title="Größte Ausdehnung", note="Unter Kaiser Trajan.",
         add=["britannia", "mauretania", "thracia", "dacia", "arabia", "mesopot"], icon="eagle", christ=1, t=117),
    dict(year="~300 n. Chr.", title="Christen im Osten", note="Etwa 10 % Christen – am stärksten\nim griechischsprachigen Osten.",
         remove=["mesopot", "dacia"], icon="chirho", christ=2, cities=["Alexandria", "Antiochia", "Karthago"], t=300, hold=1.4),
    dict(year="313 / 330 n. Chr.", title="Konstantin", note="Toleranzedikt von Mailand;\nKonstantinopel wird neue Hauptstadt.",
         icon="chirho", christ=2, cities=["Mailand", "Konstantinopel", "Nikaia"], t=320),
    dict(year="380 n. Chr.", title="Staatsreligion", note="Theodosius: Das Christentum wird\nStaatsreligion im ganzen Reich.",
         icon="chirho", christ=3, cities=["Konstantinopel"], t=380),
    dict(year="395 n. Chr.", title="Reichsteilung", note="Ostrom spricht Griechisch –\ndas Erbe Griechenlands lebt weiter.",
         icon="eagle", christ=3, split=True, cities=["Konstantinopel", "Rom"], t=395),
    dict(year="476 n. Chr.", title="Ende Westroms", note="Ostrom (Byzanz) besteht\nbis 1453 weiter.",
         icon="eagle", christ=3, split=True, fall=True, cities=["Konstantinopel", "Ravenna"], t=476, hold=1.6),
]

TITLE_SEC = 1.0
TARGET_SEC = 20.0
base = [p.get("hold", 1.0) for p in PHASES]
scale = (TARGET_SEC - TITLE_SEC) / sum(base)
for p, b in zip(PHASES, base):
    p["dur"] = b * scale

# Zustände je Phase vorausberechnen
owned = []
cur = set()
for p in PHASES:
    prev = set(cur)
    cur |= set(p.get("add", []))
    cur -= set(p.get("remove", []))
    owned.append((prev, set(cur)))


# ---------------------------------------------------------------- Hilfen
def geom_to_path(g):
    verts, codes = [], []
    polys = [g] if g.geom_type == "Polygon" else [x for x in getattr(g, "geoms", []) if x.geom_type == "Polygon"]
    for poly in polys:
        for ring in [poly.exterior, *poly.interiors]:
            c = np.asarray(ring.coords)
            if len(c) < 3:
                continue
            verts.extend(c.tolist())
            codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(c) - 2) + [MplPath.CLOSEPOLY])
    if not verts:
        return None
    return MplPath(verts, codes)


PATHS = {k: geom_to_path(g) for k, g in GEOM.items()}
LAND_PATH = geom_to_path(land)


def grid_points(g, step=1.25, offset=0.0):
    pg = prep(g)
    pts = []
    for x in np.arange(LON0 + offset, LON1, step):
        for y in np.arange(LAT0 + offset, LAT1, step * 0.8):
            xx = x + (0.5 * step if int(round((y - LAT0) / (step * 0.8))) % 2 else 0)
            if pg.contains(Point(xx, y)):
                pts.append((xx, y))
    return pts


CHRIST_PTS = {k: grid_points(g) for k, g in GEOM.items()}


def mix(c1, c2, t):
    a = np.array(matplotlib.colors.to_rgb(c1))
    b = np.array(matplotlib.colors.to_rgb(c2))
    return tuple(a + (b - a) * t)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def region_color(k, conquered):
    if not conquered:
        return GREEK_FREE
    return ROME_GREEK if k in GREEK else ROME


# ---------------------------------------------------------------- Icons
def draw_icon(ax, kind, cx, cy, s, alpha):
    """Kleine Piktogramme (in Achsen-Koordinaten 0..1)."""
    kw = dict(transform=ax.transAxes, alpha=alpha, zorder=30)
    if kind == "temple":
        # griechischer Tempel
        ax.add_patch(MplPolygon([(cx - s, cy + 0.35 * s), (cx + s, cy + 0.35 * s), (cx, cy + 0.85 * s)],
                                closed=True, fc=TEXT, ec="none", **kw))
        ax.add_patch(Rectangle((cx - s, cy + 0.22 * s), 2 * s, 0.1 * s, fc=TEXT, **kw))
        for i in range(5):
            x = cx - 0.85 * s + i * 0.425 * s
            ax.add_patch(Rectangle((x - 0.07 * s, cy - 0.55 * s), 0.14 * s, 0.75 * s, fc=TEXT, **kw))
        ax.add_patch(Rectangle((cx - 1.1 * s, cy - 0.72 * s), 2.2 * s, 0.14 * s, fc=TEXT, **kw))
    elif kind == "swords":
        ax.text(cx, cy, "⚔", fontsize=40, ha="center", va="center", color=TEXT, **kw)
    elif kind == "chirho":
        ax.text(cx, cy, "☧", fontsize=44, ha="center", va="center", color=GOLD, **kw)
    elif kind == "eagle":
        ax.text(cx, cy + 0.1 * s, "SPQR", fontsize=17, ha="center", va="center", color=GOLD,
                fontweight="bold", family="DejaVu Serif", **kw)
        ax.add_patch(Rectangle((cx - s, cy - 0.45 * s), 2 * s, 0.06 * s, fc=GOLD, **kw))
    elif kind == "wolf":
        ax.text(cx, cy + 0.05 * s, "ROMA", fontsize=17, ha="center", va="center", color=GOLD,
                fontweight="bold", family="DejaVu Serif", **kw)


# ---------------------------------------------------------------- Rendering
fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1])


def text_box(x, y, w, h, alpha=0.82):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                transform=ax.transAxes, fc="#10202e", ec="#5d7487", lw=1, alpha=alpha, zorder=20))


def draw_frame(pi, lt, global_t):
    """pi: Phasenindex (-1 = Titel), lt: lokale Zeit 0..1 in der Phase."""
    ax.clear()
    ax.set_xlim(LON0, LON1)
    ax.set_ylim(LAT0, LAT1)
    ax.axis("off")
    ax.add_patch(Rectangle((LON0, LAT0), LON1 - LON0, LAT1 - LAT0, fc=SEA, zorder=0))
    ax.add_patch(PathPatch(LAND_PATH, fc=LAND, ec=LAND_EDGE, lw=0.5, zorder=1))

    p = PHASES[max(pi, 0)]
    prev, now = owned[max(pi, 0)] if pi >= 0 else (set(), set())
    fade = ease(lt / 0.35)
    fall = p.get("fall", False) and pi >= 0

    # -- griechische Welt (noch nicht römisch)
    for k in GREEK:
        if k not in now:
            ax.add_patch(PathPatch(PATHS[k], fc=GREEK_FREE, ec="#4f7ea6", lw=0.6, alpha=0.9, zorder=2))

    # -- römische Gebiete
    for k in now | prev:
        path = PATHS[k]
        if path is None:
            continue
        if k in now and k not in prev:          # neu erobert
            start = GREEK_FREE if k in GREEK else LAND
            col = mix(start, region_color(k, True), fade)
            ax.add_patch(PathPatch(path, fc=col, ec="none", zorder=3))
            glow = 1 - ease((lt - 0.2) / 0.6)
            if glow > 0:
                ax.add_patch(PathPatch(path, fc="none", ec=GOLD, lw=2.2, alpha=glow, zorder=6))
        elif k in prev and k not in now:        # verloren
            col = mix(region_color(k, True), LAND, fade)
            ax.add_patch(PathPatch(path, fc=col, ec="none", zorder=3))
        else:
            col = region_color(k, True)
            if fall and k in FALLEN_476:
                col = mix(col, LOST_WEST, fade)
            ax.add_patch(PathPatch(path, fc=col, ec="none", zorder=3))

    # Labels für neu eroberte griechische Gebiete
    for k in now - prev:
        if k in GREEK:
            c = LABEL_POS.get(k) or GEOM[k].representative_point().coords[0]
            ax.text(c[0], c[1], R[k][2], fontsize=8.5, color="white", ha="center", va="center",
                    fontweight="bold", alpha=ease(lt / 0.3), zorder=8,
                    bbox=dict(boxstyle="round,pad=0.15", fc="#00000055", ec="none"))

    # -- Christianisierung (weiße Kreuze)
    stage = p.get("christ", 0)
    prev_stage = PHASES[pi - 1].get("christ", 0) if pi > 0 else 0
    if stage >= 2:
        regions = EAST_CHRIST_300 if stage == 2 else now
        prev_regions = (EAST_CHRIST_300 if prev_stage == 2 else (now if prev_stage >= 3 else set()))
        for k in regions & now:
            a = 0.85 if k in prev_regions else 0.85 * fade
            pts = CHRIST_PTS[k]
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, marker="+", s=34, lw=1.3, c=CHRIST, alpha=a, zorder=7)

    # -- Teilungslinie 395
    if p.get("split") and pi >= 0:
        a = 1.0 if PHASES[pi - 1].get("split") else fade
        ax.plot([19.2, 19.2, 18.6, 19.6, 20.0, 19.2], [48.5, 45.0, 43.0, 42.0, 38.0, 30.0],
                color=GOLD, lw=2.4, ls=(0, (5, 3)), alpha=a, zorder=9)
        ax.text(8.0, 48.6 if not fall else 48.6, "Westrom" if not fall else "Germanenreiche",
                fontsize=13, color=TEXT, ha="center", fontweight="bold", alpha=a, zorder=9,
                bbox=dict(boxstyle="round,pad=0.25", fc="#00000066", ec="none"))
        ax.text(33.0, 43.6, "Ostrom (griechisch)", fontsize=13, color=TEXT, ha="center",
                fontweight="bold", alpha=a, zorder=9, bbox=dict(boxstyle="round,pad=0.25", fc="#00000066", ec="none"))

    # -- Städte
    shown = {"Rom", "Athen"} | set(p.get("cities", []) if pi >= 0 else [])
    if stage >= 1:
        shown |= {"Jerusalem", "Antiochia", "Ephesos", "Korinth", "Thessaloniki"}
    for name in shown:
        x, y = CITY[name]
        hi = name in (p.get("cities", []) if pi >= 0 else [])
        is_christ = stage >= 1 and name in {"Jerusalem", "Antiochia", "Ephesos", "Korinth", "Philippi",
                                             "Thessaloniki", "Athen", "Rom", "Tarsos", "Konstantinopel", "Nikaia",
                                             "Alexandria", "Karthago", "Mailand"}
        ax.scatter([x], [y], s=26 if hi else 14, c=GOLD if is_christ else "#1b1b1b",
                   edgecolors="white", linewidths=0.8, zorder=10)
        if is_christ and hi:
            r = 0.4 + 1.2 * ease(lt / 0.6)
            ax.scatter([x], [y], s=(r * 22) ** 2, facecolors="none", edgecolors=GOLD,
                       linewidths=1.2, alpha=max(0, 1 - lt * 1.3), zorder=9)
        if hi or name in ("Rom", "Athen"):
            if name == "Philippi":
                continue
            dx, dy, ha = LABEL_OFF.get(name, (0.35, 0.3, "left"))
            ax.text(x + dx, y + dy, name, ha=ha, fontsize=8.5 if hi else 7.5, color="white", zorder=11,
                    fontweight="bold" if hi else "normal",
                    bbox=dict(boxstyle="round,pad=0.1", fc="#00000080", ec="none"))
    for name in (p.get("battle", []) if pi >= 0 else []):
        x, y = CITY[name]
        sc = 1 + 0.6 * (1 - ease(lt / 0.3))
        ax.text(x, y - 0.75, "⚔", fontsize=15 * sc, color=GOLD, ha="center", va="center", zorder=12,
                path_effects=[matplotlib.patheffects.withStroke(linewidth=2.5, foreground="black")])

    # -- Jahreszahl (oben rechts)
    if pi >= 0:
        slide = 1 - ease(lt / 0.18)
        text_box(0.67, 0.80, 0.315, 0.175)
        ax.text(0.965, 0.915, p["year"], transform=ax.transAxes, fontsize=34 if len(p["year"]) <= 12 else 27, ha="right", va="center",
                color=GOLD, fontweight="bold", family="DejaVu Serif", zorder=25, alpha=1 - slide * 0.8)
        ax.text(0.965, 0.835, p["title"], transform=ax.transAxes, fontsize=17, ha="right", va="center",
                color=TEXT, zorder=25)

        # -- Info-Box (unten links)
        text_box(0.012, 0.15, 0.43, 0.15)
        draw_icon(ax, p["icon"], 0.055, 0.225, 0.028, 1.0)
        ax.text(0.10, 0.225, p["note"], transform=ax.transAxes, fontsize=11.5, color=TEXT, va="center",
                ha="left", zorder=25, linespacing=1.35)

    # -- Zeitleiste (unten)
    T0, T1 = -753, 476
    x0, x1, yb = 0.04, 0.96, 0.05
    ax.add_patch(Rectangle((0, 0), 1, 0.13, transform=ax.transAxes, fc="#0b1620", alpha=0.9, zorder=20))
    ax.plot([x0, x1], [yb, yb], transform=ax.transAxes, color="#5d7487", lw=3, zorder=21)
    if pi >= 0:
        t_prev = PHASES[pi - 1]["t"] if pi > 0 else T0
        tnow = t_prev + (p["t"] - t_prev) * ease(lt / 0.35) if pi > 0 else T0
        xn = x0 + (x1 - x0) * (tnow - T0) / (T1 - T0)
        ax.plot([x0, xn], [yb, yb], transform=ax.transAxes, color=ROME, lw=3, zorder=22)
        ax.scatter([xn], [yb], transform=ax.transAxes, s=90, c=GOLD, edgecolors="white", zorder=23)
    for tick, lab in [(-753, "753 v."), (-500, "500 v."), (-250, "250 v."), (0, "Chr. Geb."),
                      (250, "250 n."), (476, "476 n.")]:
        xt = x0 + (x1 - x0) * (tick - T0) / (T1 - T0)
        ax.plot([xt, xt], [yb - 0.01, yb + 0.01], transform=ax.transAxes, color="#9fb3c4", lw=1, zorder=22)
        ax.text(xt, yb - 0.028, lab, transform=ax.transAxes, fontsize=8, color="#9fb3c4", ha="center",
                va="center", zorder=22)
    # Legende in der Leiste
    items = [(ROME, "römisch"), (ROME_GREEK, "römisch, ehem. griechisch/hellenistisch"),
             (GREEK_FREE, "griechisch/hellenistisch (noch frei)")]
    lx = 0.04
    for c, lab in items:
        ax.add_patch(Rectangle((lx, 0.093), 0.018, 0.022, transform=ax.transAxes, fc=c, ec="white",
                               lw=0.5, zorder=26))
        ax.text(lx + 0.025, 0.104, lab, transform=ax.transAxes, fontsize=9.5, color=TEXT, va="center", zorder=26)
        lx += 0.04 + 0.006 * len(lab)
    ax.text(lx + 0.008, 0.104, "+", transform=ax.transAxes, fontsize=13, color=CHRIST, ha="center",
            va="center", fontweight="bold", zorder=26)
    ax.text(lx + 0.025, 0.104, "christianisiert", transform=ax.transAxes, fontsize=9.5, color=TEXT,
            va="center", zorder=26)

    # -- Titelkarte
    if pi < 0:
        a = 1 - ease((lt - 0.7) / 0.3)
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc="#0b1620", alpha=0.75 * a, zorder=40))
        ax.text(0.5, 0.58, "ROM & GRIECHENLAND", transform=ax.transAxes, fontsize=46, color=GOLD,
                ha="center", va="center", family="DejaVu Serif", fontweight="bold", alpha=a, zorder=41)
        ax.text(0.5, 0.46, "753 v. Chr. – 476 n. Chr.  ·  Eroberung & Christianisierung",
                transform=ax.transAxes, fontsize=18, color=TEXT, ha="center", va="center", alpha=a, zorder=41)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    return buf.tobytes()


import matplotlib.patheffects  # noqa: E402


def main():
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           "-movflags", "+faststart", str(OUT)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n_title = int(TITLE_SEC * FPS)
    for f in range(n_title):
        proc.stdin.write(draw_frame(-1, f / n_title, 0))
    total = n_title
    for pi, p in enumerate(PHASES):
        n = max(1, round(p["dur"] * FPS))
        for f in range(n):
            proc.stdin.write(draw_frame(pi, f / n, 0))
        total += n
        print(f"{p['year']:>18}  {n / FPS:4.2f}s")
    proc.stdin.close()
    proc.wait()
    print(f"fertig: {OUT}  ({total / FPS:.1f}s, {total} Frames)")


if __name__ == "__main__":
    main()
