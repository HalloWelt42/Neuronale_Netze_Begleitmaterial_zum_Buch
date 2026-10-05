"""Ein Gitter aus 8x8 Neuronen legt sich über ein Quadrat.

Begleitprogramm zu Kapitel 12 "Kohonen-Karten: Ordnung ohne Lehrer".
Läuft mit Python 3.12 oder neuer, nur Standardbibliothek:

    python gitter_flaeche.py           # Zwischenstände als Tabelle
    python gitter_flaeche.py --tikz    # zusätzlich die Punkte als TikZ-Koordinaten

Diesmal hat jedes Neuron nur zwei Gewichte, eine x- und eine y-Koordinate. Man kann es also als
Punkt in der Ebene zeichnen und die Gitternachbarn mit Linien verbinden. Die Eingaben sind
zufällige Punkte aus dem Quadrat von (0, 0) bis (1, 1). Anfangs liegen die Neuronen wild verstreut,
das gezeichnete Gitter ist ein Knäuel. Nach und nach entwirrt es sich und spannt sich über die
ganze Fläche. Die Abbildung "Gitter über einer Fläche" im Buch zeigt genau diese Zwischenstände.

Ein zweiter Lauf zeigt, was passiert, wenn die Nachbarschaft am Anfang zu klein ist: Das Gitter
ordnet sich nur stückweise und behält eine verdrehte Stelle.

Der Lernkern (abstand, lernschritt, zeitplan, neue_karte) stammt aus farbkarte.py.
"""

import random
import sys
from dataclasses import dataclass

from farbkarte import abstand, lernschritt, neue_karte, zeitplan

SEITE = 8
STARTWERT = 3


@dataclass(frozen=True)
class Einstellung:
    """Ein Lernlauf: wie lange, wie kräftig, wie weit die Nachbarschaft anfangs reicht."""
    name: str
    schritte: int
    lernrate: tuple[float, float]  # Start, Ende
    radius: tuple[float, float]    # Start, Ende
    stadien: tuple[int, ...]       # nach so vielen Schritten wird der Stand festgehalten


GUT = Einstellung("Startradius 3", 6000, (0.1, 0.01), (3.0, 0.3), (0, 50, 300, 6000))
ZU_ENG = Einstellung("Startradius 2, Lernrate 0,3", 4000, (0.3, 0.02), (2.0, 0.3), (0, 20, 200, 4000))


def kreuzungen(karte):
    """Zählt, wie oft sich zwei gezeichnete Gitterlinien schneiden (Maß für das Knäuel)."""
    linien = [(karte[p], karte[q]) for p in karte for q in ((p[0] + 1, p[1]), (p[0], p[1] + 1))
              if q in karte]

    def seite(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    zahl = 0
    for i, (a, b) in enumerate(linien):
        for c, d in linien[i + 1:]:
            if a in (c, d) or b in (c, d):
                continue  # Linien mit gemeinsamem Endpunkt zählen nicht
            if seite(a, b, c) * seite(a, b, d) < 0 and seite(c, d, a) * seite(c, d, b) < 0:
                zahl += 1
    return zahl


def ausdehnung(karte):
    """Breite und Höhe des Rechtecks, das alle Neuronen umschließt."""
    xs = [w[0] for w in karte.values()]
    ys = [w[1] for w in karte.values()]
    return max(xs) - min(xs), max(ys) - min(ys)


def tikz(karte, titel):
    """Punkte als TikZ-Liste: Zeile/Spalte/x/y, auf zwei Stellen gerundet."""
    eintraege = [f"{z}/{s}/{w[0]:.2f}/{w[1]:.2f}" for (z, s), w in sorted(karte.items())]
    return f"% {titel} Schritte\n" + ",\n".join(
        ", ".join(eintraege[i:i + SEITE]) for i in range(0, len(eintraege), SEITE))


def lernlauf(e):
    """Gitter zufällig verstreuen und lernen lassen; liefert die festgehaltenen Stände."""
    zufall = random.Random(STARTWERT)
    karte = neue_karte(SEITE, SEITE, 2, zufall)
    stand = {}
    print(f"Lauf mit {e.name}:")
    for t in range(e.schritte + 1):
        if t in e.stadien:
            stand[t] = {p: list(w) for p, w in karte.items()}
            breite, hoehe = ausdehnung(karte)
            print(f"   nach {t:4d} Schritten: {kreuzungen(karte):4d} Kreuzungen, "
                  f"Ausdehnung {breite:.2f} x {hoehe:.2f}, "
                  f"Ecke (0,0) bei ({karte[(0, 0)][0]:.2f}, {karte[(0, 0)][1]:.2f})")
        if t == e.schritte:
            break
        x = [zufall.random(), zufall.random()]
        lernschritt(karte, x, zeitplan(t, e.schritte, *e.lernrate), zeitplan(t, e.schritte, *e.radius))
    return stand


def main():
    gut = lernlauf(GUT)
    ende = gut[GUT.schritte]
    # Proben: aus dem Knäuel wird ein glattes Gitter, das das Quadrat fast ganz ausfüllt.
    assert kreuzungen(gut[0]) > 100, "am Anfang ist das Gitter verknäuelt"
    assert kreuzungen(ende) == 0, "am Ende kreuzen sich keine Gitterlinien mehr"
    assert min(ausdehnung(ende)) > 0.7, "das Gitter spannt sich über die Fläche"
    ecken = [ende[(z, s)] for z in (0, SEITE - 1) for s in (0, SEITE - 1)]
    assert min(abstand(a, b) for i, a in enumerate(ecken) for b in ecken[i + 1:]) > 0.7, \
        "die vier Gitterecken liegen in verschiedenen Ecken des Quadrats"

    # Gegenprobe: Reicht die Nachbarschaft anfangs nicht weit genug, bleibt ein Knoten im Gitter.
    eng = lernlauf(ZU_ENG)
    assert kreuzungen(eng[ZU_ENG.schritte]) > 0, "das zu eng gekoppelte Gitter bleibt verdreht"

    if "--tikz" in sys.argv:
        for e, stand in ((GUT, gut), (ZU_ENG, eng)):
            for t in e.stadien:
                print(tikz(stand[t], f"{e.name}, {t}"))


if __name__ == "__main__":
    main()
