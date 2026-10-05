"""Eine Kohonen-Karte ordnet Farben auf einem 10x10-Gitter.

Begleitprogramm zu Kapitel 12 "Kohonen-Karten: Ordnung ohne Lehrer".
Läuft mit Python 3.12 oder neuer, nur Standardbibliothek:

    python farbkarte.py                    # Lernen mit Textbild im Terminal
    python farbkarte.py --ppm karte.ppm    # zusätzlich ein Bild im PPM-Format schreiben

Jedes der 100 Neuronen hat drei Gewichte: einen Rot-, einen Grün- und einen Blauwert zwischen
0 und 1. Am Anfang sind sie zufällig, die Karte ist ein buntes Durcheinander. Dann sieht das Netz
5000 zufällige Farben, eine nach der anderen. Niemand sagt ihm, was richtig ist. Bei jeder Farbe
sucht es das Gewinner-Neuron (die ähnlichste Farbe auf der Karte) und zieht es samt seinen
Nachbarn ein Stück zu der gezeigten Farbe hin. Am Ende liegen ähnliche Farben nebeneinander.

Das Programm druckt die ersten Lernschritte im Einzelnen, danach alle 1000 Schritte einen
Zwischenstand und zum Schluss die Karte als Textbild: Jedes Feld zeigt den Anfangsbuchstaben der
Grundfarbe, der es am nächsten kommt.

Die Funktionen abstand, gewinner, nachbarschaft und lernschritt sind allgemein gehalten
(beliebig viele Werte je Neuron). gitter_flaeche.py nutzt sie für ein Gitter in der Ebene.
"""

import math
import random
import sys

ZEILEN, SPALTEN = 10, 10
SCHRITTE = 5000
LERNRATE_START, LERNRATE_ENDE = 0.5, 0.01
RADIUS_START, RADIUS_ENDE = 5.0, 0.5
STARTWERT = 12  # fester Startwert des Zufallsgenerators: jeder Lauf liefert dieselben Zahlen

GRUNDFARBEN = {  # Buchstabe für das Textbild: (Name, Rot, Grün, Blau)
    "S": ("Schwarz", 0, 0, 0), "W": ("Weiß", 1, 1, 1),
    "R": ("Rot", 1, 0, 0), "G": ("Grün", 0, 1, 0), "B": ("Blau", 0, 0, 1),
    "Z": ("Zitronengelb", 1, 1, 0), "T": ("Türkis", 0, 1, 1), "M": ("Magenta", 1, 0, 1),
}


# --- Kern: eine Kohonen-Karte lernt -------------------------------------------------------

def abstand(a, b):
    """Abstand zweier Punkte nach Pythagoras, für beliebig viele Werte."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def gewinner(karte, x):
    """Platz (Zeile, Spalte) des Neurons, dessen Gewichte der Eingabe x am nächsten liegen."""
    return min(karte, key=lambda platz: abstand(karte[platz], x))


def nachbarschaft(platz, sieger, radius):
    """Glocke: 1 beim Gewinner, nach außen schwächer, je nach Abstand auf dem Gitter."""
    d = abstand(platz, sieger)
    return math.exp(-d * d / (2 * radius * radius))


def lernschritt(karte, x, lernrate, radius):
    """Gewinner suchen, dann jedes Neuron um Lernrate * Nachbarschaft * (x - w) verschieben."""
    sieger = gewinner(karte, x)
    for platz, w in karte.items():
        h = nachbarschaft(platz, sieger, radius)
        karte[platz] = [wi + lernrate * h * (xi - wi) for wi, xi in zip(w, x)]
    return sieger


def zeitplan(t, schritte, start, ende):
    """Wert, der von start nach ende gleichmäßig (linear) abnimmt."""
    return start + (ende - start) * t / schritte


def neue_karte(zeilen, spalten, werte, zufall):
    """Gitter aus Neuronen mit zufälligen Gewichten zwischen 0 und 1."""
    return {(z, s): [zufall.random() for _ in range(werte)]
            for z in range(zeilen) for s in range(spalten)}


# --- Messgrößen und Ausgabe ---------------------------------------------------------------

def nachbarunterschied(karte):
    """Mittlerer Farbabstand zwischen direkt benachbarten Feldern (rechts und unten)."""
    paare = [(p, (p[0] + dz, p[1] + ds)) for p in karte for dz, ds in ((0, 1), (1, 0))]
    paare = [(p, q) for p, q in paare if q in karte]
    return sum(abstand(karte[p], karte[q]) for p, q in paare) / len(paare)


def treffabstand(karte, proben):
    """Mittlerer Abstand zwischen einer Farbe und ihrem Gewinner: Wie gut deckt die Karte alle Farben ab?"""
    return sum(abstand(karte[gewinner(karte, x)], x) for x in proben) / len(proben)


def textbild(karte):
    """Jedes Feld als Anfangsbuchstabe der nächstgelegenen Grundfarbe."""
    zeilen = []
    for z in range(ZEILEN):
        zeichen = [min(GRUNDFARBEN, key=lambda k: abstand(GRUNDFARBEN[k][1:], karte[(z, s)]))
                   for s in range(SPALTEN)]
        zeilen.append("   " + " ".join(zeichen))
    return "\n".join(zeilen)


def ppm_schreiben(karte, datei, pixel_je_feld=24):
    """Karte als einfaches Bild im Textformat PPM (P3) speichern."""
    breite, hoehe = SPALTEN * pixel_je_feld, ZEILEN * pixel_je_feld
    with open(datei, "w", encoding="ascii") as f:
        f.write(f"P3\n{breite} {hoehe}\n255\n")
        for y in range(hoehe):
            reihe = []
            for x in range(breite):
                r, g, b = karte[(y // pixel_je_feld, x // pixel_je_feld)]
                reihe.append(f"{round(255 * r)} {round(255 * g)} {round(255 * b)}")
            f.write(" ".join(reihe) + "\n")


def tabelle_rgb(karte):
    """Alle Felder als 0-255-Werte, Zeile für Zeile (Grundlage der Abbildung im Buch)."""
    return "\n".join(
        " ".join("{:3d},{:3d},{:3d}".format(*(round(255 * v) for v in karte[(z, s)]))
                 for s in range(SPALTEN))
        for z in range(ZEILEN))


# --- Hauptprogramm ------------------------------------------------------------------------

def main():
    zufall = random.Random(STARTWERT)
    karte = neue_karte(ZEILEN, SPALTEN, 3, zufall)
    proben = [[p.random() for _ in range(3)] for p in [random.Random(99)] for _ in range(500)]
    vorher = nachbarunterschied(karte)
    treff_vorher = treffabstand(karte, proben)
    print(f"Start: {ZEILEN}x{SPALTEN} Neuronen mit Zufallsfarben, "
          f"Nachbarunterschied {vorher:.3f}, Treffabstand {treff_vorher:.3f}")
    print(textbild(karte))
    print("\nRGB-Werte vorher (0 bis 255):")
    print(tabelle_rgb(karte))
    print()

    verlauf = []  # Treffabstand alle 1000 Schritte
    for t in range(SCHRITTE):
        lernrate = zeitplan(t, SCHRITTE, LERNRATE_START, LERNRATE_ENDE)
        radius = zeitplan(t, SCHRITTE, RADIUS_START, RADIUS_ENDE)
        x = [zufall.random() for _ in range(3)]
        if t < 3:
            sieger = gewinner(karte, x)
            w = karte[sieger]
            print(f"Schritt {t + 1}: Farbe ({x[0]:.2f}, {x[1]:.2f}, {x[2]:.2f})  "
                  f"Gewinner {sieger} mit ({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}), "
                  f"Abstand {abstand(w, x):.3f}")
        sieger = lernschritt(karte, x, lernrate, radius)
        if t < 3:
            w = karte[sieger]
            print(f"   Lernrate {lernrate:.3f}, Radius {radius:.3f} -> Gewinner jetzt "
                  f"({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f})")
        if (t + 1) % 1000 == 0:
            verlauf.append(treffabstand(karte, proben))
            print(f"nach {t + 1:5d} Schritten: Lernrate {lernrate:.3f}, Radius {radius:.2f}, "
                  f"Nachbarunterschied {nachbarunterschied(karte):.3f}, "
                  f"Treffabstand {verlauf[-1]:.3f}")

    nachher = nachbarunterschied(karte)
    treff_nachher = treffabstand(karte, proben)
    print(f"\nFertig. Nachbarunterschied {vorher:.3f} -> {nachher:.3f}, "
          f"Treffabstand {treff_vorher:.3f} -> {treff_nachher:.3f}")
    print(textbild(karte))
    print("   " + ", ".join(f"{k} = {v[0]}" for k, v in GRUNDFARBEN.items()))
    print("\nWo liegen die Grundfarben auf der Karte?")
    plaetze = {}
    for k, (name, *rgb) in GRUNDFARBEN.items():
        plaetze[k] = gewinner(karte, rgb)
        print(f"   {name:<13} -> Feld {plaetze[k]}")
    print("\nRGB-Werte nachher (0 bis 255):")
    print(tabelle_rgb(karte))

    # Proben: Die Karte hat sich geordnet.
    assert nachher < vorher / 3, "Nachbarfelder sollten sich nach dem Lernen viel ähnlicher sein"
    assert treff_nachher < verlauf[0] / 2, "die anfangs zusammengezogene Karte breitet sich wieder aus"
    assert treff_nachher < 1.2 * treff_vorher, "die geordnete Karte deckt die Farben fast so gut ab wie der Zufall"
    assert len(set(plaetze.values())) == len(GRUNDFARBEN), "jede Grundfarbe bekommt ein eigenes Feld"
    assert abstand(plaetze["S"], plaetze["W"]) > 5, "Schwarz und Weiß liegen weit auseinander"

    if "--ppm" in sys.argv:
        datei = sys.argv[sys.argv.index("--ppm") + 1]
        ppm_schreiben(karte, datei)
        print(f"\nBild gespeichert: {datei}")


if __name__ == "__main__":
    main()
