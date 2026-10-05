"""Ein Modellneuron von Hand rechnen und seine Aktivierungsfunktionen ansehen.

Begleitprogramm zu Kapitel 4 "Das Modellneuron".
Läuft mit Python 3.12 oder neuer, nur Standardbibliothek:

    python modellneuron.py

Das Programm
1. rechnet das Freibad-Neuron aus dem Kapitel für alle acht Eingaben durch,
   einmal mit Treppenstufe, einmal mit S-Kurve,
2. prüft das Mehrheits-Neuron mit drei gleich starken Stimmen,
3. druckt eine Wertetafel der S-Kurve (auch mit steileren Varianten),
4. zeichnet die Aktivierungsfunktionen als kleine Textbilder,
5. zeigt, dass zwei hintereinander geschaltete lineare Neuronen
   wieder nur ein lineares Neuron sind.
"""

import math
from itertools import product

# --- Aktivierungsfunktionen: Netzeingabe s rein, Ausgabe raus ---------------


def stufe(s: float) -> float:
    """Treppenstufe: 1, sobald s größer als 0 ist, sonst 0."""
    return 1.0 if s > 0 else 0.0


def identitaet(s: float) -> float:
    """Gerade durch den Ursprung: die Summe wird unverändert weitergegeben."""
    return s


def rampe(s: float) -> float:
    """Stückweise linear: zwischen -0,5 und 0,5 schräg, außen flach bei 0 und 1."""
    return min(1.0, max(0.0, s + 0.5))


def sigmoid(s: float) -> float:
    """Logistische S-Kurve: weich zwischen 0 und 1, genau 0,5 bei s = 0."""
    return 1.0 / (1.0 + math.exp(-s))


def tanh(s: float) -> float:
    """S-Kurve zwischen -1 und 1."""
    return math.tanh(s)


def relu(s: float) -> float:
    """Gleichrichter: negative Werte werden 0, positive bleiben."""
    return max(0.0, s)


# --- Das Modellneuron ---------------------------------------------------------


def summe(eingaben: tuple[float, ...], gewichte: tuple[float, ...]) -> float:
    """Gewichtete Summe: jede Eingabe mal ihr Gewicht, alles zusammengezählt."""
    return sum(x * w for x, w in zip(eingaben, gewichte))


def neuron(eingaben, gewichte, schwelle, aktivierung=stufe) -> float:
    """Summe bilden, Schwelle abziehen, Aktivierungsfunktion anwenden."""
    return aktivierung(summe(eingaben, gewichte) - schwelle)


# Freibad-Neuron: Sonne, Freundin kommt mit, Mathearbeit morgen.
FREIBAD_GEWICHTE = (2, 3, -4)
FREIBAD_SCHWELLE = 4

# Mehrheits-Neuron: drei gleich starke Stimmen.
MEHRHEIT_GEWICHTE = (1, 1, 1)
MEHRHEIT_SCHWELLE = 1.5


def freibad_tafel() -> list[tuple[tuple[int, int, int], float, float, float]]:
    """Alle acht Eingaben durchrechnen: Summe, Stufe, S-Kurve."""
    zeilen = []
    print("Freibad-Neuron: Gewichte", FREIBAD_GEWICHTE, " Schwelle", FREIBAD_SCHWELLE)
    print(" Sonne Freundin Arbeit | Summe | Stufe | S-Kurve")
    for x in product((0, 1), repeat=3):
        s = summe(x, FREIBAD_GEWICHTE)
        a_stufe = neuron(x, FREIBAD_GEWICHTE, FREIBAD_SCHWELLE, stufe)
        a_sig = neuron(x, FREIBAD_GEWICHTE, FREIBAD_SCHWELLE, sigmoid)
        zeilen.append((x, s, a_stufe, a_sig))
        print(f"   {x[0]}      {x[1]}       {x[2]}    | {s:+5.1f} |  {a_stufe:.0f}    | {a_sig:.3f}")
    print()
    return zeilen


def mehrheit_pruefen() -> None:
    """Das Mehrheits-Neuron feuert genau dann, wenn mindestens zwei Eingänge 1 sind."""
    print("Mehrheits-Neuron: Gewichte", MEHRHEIT_GEWICHTE, " Schwelle", MEHRHEIT_SCHWELLE)
    for x in product((0, 1), repeat=3):
        a = neuron(x, MEHRHEIT_GEWICHTE, MEHRHEIT_SCHWELLE)
        print(f"  {x} -> {a:.0f}")
        assert a == (1.0 if sum(x) >= 2 else 0.0)
    print()


def sigmoid_wertetafel() -> None:
    """Wertetafel der S-Kurve und ihrer steileren Schwestern sigmoid(a * s)."""
    print("Wertetafel der S-Kurve")
    print("    s  | sig(s) | sig(2s) | sig(5s)")
    for s in (-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4):
        print(f" {s:+5.1f} | {sigmoid(s):.3f}  | {sigmoid(2 * s):.3f}   | {sigmoid(5 * s):.3f}")
    print()


def textbild(name: str, f, unten: float, oben: float, breite: int = 41, hoehe: int = 9) -> None:
    """Zeichnet f zwischen s = -4 und s = 4 als Textbild aus Sternchen."""
    raster = [[" "] * breite for _ in range(hoehe)]
    for spalte in range(breite):
        s = -4 + 8 * spalte / (breite - 1)
        y = min(oben, max(unten, f(s)))
        zeile = round((oben - y) / (oben - unten) * (hoehe - 1))
        raster[zeile][spalte] = "*"
    print(f"{name}  (s von -4 bis 4, Ausgabe von {unten:g} bis {oben:g})")
    for zeile in raster:
        print("  |" + "".join(zeile))
    print("  +" + "-" * breite)
    print()


def linear_hintereinander() -> None:
    """Zwei lineare Neuronen hintereinander sind wieder ein lineares Neuron."""
    print("Zwei lineare Neuronen hintereinander: erst mal 2, dann mal 3")
    for s in (-1, 0, 2, 5):
        zwei_stufen = 3 * identitaet(2 * identitaet(s))
        print(f"  s = {s:+d}:  3 * (2 * s) = {zwei_stufen:+.0f}  =  6 * s = {6 * s:+d}")
        assert zwei_stufen == 6 * s
    print()


if __name__ == "__main__":
    tafel = freibad_tafel()
    # Proben: nur Sonne und Freundin ohne Mathearbeit schicken dich ins Freibad.
    for x, s, a_stufe, a_sig in tafel:
        assert a_stufe == (1.0 if x == (1, 1, 0) else 0.0)
        assert (a_sig > 0.5) == (a_stufe == 1.0)
    assert abs(sigmoid(1) - 0.731) < 0.001
    assert sigmoid(0) == 0.5

    mehrheit_pruefen()
    sigmoid_wertetafel()
    # Proben zur S-Kurve: symmetrisch um den Punkt (0 | 0,5), tanh ist eine gestreckte S-Kurve.
    for s in (-3, -1, 0.5, 2):
        assert abs(sigmoid(-s) - (1 - sigmoid(s))) < 1e-12
        assert abs(tanh(s) - (2 * sigmoid(2 * s) - 1)) < 1e-12

    textbild("Treppenstufe", stufe, 0, 1)
    textbild("Rampe", rampe, 0, 1)
    textbild("S-Kurve (Sigmoid)", sigmoid, 0, 1)
    textbild("Tangens hyperbolicus", tanh, -1, 1)
    textbild("ReLU", relu, 0, 4)

    linear_hintereinander()
    print("Alle Proben bestanden.")
