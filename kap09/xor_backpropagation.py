"""Ein kleines Netz (2-2-1) lernt XOR mit Backpropagation.

Begleitprogramm zu Kapitel 9 "Backpropagation".
Läuft mit Python 3.12 oder neuer, nur Standardbibliothek:

    python xor_backpropagation.py

Teil 1: Ein einzelner Lernschritt mit den festen Gewichten aus dem Kapitel. Das Programm druckt
den Vorwärtsdurchlauf, die Fehlersignale des Rückwärtsdurchlaufs und alle neuen Gewichte, sodass
du das durchgerechnete Beispiel im Buch Zahl für Zahl nachprüfen kannst. Zur Probe vergleicht es
jede Steigung, die Backpropagation liefert, mit einem "Wackeltest": Gewicht ein winziges Stück
verstellen und nachmessen, wie sich der Fehler ändert.

Teil 2: Das Netz startet mit zufälligen Gewichten (fester Startwert, daher jedes Mal gleich) und
lernt XOR. Alle 500 Runden druckt es den Fehler, am Ende die vier Antworten.

Teil 3: Dasselbe mit 20 verschiedenen Startwerten. Nicht jeder Versuch gelingt: Manchmal bleibt
das Netz in einer Mulde der Fehlerlandschaft hängen. Darum geht es in Kapitel 10.
"""

import math
import random

XOR = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]
STARTWERT = 1
LERNRATE = 0.5
MAX_RUNDEN = 20000
TOLERANZ = 0.1  # gelernt, wenn jede Antwort höchstens 0,1 vom Sollwert entfernt ist


def sigma(z: float) -> float:
    """Die S-Kurve: 1 / (1 + e^(-z))."""
    return 1 / (1 + math.exp(-z))


# Ein Netz ist ein Wörterbuch mit zwei Gewichtslisten. Jede Liste endet mit dem Bias.
#   netz["verdeckt"][j] = [Gewicht von x1, Gewicht von x2, Bias]  für die verdeckten Neuronen A, B
#   netz["aus"]         = [Gewicht von A, Gewicht von B, Bias]     für das Ausgabeneuron


def vorwaerts(netz: dict, x: tuple[float, float]) -> tuple[list[float], float]:
    """Vorwärtsdurchlauf: Ausgaben der verdeckten Schicht und die Netzausgabe."""
    h = [sigma(w[0] * x[0] + w[1] * x[1] + w[2]) for w in netz["verdeckt"]]
    v = netz["aus"]
    o = sigma(v[0] * h[0] + v[1] * h[1] + v[2])
    return h, o


def fehler(netz: dict, x, soll) -> float:
    """Fehler für ein Beispiel: halbes Quadrat der Abweichung."""
    return 0.5 * (soll - vorwaerts(netz, x)[1]) ** 2


def fehlersignale(netz: dict, x, soll):
    """Rückwärtsdurchlauf ohne Gewichtsänderung: liefert h, o, delta_aus und die delta der Mitte."""
    h, o = vorwaerts(netz, x)
    d_aus = (soll - o) * o * (1 - o)  # Abweichung mal Steigung der S-Kurve
    d_h = [h[j] * (1 - h[j]) * netz["aus"][j] * d_aus for j in range(2)]  # zurückgereicht
    return h, o, d_aus, d_h


def lernschritt(netz: dict, x, soll, eta: float = LERNRATE) -> float:
    """Ein Beispiel vorwärts, Fehlersignal rückwärts, alle neun Gewichte verstellen."""
    h, o, d_aus, d_h = fehlersignale(netz, x, soll)
    for i, eingang in enumerate([h[0], h[1], 1]):
        netz["aus"][i] += eta * d_aus * eingang
    for j in range(2):
        for i, eingang in enumerate([x[0], x[1], 1]):
            netz["verdeckt"][j][i] += eta * d_h[j] * eingang
    return 0.5 * (soll - o) ** 2


def gesamtfehler(netz: dict) -> float:
    """Summe der Fehler über alle vier XOR-Beispiele."""
    return sum(fehler(netz, x, soll) for x, soll in XOR)


def zufallsnetz(rng: random.Random) -> dict:
    """Alle Gewichte zufällig zwischen -1 und 1."""
    return {
        "verdeckt": [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(2)],
        "aus": [rng.uniform(-1, 1) for _ in range(3)],
    }


def gelernt(netz: dict) -> bool:
    return all(abs(soll - vorwaerts(netz, x)[1]) <= TOLERANZ for x, soll in XOR)


def trainieren(startwert: int, drucken: bool = False, eta: float = LERNRATE):
    """Lernt XOR Runde für Runde (Reihenfolge der Beispiele jede Runde neu gemischt)."""
    rng = random.Random(startwert)
    netz = zufallsnetz(rng)
    beispiele = list(XOR)
    verlauf = [(0, gesamtfehler(netz))]
    if drucken:
        print(f"  Start:       Gesamtfehler {verlauf[0][1]:.4f}")
    for runde in range(1, MAX_RUNDEN + 1):
        rng.shuffle(beispiele)
        for x, soll in beispiele:
            lernschritt(netz, x, soll, eta)
        if runde % 100 == 0:
            verlauf.append((runde, gesamtfehler(netz)))
        if drucken and runde % 500 == 0:
            print(f"  Runde {runde:5d}: Gesamtfehler {gesamtfehler(netz):.4f}")
        if gelernt(netz):
            verlauf.append((runde, gesamtfehler(netz)))
            return netz, runde, verlauf
    return netz, None, verlauf


# ---------------------------------------------------------------------------------------------
# Teil 1: der durchgerechnete Lernschritt aus dem Kapitel
# ---------------------------------------------------------------------------------------------

def buchnetz() -> dict:
    return {
        "verdeckt": [[0.4, -0.3, 0.1],    # Neuron A
                     [-0.2, 0.6, 0.2]],   # Neuron B
        "aus": [0.7, -0.5, 0.1],          # Ausgabeneuron
    }


def wackeltest(netz: dict, x, soll, schicht: str, j: int, i: int, eps: float = 1e-6) -> float:
    """Steigung des Fehlers nach einem Gewicht, gemessen durch winziges Verstellen."""
    def zeiger(n):
        return n["aus"] if schicht == "aus" else n["verdeckt"][j]
    plus, minus = _kopie(netz), _kopie(netz)
    zeiger(plus)[i] += eps
    zeiger(minus)[i] -= eps
    return (fehler(plus, x, soll) - fehler(minus, x, soll)) / (2 * eps)


def _kopie(netz: dict) -> dict:
    return {"verdeckt": [list(w) for w in netz["verdeckt"]], "aus": list(netz["aus"])}


def teil1():
    print("Teil 1: ein Lernschritt von Hand, Eingabe (1, 0), Soll 1, Lernrate 0,5")
    netz = buchnetz()
    x, soll = (1, 0), 1
    za = 0.4 * 1 + -0.3 * 0 + 0.1
    zb = -0.2 * 1 + 0.6 * 0 + 0.2
    h, o, d_aus, d_h = fehlersignale(netz, x, soll)
    zo = 0.7 * h[0] - 0.5 * h[1] + 0.1
    print("  vorwärts:")
    print(f"    A: Summe {za:.3f}  Ausgabe {h[0]:.4f}")
    print(f"    B: Summe {zb:.3f}  Ausgabe {h[1]:.4f}")
    print(f"    Ausgabeneuron: Summe {zo:.4f}  Ausgabe {o:.4f}")
    print(f"    Abweichung Soll - Ist = {soll - o:.4f}   Fehler = {0.5 * (soll - o) ** 2:.4f}")
    print("  rückwärts:")
    print(f"    Steigung der S-Kurve am Ausgang o(1-o) = {o * (1 - o):.4f}")
    print(f"    delta_aus = {d_aus:.4f}")
    for name, j in (("A", 0), ("B", 1)):
        print(f"    delta_{name} = {h[j] * (1 - h[j]):.4f} * {netz['aus'][j]:+.1f} * {d_aus:.4f}"
              f" = {d_h[j]:+.4f}")

    # Probe: jede Steigung aus Backpropagation gegen den Wackeltest
    eingaenge_aus = [h[0], h[1], 1]
    eingaenge_mitte = [x[0], x[1], 1]
    for i in range(3):
        bp = -d_aus * eingaenge_aus[i]
        assert abs(bp - wackeltest(netz, x, soll, "aus", 0, i)) < 1e-8
    for j in range(2):
        for i in range(3):
            bp = -d_h[j] * eingaenge_mitte[i]
            assert abs(bp - wackeltest(netz, x, soll, "verdeckt", j, i)) < 1e-8
    print("    Probe: alle neun Steigungen stimmen mit dem Wackeltest überein.")

    vorher = fehler(netz, x, soll)
    lernschritt(netz, x, soll)
    nachher_h, nachher_o = vorwaerts(netz, x)
    print("  neue Gewichte:")
    print("    zum Ausgang: " + "  ".join(f"{w:+.4f}" for w in netz["aus"]))
    for name, j in (("A", 0), ("B", 1)):
        print(f"    zu {name}:       " + "  ".join(f"{w:+.4f}" for w in netz["verdeckt"][j]))
    nachher = fehler(netz, x, soll)
    print(f"  dieselbe Eingabe noch einmal: Ausgabe {nachher_o:.4f}  Fehler {nachher:.4f}"
          f" (vorher {vorher:.4f})\n")
    assert nachher < vorher
    assert netz["verdeckt"][0][1] == -0.3 and netz["verdeckt"][1][1] == 0.6  # x2 = 0: unverändert
    assert abs(o - 0.5709) < 5e-5 and abs(d_aus - 0.1051) < 5e-5


# ---------------------------------------------------------------------------------------------
# Teil 2 und 3
# ---------------------------------------------------------------------------------------------

def teil2():
    print(f"Teil 2: das Netz lernt XOR (Startwert {STARTWERT}, Lernrate {LERNRATE})")
    netz, runden, verlauf = trainieren(STARTWERT, drucken=True)
    assert runden == 3027, "die Rundenzahl, die im Buch steht"
    print(f"  gelernt nach {runden} Runden, Gesamtfehler {gesamtfehler(netz):.4f}")
    for x, soll in XOR:
        h, o = vorwaerts(netz, x)
        print(f"    Eingabe {x}: soll {soll}  ist {o:.3f}   (A = {h[0]:.3f}, B = {h[1]:.3f})")
        assert abs(soll - o) <= TOLERANZ
    print("  gelernte Gewichte:")
    print("    zum Ausgang: " + "  ".join(f"{w:+.2f}" for w in netz["aus"]))
    for name, j in (("A", 0), ("B", 1)):
        print(f"    zu {name}:       " + "  ".join(f"{w:+.2f}" for w in netz["verdeckt"][j]))
    print("  Verlauf des Gesamtfehlers (für die Abbildung im Buch):")
    print("    " + " ".join(f"({r},{f:.4f})" for r, f in verlauf if r % 200 == 0 or r == runden))
    print()


def teil3():
    print("Teil 3: zwanzig Startwerte im Vergleich")
    erfolge = []
    for startwert in range(1, 21):
        netz, runden, _ = trainieren(startwert)
        if runden:
            status = f"gelernt nach {runden:5d} Runden"
        else:
            antworten = " ".join(f"{vorwaerts(netz, x)[1]:.2f}" for x, _ in XOR)
            status = f"hängt fest, Antworten {antworten}"
        print(f"  Startwert {startwert:2d}: {status}   Gesamtfehler {gesamtfehler(netz):.4f}")
        if runden:
            erfolge.append(runden)
    print(f"  {len(erfolge)} von 20 Versuchen gelernt, im Mittel nach "
          f"{sum(erfolge) / len(erfolge):.0f} Runden.")
    assert len(erfolge) == 13  # die Zahl, die im Buch steht
    schneller = [trainieren(startwert, eta=1.0)[1] for startwert in range(1, 21)]
    anzahl = sum(r is not None for r in schneller)
    print(f"  Zum Vergleich mit Lernrate 1,0: {anzahl} von 20 Versuchen gelernt.")
    assert anzahl == 16


if __name__ == "__main__":
    teil1()
    teil2()
    teil3()
