"""Gradientenabstieg auf einer Fehlerparabel: Lernrate zu klein, passend, zu groß.

Begleitprogramm zu Kapitel 8 "Die Fehlerlandschaft".
Läuft mit Python 3.12 oder neuer, nur Standardbibliothek:

    python gradientenabstieg.py

Die Aufgabe: Ein Neuron mit einem einzigen Gewicht w soll den Preis einer Eisportion lernen,
Preis = w * Kugeln. Die Beispiele sind 1 Kugel für 2 Euro und 2 Kugeln für 4 Euro.

Teil 1: Der mittlere quadratische Fehler E(w) und seine Steigung. Das Programm nähert die
Steigung mit immer kleineren Schritten h an (Sekante wird zur Tangente) und vergleicht mit der
genauen Formel.

Teil 2: Gradientenabstieg w_neu = w - Lernrate * Steigung mit vier verschiedenen Lernraten.
Das Programm druckt die ersten Schritte und zählt, wie viele Schritte es braucht, bis w auf
0,01 genau beim besten Wert 2 liegt, oder meldet, dass der Abstieg entgleist.

Teil 3: Zwei Gewichte (Preis einer Kugel und Preis einer Sahnehaube). Das Programm geht auf der
Höhenlinienkarte bergab und druckt den Pfad, der in der Abbildung des Kapitels eingezeichnet ist.
"""

EIS = [(1, 2.0), (2, 4.0)]  # (Kugeln, Preis in Euro)


def fehler(w: float) -> float:
    """Mittlerer quadratischer Fehler des Neurons Preis = w * Kugeln."""
    return sum((w * x - y) ** 2 for x, y in EIS) / len(EIS)


def steigung(w: float) -> float:
    """Genaue Steigung von E an der Stelle w: Mittelwert von 2 * (w*x - y) * x."""
    return sum(2 * (w * x - y) * x for x, y in EIS) / len(EIS)


def steigung_genaehert(w: float, h: float) -> float:
    """Steigung der Sekante zwischen w und w + h (Differenzenquotient)."""
    return (fehler(w + h) - fehler(w)) / h


def abstieg(w: float, lernrate: float, schritte: int) -> list[float]:
    """Gradientenabstieg: immer ein Stück entgegen der Steigung gehen."""
    pfad = [w]
    for _ in range(schritte):
        w = w - lernrate * steigung(w)
        pfad.append(w)
    return pfad


def schritte_bis_ziel(lernrate: float, ziel: float = 2.0, genauigkeit: float = 0.01,
                      hoechstens: int = 1000) -> int | None:
    """Wie viele Schritte, bis w auf genauigkeit an das Ziel herankommt? None = nie."""
    w = 0.0
    for schritt in range(1, hoechstens + 1):
        w = w - lernrate * steigung(w)
        if abs(w - ziel) < genauigkeit:
            return schritt
        if abs(w) > 1e6:
            return None
    return None


# --- Teil 3: zwei Gewichte -----------------------------------------------------------
# (Kugeln, Sahnehauben) -> Preis in Euro
EIS_SAHNE = [((1, 0), 2.0), ((0, 1), 1.0), ((1, 1), 3.0)]


def fehler2(w1: float, w2: float) -> float:
    """Mittlerer quadratischer Fehler des Neurons Preis = w1 * Kugeln + w2 * Sahne."""
    return sum((w1 * x1 + w2 * x2 - y) ** 2 for (x1, x2), y in EIS_SAHNE) / len(EIS_SAHNE)


def gradient2(w1: float, w2: float) -> tuple[float, float]:
    """Steigung in w1-Richtung und in w2-Richtung (das jeweils andere Gewicht bleibt fest)."""
    n = len(EIS_SAHNE)
    g1 = sum(2 * (w1 * x1 + w2 * x2 - y) * x1 for (x1, x2), y in EIS_SAHNE) / n
    g2 = sum(2 * (w1 * x1 + w2 * x2 - y) * x2 for (x1, x2), y in EIS_SAHNE) / n
    return g1, g2


def abstieg2(w1: float, w2: float, lernrate: float, schritte: int) -> list[tuple[float, float]]:
    pfad = [(w1, w2)]
    for _ in range(schritte):
        g1, g2 = gradient2(w1, w2)
        w1, w2 = w1 - lernrate * g1, w2 - lernrate * g2
        pfad.append((w1, w2))
    return pfad


def zahl(z: float, stellen: int = 3) -> str:
    """Zahl mit Dezimalkomma, wie im Buch."""
    return f"{z:+.{stellen}f}".replace(".", ",")


if __name__ == "__main__":
    print("Teil 1: Fehler und Steigung")
    print(f"  E(0) = {zahl(fehler(0), 2)}   E(1) = {zahl(fehler(1), 2)}   E(2) = {zahl(fehler(2), 2)}")
    print("  Sekantensteigung an der Stelle w = 0:")
    for h in (1, 0.1, 0.01, 0.001):
        print(f"    h = {str(h).replace('.', ','):<6}  Steigung = {zahl(steigung_genaehert(0, h), 4)}")
    print(f"  genaue Steigung (Tangente): {zahl(steigung(0), 4)}")
    for w in (0, 1, 2, 3):
        print(f"  Steigung bei w = {w}: {zahl(steigung(w), 1)}   (Formel 5 * (w - 2) = {5 * (w - 2)})")
        assert abs(steigung(w) - 5 * (w - 2)) < 1e-12
        assert abs(steigung_genaehert(w, 1e-6) - steigung(w)) < 1e-4
    assert abs(fehler(0) - 10) < 1e-12 and fehler(2) == 0

    print("\nTeil 2: Gradientenabstieg mit verschiedenen Lernraten, Start bei w = 0")
    ergebnisse = {}
    for lernrate in (0.02, 0.1, 0.3, 0.5):
        pfad = abstieg(0.0, lernrate, 5)
        schritte = schritte_bis_ziel(lernrate)
        ergebnisse[lernrate] = schritte
        werte = "  ".join(zahl(w, 3) for w in pfad)
        fehlerwerte = "  ".join(zahl(fehler(w), 3) for w in pfad)
        ende = f"nach {schritte} Schritten auf 0,01 genau" if schritte else "entgleist"
        print(f"  Lernrate {zahl(lernrate, 2)[1:]}:  w = {werte}")
        print(f"                  E = {fehlerwerte}")
        print(f"                  {ende}")
    print("  Grenzfall Lernrate 0,4:", "  ".join(zahl(w, 1) for w in abstieg(0.0, 0.4, 5)))
    print("  Volltreffer Lernrate 0,2:", "  ".join(zahl(w, 1) for w in abstieg(0.0, 0.2, 2)))
    assert ergebnisse[0.02] > ergebnisse[0.1]   # zu klein: langsam
    assert ergebnisse[0.5] is None              # zu groß: entgleist
    assert ergebnisse[0.3] is not None          # schaukelt, kommt aber an
    assert abstieg(0.0, 0.1, 3) == [0.0, 1.0, 1.5, 1.75]

    print("\nTeil 3: zwei Gewichte, Start bei w1 = 1, w2 = 4, Lernrate 0,3")
    pfad2 = abstieg2(1.0, 4.0, 0.3, 30)
    for nr, (w1, w2) in enumerate(pfad2):
        if nr <= 8 or nr % 10 == 0:
            g1, g2 = gradient2(w1, w2)
            print(f"  Schritt {nr:2d}: w1 = {zahl(w1)}  w2 = {zahl(w2)}  E = {zahl(fehler2(w1, w2))}"
                  f"  Gradient = ({zahl(g1)}, {zahl(g2)})")
    w1, w2 = pfad2[-1]
    assert abs(w1 - 2) < 0.01 and abs(w2 - 1) < 0.01
    assert all(fehler2(*b) < fehler2(*a) for a, b in zip(pfad2, pfad2[1:]))
    print("  Pfad für die Abbildung:", " ".join(f"({a:.3f},{b:.3f})" for a, b in pfad2[:13]))
    print("\nAlle Proben bestanden.")
