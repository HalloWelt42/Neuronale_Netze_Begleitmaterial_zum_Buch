"""Netze mit einer verdeckten Schicht und fest eingestellten Gewichten.

Begleitprogramm zu Kapitel 7 "Mehrere Schichten".
Läuft mit Python 3.12 oder neuer, nur Standardbibliothek:

    python xor_netz.py

Kein Gewicht in diesem Programm wird gelernt. Alle Zahlen sind von Hand eingestellt, so wie im
Kapitel beschrieben. Das Programm zeigt, was verdeckte Neuronen leisten:

Teil 1: XOR mit zwei verdeckten Neuronen (ODER und UND). Das Programm druckt für jede Eingabe, was
die verdeckte Schicht "sieht", und prüft, dass am Ende XOR herauskommt. Dasselbe Bauprinzip
löst die Parität für beliebig viele Eingänge.

Teil 2: Das kleinere XOR-Netz mit vier Zellen: nur ein verdecktes Neuron (UND), dafür laufen die
Eingänge zusätzlich direkt zum Ausgabeneuron (Abkürzung).

Teil 3: Die Ebene falten. Zwei ReLU-Neuronen falten das Quadrat entlang der Diagonalen; danach
reicht ein einziger gerader Schnitt. Das Netz arbeitet auch für Kommazahlen zwischen 0 und 1 und
antwortet dort anders als das Netz aus Teil 1, obwohl beide XOR lösen.

Teil 4: Treppenstufen bauen eine Kurve. Viele Stufen-Neuronen nebeneinander, deren Ausgaben das
Ausgabeneuron zusammenzählt, ahmen die Hügelkurve f(x) = 4x(1 - x) immer genauer nach.
"""

from itertools import product


def stufe(s: float) -> int:
    """Treppenstufe: 1, wenn die Summe größer als 0 ist, sonst 0."""
    return 1 if s > 0 else 0


def relu(s: float) -> float:
    """ReLU: die Summe selbst, wenn sie positiv ist, sonst 0."""
    return s if s > 0 else 0.0


def neuron(gewichte, bias, eingaben, aktivierung=stufe):
    """Ein Neuron: gewichtete Summe plus Bias, dann die Aktivierungsfunktion."""
    summe = sum(w * x for w, x in zip(gewichte, eingaben)) + bias
    return aktivierung(summe)


# --- Teil 1: XOR mit zwei verdeckten Neuronen ----------------------------
def xor_netz(x1, x2):
    """Liefert die Ausgaben der verdeckten Schicht und die Netzausgabe."""
    h_oder = neuron([1, 1], -0.5, [x1, x2])        # mindestens einer an
    h_und = neuron([1, 1], -1.5, [x1, x2])         # beide an
    o = neuron([1, -1], -0.5, [h_oder, h_und])     # ODER, aber nicht UND
    return (h_oder, h_und), o


def paritaet_netz(bits):
    """Parität für beliebig viele Eingänge: Ist eine ungerade Anzahl von Bits an?

    Verdecktes Neuron k feuert, wenn mindestens k Bits an sind. Das Ausgabeneuron zählt diese
    Neuronen abwechselnd mit +1 und -1. Für zwei Eingänge ist das genau das XOR-Netz oben.
    """
    n = len(bits)
    verdeckt = [neuron([1] * n, -(k - 0.5), bits) for k in range(1, n + 1)]
    return neuron([(-1) ** (k + 1) for k in range(1, n + 1)], -0.5, verdeckt)


# --- Teil 2: XOR-Netz mit vier Zellen (Abkürzung) -----------------------
def xor_vier_zellen(x1, x2):
    h_und = neuron([1, 1], -1.5, [x1, x2])
    summe = 1 * x1 + 1 * x2 - 2 * h_und - 0.5     # Ausgabeneuron sieht x1, x2 und h
    return h_und, summe, stufe(summe)


# --- Teil 3: Falten mit ReLU ---------------------------------------------
def falt_netz(x1, x2):
    links = neuron([1, -1], 0, [x1, x2], relu)     # wie weit x1 über x2 liegt
    rechts = neuron([-1, 1], 0, [x1, x2], relu)    # wie weit x2 über x1 liegt
    abstand = links + rechts                       # = |x1 - x2|, egal welche Seite
    return abstand, neuron([1, 1], -0.5, [links, rechts])


# --- Teil 4: Treppenstufen bauen eine Kurve ------------------------------
def huegel(x):
    return 4 * x * (1 - x)


def treppen_netz(f, n):
    """Baut ein Netz mit n - 1 Stufen-Neuronen, das f auf [0, 1] durch n Treppenstufen annähert.

    Stufe k springt bei x = k/n. Ihr Gewicht ist der Höhenunterschied zwischen Block k - 1 und
    Block k; die Höhe eines Blocks ist der Kurvenwert in seiner Mitte. Der Bias des Ausgabeneurons
    ist die Höhe des ersten Blocks.
    """
    hoehen = [f((k + 0.5) / n) for k in range(n)]
    stufen = [(k / n, hoehen[k] - hoehen[k - 1]) for k in range(1, n)]  # (Sprungstelle, Gewicht)
    return hoehen[0], stufen


def netz_ausgabe(bias, stufen, x):
    """Ausgabeneuron mit Identität: Bias plus gewichtete Summe aller Stufen-Neuronen."""
    return bias + sum(w * stufe(x - c) for c, w in stufen)


def groesster_fehler(f, bias, stufen, punkte=1000):
    """Größter Abstand zwischen Kurve und Netz an 1001 gleichmäßig verteilten Stellen in [0, 1]."""
    return max(abs(f(i / punkte) - netz_ausgabe(bias, stufen, i / punkte))
               for i in range(punkte + 1))


if __name__ == "__main__":
    print("Teil 1: XOR mit zwei verdeckten Neuronen")
    print("  x1 x2 | h_ODER h_UND | Ausgabe")
    for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        (h1, h2), o = xor_netz(x1, x2)
        print(f"   {x1}  {x2} |   {h1}     {h2}   |    {o}")
        assert o == (x1 ^ x2)
    bilder = {xor_netz(*e)[0] for e in [(0, 1), (1, 0)]}
    print(f"  (0,1) und (1,0) landen in der verdeckten Schicht beide auf {bilder.pop()}.")
    for n in range(2, 7):
        assert all(paritaet_netz(bits) == sum(bits) % 2 for bits in product([0, 1], repeat=n))
        print(f"  Parität mit {n} Eingängen: {n} verdeckte Neuronen, alle {2 ** n} Fälle richtig")
    print()

    print("Teil 2: XOR-Netz mit vier Zellen")
    for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        h, s, o = xor_vier_zellen(x1, x2)
        print(f"   ({x1},{x2}): h_UND = {h}, Summe am Ausgang = {s:+.1f}, Ausgabe {o}")
        assert o == (x1 ^ x2)
    print()

    print("Teil 3: Die Ebene falten (ReLU)")
    for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1), (0.2, 0.9), (0.6, 0.5), (0.9, 0.3)]:
        abstand, o = falt_netz(x1, x2)
        print(f"   ({x1},{x2}): Unterschied |x1 - x2| = {abstand:.1f}, Ausgabe {o}")
        assert o == (1 if abs(x1 - x2) > 0.5 else 0)
    _, band = xor_netz(0.6, 0.5)
    print(f"   Zum Vergleich das Netz aus Teil 1 bei (0.6,0.5): Ausgabe {band}")
    assert band == 1 and falt_netz(0.6, 0.5)[1] == 0
    print()

    print("Teil 4: Treppenstufen bauen die Hügelkurve f(x) = 4x(1 - x)")
    bias, stufen = treppen_netz(huegel, 4)
    print(f"   n = 4: Bias {bias:.4f}, Stufen (Sprungstelle, Gewicht): "
          + ", ".join(f"({c:.2f}, {w:+.4f})" for c, w in stufen))
    assert abs(netz_ausgabe(bias, stufen, 0.6) - 0.9375) < 1e-12
    print("   Blöcke | verdeckte Neuronen | größter Fehler")
    vorher = None
    for n in (4, 8, 16, 32, 64):
        bias, stufen = treppen_netz(huegel, n)
        fehler = groesster_fehler(huegel, bias, stufen)
        print(f"   {n:6d} | {len(stufen):18d} | {fehler:.4f}")
        if vorher is not None:
            assert fehler < vorher       # mehr Neuronen, kleinerer Fehler
        vorher = fehler
    assert vorher < 0.04
