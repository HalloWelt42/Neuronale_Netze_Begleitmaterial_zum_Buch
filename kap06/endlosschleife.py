"""Ein Perzeptron versucht XOR zu lernen und merkt, dass es sich im Kreis dreht.

Begleitprogramm zu Kapitel 6 "Adaline und die Grenzen der Geraden".
Läuft mit Python 3.12 oder neuer, nur Standardbibliothek:

    python endlosschleife.py

Teil 1: Das Perzeptron aus Kapitel 5 lernt UND (klappt) und XOR (klappt nie). Das Programm
merkt sich den Stand der Gewichte zu Beginn jeder Runde. Taucht ein Stand ein zweites Mal auf,
wiederholt sich von da an alles genau gleich: eine Endlosschleife, die das Programm erkennt und
meldet, statt ewig weiterzurechnen.

Teil 2: Adaline mit der Delta-Regel. Sie kommt auch bei XOR zur Ruhe, findet dort aber nur den
Kompromiss "alles ungefähr 0" - sie weiß es auch nicht besser.

Teil 3: Wie viele Logikfunktionen lassen sich überhaupt durch eine Gerade (bei drei Eingängen:
eine Ebene) trennen? Das Programm probiert alle kleinen ganzzahligen Gewichte durch und zählt.
"""

from itertools import product

LERNRATE = 0.5


def ausgabe(gewichte: list[float], bias: float, eingabe: tuple[int, int]) -> int:
    """Perzeptron: gewichtete Summe bilden und mit der Schwelle 0 vergleichen."""
    summe = sum(w * x for w, x in zip(gewichte, eingabe)) + bias
    return 1 if summe > 0 else 0


def runde_lernen(gewichte, bias, beispiele):
    """Eine Runde Perzeptron-Lernregel; liefert neue Gewichte, neuen Bias und die Fehlerzahl."""
    fehler_in_runde = 0
    for eingabe, soll in beispiele:
        fehler = soll - ausgabe(gewichte, bias, eingabe)
        if fehler != 0:
            fehler_in_runde += 1
            gewichte = [w + LERNRATE * fehler * x for w, x in zip(gewichte, eingabe)]
            bias += LERNRATE * fehler
    return gewichte, bias, fehler_in_runde


def versuchen(beispiele, max_runden=100):
    """Lernt Runde für Runde und erkennt, wenn ein Stand vom Rundenbeginn wiederkehrt."""
    gewichte, bias = [0.0, 0.0], 0.0
    gesehen = {}  # Stand zu Rundenbeginn -> Nummer der Runde
    for runde in range(1, max_runden + 1):
        stand = (*gewichte, bias)
        if stand in gesehen:
            return "Endlosschleife", gesehen[stand], runde
        gesehen[stand] = runde
        gewichte, bias, fehler = runde_lernen(gewichte, bias, beispiele)
        print(f"  Runde {runde}: {fehler} Fehler, danach w1={gewichte[0]:+.1f}"
              f"  w2={gewichte[1]:+.1f}  b={bias:+.1f}")
        if fehler == 0:
            return "gelernt", runde, None
    return "abgebrochen", max_runden, None


def adaline(beispiele, lernrate=0.05, runden=400):
    """Delta-Regel: nach JEDEM Beispiel um Lernrate * (Soll - Summe) * Eingabe schieben."""
    gewichte, bias = [0.0, 0.0], 0.0
    for _ in range(runden):
        for eingabe, soll in beispiele:
            summe = sum(w * x for w, x in zip(gewichte, eingabe)) + bias
            delta = soll - summe
            gewichte = [w + lernrate * delta * x for w, x in zip(gewichte, eingabe)]
            bias += lernrate * delta
    return gewichte, bias


def mittlerer_quadratischer_fehler(gewichte, bias, beispiele):
    summen = [sum(w * x for w, x in zip(gewichte, e)) + bias for e, _ in beispiele]
    return sum((soll - s) ** 2 for s, (_, soll) in zip(summen, beispiele)) / len(beispiele)


def zaehle_trennbare(n: int, grenze: int = 3) -> int:
    """Zählt die Logikfunktionen mit n Eingängen, die ein einzelnes Perzeptron darstellen kann."""
    ecken = list(product([0, 1], repeat=n))
    gefunden = set()
    for gewichte in product(range(-grenze, grenze + 1), repeat=n):
        summen = [sum(w * x for w, x in zip(gewichte, ecke)) for ecke in ecken]
        for doppelte_schwelle in range(-2 * grenze * n - 1, 2 * grenze * n + 2):
            schwelle = doppelte_schwelle / 2
            gefunden.add(tuple(1 if s > schwelle else 0 for s in summen))
    return len(gefunden)


UND = [((0, 0), 0), ((0, 1), 0), ((1, 0), 0), ((1, 1), 1)]
XOR = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]

if __name__ == "__main__":
    print("Perzeptron lernt UND")
    ergebnis, runde, _ = versuchen(UND)
    print(f"  -> {ergebnis} nach {runde} Runden\n")
    assert ergebnis == "gelernt" and runde == 6

    print("Perzeptron versucht XOR")
    ergebnis, erste, wieder = versuchen(XOR)
    print(f"  -> {ergebnis}: Der Stand zu Beginn von Runde {wieder} ist derselbe wie zu Beginn"
          f" von Runde {erste}.\n     Von hier an wiederholt sich alles, ein Kreis von"
          f" {wieder - erste} Runde(n) - fertig wird das Perzeptron nie.\n")
    assert ergebnis == "Endlosschleife"

    # Adaline arbeitet mit den Antworten +1 (ja) und -1 (nein).
    print("Adaline mit Delta-Regel (Antworten +1 und -1)")
    for name, aufgabe in (("UND", UND), ("XOR", XOR)):
        beispiele = [(e, 1 if soll == 1 else -1) for e, soll in aufgabe]
        g, b = adaline(beispiele)
        mqf = mittlerer_quadratischer_fehler(g, b, beispiele)
        summen = [sum(w * x for w, x in zip(g, e)) + b for e, _ in beispiele]
        print(f"  {name}: w1={g[0]:+.2f}  w2={g[1]:+.2f}  b={b:+.2f}  "
              f"Summen {' '.join(f'{s:+.2f}' for s in summen)}  mittlerer quadratischer Fehler {mqf:.2f}")
        richtig = sum((s > 0) == (soll > 0) for s, (_, soll) in zip(summen, beispiele))
        print(f"       richtig eingeordnet: {richtig} von 4")
        if name == "UND":
            assert richtig == 4
        else:
            assert richtig < 4 and mqf > 0.9

    print("\nWie viele Logikfunktionen sind linear trennbar?")
    for n, erwartet in ((2, 14), (3, 104)):
        anzahl = zaehle_trennbare(n)
        print(f"  {n} Eingänge: {anzahl} von {2 ** 2 ** n} möglichen Funktionen")
        assert anzahl == erwartet
