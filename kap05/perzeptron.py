"""Ein Perzeptron lernt die UND- und die ODER-Verknüpfung.

Begleitprogramm zu Kapitel 5 "Das Perzeptron lernt".
Läuft mit Python 3.12 oder neuer, nur Standardbibliothek:

    python perzeptron.py

Das Programm druckt jeden Lernschritt: welches Beispiel gerade dran ist, was das Perzeptron
antwortet, ob es richtig lag und wie sich die Gewichte ändern.
"""

LERNRATE = 0.5


def ausgabe(gewichte: list[float], bias: float, eingabe: tuple[int, int]) -> int:
    """Gewichtete Summe bilden und mit der Schwelle 0 vergleichen."""
    summe = sum(w * x for w, x in zip(gewichte, eingabe)) + bias
    return 1 if summe > 0 else 0


def lernen(beispiele: list[tuple[tuple[int, int], int]], runden: int = 20, drucken: bool = True):
    """Perzeptron-Lernregel: bei jedem Fehler die Gewichte um Lernrate * Fehler * Eingabe schieben."""
    gewichte, bias = [0.0, 0.0], 0.0
    for runde in range(1, runden + 1):
        fehler_in_runde = 0
        for eingabe, soll in beispiele:
            ist = ausgabe(gewichte, bias, eingabe)
            fehler = soll - ist  # +1: zu wenig, -1: zu viel, 0: richtig
            if fehler != 0:
                fehler_in_runde += 1
                gewichte = [w + LERNRATE * fehler * x for w, x in zip(gewichte, eingabe)]
                bias += LERNRATE * fehler
            if drucken:
                zeichen = "richtig" if fehler == 0 else f"falsch, Fehler {fehler:+d}"
                print(f"Runde {runde}  Eingabe {eingabe}  soll {soll}  ist {ist}  {zeichen:<17}"
                      f"  w1={gewichte[0]:+.1f}  w2={gewichte[1]:+.1f}  b={bias:+.1f}")
        if fehler_in_runde == 0:
            if drucken:
                print(f"Fertig nach {runde} Runden: alle vier Beispiele richtig.\n")
            return gewichte, bias, runde
    return gewichte, bias, None


UND = [((0, 0), 0), ((0, 1), 0), ((1, 0), 0), ((1, 1), 1)]
ODER = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1)]

if __name__ == "__main__":
    print("UND-Verknüpfung")
    g, b, runden = lernen(UND)
    assert runden is not None and all(ausgabe(g, b, x) == y for x, y in UND)
    print("ODER-Verknüpfung")
    g, b, runden = lernen(ODER)
    assert runden is not None and all(ausgabe(g, b, x) == y for x, y in ODER)
