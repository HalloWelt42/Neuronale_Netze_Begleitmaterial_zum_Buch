"""Einfacher Gradientenabstieg gegen Abstieg mit Schwung (Momentum) in einer langen Schlucht.

Begleitprogramm zu Kapitel 10 "Wenn das Lernen hakt".
Läuft mit Python 3.12 oder neuer, nur Standardbibliothek:

    python schlucht_momentum.py

Die Fehlerlandschaft ist F(w1, w2) = 0,05 * w1^2 + 2 * w2^2: eine Schlucht, die in Richtung w1
lang und flach, in Richtung w2 schmal und steil ist. Das Programm startet bei (-10, 1) und zählt,
wie viele Schritte der Abstieg braucht, bis der Fehler unter 0,01 fällt. Es zeigt
1. den einfachen Abstieg, der quer zur Schlucht hin und her springt,
2. eine etwas größere Lernrate, bei der das Springen nie aufhört,
3. den Abstieg mit Schwung, der längs der Schlucht Fahrt aufnimmt und viel schneller ankommt.
"""

START = (-10.0, 1.0)
ZIEL = 0.01  # Fehler, unter dem wir "angekommen" sagen


def fehler(w1: float, w2: float) -> float:
    """Höhe der Fehlerlandschaft an der Stelle (w1, w2)."""
    return 0.05 * w1 * w1 + 2.0 * w2 * w2


def steigung(w1: float, w2: float) -> tuple[float, float]:
    """Steigung in w1- und in w2-Richtung (Parabel a*x^2 hat die Steigung 2*a*x)."""
    return 0.1 * w1, 4.0 * w2


def abstieg(lernrate: float, schwung: float = 0.0, schritte_max: int = 500):
    """Gradientenabstieg mit Schwung. Schwung 0 ist der einfache Abstieg.

    Schritt = Schwung * letzter Schritt - Lernrate * Steigung
    Gibt den ganzen Weg als Liste von Punkten zurück.
    """
    w1, w2 = START
    s1, s2 = 0.0, 0.0  # letzter Schritt, anfangs null
    weg = [(w1, w2)]
    for _ in range(schritte_max):
        g1, g2 = steigung(w1, w2)
        s1 = schwung * s1 - lernrate * g1
        s2 = schwung * s2 - lernrate * g2
        w1, w2 = w1 + s1, w2 + s2
        weg.append((w1, w2))
        if fehler(w1, w2) < ZIEL:
            break
    return weg


def zeige(titel: str, weg: list[tuple[float, float]], zeilen: int = 6) -> None:
    """Die ersten Schritte und das Ergebnis eines Laufs drucken."""
    print(titel)
    for nr, (w1, w2) in enumerate(weg[:zeilen]):
        print(f"  Schritt {nr:3d}:  w1 = {w1:+7.3f}   w2 = {w2:+7.3f}   F = {fehler(w1, w2):7.3f}")
    w1, w2 = weg[-1]
    angekommen = fehler(w1, w2) < ZIEL
    ende = "angekommen" if angekommen else "NICHT angekommen"
    print(f"  ... nach {len(weg) - 1} Schritten {ende}:  w1 = {w1:+.3f}   w2 = {w2:+.3f}"
          f"   F = {fehler(w1, w2):.4f}\n")


if __name__ == "__main__":
    einfach = abstieg(lernrate=0.45)
    zeige("Einfacher Abstieg, Lernrate 0,45", einfach)

    zu_gross = abstieg(lernrate=0.5, schritte_max=200)
    zeige("Einfacher Abstieg, Lernrate 0,5 (w2 springt ewig zwischen +1 und -1)", zu_gross)

    mit_schwung = abstieg(lernrate=0.45, schwung=0.7)
    zeige("Abstieg mit Schwung 0,7, Lernrate 0,45", mit_schwung)

    print("Schritte bis zum Ziel bei Lernrate 0,45 und verschiedenem Schwung:")
    ergebnis = {}
    for schwung in (0.0, 0.5, 0.7, 0.9):
        ergebnis[schwung] = len(abstieg(lernrate=0.45, schwung=schwung)) - 1
        print(f"  Schwung {schwung:.1f}:  {ergebnis[schwung]:3d} Schritte")

    # Proben
    assert len(einfach) - 1 == 68
    assert fehler(*zu_gross[-1]) > 1.9 and len(zu_gross) - 1 == 200  # springt und kommt nie an
    assert all(a[1] * b[1] < 0 for a, b in zip(einfach[:10], einfach[1:11]))  # Zickzack quer
    assert len(mit_schwung) - 1 == 14
    assert ergebnis[0.7] < ergebnis[0.5] < ergebnis[0.0] and ergebnis[0.9] > ergebnis[0.7]
    print("\nAlle Proben bestanden.")
