"""Überanpassung: Trainingsfehler und Testfehler bei einer Polynomanpassung.

Begleitprogramm zu Kapitel 10 "Wenn das Lernen hakt".
Läuft mit Python 3.12 oder neuer, nur Standardbibliothek:

    python ueberanpassung.py

Hinter den Daten steckt eine sanfte Welle, y = sin(pi * x) für x zwischen -1 und 1, dazu ein
zufälliges Rauschen wie bei echten Messungen. Zum Lernen gibt es nur 8 Punkte (Trainingsdaten),
zum Prüfen 200 neue Punkte, die beim Lernen nie vorkommen (Testdaten).

Teil 1: Polynome vom Grad 0 bis 7 werden passgenau an die 8 Punkte angelegt (Methode der
kleinsten Quadrate). Der Trainingsfehler sinkt immer weiter, beim Grad 7 auf null - die Kurve
geht durch alle Punkte. Der Testfehler sinkt nur bis Grad 3 und steigt danach wieder.

Teil 2: Das Polynom vom Grad 7 lernt schrittweise per Gradientenabstieg (mit Schwung). Alle 10
Schritte wird der Testfehler gemessen. Er ist früh am kleinsten und steigt dann, während der
Trainingsfehler weiter fällt: Hier lohnt der frühe Abbruch.
"""

import math
import random

GRAD_MAX = 7
RAUSCHEN = 0.25  # Streuung der Messfehler


def wahre_kurve(x: float) -> float:
    return math.sin(math.pi * x)


def messpunkte(anzahl: int, saat: int, gleichmaessig: bool) -> list[tuple[float, float]]:
    """Punkte auf der wahren Kurve plus Rauschen. Feste Saat: jeder Lauf gibt dieselben Zahlen."""
    zufall = random.Random(saat)
    if gleichmaessig:
        xs = [-1 + 2 * i / (anzahl - 1) for i in range(anzahl)]
    else:
        xs = [zufall.uniform(-1, 1) for _ in range(anzahl)]
    return [(x, wahre_kurve(x) + zufall.gauss(0, RAUSCHEN)) for x in xs]


TRAINING = messpunkte(8, saat=5, gleichmaessig=True)
TEST = messpunkte(200, saat=105, gleichmaessig=False)


def polynom(koeffizienten: list[float], x: float) -> float:
    """c0 + c1*x + c2*x^2 + ..."""
    return sum(c * x**k for k, c in enumerate(koeffizienten))


def mittlerer_fehler(koeffizienten: list[float], daten: list[tuple[float, float]]) -> float:
    """Mittlere quadratische Abweichung zwischen Kurve und Punkten."""
    return sum((polynom(koeffizienten, x) - y) ** 2 for x, y in daten) / len(daten)


def anpassen(grad: int, daten: list[tuple[float, float]]) -> list[float]:
    """Kleinste-Quadrate-Polynom vom gegebenen Grad (Gram-Schmidt, dann Rückwärtseinsetzen)."""
    spalten = [[x**k for x, _ in daten] for k in range(grad + 1)]
    ys = [y for _, y in daten]
    q, r = [], [[0.0] * (grad + 1) for _ in range(grad + 1)]
    for j, v in enumerate(spalten):
        for k in range(j):
            r[k][j] = sum(a * b for a, b in zip(q[k], v))
            v = [a - r[k][j] * b for a, b in zip(v, q[k])]
        r[j][j] = math.sqrt(sum(a * a for a in v))
        q.append([a / r[j][j] for a in v])
    rechts = [sum(a * b for a, b in zip(q[j], ys)) for j in range(grad + 1)]
    c = [0.0] * (grad + 1)
    for j in reversed(range(grad + 1)):
        c[j] = (rechts[j] - sum(r[j][k] * c[k] for k in range(j + 1, grad + 1))) / r[j][j]
    return c


def lernen_mit_pruefung(schritte: int, lernrate: float = 0.5, schwung: float = 0.9):
    """Grad-7-Polynom per Gradientenabstieg lernen; alle 10 Schritte beide Fehler notieren."""
    c = [0.0] * (GRAD_MAX + 1)
    s = [0.0] * (GRAD_MAX + 1)  # letzter Schritt (Schwung)
    protokoll = []
    for schritt in range(1, schritte + 1):
        steigung = [0.0] * (GRAD_MAX + 1)
        for x, y in TRAINING:
            abweichung = polynom(c, x) - y
            for k in range(GRAD_MAX + 1):
                steigung[k] += 2 * abweichung * x**k / len(TRAINING)
        s = [schwung * sk - lernrate * gk for sk, gk in zip(s, steigung)]
        c = [ck + sk for ck, sk in zip(c, s)]
        if schritt % 10 == 0:
            protokoll.append((schritt, mittlerer_fehler(c, TRAINING), mittlerer_fehler(c, TEST)))
    return protokoll


if __name__ == "__main__":
    print("Trainingspunkte (x, y):")
    print("  " + "  ".join(f"({x:+.3f}, {y:+.3f})" for x, y in TRAINING))

    print("\nTeil 1: Polynome verschiedenen Grades")
    print("  Grad   Trainingsfehler   Testfehler")
    tabelle = {}
    for grad in range(GRAD_MAX + 1):
        c = anpassen(grad, TRAINING)
        tabelle[grad] = (mittlerer_fehler(c, TRAINING), mittlerer_fehler(c, TEST))
        print(f"  {grad:4d}   {tabelle[grad][0]:15.3f}   {tabelle[grad][1]:10.3f}")
        if grad in (1, 3, 7):
            print("         Koeffizienten c0, c1, ...: " + ", ".join(f"{k:.4f}" for k in c))
    bester = min(tabelle, key=lambda g: tabelle[g][1])
    print(f"  Kleinster Testfehler bei Grad {bester}.")

    print("\nTeil 2: Grad 7 schrittweise lernen, alle 10 Schritte prüfen")
    print("  Schritt   Trainingsfehler   Testfehler")
    protokoll = lernen_mit_pruefung(30000)
    for schritt, f_train, f_test in protokoll:
        if schritt in (10, 50, 100, 200, 300, 1000, 3000, 10000, 20000, 30000):
            print(f"  {schritt:7d}   {f_train:15.3f}   {f_test:10.3f}")
    stopp = min(protokoll, key=lambda zeile: zeile[2])
    print(f"  Früher Abbruch: kleinster Testfehler {stopp[2]:.3f} nach {stopp[0]} Schritten.")

    # Proben
    trainingsfehler = [tabelle[g][0] for g in range(GRAD_MAX + 1)]
    assert all(a >= b - 1e-12 for a, b in zip(trainingsfehler, trainingsfehler[1:]))  # fällt stets
    assert tabelle[7][0] < 1e-12  # Grad 7 geht durch alle 8 Punkte
    assert bester == 3 and tabelle[7][1] > 5 * tabelle[3][1]
    assert protokoll[-1][1] < stopp[1] and protokoll[-1][2] > 3 * stopp[2]
    print("\nAlle Proben bestanden.")
