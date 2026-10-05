"""Wie schnell ist eine Matrixmultiplikation in reinem Python, wie schnell mit NumPy?

Begleitprogramm zu Kapitel 14 "Winter und Frühling".
Läuft mit Python 3.12 oder neuer und braucht NumPy (ab Teil V des Buchs):

    python matrix_messung.py

Eine Schicht eines neuronalen Netzes rechnet für viele Eingaben auf einmal gewichtete Summen aus.
Mathematisch ist das eine Matrixmultiplikation. Das Programm
1. prüft an einem kleinen Beispiel, das man von Hand nachrechnen kann, dass beide Wege dasselbe liefern,
2. multipliziert dann zufällige quadratische Matrizen wachsender Größe einmal mit drei
   ineinandergeschachtelten Python-Schleifen und einmal mit NumPy,
3. misst jeweils die beste von drei Zeiten und druckt, wie viele Multiplikationen pro Sekunde
   jeder Weg schafft und wie viel schneller NumPy ist.
Die Zeiten hängen vom Rechner ab und schwanken von Lauf zu Lauf; aussagekräftig ist die
Größenordnung des Verhältnisses.
"""

import random
import time

import numpy as np

STARTWERT = 14
GROESSEN = [50, 100, 200]
WIEDERHOLUNGEN = 3


def matrix_mal_python(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Zeile mal Spalte, Zahl für Zahl: drei Schleifen ineinander."""
    zeilen, mitte, spalten = len(a), len(b), len(b[0])
    ergebnis = [[0.0] * spalten for _ in range(zeilen)]
    for i in range(zeilen):
        for j in range(spalten):
            summe = 0.0
            for k in range(mitte):
                summe += a[i][k] * b[k][j]
            ergebnis[i][j] = summe
    return ergebnis


def beste_zeit(aufgabe, wiederholungen: int = WIEDERHOLUNGEN) -> float:
    """Die Aufgabe mehrmals ausführen und die kürzeste Zeit in Sekunden zurückgeben."""
    zeiten = []
    for _ in range(wiederholungen):
        start = time.perf_counter()
        aufgabe()
        zeiten.append(time.perf_counter() - start)
    return min(zeiten)


def zufallsmatrix(n: int, zufall: random.Random) -> list[list[float]]:
    return [[zufall.uniform(-1.0, 1.0) for _ in range(n)] for _ in range(n)]


def kleines_beispiel() -> None:
    """Eine Schicht mit drei Neuronen und zwei Eingängen, zwei Eingabebeispiele."""
    gewichte = [[1.0, 2.0],    # Neuron 1
                [0.5, -1.0],   # Neuron 2
                [3.0, 0.0]]    # Neuron 3
    eingaben = [[2.0, 1.0],    # jede Spalte ist ein Beispiel:
                [3.0, 4.0]]    # erstes (2, 3), zweites (1, 4)
    summen = matrix_mal_python(gewichte, eingaben)
    print("Kleines Beispiel: Gewichte (3 x 2) mal Eingaben (2 x 2)")
    for nummer, zeile in enumerate(summen, start=1):
        print(f"  Neuron {nummer}: Summen {zeile}")
    assert summen == [[8.0, 9.0], [-2.0, -3.5], [6.0, 3.0]]
    assert np.array_equal(np.array(gewichte) @ np.array(eingaben), np.array(summen))
    print("  Probe bestanden: reines Python und NumPy liefern dasselbe.\n")


def messen() -> list[float]:
    zufall = random.Random(STARTWERT)
    verhaeltnisse = []
    print(f"{'Größe':>7} {'Multiplikationen':>17} {'reines Python':>14} {'NumPy':>11}"
          f" {'Python/s':>10} {'NumPy/s':>10} {'NumPy schneller':>16}")
    for n in GROESSEN:
        a, b = zufallsmatrix(n, zufall), zufallsmatrix(n, zufall)
        a_np, b_np = np.array(a), np.array(b)
        assert np.allclose(np.array(matrix_mal_python(a, b)), a_np @ b_np)
        zeit_python = beste_zeit(lambda: matrix_mal_python(a, b))
        zeit_numpy = beste_zeit(lambda: a_np @ b_np, wiederholungen=50)
        multiplikationen = n ** 3
        verhaeltnis = zeit_python / zeit_numpy
        verhaeltnisse.append(verhaeltnis)
        print(f"{n:>4}x{n:<3} {multiplikationen:>17,} {zeit_python:>12.4f} s {zeit_numpy:>9.6f} s"
              f" {multiplikationen / zeit_python:>10.1e} {multiplikationen / zeit_numpy:>10.1e}"
              f" {verhaeltnis:>13,.0f}-mal".replace(",", "."))
    return verhaeltnisse


if __name__ == "__main__":
    kleines_beispiel()
    verhaeltnisse = messen()
    assert all(v > 10 for v in verhaeltnisse), "NumPy sollte deutlich schneller sein"
    print(f"\nBei der größten Matrix war NumPy etwa {verhaeltnisse[-1]:,.0f}-mal schneller."
          .replace(",", "."))
