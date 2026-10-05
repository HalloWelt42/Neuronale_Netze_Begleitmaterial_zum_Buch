"""Aufmerksamkeit von Hand: Wer gehört in "Lea ruft Tom, er kommt" zu wem?

Begleitprogramm zu Kapitel 18 "Aufmerksamkeit und Transformer".
Läuft mit Python 3.12 oder neuer und braucht NumPy:

    python aufmerksamkeit.py

Alle Zahlen sind erfunden und klein, damit du jeden Schritt mit dem Taschenrechner
nachprüfen kannst. In einem echten Netz kommen Fragen, Schlüssel und Werte aus gelernten
Gewichten; hier legen wir sie von Hand fest.

Das Programm druckt
  1. die Passung (Skalarprodukt) jeder Frage mit jedem Schlüssel,
  2. die Aufmerksamkeitsmatrix nach der Softmax (jede Zeile ergibt 1),
  3. die neuen Wortvektoren als gewichtete Mischung der Werte,
  4. zum Vergleich die Matrix mit der Skalierung durch die Wurzel der Länge.
"""

import numpy as np

WOERTER = ["Lea", "ruft", "Tom", "er", "kommt"]

# Schlüssel: was jedes Wort über sich anbietet.
# Spalten: Person, Tätigkeit, Geschlecht (+1 männlich, -1 weiblich)
K = np.array([[1, 0, -1],    # Lea
              [0, 1,  0],    # ruft
              [1, 0,  1],    # Tom
              [0, 0,  1],    # er
              [0, 1,  0]])   # kommt

# Fragen: wonach jedes Wort sucht (gleiche Spalten wie die Schlüssel).
Q = np.array([[0, 2, 0],     # Lea sucht ihre Tätigkeit
              [2, 0, 0],     # ruft sucht eine Person
              [0, 2, 0],     # Tom sucht seine Tätigkeit
              [2, 0, 2],     # er sucht eine männliche Person
              [2, 0, 0]])    # kommt sucht eine Person

# Werte: was jedes Wort weitergibt, hier einfach sein Wortvektor.
# Spalten: Person, Tätigkeit, Geschlecht, Verweis (Fürwort)
V = np.array([[1, 0, -1, 0],
              [0, 1,  0, 0],
              [1, 0,  1, 0],
              [0, 0,  1, 1],
              [0, 1,  0, 0]], dtype=float)


def softmax(zeile: np.ndarray) -> np.ndarray:
    """Macht aus beliebigen Zahlen positive Gewichte, die zusammen 1 ergeben."""
    e = np.exp(zeile - zeile.max())   # Abziehen des Maximums ändert nichts, schützt vor Überlauf
    return e / e.sum()


def aufmerksamkeit(Q, K, V, skalieren=False):
    """Passung = Q mal K quer, Softmax je Zeile, dann Werte mischen."""
    passung = Q @ K.T
    if skalieren:
        passung = passung / np.sqrt(K.shape[1])
    gewichte = np.array([softmax(z) for z in passung])
    return passung, gewichte, gewichte @ V


def tabelle(titel: str, matrix: np.ndarray, spalten: list[str], stellen: int = 2) -> None:
    """Druckt eine Matrix mit Zeilen- und Spaltennamen."""
    print(titel)
    print(" " * 7 + "".join(f"{s:>11}" for s in spalten))
    for wort, zeile in zip(WOERTER, matrix):
        print(f"{wort:>6} " + "".join(f"{x:11.{stellen}f}" for x in zeile))
    print()


def main() -> None:
    passung, A, neu = aufmerksamkeit(Q, K, V)
    tabelle("1. Passung (Skalarprodukt Frage mal Schlüssel):", passung, WOERTER, 0)
    tabelle("2. Aufmerksamkeit nach der Softmax (Zeile = wer schaut, Spalte = auf wen):",
            A, WOERTER, 3)
    tabelle("3. Neue Wortvektoren (gewichtete Mischung der Werte):",
            neu, ["Person", "Tätigkeit", "Geschlecht", "Verweis"], 3)

    _, A_skaliert, _ = aufmerksamkeit(Q, K, V, skalieren=True)
    tabelle("4. Zum Vergleich: Passung vorher durch Wurzel aus 3 geteilt:",
            A_skaliert, WOERTER, 3)

    # Proben
    assert np.allclose(A.sum(axis=1), 1.0), "jede Zeile muss 1 ergeben"
    assert (A > 0).all(), "die Softmax liefert nur positive Gewichte"
    er, tom = WOERTER.index("er"), WOERTER.index("Tom")
    assert A[er].argmax() == tom, "er soll vor allem auf Tom schauen"
    assert abs(A[er, tom] - 0.840) < 0.001
    assert neu[er, 0] > 0.8, "nach der Mischung trägt er das Merkmal Person"
    assert A_skaliert[er, tom] < A[er, tom], "Skalieren macht die Verteilung weicher"
    print(f"er schaut zu {A[er, tom]:.1%} auf Tom. Alle Proben bestanden.")


if __name__ == "__main__":
    main()
