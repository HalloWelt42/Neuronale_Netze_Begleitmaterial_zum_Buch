"""Ein Agent lernt ein kleines Labyrinth allein durch Belohnung (Q-Lernen).

Begleitprogramm zu Kapitel 21 "Lernen durch Belohnung".
Läuft mit Python 3.12 oder neuer, nur Standardbibliothek:

    python labyrinth_q_lernen.py

Der Agent startet bei S und kennt das Labyrinth nicht. Er erfährt nur: Am Ziel Z gibt es
+1 Punkt, in der Falle F gibt es -1 Punkt, sonst nichts. Nach jedem Schritt verbessert er
seine Tabelle der Q-Werte (wie gut ist welcher Zug in welchem Feld?). Das Programm druckt,
wie viele Schritte er in ausgewählten Durchläufen braucht, die Wertekarte nach wenigen und
nach vielen Durchläufen und zum Schluss den gelernten Weg als Pfeile.
Der Zufall ist mit einem festen Startwert gesät, deshalb kommt bei jedem Lauf dasselbe heraus.
"""

import random

LABYRINTH = [
    "S..#Z",
    ".#.#.",
    ".#...",
    "...#F",
    "#....",
]
ZUEGE = {"hoch": (-1, 0), "runter": (1, 0), "links": (0, -1), "rechts": (0, 1)}
PFEIL = {"hoch": "^", "runter": "v", "links": "<", "rechts": ">"}
LERNRATE = 0.5      # wie stark eine neue Erfahrung den alten Wert verschiebt
ABSCHLAG = 0.9      # eine Belohnung, die einen Schritt später kommt, zählt nur 90 Prozent
NEUGIER = 0.2       # in 20 Prozent der Fälle probiert der Agent einen zufälligen Zug
STARTWERT = 2016


def finde(zeichen: str) -> tuple[int, int]:
    for z, zeile in enumerate(LABYRINTH):
        if zeichen in zeile:
            return z, zeile.index(zeichen)
    raise ValueError(zeichen)


START, ZIEL, FALLE = finde("S"), finde("Z"), finde("F")


def schritt(feld: tuple[int, int], zug: str) -> tuple[tuple[int, int], float, bool]:
    """Die Umwelt: neues Feld, Belohnung, Durchlauf zu Ende?"""
    dz, ds = ZUEGE[zug]
    z, s = feld[0] + dz, feld[1] + ds
    if not (0 <= z < len(LABYRINTH) and 0 <= s < len(LABYRINTH[0])) or LABYRINTH[z][s] == "#":
        z, s = feld  # gegen Wand oder Rand gelaufen: stehen bleiben
    if (z, s) == ZIEL:
        return (z, s), 1.0, True
    if (z, s) == FALLE:
        return (z, s), -1.0, True
    return (z, s), 0.0, False


def bester_zug(q: dict, feld: tuple[int, int]) -> str:
    return max(ZUEGE, key=lambda zug: q[feld, zug])


def wert(q: dict, feld: tuple[int, int]) -> float:
    """Wert eines Feldes: was der beste Zug von hier aus einbringt."""
    return max(q[feld, zug] for zug in ZUEGE)


def durchlauf(q: dict, zufall: random.Random, neugier: float = NEUGIER,
              max_schritte: int = 200) -> int:
    """Ein Durchlauf vom Start bis zum Ziel oder zur Falle; liefert die Zahl der Schritte."""
    feld = START
    for anzahl in range(1, max_schritte + 1):
        if zufall.random() < neugier:
            zug = zufall.choice(list(ZUEGE))          # erkunden
        else:
            beste = wert(q, feld)                     # ausnutzen; bei Gleichstand losen
            zug = zufall.choice([z for z in ZUEGE if q[feld, z] == beste])
        neu, belohnung, fertig = schritt(feld, zug)
        ziel = belohnung if fertig else belohnung + ABSCHLAG * wert(q, neu)
        q[feld, zug] += LERNRATE * (ziel - q[feld, zug])
        feld = neu
        if fertig:
            return anzahl
    return max_schritte


def leere_tabelle() -> dict:
    return {((z, s), zug): 0.0 for z in range(len(LABYRINTH))
            for s in range(len(LABYRINTH[0])) for zug in ZUEGE}


def wertekarte(q: dict) -> str:
    zeilen = []
    for z, zeile in enumerate(LABYRINTH):
        teile = []
        for s, zeichen in enumerate(zeile):
            if zeichen == "#":
                teile.append(" #### ")
            elif zeichen in "ZF":
                teile.append(f"  {zeichen}   ")
            else:
                teile.append(f"{wert(q, (z, s)):+.2f} ".rjust(6))
        zeilen.append(" ".join(teile))
    return "\n".join(zeilen)


def gelernter_weg(q: dict) -> list[tuple[int, int]]:
    """Ohne Neugier immer den besten Zug nehmen und mitschreiben, wo der Agent landet."""
    feld, weg = START, [START]
    while feld not in (ZIEL, FALLE) and len(weg) < 50:
        feld, _, _ = schritt(feld, bester_zug(q, feld))
        weg.append(feld)
    return weg


def wegkarte(q: dict) -> str:
    weg = set(gelernter_weg(q))
    zeilen = []
    for z, zeile in enumerate(LABYRINTH):
        teile = []
        for s, zeichen in enumerate(zeile):
            if zeichen in "#ZF" or (z, s) not in weg:
                teile.append(zeichen)
            else:
                teile.append(PFEIL[bester_zug(q, (z, s))])
        zeilen.append(" ".join(teile))
    return "\n".join(zeilen)


def trainiere(startwert: int, durchlaeufe: int, neugier: float) -> tuple[dict, list[int]]:
    """Leise trainieren, nur für die Vergleichsversuche am Schluss."""
    zufall, q = random.Random(startwert), leere_tabelle()
    schritte = [durchlauf(q, zufall, neugier) for _ in range(durchlaeufe)]
    return q, schritte


if __name__ == "__main__":
    zufall = random.Random(STARTWERT)
    q = leere_tabelle()
    schritte = []
    for nummer in range(1, 301):
        schritte.append(durchlauf(q, zufall))
        if nummer in (1, 2, 3, 5, 10, 20, 50, 100, 200, 300):
            print(f"Durchlauf {nummer:3d}: {schritte[-1]:3d} Schritte")
        if nummer == 10:
            print("\nWertekarte nach 10 Durchläufen:")
            print(wertekarte(q), "\n")
    print("\nMittlere Schrittzahl je zehn Durchläufe:")
    print(" ".join(f"{sum(schritte[i:i + 10]) / 10:.1f}" for i in range(0, 300, 10)))
    print("\nWertekarte nach 300 Durchläufen:")
    print(wertekarte(q))
    print("\nGelernter Weg (Pfeile ab dem Start oben links):")
    print(wegkarte(q))
    weg = gelernter_weg(q)
    print(f"\nDer Weg hat {len(weg) - 1} Schritte und endet bei {weg[-1]}.")
    print(f"Q-Wert für den Zug in die Falle: {q[(2, 4), 'runter']:+.2f}")

    # Proben
    assert weg[-1] == ZIEL, "der gelernte Weg muss im Ziel enden"
    assert len(weg) - 1 == 8, "der kürzeste Weg durch dieses Labyrinth hat 8 Schritte"
    assert abs(wert(q, START) - ABSCHLAG ** 7) < 0.01, "Wert am Start: 0,9 hoch 7"
    assert wert(q, (1, 4)) > 0.99, "direkt neben dem Ziel ist der Wert fast 1"
    assert q[(2, 4), "runter"] < 0, "der Zug in die Falle muss sich schlecht anfühlen"

    # Vergleich: Wie oft findet der Agent den kürzesten Weg, je nach Neugier?
    print("\nVergleich über 100 Startwerte, je 300 Durchläufe:")
    for neugier in (0.0, 0.05, 0.2, 0.5):
        kurz, spaet = 0, 0
        for s in range(100):
            q_test, schritte_test = trainiere(s, 300, neugier)
            kurz += len(gelernter_weg(q_test)) - 1 == 8
            spaet += sum(schritte_test[-100:]) / 100
        print(f"Neugier {neugier:.2f}: kürzester Weg in {kurz:3d} von 100 Läufen, "
              f"im Training zuletzt {spaet / 100:.1f} Schritte je Durchlauf")
    print("Alle Proben bestanden.")
