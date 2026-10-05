"""Kantenfilter auf einem Ziffernbild und ein Mini-Faltungsnetz, das Striche unterscheidet.

Begleitprogramm zu Kapitel 15 "Faltungsnetze: Maschinen lernen sehen".
Läuft mit Python 3.12 oder neuer und braucht NumPy (pip install numpy):

    python faltungsnetz.py

Teil 1: Das kleine Rechenbeispiel aus dem Kapitel. Eine 3x3-Schablone wandert über ein Bild aus
5x5 Pixeln mit einem senkrechten Strich. Das Programm druckt jede Merkmalskarte als Zahlengitter.

Teil 2: Zwei Kantenfilter (für senkrechte und für waagerechte Kanten) laufen über eine Ziffer 4,
die als Textbild im Programm steht. Die Merkmalskarten werden als Zeichenbild gedruckt:
"+" heißt starke positive Antwort, "-" starke negative, "." keine.

Teil 3: Ein Mini-Faltungsnetz mit zwei lernbaren Schablonen, ReLU, Verdichten (2x2-Maximum) und
einem Ausgabeneuron lernt, waagerechte von senkrechten Strichen zu unterscheiden. Die Bilder sind
8x8 Pixel groß, die Striche liegen an zufälligen Stellen, dazu kommt etwas Rauschen. Das Netz lernt
mit Backpropagation; ein Wackeltest prüft vorher, ob die Steigungen stimmen. Am Ende druckt das
Programm die gelernten Schablonen. Der Startwert ist fest, deshalb ist jeder Lauf gleich.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Bausteine eines Faltungsnetzes
# ---------------------------------------------------------------------------


def falten(bild: np.ndarray, schablone: np.ndarray) -> np.ndarray:
    """Schablone über das Bild schieben; an jeder Stelle Pixel mal Gewicht, alles zusammenzählen."""
    h, b = schablone.shape
    zeilen, spalten = bild.shape[0] - h + 1, bild.shape[1] - b + 1
    karte = np.zeros((zeilen, spalten))
    for i in range(zeilen):
        for j in range(spalten):
            karte[i, j] = np.sum(bild[i:i + h, j:j + b] * schablone)
    return karte


def relu(z: np.ndarray) -> np.ndarray:
    """Negative Werte auf 0 setzen, positive durchlassen."""
    return np.maximum(z, 0.0)


def verdichten(karte: np.ndarray) -> np.ndarray:
    """Je 2x2 Felder zu ihrem größten Wert zusammenfassen (Maximum-Verdichtung)."""
    z, s = karte.shape
    return karte.reshape(z // 2, 2, s // 2, 2).max(axis=(1, 3))


def sigma(z: float) -> float:
    """Die S-Kurve aus Kapitel 4: 1 / (1 + e^(-z))."""
    return 1.0 / (1.0 + np.exp(-z))


def textbild_lesen(text: str) -> np.ndarray:
    """Textbild in ein Zahlengitter verwandeln: '#' wird 1, '.' wird 0."""
    zeilen = [z.strip() for z in text.strip().splitlines()]
    return np.array([[1.0 if c == "#" else 0.0 for c in z] for z in zeilen])


def als_zeichen(karte: np.ndarray, grenze: float = 1.5) -> str:
    """Merkmalskarte als Zeichenbild: '+' über der Grenze, '-' unter minus Grenze, sonst '.'."""
    zeilen = []
    for zeile in karte:
        zeilen.append(" ".join("+" if w > grenze else "-" if w < -grenze else "." for w in zeile))
    return "\n".join(zeilen)


def als_zahlen(karte: np.ndarray) -> str:
    """Zahlengitter mit Vorzeichen drucken (+ 0.0 macht aus -0 eine 0)."""
    return "\n".join(" ".join(f"{w + 0.0:+3.0f}" for w in zeile) for zeile in karte)


# Die beiden Kantenfilter aus dem Kapitel
SENKRECHT = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=float)   # Wert steigt nach rechts: positiv
WAAGERECHT = SENKRECHT.T.copy()                      # Wert steigt nach unten: positiv


# ---------------------------------------------------------------------------
# Teil 1: das kleine Rechenbeispiel
# ---------------------------------------------------------------------------


def teil1() -> None:
    print("=" * 72)
    print("Teil 1: Eine 3x3-Schablone wandert über ein 5x5-Bild mit senkrechtem Strich")
    print("=" * 72)
    bild = np.zeros((5, 5))
    bild[:, 2] = 1.0
    print("Bild:")
    print(als_zahlen(bild))
    print("\nSchablone für senkrechte Kanten:")
    print(als_zahlen(SENKRECHT))

    # Eine Stelle ausführlich: Schablone links oben
    ausschnitt = bild[0:3, 0:3]
    produkte = ausschnitt * SENKRECHT
    print("\nStelle (Zeile 1, Spalte 1): Ausschnitt mal Schablone, Feld für Feld:")
    print(als_zahlen(produkte))
    print(f"Summe = {produkte.sum():+.0f}")

    karte_s = falten(bild, SENKRECHT)
    karte_w = falten(bild, WAAGERECHT)
    print("\nMerkmalskarte senkrechter Filter (3x3):")
    print(als_zahlen(karte_s))
    print("\nMerkmalskarte waagerechter Filter (3x3):")
    print(als_zahlen(karte_w))
    print("\nNach ReLU bleibt vom senkrechten Filter nur die linke Kante des Strichs:")
    print(als_zahlen(relu(karte_s)))

    assert produkte.sum() == 3
    assert np.array_equal(karte_s, np.array([[3, 0, -3]] * 3, dtype=float))
    assert np.all(karte_w == 0)


# ---------------------------------------------------------------------------
# Teil 2: Kantenfilter auf einer Ziffer
# ---------------------------------------------------------------------------

ZIFFER_VIER = """
..........
..#....#..
..#....#..
..#....#..
..#....#..
..######..
.......#..
.......#..
.......#..
..........
"""


def teil2() -> None:
    print("\n" + "=" * 72)
    print("Teil 2: Zwei Kantenfilter auf einer Ziffer 4 (10x10 Pixel)")
    print("=" * 72)
    bild = textbild_lesen(ZIFFER_VIER)
    print("Bild (# = Tinte):")
    print("\n".join(" ".join("#" if w else "." for w in z) for z in bild))

    karte_s = falten(bild, SENKRECHT)
    karte_w = falten(bild, WAAGERECHT)
    print("\nSenkrechter Filter, 8x8 (+ = Wert steigt nach rechts, - = fällt):")
    print(als_zeichen(karte_s))
    print("\nWaagerechter Filter, 8x8 (+ = Wert steigt nach unten, - = fällt):")
    print(als_zeichen(karte_w))
    print("\nWaagerechter Filter als Zahlen:")
    print(als_zahlen(karte_w))

    # Die senkrechten Striche erzeugen kräftige Antworten im senkrechten Filter,
    # der Querbalken in Zeile 6 vor allem im waagerechten.
    assert karte_s.shape == (8, 8)
    assert karte_s[1, 0] == 3 and karte_s[1, 2] == -3          # linker Strich: zwei Kanten
    assert karte_w[3, 3] == 3 and karte_w[5, 3] == -3          # Querbalken: Ober- und Unterkante
    assert np.abs(karte_w[0:2, :]).max() <= 1                  # oben, wo nur Striche sind: schwach
    print(f"\nStärkste Antworten: senkrecht {np.abs(karte_s).max():.0f}, "
          f"waagerecht {np.abs(karte_w).max():.0f}")


# ---------------------------------------------------------------------------
# Teil 3: ein Mini-Faltungsnetz lernt waagerecht gegen senkrecht
# ---------------------------------------------------------------------------

STARTWERT = 15
GROESSE = 8          # Bilder 8x8 Pixel
STRICHLAENGE = 4
RAUSCHEN = 0.3       # Hintergrund zufällig zwischen 0 und 0,3
LERNRATE = 0.5
RUNDEN = 30


def strichbild(rng: np.random.Generator, waagerecht: bool) -> np.ndarray:
    """Bild mit einem Strich der Länge 4 an zufälliger Stelle, dazu etwas Rauschen."""
    bild = rng.uniform(0.0, RAUSCHEN, size=(GROESSE, GROESSE))
    lang = rng.integers(0, GROESSE - STRICHLAENGE + 1)
    quer = rng.integers(0, GROESSE)
    if waagerecht:
        bild[quer, lang:lang + STRICHLAENGE] = 1.0
    else:
        bild[lang:lang + STRICHLAENGE, quer] = 1.0
    return bild


def datensatz(rng: np.random.Generator, anzahl: int) -> list[tuple[np.ndarray, int]]:
    """Gleich viele waagerechte (Sollwert 1) und senkrechte Striche (Sollwert 0), gemischt."""
    daten = [(strichbild(rng, i % 2 == 0), 1 if i % 2 == 0 else 0) for i in range(anzahl)]
    rng.shuffle(daten)
    return daten


def neues_netz(rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "schablonen": rng.normal(0.0, 0.5, size=(2, 3, 3)),   # zwei lernbare 3x3-Schablonen
        "s_bias": np.zeros(2),
        "gewichte": rng.normal(0.0, 0.5, size=18),            # 2 Karten x 3x3 Felder -> Ausgabe
        "bias": np.zeros(1),
    }


def vorwaerts(netz: dict[str, np.ndarray], bild: np.ndarray):
    """Falten, ReLU, verdichten, alles in eine Reihe legen, ein Ausgabeneuron."""
    roh = np.array([falten(bild, k) + b for k, b in zip(netz["schablonen"], netz["s_bias"])])
    aktiv = relu(roh)                                          # 2 Karten 6x6
    dicht = np.array([verdichten(k) for k in aktiv])            # 2 Karten 3x3
    reihe = dicht.ravel()                                       # 18 Zahlen
    y = sigma(reihe @ netz["gewichte"] + netz["bias"][0])
    return y, (roh, aktiv, dicht, reihe)


def fehler(netz: dict[str, np.ndarray], bild: np.ndarray, soll: int) -> float:
    y, _ = vorwaerts(netz, bild)
    return 0.5 * (y - soll) ** 2


def steigungen(netz: dict[str, np.ndarray], bild: np.ndarray, soll: int) -> dict[str, np.ndarray]:
    """Backpropagation durch Ausgabeneuron, Verdichten, ReLU und Faltung."""
    y, (roh, aktiv, dicht, reihe) = vorwaerts(netz, bild)
    delta = (y - soll) * y * (1 - y)                            # Fehlersignal am Ausgang
    g = {"gewichte": delta * reihe, "bias": np.array([delta])}
    d_dicht = (delta * netz["gewichte"]).reshape(2, 3, 3)
    # Verdichten: nur das Feld, das das Maximum war, bekommt das Fehlersignal
    d_aktiv = np.zeros_like(aktiv)
    for k in range(2):
        for i in range(3):
            for j in range(3):
                block = aktiv[k, 2 * i:2 * i + 2, 2 * j:2 * j + 2]
                a, b = np.unravel_index(np.argmax(block), block.shape)
                d_aktiv[k, 2 * i + a, 2 * j + b] = d_dicht[k, i, j]
    d_roh = d_aktiv * (roh > 0)                                 # ReLU lässt nur aktive Stellen durch
    # Faltung: jedes Schablonengewicht hat an allen 36 Stellen mitgewirkt (gemeinsame Gewichte)
    g["schablonen"] = np.array([falten(bild, d_roh[k]) for k in range(2)])
    g["s_bias"] = d_roh.sum(axis=(1, 2))
    return g


def wackeltest(netz: dict[str, np.ndarray], bild: np.ndarray, soll: int) -> float:
    """Jede Steigung mit einem winzigen Dreh am Gewicht nachmessen; größte Abweichung zurückgeben."""
    g = steigungen(netz, bild, soll)
    eps, groesste = 1e-6, 0.0
    for name in netz:
        for idx in np.ndindex(netz[name].shape):
            alt = netz[name][idx]
            netz[name][idx] = alt + eps
            oben = fehler(netz, bild, soll)
            netz[name][idx] = alt - eps
            unten = fehler(netz, bild, soll)
            netz[name][idx] = alt
            groesste = max(groesste, abs((oben - unten) / (2 * eps) - g[name][idx]))
    return groesste


def trefferquote(netz: dict[str, np.ndarray], daten: list[tuple[np.ndarray, int]]) -> float:
    richtig = sum((vorwaerts(netz, bild)[0] > 0.5) == (soll == 1) for bild, soll in daten)
    return richtig / len(daten)


def teil3() -> None:
    print("\n" + "=" * 72)
    print("Teil 3: Ein Mini-Faltungsnetz lernt waagerechte gegen senkrechte Striche")
    print("=" * 72)
    rng = np.random.default_rng(STARTWERT)
    training = datensatz(rng, 200)
    test = datensatz(rng, 200)
    netz = neues_netz(rng)

    beispiel = strichbild(np.random.default_rng(1), waagerecht=True)
    print("Ein Beispielbild (waagerechter Strich; Wert mal 10, abgerundet):")
    print("\n".join(" ".join(f"{min(9, int(w * 10)):d}" for w in z) for z in beispiel))

    anzahl = sum(w.size for w in netz.values())
    print(f"\nLernbare Zahlen im Netz: {anzahl} "
          f"(2 Schablonen x 9 + 2 Bias + 18 Gewichte + 1 Bias)")
    print(f"Ein voll verbundenes Neuron für 8x8 Pixel hätte allein {GROESSE * GROESSE} Gewichte.")

    abweichung = wackeltest(netz, training[0][0], training[0][1])
    print(f"Wackeltest: größte Abweichung zwischen Backpropagation und Nachmessen {abweichung:.1e}")
    assert abweichung < 1e-7

    print(f"\nVor dem Lernen: Trefferquote auf Testbildern {trefferquote(netz, test):.0%}")
    for runde in range(1, RUNDEN + 1):
        summe = 0.0
        for bild, soll in training:
            g = steigungen(netz, bild, soll)
            for name in netz:
                netz[name] -= LERNRATE * g[name]
            summe += fehler(netz, bild, soll)
        if runde in (1, 2, 5, 10, 20, 30):
            print(f"Runde {runde:2d}: mittlerer Fehler {summe / len(training):.4f}   "
                  f"Treffer Training {trefferquote(netz, training):.0%}   "
                  f"Treffer Test {trefferquote(netz, test):.0%}")

    quote_test = trefferquote(netz, test)
    assert quote_test >= 0.95, quote_test

    print("\nGelernte Schablonen (gerundet):")
    for k in range(2):
        print(f"Schablone {k + 1}, Bias {netz['s_bias'][k]:+.2f}:")
        for zeile in netz["schablonen"][k]:
            print("   " + " ".join(f"{w:+5.1f}" for w in zeile))

    # Wie antwortet das fertige Netz auf zwei saubere Striche ohne Rauschen?
    sauber_w = np.zeros((8, 8)); sauber_w[3, 2:6] = 1.0
    sauber_s = np.zeros((8, 8)); sauber_s[2:6, 3] = 1.0
    y_w, y_s = vorwaerts(netz, sauber_w)[0], vorwaerts(netz, sauber_s)[0]
    print(f"\nSauberer waagerechter Strich: Ausgabe {y_w:.3f} (Soll 1)")
    print(f"Sauberer senkrechter Strich:  Ausgabe {y_s:.3f} (Soll 0)")
    assert y_w > 0.5 > y_s

    # Etwas, das das Netz nie gesehen hat: ein schräger Strich. Es muss trotzdem antworten.
    schraeg = np.zeros((8, 8))
    for k in range(4):
        schraeg[2 + k, 2 + k] = 1.0
    y_d = vorwaerts(netz, schraeg)[0]
    print(f"Schräger Strich (nie gesehen): Ausgabe {y_d:.3f} - das Netz kennt nur zwei Antworten")
    assert 0.0 < y_d < 1.0


if __name__ == "__main__":
    teil1()
    teil2()
    teil3()
    print("\nAlle Proben bestanden.")
