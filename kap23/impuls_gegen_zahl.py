"""Ein Impulsneuron und ein Rechenneuron bekommen dieselbe Szene.

Begleitprogramm zu Kapitel 23 "Ausblick: Netze und Gehirne".
Läuft mit Python 3.12 oder neuer, nur Standardbibliothek:

    python impuls_gegen_zahl.py

Ein Neuron hat 100 Eingänge. In jeder Szene ist nur ein Teil davon aktiv; jeder aktive
Eingang schickt alle 20 Millisekunden einen Impuls (50 Impulse pro Sekunde).

- Das Rechenneuron aus Teil II bekommt die Stärke jedes Eingangs als Zahl und bildet die
  gewichtete Summe. Dafür rechnet es alle 100 Produkte aus, auch die mit null.
- Das Impulsneuron ist das Integrieren-und-Feuern-Modell aus Kapitel 2. Es tut nur dann
  etwas, wenn ein Impuls ankommt: Es zählt das Gewicht zu seiner Spannung dazu.

Das Programm zählt für jede Szene, wie oft das Impulsneuron feuert und wie viel Arbeit
(Rechenschritte) jedes der beiden Neuronen hatte.
"""

RUHE, SCHWELLE, RESET = -70.0, -55.0, -75.0   # Millivolt, wie in Kapitel 2
LECK, PAUSE = 0.1, 2                          # Leck je ms, Refraktärzeit in ms
EINGAENGE = 100       # Zahl der Eingänge
GEWICHT = 2.0         # Millivolt, die ein ankommender Impuls bringt
TAKT = 20             # ein aktiver Eingang feuert alle 20 ms
DAUER = 200           # eine Szene dauert 200 ms


def impulszeiten(nummer: int) -> list[int]:
    """Regelmäßige Impulse eines aktiven Eingangs, je Eingang etwas versetzt."""
    versatz = nummer % TAKT
    return list(range(versatz, DAUER, TAKT))


def rechenneuron(staerken: list[float]) -> tuple[float, int]:
    """Gewichtete Summe über alle Eingänge; liefert Ergebnis und Zahl der Produkte."""
    summe = sum(GEWICHT * x for x in staerken)
    return summe, len(staerken)


def impulsneuron(aktive: list[int]) -> tuple[int, int]:
    """Integrieren und feuern; liefert Zahl der Spitzen und der verarbeiteten Impulse."""
    ankunft = [0] * DAUER
    for nummer in aktive:
        for t in impulszeiten(nummer):
            ankunft[t] += 1
    v, pause, spitzen = RUHE, 0, 0
    for t in range(DAUER):
        if pause > 0:
            pause -= 1
            continue
        v += LECK * (RUHE - v) + GEWICHT * ankunft[t]
        if v >= SCHWELLE:
            spitzen += 1
            v, pause = RESET, PAUSE
    return spitzen, sum(ankunft)


def szene(anzahl_aktiv: int) -> dict[str, float]:
    """Die ersten anzahl_aktiv Eingänge feuern, die übrigen schweigen."""
    aktive = list(range(anzahl_aktiv))
    staerken = [1 / TAKT if i in aktive else 0.0 for i in range(EINGAENGE)]
    summe, produkte = rechenneuron(staerken)
    spitzen, impulse = impulsneuron(aktive)
    return {"aktiv": anzahl_aktiv, "summe": summe, "produkte": produkte,
            "spitzen": spitzen, "impulse": impulse}


def main() -> None:
    print(f"{EINGAENGE} Eingänge, Szene von {DAUER} ms, aktive Eingänge feuern alle {TAKT} ms.\n")
    print("aktiv | Rechenneuron: Summe  Produkte | Impulsneuron: Spitzen  Impulse verarbeitet")
    ergebnisse = []
    for anzahl in [0, 2, 5, 10, 20, 40, 100]:
        e = szene(anzahl)
        ergebnisse.append(e)
        print(f"{e['aktiv']:5d} | {e['summe']:19.1f} {e['produkte']:9d} |"
              f" {e['spitzen']:21d} {e['impulse']:9d}")

    # Proben
    nach_anzahl = {e["aktiv"]: e for e in ergebnisse}
    assert all(e["produkte"] == EINGAENGE for e in ergebnisse), "Rechenneuron rechnet immer alles"
    assert nach_anzahl[0]["impulse"] == 0 and nach_anzahl[0]["spitzen"] == 0, "Stille kostet nichts"
    assert nach_anzahl[5]["impulse"] < nach_anzahl[5]["produkte"], "wenig Betrieb: weniger Arbeit"
    assert nach_anzahl[2]["spitzen"] == 0, "zu wenig Eingang: Impulsneuron schweigt"
    spitzen = [e["spitzen"] for e in ergebnisse]
    assert spitzen == sorted(spitzen), "mehr aktive Eingänge, nie weniger Spitzen"
    assert nach_anzahl[100]["impulse"] > nach_anzahl[100]["produkte"], "bei viel Betrieb kehrt sich der Vorteil um"
    print("\nAlle Proben bestanden.")


if __name__ == "__main__":
    main()
