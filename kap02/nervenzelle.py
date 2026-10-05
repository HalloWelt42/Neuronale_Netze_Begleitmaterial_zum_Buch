"""Eine Nervenzelle als einfaches Schwellenmodell: integrieren und feuern.

Begleitprogramm zu Kapitel 2 "Das Gehirn als Vorbild".
Läuft mit Python 3.12 oder neuer, nur Standardbibliothek:

    python nervenzelle.py

Das Modell rechnet in Schritten von einer Millisekunde. In jedem Schritt
1. sickert ein Zehntel des Abstands zum Ruhepotential wieder heraus (das "Leck"),
2. kommt der Eingangsstrom hinzu (gewichtete Summe der Synapsen),
3. feuert die Zelle, sobald die Schwelle erreicht ist, und wird danach zurückgesetzt.

Das Programm zeigt drei Dinge aus dem Kapitel:
- Ein schwacher Reiz bleibt unter der Schwelle: keine einzige Spitze (Alles oder nichts).
- Ein stärkerer Reiz erzeugt nicht größere, sondern häufigere Spitzen (Frequenzcode).
- Zwei schwache Synapsen zusammen schaffen, was keine allein schafft; eine hemmende
  Synapse kann das wieder verhindern.
"""

RUHE = -70.0       # Ruhepotential in Millivolt
SCHWELLE = -55.0   # ab hier feuert die Zelle
RESET = -75.0      # nach der Spitze kurz unter dem Ruhepotential
LECK = 0.1         # Anteil des Abstands zur Ruhe, der je Millisekunde verloren geht
PAUSE = 2          # Millisekunden Refraktärzeit: Zelle nimmt nichts an


def simuliere(strom: float, dauer: int = 100) -> tuple[list[float], list[int]]:
    """Membranpotential Millisekunde für Millisekunde; liefert Verlauf und Spitzenzeiten."""
    v, pause = RUHE, 0
    verlauf, spitzen = [], []
    for t in range(dauer):
        if pause > 0:                      # nach einer Spitze: kurz unerregbar
            pause -= 1
        else:
            v += LECK * (RUHE - v) + strom  # integrieren: Leck plus Eingang
            if v >= SCHWELLE:               # feuern: Alles oder nichts
                spitzen.append(t)
                v, pause = RESET, PAUSE
        verlauf.append(v)
    return verlauf, spitzen


def eingangsstrom(gewichte: list[float], aktiv: list[int]) -> float:
    """Gewichtete Summe: jede aktive Synapse trägt ihr Gewicht bei (negativ = hemmend)."""
    return sum(w * x for w, x in zip(gewichte, aktiv))


def zeige_verlauf(strom: float, bis: int = 30) -> None:
    """Druckt die ersten Millisekunden, damit man das Aufladen mitverfolgen kann."""
    verlauf, spitzen = simuliere(strom, bis)
    print(f"Eingangsstrom {strom} mV je ms, die ersten {bis} ms:")
    for t, v in enumerate(verlauf):
        marke = "  <- Spitze, zurückgesetzt" if t in spitzen else ""
        print(f"  t={t:2d} ms  v={v:6.1f} mV{marke}")
    print()


def main() -> None:
    zeige_verlauf(2.5, bis=16)

    print("Wie oft feuert die Zelle in 100 ms?")
    tabelle = {}
    for strom in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]:
        _, spitzen = simuliere(strom)
        tabelle[strom] = len(spitzen)
        print(f"  Strom {strom:3.1f}: {len(spitzen):2d} Spitzen")
    print()

    # Räumliche Summation: zwei erregende Synapsen (Gewicht 1,0) und eine hemmende (-1,5).
    gewichte = [1.0, 1.0, -1.5]
    faelle = {"nur A": [1, 0, 0], "A und B": [1, 1, 0], "A, B und Hemmung": [1, 1, 1]}
    print("Synapsen A, B (erregend, je 1,0) und H (hemmend, -1,5):")
    ergebnis = {}
    for name, aktiv in faelle.items():
        strom = eingangsstrom(gewichte, aktiv)
        _, spitzen = simuliere(strom)
        ergebnis[name] = len(spitzen)
        print(f"  {name:<17} Strom {strom:4.1f}  ->  {len(spitzen):2d} Spitzen in 100 ms")

    # Proben
    assert tabelle[1.0] == 0, "schwacher Reiz bleibt unter der Schwelle"
    assert tabelle[1.5] == 0, "genau an der Grenze: Gleichgewicht liegt auf der Schwelle, nie darüber"
    assert all(tabelle[a] <= tabelle[b] for a, b in [(2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, 6.0)])
    assert tabelle[6.0] > tabelle[2.0], "stärkerer Reiz, häufigere Spitzen"
    assert ergebnis["nur A"] == 0 and ergebnis["A und B"] > 0 and ergebnis["A, B und Hemmung"] == 0
    verlauf, _ = simuliere(6.0)
    assert max(verlauf) < SCHWELLE, "jede Spitze wird sofort zurückgesetzt"
    print("\nAlle Proben bestanden.")


if __name__ == "__main__":
    main()
