"""Logikgatter aus McCulloch-Pitts-Zellen und ein kleines Beispiel zu Hebbs Lernregel.

Begleitprogramm zu Kapitel 3 "Die ersten Modelle: 1943 bis 1958".
Läuft mit Python 3.12 oder neuer, nur Standardbibliothek:

    python logikgatter.py

Teil 1 baut UND, ODER und NICHT aus je einer McCulloch-Pitts-Zelle und setzt daraus
ein kleines Netz für "entweder oder" (XOR) zusammen. Für jedes Gatter wird die
Wahrheitstafel gedruckt und mit dem Sollwert verglichen.

Teil 2 zeigt Hebbs Regel an einem Neuron mit zwei Eingängen: Ein Ton, der immer
zusammen mit Futter kommt, bekommt Schritt für Schritt eine stärkere Verbindung,
bis er das Neuron allein auslösen kann. Danach wächst das Gewicht ungebremst weiter.
"""

from itertools import product


# --- Teil 1: McCulloch-Pitts-Zellen -------------------------------------------------

def mp_zelle(erregend: list[int], hemmend: list[int], schwelle: int) -> int:
    """Eine McCulloch-Pitts-Zelle.

    Ist auch nur ein hemmender Eingang aktiv, bleibt die Zelle still (Ausgabe 0).
    Sonst werden die erregenden Eingänge gezählt; erreicht die Zahl die Schwelle,
    feuert die Zelle (Ausgabe 1).
    """
    if any(hemmend):
        return 0
    return 1 if sum(erregend) >= schwelle else 0


def und(x1: int, x2: int) -> int:
    return mp_zelle([x1, x2], [], schwelle=2)


def oder(x1: int, x2: int) -> int:
    return mp_zelle([x1, x2], [], schwelle=1)


def nicht(x: int) -> int:
    return mp_zelle([], [x], schwelle=0)


def xor(x1: int, x2: int) -> int:
    """Zwei Stufen: "x1 und nicht x2", "x2 und nicht x1", dann ODER."""
    links = mp_zelle([x1], [x2], schwelle=1)
    rechts = mp_zelle([x2], [x1], schwelle=1)
    return oder(links, rechts)


SOLL = {
    "UND": lambda a, b: a and b,
    "ODER": lambda a, b: a or b,
    "XOR": lambda a, b: a != b,
}


def wahrheitstafel(name: str, gatter) -> None:
    """Druckt die Wahrheitstafel und prüft jede Zeile gegen den Sollwert."""
    print(f"{name}:")
    for x1, x2 in product((0, 1), repeat=2):
        ist = gatter(x1, x2)
        assert ist == int(SOLL[name](x1, x2)), (name, x1, x2, ist)
        print(f"  x1={x1}  x2={x2}  ->  {ist}")


# --- Teil 2: Hebbs Regel --------------------------------------------------------------

LERNRATE = 0.25
SCHWELLE = 1.0


def feuert(gewichte: dict[str, float], eingabe: dict[str, int]) -> int:
    """Gewichtete Summe der Eingänge mit der Schwelle vergleichen."""
    summe = sum(gewichte[name] * eingabe[name] for name in gewichte)
    return 1 if summe >= SCHWELLE else 0


def hebb_schritt(gewichte: dict[str, float], eingabe: dict[str, int]) -> int:
    """Hebbs Regel: Gewicht += Lernrate * Ausgabe der Vorgängerzelle * Aktivität der Zelle."""
    aktiv = feuert(gewichte, eingabe)
    for name in gewichte:
        gewichte[name] += LERNRATE * eingabe[name] * aktiv
    return aktiv


def hebb_versuch(durchgaenge: int = 8) -> dict[str, float]:
    gewichte = {"Futter": 1.0, "Ton": 0.0}
    ton_allein = {"Futter": 0, "Ton": 1}
    beides = {"Futter": 1, "Ton": 1}
    print(f"Start: Futter-Gewicht {gewichte['Futter']:.2f}, Ton-Gewicht {gewichte['Ton']:.2f}, "
          f"Ton allein löst aus: {'ja' if feuert(gewichte, ton_allein) else 'nein'}")
    for n in range(1, durchgaenge + 1):
        hebb_schritt(gewichte, beides)
        print(f"  nach Durchgang {n}: Futter {gewichte['Futter']:.2f}  Ton {gewichte['Ton']:.2f}"
              f"  Ton allein löst aus: {'ja' if feuert(gewichte, ton_allein) else 'nein'}")
    return gewichte


if __name__ == "__main__":
    print("Teil 1: Logikgatter aus McCulloch-Pitts-Zellen\n")
    wahrheitstafel("UND", und)
    wahrheitstafel("ODER", oder)
    wahrheitstafel("XOR", xor)
    print("NICHT:")
    for x in (0, 1):
        assert nicht(x) == 1 - x
        print(f"  x={x}  ->  {nicht(x)}")

    print("\nTeil 2: Hebbs Regel, Ton und Futter kommen immer zusammen\n")
    g = hebb_versuch()
    # Nach vier Durchgängen reicht der Ton allein, danach wächst das Gewicht weiter.
    assert g["Ton"] == 8 * LERNRATE and g["Futter"] == 1.0 + 8 * LERNRATE
    assert feuert(g, {"Futter": 0, "Ton": 1}) == 1
    print("\nAlle Proben bestanden.")
