"""Ein Bigramm-Sprachmodell lernt aus einem kleinen Märchen und erzeugt neue Sätze.

Begleitprogramm zu Kapitel 19 "Große Sprachmodelle".
Läuft mit Python 3.12 oder neuer, nur Standardbibliothek:

    python bigramm_maerchen.py

Das Modell zählt, welches Wort wie oft auf welches folgt. Aus den Zählungen werden
Wahrscheinlichkeiten für das nächste Wort. Beim Erzeugen zieht das Programm Wort für Wort
aus diesen Wahrscheinlichkeiten; die Temperatur bestimmt, wie mutig es dabei zieht.
Der Zufallsstartwert ist fest, deshalb erscheinen bei jedem Lauf dieselben Sätze.
"""

import random
import re
from collections import Counter, defaultdict

STARTWERT = 7

MAERCHEN = """
Es war einmal ein kleiner Fuchs, der wohnte am Rand eines großen Waldes.
Der Fuchs war klug, aber er war auch sehr neugierig.
Eines Morgens fand der Fuchs einen goldenen Schlüssel im Moos.
Der Schlüssel glänzte in der Sonne wie ein Stern.
Der Fuchs nahm den Schlüssel und lief zu der alten Eule.
Die Eule wohnte in einer hohlen Eiche mitten im Wald.
Die Eule sah den Schlüssel lange an und sagte, er gehöre zu einer Tür im Berg.
Also wanderte der Fuchs zum Berg, und die Eule flog über ihm.
Am Berg wartete ein Bär, der war groß und sehr müde.
Der Bär fragte den Fuchs, was er im Wald suche.
Der Fuchs zeigte ihm den Schlüssel, und der Bär staunte.
Gemeinsam fanden sie die Tür im Berg, und der Schlüssel passte.
Hinter der Tür lag kein Gold, sondern ein Garten voller Blumen.
Im Garten sang ein Vogel, und der Bär war nicht mehr müde.
Der Fuchs, die Eule und der Bär blieben bis zum Abend im Garten.
Am Abend gingen sie zurück in den Wald.
Der Fuchs legte den Schlüssel unter das Moos, damit ihn ein anderes Tier finden kann.
Und wenn sie nicht gestorben sind, dann leben sie noch heute.
"""


def in_tokens(text: str) -> list[str]:
    """Text in Tokens zerlegen: Wörter (Groß- und Kleinschreibung bleibt) sowie Komma und Punkt."""
    return re.findall(r"[A-Za-zÄÖÜäöüß]+|[.,]", text)


def zaehlen(tokens: list[str]) -> dict[str, Counter]:
    """Für jedes Wort zählen, welche Wörter wie oft direkt danach kommen.

    Der Punkt steht für ein Satzende; das Wort danach beginnt einen neuen Satz.
    """
    folgen: dict[str, Counter] = defaultdict(Counter)
    for vorher, nachher in zip(["."] + tokens, tokens):
        folgen[vorher][nachher] += 1
    return folgen


def wahrscheinlichkeiten(zaehler: Counter, temperatur: float = 1.0) -> dict[str, float]:
    """Zählungen in Wahrscheinlichkeiten umrechnen, mit Temperatur geschärft oder geglättet.

    Jede Anzahl wird hoch 1/Temperatur genommen und dann durch die Summe geteilt.
    Temperatur 1: genau die Anteile aus dem Text. Kleiner als 1: das Häufigste wird noch
    häufiger. Größer als 1: Seltenes kommt öfter zum Zug.
    """
    gewichte = {wort: anzahl ** (1 / temperatur) for wort, anzahl in zaehler.items()}
    summe = sum(gewichte.values())
    return {wort: g / summe for wort, g in gewichte.items()}


def satz_erzeugen(folgen: dict[str, Counter], temperatur: float, zufall: random.Random,
                  hoechstens: int = 20) -> str:
    """Ab einem Satzanfang Wort für Wort ziehen, bis ein Punkt kommt."""
    wort, satz = ".", []
    for _ in range(hoechstens):
        p = wahrscheinlichkeiten(folgen[wort], temperatur)
        wort = zufall.choices(list(p), weights=list(p.values()))[0]
        if wort == ".":
            return " ".join(satz).replace(" ,", ",") + "."
        satz.append(wort)
    return " ".join(satz).replace(" ,", ",") + " ..."  # abgebrochen: kein Satzende gefunden


def tabelle_drucken(folgen: dict[str, Counter], wort: str, temperatur: float = 1.0) -> None:
    """Die Wahrscheinlichkeiten für das Wort nach `wort` als kleine Balkentabelle drucken."""
    p = wahrscheinlichkeiten(folgen[wort], temperatur)
    print(f'Nach "{wort}" (Temperatur {temperatur}), {sum(folgen[wort].values())} Fälle im Text:')
    for nachfolger, wert in sorted(p.items(), key=lambda paar: (-paar[1], paar[0])):
        print(f"  {nachfolger:<10} {folgen[wort][nachfolger]:>2}x  {wert:6.1%}  {'#' * round(wert * 40)}")
    print()


if __name__ == "__main__":
    tokens = in_tokens(MAERCHEN)
    folgen = zaehlen(tokens)
    print(f"Der Text hat {len(tokens)} Tokens, davon {len(set(tokens))} verschiedene.")
    print(f"Das Modell kennt {sum(len(z) for z in folgen.values())} verschiedene Wortpaare.\n")

    # Proben: die Wahrscheinlichkeiten nach jedem Wort ergeben zusammen 1 (also 100 Prozent).
    for wort in folgen:
        for t in (0.3, 1.0, 2.0):
            assert abs(sum(wahrscheinlichkeiten(folgen[wort], t).values()) - 1) < 1e-9

    tabelle_drucken(folgen, "der")
    tabelle_drucken(folgen, "der", temperatur=0.5)
    tabelle_drucken(folgen, "der", temperatur=2.0)

    # Probe: niedrige Temperatur bevorzugt das häufigste Wort stärker, hohe schwächer.
    p_kalt = wahrscheinlichkeiten(folgen["der"], 0.5)["Bär"]
    p_normal = wahrscheinlichkeiten(folgen["der"], 1.0)["Bär"]
    p_heiss = wahrscheinlichkeiten(folgen["der"], 2.0)["Bär"]
    assert p_kalt > p_normal > p_heiss

    zufall = random.Random(STARTWERT)
    neue_saetze = 0
    for temperatur in (0.3, 1.0, 2.0):
        print(f"Drei Sätze mit Temperatur {temperatur}:")
        for _ in range(3):
            satz = satz_erzeugen(folgen, temperatur, zufall)
            neu = satz not in MAERCHEN
            neue_saetze += neu
            print(f"  {satz}" + ("   (so nicht im Märchen)" if neu else ""))
        print()

    # Probe: jedes erzeugte Wortpaar stammt aus dem Märchen; das Modell kennt nichts anderes.
    probe = in_tokens(satz_erzeugen(folgen, 1.0, random.Random(STARTWERT)))
    assert all(folgen[a][b] > 0 for a, b in zip(["."] + probe, probe))
    print(f"{neue_saetze} von 9 Sätzen stehen so nicht im Märchen,")
    print("obwohl jedes einzelne Wortpaar daraus stammt.")
