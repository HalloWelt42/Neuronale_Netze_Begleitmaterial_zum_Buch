"""Wortvektoren aus gemeinsamem Vorkommen zählen und nächste Nachbarn finden.

Begleitprogramm zu Kapitel 17 "Wörter als Zahlen".
Läuft mit Python 3.12 oder neuer und NumPy (ab Teil V des Buchs im Einsatz):

    python wortvektoren.py

Das Programm liest einen kleinen, eigens für das Buch geschriebenen Text über einen Königshof,
Menschen, Tiere, Brot und Wetter. Für jedes Wort zählt es, welche anderen Wörter in seiner Nähe
stehen (höchstens zwei Wörter davor oder danach, im selben Satz). Die Zählliste eines Wortes ist
sein Wortvektor: ein Pfeil mit so vielen Richtungen, wie der Text verschiedene Wörter hat.

Das Programm arbeitet in fünf Schritten und druckt jeden davon aus:

1. Zählen mit allen Wörtern: Die kleinen Füllwörter wie "der" und "die" drängen sich vor.
2. Zählen ohne Füllwörter: Ausschnitt der Zähltabelle und die nächsten Nachbarn einiger Wörter,
   gemessen am Kosinus des Winkels zwischen zwei Pfeilen.
3. Pfeilrechnung wie König - Mann + Frau, mit Treffern und einem Fehlschlag.
4. Wortkarte: Alle Pfeile werden mit NumPy auf zwei Zahlen je Wort verdichtet.
5. Personenkarte: nur die acht Personen, mit einer Richtung, die "er" und "sie" trennt.

Kosinus, Skalarprodukt und Länge sind bewusst ohne NumPy geschrieben, damit jeder Rechenschritt
sichtbar bleibt. NumPy kommt nur beim Verdichten zum Einsatz.
"""

import math
import re
from collections import Counter

import numpy as np

FENSTER = 2  # so viele Wörter links und rechts zählen als Nachbarschaft

# Kleine Wörter, die fast überall stehen und über die Bedeutung wenig verraten.
# "er" und "sie" bleiben absichtlich drin: Sie tragen den Unterschied zwischen König und Königin.
FUELLWOERTER = {"der", "die", "das", "den", "dem", "ein", "eine", "im", "in", "auf", "aus", "am",
                "zum", "und", "ist", "wir", "über", "uns", "für", "viel"}

TEXT = """
Der König wohnt im Schloss. Die Königin wohnt im Schloss.
Der König trägt eine goldene Krone. Die Königin trägt eine goldene Krone.
Der König sitzt auf dem Thron. Die Königin sitzt auf dem Thron.
Der König regiert das Land. Die Königin regiert das Land.
Er ist ein strenger König. Sie ist eine kluge Königin.
Der Prinz wohnt im Schloss. Die Prinzessin wohnt im Schloss.
Der Prinz reitet durch das Land. Die Prinzessin reitet durch das Land.
Der Prinz wird später König. Die Prinzessin wird später Königin.
Er ist ein mutiger Prinz. Sie ist eine mutige Prinzessin.
Der Mann geht in die Stadt. Die Frau geht in die Stadt.
Der Mann arbeitet im Garten. Die Frau arbeitet im Garten.
Er ist ein freundlicher Mann. Sie ist eine freundliche Frau.
Der Mann liest die Zeitung. Die Frau liest die Zeitung.
Der Junge spielt im Garten. Das Mädchen spielt im Garten.
Der Junge geht in die Schule. Das Mädchen geht in die Schule.
Er ist ein fröhlicher Junge. Sie ist ein fröhliches Mädchen.
Der Hund bellt im Hof. Die Katze miaut im Hof.
Der Hund frisst aus dem Napf. Die Katze frisst aus dem Napf.
Der Hund schläft auf dem Sofa. Die Katze schläft auf dem Sofa.
Der Hund hat ein weiches Fell. Die Katze hat ein weiches Fell.
Der Hund jagt die Maus. Die Katze jagt die Maus.
Der Bäcker backt das Brot. Der Bäcker backt den Kuchen.
Das Brot ist frisch und warm. Der Kuchen ist frisch und süß.
Wir essen das Brot zum Frühstück. Wir essen den Kuchen am Nachmittag.
Im Brot steckt viel Mehl. Im Kuchen steckt viel Zucker.
Die Sonne kommt hinter den Wolken hervor. Der Regen kommt aus den Wolken.
Heute bringt das Wetter viel Sonne. Morgen bringt das Wetter viel Regen.
Die Sonne ist gut für den Garten. Der Regen ist gut für den Garten.
Wir freuen uns über die Sonne. Wir ärgern uns über den Regen.
Am Himmel ziehen die Wolken. Am Himmel steht die Sonne.
Der Garten liegt hinter dem Haus. Der Hof liegt hinter dem Haus.
Die Kinder spielen im Garten. Die Kinder spielen im Hof.
"""


def saetze(text: str, weglassen: set[str] = frozenset()) -> list[list[str]]:
    """Text in Sätze zerlegen, jeden Satz in kleingeschriebene Wörter; Wörter aus `weglassen` streichen."""
    return [[w for w in re.findall(r"[a-zäöüß]+", satz.lower()) if w not in weglassen]
            for satz in re.split(r"[.!?]", text) if satz.strip()]


def zaehlen(liste_der_saetze: list[list[str]], fenster: int = FENSTER) -> dict[str, Counter]:
    """Für jedes Wort zählen, welche Wörter höchstens `fenster` Plätze entfernt im selben Satz stehen."""
    nachbarn: dict[str, Counter] = {}
    for satz in liste_der_saetze:
        for i, wort in enumerate(satz):
            zaehler = nachbarn.setdefault(wort, Counter())
            for j in range(max(0, i - fenster), min(len(satz), i + fenster + 1)):
                if j != i:
                    zaehler[satz[j]] += 1
    return nachbarn


def wortvektoren(nachbarn: dict[str, Counter]) -> dict[str, list[float]]:
    """Die Zählliste jedes Wortes als Pfeil: eine Zahl je Wort des Vokabulars."""
    vokabular = sorted(nachbarn)
    return {wort: [float(nachbarn[wort][anderes]) for anderes in vokabular] for wort in vokabular}


def skalarprodukt(a: list[float], b: list[float]) -> float:
    """Stelle für Stelle malnehmen und alles zusammenzählen."""
    return sum(x * y for x, y in zip(a, b))


def laenge(a: list[float]) -> float:
    """Länge eines Pfeils nach Pythagoras: Wurzel aus der Summe der Quadrate."""
    return math.sqrt(skalarprodukt(a, a))


def kosinus(a: list[float], b: list[float]) -> float:
    """Kosinus des Winkels zwischen zwei Pfeilen: 1 gleiche Richtung, 0 rechter Winkel."""
    return skalarprodukt(a, b) / (laenge(a) * laenge(b))


def normiert(a: list[float]) -> list[float]:
    """Pfeil auf Länge 1 bringen, die Richtung bleibt."""
    l = laenge(a)
    return [x / l for x in a]


def naechste_nachbarn(ziel: list[float], vektoren: dict[str, list[float]], k: int = 3,
                      ohne: tuple[str, ...] = ()) -> list[tuple[str, float]]:
    """Die k Wörter, deren Pfeile mit dem Ziel den kleinsten Winkel bilden."""
    werte = [(wort, kosinus(ziel, v)) for wort, v in vektoren.items() if wort not in ohne]
    return sorted(werte, key=lambda paar: -paar[1])[:k]


def pfeilrechnung(a: str, minus: str, plus: str, vektoren: dict[str, list[float]], k: int = 3,
                  ausschliessen: bool = True) -> list[tuple[str, float]]:
    """a - minus + plus mit gleich langen Pfeilen; die drei Ausgangswörter scheiden normalerweise aus."""
    va, vm, vp = (normiert(vektoren[w]) for w in (a, minus, plus))
    ziel = [x - y + z for x, y, z in zip(va, vm, vp)]
    return naechste_nachbarn(ziel, vektoren, k, ohne=(a, minus, plus) if ausschliessen else ())


def verdichten(vektoren: dict[str, list[float]], woerter: list[str], richtungen: tuple[int, int] = (0, 1),
               rechts: str = "könig", oben: str = "mann") -> dict[str, tuple[float, float]]:
    """Ausgewählte Pfeile mit NumPy auf zwei Zahlen je Wort verdichten (Singulärwertzerlegung).

    Jeder Pfeil wird zuerst auf Länge 1 gebracht. Dann sucht NumPy der Reihe nach die Richtungen,
    in denen sich die Wörter am stärksten unterscheiden (im Code ab 0 gezählt, Richtung 0 ist die stärkste). Die zwei
    Zahlen eines Wortes sind die Anteile seines Pfeils in den beiden gewählten Richtungen. Alles
    andere geht beim Verdichten verloren. `rechts` und `oben` legen nur fest, welches Wort auf der
    Karte rechts beziehungsweise oben liegt, damit sie bei jedem Lauf gleich aussieht.
    """
    matrix = np.array([normiert(vektoren[w]) for w in woerter])
    matrix -= matrix.mean(axis=0)
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    punkte = (u * s)[:, list(richtungen)]
    if punkte[woerter.index(rechts), 0] < 0:
        punkte[:, 0] *= -1
    if punkte[woerter.index(oben), 1] < 0:
        punkte[:, 1] *= -1
    return {w: (float(x), float(y)) for w, (x, y) in zip(woerter, punkte)}


def zeile(paare: list[tuple[str, float]]) -> str:
    return ", ".join(f"{w} ({c:.2f})" for w, c in paare)


def main() -> None:
    print("1. Zählen mit allen Wörtern")
    roh = zaehlen(saetze(TEXT))
    print(f"   {len(saetze(TEXT))} Sätze, {sum(map(len, saetze(TEXT)))} Wörter, {len(roh)} verschiedene.")
    print("   Häufigste Nachbarn von 'könig': "
          + ", ".join(f"{w} {n}" for w, n in roh["könig"].most_common(5)))
    v_roh = wortvektoren(roh)
    for wort in ["könig", "hund"]:
        print(f"   {wort:<6} -> " + zeile(naechste_nachbarn(v_roh[wort], v_roh, ohne=(wort,))))

    print("\n2. Zählen ohne Füllwörter")
    liste = saetze(TEXT, weglassen=FUELLWOERTER)
    nachbarn = zaehlen(liste)
    vektoren = wortvektoren(nachbarn)
    print(f"   {sum(map(len, liste))} Wörter, {len(vektoren)} verschiedene: "
          f"Jeder Wortvektor hat {len(vektoren)} Stellen.")
    print("   Nachbarn von 'könig': " + ", ".join(f"{w} {n}" for w, n in nachbarn["könig"].most_common()))
    print("   Nachbarn von 'königin': " + ", ".join(f"{w} {n}" for w, n in nachbarn["königin"].most_common()))
    spalten = ["schloss", "regiert", "er", "sie", "stadt", "garten", "napf", "hof"]
    print("   Ausschnitt der Zähltabelle:")
    print("   " + " " * 9 + "".join(f"{s:>8}" for s in spalten))
    for wort in ["könig", "königin", "mann", "frau", "hund", "katze"]:
        print(f"   {wort:<9}" + "".join(f"{nachbarn[wort][s]:>8}" for s in spalten))
    a, b = vektoren["könig"], vektoren["königin"]
    print(f"   Skalarprodukt {skalarprodukt(a, b):.0f}, Längen {laenge(a):.2f} und {laenge(b):.2f},"
          f" Kosinus {kosinus(a, b):.2f}")
    for wort in ["könig", "mann", "hund", "brot", "sonne"]:
        print(f"   {wort:<6} -> " + zeile(naechste_nachbarn(vektoren[wort], vektoren, ohne=(wort,))))

    print("\n3. Pfeilrechnung (die drei Ausgangswörter zählen nicht als Antwort)")
    for a_, m_, p_ in [("könig", "mann", "frau"), ("prinz", "junge", "mädchen"),
                       ("hund", "bellt", "miaut"), ("brot", "mehl", "zucker")]:
        print(f"   {a_} - {m_} + {p_} -> " + zeile(pfeilrechnung(a_, m_, p_, vektoren)))
    print("   Gegenprobe ohne Ausschluss: könig - mann + frau -> "
          + zeile(pfeilrechnung("könig", "mann", "frau", vektoren, ausschliessen=False)))

    print("\n4. Wortkarte: jeder Pfeil auf zwei Zahlen verdichtet (NumPy, Richtungen 1 und 2)")
    karte_woerter = ["könig", "königin", "prinz", "prinzessin", "mann", "frau", "junge", "mädchen",
                     "hund", "katze", "brot", "kuchen", "sonne", "regen", "wolken"]
    karte = verdichten(vektoren, karte_woerter)
    for w, (x, y) in karte.items():
        print(f"   {w:<11} ({x:+.2f}, {y:+.2f})")

    print("\n5. Nur die acht Personen, Richtungen 1 und 4 (die vierte trennt er und sie)")
    personen = ["könig", "königin", "prinz", "prinzessin", "mann", "frau", "junge", "mädchen"]
    personenkarte = verdichten(vektoren, personen, richtungen=(0, 3), rechts="könig", oben="könig")
    for w, (x, y) in personenkarte.items():
        print(f"   {w:<11} ({x:+.2f}, {y:+.2f})")

    # Proben
    assert abs(kosinus(a, a) - 1.0) < 1e-12, "ein Pfeil bildet mit sich selbst den Winkel null"
    assert all(nachbarn[w][v] == nachbarn[v][w] for w in nachbarn for v in nachbarn), "Zählung ist symmetrisch"
    assert naechste_nachbarn(v_roh["hund"], v_roh, 1, ohne=("hund",))[0][0] != "katze", "Füllwörter stören"
    assert naechste_nachbarn(vektoren["hund"], vektoren, 1, ohne=("hund",))[0][0] == "katze"
    assert pfeilrechnung("könig", "mann", "frau", vektoren, 1)[0][0] == "königin"
    assert pfeilrechnung("brot", "mehl", "zucker", vektoren, 1)[0][0] != "kuchen", "nicht jede Rechnung geht auf"
    assert math.dist(karte["könig"], karte["königin"]) < 0.01, "auf der groben Karte fallen beide zusammen"
    assert personenkarte["königin"][1] < 0 < personenkarte["könig"][1], "Richtung 4 trennt er und sie"
    print("\nAlle Proben bestanden.")


if __name__ == "__main__":
    main()
