"""Ein lernendes Modell übernimmt die Schieflage aus erfundenen Bewerbungsdaten.

Begleitprogramm zu Kapitel 22 "Grenzen, Risiken, Verantwortung".
Läuft mit Python 3.12 oder neuer, nur Standardbibliothek:

    python bewerbung_schieflage.py

Alle Daten sind erfunden. Das Programm arbeitet in vier Schritten und druckt jeden davon:
1. Es erzeugt 1000 Bewerbungen mit Eignung, Erfahrung und Gruppe (A oder B). Die früheren
   Entscheidungen sind schief: Wer zu Gruppe B gehört, brauchte mehr Punkte für eine Zusage.
2. Ein einzelnes Neuron mit Sigmoid-Ausgabe lernt aus diesen Entscheidungen.
3. Prüfung pro Gruppe: Zusagequote, Trefferquote und ein Zwillingstest, bei dem nur die Gruppe
   vertauscht wird.
4. Gegenprobe: Ein zweites Neuron sieht die Gruppe nicht, wohl aber den Wohnbezirk, der eng mit
   der Gruppe zusammenhängt. Die Schieflage schleicht sich über diese Hintertür wieder ein.
"""

import math
import random

STARTWERT = 22          # fester Zufallsstartwert: jeder Lauf liefert dieselben Zahlen
ANZAHL = 1000
STRAFE_B = 0.25         # so viel mehr Punktsumme brauchte Gruppe B früher für eine Zusage
LERNRATE = 0.5
RUNDEN = 1500


def bewerbungen_erzeugen(anzahl: int) -> list[dict]:
    """Erfundene Bewerbungen. Eignung und Erfahrung sind in beiden Gruppen gleich verteilt."""
    daten = []
    for _ in range(anzahl):
        gruppe_b = 1 if random.random() < 0.5 else 0
        eignung = random.random()                    # 0 = schwach, 1 = sehr stark
        erfahrung = random.random()                  # 0 = keine, 1 = viel
        # Wohnbezirk Nord: in Gruppe B meist ja, in Gruppe A meist nein
        bezirk_nord = 1 if random.random() < (0.85 if gruppe_b else 0.15) else 0
        punkte = 0.6 * eignung + 0.4 * erfahrung + random.gauss(0, 0.05)
        zusage = 1 if punkte > 0.5 + STRAFE_B * gruppe_b else 0   # die alte, schiefe Regel
        daten.append({"eignung": eignung, "erfahrung": erfahrung, "gruppe_b": gruppe_b,
                      "bezirk_nord": bezirk_nord, "zusage": zusage})
    return daten


def sigmoid(s: float) -> float:
    return 1 / (1 + math.exp(-s))


def vorhersage(gewichte, bias, merkmale) -> float:
    """Ein Neuron: gewichtete Summe plus Bias, dann Sigmoid (Wert zwischen 0 und 1)."""
    return sigmoid(sum(w * x for w, x in zip(gewichte, merkmale)) + bias)


def trainieren(daten, spalten):
    """Gradientenabstieg auf dem Kreuzentropie-Fehler, alle Beispiele je Runde."""
    gewichte, bias = [0.0] * len(spalten), 0.0
    for _ in range(RUNDEN):
        summen, summe_b = [0.0] * len(spalten), 0.0
        for zeile in daten:
            x = [zeile[s] for s in spalten]
            fehler = vorhersage(gewichte, bias, x) - zeile["zusage"]
            summen = [g + fehler * xi for g, xi in zip(summen, x)]
            summe_b += fehler
        gewichte = [w - LERNRATE * g / len(daten) for w, g in zip(gewichte, summen)]
        bias -= LERNRATE * summe_b / len(daten)
    return gewichte, bias


def entscheidet_zusage(gewichte, bias, zeile, spalten) -> int:
    return 1 if vorhersage(gewichte, bias, [zeile[s] for s in spalten]) > 0.5 else 0


def pruefung_pro_gruppe(daten, gewichte, bias, spalten) -> dict[str, dict[str, float]]:
    """Zusagequote in den Daten und im Modell sowie Trefferquote, getrennt nach Gruppe."""
    ergebnis = {}
    for name, wert in (("A", 0), ("B", 1)):
        teil = [z for z in daten if z["gruppe_b"] == wert]
        modell = [entscheidet_zusage(gewichte, bias, z, spalten) for z in teil]
        ergebnis[name] = {
            "anzahl": len(teil),
            "quote_daten": sum(z["zusage"] for z in teil) / len(teil),
            "quote_modell": sum(modell) / len(teil),
            "treffer": sum(m == z["zusage"] for m, z in zip(modell, teil)) / len(teil),
        }
    return ergebnis


def zwillingstest(daten, gewichte, bias, spalten) -> int:
    """Wie viele Entscheidungen kippen, wenn man nur die Gruppe vertauscht?"""
    gekippt = 0
    for zeile in daten:
        zwilling = dict(zeile, gruppe_b=1 - zeile["gruppe_b"])
        if entscheidet_zusage(gewichte, bias, zeile, spalten) != \
                entscheidet_zusage(gewichte, bias, zwilling, spalten):
            gekippt += 1
    return gekippt


def gleiche_bewerbung(gewichte, bias, spalten) -> dict[str, float]:
    """Eine mittelstarke Bewerbung (Eignung 0,7, Erfahrung 0,6), einmal als A, einmal als B.

    Der Wohnbezirk folgt dem typischen Fall: A wohnt im Süden, B im Norden.
    """
    ergebnis = {}
    for name, gruppe_b in (("A", 0), ("B", 1)):
        zeile = {"eignung": 0.7, "erfahrung": 0.6, "gruppe_b": gruppe_b, "bezirk_nord": gruppe_b}
        ergebnis[name] = vorhersage(gewichte, bias, [zeile[s] for s in spalten])
    return ergebnis


def bericht(titel, daten, spalten):
    print(f"\n=== {titel} ===")
    print(f"Merkmale: {', '.join(spalten)}")
    gewichte, bias = trainieren(daten, spalten)
    for s, w in zip(spalten, gewichte):
        print(f"  Gewicht {s:<12} {w:+7.2f}")
    print(f"  Bias                 {bias:+7.2f}")
    pruefung = pruefung_pro_gruppe(daten, gewichte, bias, spalten)
    for name, werte in pruefung.items():
        print(f"  Gruppe {name}: {werte['anzahl']:4d} Bewerbungen  Zusagen in den Daten "
              f"{werte['quote_daten']:6.1%}  im Modell {werte['quote_modell']:6.1%}  "
              f"Treffer {werte['treffer']:6.1%}")
    gekippt = zwillingstest(daten, gewichte, bias, spalten)
    print(f"  Zwillingstest: {gekippt} von {len(daten)} Entscheidungen kippen, "
          f"wenn nur die Gruppe vertauscht wird")
    gleich = gleiche_bewerbung(gewichte, bias, spalten)
    print(f"  Gleiche Bewerbung (Eignung 0,7, Erfahrung 0,6): "
          f"A (Süd) {gleich['A']:.2f}, B (Nord) {gleich['B']:.2f}")
    return gewichte, pruefung, gekippt, gleich


if __name__ == "__main__":
    random.seed(STARTWERT)
    daten = bewerbungen_erzeugen(ANZAHL)
    print(f"{len(daten)} erfundene Bewerbungen erzeugt (Startwert {STARTWERT}).")

    # Modell 1 sieht die Gruppe direkt
    g1, p1, kipp1, gleich1 = bericht("Modell 1: sieht die Gruppe", daten,
                                     ["eignung", "erfahrung", "gruppe_b"])
    # Modell 2 sieht die Gruppe nicht, aber den Wohnbezirk
    g2, p2, kipp2, gleich2 = bericht("Modell 2: Gruppe gestrichen, Wohnbezirk bleibt", daten,
                                     ["eignung", "erfahrung", "bezirk_nord"])
    # Modell 3 sieht nur Eignung und Erfahrung
    g3, p3, kipp3, gleich3 = bericht("Modell 3: nur Eignung und Erfahrung", daten,
                                     ["eignung", "erfahrung"])

    # Proben: die Schieflage der Daten taucht im Modell wieder auf
    assert p1["A"]["quote_daten"] > p1["B"]["quote_daten"] + 0.2
    assert g1[2] < 0, "Modell 1 bestraft Gruppe B mit einem negativen Gewicht"
    assert p1["A"]["quote_modell"] > p1["B"]["quote_modell"] + 0.2
    assert kipp1 > 0.1 * ANZAHL
    assert gleich1["A"] > 0.5 > gleich1["B"]
    assert g2[2] < 0, "Modell 2 bestraft den Wohnbezirk als Ersatz für die Gruppe"
    assert p2["A"]["quote_modell"] > p2["B"]["quote_modell"] + 0.05
    assert kipp3 == 0
    print("\nAlle Proben bestanden.")
