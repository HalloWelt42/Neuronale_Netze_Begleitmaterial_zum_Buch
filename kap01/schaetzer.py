"""Ein lernender Schätzer findet aus Beispielen die Regel "Zoll mal Faktor = Zentimeter".

Begleitprogramm zu Kapitel 1 "Was heißt eigentlich Lernen?".
Läuft mit Python 3.12 oder neuer, nur Standardbibliothek:

    python schaetzer.py

Bildschirmgrößen werden in Zoll angegeben. Wie viele Zentimeter ein Zoll hat, verrät dem
Programm niemand. Es bekommt nur Messungen mit dem Maßband (auf halbe Zentimeter gerundet,
also leicht ungenau) und schätzt daraus den Umrechnungsfaktor. Nach jedem Beispiel korrigiert
es seine Schätzung um einen Teil des Fehlers:

    neue Schätzung = alte Schätzung + (Beispielwert - alte Schätzung) / Anzahl der Beispiele

Das Programm druckt jeden Lernschritt und vergleicht am Ende mit der programmierten Regel
1 Zoll = 2,54 cm.
"""

# (Diagonale in Zoll, mit dem Maßband gemessene Diagonale in cm)
MESSUNGEN = [
    (5, 13.0),
    (6, 15.5),
    (10, 25.5),
    (13, 33.0),
    (15, 38.0),
    (24, 61.0),
    (32, 81.5),
    (55, 140.0),
]

REGEL_FAKTOR = 2.54  # die von Hand programmierte Regel, nur zum Vergleich


def lernen(messungen: list[tuple[float, float]], drucken: bool = True) -> list[float]:
    """Gleitender Mittelwert: jede Messung schiebt die Schätzung ein Stück zu ihrem Verhältnis."""
    schaetzung = 0.0
    verlauf = []
    for n, (zoll, cm) in enumerate(messungen, start=1):
        beispielwert = cm / zoll            # was diese eine Messung über den Faktor sagt
        fehler = beispielwert - schaetzung  # wie weit die bisherige Schätzung danebenliegt
        schaetzung += fehler / n            # nur den n-ten Teil des Fehlers übernehmen
        verlauf.append(schaetzung)
        if drucken:
            print(f"Beispiel {n}: {zoll:>2} Zoll = {cm:>5.1f} cm  ->  Verhältnis {beispielwert:.4f}"
                  f"  Fehler {fehler:+.4f}  neue Schätzung {schaetzung:.4f}")
    return verlauf


def vorhersage(faktor: float, zoll: float) -> float:
    """Die gelernte Regel anwenden."""
    return faktor * zoll


if __name__ == "__main__":
    verlauf = lernen(MESSUNGEN)
    faktor = verlauf[-1]
    print(f"\nGelernter Faktor:      {faktor:.4f} cm pro Zoll")
    print(f"Programmierte Regel:   {REGEL_FAKTOR:.4f} cm pro Zoll")
    print(f"Abweichung:            {100 * (faktor - REGEL_FAKTOR) / REGEL_FAKTOR:+.2f} Prozent")

    neu = 27  # ein Bildschirm, den das Programm nie gesehen hat
    print(f"\nNeuer Bildschirm {neu} Zoll: gelernt {vorhersage(faktor, neu):.1f} cm,"
          f" Regel {vorhersage(REGEL_FAKTOR, neu):.1f} cm")

    # Proben
    mittelwert = sum(cm / zoll for zoll, cm in MESSUNGEN) / len(MESSUNGEN)
    assert abs(faktor - mittelwert) < 1e-12, "Der Schätzer muss den Mittelwert treffen."
    assert abs(faktor - REGEL_FAKTOR) < 0.03, "Die gelernte Regel muss nahe an 2,54 liegen."
    assert abs(vorhersage(faktor, neu) - vorhersage(REGEL_FAKTOR, neu)) < 1.0
    print("Proben bestanden.")
