"""Ein Fuzzy-Regler stellt das Ventil einer Heizung nach der Raumtemperatur ein.

Begleitprogramm zu Kapitel 13 "Weitere Wege: ART, Fuzzy und Radialbasis".
Läuft mit Python 3.12 oder neuer, nur Standardbibliothek:

    python fuzzy_heizung.py

Der Regler arbeitet in drei Schritten und druckt jeden davon:
1. Unscharf machen: Wie sehr ist die Temperatur kalt, warm, heiß?
2. Regeln auswerten: Wie stark gilt jede Wenn-dann-Regel, wie weit wird ihre Ausgabe gekappt?
3. Scharf machen: Der Schwerpunkt der gekappten Ausgabe-Flächen ist die Ventilstellung.
Zum Schluss vergleicht eine Kennlinie den Fuzzy-Regler mit einem einfachen Thermostat.
"""


def rampe_ab(x: float, a: float, b: float) -> float:
    """1 bis a, fällt bis b linear auf 0."""
    return 1.0 if x <= a else 0.0 if x >= b else (b - x) / (b - a)


def rampe_auf(x: float, a: float, b: float) -> float:
    """0 bis a, steigt bis b linear auf 1."""
    return 1.0 - rampe_ab(x, a, b)


def dreieck(x: float, a: float, m: float, b: float) -> float:
    """0 außerhalb von a bis b, Spitze 1 bei m."""
    if x <= a or x >= b:
        return 0.0
    return (x - a) / (m - a) if x <= m else (b - x) / (b - m)


# Eingang: Raumtemperatur in Grad Celsius
TEMPERATUR = {
    "kalt": lambda t: rampe_ab(t, 15, 20),
    "warm": lambda t: dreieck(t, 15, 20, 25),
    "heiß": lambda t: rampe_auf(t, 20, 25),
}
# Ausgang: Ventilstellung in Prozent (0 = zu, 100 = ganz offen)
VENTIL = {
    "zu": lambda v: rampe_ab(v, 0, 30),
    "halb": lambda v: dreieck(v, 30, 50, 70),
    "auf": lambda v: rampe_auf(v, 70, 100),
}
REGELN = [("kalt", "auf"), ("warm", "halb"), ("heiß", "zu")]
SCHRITT = 0.5  # Streifenbreite für den Schwerpunkt in Prozentpunkten


def unscharf(t: float) -> dict[str, float]:
    """Schritt 1: Zugehörigkeitsgrade der Temperatur zu kalt, warm und heiß."""
    return {name: f(t) for name, f in TEMPERATUR.items()}


def regeln_auswerten(grade: dict[str, float]) -> dict[str, float]:
    """Schritt 2: Jede Regel gibt ihren Erfüllungsgrad an ihre Ausgabe-Menge weiter."""
    kappung = {name: 0.0 for name in VENTIL}
    for wenn, dann in REGELN:
        kappung[dann] = max(kappung[dann], grade[wenn])
    return kappung


def ausgabe_flaeche(kappung: dict[str, float], v: float) -> float:
    """Höhe der zusammengesetzten Ausgabe-Fläche an der Ventilstellung v."""
    return max(min(kappung[name], f(v)) for name, f in VENTIL.items())


def scharf(kappung: dict[str, float]) -> float:
    """Schritt 3: Schwerpunkt der Ausgabe-Fläche.

    Die Fläche wird in schmale Streifen der Breite SCHRITT zerlegt; jeder Streifen zählt mit
    seiner Höhe in der Streifenmitte. Schwerpunkt = Summe(Stelle * Höhe) / Summe(Höhe).
    """
    stellen = [(i + 0.5) * SCHRITT for i in range(int(100 / SCHRITT))]
    hoehen = [ausgabe_flaeche(kappung, v) for v in stellen]
    return sum(v * h for v, h in zip(stellen, hoehen)) / sum(hoehen)


def regler(t: float) -> float:
    return scharf(regeln_auswerten(unscharf(t)))


def thermostat(t: float, sollwert: float = 20.0) -> float:
    """Zum Vergleich: ein Schalter, der unter dem Sollwert ganz öffnet, sonst ganz schließt."""
    return 100.0 if t < sollwert else 0.0


def schritte_drucken(t: float) -> float:
    print(f"Raumtemperatur {t} Grad")
    grade = unscharf(t)
    print("1. Unscharf machen:  " + "   ".join(f"{n} {g:.2f}" for n, g in grade.items()))
    kappung = regeln_auswerten(grade)
    print("2. Regeln auswerten:")
    for wenn, dann in REGELN:
        print(f"   Wenn {wenn:<4} dann Ventil {dann:<4}  gilt zu {grade[wenn]:.2f}"
              f"  -> Menge '{dann}' gekappt bei {kappung[dann]:.2f}")
    stellung = scharf(kappung)
    spitzen = {"zu": 0, "halb": 50, "auf": 100}
    grob = sum(kappung[n] * spitzen[n] for n in VENTIL) / sum(kappung.values())
    print(f"3. Scharf machen:     Schwerpunkt der Fläche = {stellung:.1f} Prozent")
    print(f"   (Kopfrechnung mit den Spitzen 0, 50, 100: {grob:.1f} Prozent)\n")
    return stellung


if __name__ == "__main__":
    stellung = schritte_drucken(18.5)
    assert abs(unscharf(18.5)["kalt"] - 0.3) < 1e-9 and abs(unscharf(18.5)["warm"] - 0.7) < 1e-9
    assert abs(stellung - 61.0) < 0.05, "von Hand: 1576,45 / 25,85 = 60,98"

    print("Kennlinie: Temperatur, Fuzzy-Regler, Thermostat (Ventil in Prozent)")
    werte = []
    for zehntel in range(140, 281, 10):
        t = zehntel / 10
        werte.append(regler(t))
        print(f"   {t:4.1f} Grad   Fuzzy {werte[-1]:5.1f}   Thermostat {thermostat(t):5.1f}")
    assert all(a >= b - 1e-9 for a, b in zip(werte, werte[1:])), "je wärmer, desto weiter zu"
    assert abs(regler(20.0) - 50.0) < 1e-9, "genau warm: Ventil halb offen"
    assert abs(regler(14.0) + regler(28.0) - 100.0) < 1e-9, "kalt und heiß spiegelbildlich"
    assert abs(regler(14.0) - 90.0) < 0.05, "ganz kalt: Schwerpunkt der Rampe 70 bis 100"
    print("\nAlle Proben bestanden.")
