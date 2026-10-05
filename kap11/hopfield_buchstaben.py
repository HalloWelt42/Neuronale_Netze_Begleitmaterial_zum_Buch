"""Ein Hopfield-Netz merkt sich drei Buchstaben und erinnert sich an einen gestörten.

Begleitprogramm zu Kapitel 11 "Hopfield-Netze: Gedächtnis aus Rückkopplung".
Läuft mit Python 3.12 oder neuer, nur Standardbibliothek:

    python hopfield_buchstaben.py

Was das Programm zeigt:
1. Ein Mini-Netz aus vier Zellen, von Hand nachrechenbar (Beispiel im Kapitel).
2. Drei Buchstaben T, L und X aus je 5x5 = 25 Bildpunkten werden mit der Hebb-Regel
   gespeichert. Jeder Bildpunkt ist eine Zelle: +1 heißt dunkel (#), -1 heißt hell (.).
3. Ein gestörtes T wird Runde für Runde zurück zum echten T geführt; dabei sinkt die Energie.
4. Wie oft das gelingt, wenn immer mehr Bildpunkte zufällig umgedreht sind.
5. Wie viele Zufallsmuster ein Netz aus 100 Zellen sicher halten kann (Speicherkapazität).
"""

import random

# --- Die drei Buchstaben als Textbilder --------------------------------------------
BILDER = {
    "T": ["#####",
          "..#..",
          "..#..",
          "..#..",
          "..#.."],
    "L": ["#....",
          "#....",
          "#....",
          "#....",
          "#####"],
    "X": ["#...#",
          ".#.#.",
          "..#..",
          ".#.#.",
          "#...#"],
}


def als_muster(bild: list[str]) -> list[int]:
    """Textbild zeilenweise in eine Liste aus +1 (dunkel) und -1 (hell) umwandeln."""
    return [1 if zeichen == "#" else -1 for zeile in bild for zeichen in zeile]


def als_bild(muster: list[int], breite: int = 5) -> list[str]:
    """Liste aus +1/-1 wieder als Textbild mit # und . schreiben."""
    zeichen = "".join("#" if z > 0 else "." for z in muster)
    return [zeichen[i:i + breite] for i in range(0, len(zeichen), breite)]


def nebeneinander(bilder: list[list[str]], titel: list[str]) -> str:
    """Mehrere Textbilder mit Überschrift nebeneinander setzen."""
    breite = max(max(len(t) for t in titel), len(bilder[0][0])) + 3
    zeilen = ["".join(t.ljust(breite) for t in titel)]
    for k in range(len(bilder[0])):
        zeilen.append("".join(b[k].ljust(breite) for b in bilder))
    return "\n".join(zeilen)


# --- Der Kern: speichern, Energie, erinnern -----------------------------------------
def lernen(muster_liste):
    """Hebb-Regel: w[i][j] = Summe über alle Muster von m[i] * m[j], Diagonale bleibt 0."""
    n = len(muster_liste[0])
    w = [[0] * n for _ in range(n)]
    for m in muster_liste:
        for i in range(n):
            for j in range(n):
                if i != j:
                    w[i][j] += m[i] * m[j]
    return w


def energie(w, s):
    """E = -1/2 * Summe über alle Paare w[i][j] * s[i] * s[j]."""
    n = len(s)
    return -sum(w[i][j] * s[i] * s[j] for i in range(n) for j in range(n)) / 2


def runde(w, s, protokoll=None):
    """Jede Zelle der Reihe nach: Summe der Nachbarn bilden, Vorzeichen übernehmen."""
    wechsel = 0
    for i in range(len(s)):
        summe = sum(w[i][j] * s[j] for j in range(len(s)))
        neu = 1 if summe > 0 else -1 if summe < 0 else s[i]   # bei 0: bleibt
        if neu != s[i]:
            s[i], wechsel = neu, wechsel + 1
            if protokoll is not None:
                protokoll.append((i, summe, energie(w, s)))
    return wechsel


def erinnern(w, start, drucken=False):
    """Runden wiederholen, bis sich keine Zelle mehr ändert. Gibt Endzustand und Verlauf zurück."""
    s = list(start)
    verlauf = [(0, 0, energie(w, s), list(s))]
    nummer = 0
    while True:
        nummer += 1
        protokoll = [] if drucken else None
        wechsel = runde(w, s, protokoll)
        for i, summe, e in protokoll or []:
            print(f"    Zelle {i + 1:2d} (Zeile {i // 5 + 1}, Spalte {i % 5 + 1}): Summe {summe:+3d},"
                  f" kippt auf {'#' if summe > 0 else '.'}, Energie jetzt {e:6.1f}")
        verlauf.append((nummer, wechsel, energie(w, s), list(s)))
        if drucken:
            print(f"  Runde {nummer}: {wechsel:2d} Zellen gewechselt, Energie {energie(w, s):6.1f}")
        if wechsel == 0:
            return s, verlauf


# --- Teil 1: Mini-Netz aus vier Zellen ----------------------------------------------
def mini_beispiel():
    print("=== 1. Mini-Netz aus vier Zellen ===")
    muster = [1, 1, -1, -1]
    w = lernen([muster])
    for i, zeile in enumerate(w, start=1):
        print(f"  Zelle {i}: Gewichte {zeile}")
    gestoert = [1, -1, -1, -1]
    summe = sum(w[1][j] * gestoert[j] for j in range(4))
    print(f"  Gestört {gestoert}, Energie {energie(w, gestoert)}")
    print(f"  Zelle 2 rechnet: Summe = {summe} -> neuer Zustand {1 if summe > 0 else -1}")
    ende, _ = erinnern(w, gestoert)
    print(f"  Erinnert {ende}, Energie {energie(w, ende)}\n")
    assert w[0] == [0, 1, -1, -1] and summe == 3
    assert ende == muster and energie(w, gestoert) == 0 and energie(w, ende) == -6
    return w


# --- Teil 2 und 3: Buchstaben speichern und ein gestörtes T erinnern -----------------
def buchstaben():
    print("=== 2. Drei Buchstaben speichern ===")
    muster = {name: als_muster(bild) for name, bild in BILDER.items()}
    w = lernen(list(muster.values()))
    print(nebeneinander(list(BILDER.values()), list(BILDER)))
    werte = sorted({w[i][j] for i in range(25) for j in range(25) if i != j})
    print(f"  {25 * 24 // 2} Verbindungen, mögliche Gewichte: {werte}")
    for name, m in muster.items():
        ende, _ = erinnern(w, m)
        print(f"  {name} ist stabil: {ende == m}, Energie {energie(w, m)}")
        assert ende == m
    assert all(w[i][j] == w[j][i] for i in range(25) for j in range(25))

    print("\n=== 3. Ein gestörtes T erinnern ===")
    gestoert = list(muster["T"])
    for punkt in (0, 4, 7, 11, 13, 18):          # sechs Bildpunkte umdrehen (24 Prozent)
        gestoert[punkt] = -gestoert[punkt]
    ende, verlauf = erinnern(w, gestoert, drucken=True)
    bilder = [als_bild(z) for _, _, _, z in verlauf] + [BILDER["T"]]
    titel = [f"Start E={verlauf[0][2]:.0f}"] + [f"R{nr} E={e:.0f}" for nr, _, e, _ in verlauf[1:]] + ["Original"]
    print(nebeneinander(bilder, titel))
    abweichung = sum(a != b for a, b in zip(gestoert, muster["T"]))
    print(f"  Start wich in {abweichung} von 25 Bildpunkten ab, Ende gleich T: {ende == muster['T']}")
    assert ende == muster["T"]
    energien = [e for _, _, e, _ in verlauf]
    assert all(a >= b for a, b in zip(energien, energien[1:]))   # Energie sinkt oder bleibt

    umgedreht = [-z for z in muster["T"]]
    print(f"  Auch das Negativ von T ist stabil: {erinnern(w, umgedreht)[0] == umgedreht}")
    print(nebeneinander([als_bild(umgedreht)], ["Negativ T"]))
    return w, muster


# --- Teil 4: wie viel Rauschen verträgt das Netz? ------------------------------------
def rauschprobe(w, muster, versuche=300):
    print("\n=== 4. Erinnern bei zunehmendem Rauschen ===")
    zufall = random.Random(2026)
    quoten = {}
    for umgedreht in range(0, 13, 2):
        treffer = 0
        for m in muster.values():
            for _ in range(versuche):
                s = list(m)
                for punkt in zufall.sample(range(25), umgedreht):
                    s[punkt] = -s[punkt]
                treffer += erinnern(w, s)[0] == m
        quoten[umgedreht] = treffer / (versuche * len(muster))
        print(f"  {umgedreht:2d} von 25 Punkten umgedreht: richtig erinnert in {100 * quoten[umgedreht]:5.1f} %")
    assert quoten[0] == 1.0 and quoten[4] > 0.95 and quoten[12] < 0.5
    return quoten


# --- Teil 5: Speicherkapazität mit 100 Zellen ----------------------------------------
def kapazitaet(n=100, wiederholungen=20):
    print(f"\n=== 5. Wie viele Zufallsmuster hält ein Netz aus {n} Zellen? ===")
    zufall = random.Random(42)
    anteile = {}
    for anzahl in (5, 10, 15, 20, 25, 30):
        stabil = 0
        for _ in range(wiederholungen):
            muster = [[zufall.choice((-1, 1)) for _ in range(n)] for _ in range(anzahl)]
            w = lernen(muster)
            for m in muster:
                fehler = sum(1 for i in range(n)
                             if sum(w[i][j] * m[j] for j in range(n)) * m[i] <= 0)
                stabil += fehler == 0
        anteile[anzahl] = stabil / (anzahl * wiederholungen)
        print(f"  {anzahl:2d} Muster: {100 * anteile[anzahl]:5.1f} % bleiben ohne einen einzigen Fehler stehen")
    assert anteile[5] > anteile[30]
    return anteile


if __name__ == "__main__":
    mini_beispiel()
    w, muster = buchstaben()
    rauschprobe(w, muster)
    kapazitaet()
    print("\nAlle Proben bestanden.")
