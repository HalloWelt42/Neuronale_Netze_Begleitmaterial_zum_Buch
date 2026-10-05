"""Eindimensionale Diffusion: Punkte verrauschen und wieder ordnen.

Begleitprogramm zu Kapitel 20 "Bilder erzeugen".
Läuft mit Python 3.12 oder neuer und NumPy:

    python diffusion_punkte.py

Statt eines Bildes nehmen wir 2000 Zahlen auf einer Linie. Sie bilden zwei Häufchen, eines bei -2
und eines bei +2. Das ist unser "Bild". Das Programm

1. verrauscht die Punkte in 40 kleinen Schritten, bis nur noch eine formlose Wolke übrig ist,
2. trainiert ein kleines Netz (eine verdeckte Schicht mit 32 Neuronen), das zu einem verrauschten
   Punkt und der Schrittnummer schätzt, welches Rauschen darin steckt,
3. erzeugt mit diesem Netz neue Punkte: Es startet mit reinem Rauschen und zieht in 40 Schritten
   jeweils ein Stück des geschätzten Rauschens ab.

Zur Kontrolle rechnet das Programm die Entrauschungsrichtung auch exakt aus. Das geht nur, weil wir
die Form der Daten (zwei glockenförmige Häufchen) selbst festgelegt haben; bei echten Bildern kennt
niemand diese Formel, dort bleibt nur das gelernte Netz.

Alle Zufallszahlen kommen aus einem festen Startwert, jeder Lauf druckt dieselben Zahlen.
"""

import numpy as np

STARTWERT = 20
SCHRITTE = 40                                  # T: so viele Rauschschritte
BETA = np.linspace(0.001, 0.2, SCHRITTE)       # Rauschmenge je Schritt
ALPHA = 1.0 - BETA
ALPHA_QUER = np.cumprod(ALPHA)                 # wie viel vom Original nach t Schritten übrig ist
MITTE, BREITE = 2.0, 0.3                       # Häufchen bei -2 und +2, Streuung 0,3

zufall = np.random.default_rng(STARTWERT)


def daten(anzahl: int) -> np.ndarray:
    """Die "Bilder": Zahlen aus zwei Häufchen bei -MITTE und +MITTE."""
    seite = zufall.choice([-1.0, 1.0], size=anzahl)
    return seite * MITTE + BREITE * zufall.standard_normal(anzahl)


def verrauschen(x0: np.ndarray, t: np.ndarray, rauschen: np.ndarray) -> np.ndarray:
    """Hinweg in einem Sprung: Original schrumpfen, Rauschen dazu (t zählt von 1 bis SCHRITTE)."""
    a = ALPHA_QUER[t - 1]
    return np.sqrt(a) * x0 + np.sqrt(1.0 - a) * rauschen


# ---------------------------------------------------------------- das kleine Netz
def merkmale(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Eingabe des Netzes: der verrauschte Wert und die Schrittnummer, auf 0 bis 1 skaliert."""
    return np.stack([x, t / SCHRITTE], axis=1)


class Entrauscher:
    """Netz mit einer verdeckten Schicht (tanh), das das enthaltene Rauschen schätzt."""

    def __init__(self, verdeckt: int = 32):
        self.W1 = zufall.normal(0.0, 1.0, (2, verdeckt))
        self.b1 = np.zeros(verdeckt)
        self.W2 = zufall.normal(0.0, 0.1, (verdeckt, 1))
        self.b2 = np.zeros(1)

    def __call__(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        h = np.tanh(merkmale(x, t) @ self.W1 + self.b1)
        return (h @ self.W2 + self.b2)[:, 0]

    def lernschritt(self, x0: np.ndarray, lernrate: float) -> float:
        """Ein Schritt Gradientenabstieg auf den mittleren quadratischen Fehler."""
        t = zufall.integers(1, SCHRITTE + 1, size=len(x0))
        rauschen = zufall.standard_normal(len(x0))
        xt = verrauschen(x0, t, rauschen)
        e = merkmale(xt, t)
        h = np.tanh(e @ self.W1 + self.b1)
        schaetzung = (h @ self.W2 + self.b2)[:, 0]
        d = 2.0 * (schaetzung - rauschen)[:, None] / len(x0)    # Ableitung des Fehlers
        dh = (d @ self.W2.T) * (1.0 - h**2)                      # rückwärts durch tanh
        self.W2 -= lernrate * h.T @ d
        self.b2 -= lernrate * d.sum(axis=0)
        self.W1 -= lernrate * e.T @ dh
        self.b1 -= lernrate * dh.sum(axis=0)
        return float(np.mean((schaetzung - rauschen) ** 2))


def exaktes_rauschen(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Bestmögliche Schätzung des Rauschens, nur möglich, weil wir die Datenform kennen."""
    a = ALPHA_QUER[t - 1]
    varianz = a * BREITE**2 + (1.0 - a)
    anteil_rechts = np.tanh(np.sqrt(a) * MITTE * x / varianz)       # zwischen -1 und +1
    mitte_geschaetzt = MITTE * anteil_rechts
    x0_geschaetzt = mitte_geschaetzt + np.sqrt(a) * BREITE**2 / varianz * (x - np.sqrt(a) * mitte_geschaetzt)
    return (x - np.sqrt(a) * x0_geschaetzt) / np.sqrt(1.0 - a)


# ---------------------------------------------------------------- Rückweg
def erzeugen(schaetzer, anzahl: int) -> np.ndarray:
    """Rückweg: aus reinem Rauschen Schritt für Schritt Punkte formen."""
    x = zufall.standard_normal(anzahl)      # reines Rauschen
    for t in range(SCHRITTE, 0, -1):
        tt = np.full(anzahl, t)
        rauschen = schaetzer(x, tt)         # Netz schätzt das Rauschen
        x = (x - BETA[t-1] / np.sqrt(1.0 - ALPHA_QUER[t-1]) * rauschen) \
            / np.sqrt(ALPHA[t-1])
        if t > 1:                           # frischer Zufall, außer am Ende
            x += np.sqrt(BETA[t-1]) * zufall.standard_normal(anzahl)
    return x


# ---------------------------------------------------------------- Ausgabe als Textbild
def textbild(x: np.ndarray, titel: str) -> None:
    """Häufigkeiten in 16 Fächern von -4 bis +4 als Balken aus Rautezeichen."""
    zaehler, kanten = np.histogram(x, bins=16, range=(-4.0, 4.0))
    print(titel)
    for z, links in zip(zaehler, kanten[:-1]):
        print(f"  {links:+5.1f} | {'#' * int(round(z / 20))}")


def kennzahlen(x: np.ndarray) -> tuple[float, float, float, float]:
    """Anteil links, Anteil in der leeren Mitte, Mittelwerte der beiden Häufchen."""
    links = float(np.mean(x < 0))
    mitte = float(np.mean(np.abs(x) < 1.0))
    return links, mitte, float(x[x < 0].mean()), float(x[x > 0].mean())


if __name__ == "__main__":
    original = daten(2000)
    textbild(original, "Die Daten: zwei Häufchen")

    print("\nHinweg: Schritt für Schritt verrauschen")
    for t in (10, 20, 30, 40):
        xt = verrauschen(original, np.full(len(original), t), zufall.standard_normal(len(original)))
        print(f"  nach Schritt {t:2d}: vom Original übrig {np.sqrt(ALPHA_QUER[t - 1]):.2f}, "
              f"Streuung der Wolke {xt.std():.2f}")
    textbild(xt, "Nach 40 Schritten: kaum noch Form")

    print("\nTraining: das Netz lernt, das Rauschen zu schätzen")
    netz = Entrauscher()
    for schritt in range(1, 6001):
        fehler = netz.lernschritt(daten(256), lernrate=0.05)
        if schritt in (1, 1000, 3000, 6000):
            print(f"  Lernschritt {schritt:4d}: mittlerer quadratischer Fehler {fehler:.3f}")
    probe = daten(20000)
    t_probe = zufall.integers(1, SCHRITTE + 1, size=len(probe))
    r_probe = zufall.standard_normal(len(probe))
    x_probe = verrauschen(probe, t_probe, r_probe)
    fehler_netz = float(np.mean((netz(x_probe, t_probe) - r_probe) ** 2))
    fehler_exakt = float(np.mean((exaktes_rauschen(x_probe, t_probe) - r_probe) ** 2))
    print(f"  Probe mit 20000 neuen Punkten: Netz {fehler_netz:.3f}, "
          f"exakte Formel {fehler_exakt:.3f} (besser geht es nicht)")

    neu = erzeugen(netz, 2000)
    textbild(neu, "\nRückweg mit dem gelernten Netz: neue Punkte")
    exakt = erzeugen(exaktes_rauschen, 2000)

    for name, x in (("Daten", original), ("gelerntes Netz", neu), ("exakte Richtung", exakt)):
        links, mitte, m_links, m_rechts = kennzahlen(x)
        print(f"  {name:16s} links {links:.0%}  in der Mitte {mitte:.1%}  "
              f"Häufchen bei {m_links:+.2f} und {m_rechts:+.2f}")

    # Proben
    links, mitte, m_links, m_rechts = kennzahlen(neu)
    assert 0.35 < links < 0.65, "beide Häufchen sollen etwa gleich groß sein"
    assert mitte < 0.10, "zwischen den Häufchen soll es fast leer sein"
    assert abs(m_links + MITTE) < 0.3 and abs(m_rechts - MITTE) < 0.3
    assert kennzahlen(exakt)[1] < 0.05
    assert fehler_exakt <= fehler_netz < fehler_exakt + 0.05, "Netz fast so gut wie die exakte Formel"
    print("\nAlle Proben bestanden.")
