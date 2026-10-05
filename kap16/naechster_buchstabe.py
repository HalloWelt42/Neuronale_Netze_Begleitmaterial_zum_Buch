"""Ein rekurrentes Netz lernt, den nächsten Buchstaben eines kleinen Reims vorherzusagen.

Begleitprogramm zu Kapitel 16 "Folgen und Gedächtnis: rekurrente Netze und LSTM".
Läuft mit Python 3.12 oder neuer und braucht NumPy:

    python naechster_buchstabe.py

Was das Programm zeigt:
1. Das Echo eines einzelnen rückgekoppelten Neurons: Ein kurzer Anstoß verklingt Schritt für
   Schritt, und auch das Fehlersignal rückwärts wird bei jedem Schritt kleiner (Beispiel im Kapitel).
2. Eine Zähltabelle, die nur den letzten Buchstaben kennt (kein Gedächtnis), als Vergleich.
3. Ein rekurrentes Netz mit innerem Zustand lernt den Reim mit Lernen rückwärts durch die Zeit.
4. Das gelernte Netz schreibt den Reim aus einem kurzen Anfang selbst weiter.
5. Wie groß das Fehlersignal noch ist, wenn es 1, 2, 5, 10 oder 20 Schritte zurückgereist ist,
   vor und nach dem Lernen.
"""

import numpy as np

# Der Reim ist für dieses Buch geschrieben. Wichtig ist die Wiederholung: Nach "das " und nach
# "ist " hängt der nächste Buchstabe davon ab, was viele Zeichen vorher stand.
REIM = ("ich hab ein boot, das boot ist rot. "
        "ich hab ein schaf, das schaf ist brav. "
        "ich hab ein haus, im haus wohnt eine maus. ")

ZEICHEN = sorted(set(REIM))
NUMMER = {z: i for i, z in enumerate(ZEICHEN)}
K = len(ZEICHEN)          # Anzahl verschiedener Zeichen
H = 32                    # Größe des inneren Zustands (Anzahl der Gedächtnis-Neuronen)
LERNRATE = 0.1
RUNDEN = 600


# --- Teil 1: das Echo eines einzelnen Neurons -----------------------------------------
def echo(u: float = 0.5, schritte: int = 6) -> list[float]:
    """h_t = tanh(Eingabe_t + u * h_(t-1)); nur im ersten Schritt kommt eine 1 herein."""
    print(f"=== 1. Echo eines Neurons mit Rückkopplungsgewicht u = {u} ===")
    h, verlauf, faktor_gesamt = 0.0, [], 1.0
    for t in range(1, schritte + 1):
        eingabe = 1.0 if t == 1 else 0.0
        h = float(np.tanh(eingabe + u * h))
        verlauf.append(h)
        steigung = 1 - h * h                 # Steigung von tanh an dieser Stelle
        print(f"  Schritt {t}: Eingabe {eingabe:.0f}  Zustand h = {h:.3f}"
              f"   Steigung tanh = {steigung:.3f}")
    # Rückwärts: Bei jedem Schritt zurück wird das Fehlersignal mit u * Steigung malgenommen.
    print("  Rückwärts von Schritt 6 nach Schritt 1, Faktor je Schritt = u * Steigung:")
    for t in range(schritte, 1, -1):
        faktor = u * (1 - verlauf[t - 1] ** 2)
        faktor_gesamt *= faktor
        print(f"    {t} -> {t - 1}: Faktor {faktor:.3f}   Produkt bisher {faktor_gesamt:.6f}")
    assert verlauf[0] > 0.7 and verlauf[-1] < 0.05
    assert faktor_gesamt < 0.5 ** 5
    print("  Mit u = 0.5 kann der Faktor nie größer als 0.5 werden: 0.5^20 =", f"{0.5 ** 20:.7f}")
    print("  Zum Vergleich ein Faktor 1.5 je Schritt: 1.5^20 =", f"{1.5 ** 20:.0f}")
    for f in (0.95, 0.99):
        print(f"  Faktor {f} je Schritt: nach 20 Schritten {f ** 20:.3f}, nach 100 Schritten {f ** 100:.3f}")
    return verlauf


def lstm_von_hand() -> float:
    """Eine LSTM-Speicherzelle mit von Hand gesetzten Ventilen (0 = zu, 1 = offen)."""
    print("\n=== 1b. Eine LSTM-Zelle mit von Hand gestellten Ventilen ===")
    print("  (Ventil 1 = offen, 0 = zu. Ein offenes Vergessensventil lässt den alten Inhalt durch.)")
    #           vergessen, schreiben, lesen, neuer Inhalt
    plan = [(0.0, 1.0, 0.0, 0.8),     # Schritt 1: alles vergessen, 0.8 hineinschreiben
            (1.0, 0.0, 0.0, -0.5),    # Schritt 2: festhalten, Neues abweisen
            (1.0, 0.0, 0.0, 0.3),     # Schritt 3: festhalten, Neues abweisen
            (1.0, 0.0, 1.0, 0.1)]     # Schritt 4: festhalten und auslesen
    c = 0.0
    for t, (f, i, o, g) in enumerate(plan, 1):
        c = f * c + i * g
        h = o * float(np.tanh(c))
        print(f"  Schritt {t}: vergessen {f:.0f}  schreiben {i:.0f}  lesen {o:.0f}"
              f"  Angebot {g:+.1f}  ->  Speicher c = {c:.3f}  Ausgabe h = {h:.3f}")
    assert c == 0.8 and abs(h - 0.664) < 0.001
    return h


# --- Teil 2: Zähltabelle ohne Gedächtnis ---------------------------------------------
def softmax_beispiel() -> None:
    p = softmax(np.array([2.0, 1.0, 0.0]))
    print(f"\n  Softmax der Punktzahlen 2, 1, 0: {p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}")
    assert abs(p.sum() - 1) < 1e-12


def zaehltabelle() -> float:
    """Für jedes Zeichen merken, welches Zeichen am häufigsten darauf folgt."""
    print("\n=== 2. Vergleich ohne Gedächtnis: Was folgt am häufigsten auf ein Zeichen? ===")
    zaehler = np.zeros((K, K))
    for a, b in zip(REIM, REIM[1:] + REIM[0]):
        zaehler[NUMMER[a], NUMMER[b]] += 1
    tipp = zaehler.argmax(axis=1)
    treffer = sum(tipp[NUMMER[a]] == NUMMER[b] for a, b in zip(REIM, REIM[1:] + REIM[0]))
    quote = treffer / len(REIM)
    print(f"  Nach einem Leerzeichen tippt die Tabelle immer auf '{ZEICHEN[tipp[NUMMER[' ']]]}'.")
    print(f"  Treffer: {treffer} von {len(REIM)} Zeichen = {100 * quote:.1f} %")
    return quote


# --- Teil 3: das rekurrente Netz ------------------------------------------------------
def softmax(s: np.ndarray) -> np.ndarray:
    """Aus beliebigen Punktzahlen Wahrscheinlichkeiten machen, die zusammen 1 ergeben."""
    e = np.exp(s - s.max())
    return e / e.sum()


def schritt(netz, x, h_alt):
    """Ein Zeitschritt: neuer Zustand aus Eingabe und altem Zustand, dann Tipp fürs nächste Zeichen."""
    W_ein, W_rueck, W_aus, b_h, b_y = netz
    h = np.tanh(W_ein @ x + W_rueck @ h_alt + b_h)  # neuer Zustand
    p = softmax(W_aus @ h + b_y)                    # Tipp fürs nächste Zeichen
    return h, p


def lernschritt(netz, text, h_start):
    """Vorwärts durch die ganze Folge, dann rückwärts durch die Zeit; liefert Fehler und Steigungen."""
    W_ein, W_rueck, W_aus, b_h, b_y = netz
    xs, hs, ps = [], {-1: h_start}, []
    fehler = 0.0
    for t, (a, b) in enumerate(zip(text, text[1:])):
        x = np.zeros(K); x[NUMMER[a]] = 1
        hs[t], p = schritt(netz, x, hs[t - 1])
        xs.append(x); ps.append(p)
        fehler -= np.log(p[NUMMER[b]])          # Überraschung beim richtigen Zeichen
    d = [np.zeros_like(g) for g in netz]
    # Rückwärts durch die Zeit
    dh_spaeter = np.zeros(H)          # Signal, das aus der Zukunft kommt
    for t in reversed(range(len(xs))):
        dy = ps[t].copy()
        dy[NUMMER[text[t + 1]]] -= 1  # Tipp minus Wahrheit
        d[2] += np.outer(dy, hs[t]); d[4] += dy
        dh = W_aus.T @ dy + dh_spaeter     # Schuld von jetzt + von später
        dz = (1 - hs[t] ** 2) * dh         # durch die Steigung von tanh
        d[0] += np.outer(dz, xs[t])        # Korrekturen aufsummieren
        d[1] += np.outer(dz, hs[t - 1])
        d[3] += dz
        dh_spaeter = W_rueck.T @ dz        # einen Schritt weiter zurück
    for g in d:
        np.clip(g, -5, 5, out=g)               # Ausreißer kappen (gegen explodierende Steigungen)
    return fehler / len(xs), d


def neues_netz(zufall):
    return [zufall.normal(0, 0.1, (H, K)), zufall.normal(0, 0.1, (H, H)),
            zufall.normal(0, 0.1, (K, H)), np.zeros(H), np.zeros(K)]


def trainieren(seed: int = 7):
    print(f"\n=== 3. Rekurrentes Netz: {K} Zeichen, {H} Gedächtnis-Neuronen ===")
    zufall = np.random.default_rng(seed)
    netz = neues_netz(zufall)
    start = [g.copy() for g in netz]
    speicher = [np.zeros_like(g) for g in netz]  # Adagrad: Summe der quadrierten Steigungen
    text = REIM + REIM[0]                        # am Ende wieder vorn anfangen
    verlauf = []
    for runde in range(1, RUNDEN + 1):
        fehler, d = lernschritt(netz, text, np.zeros(H))
        for g, dg, s in zip(netz, d, speicher):
            s += dg * dg
            g -= LERNRATE * dg / np.sqrt(s + 1e-8)
        verlauf.append(fehler)
        if runde in (1, 10, 50, 100, 200, 400, RUNDEN):
            print(f"  Runde {runde:4d}: mittlere Überraschung {fehler:.3f}"
                  f"   (Wahrscheinlichkeit fürs richtige Zeichen etwa {np.exp(-fehler):.2f})")
    assert verlauf[-1] < verlauf[0] / 10
    return netz, start, verlauf


def trefferquote(netz) -> float:
    h, treffer = np.zeros(H), 0
    for a, b in zip(REIM, REIM[1:] + REIM[0]):
        x = np.zeros(K); x[NUMMER[a]] = 1
        h, p = schritt(netz, x, h)
        treffer += ZEICHEN[int(p.argmax())] == b
    quote = treffer / len(REIM)
    print(f"  Treffer des Netzes: {treffer} von {len(REIM)} Zeichen = {100 * quote:.1f} %")
    return quote


def sicher_nach(netz, anfang: str) -> None:
    """Wie sicher ist das Netz beim Zeichen nach einem Textstück aus dem Reim?"""
    h = np.zeros(H)
    for a in anfang:
        x = np.zeros(K); x[NUMMER[a]] = 1
        h, p = schritt(netz, x, h)
    beste = np.argsort(p)[::-1][:3]
    tipps = ", ".join(f"'{ZEICHEN[i]}' {p[i]:.2f}" for i in beste)
    print(f"  nach '...{anfang[-14:]}': {tipps}")


# --- Teil 4: Weiterschreiben --------------------------------------------------------
def weiterschreiben(netz, anfang: str = "ich hab ein s", laenge: int = 60) -> str:
    h = np.zeros(H)
    for a in anfang:
        x = np.zeros(K); x[NUMMER[a]] = 1
        h, p = schritt(netz, x, h)
    text = anfang
    for _ in range(laenge):
        z = ZEICHEN[int(p.argmax())]
        text += z
        x = np.zeros(K); x[NUMMER[z]] = 1
        h, p = schritt(netz, x, h)
    print(f"  Anfang '{anfang}' -> '{text}'")
    return text


# --- Teil 5: Wie weit reicht der Einfluss zurück? ------------------------------------
def rueckwaerts_signal(netz, abstaende=(1, 2, 5, 10, 20)) -> dict[int, float]:
    """Fehlersignal eines Zeitschritts rückwärts schicken: Wie groß kommt es k Schritte früher an?"""
    W_ein, W_rueck, W_aus, b_h, b_y = netz
    h, zustaende, ps = np.zeros(H), [], []
    for a in REIM * 2:
        x = np.zeros(K); x[NUMMER[a]] = 1
        h, p = schritt(netz, x, h)
        zustaende.append(h); ps.append(p)
    groesse = {k: [] for k in abstaende}
    folge = (REIM * 2)[1:] + REIM[0]
    for t in range(len(REIM), 2 * len(REIM)):
        dy = ps[t].copy(); dy[NUMMER[folge[t]]] -= 1
        dh = W_aus.T @ dy                       # Signal am Zustand zum Zeitpunkt t
        start = np.linalg.norm(dh)
        for k in range(1, max(abstaende) + 1):  # Schritt für Schritt in die Vergangenheit
            dh = W_rueck.T @ ((1 - zustaende[t - k + 1] ** 2) * dh)
            if k in groesse:
                groesse[k].append(np.linalg.norm(dh) / start)
    ergebnis = {k: float(np.mean(v)) for k, v in groesse.items()}
    for k, v in ergebnis.items():
        print(f"  {k:2d} Schritte zurück: Anteil des Fehlersignals {v:.7f}")
    return ergebnis


if __name__ == "__main__":
    echo()
    lstm_von_hand()
    softmax_beispiel()
    quote_tabelle = zaehltabelle()
    netz, startnetz, _ = trainieren()
    quote_netz = trefferquote(netz)
    for stueck in ("ich hab ein boot, das ", "ich hab ein schaf, das ",
                   "ich hab ein boot, das boot ist ", "ich hab ein schaf, das schaf ist "):
        sicher_nach(netz, stueck)
    assert quote_netz > quote_tabelle + 0.2
    print("\n=== 4. Das Netz schreibt weiter (immer das wahrscheinlichste Zeichen) ===")
    text = weiterschreiben(netz)
    assert text.startswith("ich hab ein schaf, das schaf ist brav.")
    weiterschreiben(netz, "ich hab ein auto", 30)    # kommt im Reim nicht vor
    print("\n=== 5. Wie weit reicht das Fehlersignal zurück? ===")
    print("  Vor dem Lernen (kleine Zufallsgewichte):")
    vorher = rueckwaerts_signal(startnetz)
    assert vorher[20] < vorher[1] / 1000
    print("  Nach dem Lernen:")
    nachher = rueckwaerts_signal(netz)
    assert nachher[20] < nachher[1]
    print("\nAlle Proben bestanden.")
