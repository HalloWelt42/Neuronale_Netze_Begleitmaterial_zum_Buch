# Neuronale Netze - Begleitmaterial zum Buch

Programme zum Buch "Neuronale Netze - Von der Nervenzelle zur modernen KI" von HalloWelt42.

Jedes Kapitel mit Rechenkern hat hier einen eigenen Ordner. Im Buch steht jeweils nur der Kern eines
Programms; hier liegt die vollständige, lauffähige Fassung. Jedes Programm prüft sich selbst
(`assert`) und druckt nachvollziehbare Zwischenschritte.

## Voraussetzungen

- Python 3.12 oder neuer
- Teil I bis IV: nur die Standardbibliothek
- ab Teil V: zusätzlich NumPy (`pip install numpy`), im jeweiligen Programm angekündigt

## Inhalt

<!-- inhalt:anfang -->
| Ordner | Kapitel | Programm |
|---|---|---|
| `kap01/` | 1 Was heißt eigentlich Lernen? | `schaetzer.py` - Ein lernender Schätzer findet aus Beispielen die Regel "Zoll mal Faktor = Zentimeter" |
| `kap02/` | 2 Das Gehirn als Vorbild | `nervenzelle.py` - Eine Nervenzelle als einfaches Schwellenmodell: integrieren und feuern |
| `kap03/` | 3 Die ersten Modelle: 1943 bis 1958 | `logikgatter.py` - Logikgatter aus McCulloch-Pitts-Zellen und ein kleines Beispiel zu Hebbs Lernregel |
| `kap04/` | 4 Das Modellneuron | `modellneuron.py` - Ein Modellneuron von Hand rechnen und seine Aktivierungsfunktionen ansehen |
| `kap05/` | 5 Das Perzeptron lernt | `perzeptron.py` - Ein Perzeptron lernt die UND- und die ODER-Verknüpfung |
| `kap06/` | 6 Adaline und die Grenzen der Geraden | `endlosschleife.py` - Ein Perzeptron versucht XOR zu lernen und merkt, dass es sich im Kreis dreht |
| `kap07/` | 7 Mehrere Schichten | `xor_netz.py` - Netze mit einer verdeckten Schicht und fest eingestellten Gewichten |
| `kap08/` | 8 Die Fehlerlandschaft | `gradientenabstieg.py` - Gradientenabstieg auf einer Fehlerparabel: Lernrate zu klein, passend, zu groß |
| `kap09/` | 9 Backpropagation | `xor_backpropagation.py` - Ein kleines Netz (2-2-1) lernt XOR mit Backpropagation |
| `kap10/` | 10 Wenn das Lernen hakt | `schlucht_momentum.py` - Einfacher Gradientenabstieg gegen Abstieg mit Schwung (Momentum) in einer langen Schlucht |
| `kap10/` | 10 Wenn das Lernen hakt | `ueberanpassung.py` - Überanpassung: Trainingsfehler und Testfehler bei einer Polynomanpassung |
| `kap11/` | 11 Hopfield-Netze: Gedächtnis aus Rückkopplung | `hopfield_buchstaben.py` - Ein Hopfield-Netz merkt sich drei Buchstaben und erinnert sich an einen gestörten |
| `kap12/` | 12 Kohonen-Karten: Ordnung ohne Lehrer | `farbkarte.py` - Eine Kohonen-Karte ordnet Farben auf einem 10x10-Gitter |
| `kap12/` | 12 Kohonen-Karten: Ordnung ohne Lehrer | `gitter_flaeche.py` - Ein Gitter aus 8x8 Neuronen legt sich über ein Quadrat |
| `kap13/` | 13 Weitere Wege: ART, Fuzzy und Radialbasis | `fuzzy_heizung.py` - Ein Fuzzy-Regler stellt das Ventil einer Heizung nach der Raumtemperatur ein |
<!-- inhalt:ende -->

## Starten

```bash
cd kap05
python perzeptron.py
```

## Lizenz

Siehe `LICENSE`.
