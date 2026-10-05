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

| Ordner | Kapitel | Programm |
|---|---|---|
| `kap05/` | 5 Das Perzeptron lernt | `perzeptron.py` - Perzeptron lernt UND und ODER, jeder Lernschritt wird gedruckt |

## Starten

```bash
cd kap05
python perzeptron.py
```

## Lizenz

Siehe `LICENSE`.
