# OCEAN – Persönlichkeitstyp-Vorhersage

Persönliches Projekt von **mperos1978**: [ocean-personality auf GitHub](https://github.com/mperos1978/ocean-personality).

## 1. Überblick

Dieses Projekt untersucht eine überwachte Klassifikationsaufgabe: Aus 19 Frageantworten sowie Alter, Geschlecht und Schreibhand wird ein Persönlichkeitstyp vorhergesagt. Eine deutschsprachige Streamlit-App stellt den Fragebogen bereit und zeigt die Vorhersage mit einem animierten Unterwasserdesign an.

### Daten und Zielklassen

Die Kursdaten basieren laut Projektbeschreibung auf einem auf Kaggle veröffentlichten Big-Five-Persönlichkeitstest. Eine Zeile beschreibt eine Person. Die Eingaben bestehen aus 19 Antwortwerten auf einer Skala von 1 bis 5 sowie `age`, `gender` und `hand`. Die Zielspalte heißt `target` und enthält vier Klassen:

| Deutsch | Klassenname im Modell |
|---|---|
| Moderat | Moderate |
| Resilient | Resilient |
| Überkontrolliert | Overcontroller |
| Unterkontrolliert | Undercontroller |

Die Reproduktion beginnt mit der bereitgestellten Ausgangsdatei `data/data.csv`. Das EDA-Notebook erzeugt daraus `data/data_clean_MP.csv` mit 19.587 Personen und 23 Spalten einschließlich `target`. Das Modelling verwendet 22 Eingabemerkmale, 15.669 Trainingsfälle und 3.918 zurückgehaltene Testfälle.

Die Zielklassen wurden aus den Frageantworten abgeleitet, die auch als Eingaben verwendet werden. Die Ergebnisse beschreiben deshalb die Nachbildung dieser vorhandenen Klassenzuordnung; sie sind keine unabhängige psychologische Validierung.

### Vorgehen

1. Daten untersuchen und bereinigen im EDA-Notebook.
2. Eingabemerkmale und Zielvariable trennen; 80 % Trainingsdaten und 20 % Testdaten bilden.
3. Fehlende Werte und die erforderliche Kodierung beziehungsweise Skalierung innerhalb der Modell-Pipelines behandeln.
4. Logistic Regression, KNN, Random Forest und HistGradientBoosting mit fünfteiliger stratifizierter Cross-Validation vergleichen.
5. Hyperparameter mit Grid Search, Randomized Search und Hyperopt optimieren.
6. Die ausgewählte Pipeline auf den Trainingsdaten trainieren, einmal auf dem zurückgehaltenen Testsatz bewerten und inklusive Vorverarbeitung speichern.
7. Die gespeicherte Pipeline in der Streamlit-App für Vorhersagen verwenden.

Die Modellwahl erfolgt anhand von **F1-Macro**. Der F1-Score verbindet zwei Fragen: Wie viele tatsächliche Fälle einer Klasse werden erkannt, und wie viele Vorhersagen dieser Klasse sind richtig? Für F1-Macro werden die F1-Scores aller vier Klassen gleich gewichtet und gemittelt. Der Wert liegt zwischen 0 und 1; höher ist besser. Als Prozentwert dargestellt bedeutet er jedoch nicht den Anteil aller richtigen Vorhersagen – diesen beschreibt die Accuracy.

Zufallsgesteuerte Schritte verwenden den Wert 42. Zusatzanalysen untersuchen den Vergleich ohne jeweils `age`, `gender` oder `hand`.

### Ergebnisse und Modellentscheidung

Die folgenden Werte sind die dokumentierten Ergebnisse aus dem Modelling-Notebook.

| Modell | Bestes Verfahren im Hauptvergleich | CV F1-Macro |
|---|---|---:|
| HistGradientBoosting | Hyperopt | **0,8028** |
| Logistic Regression | Hyperopt | 0,7657 |
| Random Forest | Ohne Tuning / Grid Search (gleicher Score) | 0,7380 |
| KNN | Hyperopt | 0,7378 |

Die gewählte Pipeline ist **HistGradientBoosting mit Hyperopt**. Ihr CV-F1-Macro beträgt **0,8028**, die Streuung über die fünf Folds **0,0057**. Gegenüber HistGradientBoosting ohne Tuning (0,7992) verbessert sich der CV-F1-Macro um rund 0,0036. Der Vorsprung gegenüber Randomized Search beträgt rund 0,0020; er ist klein im Verhältnis zur CV-Streuung.

Die ausgewählten Modellparameter sind:

| Hyperparameter | Wert |
|---|---:|
| `learning_rate` | 0.07814458302117205 |
| `max_leaf_nodes` | 31 |
| `l2_regularization` | 1.2994942824267277 |
| `max_iter` | 200 |
| `random_state` | 42 |

Auf dem zurückgehaltenen Testsatz erreichte die Pipeline **F1-Macro = 0,8030** und **Accuracy = 0,8612 (86,12 %)**. Das entspricht 3.374 richtigen Vorhersagen bei 3.918 Testfällen. Die Confusion Matrix zeigt insbesondere eine schwächere Erkennung von Undercontroller: 200 von 452 Fällen werden richtig erkannt (Recall 44,2 %); 231 werden als Moderate vorhergesagt. Die ausführliche Modellentscheidung und Klassenauswertung stehen im Modelling-Notebook.

CatBoost wurde ergänzend in einem separat aufbewahrten Notebook untersucht. Diese Zusatzuntersuchung ist nicht Bestandteil des hier reproduzierbaren Hauptvergleichs. Die dokumentierte Modellwahl beruht auf den vier Modellen im enthaltenen Modelling-Notebook. CatBoost ist für diesen Hauptworkflow und die App nicht erforderlich, bleibt jedoch als Bibliothek der bisherigen Projektumgebung in `MP_requirements.txt` enthalten.

### App verwenden

Alter eingeben, Geschlecht und Schreibhand auswählen und die 19 Aussagen bewerten. Anschließend die Vorhersage auslösen. Die App zeigt den deutschen Persönlichkeitstyp mit englischer Bezeichnung sowie weitere abrufbare Erläuterungen. Sie lädt ausschließlich die gespeicherte Pipeline und trainiert keine Modelle. Das Unterwasserdesign enthält bewegte Fische, Einsiedlerkrebse, Taucher, eine langsam diagonal treibende Mine und ein sinkendes Schiff.

## 2. Einrichtung

Die folgenden Schritte beziehen sich auf Windows 11 mit VS Code und Python 3.14. Alle Terminalbefehle werden im Hauptordner dieses persönlichen Repositorys ausgeführt: **`ocean-personality`**. App und Notebooks liegen direkt dort; ein zusätzlicher Unterordner `MP` ist nicht erforderlich. Für den lokalen Betrieb ist kein Hosting erforderlich.

Das Repository über GitHub herunterladen oder mit Git klonen. Zum Klonen im Terminal des gewünschten übergeordneten Ordners ausführen:

```powershell
git clone https://github.com/mperos1978/ocean-personality.git
cd ocean-personality
```

Beim ZIP-Download das Archiv zunächst entpacken. In VS Code über **Datei → Ordner öffnen** den entpackten Projektordner öffnen, in dem `MP_app.py` liegt. Danach **Terminal → Neues Terminal** wählen. Der Ordnername kann beim ZIP-Download `ocean-personality-main` lauten; die relativen Dateipfade funktionieren dort ebenso.

### 1. Daten beschaffen

Die Ausgangsdaten werden über den in der Projektbeschreibung genannten [Google-Drive-Ordner](https://drive.google.com/drive/folders/1KhwTPAG07EdaENW_XX9nVvKhC-DP1Ags?usp=sharing) bereitgestellt.

Den Unterordner `data` im Projektordner erstellen:

```powershell
New-Item -ItemType Directory -Force data
```

Die Ausgangs-CSV aus dem Drive-Ordner darin als **`data.csv`** ablegen. Genau diesen relativen Pfad liest das EDA-Notebook: `data/data.csv`.

Die Daten und die trainierte Joblib-Datei bleiben lokal und werden gemäß Projektbeschreibung **nicht in GitHub committed**.

### 2. Python-Umgebung erstellen

Die vorhandene App-Umgebung wurde mit Python 3.14 verwendet. Python und VS Code mit den Erweiterungen Python und Jupyter installieren. Den Projektordner über **Datei → Ordner öffnen** öffnen und anschließend **Terminal → Neues Terminal** wählen.

```powershell
py -3.14 -m venv .venv
```

Die folgenden Befehle verwenden die virtuelle Umgebung direkt; eine PowerShell-Aktivierung ist dafür nicht nötig.

### 3. Abhängigkeiten installieren

Die versionsgebundenen Abhängigkeiten für EDA, Modelling, Notebook-Ausführung und App installieren:

```powershell
.\.venv\Scripts\python.exe -m pip install -r MP_requirements.txt
```

`MP_requirements.txt` enthält die tatsächlich verwendeten Versionen von NumPy, pandas, Matplotlib, seaborn, scikit-learn, SciPy, Hyperopt, joblib, Streamlit, IPython, ipywidgets, ipykernel und CatBoost. Weitere benötigte Unterabhängigkeiten installiert `pip` automatisch. CatBoost wird für die separate Zusatzanalyse verwendet und ist für die Vorhersage in der App nicht erforderlich.

### 4. Notebooks in der richtigen Reihenfolge ausführen

In VS Code den Repository-Hauptordner öffnen und für beide Notebooks oben rechts den Python-Kernel aus `.venv` wählen. Der Arbeitsordner muss der Repository-Hauptordner sein, damit die relativen CSV-Pfade stimmen. Jedes Notebook mit frisch gestartetem Kernel von oben nach unten ausführen.

**Zuerst `MP EDA Personality.ipynb`:** Alle Zellen von oben nach unten ausführen. Die Speicherzelle erzeugt die bereinigte Datei unter `data/data_clean_MP.csv`.

**Danach `MP_modelling.ipynb`:** Die Ladezelle liest `data/data_clean_MP.csv`. Ein manueller Kopierschritt ist nicht erforderlich.

Den Hauptworkflow bis einschließlich **„Gewinner-Pipeline speichern“** von oben nach unten ausführen. Der anschließende Abschnitt **„Abschlussdokumentation des Hauptworkflows“** zeigt die Ergebnisse erneut an. Die spätere **„Zusatzanalyse: Modellvergleich ohne eine Spalte“** ist optional und für die App nicht erforderlich. Das Modelltraining kann mehrere Minuten dauern. Der Speicherschritt erstellt den Modellordner automatisch und erzeugt:

```text
models/best_pipeline.joblib
```

Die Datei enthält die trainierte Vorverarbeitung und das Modell zusammen. Liegt sie bereits aus einem abgeschlossenen Training vor, muss für einen reinen App-Start nicht erneut trainiert werden.

### 5. App lokal starten

```powershell
.\.venv\Scripts\python.exe -m streamlit run MP_app.py
```

Die App öffnet sich normalerweise automatisch. Andernfalls die im Terminal angezeigte lokale Adresse im Browser öffnen, üblicherweise `http://localhost:8501`.

Das Terminal während der Nutzung geöffnet lassen. Mit **Strg + C** wird die App beendet.

### Lokale Ordnerstruktur

```text
ocean-personality/             # Hauptordner des persönlichen Repositorys
├── .gitignore
├── MP_readme.md
├── MP_requirements.txt
├── MP_app.py
├── MP EDA Personality.ipynb
├── MP_modelling.ipynb
├── data/                       # lokal beschaffen beziehungsweise erzeugen
│   ├── data.csv
│   └── data_clean_MP.csv
├── models/                     # lokal durch Training erzeugen
│   └── best_pipeline.joblib
└── .venv/                      # lokal erzeugte Python-Umgebung
```

Die App findet die Pipeline relativ zu ihrem eigenen Speicherort: im Unterordner `models` neben `MP_app.py`. Der gesamte Ordner kann daher verschoben werden, solange diese Struktur erhalten bleibt. Die Notebooks erwarten den Projektordner als Arbeitsverzeichnis. Die bestehende virtuelle Umgebung kann sich auch außerhalb des Projektordners befinden; für eine neue Einrichtung zeigen die Befehle oben eine eigene `.venv` im Projektordner.

### Häufige Startprobleme

| Meldung oder Problem | Lösung |
|---|---|
| Pipeline fehlt | Modelling ausführen oder prüfen, ob `models/best_pipeline.joblib` vorhanden ist. |
| CSV nicht gefunden | `data/data.csv` für EDA beziehungsweise `data/data_clean_MP.csv` für Modelling prüfen. |
| Bibliothek fehlt | Abhängigkeiten in der ausgewählten Python-Umgebung installieren. |
| Andere Python-Umgebung verwendet | Die Befehle mit `.\.venv\Scripts\python.exe` ausführen und denselben Notebook-Kernel wählen. |
| Browser öffnet sich nicht | Die lokale Adresse aus dem Streamlit-Terminal öffnen. |

## 3. Persönliches Repository und Prüfung

Dieses Repository ist die persönliche Präsentation des im Rahmen eines Gruppenprojekts erarbeiteten Beitrags. Es wird unabhängig vom gemeinsamen Repository gepflegt. Die Projektdateien liegen direkt auf der Branch **`main`** im Repository-Hauptordner. Die Dateinamen mit `MP` bleiben erhalten.

In GitHub gehören die Dokumentation, App, Requirements, Notebooks und `.gitignore`. Datensätze und Joblib-Dateien sowie die virtuelle Umgebung und generierte Zwischendateien werden nicht hochgeladen. Die im Strukturbeispiel gezeigten Daten- und Modelldateien entstehen lokal durch die beschriebenen Schritte. Die `.gitignore` schließt unter anderem `data/`, `models/` und `.venv/` aus.

Der lokale Ablauf wurde in einer neu eingerichteten Python-Umgebung geprüft: Requirements installieren, EDA ausführen, Modelling ausführen, Pipeline speichern und App starten. Auch die Erstellung und Aktualisierung der Vorhersage wurden getestet.

Die App berechnet das Ergebnis nach dem Klick auf **„Persönlichkeitstyp vorhersagen“**. Werden Antworten geändert, muss der Button erneut angeklickt werden. Ein berechnetes Ergebnis bleibt in derselben Browser-Sitzung für die zusätzlichen Typ-Erklärungen erhalten. Eine neue Sitzung beginnt ohne Ergebnis.

Eine Online-Veröffentlichung ist ein optionaler späterer Schritt; für die lokale Nutzung ist sie nicht erforderlich.
