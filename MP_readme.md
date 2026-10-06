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

Für die Reproduktion wird die bereinigte Datei `data/data_clean_MP.csv` mit 23 Spalten einschließlich `target` verwendet.

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
| HistGradientBoosting | Hyperopt | **0,8014 (80,14 %)** |
| Logistic Regression | Hyperopt | 0,7673 (76,73 %) |
| Random Forest | Grid Search | 0,7443 (74,43 %) |
| KNN | Grid Search | 0,7371 (73,71 %) |

Die gewählte Pipeline ist **HistGradientBoosting mit Hyperopt**. Im dokumentierten Lauf erreichte sie auf dem zurückgehaltenen Testsatz einen **F1-Macro von 0,8094 (80,94 %)** und eine **Accuracy von 0,8701 (87,01 %)**. Die Accuracy bedeutet, dass 87,01 % aller Vorhersagen richtig waren. Die dazugehörige Confusion Matrix zeigt insbesondere eine schwächere Erkennung von Undercontroller, die häufig als Moderate vorhergesagt werden. Die ausführliche Modellentscheidung und Klassenauswertung stehen im Modelling-Notebook.

In einer separat gespeicherten Zusatzuntersuchung wurde außerdem CatBoost mit Randomized Search, Grid Search und Hyperopt untersucht. CatBoost bestimmte seine Lernrate automatisch; optimiert wurden Baumzahl und Baumtiefe. Die beste Kombination aus 600 Bäumen und Tiefe 4 erreichte einen CV-F1-Macro von 0,8100 (81,00 %). Wegen des kleinen Leistungszuwachses bei deutlich höherem Rechenaufwand wurde für dieses Projekt die bestehende HistGradientBoosting-Pipeline beibehalten. Das CatBoost-Notebook wird separat aufbewahrt und gehört nicht zu diesem Abgabeordner. CatBoost ist als verwendete Bibliothek dennoch in `MP_requirements.txt` aufgeführt.

### App verwenden

Alter eingeben, Geschlecht und Schreibhand auswählen und die 19 Aussagen bewerten. Anschließend die Vorhersage auslösen. Die App zeigt den deutschen Persönlichkeitstyp mit englischer Bezeichnung sowie weitere abrufbare Erläuterungen. Sie lädt ausschließlich die gespeicherte Pipeline und trainiert keine Modelle.

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

Den Unterordner `data` im Projektordner erstellen. Die Ausgangs-CSV darin als **`data.csv`** ablegen. Genau diesen relativen Pfad liest das EDA-Notebook: `data/data.csv`.

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

In VS Code für beide Notebooks oben rechts den Python-Kernel aus `.venv` wählen.

**Zuerst `MP EDA Personality.ipynb`:** Alle Zellen von oben nach unten ausführen. Die Speicherzelle erzeugt die bereinigte Datei unter `data/data_clean_MP.csv`.

**Danach `MP_modelling.ipynb`:** Die Ladezelle liest `data/data_clean_MP.csv`. Ein manueller Kopierschritt ist nicht erforderlich.

Den Hauptworkflow bis einschließlich **Schritt 15** von oben nach unten ausführen. Die optionale Zusatzanalyse in Schritt 16 ist für die App nicht erforderlich. Das Modelltraining kann mehrere Minuten dauern. Der Speicherschritt erzeugt:

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
