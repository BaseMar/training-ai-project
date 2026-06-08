# Training AI Project

System analizy danych treningowych i generowania demonstracyjnych rekomendacji planu siłowego z wykorzystaniem syntetycznych danych, modeli regresyjnych i dashboardu Streamlit.

## Opis projektu

`Training AI Project` to projekt portfolio z obszaru Data Science / Python, który pokazuje kompletny proces pracy z danymi treningowymi: od wygenerowania syntetycznego zbioru danych, przez EDA i feature engineering, po model predykcyjny oraz hybrydowy system rekomendacyjny.

Projekt rozwiązuje problem analizy historii treningu siłowego i wspomagania decyzji o kolejnych parametrach treningu: doborze splitu, ćwiczeń, liczby serii, liczby powtórzeń, docelowego RIR, poziomu zmęczenia oraz sugerowanego ciężaru. Rekomendacje mają charakter demonstracyjny i nie zastępują trenera, fizjoterapeuty ani konsultacji medycznej.

Z perspektywy portfolio projekt pokazuje:

- generowanie i walidację syntetycznego datasetu,
- eksploracyjną analizę danych i wizualizacje,
- przygotowanie cech historycznych dla problemu regresji,
- porównanie modeli ML i zapis najlepszego pipeline'u,
- implementację regułowego i modelowego systemu rekomendacyjnego,
- aplikację Streamlit prezentującą dane, model i scenariusze rekomendacji.

## Główne funkcjonalności

- **Generator syntetycznych danych treningowych**  
  Moduł `generator/` symuluje użytkowników, sesje treningowe i serie ćwiczeń z uwzględnieniem poziomu zaawansowania, splitu, fazy treningowej, zmęczenia, RIR, progresji oraz losowych wahań dyspozycji.

- **Kanoniczny dataset treningowy**  
  Plik `data/FINAL_ENGINE_V4.csv` zawiera dane na poziomie pojedynczej serii treningowej. W aktualnej wersji ma 1 215 602 rekordy, 100 syntetycznych użytkowników, 61 951 sesji i 15 ćwiczeń.

- **Eksploracyjna analiza danych**  
  Skrypt `scripts/01_eda.py` analizuje strukturę danych, braki, duplikaty, rozkłady zmiennych, trendy objętości, relacje między poziomem, splitem, fazą i płcią oraz tworzy cechy `volume` i `e1rm_epley`.

- **Modelowanie predykcyjne**  
  Skrypt `scripts/02_modeling_and_recommendation.py` przygotowuje cechy historyczne, wykonuje czasowy podział train/test, porównuje modele regresyjne i zapisuje najlepszy model jako `models/best_weight_prediction_model.joblib`.

- **Hybrydowy rekomender planów treningowych**  
  Moduł `src/recommendation_engine.py` łączy wynik modelu ML, dane podobnych użytkowników, historię konkretnego użytkownika, kalibrację siłową i reguły bezpieczeństwa.

- **Demo end-to-end**  
  Skrypt `scripts/03_system_demo.py` ładuje dataset i model, wykonuje sanity check modelu, generuje kilka scenariuszy planów tygodniowych i zapisuje wyniki do `outputs/stage3_outputs/`.

- **Dashboard Streamlit**  
  Aplikacja `app/streamlit_app.py` prezentuje dataset, EDA, wyniki modelu, scenariusze rekomendacji, reguły bezpieczeństwa oraz formularz `Live Generator` do generowania planu na żywo, jeśli lokalnie dostępny jest model `.joblib`.

## Technologie

Technologie i biblioteki widoczne w repozytorium:

- Python
- pandas
- NumPy
- scikit-learn
- joblib
- Streamlit
- matplotlib
- seaborn
- CSV
- Markdown
- PDF / DOCX jako formaty raportu

## Struktura projektu

```text
TrainingAIProject/
|-- app/
|   |-- demo_assets/
|   |   |-- scenario_comparison.csv
|   |   `-- plan_*.csv
|   `-- streamlit_app.py
|-- data/
|   `-- FINAL_ENGINE_V4.csv
|-- generator/
|   |-- README_generator.md
|   |-- config.py
|   |-- generator.py
|   |-- main.py
|   |-- models.py
|   `-- session.py
|-- models/
|   `-- best_weight_prediction_model.joblib       # generowany lokalnie, ignorowany przez Git
|-- outputs/
|   |-- eda_outputs/                              # generowane lokalnie
|   |-- stage2_outputs/                           # generowane lokalnie
|   `-- stage3_outputs/                           # generowane lokalnie
|-- presentation/
|-- report/
|   |-- final_report.md
|   |-- final_report.pdf
|   |-- final_report.docx
|   `-- images/
|-- scripts/
|   |-- 01_eda.py
|   |-- 02_modeling_and_recommendation.py
|   `-- 03_system_demo.py
|-- src/
|   `-- recommendation_engine.py
|-- .gitignore
|-- README.md
`-- requirements.txt
```

Najważniejsze elementy:

- `generator/` - logika tworzenia syntetycznych danych treningowych.
- `data/FINAL_ENGINE_V4.csv` - główne wejście dla analizy, modelowania i dashboardu.
- `scripts/` - uruchamialne etapy projektu: EDA, modelowanie, demo systemu.
- `src/recommendation_engine.py` - wielokrotnego użytku logika rekomendacyjna wykorzystywana przez dashboard.
- `app/` - aplikacja Streamlit oraz małe pliki CSV z gotowymi scenariuszami demo.
- `report/` - raport projektu i obrazy używane do prezentacji wyników.
- `outputs/` - lokalne wyniki skryptów; katalogi są ignorowane przez Git.
- `models/*.joblib` - lokalne artefakty modeli; ignorowane przez Git ze względu na rozmiar.

## Instalacja i uruchomienie

### 1. Klonowanie repozytorium

```bash
git clone <repository-url>
cd TrainingAIProject
```

### 2. Utworzenie środowiska wirtualnego

Projekt używa składni typów dostępnej w Pythonie 3.10+, więc taka wersja Pythona jest minimalnym sensownym wymaganiem. Repozytorium nie zawiera `pyproject.toml` ani dokładnej deklaracji wersji Pythona.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Uruchomienie generatora danych

```bash
python generator/main.py --users 100 --years 3 --output data/FINAL_ENGINE_V4.csv
```

Szybki test bez zapisu pliku:

```bash
python generator/main.py --users 5 --years 1 --no-save
```

### 4. Uruchomienie EDA

```bash
python scripts/01_eda.py
```

Wyniki robocze są zapisywane do:

```text
outputs/eda_outputs/
```

### 5. Trenowanie modelu i rekomender Stage 2

```bash
python scripts/02_modeling_and_recommendation.py
```

Skrypt zapisuje wyniki do `outputs/stage2_outputs/` oraz model do:

```text
models/best_weight_prediction_model.joblib
```

Plik modelu jest duży i znajduje się w `.gitignore`, dlatego po świeżym klonie może wymagać ponownego wygenerowania przez uruchomienie Stage 2.

### 6. Uruchomienie demo end-to-end

```bash
python scripts/03_system_demo.py
```

Ten etap wymaga istniejącego modelu `models/best_weight_prediction_model.joblib`. Jeżeli go brakuje, najpierw uruchom Stage 2.

### 7. Uruchomienie dashboardu Streamlit

```bash
streamlit run app/streamlit_app.py
```

Dashboard działa w dwóch trybach:

- tryb prezentacyjny na plikach `app/demo_assets/*.csv`,
- tryb `Live Generator`, który wymaga lokalnego modelu `models/best_weight_prediction_model.joblib`.

### 8. Raport i materiały

Raport projektu można otworzyć z katalogu:

```text
report/final_report.md
report/final_report.pdf
report/final_report.docx
```

Obrazy raportowe i screeny dashboardu znajdują się w:

```text
report/images/
```

## Dane

Główny dataset projektu:

```text
data/FINAL_ENGINE_V4.csv
```

Dane są syntetyczne i lokalne. Zostały wygenerowane przez moduł `generator/`, a nie pobrane z rzeczywistych dzienników treningowych.

Aktualny plik danych zawiera:

| Cecha | Wartość |
| --- | ---: |
| Liczba rekordów | 1 215 602 |
| Liczba kolumn | 13 |
| Liczba użytkowników | 100 |
| Liczba sesji | 61 951 |
| Liczba ćwiczeń | 15 |
| Zakres dat | 2022-01-01 - 2024-12-30 |

Kolumny datasetu:

| Kolumna | Znaczenie |
| --- | --- |
| `user_id` | Identyfikator syntetycznego użytkownika. |
| `session_id` | Identyfikator sesji treningowej. |
| `date` | Data treningu. |
| `exercise` | Nazwa ćwiczenia. |
| `set_number` | Numer serii w ramach ćwiczenia. |
| `reps` | Liczba powtórzeń. |
| `weight` | Ciężar roboczy w kilogramach. |
| `fatigue` | Poziom zmęczenia używany w symulacji. |
| `rir` | Reps In Reserve, czyli liczba powtórzeń w zapasie. |
| `level` | Poziom zaawansowania: `beginner`, `intermediate`, `advanced`. |
| `split` | Split treningowy: `fbw`, `ppl`, `upper_lower`. |
| `phase` | Faza treningowa: `hypertrophy`, `strength`, `deload`. |
| `sex` | Płeć syntetycznego użytkownika: `female`, `male`. |

W katalogu `app/demo_assets/` znajdują się mniejsze pliki CSV z gotowymi scenariuszami demonstracyjnymi:

- `scenario_comparison.csv`,
- `plan_beginner_female_hypertrophy.csv`,
- `plan_intermediate_male_strength.csv`,
- `plan_advanced_male_deload.csv`,
- `plan_older_beginner_hypertrophy.csv`,
- `plan_existing_user_with_history.csv`.

## Sposób działania

Ogólny przepływ projektu:

```text
generator/ -> data/FINAL_ENGINE_V4.csv -> EDA -> feature engineering
-> model regresyjny -> rekomender hybrydowy -> demo CSV / dashboard Streamlit
```

1. `generator/main.py` uruchamia symulację użytkowników i zapisuje dane do CSV.
2. `scripts/01_eda.py` sprawdza jakość danych, rozkłady, trendy oraz tworzy pierwsze cechy analityczne.
3. `scripts/02_modeling_and_recommendation.py` tworzy cechy historyczne, trenuje modele regresyjne i wybiera najlepszy model według MAE.
4. Model przewiduje ciężar pasujący do planowanych parametrów serii: ćwiczenia, liczby powtórzeń, RIR, zmęczenia, fazy i historii użytkownika.
5. Rekomender łączy predykcję modelu z regułami biznesowymi: wyborem splitu, doborem ćwiczeń, fallbackiem na podobnych użytkowników i limitami bezpieczeństwa.
6. `scripts/03_system_demo.py` generuje przykładowe plany tygodniowe.
7. `app/streamlit_app.py` prezentuje dataset, EDA, model, gotowe plany oraz formularz live generatora.

## Wyniki, dashboard i przykładowe użycie

Repozytorium zawiera raport i obrazy prezentujące wyniki analizy oraz dashboard:

![Dashboard overview](report/images/dashboard_overview.png)

Przykładowe artefakty wynikowe:

- raport końcowy w `report/final_report.md`, `report/final_report.pdf` i `report/final_report.docx`,
- wykresy EDA w `report/images/`, m.in. rozkłady poziomów, faz, ciężaru, RIR i trend dziennej objętości,
- obrazy porównania modeli w `report/images/model_mae_comparison.png` i `report/images/model_within_5kg_comparison.png`,
- screeny dashboardu w `report/images/dashboard_*.png`,
- scenariusze planów w `app/demo_assets/`.

W lokalnych wynikach Stage 2 porównano:

| Model | MAE | RMSE | R2 | Predykcje w granicy 5 kg |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 3.8604 | 7.1335 | 0.9625 | 77.287% |
| Ridge Regression | 3.8780 | 6.9160 | 0.9648 | 77.344% |
| HistGradientBoosting | 3.9486 | 7.7548 | 0.9557 | 77.449% |
| Naive previous weight | 4.6917 | 8.5286 | 0.9464 | 71.867% |

Według metryki MAE najlepszy był model `Random Forest`, który został zapisany jako `models/best_weight_prediction_model.joblib`.

## Najważniejsze decyzje projektowe

- **Dane syntetyczne zamiast rzeczywistych**  
  Projekt nie używa danych osobowych ani realnych dzienników treningowych. Generator pozwala kontrolować strukturę danych i testować pełny pipeline bez ryzyka prywatności.

- **Poziom pojedynczej serii jako jednostka obserwacji**  
  Każdy rekord opisuje jedną serię. Dzięki temu dane można agregować do poziomu ćwiczenia, sesji, tygodnia lub użytkownika.

- **Cechy historyczne bez przecieku danych**  
  Skrypty modelujące używają `shift(1)` oraz średnich kroczących z poprzednich obserwacji, aby model nie widział informacji z aktualnie przewidywanej serii.

- **Predykcja ciężaru jako problem regresyjny**  
  Model odpowiada na pytanie: jaki ciężar pasuje do planowanych parametrów serii i historii użytkownika.

- **Rekomendacja hybrydowa**  
  Finalny plan nie jest czystą predykcją ML. System łączy model, podobnych użytkowników, historię, kalibrację siłową oraz reguły bezpieczeństwa.

- **Reguły bezpieczeństwa jako ograniczenia rekomendacji**  
  Rekomender ogranicza progresję przy wysokim zmęczeniu, niskim RIR, fazie `deload` oraz dla starszych użytkowników.

- **Oddzielenie artefaktów roboczych od repozytorium**  
  `outputs/` i `models/*.joblib` są ignorowane przez Git. Repozytorium zawiera kod, dane kanoniczne, raport i lekkie artefakty demo.

## Możliwe usprawnienia

- Dodanie deklaracji minimalnej wersji Pythona i wersji bibliotek w `requirements.txt`.
- Udostępnienie modelu `.joblib` przez GitHub Release albo Git LFS.
- Dodanie testów jednostkowych dla generatora i rekomendera.
- Rozbudowanie walidacji danych wejściowych w dashboardzie.
- Dodanie osobnej tabeli profili użytkowników, np. z wiekiem, celem i ograniczeniami treningowymi.
- Walidacja rekomendacji na danych rzeczywistych lub przez eksperta treningowego.
- Dodanie automatycznego procesu odświeżania `app/demo_assets/` po uruchomieniu Stage 3.

## Status projektu

Projekt portfolio / demonstrator Data Science w trakcie rozwoju. Główne elementy pipeline'u są zaimplementowane: generator danych, EDA, modelowanie, rekomender, demo end-to-end, dashboard Streamlit oraz raport końcowy.

Ograniczenia aktualnej wersji:

- dane są syntetyczne,
- model i rekomender są prototypem decyzyjno-analitycznym,
- dataset nie zawiera kolumny `age`; wiek jest używany tylko jako zewnętrzny parametr formularza i reguł bezpieczeństwa,
- system nie jest narzędziem medycznym ani produkcyjnym systemem trenerskim.

## Autorzy

Authors: Martino Sebastiani, Zuzanna Klimaszewska

Projekt przygotowany w kontekście studiów podyplomowych WSB Merito: Analiza Danych / Data Science z elementami AI.
