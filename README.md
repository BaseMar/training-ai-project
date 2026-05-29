# Training AI Project

`training-ai-project` to repozytorium projektu dotyczącego budowy systemu AI do
analizy danych treningowych, predykcji progresu siłowego oraz generowania
spersonalizowanych rekomendacji treningowych.

Temat projektu:

**Projekt i implementacja systemu sztucznej inteligencji do analizy danych treningowych.**

## Opis projektu

Projekt dotyczy zaprojektowania i implementacji systemu AI wspierającego analizę
danych treningu siłowego. System bazuje na syntetycznym zbiorze danych, wykonuje
analizę eksploracyjną, przygotowuje dane do modelowania, trenuje model predykcyjny
oraz docelowo generuje rekomendacje treningowe.

Pierwszym ukończonym komponentem jest generator syntetycznych danych. Generator
symuluje użytkowników, sesje treningowe, ćwiczenia i pojedyncze serie. Zbiór
danych wygenerowany przez ten moduł stanowi punkt wejścia do kolejnych etapów
projektu.

## Cele projektu

Główne cele projektu:

- stworzenie syntetycznego zbioru danych treningowych,
- wykonanie eksploracyjnej analizy danych (EDA),
- przygotowanie cech treningowych do uczenia maszynowego,
- budowa modelu regresyjnego do predykcji ciężaru treningowego,
- stworzenie hybrydowego systemu rekomendacyjnego,
- przygotowanie demonstratora końcowego systemu.

## Architektura systemu

```text
generator/
    -> data/FINAL_ENGINE_V4.csv
        -> EDA
            -> inżynieria cech
                -> model ML
                    -> silnik rekomendacyjny
                        -> demonstrator systemu
```

Elementy przepływu danych:

- `generator/` tworzy syntetyczne dane treningowe na poziomie pojedynczej serii.
- `data/FINAL_ENGINE_V4.csv` jest kanonicznym zbiorem danych projektu.
- EDA analizuje rozkłady, jakość danych, trendy treningowe i zależności między zmiennymi.
- Inżynieria cech przygotowuje zmienne opisujące historię treningową, objętość, progres i zmęczenie.
- Model ML przewiduje docelowo ciężar lub progres w kolejnych jednostkach treningowych.
- Silnik rekomendacyjny łączy wynik modelu z regułami treningowymi i zasadami bezpieczeństwa.
- Demonstrator systemu prezentuje kompletny przepływ: dane, predykcja i rekomendacja.

## Struktura repozytorium

Aktualna i planowana struktura repozytorium:

```text
training-ai-project/
|-- data/
|   `-- FINAL_ENGINE_V4.csv
|-- generator/
|   |-- README_generator.md
|   |-- config.py
|   |-- generator.py
|   |-- main.py
|   |-- models.py
|   `-- session.py
|-- scripts/
|   |-- 01_eda.py
|   `-- 02_modeling_and_recommendation.py
|-- notebooks/
|-- outputs/
|   |-- eda_outputs/
|   |-- stage2_outputs/
|   `-- stage3_outputs/
|-- src/
|-- models/
|-- presentation/
|-- report/
|-- README.md
`-- requirements.txt
```

Opis katalogów:

- `generator/` - moduł generowania syntetycznych danych treningowych.
- `data/` - kanoniczny zbiór danych projektu: `FINAL_ENGINE_V4.csv`.
- `scripts/` - uruchamialne skrypty analityczne: `01_eda.py` oraz `02_modeling_and_recommendation.py`.
- `notebooks/` - planowane notebooki dla kolejnych etapów analizy i demonstracji.
- `src/` - planowane moduły pomocnicze do przetwarzania danych, inżynierii cech, trenowania modeli, rekomendacji i reguł bezpieczeństwa.
- `models/` - planowany katalog na zapisane modele predykcyjne, np. `best_weight_prediction_model.joblib`.
- `outputs/` - katalogi na wyniki EDA, modelowania i demonstratora.
- `report/` - obecny katalog roboczy na materiały raportowe.
- `final_report/` - TODO: katalog na pełny raport akademicki/projektowy.
- `presentation/` - planowany katalog na finalną prezentację projektu.

Planowane notebooki:

- `notebooks/01_eda.ipynb` - eksploracyjna analiza danych.
- `notebooks/02_modeling_and_recommendation.ipynb` - model ML i system rekomendacyjny.
- `notebooks/03_system_demo.ipynb` - demonstrator końcowy systemu.

Obecnie w repozytorium znajdują się skrypty dla Etapu 1 EDA oraz Etapu 2
modelowania ML i rekomendacji treningowych.

## Generator danych

Generator tworzy syntetyczne dane treningowe na poziomie pojedynczej serii
treningowej. Nie generuje wyłącznie losowych wartości. Symuluje proces treningowy
z uwzględnieniem profilu użytkownika, poziomu zaawansowania, splitu, fazy treningu,
zmęczenia, RIR, progresji siłowej i losowych wahań dyspozycji.

Generator tworzy następujące kolumny:

```text
user_id, session_id, date, exercise, set_number, reps, weight,
fatigue, rir, level, split, phase, sex
```

Szczegółowa dokumentacja generatora znajduje się w pliku
`generator/README_generator.md`.

Przykładowe uruchomienie:

```bash
python generator/main.py --users 100 --years 3
```

## Zbiór danych

Kanoniczny zbiór danych projektu znajduje się w pliku:

```text
data/FINAL_ENGINE_V4.csv
```

Zbiór danych został wygenerowany przez moduł `generator/` i jest używany jako
wejście do EDA, modelowania oraz demonstratora systemu.

Projekt świadomie nie używa osobnych katalogów `raw/`, `processed/` i `final/`.
Generator tworzy dane zgodne z ustalonym schematem i gotowe do dalszej analizy.
Mimo tego dane są walidowane w notebookach analitycznych przed użyciem w kolejnych
etapach.

## Etapy projektu

1. **Etap 1 - EDA**

   Analiza struktury danych, rozkładów, wartości skrajnych, zależności między
   zmiennymi oraz podstawowych trendów treningowych.

2. **Etap 2 - Model ML i system rekomendacyjny**

   Przygotowanie cech, trenowanie modelu regresyjnego do predykcji ciężaru oraz
   budowa hybrydowego rekomendera łączącego model ML z regułami treningowymi.

3. **Etap 3 - Demonstrator systemu**

   Przygotowanie końcowego przepływu pokazującego, jak system analizuje dane,
   wykonuje predykcję i generuje rekomendację treningową.

## Jak uruchomić projekt

### 1. Utworzenie środowiska Python

Utwórz i aktywuj środowisko Python zgodnie z lokalną konfiguracją systemu.
Można użyć na przykład `venv` albo Conda.

### 2. Instalacja zależności

Plik `requirements.txt` istnieje, ale obecnie jest pusty.

TODO: uzupełnić `requirements.txt` po ustaleniu finalnego zestawu bibliotek dla
EDA, modelowania i demonstratora.

Po uzupełnieniu zależności instalacja będzie wyglądała standardowo:

```bash
pip install -r requirements.txt
```

### 3. Uruchomienie generatora

```bash
python generator/main.py --users 100 --years 3
```

Wynik zostanie zapisany domyślnie do:

```text
data/FINAL_ENGINE_V4.csv
```

### 4. Uruchomienie analiz i notebooków

Aktualnie dostępny jest pierwszy etap EDA jako skrypt:

```bash
python scripts/01_eda.py
```

Skrypt zapisuje lokalne wyniki EDA do `outputs/eda_outputs/`.
Katalogi `outputs/*` zawierają odtwarzalne artefakty robocze i nie są commitowane.
Wybrane finalne wykresy lub tabele do raportu mogą zostać później ręcznie
przeniesione do `final_report/figures/` albo właściwego katalogu raportowego.

TODO: dodać wersje notebookowe `.ipynb` dla etapów 1-3.

## Główne wyniki projektu

Projekt generuje lub docelowo będzie generował:

- wykresy i tabele EDA,
- walidację jakości danych syntetycznych,
- dane przygotowane do modelowania,
- porównanie modeli predykcyjnych,
- zapisany model, np. `models/best_weight_prediction_model.joblib`,
- przykładowe rekomendacje i plany treningowe,
- lokalne artefakty robocze w `outputs/`, które można odtwórzyć przez ponowne
  uruchomienie odpowiednich skryptów.

Etap 2 jest dostępny w `scripts/02_modeling_and_recommendation.py`. Skrypt
zapisuje robocze wyniki do `outputs/stage2_outputs/`, a finalny model generuje
lokalnie jako `models/best_weight_prediction_model.joblib`. Plik `.joblib` nie
jest commitowany zwykłym Gitem ze względu na rozmiar; w przyszłości może zostać
udostępniony przez GitHub Release albo Git LFS.

## Ograniczenia

Najważniejsze ograniczenia projektu:

- dane są syntetyczne i nie pochodzą z rzeczywistych dzienników treningowych,
- model predykcyjny i rekomender mają charakter prototypowy,
- dane historyczne nie zawierają wieku użytkownika,
- system nie zastępuje profesjonalnej opieki trenerskiej, fizjoterapeutycznej ani medycznej,
- rekomendacje wymagają dalszej walidacji na danych rzeczywistych,
- aktualna struktura zależności i notebooków jest nadal rozwijana.

## Autorzy

- **Martino Sebastiani**
- **Zuzanna Klimaszewska**

Studia podyplomowe WSB Merito w Poznaniu:
**Analiza Danych / Data Science z elementami AI**
