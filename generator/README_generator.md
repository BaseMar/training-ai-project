# Dokumentacja generatora syntetycznych danych treningowych

## Kontekst projektu

Generator jest pierwszym etapem projektu zaliczeniowego realizowanego na studiach
podyplomowych WSB Merito w Poznaniu **Analiza Danych - Data Science z Elementami AI**.

Generator został przygotowany jako źródło syntetycznych próbek treningowych, które
mogą zostać wykorzystane w dalszych etapach projektu: eksploracyjnej analizie danych
(EDA), budowie cech, trenowaniu modeli predykcyjnych oraz projektowaniu modułu
rekomendacji planów treningowych.

## 1. Cel generatora

Celem generatora jest stworzenie realistycznego, syntetycznego zbioru danych
odwzorowującego przebieg treningu siłowego wielu użytkowników w dłuższym okresie.
Dane nie są generowane jako proste losowe wartości. Generator symuluje stan osoby
trenującej i aktualizuje go po każdej sesji.

W modelu uwzględniono między innymi:

- indywidualny profil użytkownika,
- poziom zaawansowania treningowego,
- masę ciała i płeć,
- systematyczność treningową,
- zdolność regeneracji,
- tempo adaptacji siłowej,
- fazy treningowe: hipertrofia, siła i deload,
- zmęczenie ostre i przewlekłe,
- efekt plateau,
- homeostazę i superkompensację,
- losowe wahania dyspozycji dnia,
- możliwość wystąpienia drobnej kontuzji.

Takie podejście pozwala wygenerować dane, które lepiej przypominają rzeczywiste
dzienniki treningowe niż dane tworzone wyłącznie przez losowanie ciężaru i liczby
powtórzeń.

## 2. Jakie dane generuje

Generator tworzy dane na poziomie pojedynczej serii ćwiczenia. Oznacza to, że jeden
wiersz w pliku wynikowym odpowiada jednej wykonanej serii w ramach konkretnej sesji
treningowej.

Proces generowania obejmuje:

1. Utworzenie populacji syntetycznych użytkowników.
2. Przypisanie każdemu użytkownikowi profilu biologicznego i treningowego.
3. Wybór splitu treningowego, np. `ppl`, `upper_lower` albo `fbw`.
4. Symulację kolejnych dni w zadanym przedziale lat.
5. Sprawdzenie, czy użytkownik danego dnia wykona trening zgodnie z poziomem
   systematyczności.
6. Wybór typu treningu i ćwiczeń z planu.
7. Wygenerowanie serii, powtórzeń, ciężaru, RIR i zmęczenia.
8. Aktualizację stanu użytkownika po sesji.

Wygenerowany zbiór zawiera informacje przydatne do analizy:

- historii treningowej użytkownika,
- zmian ciężaru i liczby powtórzeń w czasie,
- wpływu zmęczenia na wyniki,
- różnic między ćwiczeniami,
- różnic między poziomami zaawansowania,
- zależności między fazą treningową a parametrami sesji.

## 3. Jakie kolumny tworzy

Aktualna wersja generatora tworzy plik CSV z następującymi kolumnami:

| Kolumna | Typ danych | Opis |
| --- | --- | --- |
| `user_id` | liczba całkowita | Identyfikator syntetycznego użytkownika. |
| `session_id` | liczba całkowita | Identyfikator sesji treningowej. |
| `date` | data | Data wykonania sesji. |
| `exercise` | tekst | Nazwa ćwiczenia, np. `Bench Press`, `Squat`, `Deadlift`. |
| `set_number` | liczba całkowita | Numer serii w ramach danego ćwiczenia. |
| `reps` | liczba całkowita | Liczba wykonanych powtórzeń. |
| `weight` | liczba zmiennoprzecinkowa | Ciężar roboczy w kilogramach. |
| `fatigue` | liczba zmiennoprzecinkowa | Aktualny poziom zmęczenia użytkownika. |
| `rir` | liczba zmiennoprzecinkowa | Reps In Reserve, czyli liczba powtórzeń pozostających w zapasie. |
| `level` | tekst | Poziom zaawansowania: `beginner`, `intermediate` albo `advanced`. |
| `split` | tekst | Rodzaj planu treningowego: `ppl`, `upper_lower` albo `fbw`. |
| `phase` | tekst | Aktualna faza treningowa: `hypertrophy`, `strength` albo `deload`. |
| `sex` | tekst | Płeć syntetycznego użytkownika: `male` albo `female`. |

Przykładowy rekord opisuje więc konkretną serię wykonaną przez konkretnego
użytkownika w konkretnej sesji. Dzięki temu możliwe jest późniejsze agregowanie
danych na różnych poziomach, np. seria, ćwiczenie, sesja, tydzień albo użytkownik.

## 4. Jakie parametry można zmieniać

Najważniejsze parametry generatora znajdują się w pliku `generator/config.py`.

### Parametry uruchomieniowe

Podstawowe parametry można zmienić z poziomu CLI:

```bash
python generator/main.py --users 100 --years 3 --output data/FINAL_ENGINE_V4.csv
```

| Parametr | Domyślna wartość | Znaczenie |
| --- | --- | --- |
| `--users` | `100` | Liczba syntetycznych użytkowników. |
| `--years` | `3` | Liczba lat symulacji. |
| `--output` | `data/FINAL_ENGINE_V4.csv` | Ścieżka zapisu pliku CSV. |
| `--no-save` | brak | Uruchamia generator bez zapisywania pliku wynikowego. |

### Parametry w `SimConfig`

| Parametr | Domyślna wartość | Znaczenie |
| --- | --- | --- |
| `users` | `100` | Liczba użytkowników. |
| `years` | `3` | Czas trwania symulacji w latach. |
| `start_date` | `2022-01-01` | Data początkowa generowanych treningów. |
| `fatigue_cap` | `2.0` | Maksymalny poziom zmęczenia. |
| `level_up_ratio` | `1.35` | Próg awansu poziomu zaawansowania. |
| `level_down_ratio` | `0.80` | Próg spadku poziomu zaawansowania. |
| `bad_day_prob` | `0.07` | Prawdopodobieństwo słabszego dnia treningowego. |
| `great_day_prob` | `0.04` | Prawdopodobieństwo bardzo dobrego dnia treningowego. |
| `injury_prob_base` | `0.0018` | Bazowe prawdopodobieństwo drobnej kontuzji. |
| `base_progression_rate` | `0.012` | Bazowe tempo progresji siłowej. |
| `weight_clip` | `(2.5, 300.0)` | Minimalny i maksymalny ciężar w danych wyjściowych. |
| `reps_clip` | `(1, 25)` | Minimalna i maksymalna liczba powtórzeń. |
| `output_path` | `data/FINAL_ENGINE_V4.csv` | Domyślna ścieżka zapisu wyniku. |

### Fazy treningowe

Generator korzysta z trzech faz:

| Faza | Rola w symulacji |
| --- | --- |
| `hypertrophy` | Większa objętość i wyższy zakres powtórzeń. |
| `strength` | Większa intensywność i niższy zakres powtórzeń. |
| `deload` | Obniżenie obciążenia i zatrzymanie progresji. |

Dla każdej fazy można zmienić:

- długość fazy w liczbie sesji,
- modyfikator liczby powtórzeń,
- modyfikator intensywności,
- wpływ fazy na tempo progresji.

### Profile ćwiczeń

Każde ćwiczenie ma własny profil zawierający:

- `mult` - względny poziom siły względem bazowego 1RM użytkownika,
- `fatigue` - wpływ ćwiczenia na zmęczenie,
- `type` - typ ćwiczenia,
- `rep_range` - preferowany zakres powtórzeń,
- `intensity_center` - typowy poziom intensywności,
- `rir_target` - docelowy zakres RIR,
- `progression_rate` - względne tempo progresji.

Zmiana tych wartości pozwala sterować charakterem danych. Przykładowo zwiększenie
`fatigue` dla martwego ciągu spowoduje szybszą akumulację zmęczenia po tym ćwiczeniu,
a zmiana `progression_rate` wpłynie na tempo wzrostu szacowanego 1RM.

### Plany treningowe

W pliku `config.py` znajdują się również definicje planów:

- `ppl`,
- `upper_lower`,
- `fbw`.

Każdy plan jest zdefiniowany osobno dla poziomów:

- `beginner`,
- `intermediate`,
- `advanced`.

Dzięki temu można testować różne struktury treningowe i analizować, jak wpływają
na wygenerowane dane.

## 5. Gdzie zapisuje wynik

Domyślnie generator zapisuje wynik do pliku:

```text
data/FINAL_ENGINE_V4.csv
```

Ścieżkę można zmienić parametrem `--output`, np.:

```bash
python generator/main.py --users 50 --years 1 --output data/sample_training_data.csv
```

Jeżeli katalog docelowy nie istnieje, generator tworzy go automatycznie.

Możliwe jest również uruchomienie generatora bez zapisu:

```bash
python generator/main.py --users 10 --years 1 --no-save
```

Taki tryb jest przydatny do szybkiego sprawdzenia działania generatora oraz podglądu
kształtu danych.

## 6. Jak dane trafiają do EDA i modelowania

Plik `data/FINAL_ENGINE_V4.csv` stanowi punkt wejścia do kolejnych etapów projektu.
Po wygenerowaniu dane mogą zostać wczytane w notebooku lub skrypcie analitycznym,
np. za pomocą biblioteki `pandas`:

```python
import pandas as pd

df = pd.read_csv("data/FINAL_ENGINE_V4.csv")
```

### EDA

W etapie eksploracyjnej analizy danych można badać między innymi:

- rozkład liczby sesji na użytkownika,
- rozkład ćwiczeń i splitów treningowych,
- zmiany ciężaru w czasie,
- zależność między zmęczeniem a wynikiem serii,
- różnice między poziomami zaawansowania,
- wpływ fazy treningowej na liczbę powtórzeń i ciężar,
- objętość treningową liczoną jako `weight * reps`.

Przykładowe agregacje:

```python
df["volume"] = df["weight"] * df["reps"]

weekly_volume = (
    df.assign(date=pd.to_datetime(df["date"]))
      .set_index("date")
      .groupby("user_id")
      .resample("W")["volume"]
      .sum()
      .reset_index()
)
```

### Modelowanie

Dane z generatora mogą zostać wykorzystane do przygotowania modeli predykcyjnych.
Przykładowe zadania modelowania:

- predykcja ciężaru w kolejnej serii,
- predykcja ciężaru w kolejnej sesji dla danego ćwiczenia,
- przewidywanie progresu siłowego użytkownika,
- klasyfikacja poziomu zaawansowania,
- ocena ryzyka stagnacji lub spadku wyniku,
- rekomendacja parametrów kolejnego treningu.

Potencjalne cechy wejściowe do modeli:

- `exercise`,
- `reps`,
- `weight`,
- `fatigue`,
- `rir`,
- `level`,
- `split`,
- `phase`,
- `sex`,
- historia poprzednich wyników użytkownika,
- średnia objętość z ostatnich sesji,
- trend ciężaru dla danego ćwiczenia,
- liczba dni od poprzedniego treningu.

Przykładowa zmienna celu:

```python
df = df.sort_values(["user_id", "exercise", "date", "session_id", "set_number"])
df["next_weight"] = df.groupby(["user_id", "exercise"])["weight"].shift(-1)
```

W ten sposób generator dostarcza dane bazowe, które można następnie przekształcić
w zbiór uczący dla modeli regresyjnych, klasyfikacyjnych lub rekomendacyjnych.

## Struktura modułów generatora

| Plik | Rola |
| --- | --- |
| `config.py` | Konfiguracja faz, ćwiczeń, splitów, planów i parametrów symulacji. |
| `models.py` | Modele użytkownika, stanu ćwiczeń, zmęczenia, adaptacji i progresji. |
| `session.py` | Symulacja pojedynczej sesji treningowej i pojedynczych serii. |
| `generator.py` | Orkiestracja całej symulacji i zapis danych do CSV. |
| `main.py` | Interfejs uruchomieniowy z poziomu konsoli. |

## Uruchomienie generatora

Przykładowe uruchomienie pełnej symulacji:

```bash
python generator/main.py --users 100 --years 3
```

Przykładowe uruchomienie krótkiej symulacji testowej:

```bash
python generator/main.py --users 5 --years 1 --no-save
```

Po poprawnym uruchomieniu program wypisuje rozmiar wygenerowanej ramki danych oraz
podgląd pierwszych rekordów.
