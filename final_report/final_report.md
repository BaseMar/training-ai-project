# Projekt i implementacja systemu sztucznej inteligencji do analizy danych treningowych

## Strona tytułowa

TODO: Uzupełnić tytuł projektu, autora, kierunek, promotora lub prowadzącego, uczelnię oraz rok akademicki.

# Streszczenie

TODO: Krótko opisać cel projektu, syntetyczny zbiór danych, etapy prac, najważniejsze wyniki oraz ograniczenia.

# 1. Wprowadzenie

TODO: Wprowadzić temat projektu i wyjaśnić, dlaczego analiza danych treningowych jest użyteczna.

## 1.1. Kontekst problemu

TODO: Opisać problem analizy treningu, personalizacji planów i monitorowania postępów.

## 1.2. Motywacja projektu

TODO: Uzasadnić praktyczną wartość wykorzystania danych treningowych do wspierania decyzji użytkownika.

## 1.3. Cel projektu

TODO: Sformułować główny cel: demonstracyjny system AI do analizy danych treningowych i generowania planów.

## 1.4. Zakres projektu

TODO: Wskazać elementy objęte projektem oraz jawnie wymienić ograniczenia zakresu.

## 1.5. Struktura raportu

TODO: Krótko zapowiedzieć zawartość kolejnych rozdziałów raportu.

# 2. Cele projektu

TODO: Opisać cele biznesowe, techniczne i pytania projektowe.

## 2.1. Cel biznesowy

TODO: Wyjaśnić praktyczną wartość systemu dla użytkownika i prezentacji wyników.

## 2.2. Cel naukowy i techniczny

TODO: Opisać predykcję obciążeń, porównanie modeli, cechy historyczne i rekomender.

## 2.3. Pytania badawcze / projektowe

TODO: Wypisać pytania dotyczące predykcji ciężaru, wyboru modelu, reguł bezpieczeństwa, cold start i dashboardu.

# 3. Architektura systemu

TODO: Przedstawić ogólny przepływ danych i rolę głównych komponentów.

## 3.1. Ogólna architektura rozwiązania

TODO: Opisać pipeline od generatora danych do dashboardu Streamlit.

## 3.2. Etapy przetwarzania danych

TODO: Opisać etapy: EDA, modelowanie i rekomender, demonstrator oraz dashboard.

## 3.3. Rola generatora danych

TODO: Wyjaśnić, jak generator tworzy syntetyczne dane treningowe.

## 3.4. Rola modelu ML

TODO: Opisać model regresyjny jako komponent predykcji sugerowanego ciężaru.

## 3.5. Rola systemu rekomendacyjnego

TODO: Opisać, jak rekomender łączy profil użytkownika, historię, podobnych użytkowników, model ML i reguły bezpieczeństwa.

## 3.6. Rola dashboardu

TODO: Opisać dashboard jako warstwę prezentacyjną i demonstracyjną.

# 4. Charakterystyka danych

TODO: Scharakteryzować pochodzenie, strukturę i ograniczenia datasetu.

## 4.1. Źródło danych

TODO: Wskazać, że dane są syntetyczne i pochodzą z generatora projektu.

## 4.2. Struktura zbioru danych

TODO: Uzupełnić liczbę rekordów, użytkowników, sesji, ćwiczeń, zakres dat i główne wymiary.

## 4.3. Opis kolumn

TODO: Dodać tabelę opisującą najważniejsze kolumny datasetu.

## 4.4. Dane syntetyczne - zalety

TODO: Opisać kontrolę nad danymi, skalowalność, brak problemów prywatności i symulację profili.

## 4.5. Dane syntetyczne - ograniczenia

TODO: Opisać brak danych realnych, zależność od jakości generatora i ostrożność interpretacji.

## 4.6. Problem realizmu danych treningowych

TODO: Wyjaśnić, że poziom zaawansowania nie wystarcza do określenia realnej siły użytkownika.

# 5. Eksploracyjna analiza danych - EDA

TODO: Opisać cele, najważniejsze rozkłady, trendy i wnioski z EDA.

## 5.1. Cel analizy EDA

TODO: Wyjaśnić, po co analizowano strukturę danych przed modelowaniem.

## 5.2. Podstawowe statystyki datasetu

TODO: Uzupełnić liczby rekordów, użytkowników, sesji i ćwiczeń.

## 5.3. Rozkład poziomów zaawansowania

TODO: Opisać rozkład `level` i dodać miejsce na wykres.

## 5.4. Rozkład splitów treningowych

TODO: Opisać rozkład `split` i dodać miejsce na wykres.

## 5.5. Rozkład faz treningowych

TODO: Opisać rozkład `phase` i dodać miejsce na wykres.

## 5.6. Analiza ćwiczeń

TODO: Opisać najczęściej występujące ćwiczenia i dodać miejsce na wykres top 10.

## 5.7. Analiza objętości treningowej

TODO: Opisać zmienną `volume` oraz trend objętości w czasie.

## 5.8. Analiza użytkowników

TODO: Opisać różnice między użytkownikami i przykład analizy wybranego `user_id`.

## 5.9. Wnioski z EDA

TODO: Podsumować, co EDA mówi o modelowaniu, personalizacji i ograniczeniach zmiennej `level`.

# 6. Przygotowanie danych do modelowania

TODO: Opisać definicję problemu, target, cechy, feature engineering, split i baseline.

## 6.1. Definicja problemu predykcyjnego

TODO: Zdefiniować zadanie regresji przewidujące sugerowany ciężar.

## 6.2. Zmienna docelowa

TODO: Opisać `weight` jako target modelu.

## 6.3. Cechy wejściowe

TODO: Wymienić cechy kategoryczne, numeryczne i historyczne użyte w modelowaniu.

## 6.4. Feature engineering

TODO: Opisać cechy takie jak `volume`, `e1rm_epley`, cechy opóźnione i rolling features.

## 6.5. Cechy historyczne

TODO: Wymienić i wyjaśnić cechy poprzednich serii oraz średnie kroczące.

## 6.6. Podział train/test

TODO: Opisać czasowy podział danych na zbiór treningowy i testowy.

## 6.7. Baseline

TODO: Opisać model bazowy `naive_prev_weight` i jego znaczenie porównawcze.

# 7. Modelowanie i wybór modelu

TODO: Przedstawić testowane modele, metryki, wyniki i wybór modelu finalnego.

## 7.1. Testowane modele

TODO: Opisać Ridge Regression, Random Forest, HistGradientBoosting i baseline.

## 7.2. Metryki oceny

TODO: Wyjaśnić MAE, RMSE, R² oraz trafienia w progach 2.5 kg, 5 kg i 10 kg.

## 7.3. Wyniki porównania modeli

TODO: Dodać tabelę wyników porównującą modele według głównych metryk.

## 7.4. Wybór modelu finalnego

TODO: Uzasadnić wybór modelu finalnego na podstawie wyników.

## 7.5. Interpretacja wyników

TODO: Zinterpretować błąd w kilogramach, przewagę nad baseline i ograniczenia metryk.

## 7.6. Ewaluacja grupowa

TODO: Opisać wyniki dla grup takich jak `level`, `phase`, `sex` i `exercise`.

## 7.7. Wnioski z modelowania

TODO: Podsumować znaczenie historii użytkownika i rolę modelu jako komponentu systemu.

# 8. System rekomendacyjny

TODO: Opisać założenia rekomendera, dobór planu, integrację ML i reguły bezpieczeństwa.

## 8.1. Założenia systemu rekomendacyjnego

TODO: Opisać generowanie planu, dobór splitu, ćwiczeń, serii, powtórzeń, RIR, fatigue i ciężaru.

## 8.2. Rekomendacja splitu

TODO: Opisać reguły doboru splitu na podstawie liczby dni treningowych.

## 8.3. Dobór ćwiczeń

TODO: Wyjaśnić dobór ćwiczeń według kategorii ruchu, splitu, podobnych użytkowników i fallbacku.

## 8.4. Dobór parametrów serii

TODO: Opisać dobór serii, powtórzeń, RIR i fatigue zależnie od fazy treningowej.

## 8.5. Wykorzystanie modelu ML w rekomenderze

TODO: Opisać użycie `model_predicted_weight` jako jednego ze źródeł rekomendacji.

## 8.6. Reguły bezpieczeństwa

TODO: Opisać ograniczenia progresji, deload, wysokie fatigue, niski RIR, wiek i zaokrąglanie ciężaru.

## 8.7. Problem cold start

TODO: Wyjaśnić ograniczenia rekomendacji dla użytkownika bez historii.

## 8.8. Kalibracja siłowa nowych użytkowników

TODO: Opisać wykorzystanie ciężarów roboczych podanych przez użytkownika jako punktów odniesienia.

## 8.9. Źródła finalnego ciężaru

TODO: Opisać źródła takie jak `user_history`, `strength_calibration`, `model_prediction` i `fallback_median`.

## 8.10. Wnioski dotyczące rekomendera

TODO: Podsumować hybrydowe podejście łączące dane, model, historię, kalibrację i reguły.

# 9. Demonstrator systemu - Etap 3

TODO: Opisać demonstrator pokazujący działanie systemu end-to-end.

## 9.1. Cel demonstratora

TODO: Wyjaśnić, co pokazuje Etap 3 i jak łączy elementy projektu.

## 9.2. Scenariusze demonstracyjne

TODO: Opisać przygotowane scenariusze użytkowników.

## 9.3. Porównanie scenariuszy

TODO: Dodać miejsce na tabelę `scenario_comparison.csv`.

## 9.4. Przykładowe plany treningowe

TODO: Wstawić i omówić jeden lub dwa przykładowe wygenerowane plany.

## 9.5. Reguły bezpieczeństwa w demonstratorze

TODO: Opisać przykłady zastosowanych korekt i ich znaczenie.

## 9.6. Wnioski z demonstratora

TODO: Podsumować, czy demonstrator potwierdza działanie przepływu od profilu użytkownika do planu.

# 10. Dashboard Streamlit - Etap 4

TODO: Opisać dashboard jako warstwę integrującą i prezentującą projekt.

## 10.1. Cel dashboardu

TODO: Wyjaśnić rolę dashboardu w prezentacji wyników i działania systemu.

## 10.2. Struktura aplikacji

TODO: Wymienić i krótko opisać zakładki aplikacji.

## 10.3. Zakładka Overview

TODO: Opisać pipeline projektu, KPI datasetu i cztery etapy projektu.

## 10.4. Zakładka Dataset

TODO: Opisać statystyki globalne, filtry, analizę użytkownika, ranking i podgląd danych.

## 10.5. Zakładka EDA

TODO: Opisać rozkłady, top ćwiczeń i trend volume.

## 10.6. Zakładka ML Model

TODO: Opisać status modelu, metryki, porównanie modeli i ewaluację grupową.

## 10.7. Zakładka Recommendation Demo

TODO: Opisać scenariusze, KPI planu, listę dni treningowych i tabelę techniczną.

## 10.8. Zakładka Safety Rules

TODO: Opisać rozkład korekt, ćwiczenia z korektą i interpretację reguł.

## 10.9. Zakładka Live Generator

TODO: Opisać tryb użytkownika z historią, tryb kalibracji, generowanie planu i eksport CSV.

## 10.10. Wnioski z dashboardu

TODO: Podsumować, jak dashboard integruje etapy projektu w jednym widoku demonstracyjnym.

# 11. Ograniczenia projektu

TODO: Zebrać ograniczenia danych, generatora, modelu, rekomendera i dashboardu.

## 11.1. Syntetyczny charakter danych

TODO: Opisać konsekwencje pracy na danych syntetycznych.

## 11.2. Ograniczenia generatora

TODO: Wskazać, że generator może nie odtworzyć wszystkich zależności treningowych.

## 11.3. Ograniczenia modelu regresyjnego

TODO: Opisać zależność modelu od dostępnych cech i jakości danych.

## 11.4. Problem cold start

TODO: Wyjaśnić, dlaczego nowi użytkownicy wymagają kalibracji siłowej.

## 11.5. Ograniczenia zmiennej `level`

TODO: Opisać, dlaczego `level` nie zastępuje informacji o realnej sile.

## 11.6. Brak walidacji eksperckiej

TODO: Zaznaczyć potrzebę walidacji przez trenera i brak charakteru porady medycznej.

## 11.7. Ograniczenia dashboardu

TODO: Opisać dashboard jako demonstrator, a nie aplikację produkcyjną.

# 12. Możliwości rozwoju

TODO: Opisać kierunki dalszego rozwoju danych, modeli, rekomendera i dashboardu.

## 12.1. Użycie danych rzeczywistych

TODO: Opisać możliwość zastąpienia danych syntetycznych rzeczywistymi danymi treningowymi.

## 12.2. Dodanie nowych cech użytkownika

TODO: Wymienić potencjalne cechy, np. masa ciała, wzrost, staż, wyniki bazowe, kontuzje i cel.

## 12.3. Rozbudowa modelu personalizacji

TODO: Opisać możliwe modele sekwencyjne, rekomendacje użytkownik-ćwiczenie i personalizację progresji.

## 12.4. Rozbudowa systemu rekomendacyjnego

TODO: Opisać lepszy dobór ćwiczeń, warianty planów, periodyzację i autoregulację.

## 12.5. Walidacja ekspercka

TODO: Opisać możliwość konsultacji i oceny systemu przez trenera personalnego.

## 12.6. Rozbudowa dashboardu

TODO: Opisać możliwe ulepszenia UI, zapis użytkowników, historię planów i eksport PDF.

# 13. Podsumowanie i wnioski końcowe

TODO: Podsumować wykonane prace, wyniki, wnioski metodologiczne i finalną ocenę projektu.

## 13.1. Podsumowanie wykonanych prac

TODO: Wymienić wykonane etapy: generator, EDA, modelowanie, rekomender i dashboard.

## 13.2. Najważniejsze wyniki

TODO: Opisać finalny model, jego główne metryki, demonstrator i dashboard.

## 13.3. Wnioski metodologiczne

TODO: Podkreślić znaczenie historii użytkownika, ograniczenia `level`, cold start i podejście hybrydowe.

## 13.4. Wniosek końcowy

TODO: Sformułować końcowy wniosek o spełnieniu celu projektu demonstracyjnego.

# Bibliografia

TODO: Uzupełnić źródła dotyczące scikit-learn, Streamlit, Random Forest, metryk regresji, RIR, treningu siłowego i systemów rekomendacyjnych.

# Załączniki

TODO: Dodać materiały uzupełniające, które wspierają raport główny.

## Załącznik A. Struktura repozytorium

TODO: Opisać najważniejsze katalogi i pliki projektu.

## Załącznik B. Najważniejsze skrypty

TODO: Opisać skrypty EDA, modelowania, demonstratora i aplikację Streamlit.

## Załącznik C. Przykładowy wygenerowany plan

TODO: Dodać jeden przykładowy plan wygenerowany przez system.

## Załącznik D. Wybrane metryki modelu

TODO: Dodać tabelę porównania modeli i/lub ewaluację grupową.
