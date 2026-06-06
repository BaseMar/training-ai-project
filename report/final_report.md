# Projekt i implementacja systemu sztucznej inteligencji do analizy danych treningowych

## Strona tytułowa

TODO: Uzupełnić tytuł projektu, autora, kierunek, promotora lub prowadzącego, uczelnię oraz rok akademicki.

# Streszczenie

Celem projektu było zaprojektowanie i zaimplementowanie demonstracyjnego systemu sztucznej inteligencji do analizy danych treningowych oraz generowania spersonalizowanych planów treningowych. Projekt obejmuje pełny proces typowy dla data science: przygotowanie danych, eksploracyjną analizę, feature engineering, modelowanie regresyjne, implementację systemu rekomendacyjnego oraz przygotowanie dashboardu prezentacyjnego.

Ze względu na brak rzeczywistych danych użytkowników wykorzystano syntetyczny zbiór danych wygenerowany specjalnie na potrzeby projektu. Dane opisują treningi na poziomie pojedynczej serii i zawierają informacje potrzebne do analizy obciążeń, zachowania użytkowników oraz parametrów wykonywanych ćwiczeń.

W części modelującej zdefiniowano problem regresyjny polegający na przewidywaniu ciężaru treningowego. Porównano kilka modeli, w tym Ridge Regression, Random Forest, HistGradientBoosting oraz prosty baseline oparty na poprzednim ciężarze. Najlepszy wynik według metryki MAE uzyskał model Random Forest, osiągając średni błąd bezwzględny około 3.86 kg.

Na podstawie przygotowanych danych i modelu zbudowano hybrydowy system rekomendacyjny, który generuje tygodniowe plany treningowe. Rekomendacje uwzględniają profil użytkownika, dostępne dni treningowe, fazę treningową, historię lub dane podobnych użytkowników, predykcję modelu ML oraz reguły bezpieczeństwa. Całość została zaprezentowana w dashboardzie Streamlit, który umożliwia przegląd danych, wyników analizy, informacji o modelu oraz generowanie planu treningowego na żywo.

Projekt potwierdził możliwość zbudowania kompletnego demonstratora systemu AI wspierającego analizę danych treningowych i generowanie planów. Jednocześnie wskazał ograniczenia wynikające z syntetycznego charakteru danych, problemu cold start, potrzeby kalibracji siłowej nowych użytkowników oraz braku walidacji rekomendacji na danych rzeczywistych i przez ekspertów treningowych.

# 1. Wprowadzenie
## 1.1. Kontekst problemu

Trening siłowy oraz ogólnie rozumiana aktywność fizyczna generują dużą ilość danych, które mogą być wykorzystywane do monitorowania postępów użytkownika, analizy obciążeń treningowych oraz wspomagania decyzji dotyczących dalszego planowania treningu. W praktyce dane takie mogą obejmować między innymi informacje o wykonywanych ćwiczeniach, liczbie serii, liczbie powtórzeń, użytym ciężarze, poziomie zmęczenia, subiektywnej trudności serii oraz historii wcześniejszych treningów.

W tradycyjnym podejściu decyzje dotyczące doboru ćwiczeń, objętości, intensywności oraz progresji podejmowane są przez użytkownika samodzielnie albo przy wsparciu trenera. Wymaga to doświadczenia, umiejętności interpretacji własnych wyników oraz regularnej kontroli historii treningowej. U osób początkujących i średniozaawansowanych może to prowadzić do zbyt szybkiej progresji, braku systematyczności, przypadkowego doboru ćwiczeń albo trudności w ocenie, czy aktualne obciążenie jest adekwatne do poziomu użytkownika.

Z punktu widzenia analizy danych i uczenia maszynowego problem ten można potraktować jako zagadnienie personalizacji. System powinien uwzględniać profil użytkownika, jego poziom zaawansowania, historię treningową, rodzaj ćwiczenia, fazę treningową oraz parametry serii. Rekomendacje treningowe nie powinny jednak wynikać wyłącznie z działania modelu ML. W praktyce muszą być interpretowane w kontekście użytkownika i ograniczane przez reguły bezpieczeństwa.

W niniejszym projekcie zaprojektowano i zaimplementowano system pokazujący, jak takie podejście może działać w praktyce: od przygotowania danych, przez analizę i modelowanie, aż po rekomendację planu treningowego oraz jego prezentację w aplikacji.

## 1.2. Motywacja projektu

Motywacją do realizacji projektu była chęć połączenia praktycznego problemu z obszaru treningu siłowego z metodami analizy danych i uczenia maszynowego. Dane treningowe mają charakter sekwencyjny i historyczny, ponieważ decyzja dotycząca kolejnej jednostki treningowej powinna wynikać z wcześniejszych wyników użytkownika. Oznacza to, że dane takie dobrze nadają się do eksperymentów z analizą trendów, cechami historycznymi oraz modelami predykcyjnymi.

Drugim istotnym aspektem była możliwość pokazania pełnego procesu projektowego typowego dla projektów data science. Projekt nie ogranicza się do wytrenowania modelu. Obejmuje również przygotowanie danych, interpretację wyników, porównanie modeli, implementację logiki rekomendacyjnej oraz warstwę prezentacyjną. Dzięki temu powstał kompletny demonstrator systemu, a nie jedynie pojedynczy notebook analityczny.

Dodatkową motywacją była analiza ograniczeń systemów rekomendacyjnych w kontekście nowych użytkowników. W projekcie pojawia się problem cold start, ponieważ użytkownik bez historii treningowej nie dostarcza informacji o swoim rzeczywistym poziomie siły. Z tego powodu w systemie przewidziano mechanizm kalibracji siłowej, który zostanie szerzej omówiony w rozdziale dotyczącym rekomendera.

## 1.3. Cel projektu

Głównym celem projektu było zaprojektowanie i zaimplementowanie demonstracyjnego systemu sztucznej inteligencji do analizy danych treningowych oraz generowania spersonalizowanych planów.

Cel ten został zrealizowany poprzez wykonanie kilku powiązanych etapów:

1. przygotowanie generatora syntetycznych danych treningowych,
2. wygenerowanie kanonicznego zbioru danych wykorzystywanego w dalszych etapach,
3. przeprowadzenie eksploracyjnej analizy danych,
4. przygotowanie cech wykorzystywanych w modelowaniu,
5. porównanie modeli regresyjnych przewidujących sugerowany ciężar,
6. wybór najlepszego modelu według przyjętej metryki jakości,
7. przygotowanie hybrydowego systemu rekomendacyjnego,
8. dodanie reguł bezpieczeństwa ograniczających rekomendacje,
9. przygotowanie demonstratora systemu,
10. stworzenie dashboardu Streamlit prezentującego działanie całego rozwiązania.

Model uczenia maszynowego jest w tym rozwiązaniu jednym z komponentów systemu, a nie samodzielnym trenerem personalnym. Finalna rekomendacja powstaje dopiero po połączeniu predykcji modelu z historią użytkownika, danymi podobnych użytkowników, kalibracją siłową oraz regułami bezpieczeństwa.

## 1.4. Zakres projektu

Zakres projektu obejmuje następujące elementy:

* implementację generatora danych syntetycznych,
* przygotowanie zbioru danych treningowych,
* eksploracyjną analizę danych,
* przygotowanie cech historycznych,
* modelowanie regresyjne,
* porównanie kilku modeli ML,
* implementację systemu rekomendacyjnego,
* przygotowanie scenariuszy demonstracyjnych,
* implementację dashboardu Streamlit.

Projekt ma charakter demonstracyjny i edukacyjny. Oznacza to, że jego celem jest pokazanie kompletnego procesu analizy danych i budowy systemu AI, a nie dostarczenie gotowej aplikacji produkcyjnej. System nie zastępuje trenera personalnego, fizjoterapeuty ani lekarza. Wygenerowane plany i rekomendacje powinny być traktowane jako przykład działania systemu, a nie jako gotowe zalecenia treningowe do stosowania bez konsultacji eksperckiej.

Projekt nie obejmuje:

* pracy na rzeczywistych danych użytkowników,
* integracji z aplikacją mobilną,
* logowania użytkowników,
* zapisu historii wygenerowanych planów w bazie danych,
* walidacji rekomendacji przez trenera personalnego,
* walidacji medycznej,
* wdrożenia produkcyjnego.

## 1.5. Struktura raportu

Raport został podzielony na kilka części odpowiadających kolejnym etapom projektu.

W pierwszych rozdziałach przedstawiono kontekst problemu, motywację, cele projektu oraz ogólną architekturę rozwiązania. Następnie opisano charakterystykę danych, sposób ich wygenerowania oraz strukturę zbioru. Kolejna część raportu dotyczy eksploracyjnej analizy danych, w tym rozkładów zmiennych, struktury użytkowników, ćwiczeń i faz treningowych.

W dalszej części opisano przygotowanie danych do modelowania, zdefiniowanie problemu predykcyjnego, utworzenie cech historycznych oraz wybór metryk oceny. Następnie przedstawiono porównanie modeli regresyjnych i wybór modelu finalnego. Kolejne rozdziały opisują system rekomendacyjny, sposób generowania planów, reguły bezpieczeństwa oraz problem cold start.

Ostatnie części raportu dotyczą demonstratora systemu, dashboardu Streamlit, ograniczeń projektu, możliwych kierunków rozwoju oraz końcowych wniosków.

# 2. Cele projektu
## 2.1. Cel biznesowy
Celem biznesowym projektu było przygotowanie systemu wspierającego analizę danych treningowych oraz generowanie planów dopasowanych do profilu użytkownika. W praktycznym ujęciu taki system może pomagać w lepszym zrozumieniu historii treningowej, monitorowaniu postępów oraz podejmowaniu decyzji dotyczących kolejnych jednostek treningowych.

System ma wspierać odpowiedzi na pytania takie jak:

* jakie ćwiczenia powinny znaleźć się w planie treningowym,
* jaki split treningowy jest odpowiedni dla liczby dostępnych dni,
* jak dobrać liczbę serii i powtórzeń,
* jak uwzględnić fazę treningową,
* jak interpretować poziom zmęczenia i RIR,
* kiedy ograniczyć progresję,
* jak wykorzystać historię użytkownika do personalizacji planu.

Z perspektywy użytkownika końcowego najważniejszym efektem projektu jest plan przedstawiony w czytelnej formie w dashboardzie. Zawiera on podział na dni treningowe, listę ćwiczeń, liczbę serii, liczbę powtórzeń, docelowy RIR, poziom zmęczenia oraz sugerowane obciążenie, jeżeli możliwe jest jego wiarygodne oszacowanie.

Z perspektywy osoby oceniającej projekt istotne jest również to, że system pokazuje pełny proces analityczny: od danych, przez analizę i modelowanie, aż po końcowy demonstrator.

## 2.2. Cel naukowy i techniczny

Celem naukowym i technicznym projektu było sprawdzenie, w jaki sposób metody analizy danych i uczenia maszynowego mogą zostać wykorzystane do wspomagania decyzji treningowych.

W projekcie zdefiniowano problem regresyjny polegający na przewidywaniu ciężaru użytego w serii treningowej. Zmienną docelową modelu jest weight, natomiast cechami wejściowymi są między innymi informacje o ćwiczeniu, poziomie użytkownika, fazie treningowej, liczbie powtórzeń, RIR, zmęczeniu oraz cechy historyczne, takie jak poprzedni ciężar i średnie kroczące z ostatnich treningów.

W ramach projektu porównano kilka podejść modelujących:

* prosty baseline oparty na poprzednim ciężarze,
* model liniowy Ridge Regression,
* Random Forest,
* HistGradientBoosting.

Jako główną metrykę wyboru modelu przyjęto MAE, ponieważ jest ona łatwa do interpretacji w kontekście treningu siłowego. MAE informuje, o ile kilogramów średnio myli się model przy przewidywaniu ciężaru. Najlepszy wynik uzyskał Random Forest, który został wykorzystany jako komponent wspierający rekomendację obciążenia.

Celem technicznym było również zaprojektowanie hybrydowego systemu rekomendacyjnego. System ten nie opiera się wyłącznie na predykcji modelu ML, lecz łączy kilka źródeł informacji:

* profil użytkownika,
* historię użytkownika,
* dane podobnych użytkowników,
* model regresyjny,
* kalibrację siłową dla nowych użytkowników,
* reguły bezpieczeństwa.

W tej części projektu kluczowe były pytania o to, jak dobrze można przewidywać ciężar treningowy na podstawie danych historycznych, który model sprawdza się najlepiej według przyjętej metryki, jak połączyć predykcję z regułami bezpieczeństwa oraz jak obsłużyć użytkowników bez historii treningowej. Ważnym pytaniem było także to, czy wyniki można przedstawić w sposób zrozumiały w dashboardzie.

Takie podejście jest bardziej praktyczne niż użycie samego modelu, ponieważ rekomendacje treningowe muszą uwzględniać ograniczenia związane z bezpieczeństwem, zmęczeniem, poziomem użytkownika i brakiem historii treningowej.

# 3. Architektura systemu
## 3.1. Ogólna architektura rozwiązania

Projekt został zaprojektowany jako wieloetapowy pipeline obejmujący generowanie danych, analizę, modelowanie, rekomendację oraz wizualizację wyników. Przepływ systemu można przedstawić następująco:

Generator danych → Dataset → EDA → Feature Engineering → Model ML → Rekomender → Reguły bezpieczeństwa → Dashboard Streamlit

Każdy etap pełni osobną rolę, ale jednocześnie przekazuje wyniki do kolejnych komponentów. Generator przygotowuje dane syntetyczne, EDA pozwala zrozumieć ich strukturę, a feature engineering przekształca je do postaci użytecznej dla modelowania. Model ML przewiduje sugerowany ciężar, natomiast rekomender łączy tę predykcję z informacjami o użytkowniku i regułami bezpieczeństwa. Ostatnim elementem jest dashboard, który prezentuje działanie systemu w formie interaktywnej aplikacji.

## 3.2. Etapy przetwarzania danych

Projekt został podzielony na cztery główne etapy.

Pierwszy etap obejmuje eksploracyjną analizę danych. Jego celem jest sprawdzenie jakości zbioru, poznanie rozkładów zmiennych oraz analiza ćwiczeń, poziomów zaawansowania, splitów i faz treningowych.

Drugi etap obejmuje przygotowanie danych do modelowania oraz budowę modeli ML. Utworzono w nim cechy historyczne, przygotowano zbiór modelowy, wykonano czasowy podział train/test, porównano kilka modeli regresyjnych oraz wybrano model finalny według metryki MAE.

Trzeci etap obejmuje demonstrator systemu end-to-end. Skrypt demonstracyjny nie trenuje modeli od nowa, lecz korzysta z modelu przygotowanego w etapie drugim. Dla kilku scenariuszy użytkowników generowane są przykładowe plany treningowe oraz tabela porównawcza.

Czwarty etap obejmuje dashboard Streamlit. Dashboard stanowi warstwę prezentacyjną projektu i integruje wyniki poprzednich etapów. Pozwala zaprezentować dataset, wyniki EDA, informacje o modelu, scenariusze rekomendacyjne, reguły bezpieczeństwa oraz generator planu działający na żywo.

## 3.3. Rola generatora danych

Generator danych pełni rolę źródła syntetycznego zbioru treningowego. Został wykorzystany dlatego, że projekt nie bazuje na rzeczywistych danych użytkowników. Dane syntetyczne pozwalają kontrolować strukturę datasetu oraz uwzględnić wielu użytkowników, różne poziomy zaawansowania, splity, fazy treningowe i ćwiczenia.

Dane generowane są na poziomie pojedynczej serii treningowej. Oznacza to, że jeden rekord odpowiada jednej serii konkretnego ćwiczenia wykonanej przez określonego użytkownika w ramach danej sesji treningowej. Taka granularność danych umożliwia analizę zarówno pojedynczych serii, jak i agregację do poziomu sesji, użytkownika lub ćwiczenia.

Generator uwzględnia między innymi:

* użytkowników,
* sesje treningowe,
* daty treningów,
* ćwiczenia,
* liczbę serii,
* liczbę powtórzeń,
* ciężar,
* poziom zmęczenia,
* RIR,
* poziom zaawansowania,
* split treningowy,
* fazę treningową,
* płeć.

W projekcie dataset wygenerowany przez generator jest traktowany jako kanoniczny zbiór wejściowy dla EDA, modelowania i demonstratora.

## 3.4. Rola modelu ML

Model uczenia maszynowego pełni funkcję komponentu wspierającego dobór sugerowanego obciążenia treningowego. Problem został zdefiniowany jako regresja, w której zmienną docelową jest weight, czyli ciężar użyty w serii.

Model nie jest jednak jedynym źródłem decyzji. W praktyce treningowej samo przewidywanie wartości liczbowej nie wystarcza, ponieważ rekomendacja musi być oceniona w kontekście historii użytkownika, fazy treningowej, zmęczenia, RIR oraz zasad bezpieczeństwa.

W projekcie model ML odpowiada za wyznaczenie wartości model_predicted_weight. Następnie wartość ta może zostać skorygowana przez system rekomendacyjny. W zależności od sytuacji finalny ciężar może pochodzić z historii użytkownika, kalibracji siłowej, predykcji modelu lub wartości orientacyjnej z danych.

## 3.5. Rola systemu rekomendacyjnego

System rekomendacyjny jest warstwą łączącą dane, model ML oraz reguły eksperckie. Jego zadaniem jest wygenerowanie planu treningowego dla użytkownika na podstawie informacji przygotowanych we wcześniejszych etapach.

Rekomender wykonuje kilka kroków:

1. analizuje profil użytkownika,
2. dobiera split treningowy na podstawie liczby dni treningowych,
3. wybiera ćwiczenia na podstawie splitu i danych podobnych użytkowników,
4. dobiera liczbę serii, powtórzeń, RIR i poziom zmęczenia,
5. wykorzystuje model ML do predykcji sugerowanego ciężaru,
6. sprawdza historię użytkownika lub kalibrację siłową,
7. stosuje reguły bezpieczeństwa,
8. zwraca finalny plan treningowy.

System ma charakter hybrydowy. Oznacza to, że nie jest wyłącznie modelem ML ani prostym zestawem reguł. Łączy podejście data-driven z informacją o użytkowniku i ograniczeniami bezpieczeństwa.

## 3.6. Rola dashboardu

Dashboard Streamlit pełni rolę końcowej warstwy prezentacyjnej projektu. Jego celem jest pokazanie działania rozwiązania w formie interaktywnej aplikacji.

Dashboard zawiera kilka zakładek:

* Overview,
* Dataset,
* EDA,
* ML Model,
* Recommendation Demo,
* Safety Rules,
* Live Generator.

Dzięki temu możliwe jest przejście od ogólnego opisu projektu, przez analizę danych i modelowanie, aż do wygenerowanego planu treningowego. Dashboard pozwala również pokazać różnicę między użytkownikiem z historią treningową a nowym użytkownikiem, dla którego potrzebna jest kalibracja siłowa.

Warstwa dashboardu jest istotna z punktu widzenia prezentacji projektu, ponieważ pozwala pokazać rezultat nie tylko jako kod i tabele, ale jako działający demonstrator systemu AI.

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
