"""Etap 2 projektu: modelowanie ML i prototyp rekomendacji treningowych.

Skrypt buduje komponent ML/AI po Etapie 1 EDA. Wejściem jest kanoniczny zbiór
danych `data/FINAL_ENGINE_V4.csv`, a wyniki robocze są zapisywane do
`outputs/stage2_outputs`. Finalny model predykcji ciężaru jest zapisywany jako
`models/best_weight_prediction_model.joblib`.

Ten etap obejmuje przygotowanie cech historycznych, trenowanie i porównanie
modeli regresyjnych, wybór najlepszego modelu, ewaluację per grupa, prostą
interpretację oraz prototyp hybrydowego rekomendera. Rekomender jest prototypem
wspierającym decyzję treningową, a nie narzędziem medycznym, fizjoterapeutycznym
ani produkcyjnym systemem trenerskim.
"""

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 140)
pd.set_option("display.width", 200)
sns.set_theme(style="whitegrid")


# ============================================================
# 1. Konfiguracja Etapu 2
# ============================================================

DATA_PATH = "data/FINAL_ENGINE_V4.csv"
OUTPUT_DIR = "outputs/stage2_outputs"
MODEL_FILENAME = "models/best_weight_prediction_model.joblib"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_FILENAME), exist_ok=True)

RANDOM_STATE = 42
MAX_TRAIN_SAMPLE = 250_000
MAX_TEST_SAMPLE = 100_000

# Domyślny tryb Etapu 2 zakłada trenowanie modeli. Tryb False służy tylko do
# ponownego użycia wcześniej zapisanego modelu i wymaga istniejącego pliku joblib.
RUN_MODEL_TRAINING = True


def print_section(title):
    """Wypisuje czytelny separator kolejnej części skryptu."""
    print("\n" + "#" * 110)
    print(title)
    print("#" * 110)


def display_table(obj, title=None):
    """Pokazuje tabelę w notebooku, a w konsoli wypisuje ją tekstowo."""
    if title:
        print("\n" + "=" * 110)
        print(title)
        print("=" * 110)
    try:
        display(obj)
    except NameError:
        print(obj)


def save_plot(filename):
    """Zapisuje aktualny wykres do katalogu wynikowego Etapu 2."""
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def make_one_hot_encoder():
    """Tworzy OneHotEncoder zgodny ze starszymi i nowszymi wersjami sklearn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def round_to_nearest(value, step=2.5):
    """Zaokrągla ciężar do najbliższego praktycznego kroku, np. 2.5 kg."""
    if pd.isna(value):
        return np.nan
    return round(float(value) / step) * step


def safe_median(series, default_value=0):
    """Zwraca medianę serii albo wartość domyślną dla pustych danych."""
    series = pd.Series(series).dropna()
    if len(series) == 0:
        return default_value
    return float(series.median())


def safe_mode(series, default_value=None):
    """Zwraca najczęstszą wartość serii albo wartość domyślną."""
    series = pd.Series(series).dropna()
    if len(series) == 0:
        return default_value
    return series.mode().iloc[0]


# ============================================================
# 2. Wczytanie i walidacja danych
# ============================================================

print_section("WCZYTANIE I WALIDACJA DANYCH")

df = pd.read_csv(DATA_PATH)

print(f"Liczba rekordów: {len(df):,}")
print(f"Liczba kolumn: {df.shape[1]:,}")
display_table(df.head(), "Pierwsze rekordy")

required_columns = [
    "user_id", "session_id", "date",
    "exercise", "set_number", "reps", "weight",
    "fatigue", "rir", "level", "split", "phase", "sex",
]

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Brakuje wymaganych kolumn: {missing_columns}")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
if df["date"].isna().sum() > 0:
    print("UWAGA: Występują niepoprawne daty po konwersji.")

print("Zakres dat:")
print("Od:", df["date"].min())
print("Do:", df["date"].max())

missing_table = df.isna().sum().to_frame("missing_count")
display_table(missing_table, "Braki danych")
print("\nLiczba duplikatów:", df.duplicated().sum())

if "age" not in df.columns:
    print("\nUWAGA:")
    print("W danych nie ma kolumny age.")
    print(
        "Wiek będzie używany tylko w regułach bezpieczeństwa jako dane profilu "
        "użytkownika, a nie jako cecha trenowana przez model."
    )


# ============================================================
# 3. Feature engineering pod model ML
# ============================================================

print_section("FEATURE ENGINEERING POD MODEL ML")

# volume opisuje prostą objętość serii i jest później używany w cechach historii.
df["volume"] = df["reps"] * df["weight"]
df["e1rm_epley"] = df["weight"] * (1 + df["reps"] / 30)

df = (
    df
    .sort_values(["user_id", "exercise", "date", "session_id", "set_number"])
    .reset_index(drop=True)
)

# Cechy historyczne pozwalają modelowi korzystać z poprzednich zachowań użytkownika
# dla danego ćwiczenia, zamiast uczyć się tylko z bieżącego rekordu.
df["prev_weight"] = df.groupby(["user_id", "exercise"])["weight"].shift(1)
df["prev_reps"] = df.groupby(["user_id", "exercise"])["reps"].shift(1)
df["prev_rir"] = df.groupby(["user_id", "exercise"])["rir"].shift(1)
df["prev_fatigue"] = df.groupby(["user_id", "exercise"])["fatigue"].shift(1)
df["prev_volume"] = df.groupby(["user_id", "exercise"])["volume"].shift(1)

for col in ["weight", "reps", "rir", "fatigue", "volume"]:
    # shift(1) przed rolling averages ogranicza data leakage: średnia krocząca
    # nie zawiera serii, dla której aktualnie przewidujemy ciężar.
    df[f"rolling_{col}_3"] = (
        df
        .groupby(["user_id", "exercise"])[col]
        .transform(lambda x: x.shift(1).rolling(3).mean())
    )

display_table(
    df[
        [
            "user_id", "exercise", "date", "reps", "weight",
            "prev_weight", "rolling_weight_3", "volume", "e1rm_epley",
        ]
    ].head(15),
    "Przykład cech historycznych",
)


# ============================================================
# 4. Przygotowanie zbioru modelowego
# ============================================================

print_section("PRZYGOTOWANIE ZBIORU MODELOWEGO")

# reps, rir i fatigue traktujemy tutaj jako planowane parametry serii, a nie jako
# znane przyszłe wyniki. Model odpowiada na pytanie: jaki ciężar dobrać dla
# zaplanowanej liczby powtórzeń i docelowej trudności.
FEATURES = [
    "exercise", "level", "split", "phase", "sex",
    "set_number", "reps", "fatigue", "rir",
    "prev_weight", "prev_reps", "prev_rir", "prev_fatigue", "prev_volume",
    "rolling_weight_3", "rolling_reps_3", "rolling_rir_3",
    "rolling_fatigue_3", "rolling_volume_3",
]
TARGET = "weight"

categorical_features = ["exercise", "level", "split", "phase", "sex"]
numeric_features = [col for col in FEATURES if col not in categorical_features]

model_ready = df.dropna(subset=FEATURES + [TARGET]).copy()

print(f"Liczba rekordów gotowych do modelowania: {len(model_ready):,}")
display_table(model_ready[FEATURES + [TARGET]].head(), "Dane modelowe")

model_ready.to_csv(
    os.path.join(OUTPUT_DIR, "model_ready_stage2.csv"),
    index=False,
)


# ============================================================
# 5. Podział czasowy train/test
# ============================================================

print_section("PODZIAŁ CZASOWY TRAIN/TEST")

# Podział czasowy lepiej przypomina realne użycie systemu: model uczy się na
# wcześniejszych treningach i przewiduje późniejsze rekordy.
cutoff_date = model_ready["date"].quantile(0.8)
train_df = model_ready[model_ready["date"] <= cutoff_date].copy()
test_df = model_ready[model_ready["date"] > cutoff_date].copy()

print("Data odcięcia:", cutoff_date)
print(f"Train: {len(train_df):,}")
print(f"Test: {len(test_df):,}")

if len(train_df) == 0 or len(test_df) == 0:
    raise ValueError(
        "Po podziale czasowym train/test jedna z części jest pusta. "
        "Sprawdź zakres dat w danych."
    )

train_df_sample = train_df.sample(
    min(len(train_df), MAX_TRAIN_SAMPLE),
    random_state=RANDOM_STATE,
)
test_df_sample = test_df.sample(
    min(len(test_df), MAX_TEST_SAMPLE),
    random_state=RANDOM_STATE,
)

print(f"Train po próbkowaniu: {len(train_df_sample):,}")
print(f"Test po próbkowaniu: {len(test_df_sample):,}")

X_train = train_df_sample[FEATURES]
y_train = train_df_sample[TARGET]
X_test = test_df_sample[FEATURES]
y_test = test_df_sample[TARGET]


# ============================================================
# 6. Pipeline preprocessingowy
# ============================================================

print_section("PIPELINE PREPROCESSINGOWY")


def build_preprocessor(scale_numeric=False):
    """Buduje preprocessing dla cech kategorycznych i numerycznych."""
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("cat", make_one_hot_encoder(), categorical_features),
            ("num", numeric_transformer, numeric_features),
        ]
    )


# ============================================================
# 7. Definicja modeli
# ============================================================

print_section("DEFINICJA MODELI")

# Porównujemy prosty model liniowy, model baggingowy i boostingowy. Daje to
# sensowny przekrój: interpretowalny baseline ML, stabilny model drzewiasty i
# mocniejszy model gradientowy.
models = {
    "ridge_regression": Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            ("model", Ridge(alpha=1.0)),
        ]
    ),
    "random_forest": Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=120,
                    max_depth=16,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
    "hist_gradient_boosting": Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=250,
                    learning_rate=0.06,
                    max_leaf_nodes=31,
                    l2_regularization=0.1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
}

print("Modele do porównania:")
for name in models.keys():
    print("-", name)


# ============================================================
# 8. Metryki i funkcje ewaluacyjne
# ============================================================

print_section("METRYKI I FUNKCJE EWALUACYJNE")


def calculate_regression_metrics(model_name, y_true, y_pred):
    """Liczy główne metryki regresji dla predykcji ciężaru."""
    y_pred = np.maximum(np.asarray(y_pred), 0)
    y_true = np.asarray(y_true)
    abs_error = np.abs(y_true - y_pred)

    # MAE jest główną metryką, bo jest łatwa do interpretacji w kilogramach.
    # Progi within_* pokazują praktyczną użyteczność predykcji dla treningu.
    return {
        "model": model_name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "within_2_5kg_percent": np.mean(abs_error <= 2.5) * 100,
        "within_5kg_percent": np.mean(abs_error <= 5.0) * 100,
        "within_10kg_percent": np.mean(abs_error <= 10.0) * 100,
    }


def evaluate_regression_model(model_name, model, X_test, y_test):
    """Uruchamia predykcję modelu i zwraca metryki oraz tabelę predykcji."""
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)
    results = calculate_regression_metrics(model_name, y_test, y_pred)
    predictions = pd.DataFrame(
        {
            "actual_weight": y_test.values,
            "predicted_weight": y_pred,
            "absolute_error": np.abs(y_test.values - y_pred),
        }
    )
    return results, predictions


def evaluate_naive_baseline(test_data):
    """Ewaluuje baseline, który jako predykcję przyjmuje poprzedni ciężar."""
    y_true = test_data[TARGET].values
    y_pred = test_data["prev_weight"].values
    y_pred = np.maximum(y_pred, 0)
    results = calculate_regression_metrics("naive_prev_weight", y_true, y_pred)
    predictions = pd.DataFrame(
        {
            "actual_weight": y_true,
            "predicted_weight": y_pred,
            "absolute_error": np.abs(y_true - y_pred),
        }
    )
    return results, predictions


# ============================================================
# 9. Trenowanie i porównanie modeli
# ============================================================

print_section("TRENOWANIE I PORÓWNANIE MODELI")

all_results = []
all_predictions = {}
trained_models = {}

# Naiwny baseline prev_weight jest prostym punktem odniesienia: dobry model ML
# powinien uzasadnić swoją złożoność poprawą względem poprzedniego ciężaru.
naive_results, naive_predictions = evaluate_naive_baseline(test_df_sample)
all_results.append(naive_results)
all_predictions["naive_prev_weight"] = naive_predictions
print("Baseline naiwny:", naive_results)

if RUN_MODEL_TRAINING:
    for model_name, pipeline in models.items():
        print(f"\nTrenowanie modelu: {model_name}")
        model = clone(pipeline)
        model.fit(X_train, y_train)

        results, predictions = evaluate_regression_model(
            model_name=model_name,
            model=model,
            X_test=X_test,
            y_test=y_test,
        )
        all_results.append(results)
        all_predictions[model_name] = predictions
        trained_models[model_name] = model
        print(results)

    results_df = pd.DataFrame(all_results).sort_values("MAE")
    display_table(results_df, "Porównanie modeli regresyjnych z baseline naiwnym")
    results_df.to_csv(
        os.path.join(OUTPUT_DIR, "model_comparison_results.csv"),
        index=False,
    )

    # Najlepszy model wybieramy tylko spośród modeli ML, ale porównujemy go także
    # z naiwnym baseline'em prev_weight.
    ml_results_df = results_df[results_df["model"] != "naive_prev_weight"].copy()
    best_model_name = ml_results_df.iloc[0]["model"]
    best_pipeline = trained_models[best_model_name]

    print("\nNajlepszy model ML według MAE:", best_model_name)
    joblib.dump(best_pipeline, MODEL_FILENAME)
    print("Model zapisano do:", MODEL_FILENAME)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=results_df, x="model", y="MAE")
    plt.title("Porównanie modeli - MAE")
    plt.xlabel("Model")
    plt.ylabel("MAE")
    plt.xticks(rotation=25)
    save_plot("model_comparison_mae.png")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=results_df, x="model", y="within_5kg_percent")
    plt.title("Odsetek predykcji z błędem do 5 kg")
    plt.xlabel("Model")
    plt.ylabel("% predykcji w granicy 5 kg")
    plt.xticks(rotation=25)
    save_plot("model_comparison_within_5kg.png")
else:
    if not os.path.exists(MODEL_FILENAME):
        raise FileNotFoundError(
            f"RUN_MODEL_TRAINING=False wymaga istniejącego modelu: {MODEL_FILENAME}"
        )
    best_pipeline = joblib.load(MODEL_FILENAME)
    best_model_name = "loaded_model"
    print("Wczytano zapisany model:", MODEL_FILENAME)

    loaded_results, loaded_predictions = evaluate_regression_model(
        model_name=best_model_name,
        model=best_pipeline,
        X_test=X_test,
        y_test=y_test,
    )
    all_results.append(loaded_results)
    all_predictions[best_model_name] = loaded_predictions
    results_df = pd.DataFrame(all_results).sort_values("MAE")


# ============================================================
# 10. Analiza najlepszego modelu
# ============================================================

print_section("ANALIZA NAJLEPSZEGO MODELU")

best_predictions = all_predictions[best_model_name].copy()

display_table(best_predictions.head(20), "Przykładowe predykcje najlepszego modelu")
best_predictions.to_csv(
    os.path.join(OUTPUT_DIR, "best_model_predictions.csv"),
    index=False,
)

plot_sample = best_predictions.sample(
    min(20_000, len(best_predictions)),
    random_state=RANDOM_STATE,
)

plt.figure(figsize=(7, 7))
sns.scatterplot(
    data=plot_sample,
    x="actual_weight",
    y="predicted_weight",
    alpha=0.25,
)
plt.title(f"Rzeczywisty ciężar vs predykcja - {best_model_name}")
plt.xlabel("Rzeczywisty ciężar")
plt.ylabel("Przewidywany ciężar")
save_plot("best_model_actual_vs_predicted.png")

plt.figure(figsize=(9, 5))
sns.histplot(best_predictions["absolute_error"], bins=60, kde=True)
plt.title(f"Rozkład błędów bezwzględnych - {best_model_name}")
plt.xlabel("Błąd bezwzględny")
plt.ylabel("Liczba obserwacji")
save_plot("best_model_absolute_error_distribution.png")


# ============================================================
# 11. Ewaluacja per grupa
# ============================================================

print_section("EWALUACJA PER GRUPA")

test_eval_df = test_df_sample.copy().reset_index(drop=True)
test_eval_df["predicted_weight"] = best_pipeline.predict(test_eval_df[FEATURES])
test_eval_df["predicted_weight"] = np.maximum(test_eval_df["predicted_weight"], 0)
test_eval_df["absolute_error"] = (
    test_eval_df[TARGET] - test_eval_df["predicted_weight"]
).abs()


def group_evaluation(data, group_col, min_records=100):
    """Liczy metryki modelu osobno dla wybranej grupy danych."""
    rows = []
    for group_value, group_df in data.groupby(group_col):
        if len(group_df) < min_records:
            continue
        metrics = calculate_regression_metrics(
            model_name=f"{best_model_name}_{group_col}_{group_value}",
            y_true=group_df[TARGET],
            y_pred=group_df["predicted_weight"],
        )
        metrics[group_col] = group_value
        metrics["records"] = len(group_df)
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("MAE") if rows else pd.DataFrame()


# Ewaluacja per grupa pokazuje, czy model działa podobnie dla różnych poziomów,
# faz, płci i ćwiczeń, a nie tylko dobrze wygląda w średniej globalnej.
for group_col in ["level", "phase", "sex", "exercise"]:
    group_results = group_evaluation(
        test_eval_df,
        group_col=group_col,
        min_records=100,
    )
    if len(group_results) > 0:
        display_table(group_results.head(30), f"Wyniki modelu według grupy: {group_col}")
        group_results.to_csv(
            os.path.join(OUTPUT_DIR, f"group_evaluation_by_{group_col}.csv"),
            index=False,
        )


# ============================================================
# 12. Interpretacja modelu
# ============================================================

print_section("INTERPRETACJA MODELU")


def get_feature_names_from_preprocessor(preprocessor):
    """Odtwarza nazwy cech po transformacji OneHotEncoder i passthrough."""
    feature_names = []
    cat_transformer = preprocessor.named_transformers_["cat"]
    cat_names = cat_transformer.get_feature_names_out(categorical_features)
    feature_names.extend(cat_names)
    feature_names.extend(numeric_features)
    return np.array(feature_names)


try:
    preprocessor = best_pipeline.named_steps["preprocessor"]
    model = best_pipeline.named_steps["model"]
    feature_names = get_feature_names_from_preprocessor(preprocessor)

    if hasattr(model, "feature_importances_"):
        importance_df = (
            pd.DataFrame(
                {"feature": feature_names, "importance": model.feature_importances_}
            )
            .sort_values("importance", ascending=False)
        )
        display_table(importance_df.head(25), "Najważniejsze cechy modelu")
        importance_df.to_csv(
            os.path.join(OUTPUT_DIR, "feature_importance.csv"),
            index=False,
        )

        plt.figure(figsize=(10, 7))
        sns.barplot(data=importance_df.head(20), x="importance", y="feature")
        plt.title("Top 20 cech według ważności")
        plt.xlabel("Ważność")
        plt.ylabel("Cecha")
        save_plot("feature_importance_top20.png")

    elif hasattr(model, "coef_"):
        coef_df = pd.DataFrame({"feature": feature_names, "coefficient": model.coef_})
        coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
        coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
        display_table(coef_df.head(25), "Najważniejsze współczynniki modelu")
        coef_df.to_csv(
            os.path.join(OUTPUT_DIR, "model_coefficients.csv"),
            index=False,
        )
    else:
        print("Wybrany model nie udostępnia prostych ważności cech.")
except Exception as e:
    print("Nie udało się wyliczyć interpretacji modelu.")
    print("Powód:", e)


# ============================================================
# 13. Moduł rekomendacyjny
# ============================================================

print_section("MODUŁ REKOMENDACYJNY")

# Rekomender nie bazuje wyłącznie na ML. Łączy podobnych użytkowników, statystyki
# historyczne, predykcję ciężaru i reguły bezpieczeństwa.
exercise_category_map = {
    "Bench Press": "push",
    "Incline Bench Press": "push",
    "Shoulder Press": "push",
    "Triceps Pushdown": "push",
    "Lateral Raise": "push",
    "Lat Pulldown": "pull",
    "Seated Row": "pull",
    "Barbell Row": "pull",
    "Rear Delt Row": "pull",
    "Biceps Curl": "pull",
    "Squat": "legs",
    "Leg Press": "legs",
    "Deadlift": "legs",
    "Leg Curl": "legs",
    "Calf Raise": "legs",
}
df["exercise_category"] = df["exercise"].map(exercise_category_map).fillna("other")


def filter_similar_users(data, sex=None, level=None, phase=None, split=None):
    """Znajduje możliwie podobne rekordy, stopniowo luzując kryteria."""
    filter_sets = [
        {"sex": sex, "level": level, "phase": phase, "split": split},
        {"level": level, "phase": phase, "split": split},
        {"level": level, "phase": phase},
        {"level": level},
        {},
    ]

    for filters in filter_sets:
        temp = data.copy()
        for col, value in filters.items():
            if value is not None and col in temp.columns:
                temp = temp[temp[col] == value]
        if len(temp) > 0:
            return temp, filters
    return data.copy(), {}


def choose_recommended_split(data, sex, level, phase, days_per_week):
    """Dobiera split z uwzględnieniem liczby dni treningowych i podobnych rekordów."""
    similar, filters_used = filter_similar_users(
        data=data,
        sex=sex,
        level=level,
        phase=phase,
        split=None,
    )
    available_splits = set(data["split"].dropna().unique())

    if days_per_week <= 2:
        preferred_split = "fbw"
    elif days_per_week == 3:
        preferred_split = "ppl"
    elif days_per_week == 4:
        preferred_split = "upper_lower"
    else:
        preferred_split = "ppl"

    if preferred_split in available_splits:
        recommended_split = preferred_split
        reason = (
            f"Wybrano split {preferred_split}, ponieważ pasuje do liczby dni "
            f"treningowych: {days_per_week}."
        )
    else:
        recommended_split = safe_mode(
            similar["split"],
            default_value=data["split"].mode().iloc[0],
        )
        reason = (
            "Wybrano najczęstszy split wśród podobnych użytkowników: "
            f"{recommended_split}."
        )
    return recommended_split, reason, filters_used


def select_exercises_for_plan(data, split, max_exercises=6):
    """Wybiera ćwiczenia do planu na podstawie splitu i popularności w danych."""
    selected = []

    def top_exercises(category, n):
        temp = data[data["exercise_category"] == category]
        return temp["exercise"].value_counts().head(n).index.tolist()

    if split == "ppl":
        selected.extend(top_exercises("push", 2))
        selected.extend(top_exercises("pull", 2))
        selected.extend(top_exercises("legs", 2))
    elif split == "upper_lower":
        upper = data[data["exercise_category"].isin(["push", "pull"])]
        lower = data[data["exercise_category"] == "legs"]
        selected.extend(upper["exercise"].value_counts().head(3).index.tolist())
        selected.extend(lower["exercise"].value_counts().head(3).index.tolist())
    else:
        selected.extend(top_exercises("push", 2))
        selected.extend(top_exercises("pull", 2))
        selected.extend(top_exercises("legs", 2))

    selected = list(dict.fromkeys(selected))
    if len(selected) < max_exercises:
        for ex in data["exercise"].value_counts().index.tolist():
            if ex not in selected:
                selected.append(ex)
            if len(selected) >= max_exercises:
                break
    return selected[:max_exercises]


def get_exercise_training_parameters(data, exercise, phase):
    """Wyznacza bazowe serie, powtórzenia, RIR i fatigue dla ćwiczenia."""
    ex_data = data[data["exercise"] == exercise].copy()
    if len(ex_data) == 0:
        return {"sets": 3, "reps": 8, "target_rir": 2, "target_fatigue": 6}

    sets_per_session = ex_data.groupby(["session_id", "exercise"])["set_number"].max()
    recommended_sets = int(round(safe_median(sets_per_session, default_value=3)))
    recommended_reps = int(round(safe_median(ex_data["reps"], default_value=8)))

    if phase == "deload":
        target_rir = 4
        target_fatigue = 3
    elif phase == "strength":
        target_rir = 2
        target_fatigue = 7
    else:
        target_rir = 2
        target_fatigue = 6

    recommended_sets = max(2, min(recommended_sets, 5))
    recommended_reps = max(3, min(recommended_reps, 15))

    return {
        "sets": recommended_sets,
        "reps": recommended_reps,
        "target_rir": target_rir,
        "target_fatigue": target_fatigue,
    }


def get_history_features_for_user_exercise(data, user_id, exercise, fallback_data):
    """Pobiera historię użytkownika dla ćwiczenia albo sensowny fallback z grupy."""
    if user_id is not None and user_id in data["user_id"].unique():
        hist = (
            data[(data["user_id"] == user_id) & (data["exercise"] == exercise)]
            .sort_values(["date", "session_id", "set_number"])
        )
        if len(hist) > 0:
            last = hist.iloc[-1]
            recent = hist.tail(3)
            return {
                "prev_weight": float(last["weight"]),
                "prev_reps": float(last["reps"]),
                "prev_rir": float(last["rir"]),
                "prev_fatigue": float(last["fatigue"]),
                "prev_volume": float(last["volume"]),
                "rolling_weight_3": float(recent["weight"].mean()),
                "rolling_reps_3": float(recent["reps"].mean()),
                "rolling_rir_3": float(recent["rir"].mean()),
                "rolling_fatigue_3": float(recent["fatigue"].mean()),
                "rolling_volume_3": float(recent["volume"].mean()),
                "history_available": True,
            }

    ex_fallback = fallback_data[fallback_data["exercise"] == exercise]
    if len(ex_fallback) == 0:
        ex_fallback = fallback_data.copy()

    median_weight = safe_median(ex_fallback["weight"], default_value=40)
    median_reps = safe_median(ex_fallback["reps"], default_value=8)
    median_rir = safe_median(ex_fallback["rir"], default_value=2)
    median_fatigue = safe_median(ex_fallback["fatigue"], default_value=6)
    median_volume = safe_median(
        ex_fallback["volume"],
        default_value=median_weight * median_reps,
    )

    return {
        "prev_weight": median_weight,
        "prev_reps": median_reps,
        "prev_rir": median_rir,
        "prev_fatigue": median_fatigue,
        "prev_volume": median_volume,
        "rolling_weight_3": median_weight,
        "rolling_reps_3": median_reps,
        "rolling_rir_3": median_rir,
        "rolling_fatigue_3": median_fatigue,
        "rolling_volume_3": median_volume,
        "history_available": False,
    }


# ============================================================
# 14. Reguły bezpieczeństwa
# ============================================================

print_section("REGUŁY BEZPIECZEŃSTWA")


def apply_safety_rules(predicted_weight, prev_weight, recent_rir, recent_fatigue, phase, level, age=None):
    """Ogranicza rekomendowany ciężar do bezpiecznego zakresu progresji."""
    predicted_weight = max(float(predicted_weight), 0)

    if pd.isna(prev_weight) or prev_weight <= 0:
        return round_to_nearest(predicted_weight, step=2.5)

    if level == "beginner":
        max_increase = 0.03
    elif level == "intermediate":
        max_increase = 0.04
    else:
        max_increase = 0.05

    # Wiek nie jest cechą modelu, bo nie występuje w historii treningowej.
    # Może natomiast działać jako zewnętrzna reguła bezpieczeństwa z profilu.
    if age is not None:
        if age >= 60:
            max_increase = min(max_increase, 0.02)
        elif age >= 45:
            max_increase = min(max_increase, 0.03)

    if phase == "deload":
        safe_weight = min(predicted_weight, prev_weight * 0.85)
        return round_to_nearest(safe_weight, step=2.5)

    # Reguły bezpieczeństwa ograniczają ciężar, gdy ostatnie serie były bardzo
    # trudne albo użytkownik ma wysokie zmęczenie.
    if recent_fatigue >= 8 or recent_rir <= 1:
        safe_weight = min(predicted_weight, prev_weight)
    else:
        upper_cap = prev_weight * (1 + max_increase)
        safe_weight = min(predicted_weight, upper_cap)

    lower_cap = prev_weight * 0.9
    safe_weight = max(safe_weight, lower_cap)
    return round_to_nearest(safe_weight, step=2.5)


def assign_training_day(split, exercise_category):
    """Przypisuje ćwiczenie do dnia treningowego zgodnie ze splitem."""
    if split == "ppl":
        if exercise_category == "push":
            return "Push"
        if exercise_category == "pull":
            return "Pull"
        if exercise_category == "legs":
            return "Legs"
        return "Accessory"

    if split == "upper_lower":
        if exercise_category in ["push", "pull"]:
            return "Upper"
        if exercise_category == "legs":
            return "Lower"
        return "Accessory"

    return "Full Body"


def generate_training_plan(profile, model, data, user_id=None, max_exercises=6):
    """Generuje przykładowy plan z predykcją ciężaru i korektą bezpieczeństwa."""
    age = profile.get("age", None)
    sex = profile.get("sex", safe_mode(data["sex"]))
    level = profile.get("level", safe_mode(data["level"]))
    phase = profile.get("phase", "hypertrophy")
    days_per_week = profile.get("days_per_week", 3)

    recommended_split, split_reason, split_filters = choose_recommended_split(
        data=data,
        sex=sex,
        level=level,
        phase=phase,
        days_per_week=days_per_week,
    )

    similar_data, filters_used = filter_similar_users(
        data=data,
        sex=sex,
        level=level,
        phase=phase,
        split=recommended_split,
    )

    selected_exercises = select_exercises_for_plan(
        data=similar_data,
        split=recommended_split,
        max_exercises=max_exercises,
    )

    plan_rows = []
    for exercise in selected_exercises:
        params = get_exercise_training_parameters(similar_data, exercise, phase)
        history_features = get_history_features_for_user_exercise(
            data,
            user_id,
            exercise,
            similar_data,
        )
        exercise_category = exercise_category_map.get(exercise, "other")
        training_day = assign_training_day(recommended_split, exercise_category)

        model_input = pd.DataFrame(
            [
                {
                    "exercise": exercise,
                    "level": level,
                    "split": recommended_split,
                    "phase": phase,
                    "sex": sex,
                    "set_number": 1,
                    "reps": params["reps"],
                    "fatigue": params["target_fatigue"],
                    "rir": params["target_rir"],
                    "prev_weight": history_features["prev_weight"],
                    "prev_reps": history_features["prev_reps"],
                    "prev_rir": history_features["prev_rir"],
                    "prev_fatigue": history_features["prev_fatigue"],
                    "prev_volume": history_features["prev_volume"],
                    "rolling_weight_3": history_features["rolling_weight_3"],
                    "rolling_reps_3": history_features["rolling_reps_3"],
                    "rolling_rir_3": history_features["rolling_rir_3"],
                    "rolling_fatigue_3": history_features["rolling_fatigue_3"],
                    "rolling_volume_3": history_features["rolling_volume_3"],
                }
            ]
        )

        predicted_weight = float(model.predict(model_input)[0])
        final_weight = apply_safety_rules(
            predicted_weight=predicted_weight,
            prev_weight=history_features["prev_weight"],
            recent_rir=history_features["rolling_rir_3"],
            recent_fatigue=history_features["rolling_fatigue_3"],
            phase=phase,
            level=level,
            age=age,
        )

        plan_rows.append(
            {
                "day": training_day,
                "split": recommended_split,
                "exercise": exercise,
                "sets": params["sets"],
                "reps": params["reps"],
                "target_rir": params["target_rir"],
                "target_fatigue": params["target_fatigue"],
                "model_predicted_weight": round(predicted_weight, 2),
                "final_recommended_weight": final_weight,
                "history_available": history_features["history_available"],
                "recommendation_reason": (
                    "Ciężar wyznaczony przez model regresyjny i skorygowany "
                    "regułami bezpieczeństwa."
                ),
            }
        )

    plan_df = pd.DataFrame(plan_rows)
    metadata = {
        "profile": profile,
        "recommended_split": recommended_split,
        "split_reason": split_reason,
        "filters_used_for_similar_users": filters_used,
        "split_filters_used": split_filters,
        "user_id_used": user_id,
    }
    return plan_df, metadata


# ============================================================
# 15. Generowanie przykładowych planów
# ============================================================

print_section("GENEROWANIE PRZYKŁADOWYCH PLANÓW")

available_sexes = df["sex"].dropna().unique().tolist()
female_value = "female" if "female" in available_sexes else df["sex"].mode().iloc[0]
male_value = "male" if "male" in available_sexes else df["sex"].mode().iloc[0]

example_profiles = [
    {
        "name": "beginner_female_hypertrophy",
        "age": 24,
        "sex": female_value,
        "level": "beginner",
        "phase": "hypertrophy",
        "days_per_week": 3,
    },
    {
        "name": "intermediate_male_strength",
        "age": 32,
        "sex": male_value,
        "level": "intermediate",
        "phase": "strength",
        "days_per_week": 4,
    },
    {
        "name": "advanced_user_deload",
        "age": 41,
        "sex": male_value,
        "level": "advanced",
        "phase": "deload",
        "days_per_week": 3,
    },
    {
        "name": "older_beginner_hypertrophy",
        "age": 62,
        "sex": male_value,
        "level": "beginner",
        "phase": "hypertrophy",
        "days_per_week": 2,
    },
]

generated_plans = {}

for profile in example_profiles:
    profile_name = profile["name"]
    plan_df, metadata = generate_training_plan(
        profile=profile,
        model=best_pipeline,
        data=df,
        user_id=None,
        max_exercises=6,
    )
    generated_plans[profile_name] = {"plan": plan_df, "metadata": metadata}
    display_table(plan_df, f"Plan treningowy: {profile_name}")
    plan_df.to_csv(
        os.path.join(OUTPUT_DIR, f"recommended_plan_{profile_name}.csv"),
        index=False,
    )
    print("Metadata:")
    print(metadata)


print_section("REKOMENDACJA DLA ISTNIEJĄCEGO UŻYTKOWNIKA")

example_existing_user = df["user_id"].value_counts().index[0]
user_profile_from_data = {
    "name": "existing_user_recommendation",
    "age": 30,
    "sex": safe_mode(df[df["user_id"] == example_existing_user]["sex"]),
    "level": safe_mode(df[df["user_id"] == example_existing_user]["level"]),
    "phase": "hypertrophy",
    "days_per_week": 3,
}

existing_user_plan, existing_user_metadata = generate_training_plan(
    profile=user_profile_from_data,
    model=best_pipeline,
    data=df,
    user_id=example_existing_user,
    max_exercises=6,
)

display_table(
    existing_user_plan,
    f"Plan dla istniejącego użytkownika: {example_existing_user}",
)
existing_user_plan.to_csv(
    os.path.join(OUTPUT_DIR, "recommended_plan_existing_user.csv"),
    index=False,
)
print("Metadata:")
print(existing_user_metadata)


# ============================================================
# 16. Podsumowanie Etapu 2
# ============================================================

print_section("PODSUMOWANIE ETAPU 2")

best_ml_row = results_df[results_df["model"] == best_model_name].iloc[0]
naive_row = results_df[results_df["model"] == "naive_prev_weight"].iloc[0]
mae_improvement = naive_row["MAE"] - best_ml_row["MAE"]
mae_improvement_percent = (
    mae_improvement / naive_row["MAE"] * 100 if naive_row["MAE"] != 0 else np.nan
)

summary_text = f"""
ETAP 2 - PODSUMOWANIE

W tym etapie zbudowano właściwy komponent ML/AI systemu treningowego.

Porównane modele:
- Ridge Regression,
- Random Forest,
- HistGradientBoosting,
- baseline naiwny prev_weight jako punkt odniesienia.

Najlepszy model ML według MAE:
- {best_model_name}

Wyniki najlepszego modelu ML:
- MAE: {best_ml_row['MAE']:.4f}
- RMSE: {best_ml_row['RMSE']:.4f}
- R2: {best_ml_row['R2']:.4f}
- Predykcje w granicy 2.5 kg: {best_ml_row['within_2_5kg_percent']:.2f}%
- Predykcje w granicy 5 kg: {best_ml_row['within_5kg_percent']:.2f}%
- Predykcje w granicy 10 kg: {best_ml_row['within_10kg_percent']:.2f}%

Porównanie z naiwnym baseline'em:
- MAE baseline'u naiwnego: {naive_row['MAE']:.4f}
- Poprawa MAE względem baseline'u: {mae_improvement:.4f}
- Poprawa procentowa: {mae_improvement_percent:.2f}%

System rekomendacyjny:
Rekomender nie bazuje wyłącznie na modelu ML. Łączy model predykcyjny, dane
podobnych użytkowników oraz reguły bezpieczeństwa, dzięki czemu jest bardziej
praktyczny niż sama regresja ciężaru.

Ograniczenia:
1. Wiek nie występuje w danych historycznych, dlatego jest używany wyłącznie jako
   reguła bezpieczeństwa pochodząca z profilu użytkownika.
2. Mapa kategorii ćwiczeń jest ręczną regułą ekspercką i w przyszłości może
   zostać zastąpiona automatyczną klasyfikacją ćwiczeń.
3. Cechy reps, rir i fatigue są używane jako planowane parametry serii. Model
   odpowiada na pytanie: jaki ciężar dobrać dla zaplanowanej liczby powtórzeń i
   docelowej trudności.
4. System jest prototypem wspierającym decyzję. Nie jest narzędziem medycznym,
   fizjoterapeutycznym ani produkcyjnym systemem trenerskim.
"""

print(summary_text)

with open(os.path.join(OUTPUT_DIR, "stage2_summary.txt"), "w", encoding="utf-8") as f:
    f.write(summary_text)

print(f"\nWyniki zapisano w folderze: {OUTPUT_DIR}")
