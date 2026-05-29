"""Stage 2 of the project: ML modeling and training recommendations.

This script builds the ML/AI layer after the exploratory analysis from Stage 1.
It reads the canonical dataset from `data/FINAL_ENGINE_V4.csv`, writes working
outputs to `outputs/stage2_outputs`, and stores the final weight prediction
model in `models/best_weight_prediction_model.joblib`.

The workflow prepares historical features, compares regression models, evaluates
the best model across groups, adds a lightweight interpretation step, and builds
a prototype hybrid recommender. The recommender supports training decisions; it
is not a medical tool or a production coaching system.
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
# 1. Stage 2 configuration
# ============================================================

DATA_PATH = "data/FINAL_ENGINE_V4.csv"
OUTPUT_DIR = "outputs/stage2_outputs"
MODEL_FILENAME = "models/best_weight_prediction_model.joblib"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_FILENAME), exist_ok=True)

RANDOM_STATE = 42
MAX_TRAIN_SAMPLE = 250_000
MAX_TEST_SAMPLE = 100_000

# Default mode trains the models. Set this to False only when a saved joblib
# model already exists and should be reused.
RUN_MODEL_TRAINING = True


def print_section(title):
    """Print a readable separator for the next script section."""
    print("\n" + "#" * 110)
    print(title)
    print("#" * 110)


def display_table(obj, title=None):
    """Show a table in a notebook, or print it in the console."""
    if title:
        print("\n" + "=" * 110)
        print(title)
        print("=" * 110)
    try:
        display(obj)
    except NameError:
        print(obj)


def save_plot(filename):
    """Save the current plot in the Stage 2 output directory."""
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def make_one_hot_encoder():
    """Create a OneHotEncoder compatible with older and newer sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def round_to_nearest(value, step=2.5):
    """Round a weight to the nearest practical step, for example 2.5 kg."""
    if pd.isna(value):
        return np.nan
    return round(float(value) / step) * step


def safe_median(series, default_value=0):
    """Return the median, or a default value when the series is empty."""
    series = pd.Series(series).dropna()
    if len(series) == 0:
        return default_value
    return float(series.median())


def safe_mode(series, default_value=None):
    """Return the most common value, or a default value when needed."""
    series = pd.Series(series).dropna()
    if len(series) == 0:
        return default_value
    return series.mode().iloc[0]


# ============================================================
# 2. Data loading and validation
# ============================================================

print_section("DATA LOADING AND VALIDATION")

df = pd.read_csv(DATA_PATH)

print(f"Rows: {len(df):,}")
print(f"Columns: {df.shape[1]:,}")
display_table(df.head(), "First rows")

required_columns = [
    "user_id", "session_id", "date",
    "exercise", "set_number", "reps", "weight",
    "fatigue", "rir", "level", "split", "phase", "sex",
]

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
if df["date"].isna().sum() > 0:
    print("WARNING: Some dates could not be parsed.")

print("Date range:")
print("From:", df["date"].min())
print("To:", df["date"].max())

missing_table = df.isna().sum().to_frame("missing_count")
display_table(missing_table, "Missing values")
print("\nDuplicate rows:", df.duplicated().sum())

if "age" not in df.columns:
    print("\nWARNING:")
    print("The dataset does not contain an age column.")
    print(
        "Age is used only as an external safety-rule input from the user profile, "
        "not as a trained model feature."
    )


# ============================================================
# 3. Feature engineering for ML
# ============================================================

print_section("FEATURE ENGINEERING FOR ML")

# Volume is a simple set-level workload measure used later in history features.
df["volume"] = df["reps"] * df["weight"]
# Epley e1RM gives a rough strength estimate for quick inspection.
df["e1rm_epley"] = df["weight"] * (1 + df["reps"] / 30)

df = (
    df
    .sort_values(["user_id", "exercise", "date", "session_id", "set_number"])
    .reset_index(drop=True)
)

# Historical features describe what the user recently did for the same exercise.
df["prev_weight"] = df.groupby(["user_id", "exercise"])["weight"].shift(1)
df["prev_reps"] = df.groupby(["user_id", "exercise"])["reps"].shift(1)
df["prev_rir"] = df.groupby(["user_id", "exercise"])["rir"].shift(1)
df["prev_fatigue"] = df.groupby(["user_id", "exercise"])["fatigue"].shift(1)
df["prev_volume"] = df.groupby(["user_id", "exercise"])["volume"].shift(1)

for col in ["weight", "reps", "rir", "fatigue", "volume"]:
    # shift(1) keeps the current set out of the rolling average.
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
    "Example historical features",
)


# ============================================================
# 4. Model dataset preparation
# ============================================================

print_section("MODEL DATASET PREPARATION")

# reps, rir, and fatigue are treated as planned set parameters. The model answers:
# which weight fits the planned reps and target difficulty?
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

print(f"Model-ready rows: {len(model_ready):,}")
display_table(model_ready[FEATURES + [TARGET]].head(), "Model dataset")

model_ready.to_csv(
    os.path.join(OUTPUT_DIR, "model_ready_stage2.csv"),
    index=False,
)


# ============================================================
# 5. Time-based train/test split
# ============================================================

print_section("TIME-BASED TRAIN/TEST SPLIT")

# Time-based split is closer to real use: train on earlier workouts, predict later ones.
cutoff_date = model_ready["date"].quantile(0.8)
train_df = model_ready[model_ready["date"] <= cutoff_date].copy()
test_df = model_ready[model_ready["date"] > cutoff_date].copy()

print("Cutoff date:", cutoff_date)
print(f"Train: {len(train_df):,}")
print(f"Test: {len(test_df):,}")

if len(train_df) == 0 or len(test_df) == 0:
    raise ValueError(
        "After the time-based train/test split, one split is empty. "
        "Check the date range in the dataset."
    )

train_df_sample = train_df.sample(
    min(len(train_df), MAX_TRAIN_SAMPLE),
    random_state=RANDOM_STATE,
)
test_df_sample = test_df.sample(
    min(len(test_df), MAX_TEST_SAMPLE),
    random_state=RANDOM_STATE,
)

# Sampling keeps the comparison fast on larger synthetic datasets.
print(f"Train after sampling: {len(train_df_sample):,}")
print(f"Test after sampling: {len(test_df_sample):,}")

X_train = train_df_sample[FEATURES]
y_train = train_df_sample[TARGET]
X_test = test_df_sample[FEATURES]
y_test = test_df_sample[TARGET]


# ============================================================
# 6. Preprocessing pipeline
# ============================================================

print_section("PREPROCESSING PIPELINE")


def build_preprocessor(scale_numeric=False):
    """Build preprocessing for categorical and numeric features."""
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("cat", make_one_hot_encoder(), categorical_features),
            ("num", numeric_transformer, numeric_features),
        ]
    )


# ============================================================
# 7. Model definitions
# ============================================================

print_section("MODEL DEFINITIONS")

# Compare a linear model, bagging model, and gradient boosting model.
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

print("Models to compare:")
for name in models.keys():
    print("-", name)


# ============================================================
# 8. Metrics and evaluation helpers
# ============================================================

print_section("METRICS AND EVALUATION HELPERS")


def calculate_regression_metrics(model_name, y_true, y_pred):
    """Calculate regression metrics for weight prediction."""
    y_pred = np.maximum(np.asarray(y_pred), 0)
    y_true = np.asarray(y_true)
    abs_error = np.abs(y_true - y_pred)

    # MAE is the main metric because it is easy to read in kilograms.
    # The within_* thresholds show practical prediction quality.
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
    """Run model predictions and return metrics with prediction rows."""
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
    """Evaluate the baseline that reuses the previous weight as prediction."""
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
# 9. Model training and comparison
# ============================================================

print_section("MODEL TRAINING AND COMPARISON")

all_results = []
all_predictions = {}
trained_models = {}

# prev_weight is a simple reference point for checking whether ML adds value.
naive_results, naive_predictions = evaluate_naive_baseline(test_df_sample)
all_results.append(naive_results)
all_predictions["naive_prev_weight"] = naive_predictions
print("Naive baseline:", naive_results)

if RUN_MODEL_TRAINING:
    for model_name, pipeline in models.items():
        print(f"\nTraining model: {model_name}")
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
    display_table(results_df, "Regression model comparison with naive baseline")
    results_df.to_csv(
        os.path.join(OUTPUT_DIR, "model_comparison_results.csv"),
        index=False,
    )

    # The best model is selected only from ML models, then compared with prev_weight.
    ml_results_df = results_df[results_df["model"] != "naive_prev_weight"].copy()
    best_model_name = ml_results_df.iloc[0]["model"]
    best_pipeline = trained_models[best_model_name]

    print("\nBest ML model by MAE:", best_model_name)
    joblib.dump(best_pipeline, MODEL_FILENAME)
    print("Model saved to:", MODEL_FILENAME)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=results_df, x="model", y="MAE")
    plt.title("Model comparison - MAE")
    plt.xlabel("Model")
    plt.ylabel("MAE")
    plt.xticks(rotation=25)
    save_plot("model_comparison_mae.png")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=results_df, x="model", y="within_5kg_percent")
    plt.title("Predictions within 5 kg error")
    plt.xlabel("Model")
    plt.ylabel("% predictions within 5 kg")
    plt.xticks(rotation=25)
    save_plot("model_comparison_within_5kg.png")
else:
    if not os.path.exists(MODEL_FILENAME):
        raise FileNotFoundError(
            f"RUN_MODEL_TRAINING=False requires an existing model: {MODEL_FILENAME}"
        )
    best_pipeline = joblib.load(MODEL_FILENAME)
    best_model_name = "loaded_model"
    print("Loaded saved model:", MODEL_FILENAME)

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
# 10. Best model analysis
# ============================================================

print_section("BEST MODEL ANALYSIS")

best_predictions = all_predictions[best_model_name].copy()

display_table(best_predictions.head(20), "Sample predictions from the best model")
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
plt.title(f"Actual weight vs prediction - {best_model_name}")
plt.xlabel("Actual weight")
plt.ylabel("Predicted weight")
save_plot("best_model_actual_vs_predicted.png")

plt.figure(figsize=(9, 5))
sns.histplot(best_predictions["absolute_error"], bins=60, kde=True)
plt.title(f"Absolute error distribution - {best_model_name}")
plt.xlabel("Absolute error")
plt.ylabel("Number of observations")
save_plot("best_model_absolute_error_distribution.png")


# ============================================================
# 11. Group-level evaluation
# ============================================================

print_section("GROUP-LEVEL EVALUATION")

test_eval_df = test_df_sample.copy().reset_index(drop=True)
test_eval_df["predicted_weight"] = best_pipeline.predict(test_eval_df[FEATURES])
test_eval_df["predicted_weight"] = np.maximum(test_eval_df["predicted_weight"], 0)
test_eval_df["absolute_error"] = (
    test_eval_df[TARGET] - test_eval_df["predicted_weight"]
).abs()


def group_evaluation(data, group_col, min_records=100):
    """Calculate model metrics separately for one grouping column."""
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


# Group-level metrics show whether the model is consistent across key segments.
for group_col in ["level", "phase", "sex", "exercise"]:
    group_results = group_evaluation(
        test_eval_df,
        group_col=group_col,
        min_records=100,
    )
    if len(group_results) > 0:
        display_table(group_results.head(30), f"Model results by group: {group_col}")
        group_results.to_csv(
            os.path.join(OUTPUT_DIR, f"group_evaluation_by_{group_col}.csv"),
            index=False,
        )


# ============================================================
# 12. Model interpretation
# ============================================================

print_section("MODEL INTERPRETATION")


def get_feature_names_from_preprocessor(preprocessor):
    """Recover feature names after one-hot encoding and passthrough columns."""
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
        display_table(importance_df.head(25), "Most important model features")
        importance_df.to_csv(
            os.path.join(OUTPUT_DIR, "feature_importance.csv"),
            index=False,
        )

        plt.figure(figsize=(10, 7))
        sns.barplot(data=importance_df.head(20), x="importance", y="feature")
        plt.title("Top 20 features by importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        save_plot("feature_importance_top20.png")

    elif hasattr(model, "coef_"):
        coef_df = pd.DataFrame({"feature": feature_names, "coefficient": model.coef_})
        coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
        coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
        display_table(coef_df.head(25), "Largest model coefficients")
        coef_df.to_csv(
            os.path.join(OUTPUT_DIR, "model_coefficients.csv"),
            index=False,
        )
    else:
        print("The selected model does not expose simple feature importance.")
except Exception as e:
    print("Could not calculate model interpretation.")
    print("Reason:", e)


# ============================================================
# 13. Recommendation module
# ============================================================

print_section("RECOMMENDATION MODULE")

# The recommender combines ML output with similar users and safety rules.
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
    """Find similar records, relaxing filters when needed."""
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
    """Choose a split based on training days and similar records."""
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
            f"Selected split {preferred_split} because it fits the planned "
            f"training days per week: {days_per_week}."
        )
    else:
        recommended_split = safe_mode(
            similar["split"],
            default_value=data["split"].mode().iloc[0],
        )
        reason = (
            "Selected the most common split among similar users: "
            f"{recommended_split}."
        )
    return recommended_split, reason, filters_used


def select_exercises_for_plan(data, split, max_exercises=6):
    """Select exercises for the plan using split logic and dataset popularity."""
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
    """Set baseline sets, reps, RIR, and fatigue for one exercise."""
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
    """Use user exercise history, or fall back to similar records."""
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
# 14. Safety rules
# ============================================================

print_section("SAFETY RULES")


def apply_safety_rules(predicted_weight, prev_weight, recent_rir, recent_fatigue, phase, level, age=None):
    """Limit recommended weight to a practical progression range."""
    predicted_weight = max(float(predicted_weight), 0)

    if pd.isna(prev_weight) or prev_weight <= 0:
        return round_to_nearest(predicted_weight, step=2.5)

    if level == "beginner":
        max_increase = 0.03
    elif level == "intermediate":
        max_increase = 0.04
    else:
        max_increase = 0.05

    # Age is not in training history, so it is used only as an external safety rule.
    if age is not None:
        if age >= 60:
            max_increase = min(max_increase, 0.02)
        elif age >= 45:
            max_increase = min(max_increase, 0.03)

    if phase == "deload":
        safe_weight = min(predicted_weight, prev_weight * 0.85)
        return round_to_nearest(safe_weight, step=2.5)

    # Hard recent sets or high fatigue should block aggressive jumps.
    if recent_fatigue >= 8 or recent_rir <= 1:
        safe_weight = min(predicted_weight, prev_weight)
    else:
        upper_cap = prev_weight * (1 + max_increase)
        safe_weight = min(predicted_weight, upper_cap)

    lower_cap = prev_weight * 0.9
    safe_weight = max(safe_weight, lower_cap)
    return round_to_nearest(safe_weight, step=2.5)


def assign_training_day(split, exercise_category):
    """Assign an exercise to a training day based on the selected split."""
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
    """Generate a sample plan with predicted and safety-adjusted weights."""
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
                    "Weight predicted by the regression model and adjusted "
                    "with safety rules."
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
# 15. Sample plan generation
# ============================================================

print_section("SAMPLE PLAN GENERATION")

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
    display_table(plan_df, f"Training plan: {profile_name}")
    plan_df.to_csv(
        os.path.join(OUTPUT_DIR, f"recommended_plan_{profile_name}.csv"),
        index=False,
    )
    print("Metadata:")
    print(metadata)


print_section("RECOMMENDATION FOR AN EXISTING USER")

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
    f"Plan for existing user: {example_existing_user}",
)
existing_user_plan.to_csv(
    os.path.join(OUTPUT_DIR, "recommended_plan_existing_user.csv"),
    index=False,
)
print("Metadata:")
print(existing_user_metadata)


# ============================================================
# 16. Stage 2 summary
# ============================================================

print_section("STAGE 2 SUMMARY")

best_ml_row = results_df[results_df["model"] == best_model_name].iloc[0]
naive_row = results_df[results_df["model"] == "naive_prev_weight"].iloc[0]
mae_improvement = naive_row["MAE"] - best_ml_row["MAE"]
mae_improvement_percent = (
    mae_improvement / naive_row["MAE"] * 100 if naive_row["MAE"] != 0 else np.nan
)

summary_text = f"""
STAGE 2 SUMMARY

This stage builds the main ML/AI component of the training system.

Compared models:
- Ridge Regression,
- Random Forest,
- HistGradientBoosting,
- naive prev_weight baseline as a reference point.

Best ML model by MAE:
- {best_model_name}

Best ML model results:
- MAE: {best_ml_row['MAE']:.4f}
- RMSE: {best_ml_row['RMSE']:.4f}
- R2: {best_ml_row['R2']:.4f}
- Predictions within 2.5 kg: {best_ml_row['within_2_5kg_percent']:.2f}%
- Predictions within 5 kg: {best_ml_row['within_5kg_percent']:.2f}%
- Predictions within 10 kg: {best_ml_row['within_10kg_percent']:.2f}%

Comparison with the naive baseline:
- Naive baseline MAE: {naive_row['MAE']:.4f}
- MAE improvement over baseline: {mae_improvement:.4f}
- Percentage improvement: {mae_improvement_percent:.2f}%

Recommendation system:
The recommender does not rely only on the ML model. It combines weight
prediction, similar-user data, and safety rules, which makes it more practical
than pure weight regression.

Limitations:
1. Age is not available in historical training data, so it is used only as an
   external safety rule from the user profile.
2. The exercise category map is a manual expert rule and can later be replaced
   with automatic exercise classification.
3. reps, rir, and fatigue are used as planned set parameters. The model answers:
   which weight fits the planned reps and target difficulty?
4. The system is a decision-support prototype. It is not a medical tool or a
   production coaching system.
"""

print(summary_text)

with open(os.path.join(OUTPUT_DIR, "stage2_summary.txt"), "w", encoding="utf-8") as f:
    f.write(summary_text)

print(f"\nOutputs saved in: {OUTPUT_DIR}")
