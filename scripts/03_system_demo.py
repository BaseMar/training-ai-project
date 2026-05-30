"""Stage 3 of the project: end-to-end AI system demo.

This script demonstrates the final project flow:

user profile -> load trained model -> prepare features -> recommend plan
-> apply safety rules -> save outputs.

Stage 3 does not train models. It loads the model prepared in Stage 2 from
`models/best_weight_prediction_model.joblib` and uses it inside a lightweight
hybrid recommendation demo. The recommendations are illustrative and require
expert validation before any real training use.
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


sns.set_theme(style="whitegrid")


# ============================================================
# 1. Stage 3 configuration
# ============================================================

DATA_PATH = "data/FINAL_ENGINE_V4.csv"
MODEL_PATH = "models/best_weight_prediction_model.joblib"
OUTPUT_DIR = "outputs/stage3_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = [
    "exercise", "level", "split", "phase", "sex",
    "set_number", "reps", "fatigue", "rir",
    "prev_weight", "prev_reps", "prev_rir", "prev_fatigue", "prev_volume",
    "rolling_weight_3", "rolling_reps_3", "rolling_rir_3",
    "rolling_fatigue_3", "rolling_volume_3",
]
TARGET = "weight"

CATEGORICAL_FEATURES = ["exercise", "level", "split", "phase", "sex"]
NUMERIC_FEATURES = [feature for feature in FEATURES if feature not in CATEGORICAL_FEATURES]

EXERCISE_CATEGORY_MAP = {
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


def print_section(title):
    """Print a readable separator for the next demo section."""
    print("\n" + "#" * 100)
    print(title)
    print("#" * 100)


def save_plot(filename):
    """Save the current plot in the Stage 3 output directory."""
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()


def round_to_nearest(value, step=2.5):
    """Round a weight to the nearest practical training increment."""
    return round(float(value) / step) * step


def safe_median(series, default_value=0):
    """Return the median, or a default when the series is empty."""
    clean = pd.Series(series).dropna()
    if len(clean) == 0:
        return default_value
    return float(clean.median())


def safe_mode(series, default_value=None):
    """Return the most common value, or a default when needed."""
    clean = pd.Series(series).dropna()
    if len(clean) == 0:
        return default_value
    return clean.mode().iloc[0]


def load_inputs(data_path, model_path):
    """Load the canonical dataset and the trained Stage 2 model."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Stage 3 is a demo and does not train models.\n"
            "First run: python scripts/02_modeling_and_recommendation.py\n"
            "Alternatively, download the model from GitHub Release if available.\n"
            f"Place the file at: {model_path}"
        )

    data = pd.read_csv(data_path)
    model = joblib.load(model_path)
    return data, model


def prepare_model_ready_data(data):
    """Rebuild the historical features expected by the Stage 2 model."""
    prepared = data.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared["volume"] = prepared["reps"] * prepared["weight"]
    prepared = (
        prepared
        .sort_values(["user_id", "exercise", "date", "session_id", "set_number"])
        .reset_index(drop=True)
    )

    prepared["prev_weight"] = prepared.groupby(["user_id", "exercise"])["weight"].shift(1)
    prepared["prev_reps"] = prepared.groupby(["user_id", "exercise"])["reps"].shift(1)
    prepared["prev_rir"] = prepared.groupby(["user_id", "exercise"])["rir"].shift(1)
    prepared["prev_fatigue"] = prepared.groupby(["user_id", "exercise"])["fatigue"].shift(1)
    prepared["prev_volume"] = prepared.groupby(["user_id", "exercise"])["volume"].shift(1)

    for column in ["weight", "reps", "rir", "fatigue", "volume"]:
        # shift(1) keeps the current set out of its own history.
        prepared[f"rolling_{column}_3"] = (
            prepared
            .groupby(["user_id", "exercise"])[column]
            .transform(lambda values: values.shift(1).rolling(3).mean())
        )

    return prepared.dropna(subset=FEATURES + [TARGET]).copy()


def add_exercise_categories(data):
    """Add helper columns used by the demo recommender."""
    categorized = data.copy()
    categorized["date"] = pd.to_datetime(categorized["date"], errors="coerce")
    categorized["volume"] = categorized["reps"] * categorized["weight"]
    categorized["exercise_category"] = (
        categorized["exercise"].map(EXERCISE_CATEGORY_MAP).fillna("other")
    )
    return categorized


def calculate_regression_metrics(y_true, y_pred):
    """Calculate compact regression metrics for the sanity check."""
    y_pred = np.maximum(np.asarray(y_pred), 0)
    y_true = np.asarray(y_true)
    abs_error = np.abs(y_true - y_pred)

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "within_2_5kg_percent": np.mean(abs_error <= 2.5) * 100,
        "within_5kg_percent": np.mean(abs_error <= 5.0) * 100,
        "within_10kg_percent": np.mean(abs_error <= 10.0) * 100,
    }


def run_model_sanity_check(model_ready, model):
    """Check whether the loaded model produces reasonable predictions."""
    x_test = model_ready[FEATURES]
    y_test = model_ready[TARGET]
    y_pred = np.maximum(model.predict(x_test), 0)
    abs_error = np.abs(y_test - y_pred)

    metrics = calculate_regression_metrics(y_test, y_pred)
    metrics_df = pd.DataFrame(list(metrics.items()), columns=["metric", "value"])
    metrics_df.to_csv(
        os.path.join(OUTPUT_DIR, "model_sanity_check_metrics.csv"),
        index=False,
    )
    print(metrics_df)

    plt.figure(figsize=(7, 7))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.25)
    plt.xlabel("Actual weight")
    plt.ylabel("Predicted weight")
    plt.title("Model sanity check: actual vs predicted weight")
    save_plot("scatter_actual_vs_predicted.png")

    plt.figure(figsize=(9, 5))
    sns.histplot(abs_error, bins=60, kde=True)
    plt.xlabel("Absolute error")
    plt.ylabel("Number of observations")
    plt.title("Model sanity check: prediction error distribution")
    save_plot("error_distribution.png")

    return metrics_df


def get_feature_names(preprocessor):
    """Recover feature names after preprocessing."""
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(
        CATEGORICAL_FEATURES
    )
    return list(cat_names) + NUMERIC_FEATURES


def save_feature_importance(model):
    """Save feature importance when the model exposes it."""
    model_step = model.named_steps["model"]
    if not hasattr(model_step, "feature_importances_"):
        print("The selected model does not expose feature importance. Skipping.")
        return None

    feature_names = get_feature_names(model.named_steps["preprocessor"])
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": model_step.feature_importances_})
        .sort_values("importance", ascending=False)
    )
    importance_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

    plt.figure(figsize=(10, 7))
    sns.barplot(data=importance_df.head(20), x="importance", y="feature")
    plt.title("Top 20 features by importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    save_plot("top20_feature_importance.png")

    return importance_df


def choose_recommended_split(days_per_week):
    """Choose a simple split based on available training days."""
    if days_per_week <= 2:
        return "fbw", "Selected FBW because the profile has at most 2 training days."
    if days_per_week == 3:
        return "ppl", "Selected PPL because the profile has 3 training days."
    if days_per_week == 4:
        return "upper_lower", "Selected upper/lower because the profile has 4 training days."
    return "ppl", "Selected PPL because the profile has 5 or more training days."


def filter_similar_users(data, profile, split):
    """Find similar training records, relaxing filters when needed."""
    filter_sets = [
        {
            "sex": profile.get("sex"),
            "level": profile.get("level"),
            "phase": profile.get("phase"),
            "split": split,
        },
        {
            "level": profile.get("level"),
            "phase": profile.get("phase"),
            "split": split,
        },
        {"level": profile.get("level"), "phase": profile.get("phase")},
        {"level": profile.get("level")},
        {},
    ]

    for filters in filter_sets:
        filtered = data.copy()
        for column, value in filters.items():
            if value is not None and column in filtered.columns:
                filtered = filtered[filtered[column] == value]
        if len(filtered) > 0:
            return filtered, filters

    return data.copy(), {}


def select_exercises_for_plan(similar_data, fallback_data, split, max_exercises=6):
    """Select exercises from similar users, falling back to the whole dataset."""
    source = similar_data if len(similar_data) > 0 else fallback_data

    def top_exercises(data, category, count):
        temp = data[data["exercise_category"] == category]
        return temp["exercise"].value_counts().head(count).index.tolist()

    selected = []
    if split == "upper_lower":
        upper = source[source["exercise_category"].isin(["push", "pull"])]
        lower = source[source["exercise_category"] == "legs"]
        selected.extend(upper["exercise"].value_counts().head(3).index.tolist())
        selected.extend(lower["exercise"].value_counts().head(3).index.tolist())
    else:
        selected.extend(top_exercises(source, "push", 2))
        selected.extend(top_exercises(source, "pull", 2))
        selected.extend(top_exercises(source, "legs", 2))

    selected = list(dict.fromkeys(selected))
    if len(selected) < max_exercises:
        for exercise in fallback_data["exercise"].value_counts().index.tolist():
            if exercise not in selected:
                selected.append(exercise)
            if len(selected) >= max_exercises:
                break

    return selected[:max_exercises]


def get_exercise_parameters(similar_data, fallback_data, exercise, phase):
    """Choose set targets from similar records and the current phase."""
    exercise_data = similar_data[similar_data["exercise"] == exercise]
    if len(exercise_data) == 0:
        exercise_data = fallback_data[fallback_data["exercise"] == exercise]
    if len(exercise_data) == 0:
        return {"sets": 3, "reps": 8, "target_rir": 2, "target_fatigue": 6}

    sets_per_session = exercise_data.groupby(["session_id", "exercise"])["set_number"].max()
    sets = int(round(safe_median(sets_per_session, default_value=3)))
    reps = int(round(safe_median(exercise_data["reps"], default_value=8)))

    if phase == "deload":
        target_rir = 4
        target_fatigue = 3
    elif phase == "strength":
        target_rir = 2
        target_fatigue = 7
    else:
        target_rir = 2
        target_fatigue = 6

    return {
        "sets": max(2, min(sets, 5)),
        "reps": max(3, min(reps, 15)),
        "target_rir": target_rir,
        "target_fatigue": target_fatigue,
    }


def get_history_features(data, user_id, exercise, similar_data, fallback_data):
    """Use user history when available, otherwise use medians from similar users."""
    if user_id is not None and user_id in set(data["user_id"].unique()):
        history = (
            data[(data["user_id"] == user_id) & (data["exercise"] == exercise)]
            .sort_values(["date", "session_id", "set_number"])
        )
        if len(history) > 0:
            last = history.iloc[-1]
            recent = history.tail(3)
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

    exercise_fallback = similar_data[similar_data["exercise"] == exercise]
    if len(exercise_fallback) == 0:
        exercise_fallback = fallback_data[fallback_data["exercise"] == exercise]
    if len(exercise_fallback) == 0:
        exercise_fallback = fallback_data.copy()

    median_weight = safe_median(exercise_fallback["weight"], default_value=40)
    median_reps = safe_median(exercise_fallback["reps"], default_value=8)
    median_rir = safe_median(exercise_fallback["rir"], default_value=2)
    median_fatigue = safe_median(exercise_fallback["fatigue"], default_value=6)
    median_volume = safe_median(
        exercise_fallback["volume"],
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


def apply_safety_rules(predicted_weight, history, profile):
    """Limit the model output with simple training safety rules."""
    predicted_weight = max(float(predicted_weight), 0)
    prev_weight = history["prev_weight"]
    recent_rir = history["rolling_rir_3"]
    recent_fatigue = history["rolling_fatigue_3"]
    level = profile.get("level", "intermediate")
    phase = profile.get("phase", "hypertrophy")
    age = profile.get("age")

    if pd.isna(prev_weight) or prev_weight <= 0:
        return round_to_nearest(predicted_weight), "no_adjustment"

    if level == "beginner":
        max_increase = 0.03
    elif level == "advanced":
        max_increase = 0.05
    else:
        max_increase = 0.04

    age_adjusted = False
    if age is not None:
        if age >= 60:
            max_increase = min(max_increase, 0.02)
            age_adjusted = True
        elif age >= 45:
            max_increase = min(max_increase, 0.03)
            age_adjusted = True

    if phase == "deload":
        safe_weight = min(predicted_weight, prev_weight * 0.85)
        return round_to_nearest(safe_weight), "deload_reduction"

    if recent_fatigue >= 8 or recent_rir <= 1:
        safe_weight = min(predicted_weight, prev_weight)
        adjustment = (
            "limited_by_high_fatigue_or_low_rir"
            if predicted_weight > prev_weight
            else "no_adjustment"
        )
        return round_to_nearest(safe_weight), adjustment

    upper_cap = prev_weight * (1 + max_increase)
    if predicted_weight > upper_cap:
        adjustment = (
            "age_adjusted_progression_cap"
            if age_adjusted
            else "limited_by_progression_cap"
        )
        return round_to_nearest(upper_cap), adjustment

    return round_to_nearest(predicted_weight), "no_adjustment"


def assign_training_day(split, exercise_category):
    """Map an exercise category to a simple training day label."""
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
    """Generate a demo plan and return it with metadata."""
    user_id = user_id if user_id is not None else profile.get("user_id")
    recommended_split, split_reason = choose_recommended_split(profile["days_per_week"])
    similar_data, filters_used = filter_similar_users(data, profile, recommended_split)
    selected_exercises = select_exercises_for_plan(
        similar_data=similar_data,
        fallback_data=data,
        split=recommended_split,
        max_exercises=max_exercises,
    )

    plan_rows = []
    history_used = False

    for exercise in selected_exercises:
        params = get_exercise_parameters(similar_data, data, exercise, profile["phase"])
        history = get_history_features(data, user_id, exercise, similar_data, data)
        history_used = history_used or history["history_available"]

        model_input = pd.DataFrame(
            [
                {
                    "exercise": exercise,
                    "level": profile["level"],
                    "split": recommended_split,
                    "phase": profile["phase"],
                    "sex": profile["sex"],
                    "set_number": 1,
                    "reps": params["reps"],
                    "fatigue": params["target_fatigue"],
                    "rir": params["target_rir"],
                    "prev_weight": history["prev_weight"],
                    "prev_reps": history["prev_reps"],
                    "prev_rir": history["prev_rir"],
                    "prev_fatigue": history["prev_fatigue"],
                    "prev_volume": history["prev_volume"],
                    "rolling_weight_3": history["rolling_weight_3"],
                    "rolling_reps_3": history["rolling_reps_3"],
                    "rolling_rir_3": history["rolling_rir_3"],
                    "rolling_fatigue_3": history["rolling_fatigue_3"],
                    "rolling_volume_3": history["rolling_volume_3"],
                }
            ]
        )

        predicted_weight = float(model.predict(model_input)[0])
        final_weight, safety_adjustment = apply_safety_rules(
            predicted_weight,
            history,
            profile,
        )
        exercise_category = EXERCISE_CATEGORY_MAP.get(exercise, "other")

        plan_rows.append(
            {
                "day": assign_training_day(recommended_split, exercise_category),
                "split": recommended_split,
                "exercise": exercise,
                "sets": params["sets"],
                "reps": params["reps"],
                "target_rir": params["target_rir"],
                "target_fatigue": params["target_fatigue"],
                "model_predicted_weight": round(predicted_weight, 2),
                "final_recommended_weight": final_weight,
                "history_available": history["history_available"],
                "safety_adjustment": safety_adjustment,
                "recommendation_reason": (
                    "Hybrid demo recommendation based on the loaded ML model, "
                    "similar-user statistics, available user history, and safety rules."
                ),
            }
        )

    plan_df = pd.DataFrame(plan_rows)
    metadata = {
        "profile": profile,
        "recommended_split": recommended_split,
        "split_reason": split_reason,
        "similar_user_filters": filters_used,
        "used_user_history": history_used,
        "exercise_count": len(plan_df),
    }
    return plan_df, metadata


def build_demo_scenarios(data):
    """Create the fixed demo scenarios used in the final system demo."""
    existing_user_id = data["user_id"].value_counts().index[0]
    existing_user_data = data[data["user_id"] == existing_user_id]

    return [
        {
            "name": "beginner_female_hypertrophy",
            "age": 24,
            "sex": "female",
            "level": "beginner",
            "phase": "hypertrophy",
            "days_per_week": 3,
        },
        {
            "name": "intermediate_male_strength",
            "age": 32,
            "sex": "male",
            "level": "intermediate",
            "phase": "strength",
            "days_per_week": 4,
        },
        {
            "name": "advanced_male_deload",
            "age": 38,
            "sex": "male",
            "level": "advanced",
            "phase": "deload",
            "days_per_week": 3,
        },
        {
            "name": "older_beginner_hypertrophy",
            "age": 62,
            "sex": "male",
            "level": "beginner",
            "phase": "hypertrophy",
            "days_per_week": 2,
        },
        {
            "name": "existing_user_with_history",
            "age": 30,
            "sex": safe_mode(existing_user_data["sex"], default_value="male"),
            "level": safe_mode(existing_user_data["level"], default_value="intermediate"),
            "phase": "hypertrophy",
            "days_per_week": 3,
            "user_id": existing_user_id,
        },
    ]


def build_scenario_summary_row(profile, plan_df, metadata):
    """Create one row for scenario comparison."""
    return {
        "scenario_name": profile["name"],
        "age": profile["age"],
        "sex": profile["sex"],
        "level": profile["level"],
        "phase": profile["phase"],
        "days_per_week": profile["days_per_week"],
        "recommended_split": metadata["recommended_split"],
        "exercise_count": metadata["exercise_count"],
        "avg_recommended_weight": plan_df["final_recommended_weight"].mean(),
        "avg_target_rir": plan_df["target_rir"].mean(),
        "history_used": metadata["used_user_history"],
        "notes": metadata["split_reason"],
    }


def run_demo_scenarios(scenarios, model, data):
    """Generate all demo plans and save the scenario comparison table."""
    summary_rows = []

    for profile in scenarios:
        plan_df, metadata = generate_training_plan(
            profile=profile,
            model=model,
            data=data,
            user_id=profile.get("user_id"),
            max_exercises=6,
        )
        scenario_name = profile["name"]
        plan_path = os.path.join(OUTPUT_DIR, f"plan_{scenario_name}.csv")
        plan_df.to_csv(plan_path, index=False)

        print(f"\nScenario: {scenario_name}")
        print(plan_df)
        summary_rows.append(build_scenario_summary_row(profile, plan_df, metadata))

    comparison_df = pd.DataFrame(summary_rows)
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, "scenario_comparison.csv"), index=False)
    return comparison_df


def save_stage3_summary(scenario_count):
    """Save a short text summary of the Stage 3 demo run."""
    summary_text = f"""
STAGE 3 SUMMARY

Generated scenarios: {scenario_count}
Plan files: {OUTPUT_DIR}/plan_<scenario_name>.csv
Scenario comparison: {OUTPUT_DIR}/scenario_comparison.csv

The trained model was loaded from:
{MODEL_PATH}

Stage 3 does not train models. It demonstrates the end-to-end system flow using
the model produced in Stage 2.

Recommendations are demonstrational and require expert validation before real
training use.
"""
    with open(os.path.join(OUTPUT_DIR, "stage3_summary.txt"), "w", encoding="utf-8") as file:
        file.write(summary_text.strip() + "\n")
    return summary_text


def main():
    """Run the Stage 3 end-to-end system demo."""
    print_section("LOAD DATA AND MODEL")
    data, best_model = load_inputs(DATA_PATH, MODEL_PATH)
    data = add_exercise_categories(data)
    print(f"Rows loaded: {len(data):,}")
    print(f"Model loaded from: {MODEL_PATH}")

    print_section("PREPARE MODEL FEATURES")
    model_ready = prepare_model_ready_data(data)
    print(f"Model-ready rows: {len(model_ready):,}")

    print_section("MODEL SANITY CHECK")
    run_model_sanity_check(model_ready, best_model)

    print_section("OPTIONAL MODEL INTERPRETATION")
    save_feature_importance(best_model)

    print_section("RUN DEMO SCENARIOS")
    scenarios = build_demo_scenarios(data)
    comparison_df = run_demo_scenarios(scenarios, best_model, data)
    print("\nScenario comparison:")
    print(comparison_df)

    print_section("STAGE 3 SUMMARY")
    summary_text = save_stage3_summary(len(scenarios))
    print(summary_text)
    print(f"\nOutputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
