"""Reusable recommendation helpers for the Stage 4 Streamlit dashboard.

The module contains the lightweight Stage 3 recommendation logic without any
file writes or model training. It expects a trained model object loaded by the
caller and the canonical dataset as a pandas DataFrame.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


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

PHASE_WEIGHT_FACTORS = {
    "strength": 1.0,
    "hypertrophy": 0.88,
    "deload": 0.7,
}

CALIBRATION_ANCHORS = {
    "Bench Press": [("Bench Press", 1.0)],
    "Incline Bench Press": [("Bench Press", 0.8)],
    "Shoulder Press": [("Shoulder Press", 1.0)],
    "Lateral Raise": [("Shoulder Press", 0.25)],
    "Triceps Pushdown": [("Bench Press", 0.35)],
    "Barbell Row": [("Barbell Row", 1.0)],
    "Lat Pulldown": [("Lat Pulldown", 1.0), ("Barbell Row", 0.75)],
    "Seated Row": [("Barbell Row", 0.75)],
    "Rear Delt Row": [("Barbell Row", 0.35)],
    "Biceps Curl": [("Barbell Row", 0.25)],
    "Squat": [("Squat", 1.0)],
    "Deadlift": [("Deadlift", 1.0)],
    "Leg Press": [("Leg Press", 1.0), ("Squat", 1.2)],
    "Leg Curl": [("Squat", 0.25)],
    "Calf Raise": [("Squat", 0.5)],
}


def round_to_nearest(value: float, step: float = 2.5) -> float:
    """Round a weight to the nearest practical training increment."""
    return round(float(value) / step) * step


def safe_median(series: pd.Series, default_value: float = 0) -> float:
    """Return the median, or a default when the series is empty."""
    clean = pd.Series(series).dropna()
    if len(clean) == 0:
        return default_value
    return float(clean.median())


def safe_mode(series: pd.Series, default_value: Any = None) -> Any:
    """Return the most common value, or a default when needed."""
    clean = pd.Series(series).dropna()
    if len(clean) == 0:
        return default_value
    return clean.mode().iloc[0]


def normalize_strength_calibration(calibration: dict[str, Any] | None) -> dict[str, float]:
    """Return positive strength calibration values keyed by exercise name."""
    if not calibration:
        return {}

    normalized = {}
    for exercise, value in calibration.items():
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if numeric_value > 0:
            normalized[exercise] = numeric_value
    return normalized


def has_valid_strength_calibration(calibration: dict[str, Any] | None) -> bool:
    """Return whether at least one usable strength anchor was provided."""
    return len(normalize_strength_calibration(calibration)) > 0


def get_calibrated_weight(
    exercise: str,
    calibration: dict[str, Any] | None,
    phase: str,
) -> tuple[float | None, float | None]:
    """Estimate an exercise weight from user-provided strength anchors."""
    normalized = normalize_strength_calibration(calibration)
    if not normalized:
        return None, None

    for anchor_name, ratio in CALIBRATION_ANCHORS.get(exercise, []):
        anchor_weight = normalized.get(anchor_name)
        if anchor_weight is None:
            continue
        reference_weight = anchor_weight * ratio
        phase_factor = PHASE_WEIGHT_FACTORS.get(phase, 0.88)
        return reference_weight * phase_factor, reference_weight

    return None, None


def add_exercise_categories(data: pd.DataFrame) -> pd.DataFrame:
    """Add helper columns used by the demo recommender."""
    categorized = data.copy()
    categorized["date"] = pd.to_datetime(categorized["date"], errors="coerce")
    categorized["volume"] = (
        pd.to_numeric(categorized["reps"], errors="coerce")
        * pd.to_numeric(categorized["weight"], errors="coerce")
    )
    categorized["exercise_category"] = (
        categorized["exercise"].map(EXERCISE_CATEGORY_MAP).fillna("other")
    )
    return categorized


def choose_recommended_split(days_per_week: int) -> tuple[str, str]:
    """Choose a simple split based on available training days."""
    if days_per_week <= 2:
        return "fbw", "Selected FBW because the profile has at most 2 training days."
    if days_per_week == 3:
        return "ppl", "Selected PPL because the profile has 3 training days."
    if days_per_week == 4:
        return "upper_lower", "Selected upper/lower because the profile has 4 training days."
    return "ppl", "Selected PPL because the profile has 5 or more training days."


def build_weekly_day_templates(split: str, days_per_week: int) -> list[dict[str, Any]]:
    """Build the training days that make up a full weekly plan."""
    if split == "ppl":
        cycle = [
            {"day_focus": "Push", "categories": ["push"], "target_exercises": 6},
            {"day_focus": "Pull", "categories": ["pull"], "target_exercises": 6},
            {"day_focus": "Legs", "categories": ["legs"], "target_exercises": 5},
        ]
    elif split == "upper_lower":
        cycle = [
            {"day_focus": "Upper", "categories": ["push", "pull"], "target_exercises": 6},
            {"day_focus": "Lower", "categories": ["legs"], "target_exercises": 5},
        ]
    else:
        cycle = [
            {
                "day_focus": "Full Body",
                "categories": ["push", "pull", "legs"],
                "target_exercises": 6,
            }
        ]

    templates = []
    for day_index in range(days_per_week):
        template = cycle[day_index % len(cycle)].copy()
        template["day_number"] = day_index + 1
        template["day_name"] = f"Day {day_index + 1} - {template['day_focus']}"
        templates.append(template)
    return templates


def filter_similar_users(
    data: pd.DataFrame,
    profile: dict[str, Any],
    split: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
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


def select_exercises_for_day(
    similar_data: pd.DataFrame,
    fallback_data: pd.DataFrame,
    categories: list[str],
    target_count: int,
    day_number: int,
) -> list[str]:
    """Select exercises for one training day from similar-user data."""
    source = similar_data if len(similar_data) > 0 else fallback_data

    def ranked_exercises(data: pd.DataFrame) -> list[str]:
        temp = data[data["exercise_category"].isin(categories)]
        return temp["exercise"].value_counts().index.tolist()

    selected = ranked_exercises(source)
    if len(selected) == 0:
        selected = ranked_exercises(fallback_data)

    selected = list(dict.fromkeys(selected))
    if len(selected) > 0:
        offset = ((day_number - 1) * 2) % len(selected)
        selected = selected[offset:] + selected[:offset]

    if len(selected) < target_count:
        for exercise in ranked_exercises(fallback_data):
            if exercise not in selected:
                selected.append(exercise)
            if len(selected) >= target_count:
                break

    return selected[:target_count]


def get_exercise_parameters(
    similar_data: pd.DataFrame,
    fallback_data: pd.DataFrame,
    exercise: str,
    phase: str,
) -> dict[str, int]:
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


def get_history_features(
    data: pd.DataFrame,
    user_id: Any,
    exercise: str,
    similar_data: pd.DataFrame,
    fallback_data: pd.DataFrame,
) -> dict[str, Any]:
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


def apply_safety_rules(
    predicted_weight: float,
    history: dict[str, Any],
    profile: dict[str, Any],
) -> tuple[float, str]:
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


def generate_training_plan(
    profile: dict[str, Any],
    model: Any,
    data: pd.DataFrame,
    user_id: Any = None,
    strength_calibration: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate a full weekly demo plan and return it with metadata."""
    prepared_data = data.copy()
    if "exercise_category" not in prepared_data.columns:
        prepared_data = add_exercise_categories(prepared_data)

    selected_user_id = user_id if user_id is not None else profile.get("user_id")
    recommended_split, split_reason = choose_recommended_split(profile["days_per_week"])
    similar_data, filters_used = filter_similar_users(
        prepared_data,
        profile,
        recommended_split,
    )
    weekly_templates = build_weekly_day_templates(
        split=recommended_split,
        days_per_week=profile["days_per_week"],
    )
    normalized_calibration = normalize_strength_calibration(strength_calibration)

    plan_rows = []
    history_used = False
    calibration_used = False

    for template in weekly_templates:
        selected_exercises = select_exercises_for_day(
            similar_data=similar_data,
            fallback_data=prepared_data,
            categories=template["categories"],
            target_count=template["target_exercises"],
            day_number=template["day_number"],
        )

        for exercise_order, exercise in enumerate(selected_exercises, start=1):
            params = get_exercise_parameters(
                similar_data,
                prepared_data,
                exercise,
                profile["phase"],
            )
            history = get_history_features(
                prepared_data,
                selected_user_id,
                exercise,
                similar_data,
                prepared_data,
            )
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
            calibrated_weight, calibration_reference = get_calibrated_weight(
                exercise,
                normalized_calibration,
                profile["phase"],
            )

            if history["history_available"]:
                final_weight, safety_adjustment = apply_safety_rules(
                    predicted_weight,
                    history,
                    profile,
                )
                weight_source = "user_history"
            elif calibrated_weight is not None and calibration_reference is not None:
                safety_history = history.copy()
                safety_history["prev_weight"] = calibration_reference
                safety_history["rolling_weight_3"] = calibration_reference
                safety_history["rolling_rir_3"] = params["target_rir"]
                safety_history["rolling_fatigue_3"] = params["target_fatigue"]
                final_weight, safety_adjustment = apply_safety_rules(
                    calibrated_weight,
                    safety_history,
                    profile,
                )
                weight_source = "strength_calibration"
                calibration_used = True
            else:
                final_weight, safety_adjustment = apply_safety_rules(
                    predicted_weight,
                    history,
                    profile,
                )
                weight_source = "fallback_median"

            exercise_category = EXERCISE_CATEGORY_MAP.get(exercise, "other")

            plan_rows.append(
                {
                    "day_number": template["day_number"],
                    "day_name": template["day_name"],
                    "day_focus": template["day_focus"],
                    "exercise_order": exercise_order,
                    "split": recommended_split,
                    "exercise": exercise,
                    "exercise_category": exercise_category,
                    "sets": params["sets"],
                    "reps": params["reps"],
                    "target_rir": params["target_rir"],
                    "target_fatigue": params["target_fatigue"],
                    "model_predicted_weight": round(predicted_weight, 2),
                    "final_recommended_weight": final_weight,
                    "weight_source": weight_source,
                    "history_available": history["history_available"],
                    "safety_adjustment": safety_adjustment,
                    "recommendation_reason": (
                        "Weekly demo recommendation based on split structure, "
                        "the loaded ML model, similar-user data, user history, "
                        "and safety rules."
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
        "used_strength_calibration": calibration_used,
        "strength_calibration": normalized_calibration,
        "day_count": len(weekly_templates),
        "total_exercises": len(plan_df),
        "avg_exercises_per_day": (
            len(plan_df) / len(weekly_templates) if len(weekly_templates) else 0
        ),
    }
    return plan_df, metadata
