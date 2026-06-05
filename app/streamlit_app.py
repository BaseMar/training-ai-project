from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.recommendation_engine import add_exercise_categories, generate_training_plan


DATASET_PATH = BASE_DIR / "data" / "FINAL_ENGINE_V4.csv"
MODEL_PATH = BASE_DIR / "models" / "best_weight_prediction_model.joblib"
DEMO_ASSETS_DIR = BASE_DIR / "app" / "demo_assets"
STAGE2_OUTPUTS_DIR = BASE_DIR / "outputs" / "stage2_outputs"

SCENARIOS = {
    "beginner_female_hypertrophy": "plan_beginner_female_hypertrophy.csv",
    "intermediate_male_strength": "plan_intermediate_male_strength.csv",
    "advanced_male_deload": "plan_advanced_male_deload.csv",
    "older_beginner_hypertrophy": "plan_older_beginner_hypertrophy.csv",
    "existing_user_with_history": "plan_existing_user_with_history.csv",
}

PLAN_DISPLAY_COLUMNS = [
    "day_name",
    "exercise",
    "sets",
    "reps",
    "target_rir",
    "target_fatigue",
    "final_recommended_weight",
    "safety_adjustment",
]

SAFETY_RULE_DESCRIPTIONS = {
    "no_adjustment": "Brak korekty. Rekomendacja modelu mieści się w bezpiecznym zakresie.",
    "limited_by_progression_cap": "Ciężar ograniczony przez maksymalny dopuszczalny skok progresji.",
    "limited_by_high_fatigue_or_low_rir": "Ciężar ograniczony przez wysokie zmęczenie albo niski RIR.",
    "deload_reduction": "Ciężar obniżony, ponieważ plan jest w fazie deload.",
    "age_adjusted_progression_cap": "Limit progresji zaostrzony ze względu na wiek użytkownika.",
}


st.set_page_config(
    page_title="Training AI Project",
    page_icon="T",
    layout="wide",
)

sns.set_theme(style="whitegrid")


@st.cache_data
def load_dataset() -> pd.DataFrame | None:
    """Load the canonical project dataset if it is available."""
    if not DATASET_PATH.exists():
        return None
    return pd.read_csv(DATASET_PATH)


@st.cache_data
def load_demo_assets() -> dict[str, pd.DataFrame | None]:
    """Load small demo CSV assets used by the presentation dashboard."""
    assets: dict[str, pd.DataFrame | None] = {"scenario_comparison": None}
    comparison_path = DEMO_ASSETS_DIR / "scenario_comparison.csv"

    if comparison_path.exists():
        assets["scenario_comparison"] = pd.read_csv(comparison_path)

    for scenario_name, file_name in SCENARIOS.items():
        plan_path = DEMO_ASSETS_DIR / file_name
        assets[scenario_name] = pd.read_csv(plan_path) if plan_path.exists() else None

    return assets


@st.cache_data
def load_plan(scenario_name: str) -> pd.DataFrame | None:
    """Load one demo training plan by scenario name."""
    file_name = SCENARIOS.get(scenario_name)
    if file_name is None:
        return None

    plan_path = DEMO_ASSETS_DIR / file_name
    if not plan_path.exists():
        return None

    return pd.read_csv(plan_path)


@st.cache_data
def load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    """Load a CSV file only when it exists."""
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_local_model() -> Any | None:
    """Load the local model artifact for live recommendations."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def model_exists() -> bool:
    """Return whether the local trained model artifact is present."""
    return MODEL_PATH.exists()


def safe_nunique(df: pd.DataFrame, column: str) -> int | None:
    """Return a nunique value only when the column exists."""
    if column not in df.columns:
        return None
    return int(df[column].nunique())


def numeric_mean(df: pd.DataFrame, column: str) -> float | None:
    """Return a numeric mean for an existing column."""
    if column not in df.columns:
        return None
    value = pd.to_numeric(df[column], errors="coerce").mean()
    return None if pd.isna(value) else float(value)


def format_number(value: float | int | None, suffix: str = "", decimals: int = 1) -> str:
    """Format a metric value for Streamlit."""
    if value is None or pd.isna(value):
        return "brak"
    if isinstance(value, int):
        return f"{value:,}".replace(",", " ")
    return f"{value:,.{decimals}f}{suffix}".replace(",", " ")


def metric_value(df: pd.DataFrame, column: str, label: str) -> None:
    """Render a metric for a unique count if the source column exists."""
    value = safe_nunique(df, column)
    st.metric(label, "brak" if value is None else format_number(value))


def add_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Add a volume column when reps and weight are available."""
    volume_df = df.copy()
    if {"reps", "weight"}.issubset(volume_df.columns):
        volume_df["volume"] = (
            pd.to_numeric(volume_df["reps"], errors="coerce")
            * pd.to_numeric(volume_df["weight"], errors="coerce")
        )
    return volume_df


def monthly_volume(df: pd.DataFrame) -> pd.Series:
    """Calculate monthly training volume."""
    if not {"date", "volume"}.issubset(df.columns):
        return pd.Series(dtype=float)

    temp = df.copy()
    temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
    return (
        temp.dropna(subset=["date"])
        .set_index("date")
        .resample("MS")["volume"]
        .sum()
        .rename("volume")
    )


def compact_bar_chart(
    series: pd.Series,
    title: str,
    x_label: str,
    y_label: str,
    color: str = "#3B82F6",
    height: float = 3.0,
) -> None:
    """Render a compact bar chart for dashboard panels."""
    if series.empty:
        st.info(f"Brak danych dla wykresu: {title}.")
        return

    fig, ax = plt.subplots(figsize=(6.2, height))
    sns.barplot(x=series.index.astype(str), y=series.values, ax=ax, color=color)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def compact_model_bar_chart(
    df: pd.DataFrame,
    metric: str,
    title: str,
    color: str,
) -> None:
    """Render a compact model-comparison chart."""
    if "model" not in df.columns or metric not in df.columns:
        st.info(f"Brak kolumn wymaganych dla wykresu `{metric}`.")
        return

    chart_df = df.copy()
    chart_df[metric] = pd.to_numeric(chart_df[metric], errors="coerce")
    chart_df = chart_df.dropna(subset=[metric]).sort_values(metric)
    if chart_df.empty:
        st.info(f"Brak danych liczbowych dla `{metric}`.")
        return

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    sns.barplot(data=chart_df, x=metric, y="model", ax=ax, color=color)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(metric, fontsize=9)
    ax.set_ylabel("model", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def show_missing_demo_assets_info() -> None:
    """Render instructions for recreating missing stage 3 demo assets."""
    st.warning("Brakuje wymaganych plików demo w `app/demo_assets/`.")
    st.markdown(
        "Uruchom lokalnie `python scripts/03_system_demo.py`, a następnie skopiuj "
        "małe pliki CSV z `outputs/stage3_outputs/` do `app/demo_assets/`."
    )


def get_plan_display(plan_df: pd.DataFrame) -> pd.DataFrame:
    """Return a simplified plan table for presentation."""
    columns = [column for column in PLAN_DISPLAY_COLUMNS if column in plan_df.columns]
    return plan_df[columns].copy()


def plan_metrics(plan_df: pd.DataFrame) -> dict[str, float | int | None]:
    """Calculate presentation KPIs for a training plan."""
    safety_count = 0
    if "safety_adjustment" in plan_df.columns:
        safety_count = int(plan_df["safety_adjustment"].ne("no_adjustment").sum())

    return {
        "day_count": safe_nunique(plan_df, "day_number"),
        "exercise_count": safe_nunique(plan_df, "exercise"),
        "avg_weight": numeric_mean(plan_df, "final_recommended_weight"),
        "avg_rir": numeric_mean(plan_df, "target_rir"),
        "history_count": (
            int(plan_df["history_available"].astype(str).str.lower().eq("true").sum())
            if "history_available" in plan_df.columns
            else None
        ),
        "safety_count": safety_count,
        "safety_percent": (safety_count / len(plan_df) * 100) if len(plan_df) else None,
    }


def render_plan_kpis(plan_df: pd.DataFrame) -> None:
    """Render training-plan KPIs."""
    metrics = plan_metrics(plan_df)
    cols = st.columns(6)
    cols[0].metric("Dni", format_number(metrics["day_count"]))
    cols[1].metric("Ćwiczenia", format_number(metrics["exercise_count"]))
    cols[2].metric("Śr. ciężar", format_number(metrics["avg_weight"], " kg"))
    cols[3].metric("Śr. RIR", format_number(metrics["avg_rir"]))
    cols[4].metric("Z historią", format_number(metrics["history_count"]))
    cols[5].metric(
        "Korekty bezpieczeństwa",
        format_number(metrics["safety_count"]),
        format_number(metrics["safety_percent"], "%"),
    )


def render_overview_tab(dataset_df: pd.DataFrame | None) -> None:
    st.title("Training AI Project")
    st.subheader("Projekt i implementacja systemu sztucznej inteligencji do analizy danych treningowych")

    st.markdown(
        "Dashboard pełni rolę warstwy prezentacyjnej projektu i integruje wyniki poprzednich etapów."
    )
    st.markdown(
        "`Generator` -> `Dataset` -> `EDA` -> `ML Model` -> "
        "`Recommendation Engine` -> `Streamlit Dashboard`"
    )

    stage_cols = st.columns(4)
    stage_cols[0].info("Etap 1\n\nEDA i kontrola jakości danych.")
    stage_cols[1].info("Etap 2\n\nModel ML i recommender.")
    stage_cols[2].info("Etap 3\n\nScenariusze demo i plany.")
    stage_cols[3].info("Etap 4\n\nDashboard prezentacyjny.")

    if dataset_df is None:
        st.warning("KPI datasetu pojawią się po dodaniu `data/FINAL_ENGINE_V4.csv`.")
        return

    st.subheader("KPI datasetu")
    cols = st.columns(4)
    cols[0].metric("Rekordy", format_number(len(dataset_df)))
    cols[1].metric("Użytkownicy", format_number(safe_nunique(dataset_df, "user_id")))
    cols[2].metric("Sesje", format_number(safe_nunique(dataset_df, "session_id")))
    cols[3].metric("Ćwiczenia", format_number(safe_nunique(dataset_df, "exercise")))


def render_dataset_tab(dataset_df: pd.DataFrame | None) -> None:
    st.header("Dataset")

    if dataset_df is None:
        st.warning("Brakuje `data/FINAL_ENGINE_V4.csv`. Wygeneruj dataset lub dodaj plik do katalogu `data/`.")
        return

    dataset_df = add_volume(dataset_df)
    st.info("Dataset jest syntetyczny i został wygenerowany przez moduł generatora danych.")

    st.subheader("Statystyki globalne")
    cols = st.columns(6)
    with cols[0]:
        metric_value(dataset_df, "user_id", "Użytkownicy")
    with cols[1]:
        metric_value(dataset_df, "session_id", "Sesje")
    with cols[2]:
        metric_value(dataset_df, "exercise", "Ćwiczenia")
    with cols[3]:
        metric_value(dataset_df, "level", "Poziomy")
    with cols[4]:
        metric_value(dataset_df, "split", "Splity")
    with cols[5]:
        metric_value(dataset_df, "phase", "Fazy")

    if "date" in dataset_df.columns:
        dates = pd.to_datetime(dataset_df["date"], errors="coerce")
        if dates.notna().any():
            st.caption(f"Zakres dat datasetu: {dates.min().date()} - {dates.max().date()}")

    with st.expander("Podgląd całego datasetu", expanded=False):
        st.dataframe(dataset_df.head(100), use_container_width=True)

    with st.expander("Typy kolumn", expanded=False):
        dtypes_df = dataset_df.dtypes.astype(str).reset_index()
        dtypes_df.columns = ["column", "dtype"]
        st.dataframe(dtypes_df, use_container_width=True)

    if "user_id" not in dataset_df.columns:
        st.warning("Brak kolumny `user_id`, więc analiza użytkownika jest niedostępna.")
        return

    st.subheader("Analiza wybranego użytkownika")
    users = sorted(dataset_df["user_id"].dropna().unique().tolist())
    selected_user = st.selectbox("Wybierz user_id", users, key="dataset_user_id")
    user_df = dataset_df[dataset_df["user_id"] == selected_user].copy()

    user_dates = (
        pd.to_datetime(user_df["date"], errors="coerce")
        if "date" in user_df.columns
        else pd.Series(dtype="datetime64[ns]")
    )
    date_range = (
        f"{user_dates.min().date()} - {user_dates.max().date()}"
        if user_dates.notna().any()
        else "brak"
    )

    user_cols = st.columns(6)
    user_cols[0].metric("Rekordy", format_number(len(user_df)))
    user_cols[1].metric("Sesje", format_number(safe_nunique(user_df, "session_id")))
    user_cols[2].metric("Ćwiczenia", format_number(safe_nunique(user_df, "exercise")))
    user_cols[3].metric("Zakres dat", date_range)
    user_cols[4].metric("Śr. RIR", format_number(numeric_mean(user_df, "rir")))
    user_cols[5].metric("Śr. fatigue", format_number(numeric_mean(user_df, "fatigue")))

    st.metric("Suma volume użytkownika", format_number(user_df["volume"].sum() if "volume" in user_df.columns else None))

    left_col, right_col = st.columns([1, 1])
    with left_col:
        if "exercise" in user_df.columns:
            compact_bar_chart(
                user_df["exercise"].value_counts().head(10),
                "Top ćwiczeń użytkownika",
                "Ćwiczenie",
                "Liczba serii",
                color="#0F766E",
            )
            st.caption("Najczęściej wykonywane ćwiczenia dla wybranego użytkownika.")
    with right_col:
        comparison_rows = []
        for column, label in [
            ("rir", "avg RIR"),
            ("fatigue", "avg fatigue"),
            ("weight", "avg weight"),
            ("reps", "avg reps"),
        ]:
            comparison_rows.append(
                {
                    "metric": label,
                    "użytkownik": numeric_mean(user_df, column),
                    "cały dataset": numeric_mean(dataset_df, column),
                }
            )
        comparison_df = pd.DataFrame(comparison_rows)
        st.dataframe(comparison_df, use_container_width=True)
        st.caption("Porównanie średnich wartości użytkownika z całym datasetem.")

    user_monthly_volume = monthly_volume(user_df)
    if not user_monthly_volume.empty:
        st.subheader("Trend miesięcznego volume użytkownika")
        st.line_chart(user_monthly_volume, use_container_width=True)
        st.caption("Suma objętości treningowej użytkownika w kolejnych miesiącach.")

    with st.expander("Podgląd danych wybranego użytkownika", expanded=False):
        st.dataframe(user_df.head(200), use_container_width=True)


def render_eda_tab(dataset_df: pd.DataFrame | None) -> None:
    st.header("EDA")

    if dataset_df is None:
        st.warning("Brakuje datasetu `data/FINAL_ENGINE_V4.csv`, więc sekcja EDA jest niedostępna.")
        return

    dataset_df = add_volume(dataset_df)
    first_row = st.columns(2)
    second_row = st.columns(2)
    chart_specs = [
        (first_row[0], "level", "Rozkład poziomu", "Poziom", "#2563EB", "Pokazuje udział poziomów zaawansowania w danych."),
        (first_row[1], "split", "Rozkład splitów", "Split", "#16A34A", "Pokazuje, jakie struktury treningowe dominują w datasencie."),
        (second_row[0], "phase", "Rozkład faz", "Faza", "#9333EA", "Pokazuje rozkład faz treningowych."),
        (second_row[1], "exercise", "Top 10 ćwiczeń", "Ćwiczenie", "#EA580C", "Pokazuje najczęstsze ćwiczenia w danych."),
    ]

    for container, column, title, x_label, color, caption in chart_specs:
        with container:
            if column not in dataset_df.columns:
                st.info(f"Brak kolumny `{column}` - pomijam wykres.")
                continue
            values = dataset_df[column].value_counts().head(10 if column == "exercise" else None)
            compact_bar_chart(values, title, x_label, "Liczba rekordów", color=color)
            st.caption(caption)

    total_monthly_volume = monthly_volume(dataset_df)
    if not total_monthly_volume.empty:
        st.subheader("Trend całkowitej objętości po miesiącach")
        st.line_chart(total_monthly_volume, use_container_width=True)
        st.caption("Suma `reps * weight` dla całego datasetu w ujęciu miesięcznym.")
    else:
        st.info("Trend objętości wymaga kolumn `date`, `reps` i `weight`.")


def find_model_metrics() -> pd.DataFrame | None:
    """Find model-comparison metrics in demo assets or Stage 2 outputs."""
    for path in [
        DEMO_ASSETS_DIR / "model_comparison_results.csv",
        STAGE2_OUTPUTS_DIR / "model_comparison_results.csv",
    ]:
        metrics_df = load_csv_if_exists(path)
        if metrics_df is not None:
            return metrics_df
    return None


def render_ml_model_tab() -> None:
    st.header("ML Model")
    st.markdown(
        "Model regresyjny przewiduje sugerowany ciężar treningowy `weight`. "
        "Jest komponentem systemu rekomendacyjnego, a nie samodzielnym trenerem."
    )

    if model_exists():
        st.success("Model dostępny lokalnie: `models/best_weight_prediction_model.joblib`")
    else:
        st.warning("Model `models/best_weight_prediction_model.joblib` nie jest dostępny lokalnie.")
        st.markdown(
            "Model nie jest commitowany do zwykłego Gita, bo jest duży. Można go wygenerować przez "
            "`python scripts/02_modeling_and_recommendation.py` albo pobrać z GitHub Release, jeśli będzie udostępniony."
        )

    metrics_df = find_model_metrics()
    if metrics_df is None:
        st.info("Brak metryk modelu. Możesz je wygenerować przez `python scripts/02_modeling_and_recommendation.py`.")
    else:
        ranked = metrics_df.copy()
        ranked["MAE"] = pd.to_numeric(ranked["MAE"], errors="coerce")
        best_row = ranked.sort_values("MAE").iloc[0]
        cols = st.columns(4)
        cols[0].metric("Najlepszy MAE", format_number(best_row.get("MAE")))
        cols[1].metric("RMSE", format_number(pd.to_numeric(best_row.get("RMSE"), errors="coerce")))
        cols[2].metric("R2", format_number(pd.to_numeric(best_row.get("R2"), errors="coerce"), decimals=3))
        cols[3].metric(
            "Within 5 kg",
            format_number(pd.to_numeric(best_row.get("within_5kg_percent"), errors="coerce"), "%"),
        )
        st.caption(f"Najlepszy model według MAE: `{best_row.get('model')}`.")

        chart_cols = st.columns(2)
        with chart_cols[0]:
            compact_model_bar_chart(metrics_df, "MAE", "Porównanie modeli według MAE", "#2563EB")
        with chart_cols[1]:
            compact_model_bar_chart(
                metrics_df,
                "within_5kg_percent",
                "Trafienia w zakresie 5 kg",
                "#16A34A",
            )

        with st.expander("Pełna tabela porównania modeli", expanded=False):
            st.dataframe(metrics_df, use_container_width=True)

    group_files = [
        STAGE2_OUTPUTS_DIR / "group_evaluation_by_level.csv",
        STAGE2_OUTPUTS_DIR / "group_evaluation_by_phase.csv",
        STAGE2_OUTPUTS_DIR / "group_evaluation_by_sex.csv",
        STAGE2_OUTPUTS_DIR / "group_evaluation_by_exercise.csv",
    ]
    existing_group_files = [path for path in group_files if path.exists()]
    if existing_group_files:
        with st.expander("Ewaluacja grupowa", expanded=False):
            for path in existing_group_files:
                st.markdown(f"**{path.name}**")
                st.dataframe(pd.read_csv(path), use_container_width=True)


def render_recommendation_tab(demo_assets: dict[str, pd.DataFrame | None]) -> None:
    st.header("Recommendation Demo")
    st.markdown(
        "Plan jest generowany na podstawie profilu, podobnych użytkowników, predykcji modelu i reguł bezpieczeństwa."
    )

    comparison_df = demo_assets.get("scenario_comparison")
    available_plans = [scenario for scenario in SCENARIOS if demo_assets.get(scenario) is not None]

    if comparison_df is None and not available_plans:
        show_missing_demo_assets_info()
        return

    if comparison_df is not None:
        st.subheader("Porównanie scenariuszy")
        st.dataframe(comparison_df, use_container_width=True)
    else:
        st.warning("Brakuje `app/demo_assets/scenario_comparison.csv`.")

    selected_scenario = st.selectbox(
        "Scenariusz",
        list(SCENARIOS.keys()),
        key="selected_scenario",
    )

    selected_plan = load_plan(selected_scenario)
    if selected_plan is None:
        st.warning(f"Brakuje pliku planu dla scenariusza `{selected_scenario}`.")
        st.markdown(
            f"Oczekiwany plik: `app/demo_assets/{SCENARIOS[selected_scenario]}`. "
            "Możesz go odtworzyć przez `python scripts/03_system_demo.py` i skopiować z `outputs/stage3_outputs/`."
        )
        return

    st.subheader("KPI planu")
    render_plan_kpis(selected_plan)

    st.subheader("Uproszczony plan treningowy")
    st.dataframe(get_plan_display(selected_plan), use_container_width=True)

    with st.expander("Pełna tabela techniczna planu", expanded=False):
        st.dataframe(selected_plan, use_container_width=True)


def render_safety_rules_tab() -> None:
    st.header("Safety Rules")
    st.markdown(
        "System nie ufa ślepo modelowi ML. Model przewiduje ciężar, ale finalna rekomendacja "
        "jest korygowana przez reguły zależne od poziomu, fazy, zmęczenia, RIR i wieku."
    )

    scenario_default = st.session_state.get("selected_scenario", "beginner_female_hypertrophy")
    scenario_names = list(SCENARIOS.keys())
    default_index = scenario_names.index(scenario_default) if scenario_default in scenario_names else 0
    selected_scenario = st.selectbox(
        "Scenariusz do analizy bezpieczeństwa",
        scenario_names,
        index=default_index,
        key="safety_scenario",
    )
    selected_plan = load_plan(selected_scenario)

    if selected_plan is None:
        show_missing_demo_assets_info()
    elif "safety_adjustment" not in selected_plan.columns:
        st.warning("Wybrany plan nie zawiera kolumny `safety_adjustment`.")
    else:
        chart_col, table_col = st.columns([1, 1.4])
        with chart_col:
            compact_bar_chart(
                selected_plan["safety_adjustment"].value_counts(),
                "Rozkład safety_adjustment",
                "Korekta",
                "Liczba ćwiczeń",
                color="#DC2626",
                height=3.2,
            )
            st.caption("Liczba ćwiczeń według typu zastosowanej korekty.")

        with table_col:
            adjusted = selected_plan[selected_plan["safety_adjustment"] != "no_adjustment"]
            st.subheader("Ćwiczenia z korektą")
            if adjusted.empty:
                st.info("W wybranym planie nie ma korekt innych niż `no_adjustment`.")
            else:
                columns_to_show = [
                    column
                    for column in [
                        "day_name",
                        "exercise",
                        "model_predicted_weight",
                        "final_recommended_weight",
                        "target_rir",
                        "target_fatigue",
                        "safety_adjustment",
                    ]
                    if column in adjusted.columns
                ]
                st.dataframe(adjusted[columns_to_show], use_container_width=True)

    st.subheader("Znaczenie reguł")
    for rule_name, description in SAFETY_RULE_DESCRIPTIONS.items():
        st.markdown(f"- `{rule_name}` - {description}")


def render_live_generator_tab(dataset_df: pd.DataFrame | None) -> None:
    st.header("Live Generator")

    if not model_exists():
        st.warning("Live generator jest niedostępny, ponieważ brakuje lokalnego modelu `.joblib`.")
        st.markdown(
            "Dashboard działa nadal w trybie prezentacyjnym na `app/demo_assets/`. "
            "Aby włączyć live generator, uruchom `python scripts/02_modeling_and_recommendation.py` "
            "albo pobierz model z GitHub Release i umieść go w `models/best_weight_prediction_model.joblib`."
        )
        return

    if dataset_df is None:
        st.warning("Model jest dostępny, ale live generator wymaga także `data/FINAL_ENGINE_V4.csv`.")
        return

    model = load_local_model()
    if model is None:
        st.warning("Nie udało się wczytać modelu lokalnego.")
        return

    st.success("Model dostępny lokalnie. Możesz wygenerować demonstracyjny plan na żywo.")
    user_options: list[Any] = ["brak"]
    if "user_id" in dataset_df.columns:
        user_options.extend(sorted(dataset_df["user_id"].dropna().unique().tolist()))

    with st.form("live_generator_form"):
        col_a, col_b, col_c = st.columns(3)
        age = col_a.number_input("Wiek", min_value=16, max_value=90, value=30)
        sex = col_b.selectbox("Płeć", ["female", "male"])
        level = col_c.selectbox("Poziom", ["beginner", "intermediate", "advanced"])

        col_d, col_e, col_f = st.columns(3)
        phase = col_d.selectbox("Faza", ["hypertrophy", "strength", "deload"])
        days_per_week = col_e.number_input("Dni treningowe w tygodniu", min_value=2, max_value=6, value=3)
        selected_user_id = col_f.selectbox("Historia user_id", user_options)

        submitted = st.form_submit_button("Generuj plan")

    if not submitted:
        st.info("Formularz używa modelu z Etapu 2 oraz lekkiej logiki rekomendacyjnej z Etapu 3.")
        return

    profile = {
        "name": "live_streamlit_plan",
        "age": int(age),
        "sex": sex,
        "level": level,
        "phase": phase,
        "days_per_week": int(days_per_week),
    }
    user_id = None if selected_user_id == "brak" else selected_user_id
    if user_id is not None:
        profile["user_id"] = user_id

    try:
        plan_df, metadata = generate_training_plan(
            profile=profile,
            model=model,
            data=add_exercise_categories(dataset_df),
            user_id=user_id,
        )
    except Exception as error:
        st.error(f"Nie udało się wygenerować planu: {error}")
        return

    st.subheader("Wygenerowany plan")
    meta_cols = st.columns(3)
    meta_cols[0].metric("Rekomendowany split", metadata["recommended_split"])
    meta_cols[1].metric("Dni", metadata["day_count"])
    meta_cols[2].metric("Ćwiczenia", metadata["total_exercises"])
    st.caption(metadata["split_reason"])

    render_plan_kpis(plan_df)
    st.dataframe(get_plan_display(plan_df), use_container_width=True)

    with st.expander("Pełna tabela techniczna planu", expanded=False):
        st.dataframe(plan_df, use_container_width=True)

    st.download_button(
        "Pobierz plan CSV",
        data=plan_df.to_csv(index=False).encode("utf-8"),
        file_name="live_training_plan.csv",
        mime="text/csv",
    )


def main() -> None:
    dataset_df = load_dataset()
    demo_assets = load_demo_assets()

    tabs = st.tabs(
        [
            "Overview",
            "Dataset",
            "EDA",
            "ML Model",
            "Recommendation Demo",
            "Safety Rules",
            "Live Generator",
        ]
    )

    with tabs[0]:
        render_overview_tab(dataset_df)
    with tabs[1]:
        render_dataset_tab(dataset_df)
    with tabs[2]:
        render_eda_tab(dataset_df)
    with tabs[3]:
        render_ml_model_tab()
    with tabs[4]:
        render_recommendation_tab(demo_assets)
    with tabs[5]:
        render_safety_rules_tab()
    with tabs[6]:
        render_live_generator_tab(dataset_df)


if __name__ == "__main__":
    main()
