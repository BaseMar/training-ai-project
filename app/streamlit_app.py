from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
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

SAFETY_RULE_DESCRIPTIONS = {
    "no_adjustment": "Brak korekty - rekomendacja modelu mieści się w bezpiecznym zakresie.",
    "limited_by_progression_cap": "Ciężar ograniczony przez maksymalny dopuszczalny skok progresji.",
    "limited_by_high_fatigue_or_low_rir": "Ciężar ograniczony przez wysokie zmęczenie albo niski RIR.",
    "deload_reduction": "Ciężar obniżony, ponieważ scenariusz jest w fazie deload.",
    "age_adjusted_progression_cap": "Limit progresji zaostrzony ze względu na wiek użytkownika.",
}


st.set_page_config(
    page_title="Training AI Project",
    page_icon="T",
    layout="wide",
)


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


def model_exists() -> bool:
    """Return whether the local trained model artifact is present."""
    return MODEL_PATH.exists()


def safe_nunique(df: pd.DataFrame, column: str) -> int | None:
    """Return a nunique value only when the column exists."""
    if column not in df.columns:
        return None
    return int(df[column].nunique())


def metric_value(df: pd.DataFrame, column: str, label: str) -> None:
    """Render a metric for a unique count if the source column exists."""
    value = safe_nunique(df, column)
    st.metric(label, "brak kolumny" if value is None else value)


def show_missing_demo_assets_info() -> None:
    """Render instructions for recreating missing stage 3 demo assets."""
    st.warning("Brakuje wymaganych plików demo w `app/demo_assets/`.")
    st.markdown(
        "Uruchom lokalnie `python scripts/03_system_demo.py`, a następnie skopiuj "
        "małe pliki CSV z `outputs/stage3_outputs/` do `app/demo_assets/`."
    )


def render_bar_chart(series: pd.Series, title: str, x_label: str, y_label: str) -> None:
    """Render a compact seaborn bar chart from a value-count series."""
    if series.empty:
        st.info(f"Brak danych dla wykresu: {title}.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=series.index.astype(str), y=series.values, ax=ax, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)
    plt.close(fig)


def plan_kpis(plan_df: pd.DataFrame) -> None:
    """Render basic KPIs for a selected recommendation plan."""
    day_count = safe_nunique(plan_df, "day_number")
    exercise_count = safe_nunique(plan_df, "exercise")
    avg_weight = (
        pd.to_numeric(plan_df["final_recommended_weight"], errors="coerce").mean()
        if "final_recommended_weight" in plan_df.columns
        else None
    )
    avg_rir = (
        pd.to_numeric(plan_df["target_rir"], errors="coerce").mean()
        if "target_rir" in plan_df.columns
        else None
    )
    history_used = (
        int(plan_df["history_available"].astype(str).str.lower().eq("true").sum())
        if "history_available" in plan_df.columns
        else None
    )
    most_common_adjustment = (
        plan_df["safety_adjustment"].mode().iat[0]
        if "safety_adjustment" in plan_df.columns and not plan_df["safety_adjustment"].dropna().empty
        else None
    )

    cols = st.columns(6)
    cols[0].metric("Liczba dni", "brak" if day_count is None else day_count)
    cols[1].metric("Liczba ćwiczeń", "brak" if exercise_count is None else exercise_count)
    cols[2].metric(
        "Śr. ciężar",
        "brak" if pd.isna(avg_weight) or avg_weight is None else f"{avg_weight:.1f} kg",
    )
    cols[3].metric("Śr. RIR", "brak" if pd.isna(avg_rir) or avg_rir is None else f"{avg_rir:.1f}")
    cols[4].metric("Historia", "brak" if history_used is None else history_used)
    cols[5].metric("Safety", most_common_adjustment or "brak")


def render_overview_tab(dataset_df: pd.DataFrame | None) -> None:
    st.title("Training AI Project")
    st.subheader("Projekt i implementacja systemu sztucznej inteligencji do analizy danych treningowych")

    st.markdown(
        "`Generator` → `Dataset` → `EDA` → `ML Model` → "
        "`Recommendation Engine` → `Streamlit Dashboard`"
    )

    st.markdown(
        """
        - **Etap 1: EDA** - analiza struktury danych, rozkładów i trendów treningowych.
        - **Etap 2: ML + recommender** - model regresyjny i hybrydowy system rekomendacyjny.
        - **Etap 3: system demo** - przykładowe scenariusze użytkowników i wygenerowane plany.
        - **Etap 4: dashboard** - warstwa wizualna prezentująca cały projekt w jednym miejscu.
        """
    )

    if dataset_df is None:
        st.info("KPI datasetu pojawią się po dodaniu `data/FINAL_ENGINE_V4.csv`.")
        return

    cols = st.columns(4)
    cols[0].metric("Rekordy", len(dataset_df))
    cols[1].metric("Użytkownicy", safe_nunique(dataset_df, "user_id") or 0)
    cols[2].metric("Sesje", safe_nunique(dataset_df, "session_id") or 0)
    cols[3].metric("Ćwiczenia", safe_nunique(dataset_df, "exercise") or 0)


def render_dataset_tab(dataset_df: pd.DataFrame | None) -> None:
    st.header("Dataset")

    if dataset_df is None:
        st.warning("Brakuje `data/FINAL_ENGINE_V4.csv`. Wygeneruj dataset lub dodaj plik do katalogu `data/`.")
        return

    st.info("Dataset jest syntetyczny i został wygenerowany przez moduł generatora danych.")
    st.dataframe(dataset_df.head(), use_container_width=True)

    st.subheader("Typy kolumn")
    dtypes_df = dataset_df.dtypes.astype(str).reset_index()
    dtypes_df.columns = ["column", "dtype"]
    st.dataframe(dtypes_df, use_container_width=True)

    if "date" in dataset_df.columns:
        dates = pd.to_datetime(dataset_df["date"], errors="coerce")
        if dates.notna().any():
            st.write(f"Zakres dat: **{dates.min().date()} - {dates.max().date()}**")
        else:
            st.warning("Kolumna `date` istnieje, ale nie udało się odczytać zakresu dat.")
    else:
        st.info("Brak kolumny `date` w datasencie.")

    st.subheader("Podstawowe statystyki")
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


def render_eda_tab(dataset_df: pd.DataFrame | None) -> None:
    st.header("EDA")

    if dataset_df is None:
        st.warning("Brakuje datasetu `data/FINAL_ENGINE_V4.csv`, więc sekcja EDA jest niedostępna.")
        return

    chart_columns = [
        ("level", "Rozkład poziomu zaawansowania", "Poziom"),
        ("split", "Rozkład splitów treningowych", "Split"),
        ("phase", "Rozkład faz treningowych", "Faza"),
    ]

    for column, title, x_label in chart_columns:
        if column in dataset_df.columns:
            render_bar_chart(dataset_df[column].value_counts(), title, x_label, "Liczba rekordów")
        else:
            st.info(f"Brak kolumny `{column}` - pomijam wykres.")

    if "exercise" in dataset_df.columns:
        render_bar_chart(
            dataset_df["exercise"].value_counts().head(10),
            "Top 10 ćwiczeń",
            "Ćwiczenie",
            "Liczba serii",
        )

    required_volume_columns = {"reps", "weight", "date"}
    if required_volume_columns.issubset(dataset_df.columns):
        volume_df = dataset_df.copy()
        volume_df["date"] = pd.to_datetime(volume_df["date"], errors="coerce")
        volume_df["volume"] = (
            pd.to_numeric(volume_df["reps"], errors="coerce")
            * pd.to_numeric(volume_df["weight"], errors="coerce")
        )
        monthly_volume = (
            volume_df.dropna(subset=["date"])
            .set_index("date")
            .resample("MS")["volume"]
            .sum()
            .rename("volume")
        )

        if not monthly_volume.empty:
            st.subheader("Trend całkowitej objętości po miesiącach")
            st.line_chart(monthly_volume)
        else:
            st.info("Nie udało się policzyć trendu objętości po miesiącach.")
    else:
        st.info("Trend objętości wymaga kolumn `date`, `reps` i `weight`.")


def render_ml_model_tab() -> None:
    st.header("ML Model")
    st.markdown(
        "Model regresyjny przewiduje sugerowany ciężar treningowy (`weight`) na podstawie "
        "profilu, ćwiczenia, parametrów serii i cech historycznych."
    )

    if model_exists():
        st.success("Model available locally")
    else:
        st.warning("Model `models/best_weight_prediction_model.joblib` nie jest dostępny lokalnie.")
        st.markdown(
            "Model nie jest commitowany do zwykłego Gita, bo jest duży. Można go wygenerować przez "
            "`python scripts/02_modeling_and_recommendation.py` albo pobrać z GitHub Release, jeśli będzie udostępniony."
        )

    metrics_candidates = [
        DEMO_ASSETS_DIR / "model_comparison_results.csv",
        DEMO_ASSETS_DIR / "model_sanity_check_metrics.csv",
        STAGE2_OUTPUTS_DIR / "model_comparison_results.csv",
        STAGE2_OUTPUTS_DIR / "group_evaluation_by_level.csv",
    ]
    shown_any = False
    for path in metrics_candidates:
        if path.exists():
            st.subheader(path.name)
            st.dataframe(pd.read_csv(path), use_container_width=True)
            shown_any = True

    if not shown_any:
        st.info("Brak plików metryk w `app/demo_assets/` lub `outputs/stage2_outputs/`. Dashboard pokazuje opis roli modelu.")


def render_recommendation_tab(demo_assets: dict[str, pd.DataFrame | None]) -> None:
    st.header("Recommendation Demo")

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
    plan_kpis(selected_plan)
    st.subheader("Plan treningowy")
    st.dataframe(selected_plan, use_container_width=True)


def render_safety_rules_tab() -> None:
    st.header("Safety Rules")
    st.markdown(
        "Model przewiduje ciężar, ale finalna rekomendacja jest korygowana przez reguły "
        "bezpieczeństwa zależne od poziomu, fazy, zmęczenia, RIR i wieku."
    )

    selected_scenario = st.session_state.get("selected_scenario", "beginner_female_hypertrophy")
    selected_plan = load_plan(selected_scenario)

    if selected_plan is None:
        show_missing_demo_assets_info()
    elif "safety_adjustment" not in selected_plan.columns:
        st.warning("Wybrany plan nie zawiera kolumny `safety_adjustment`.")
    else:
        st.subheader(f"Rozkład korekt bezpieczeństwa: {selected_scenario}")
        render_bar_chart(
            selected_plan["safety_adjustment"].value_counts(),
            "Safety adjustment",
            "Korekta",
            "Liczba ćwiczeń",
        )

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


def render_live_generator_tab() -> None:
    st.header("Live Generator")

    if not model_exists():
        st.warning("Live generator jest niedostępny, ponieważ brakuje lokalnego modelu `.joblib`.")
        st.markdown(
            "Wygeneruj model przez `python scripts/02_modeling_and_recommendation.py` albo pobierz "
            "`models/best_weight_prediction_model.joblib` z GitHub Release, jeśli będzie udostępniony."
        )
        return

    with st.form("live_generator_form"):
        col_a, col_b, col_c = st.columns(3)
        age = col_a.number_input("Wiek", min_value=16, max_value=90, value=30)
        sex = col_b.selectbox("Płeć", ["female", "male"])
        level = col_c.selectbox("Poziom", ["beginner", "intermediate", "advanced"])

        col_d, col_e = st.columns(2)
        phase = col_d.selectbox("Faza", ["hypertrophy", "strength", "deload"])
        days_per_week = col_e.number_input("Dni treningowe w tygodniu", min_value=2, max_value=6, value=3)

        submitted = st.form_submit_button("Pokaż status")

    if submitted:
        st.info(
            "Live generator will be connected to the Stage 3 recommendation logic in a later refactor."
        )
        st.json(
            {
                "age": age,
                "sex": sex,
                "level": level,
                "phase": phase,
                "days_per_week": days_per_week,
            }
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
        render_live_generator_tab()


if __name__ == "__main__":
    main()
