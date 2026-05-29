"""Per-session workout simulation.

The session layer converts a lifter's current state into set-level rows. It
samples the planned exercises, chooses reps and intensity, applies fatigue and
progression updates, and returns records ready for the dataset generator.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np

from config import (
    EXERCISE_PROFILES,
    PHASE_REP_MODIFIER,
    SimConfig,
    TrainingPhase,
)
from models import Lifter, adaptive_noise


def day_quality(cfg: SimConfig) -> float:
    """Return a session-level readiness multiplier around the lifter baseline."""
    if np.random.rand() < cfg.bad_day_prob:
        return float(np.random.uniform(0.85, 0.95))

    if np.random.rand() < cfg.great_day_prob:
        return float(np.random.uniform(1.02, 1.06))

    noise = np.random.normal(0.0, 0.02)
    return float(np.clip(1.0 + noise, 0.90, 1.08))


def reps_to_intensity(reps: float) -> float:
    """Map repetitions to an approximate intensity fraction."""
    return 1.0 / (1.0 + reps / 30.0)


def _simulate_set(
    lifter: Lifter,
    exercise: str,
    set_number: int,
    phase: TrainingPhase,
    dq: float,
    session_id: int,
    session_date: date,
) -> dict[str, Any]:
    """Simulate one set and mutate the lifter's fatigue and strength state."""

    profile = EXERCISE_PROFILES[exercise]
    rep_min, rep_max = profile["rep_range"]
    rir_lo, rir_hi = profile["rir_target"]

    # Repetition targets come from the exercise profile, then shift by phase.
    # Fatigue reduces output, while execution noise makes the rows less uniform.
    phase_mod = PHASE_REP_MODIFIER[phase]

    adj_min = max(1, int(rep_min * phase_mod))
    adj_max = max(adj_min + 1, int(rep_max * phase_mod))

    base_reps = np.random.uniform(adj_min, adj_max)
    fatigue_shift = lifter.fatigue * np.random.uniform(0.5, 1.5)
    execution_noise = np.random.normal(0.0, 0.6 * lifter.fatigue)
    actual_reps = int(
        np.clip(round(base_reps - fatigue_shift + execution_noise), adj_min, adj_max)
    )

    # Heavy lower-body movements are allowed a slightly different empirical
    # rep distribution so they do not look identical to machine/accessory work.
    if exercise == "Deadlift":
        actual_reps = int(np.clip(actual_reps + np.random.choice([0, 1, 1, 2]), 2, 8))

    if exercise == "Squat":
        actual_reps = int(
            np.clip(actual_reps + np.random.choice([0, 1, 1, 1, 2]), 3, 10)
        )

    base_rir = np.random.uniform(rir_lo, rir_hi)
    fatigue_effect = -lifter.fatigue * 0.35
    experience_effect = -lifter.level * 0.08
    actual_rir = float(
        np.clip(
            base_rir
            + fatigue_effect
            + experience_effect
            + adaptive_noise(0.45, lifter.fatigue),
            0.0,
            4.0,
        )
    )

    # Intensity is sampled independently from reps so the final data keeps both
    # signals useful for downstream analysis.
    exercise_base = np.random.normal(
        profile["intensity_center"],
        0.04 * (1.0 + lifter.fatigue * 0.7),
    )
    individual_bias = np.random.normal(lifter.form_state * 0.05, 0.02)
    exercise_base += individual_bias
    experience_bonus = lifter.level * 0.008
    fatigue_penalty = lifter.fatigue * 0.03
    day_effect = (dq - 1.0) * 0.10
    rir_effect = np.tanh(2.0 - actual_rir) * 0.04

    intensity = np.clip(
        exercise_base
        + experience_bonus
        - fatigue_penalty
        + day_effect
        + rir_effect,
        0.50,
        0.95,
    )

    injury_mod = 0.65 if exercise in lifter.injured_exercises else 1.0
    fatigue_weight_penalty = np.clip(
        1.0 - lifter.fatigue * 0.08 + np.random.normal(0.0, 0.01),
        0.68,
        1.05,
    )

    base_weight = lifter.effective_1rm(exercise) * intensity * injury_mod
    weight = float(
        np.clip(
            base_weight * fatigue_weight_penalty,
            *lifter.cfg.weight_clip,
        )
    )

    volume = actual_reps * weight
    lifter.apply_exercise_fatigue(exercise, volume)
    lifter.progress_strength(exercise, actual_reps, actual_rir)

    return {
        "user_id": lifter.id,
        "session_id": session_id,
        "date": session_date,
        "exercise": exercise,
        "set_number": set_number,
        "reps": actual_reps,
        "weight": round(weight, 2),
        "fatigue": round(lifter.fatigue, 3),
        "rir": round(actual_rir, 2),
        "level": lifter.level_name,
        "split": lifter.split,
        "phase": phase.value,
        "sex": lifter.profile.sex,
    }


class WorkoutSession:
    """Run one planned training session for a single lifter."""

    def __init__(
        self,
        lifter: Lifter,
        session_id: int,
        day_index: int,
        day_of_year: int,
        session_date: date,
        cfg: SimConfig,
    ) -> None:
        self.lifter = lifter
        self.session_id = session_id
        self.day_index = day_index
        self.day_of_year = day_of_year
        self.session_date = session_date
        self.cfg = cfg

    def run(self) -> list[dict[str, Any]]:
        """Execute the session and return generated set records."""

        l = self.lifter
        plan = l.current_plan()
        day_type = l.day_type_for(self.day_index)

        if day_type not in plan:
            return []

        l.tick_injury()
        l.maybe_trigger_injury()

        dq = day_quality(self.cfg)

        records: list[dict[str, Any]] = []

        for group in plan[day_type]:
            exercise = str(np.random.choice(group))
            records.extend(self._run_exercise(exercise, dq))

        l.training_days += 1
        l.session_count += 1

        l.update_level()
        l.commit_fatigue()
        l.recover()
        l.advance_phase()

        return records

    def _run_exercise(self, exercise: str, dq: float) -> list[dict[str, Any]]:
        """Generate all working sets for one exercise in the session."""

        l = self.lifter
        phase_bonus = {
            "hypertrophy": 1,
            "strength": 0,
            "deload": -1,
        }[l.phase.value]

        base_sets = 2 + l.level + phase_bonus
        num_sets = int(np.clip(base_sets + np.random.randint(0, 2), 1, 6))

        return [
            _simulate_set(
                lifter=l,
                exercise=exercise,
                set_number=i + 1,
                phase=l.phase,
                dq=dq,
                session_id=self.session_id,
                session_date=self.session_date,
            )
            for i in range(num_sets)
        ]
