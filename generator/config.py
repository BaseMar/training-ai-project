"""Static configuration for the synthetic training simulation.

The values in this module define the training domain used by the generator:
exercise profiles, training phases, split templates, and global simulation
defaults. Runtime state belongs in ``models.py`` and per-session mechanics live
in ``session.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TrainingPhase(Enum):
    """Macro-cycle phases used to vary reps, intensity, and progression."""

    HYPERTROPHY = "hypertrophy"
    STRENGTH = "strength"
    DELOAD = "deload"


PHASE_CYCLE: list[TrainingPhase] = [
    TrainingPhase.HYPERTROPHY,
    TrainingPhase.STRENGTH,
    TrainingPhase.DELOAD,
]

# Session windows are intentionally expressed in sessions, not weeks. This lets
# the lifter's compliance determine how long each phase lasts in calendar time.
PHASE_DURATION_SESSIONS: dict[TrainingPhase, tuple[int, int]] = {
    TrainingPhase.HYPERTROPHY: (16, 28),  # about 4-7 weeks for frequent lifters
    TrainingPhase.STRENGTH: (12, 20),     # about 3-5 weeks
    TrainingPhase.DELOAD: (3, 5),         # roughly one lighter week
}

# Applied to the center of an exercise's preferred rep range before sampling.
PHASE_REP_MODIFIER: dict[TrainingPhase, float] = {
    TrainingPhase.HYPERTROPHY: 1.10,
    TrainingPhase.STRENGTH: 0.72,
    TrainingPhase.DELOAD: 1.15,
}

# Kept as a phase-level intensity tuning point for session simulation.
PHASE_INTENSITY_MODIFIER: dict[TrainingPhase, float] = {
    TrainingPhase.HYPERTROPHY: 0.96,
    TrainingPhase.STRENGTH: 1.05,
    TrainingPhase.DELOAD: 0.72,
}

# Fraction of the base progression rate available in each phase.
PHASE_PROGRESSION_RATE: dict[TrainingPhase, float] = {
    TrainingPhase.HYPERTROPHY: 1.00,
    TrainingPhase.STRENGTH: 0.75,
    TrainingPhase.DELOAD: 0.00,
}


EXERCISES: list[str] = [
    "Bench Press", "Incline Bench Press", "Squat", "Deadlift",
    "Barbell Row", "Lat Pulldown", "Seated Row",
    "Shoulder Press", "Lateral Raise", "Biceps Curl",
    "Triceps Pushdown", "Leg Press", "Leg Curl",
    "Calf Raise", "Rear Delt Row",
]

# Exercise profile fields:
# - mult: bodyweight-based multiplier for initial exercise 1RM
# - fatigue: systemic fatigue contribution from completed volume
# - type: coarse exercise category for downstream analytics
# - rep_range: preferred working-set repetition range
# - intensity_center: typical load anchor, independent of exact reps
# - rir_target: common reps-in-reserve target range
# - progression_rate: relative adaptation speed for this lift
EXERCISE_PROFILES: dict[str, dict[str, Any]] = {
    "Bench Press": {
        "mult": 1.00,
        "fatigue": 1.0,
        "type": "compound_medium",
        "rep_range": (5, 10),
        "intensity_center": 0.79,
        "rir_target": (1, 3),
        "progression_rate": 1.0,
    },
    "Incline Bench Press": {
        "mult": 0.85,
        "fatigue": 1.0,
        "type": "compound_medium",
        "rep_range": (6, 12),
        "intensity_center": 0.76,
        "rir_target": (1, 3),
        "progression_rate": 0.9,
    },
    "Squat": {
        "mult": 1.50,
        "fatigue": 1.2,
        "type": "compound_heavy",
        "rep_range": (4, 8),
        "intensity_center": 0.83,
        "rir_target": (1, 3),
        "progression_rate": 0.9,
    },
    "Deadlift": {
        "mult": 1.80,
        "fatigue": 1.3,
        "type": "compound_heavy",
        "rep_range": (3, 6),
        "intensity_center": 0.86,
        "rir_target": (1, 3),
        "progression_rate": 0.7,
    },
    "Barbell Row": {
        "mult": 1.10,
        "fatigue": 1.0,
        "type": "compound_medium",
        "rep_range": (5, 10),
        "intensity_center": 0.80,
        "rir_target": (1, 3),
        "progression_rate": 0.9,
    },
    "Lat Pulldown": {
        "mult": 0.80,
        "fatigue": 0.9,
        "type": "compound_light",
        "rep_range": (8, 13),
        "intensity_center": 0.73,
        "rir_target": (1, 2),
        "progression_rate": 0.8,
    },
    "Seated Row": {
        "mult": 0.85,
        "fatigue": 0.9,
        "type": "compound_light",
        "rep_range": (8, 13),
        "intensity_center": 0.73,
        "rir_target": (1, 2),
        "progression_rate": 0.8,
    },
    "Shoulder Press": {
        "mult": 0.65,
        "fatigue": 0.9,
        "type": "compound_light",
        "rep_range": (6, 12),
        "intensity_center": 0.75,
        "rir_target": (1, 3),
        "progression_rate": 0.8,
    },
    "Lateral Raise": {
        "mult": 0.30,
        "fatigue": 0.8,
        "type": "accessory",
        "rep_range": (12, 20),
        "intensity_center": 0.63,
        "rir_target": (0, 2),
        "progression_rate": 0.6,
    },
    "Biceps Curl": {
        "mult": 0.35,
        "fatigue": 0.7,
        "type": "accessory",
        "rep_range": (10, 15),
        "intensity_center": 0.67,
        "rir_target": (0, 2),
        "progression_rate": 0.7,
    },
    "Triceps Pushdown": {
        "mult": 0.40,
        "fatigue": 0.7,
        "type": "accessory",
        "rep_range": (10, 15),
        "intensity_center": 0.67,
        "rir_target": (0, 2),
        "progression_rate": 0.7,
    },
    "Leg Press": {
        "mult": 1.60,
        "fatigue": 1.1,
        "type": "compound_light",
        "rep_range": (8, 15),
        "intensity_center": 0.74,
        "rir_target": (1, 2),
        "progression_rate": 0.8,
    },
    "Leg Curl": {
        "mult": 0.55,
        "fatigue": 0.8,
        "type": "accessory",
        "rep_range": (10, 15),
        "intensity_center": 0.66,
        "rir_target": (0, 2),
        "progression_rate": 0.7,
    },
    "Calf Raise": {
        "mult": 1.00,
        "fatigue": 0.9,
        "type": "high_rep",
        "rep_range": (12, 20),
        "intensity_center": 0.62,
        "rir_target": (0, 2),
        "progression_rate": 0.6,
    },
    "Rear Delt Row": {
        "mult": 0.40,
        "fatigue": 0.7,
        "type": "accessory",
        "rep_range": (12, 18),
        "intensity_center": 0.64,
        "rir_target": (0, 2),
        "progression_rate": 0.6,
    },
}


LEVELS: list[str] = ["beginner", "intermediate", "advanced"]
SPLITS: list[str] = ["ppl", "upper_lower", "fbw"]

# Plan layout:
# split -> level index -> training day type -> alternative exercise groups.
# A group with multiple exercises means the session can choose one variant.
Plans = dict[int, dict[str, list[list[str]]]]

PLANS: dict[str, Plans] = {
    "ppl": {
        0: {
            "push": [["Bench Press"], ["Shoulder Press"], ["Triceps Pushdown"]],
            "pull": [["Lat Pulldown"], ["Seated Row"], ["Biceps Curl"]],
            "legs": [["Squat"], ["Leg Press"], ["Calf Raise"]],
        },
        1: {
            "push": [
                ["Bench Press", "Incline Bench Press"],
                ["Shoulder Press"], ["Lateral Raise"], ["Triceps Pushdown"],
            ],
            "pull": [
                ["Barbell Row", "Seated Row"],
                ["Lat Pulldown"], ["Rear Delt Row"], ["Biceps Curl"],
            ],
            "legs": [
                ["Squat"], ["Deadlift"], ["Leg Press"], ["Leg Curl"], ["Calf Raise"],
            ],
        },
        2: {
            "push": [
                ["Bench Press", "Incline Bench Press"],
                ["Shoulder Press"], ["Lateral Raise"], ["Triceps Pushdown"],
            ],
            "pull": [
                ["Barbell Row"], ["Lat Pulldown"], ["Seated Row"],
                ["Rear Delt Row"], ["Biceps Curl"],
            ],
            "legs": [
                ["Squat"], ["Deadlift"], ["Leg Press"], ["Leg Curl"], ["Calf Raise"],
            ],
        },
    },
    "upper_lower": {
        0: {
            "upper": [
                ["Bench Press"], ["Seated Row"], ["Shoulder Press"], ["Biceps Curl"],
            ],
            "lower": [["Squat"], ["Leg Press"], ["Calf Raise"]],
        },
        1: {
            "upper": [
                ["Bench Press", "Incline Bench Press"],
                ["Barbell Row"], ["Lat Pulldown"], ["Shoulder Press"], ["Biceps Curl"],
            ],
            "lower": [
                ["Squat"], ["Deadlift"], ["Leg Press"], ["Leg Curl"], ["Calf Raise"],
            ],
        },
        2: {
            "upper": [
                ["Bench Press", "Incline Bench Press"],
                ["Barbell Row"], ["Lat Pulldown"], ["Seated Row"],
                ["Shoulder Press"], ["Lateral Raise"],
            ],
            "lower": [
                ["Squat"], ["Deadlift"], ["Leg Press"], ["Leg Curl"], ["Calf Raise"],
            ],
        },
    },
    "fbw": {
        0: {
            "fbw": [
                ["Squat"], ["Bench Press"], ["Seated Row"], ["Shoulder Press"],
            ],
        },
        1: {
            "fbw": [
                ["Squat"], ["Bench Press", "Incline Bench Press"],
                ["Barbell Row"], ["Shoulder Press"], ["Leg Curl"],
            ],
        },
        2: {
            "fbw": [
                ["Squat"], ["Deadlift"],
                ["Bench Press", "Incline Bench Press"],
                ["Barbell Row"], ["Lat Pulldown"], ["Shoulder Press"],
            ],
        },
    },
}


@dataclass(frozen=True)
class SimConfig:
    """Runtime settings shared by all generator components."""

    # Population and date range.
    users: int = 100
    years: int = 3
    start_date: datetime = field(default_factory=lambda: datetime(2022, 1, 1))

    # Fatigue model bounds.
    fatigue_decay: float = 0.96
    fatigue_cap: float = 2.0

    # Level changes are based on average 1RM relative to the lifter baseline.
    level_up_ratio: float = 1.35
    level_down_ratio: float = 0.80

    # Random events and day-to-day performance variability.
    bad_day_prob: float = 0.07
    great_day_prob: float = 0.04
    injury_prob_base: float = 0.0018
    seasonal_amplitude: float = 0.025

    # Approximate per-set strength gain for a beginner under full stimulus.
    base_progression_rate: float = 0.012

    # Hard output limits keep generated records within plausible CSV ranges.
    weight_clip: tuple[float, float] = (2.5, 300.0)
    reps_clip: tuple[int, int] = (1, 25)

    output_path: str = "data/FINAL_ENGINE_V4.csv"
