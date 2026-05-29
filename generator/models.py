"""Domain models for the synthetic lifter simulation.

The generator keeps long-lived athlete state here and leaves session-specific
sampling to ``session.py``. A ``Lifter`` owns profile traits, fatigue, injury
status, training phase, and per-exercise 1RM estimates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from config import (
    EXERCISES,
    EXERCISE_PROFILES,
    LEVELS,
    PHASE_CYCLE,
    PHASE_DURATION_SESSIONS,
    PHASE_PROGRESSION_RATE,
    PLANS,
    SPLITS,
    SimConfig,
    TrainingPhase,
)


def _safe(x: float) -> float:
    """Return ``0.0`` for non-finite values."""
    return 0.0 if (math.isnan(x) or math.isinf(x)) else x


@dataclass(frozen=True)
class UserProfile:
    """Immutable biological and behavioural profile for one simulated lifter."""

    sex: str
    bodyweight: float
    training_age_months: int
    recovery_score: float
    compliance: float
    adaptation_rate: float

    @classmethod
    def generate(cls) -> "UserProfile":
        """Sample a realistic but intentionally broad user profile."""
        sex = np.random.choice(["male", "female"], p=[0.68, 0.32])
        bodyweight = float(
            np.clip(np.random.normal(78 if sex == "male" else 63, 11), 45, 135)
        )
        training_age = max(0, int(np.random.lognormal(2.5, 1.0)))
        recovery = float(np.clip(np.random.beta(3, 2), 0.20, 1.0))
        compliance = float(np.clip(np.random.beta(4, 3), 0.25, 0.95))
        adaptation = float(np.clip(np.random.lognormal(-0.10, 0.35), 0.40, 2.0))
        return cls(sex, bodyweight, training_age, recovery, compliance, adaptation)

    @property
    def sex_multiplier(self) -> float:
        """Strength baseline multiplier used for initial 1RM estimates."""
        return 1.00 if self.sex == "male" else 0.62


@dataclass
class ExerciseState:
    """Mutable state tracked separately for each exercise."""

    strength: float

    def __post_init__(self) -> None:
        """Keep generated 1RM estimates above a minimal plausible floor."""
        self.strength = max(5.0, self.strength)


class Lifter:
    """Full simulation state for one athlete.

    Strength is stored as per-exercise 1RM estimates. The session layer reads
    those estimates to choose working weights and calls back into this object to
    apply fatigue, adaptation, phase changes, and injury state.
    """

    def __init__(self, user_id: int, cfg: SimConfig) -> None:
        self.id = user_id
        self.cfg = cfg
        self.profile = UserProfile.generate()

        self.level = self._level_from_training_age(self.profile.training_age_months)
        self.split = str(np.random.choice(SPLITS, p=[0.50, 0.30, 0.20]))

        level_mult = {0: 0.55, 1: 0.80, 2: 1.15}[self.level]
        self.baseline_1rm = (
            self.profile.bodyweight * self.profile.sex_multiplier * level_mult
        )

        self.fatigue = float(np.random.uniform(0.05, 0.30))
        self.fatigue_acute = 0.0
        self.fatigue_chronic = 0.0

        self.training_days = 0
        self.session_count = 0

        self._phase_index = 0
        self.phase: TrainingPhase = PHASE_CYCLE[0]
        self.phase_sessions_remaining = int(
            np.random.randint(*PHASE_DURATION_SESSIONS[self.phase])
        )

        self.injury_sessions_remaining = 0
        self.injured_exercises: set[str] = set()

        self.exercises: dict[str, ExerciseState] = self._init_exercises()

        # Readiness and life-noise state lets identical profiles diverge over
        # time without hard-coding deterministic progress paths.
        self.form_state = float(np.random.normal(0.0, 0.05))
        self.fitness_trend = float(np.random.normal(0.0, 0.01))
        self.life_noise = float(np.random.uniform(-0.02, 0.02))
        self.fatigue_history: list[float] = []
        self.recent_stimulus: list[float] = []

        # Individual response parameters alter fatigue cost and progression.
        self.fatigue_sensitivity = float(
            np.clip(np.random.normal(1.0, 0.25), 0.5, 1.8)
        )
        self.progression_ceiling = float(np.random.normal(1.0, 0.12))
        self.plateau_resistance = float(np.clip(np.random.beta(2, 5), 0.15, 0.95))
        self.consistency_drift = float(np.random.normal(0.0, 0.01))
        self.neural_efficiency = float(np.clip(np.random.normal(1.0, 0.15), 0.7, 1.4))

        # Homeostasis state smooths long-term adaptation and rebound effects.
        self.adaptation_state = 0.0
        self.recovery_debt = 0.0
        self.supercomp_window = 0
        self.performance_baseline = 1.0

    @staticmethod
    def _level_from_training_age(months: int) -> int:
        """Map prior training age to an initial level with overlap."""
        if months < 12:
            probs = [0.80, 0.18, 0.02]
        elif months < 36:
            probs = [0.25, 0.55, 0.20]
        else:
            probs = [0.05, 0.40, 0.55]
        return int(np.random.choice([0, 1, 2], p=probs))

    def _init_exercises(self) -> dict[str, ExerciseState]:
        """Create initial per-exercise 1RM estimates from the lifter baseline."""
        base = self.baseline_1rm
        return {
            ex: ExerciseState(
                strength=float(
                    np.random.normal(
                        base * EXERCISE_PROFILES[ex]["mult"],
                        base * 0.08,
                    )
                )
            )
            for ex in EXERCISES
        }

    @property
    def day_types(self) -> list[str]:
        """Ordered training-day labels for the lifter's selected split."""
        return {
            "ppl": ["push", "pull", "legs"],
            "upper_lower": ["upper", "lower"],
            "fbw": ["fbw"],
        }[self.split]

    def day_type_for(self, day_index: int) -> str:
        """Return the scheduled day type for a calendar offset."""
        return self.day_types[day_index % len(self.day_types)]

    def current_plan(self) -> dict[str, list[list[str]]]:
        """Return the active split template for the lifter's current level."""
        return PLANS[self.split][self.level]

    def apply_exercise_fatigue(self, exercise: str, volume: float) -> None:
        """Add acute fatigue from one completed set."""
        base_strength = self.exercises[exercise].strength + 1e-6
        stimulus = np.log1p(volume / base_strength)
        fatigue_mult = EXERCISE_PROFILES[exercise]["fatigue"]
        acute = float(np.clip(stimulus * fatigue_mult * 0.08, 0.0, 0.25))
        self.fatigue_acute = _safe(self.fatigue_acute + acute)

    def commit_fatigue(self) -> None:
        """Blend acute session fatigue into chronic fatigue memory."""
        self.fatigue_chronic = self.fatigue_chronic * 0.88 + self.fatigue_acute * 0.16
        self.fatigue_acute *= 0.28
        raw = 0.58 * self.fatigue_chronic + 0.22 * self.fatigue_acute
        noise = float(np.random.normal(0.0, 0.015))
        self.fatigue = _safe(float(np.clip(raw + noise, 0.0, self.cfg.fatigue_cap)))
        self.update_chaos_state()

        self.fatigue_history.append(self.fatigue)
        self.update_adaptation_system()

        # Slow drift creates durable individual differences across long runs.
        self.consistency_drift += np.random.normal(0.0, 0.0005)
        self.consistency_drift *= 0.9995
        self.form_state += self.consistency_drift

        if len(self.fatigue_history) > 14:
            self.fatigue_history.pop(0)

    def recover(self) -> None:
        """Apply passive recovery at the end of a completed session."""
        stress = self.fatigue + 0.5 * self.fatigue_acute
        base = 0.03 + 0.025 * (2 - self.level) + 0.02 * self.profile.recovery_score
        overreach_penalty = 1.0 + min(0.45, self.phase_sessions_remaining * 0.018)
        recovery = float(
            np.clip(
                base / ((1.0 + stress) * overreach_penalty)
                + np.random.normal(0.0, 0.004),
                0.005,
                0.14,
            )
        )
        self.fatigue = _safe(self.fatigue * (1.0 - recovery))

    def progress_strength(self, exercise: str, reps: int, rir: float) -> None:
        """Update an exercise's 1RM estimate from one training stimulus."""
        s = self.exercises[exercise]

        proximity = float(np.clip(1.0 / (1.0 + rir * 0.40), 0.25, 1.0))

        phase_mult = PHASE_PROGRESSION_RATE[self.phase]
        if phase_mult == 0.0:
            return

        level_decay = max(0.25, 1.0 - self.level * 0.33)
        recovery_mod = float(np.clip(1.0 - self.fatigue * 0.14, 0.45, 1.0))
        ex_rate = EXERCISE_PROFILES[exercise]["progression_rate"]

        plateau_prob = 0.04 + self.level * 0.02 + self.fatigue * 0.03
        plateau_prob = float(np.clip(plateau_prob, 0.03, 0.18))

        base_gain = (
            self.cfg.base_progression_rate
            * proximity
            * phase_mult
            * level_decay
            * self.profile.adaptation_rate
            * recovery_mod
            * ex_rate
        )

        base_gain *= self.progression_ceiling
        base_gain *= np.exp(-self.fatigue * 0.8 * self.fatigue_sensitivity)

        # Plateau pressure softens gains instead of forcing abrupt regressions.
        plateau_pressure = (
            self.fatigue * 0.25
            + self.level * 0.15
            + (1.0 - self.plateau_resistance)
        )

        if np.random.rand() < plateau_pressure:
            base_gain *= np.random.uniform(0.6, 0.95)

        if len(self.recent_stimulus) >= 8:
            recent_var = np.std(self.recent_stimulus)
            recent_mean = np.mean(self.recent_stimulus)

            if recent_mean < 0.0002 and recent_var < 0.0005:
                if np.random.rand() < (0.25 * (1.0 - self.plateau_resistance)):
                    base_gain *= np.random.uniform(0.7, 0.95)

        homeostasis_factor = np.tanh(self.adaptation_state * 2.5)
        gain = base_gain * (1.0 + homeostasis_factor * 0.25)
        gain *= 1.0 + self.recovery_debt * 0.02

        if np.random.rand() < plateau_prob:
            gain *= np.random.uniform(0.75, 0.95)

        if self.fatigue > 1.80 and np.random.rand() < 0.08:
            gain *= 0.85

        self.recent_stimulus.append(gain)

        if len(self.recent_stimulus) > 10:
            self.recent_stimulus.pop(0)

        trend_signal = float(np.mean(self.recent_stimulus)) if self.recent_stimulus else gain
        effective_gain = 0.7 * gain + 0.3 * trend_signal

        if len(self.recent_stimulus) >= 5:
            inertia = np.std(self.recent_stimulus)
            effective_gain *= 1.0 - min(inertia * 0.10, 0.18)

        s.strength = float(np.clip(s.strength * (1.0 + effective_gain), 1.0, 600.0))

    def effective_1rm(self, exercise: str) -> float:
        """Return the lifter's day-specific usable 1RM for an exercise."""

        fatigue_state = self.fatigue_pressure()
        fatigue_penalty = (
            (fatigue_state ** 1.6)
            * 0.22
            * self.fatigue_sensitivity
        )

        neural_drop = 1.0 - (self.form_state * 0.08)
        chaos = (
            1.0
            + 0.35 * self.form_state
            + 0.25 * self.life_noise
            + np.random.normal(0.0, 0.015)
        )

        effective = (
            self.exercises[exercise].strength
            * chaos
            * neural_drop
            * (1.0 - fatigue_penalty)
            * self.neural_efficiency
        )

        supercomp = 1.0
        if self.supercomp_window > 0:
            supercomp += 0.03 * (self.recovery_debt / 2.0)

        self.adaptation_state = (
            self.adaptation_state * 0.995
            + (self.recovery_debt - self.fatigue) * 0.002
        )
        adaptation_multiplier = 1.0 + np.tanh(self.adaptation_state * 2.0) * 0.08

        return float(max(1.0, effective * supercomp * adaptation_multiplier))

    def update_level(self) -> None:
        """Promote or demote the lifter when average strength drifts enough."""
        avg_1rm = float(np.mean([s.strength for s in self.exercises.values()]))
        ratio = avg_1rm / (self.baseline_1rm + 1e-6)

        if ratio > self.cfg.level_up_ratio and self.level < 2:
            self.level += 1
            self.baseline_1rm *= 1.10
            for s in self.exercises.values():
                s.strength *= 1.02

        elif ratio < self.cfg.level_down_ratio and self.level > 0:
            self.level -= 1
            self.baseline_1rm *= 0.95

    def advance_phase(self) -> None:
        """Advance the macro-cycle when the current phase has enough sessions."""
        self.phase_sessions_remaining -= 1
        if self.phase_sessions_remaining <= 0:
            self._phase_index = (self._phase_index + 1) % len(PHASE_CYCLE)
            self.phase = PHASE_CYCLE[self._phase_index]
            self.phase_sessions_remaining = int(
                np.random.randint(*PHASE_DURATION_SESSIONS[self.phase])
            )

    def tick_injury(self) -> None:
        """Reduce active injury duration by one session and clear healed state."""
        if self.injury_sessions_remaining > 0:
            self.injury_sessions_remaining -= 1
            if self.injury_sessions_remaining == 0:
                self.injured_exercises.clear()

    def maybe_trigger_injury(self) -> None:
        """Randomly assign a minor injury when the lifter is currently healthy."""
        if self.injury_sessions_remaining > 0:
            return
        prob = self.cfg.injury_prob_base * (1.0 + self.fatigue * 0.6)
        if np.random.rand() < prob:
            n_affected = np.random.randint(1, 3)
            self.injured_exercises = set(
                np.random.choice(EXERCISES, n_affected, replace=False).tolist()
            )
            self.injury_sessions_remaining = int(np.random.randint(7, 22))

    @property
    def level_name(self) -> str:
        """Human-readable label for the current numeric level."""
        return LEVELS[self.level]

    def __repr__(self) -> str:
        return (
            f"Lifter(id={self.id}, {self.profile.sex}, {self.profile.bodyweight:.0f}kg, "
            f"level={self.level_name}, split={self.split}, "
            f"phase={self.phase.value}, fatigue={self.fatigue:.2f})"
        )

    def update_chaos_state(self) -> None:
        """Update readiness drift that is not directly caused by training load."""

        self.fitness_trend += np.random.normal(0.0, 0.003)
        self.fitness_trend *= 0.995

        self.form_state = float(
            np.clip(
                self.form_state * 0.8
                + self.fitness_trend
                + np.random.normal(0.0, 0.03),
                -0.25,
                0.25,
            )
        )

        self.form_state = float(np.clip(self.form_state, -0.25, 0.25))
        self.fitness_trend = float(np.clip(self.fitness_trend, -0.15, 0.15))

    def fatigue_pressure(self) -> float:
        """Return rolling fatigue memory, falling back to current fatigue."""
        if not self.fatigue_history:
            return self.fatigue

        return float(np.mean(self.fatigue_history))

    def update_adaptation_system(self) -> None:
        """Convert accumulated fatigue into recovery debt and rebound windows."""

        fatigue = self.fatigue_pressure()

        self.recovery_debt += fatigue * 0.03
        self.recovery_debt *= 0.985

        recovery_power = self.profile.recovery_score * (2.0 - self.level * 0.3)

        if self.fatigue < 0.9:
            self.recovery_debt -= recovery_power * 0.015

        self.recovery_debt = float(np.clip(self.recovery_debt, 0.0, 3.0))

        if self.recovery_debt > 1.2:
            self.supercomp_window = np.random.randint(2, 5)

        if self.supercomp_window > 0:
            self.supercomp_window -= 1


def adaptive_noise(scale: float, fatigue: float, rng=np.random) -> float:
    """Sample noise that grows as fatigue makes performance less predictable."""
    return float(rng.normal(0.0, scale * (1.0 + fatigue * 0.85)))
