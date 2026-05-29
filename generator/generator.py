"""High-level orchestration for the synthetic training dataset generator.

This module intentionally contains very little physiology logic. Its job is to
create lifters, move them through calendar days, collect set-level records, and
turn those records into a CSV-friendly ``pandas.DataFrame``.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from config import SimConfig
from models import Lifter
from session import WorkoutSession


class DatasetGenerator:
    """Generate set-level workout history for a synthetic population.

    Parameters
    ----------
    cfg:
        Optional simulation settings. When omitted, the default ``SimConfig`` is
        used so the generator remains convenient for quick local runs.
    """

    def __init__(self, cfg: SimConfig | None = None) -> None:
        self.cfg = cfg or SimConfig()

    def run(self, save: bool = True) -> pd.DataFrame:
        """Run the simulation and return the generated records.

        Parameters
        ----------
        save:
            When ``True``, persist the generated dataset to ``cfg.output_path``.

        Returns
        -------
        pandas.DataFrame
            One row per completed exercise set.
        """
        records = list(self._generate_records())
        df = self._build_dataframe(records)

        if save:
            self._save(df)

        return df

    def _generate_records(self) -> Iterator[dict]:
        """Yield set records for every compliant training day."""
        session_id = 1
        total_days = self.cfg.years * 365

        for user_id in range(self.cfg.users):
            lifter = Lifter(user_id, self.cfg)

            for day_offset in range(total_days):
                # Compliance is sampled daily. A skipped day produces no session
                # and therefore no rows in the final dataset.
                if np.random.rand() > lifter.profile.compliance:
                    continue

                session_date = self.cfg.start_date + timedelta(days=day_offset)

                session = WorkoutSession(
                    lifter=lifter,
                    session_id=session_id,
                    day_index=day_offset,
                    day_of_year=day_offset % 365,
                    session_date=session_date,
                    cfg=self.cfg,
                )
                yield from session.run()
                session_id += 1

    def _build_dataframe(self, records: list[dict]) -> pd.DataFrame:
        """Create the output frame and enforce configured value bounds."""
        df = pd.DataFrame(records)
        df["weight"] = df["weight"].clip(*self.cfg.weight_clip)
        df["reps"] = df["reps"].clip(*self.cfg.reps_clip)
        return df

    def _save(self, df: pd.DataFrame) -> None:
        """Write the dataset to disk, creating parent directories as needed."""
        output_path = Path(self.cfg.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df):,} rows -> {output_path}")
