"""Command-line entry point for the synthetic gym dataset generator.

Examples
--------
Run with defaults:
    python main.py

Generate a smaller dataset:
    python main.py --users 50 --years 1

Inspect generated data without writing a CSV:
    python main.py --no-save
"""

from __future__ import annotations

import argparse

from config import SimConfig
from generator import DatasetGenerator


def parse_args() -> argparse.Namespace:
    """Parse CLI options used to build ``SimConfig``."""
    parser = argparse.ArgumentParser(
        description="Generate a synthetic set-level gym training dataset."
    )
    parser.add_argument(
        "--users",
        type=int,
        default=100,
        help="Number of synthetic lifters to simulate.",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=3,
        help="Number of full calendar years to simulate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/FINAL_ENGINE_V4.csv",
        help="Path where the generated CSV should be saved.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run the simulation without writing a CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    """Build the simulation config, run the generator, and print a preview."""
    args = parse_args()

    cfg = SimConfig(
        users=args.users,
        years=args.years,
        output_path=args.output,
    )

    print(f"Generating dataset: {cfg.users} users x {cfg.years} years...")
    df = DatasetGenerator(cfg).run(save=not args.no_save)

    print(f"Shape: {df.shape}")
    print(df.head())


if __name__ == "__main__":
    main()
