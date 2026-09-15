"""Create a reproducible shifted batch for demonstrating drift monitoring."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=Path("artifacts/model/reference.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/current.csv"))
    args = parser.parse_args()
    frame = pd.read_csv(args.reference).sample(frac=0.5, random_state=42).copy()
    rng = np.random.default_rng(42)
    frame["Amount"] = frame["Amount"] * rng.lognormal(0.2, 0.15, len(frame))
    frame["V1"] = frame["V1"] + 0.35
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
