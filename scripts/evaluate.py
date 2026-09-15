"""Print the frozen, pre-generated test metrics without recomputing them live."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/model/metrics.json"))
    args = parser.parse_args()
    print(json.dumps(json.loads(args.metrics.read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
