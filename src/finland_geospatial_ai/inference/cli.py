"""Command-line entry point for georeferenced orthophoto inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from finland_geospatial_ai.inference import Predictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--confidence", type=Path)
    args = parser.parse_args()
    print(
        Predictor(args.model).predict(
            args.input, args.output, args.tile_size, args.overlap, args.confidence
        )
    )


if __name__ == "__main__":
    main()
