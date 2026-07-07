"""Plotting entry point placeholder for saved experiment artifacts."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report figures from saved results.")
    parser.add_argument("--experiments", default="results/experiments.csv")
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()
    raise NotImplementedError(
        "Plot generation is scaffolded but not implemented yet; "
        f"received {args.experiments} and {args.output_dir}."
    )


if __name__ == "__main__":
    main()
