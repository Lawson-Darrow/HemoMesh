"""Tuning entry point placeholder for resumable experiment sweeps."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a HemoMesh tuning sweep.")
    parser.add_argument("--config", required=True, help="Path to a YAML sweep config.")
    args = parser.parse_args()
    raise NotImplementedError(
        f"Tuning loop is scaffolded but not implemented yet; received config {args.config}."
    )


if __name__ == "__main__":
    main()
