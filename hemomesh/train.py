"""Training entry point placeholder for the HemoMesh experiment harness."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a HemoMesh model from a config file.")
    parser.add_argument("--config", required=True, help="Path to a YAML training config.")
    args = parser.parse_args()
    raise NotImplementedError(
        f"Training loop is scaffolded but not implemented yet; received config {args.config}."
    )


if __name__ == "__main__":
    main()
