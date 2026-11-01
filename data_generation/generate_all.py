"""Generate and inspect all RIO synthetic datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .common import DEFAULT_SEED
from .generate_intents import generate as generate_intents
from .generate_phishing import generate as generate_phishing
from .generate_transitions import generate as generate_transitions


def inspect_dataset(name: str, frame: pd.DataFrame, distribution_column: str) -> None:
    print(f"\n=== {name} ===")
    print(f"rows={len(frame)} columns={len(frame.columns)}")
    print("distribution:")
    print(frame[distribution_column].value_counts().to_string())
    print(f"missing_values={int(frame.isna().sum().sum())}")
    print(f"duplicate_rows={int(frame.duplicated().sum())}")
    print(f"template_families={frame['template_family'].nunique()}")
    print("representative_samples:")
    print(frame.head(3).to_string(index=False))


def main(output_dir: Path, seed: int) -> None:
    phishing = generate_phishing(output_dir, seed)
    intents = generate_intents(output_dir, seed)
    transitions = generate_transitions(output_dir, seed)
    inspect_dataset("Dataset A - phishing emails", phishing, "label_name")
    inspect_dataset("Dataset B - intent extraction", intents, "intent_label")
    inspect_dataset("Dataset C - intent transitions", transitions, "transition_label")
    print("\nthread_consistency=" + str(
        bool((transitions["thread_id"].nunique() == len(transitions)) and
             (transitions["previous_email_id"].nunique() == len(transitions)) and
             (transitions["current_email_id"].nunique() == len(transitions)))
    ))
    print("\nSynthetic data only: metrics from these datasets must not be presented as real-world performance.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    main(args.output_dir, args.seed)
