"""Evaluate the 14-channel Extra Trees classifier on held-out recordings."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from extra_trees_no_middle_model import DATA_DIR, _features, _new_classifier


def evaluate(data_dir: Path = DATA_DIR, test_fraction: float = 0.2) -> None:
    """Train an evaluation copy and report metrics on per-class held-out data."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    train_features: list[np.ndarray] = []
    train_labels: list[str] = []
    test_features: list[np.ndarray] = []
    test_labels: list[str] = []
    rng = np.random.default_rng(42)

    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        examples: list[np.ndarray] = []
        for recording in sorted(class_dir.glob("*.csv")):
            frame = pd.read_csv(recording)
            if frame.empty:
                warnings.warn(f"Skipping empty recording: {recording}", stacklevel=1)
                continue
            examples.append(_features(frame))

        if not examples:
            continue

        order = rng.permutation(len(examples))
        test_count = min(
            len(examples) - 1,
            max(1, round(len(examples) * test_fraction)),
        )
        test_indices = set(order[:test_count])

        for index, features in enumerate(examples):
            if index in test_indices:
                test_features.append(features)
                test_labels.append(class_dir.name)
            else:
                train_features.append(features)
                train_labels.append(class_dir.name)

    if not test_features:
        raise ValueError(f"No held-out recordings available in {data_dir}.")
    if not train_features:
        raise ValueError(f"No training recordings available in {data_dir}.")

    classifier = _new_classifier()
    classifier.fit(np.vstack(train_features), train_labels)
    predictions = classifier.predict(np.vstack(test_features))

    print(f"Training recordings: {len(train_labels)}")
    print(f"Held-out recordings: {len(test_labels)}")
    print(f"Signs evaluated: {len(set(test_labels))}")
    print("Sensor features per recording: 448 (14 channels x 32 time points)")
    print(f"Accuracy: {accuracy_score(test_labels, predictions):.1%}")
    print(classification_report(test_labels, predictions, zero_division=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="Dataset directory.")
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of each sign's recordings held out for testing.",
    )
    args = parser.parse_args()
    evaluate(args.data, args.test_fraction)


if __name__ == "__main__":
    main()
