"""Evaluate sign recognition on recordings held out from training."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from sign_language_model import BASE_DIR, DATA_DIR, _features


def evaluate(data_dir: Path = DATA_DIR, test_fraction: float = 0.2) -> None:
    """Train an evaluation copy and report metrics on unseen recordings."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    labels_path = BASE_DIR / "labels.json"
    labels = set(json.loads(labels_path.read_text(encoding="utf-8"))) if labels_path.exists() else None

    train_features: list[np.ndarray] = []
    train_labels: list[str] = []
    test_features: list[np.ndarray] = []
    test_labels: list[str] = []
    rng = np.random.default_rng(42)

    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        if labels is not None and class_dir.name not in labels:
            continue

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
        raise ValueError(f"No labeled recordings available for testing in {data_dir}.")

    classifier = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    classifier.fit(np.vstack(train_features), train_labels)
    predictions = classifier.predict(np.vstack(test_features))

    print(f"Training recordings: {len(train_labels)}")
    print(f"Held-out recordings: {len(test_labels)}")
    print(f"Signs evaluated: {len(set(test_labels))}")
    print(f"Accuracy: {accuracy_score(test_labels, predictions):.1%}")
    print(
        classification_report(
            test_labels,
            predictions,
            zero_division=0,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="Dataset directory.")
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of each sign's recordings held out for testing (default: 0.2).",
    )
    args = parser.parse_args()
    evaluate(args.data, args.test_fraction)


if __name__ == "__main__":
    main()
