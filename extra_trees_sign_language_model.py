from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from sign_language_model import BASE_DIR, SENSOR_COLUMNS, SEQUENCE_LENGTH, _features


DATA_DIR = BASE_DIR / "Datasets" / "sign_data"
MODEL_PATH = BASE_DIR / "extra_trees_sign_language_model.joblib"


def train(data_dir: Path = DATA_DIR, model_path: Path = MODEL_PATH) -> None:
    """Train Extra Trees on CSV recordings in per-sign subdirectories."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    feature_rows: list[np.ndarray] = []
    targets: list[str] = []
    skipped_empty = 0

    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for recording in sorted(class_dir.glob("*.csv")):
            frame = pd.read_csv(recording)
            if frame.empty:
                warnings.warn(f"Skipping empty recording: {recording}", stacklevel=1)
                skipped_empty += 1
                continue
            feature_rows.append(_features(frame))
            targets.append(class_dir.name)

    if not feature_rows:
        raise ValueError(f"No usable labeled recordings found in {data_dir}.")

    classifier = ExtraTreesClassifier(
        n_estimators=300,
        criterion="gini",
        max_features="sqrt",
        bootstrap=False,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    classifier.fit(np.vstack(feature_rows), targets)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "classifier": classifier,
            "sensor_columns": SENSOR_COLUMNS,
            "sequence_length": SEQUENCE_LENGTH,
        },
        model_path,
    )

    print(f"Trained on {len(targets)} recordings across {len(classifier.classes_)} signs.")
    if skipped_empty:
        print(f"Skipped {skipped_empty} empty recording(s).")
    print(f"Saved model to: {model_path}")


def predict_samples(
    samples: pd.DataFrame | np.ndarray,
    model_path: Path = MODEL_PATH,
) -> tuple[str, float]:
    """Predict a sign from a DataFrame or array with the 15 sensor channels."""
    bundle: dict[str, Any] = joblib.load(model_path)
    features = _features(samples)
    classifier: ExtraTreesClassifier = bundle["classifier"]
    probabilities = classifier.predict_proba(features.reshape(1, -1))[0]
    best_index = int(np.argmax(probabilities))
    return str(classifier.classes_[best_index]), float(probabilities[best_index])


def predict_recording(
    recording_path: Path,
    model_path: Path = MODEL_PATH,
) -> tuple[str, float]:
    """Predict the sign in a sensor CSV recording."""
    return predict_samples(pd.read_csv(recording_path), model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and save the Extra Trees model.")
    train_parser.add_argument("--data", type=Path, default=DATA_DIR, help="Dataset directory.")
    train_parser.add_argument("--output", type=Path, default=MODEL_PATH, help="Model file path.")

    predict_parser = subparsers.add_parser("predict", help="Predict the sign in a sensor CSV.")
    predict_parser.add_argument("recording", type=Path, help="CSV recording to classify.")
    predict_parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Saved model path.")

    args = parser.parse_args()
    if args.command == "train":
        train(args.data, args.output)
    else:
        label, confidence = predict_recording(args.recording, args.model)
        print(f"Prediction: {label} (confidence: {confidence:.1%})")


if __name__ == "__main__":
    main()
