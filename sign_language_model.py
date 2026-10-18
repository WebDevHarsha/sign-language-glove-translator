
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Datasets" / "data"
MODEL_PATH = BASE_DIR / "sign_language_model.joblib"
SEQUENCE_LENGTH = 32
SENSOR_COLUMNS = (
    "thumb_mV",
    "index_mV",
    "middle_mV",
    "ring_mV",
    "little_mV",
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
)


def _features(samples: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Convert one variable-length sensor recording into a fixed-size vector."""
    if isinstance(samples, pd.DataFrame):
        missing = [column for column in SENSOR_COLUMNS if column not in samples.columns]
        if missing:
            raise ValueError(f"Recording is missing sensor columns: {', '.join(missing)}")
        values = samples.loc[:, SENSOR_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=np.float64
        )
    else:
        values = np.asarray(samples, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(SENSOR_COLUMNS):
            raise ValueError(
                f"Expected a 2D array with {len(SENSOR_COLUMNS)} sensor columns."
            )

    values = values[np.isfinite(values).all(axis=1)]
    if len(values) == 0:
        raise ValueError("Recording contains no complete numeric sensor samples.")

    source_positions = np.linspace(0.0, 1.0, num=len(values))
    target_positions = np.linspace(0.0, 1.0, num=SEQUENCE_LENGTH)
    resampled = np.column_stack(
        [
            np.interp(target_positions, source_positions, values[:, column])
            for column in range(values.shape[1])
        ]
    )
    return resampled.ravel()


def train(data_dir: Path = DATA_DIR, model_path: Path = MODEL_PATH) -> None:
    """Train on one CSV recording per labeled subdirectory and save the model."""
    labels_path = BASE_DIR / "labels.json"
    if labels_path.exists():
        labels = set(json.loads(labels_path.read_text(encoding="utf-8")))
    else:
        labels = None

    feature_rows: list[np.ndarray] = []
    targets: list[str] = []
    skipped_empty = 0

    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        if labels is not None and class_dir.name not in labels:
            continue

        recordings = sorted(class_dir.glob("*.csv"))
        if not recordings:
            continue

        for recording in recordings:
            frame = pd.read_csv(recording)
            if frame.empty:
                warnings.warn(f"Skipping empty recording: {recording}", stacklevel=1)
                skipped_empty += 1
                continue
            feature_rows.append(_features(frame))
            targets.append(class_dir.name)

    if not feature_rows:
        raise ValueError(f"No usable labeled recordings found in {data_dir}.")

    classifier = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
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
    samples: pd.DataFrame | np.ndarray, model_path: Path = MODEL_PATH
) -> tuple[str, float]:
    """Predict a sign from a DataFrame or array with the 15 sensor channels."""
    bundle: dict[str, Any] = joblib.load(model_path)
    features = _features(samples)
    classifier: RandomForestClassifier = bundle["classifier"]
    probabilities = classifier.predict_proba(features.reshape(1, -1))[0]
    best_index = int(np.argmax(probabilities))
    return str(classifier.classes_[best_index]), float(probabilities[best_index])


def predict_recording(recording_path: Path, model_path: Path = MODEL_PATH) -> tuple[str, float]:
    """Predict a sign from a sensor CSV recording."""
    return predict_samples(pd.read_csv(recording_path), model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and save the sign model.")
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
