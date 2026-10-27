"""Train and use an Extra Trees sign classifier without the middle-finger sensor."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Datasets" / "sign_data"
MODEL_PATH = BASE_DIR / "extra_trees_no_middle_model.joblib"
SEQUENCE_LENGTH = 32
SENSOR_COLUMNS = (
    "thumb_mV",
    "index_mV",
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
    """Convert one recording into 448 features, excluding middle_mV."""
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


def _new_classifier() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=300,
        criterion="gini",
        max_features="sqrt",
        bootstrap=False,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )


def train(data_dir: Path = DATA_DIR, model_path: Path = MODEL_PATH) -> None:
    """Train using every CSV recording in each sign-labeled subdirectory."""
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

    classifier = _new_classifier()
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
    print(f"Sensor features per recording: {len(SENSOR_COLUMNS) * SEQUENCE_LENGTH}")
    if skipped_empty:
        print(f"Skipped {skipped_empty} empty recording(s).")
    print(f"Saved model to: {model_path}")


def load_classifier(model_path: Path = MODEL_PATH) -> ExtraTreesClassifier:
    """Load and validate a model trained with this 14-channel feature schema."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}; train it first.")

    bundle: dict[str, Any] = joblib.load(model_path)
    if tuple(bundle.get("sensor_columns", ())) != SENSOR_COLUMNS:
        raise ValueError(
            f"Model sensor schema does not match the {len(SENSOR_COLUMNS)}-channel "
            "schema that excludes middle_mV."
        )
    classifier = bundle.get("classifier")
    if not isinstance(classifier, ExtraTreesClassifier):
        raise ValueError("Model bundle does not contain an ExtraTreesClassifier.")
    expected_features = len(SENSOR_COLUMNS) * SEQUENCE_LENGTH
    if classifier.n_features_in_ != expected_features:
        raise ValueError(
            f"Model expects {classifier.n_features_in_} features; expected {expected_features}."
        )
    return classifier


def predict_samples(
    samples: pd.DataFrame | np.ndarray,
    model_path: Path = MODEL_PATH,
) -> tuple[str, float]:
    """Predict a sign from a recording with the 14 required sensor channels."""
    classifier = load_classifier(model_path)
    features = _features(samples)
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

    train_parser = subparsers.add_parser("train", help="Train and save the 14-channel model.")
    train_parser.add_argument("--data", type=Path, default=DATA_DIR, help="Dataset directory.")
    train_parser.add_argument("--output", type=Path, default=MODEL_PATH, help="Model file path.")

    predict_parser = subparsers.add_parser("predict", help="Predict a sensor CSV recording.")
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
