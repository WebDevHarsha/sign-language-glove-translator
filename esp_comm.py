from __future__ import annotations

import argparse
import json
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import serial

from excel_sign_translator import _format_sentence
from sign_language_model import MODEL_PATH, SENSOR_COLUMNS, _features


DEFAULT_PORT = "COM8"
DEFAULT_BAUD_RATE = 115200
SAMPLE_RATE_HZ = 30
DEFAULT_WINDOW_SIZE = 128
DEFAULT_STRIDE = 32
PREDICTION_HISTORY = 3


def _parse_sensor_sample(line: str) -> np.ndarray:
    """Parse one JSON object containing the model's 15 sensor fields."""
    try:
        data: Any = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError("not valid JSON") from error

    if not isinstance(data, dict):
        raise ValueError("expected a JSON object keyed by sensor column names")

    missing = [column for column in SENSOR_COLUMNS if column not in data]
    if missing:
        raise ValueError(f"missing sensor fields: {', '.join(missing)}")

    try:
        sample = np.asarray([float(data[column]) for column in SENSOR_COLUMNS], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("sensor fields must contain numeric values") from error

    if not np.isfinite(sample).all():
        raise ValueError("sensor fields must contain finite numeric values")
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=DEFAULT_PORT, help="ESP32 serial port (default: COM8).")
    parser.add_argument(
        "--baud-rate",
        type=int,
        default=DEFAULT_BAUD_RATE,
        help="Serial baud rate (default: 115200).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="Saved classifier file (default: sign_language_model.joblib).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Samples per prediction window (128 is about 4.3 seconds at 30 Hz).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help="New samples between predictions (32 is about 1.1 seconds at 30 Hz).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Ignore predictions below this confidence (0 to 1).",
    )
    args = parser.parse_args()

    if args.window_size < 1:
        parser.error("--window-size must be at least 1")
    if not 1 <= args.stride <= args.window_size:
        parser.error("--stride must be between 1 and --window-size")
    if not 0.0 <= args.min_confidence <= 1.0:
        parser.error("--min-confidence must be between 0 and 1")
    if not args.model.is_file():
        parser.error(f"trained model not found: {args.model}; train it first")

    bundle: dict[str, Any] = joblib.load(args.model)
    classifier = bundle["classifier"]
    sample_buffer: deque[np.ndarray] = deque(maxlen=args.window_size)
    prediction_history: deque[str] = deque(maxlen=PREDICTION_HISTORY)
    translated_labels: list[str] = []
    new_samples_since_prediction = 0

    print(f"Connecting to ESP32 on {args.port} at {args.baud_rate} baud...")
    try:
        with serial.Serial(args.port, args.baud_rate, timeout=1) as esp32:
            time.sleep(2)
            esp32.reset_input_buffer()
            print(
                f"Connected. Expecting {SAMPLE_RATE_HZ} JSON sensor samples per second "
                f"with fields: {', '.join(SENSOR_COLUMNS)}"
            )
            print("Press Ctrl+C to stop.\n")

            while True:
                line = esp32.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                if not line.startswith("{"):
                    print(f"ESP32: {line}")
                    continue

                try:
                    sample_buffer.append(_parse_sensor_sample(line))
                except ValueError as error:
                    print(f"Skipping invalid sensor line: {error}")
                    continue

                new_samples_since_prediction += 1
                if (
                    len(sample_buffer) < args.window_size
                    or new_samples_since_prediction < args.stride
                ):
                    continue

                new_samples_since_prediction = 0
                features = _features(np.vstack(sample_buffer)).reshape(1, -1)
                probabilities = classifier.predict_proba(features)[0]
                best_index = int(np.argmax(probabilities))
                confidence = float(probabilities[best_index])
                if confidence < args.min_confidence:
                    continue

                label = str(classifier.classes_[best_index])
                prediction_history.append(label)
                votes = Counter(prediction_history)
                stable_label, vote_count = votes.most_common(1)[0]
                if vote_count < 2:
                    continue
                if translated_labels and translated_labels[-1] == stable_label:
                    continue

                translated_labels.append(stable_label)
                sentence = _format_sentence(translated_labels)
                print(
                    f"Prediction: {label} ({confidence:.1%}) | "
                    f"Translation: {sentence}"
                )

    except serial.SerialException as error:
        print(f"Serial connection error: {error}")
        print(f"Check {args.port} and close the Arduino Serial Monitor.")
    except KeyboardInterrupt:
        print("\nStopped reading sensor data.")


if __name__ == "__main__":
    main()
