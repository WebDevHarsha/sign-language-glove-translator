"""Translate live ESP32 glove readings with the 14-channel Extra Trees model."""

from __future__ import annotations

import argparse
import base64
import json
import queue
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
import serial

from excel_sign_translator import _format_sentence
from extra_trees_no_middle_model import MODEL_PATH, _features, load_classifier
from extra_trees_no_middle_model import SENSOR_COLUMNS


DEFAULT_PORT = "COM8"
DEFAULT_BAUD_RATE = 115200
SAMPLE_RATE_HZ = 30
DEFAULT_WINDOW_SIZE = 128
DEFAULT_STRIDE = 32
PREDICTION_HISTORY = 3
SPEECH_QUIET_SECONDS = 1.5


def _parse_sensor_sample(line: str) -> np.ndarray:
    """Parse one JSON object containing the model's 14 required sensor fields."""
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
        sample = np.asarray(
            [float(data[column]) for column in SENSOR_COLUMNS],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("sensor fields must contain numeric values") from error

    if not np.isfinite(sample).all():
        raise ValueError("sensor fields must contain finite numeric values")
    return sample


def _speech_suffix(sentence: str, previous_sentence: str) -> str:
    """Return only newly added words when a translation extends prior text."""
    current_words = sentence.rstrip(".!?").split()
    previous_words = previous_sentence.rstrip(".!?").split()
    if current_words[: len(previous_words)] == previous_words:
        current_words = current_words[len(previous_words) :]
    return " ".join(current_words)


class SpeechOutput:
    """Speak settled translation updates asynchronously using Windows SAPI."""

    def __init__(self) -> None:
        self._updates: queue.Queue[str | None] = queue.Queue(maxsize=1)
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def update(self, sentence: str) -> None:
        """Queue the latest translation, replacing one not yet spoken."""
        try:
            self._updates.put_nowait(sentence)
        except queue.Full:
            try:
                self._updates.get_nowait()
            except queue.Empty:
                pass
            self._updates.put_nowait(sentence)

    def _run(self) -> None:
        previous_sentence = ""
        while True:
            sentence = self._updates.get()
            if sentence is None:
                return

            while True:
                try:
                    newer_sentence = self._updates.get(timeout=SPEECH_QUIET_SECONDS)
                except queue.Empty:
                    break
                if newer_sentence is None:
                    return
                sentence = newer_sentence

            text = _speech_suffix(sentence, previous_sentence)
            if not text:
                continue

            encoded_text = base64.b64encode(text.encode("utf-8")).decode("ascii")
            script = (
                "$voice = New-Object -ComObject SAPI.SpVoice; "
                f"$text = [Text.Encoding]::UTF8.GetString("
                f"[Convert]::FromBase64String('{encoded_text}')); "
                "[void]$voice.Speak($text)"
            )
            encoded_script = base64.b64encode(script.encode("utf-16le")).decode("ascii")
            try:
                subprocess.run(
                    [
                        "powershell.exe",
                        "-NoProfile",
                        "-NonInteractive",
                        "-EncodedCommand",
                        encoded_script,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=30,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except (OSError, subprocess.SubprocessError) as error:
                print(f"Text-to-speech error: {error}", file=sys.stderr)
                continue

            previous_sentence = sentence


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
        help="Saved 14-channel Extra Trees model file.",
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
    parser.add_argument(
        "--no-speech",
        action="store_true",
        help="Disable Windows text-to-speech.",
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
    if not args.no_speech and sys.platform != "win32":
        parser.error("Text-to-speech uses Windows SAPI; use --no-speech on other platforms.")

    try:
        classifier = load_classifier(args.model)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))

    sample_buffer: deque[np.ndarray] = deque(maxlen=args.window_size)
    prediction_history: deque[str] = deque(maxlen=PREDICTION_HISTORY)
    translated_labels: list[str] = []
    new_samples_since_prediction = 0
    speech = None if args.no_speech else SpeechOutput()

    print(f"Connecting to ESP32 on {args.port} at {args.baud_rate} baud...")
    try:
        with serial.Serial(args.port, args.baud_rate, timeout=1) as esp32:
            time.sleep(2)
            esp32.reset_input_buffer()
            print(
                f"Connected. Expecting {SAMPLE_RATE_HZ} JSON sensor samples per second "
                f"with fields: {', '.join(SENSOR_COLUMNS)}"
            )
            print("The middle-finger channel is ignored. Press Ctrl+C to stop.\n")

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
                if speech is not None:
                    speech.update(sentence)

    except serial.SerialException as error:
        print(f"Serial connection error: {error}")
        print(f"Check {args.port} and close the Arduino Serial Monitor.")
    except KeyboardInterrupt:
        print("\nStopped reading sensor data.")


if __name__ == "__main__":
    main()
