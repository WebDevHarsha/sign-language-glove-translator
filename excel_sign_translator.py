from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sign_language_model import MODEL_PATH, SENSOR_COLUMNS, _features


MIN_STABLE_WINDOWS = 3
STRONG_CONFIDENCE = 0.55
NON_LANGUAGE_SIGNS = {"CAL_BENT", "CAL_FLAT"}
SPELLING_WORDS = {
    "a", "about", "after", "again", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "best", "better",
    "big", "but", "by", "call", "can", "come", "could", "day", "deaf", "did",
    "do", "drink", "each", "eat", "else", "even", "every", "feel", "few",
    "find", "first", "for", "forget", "from", "get", "give", "go", "good",
    "got", "great", "had", "happy", "has", "have", "he", "hello", "her", "here", "hi",
    "him", "his", "home", "how", "hungry", "i", "if", "in", "is", "it",
    "its", "just", "know", "like", "little", "look", "love", "make", "many",
    "may", "me", "more", "most", "mother", "much", "my", "name", "need",
    "never", "new", "no", "not", "now", "of", "off", "oh", "on", "one",
    "only", "or", "other", "our", "out", "over", "please", "pretty", "right",
    "said", "same", "see", "she", "should", "sick", "since", "so", "some",
    "sorry", "stop", "take", "tell", "thank", "than", "that", "the", "their",
    "them", "then", "there", "these", "they", "thing", "think", "this",
    "those", "time", "to", "today", "too", "try", "u", "up", "us", "very", "want",
    "water", "we", "well", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "would", "yes", "you", "your",
}


def _window_starts(row_count: int, window_size: int, stride: int) -> list[int]:
    if row_count <= window_size:
        return [0]

    starts = list(range(0, row_count - window_size + 1, stride))
    final_start = row_count - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _remove_transient_predictions(
    predictions: list[str], confidences: list[float]
) -> list[str]:
    """Drop short, low-confidence label runs, usually caused by gesture transitions."""
    stable: list[str] = []
    start = 0
    while start < len(predictions):
        end = start + 1
        while end < len(predictions) and predictions[end] == predictions[start]:
            end += 1

        run_confidence = max(confidences[start:end])
        if end - start >= MIN_STABLE_WINDOWS or run_confidence >= STRONG_CONFIDENCE:
            stable.extend(predictions[start:end])
        start = end

    return stable


def _spell_word(letters: str) -> list[str]:
    """Split a fingerspelled character run into likely common English words."""
    text = letters.lower()
    words: list[str] = []
    position = 0
    while position < len(text):
        matches = [
            word
            for word in SPELLING_WORDS
            if text.startswith(word, position)
        ]
        if matches:
            word = max(matches, key=len)
            words.append(word)
            position += len(word)
        else:
            words.append(text[position])
            position += 1
    return words


def _format_sentence(labels: list[str]) -> str:
    """Turn stable sign labels into readable words and spelled-out text."""
    words: list[str] = []
    letters = ""

    def flush_letters() -> None:
        nonlocal letters
        if letters:
            words.extend(_spell_word(letters))
            letters = ""

    for label in labels:
        if label == "QQ":
            label = "Q"
        if len(label) == 1 and label.isalpha():
            letters += label
        else:
            flush_letters()
            if label in NON_LANGUAGE_SIGNS:
                continue
            words.append(label.lower())
    flush_letters()

    if not words:
        return ""
    sentence = " ".join(words)
    return sentence[0].upper() + sentence[1:] + "."


def translate_excel(
    excel_path: Path,
    model_path: Path = MODEL_PATH,
    sheet_name: str | int = 0,
    window_size: int = 128,
    stride: int = 32,
    min_confidence: float = 0.0,
) -> str:
    """Classify overlapping sensor windows and collapse repeated predictions."""
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")
    if stride < 1 or stride > window_size:
        raise ValueError("stride must be between 1 and window_size.")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1.")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. Run sign_language_model.py train first."
        )

    try:
        readings = pd.read_excel(excel_path, sheet_name=sheet_name)
    except ImportError as error:
        raise RuntimeError(
            "Reading Excel files requires openpyxl. Install it with "
            "'.venv\\Scripts\\python.exe -m pip install openpyxl'."
        ) from error

    missing_columns = [column for column in SENSOR_COLUMNS if column not in readings.columns]
    if missing_columns:
        raise ValueError(
            "Excel sheet is missing the required sensor columns: "
            + ", ".join(missing_columns)
        )
    if readings.empty:
        raise ValueError("The selected Excel sheet contains no sensor readings.")

    bundle: dict[str, Any] = joblib.load(model_path)
    classifier = bundle["classifier"]
    predictions: list[str] = []
    confidences: list[float] = []

    for start in _window_starts(len(readings), window_size, stride):
        window = readings.iloc[start : start + window_size]
        try:
            features = _features(window).reshape(1, -1)
        except ValueError as error:
            if "no complete numeric sensor samples" in str(error):
                continue
            raise

        probabilities = classifier.predict_proba(features)[0]
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        if confidence >= min_confidence:
            predictions.append(str(classifier.classes_[best_index]))
            confidences.append(confidence)

    if not predictions:
        raise ValueError("No usable sensor readings produced a prediction.")

    stable = _remove_transient_predictions(predictions, confidences)
    if not stable:
        raise ValueError(
            "No stable sign predictions were found. Try a longer recording or "
            "lower the --min-confidence value."
        )

    # Overlapping windows repeat a sign for many steps; keep one label per run.
    collapsed = [
        label
        for index, label in enumerate(stable)
        if index == 0 or label != stable[index - 1]
    ]
    sentence = _format_sentence(collapsed)
    if not sentence:
        raise ValueError("Predictions contained no translatable sign labels.")
    return sentence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("excel", type=Path, help="Excel workbook containing sensor readings.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Trained model file.")
    parser.add_argument("--sheet", default=0, help="Sheet name or zero-based sheet index.")
    parser.add_argument(
        "--window-size",
        type=int,
        default=128,
        help="Sensor rows in each prediction window (default: 128).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=32,
        help="Rows to move between windows (default: 32).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Ignore window predictions below this confidence (default: 0).",
    )
    args = parser.parse_args()

    sheet_name: str | int = args.sheet
    if isinstance(sheet_name, str) and sheet_name.isdecimal():
        sheet_name = int(sheet_name)

    translation = translate_excel(
        args.excel,
        model_path=args.model,
        sheet_name=sheet_name,
        window_size=args.window_size,
        stride=args.stride,
        min_confidence=args.min_confidence,
    )
    print(f"Translation: {translation}")


if __name__ == "__main__":
    main()
