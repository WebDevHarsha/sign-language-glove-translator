# Sign Language Glove Translator

A Python project for recognizing sign-language gestures from sensor readings produced by a data glove. It trains a Random Forest classifier from labeled CSV recordings and can classify individual recordings, translate sensor data in an Excel workbook, or make live predictions from an ESP32 over serial.

> This is a prototype sign-recognition project, not a general-purpose sign-language translator. Recognition is limited to the labels and examples in the supplied dataset.

## How the model works

The classifier is a **scikit-learn `RandomForestClassifier`**, not a neural network. Training reads each CSV recording inside a labeled subdirectory and treats that directory's name as the sign label. The optional `labels.json` file restricts training to directory names listed in that file.

Each recording is represented by 15 sensor channels:

| Channel group | Columns |
| --- | --- |
| Finger bend sensors | `thumb_mV`, `index_mV`, `middle_mV`, `ring_mV`, `little_mV` |
| Accelerometer | `accel_x`, `accel_y`, `accel_z` |
| Gyroscope | `gyro_x`, `gyro_y`, `gyro_z` |
| Orientation quaternion | `quat_w`, `quat_x`, `quat_y`, `quat_z` |

For every recording or inference window, the feature pipeline:

1. Selects those columns in the order shown above and converts values to numbers.
2. Drops rows that do not contain a complete set of finite numeric readings.
3. Linearly resamples each channel to 32 evenly spaced time points.
4. Flattens the result into a fixed-length vector of 480 features (32 points × 15 channels).

The model is trained with 300 trees, balanced-sub-sample class weights, and a fixed random seed. The saved `sign_language_model.joblib` bundle contains the classifier and its expected sensor schema. Training writes this file to the repository root by default; generated model files are git-ignored.

## Repository contents

- `sign_language_model.py` — feature extraction, model training, and single-recording prediction.
- `excel_sign_translator.py` — overlapping-window Excel classification and sentence formatting.
- `esp_comm.py` — live serial input and streaming predictions from an ESP32.
- `test_sign_language_model.py` — held-out recording evaluation.
- `labels.json` — allowed class directory names for training and evaluation.
- `Datasets/data/` — training recordings, arranged in one directory per sign.
- `Datasets/ASL-Sensor-Dataglove-Dataset/` — included source dataset.

## Requirements and setup

Python 3.10 or later is recommended. From the repository root, create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn joblib openpyxl pyserial
```

`openpyxl` is needed to read `.xlsx` workbooks; `pyserial` is needed only for live ESP32 input.

## Train the model

The default dataset directory is `Datasets/data`, and the default output is `sign_language_model.joblib`:

```powershell
python sign_language_model.py train
```

To specify another dataset or output path:

```powershell
python sign_language_model.py train --data .\Datasets\data --output .\sign_language_model.joblib
```

Expected dataset layout:

```text
Datasets/data/
  HELLO/
    recording_01.csv
    recording_02.csv
  A/
    recording_01.csv
```

Each CSV must have the 15 sensor columns listed above. Additional columns such as timestamps are allowed and ignored. Each recording must contain at least one row with numeric values for all 15 channels. Directory names must match entries in `labels.json` when that file is present.

**Label naming note:** the current dataset contains a `THANK U` directory, while `labels.json` lists `THANK YOU`. Since names must match exactly, that directory is excluded from training until the names are made consistent.

## Evaluate recognition

Evaluate on a deterministic, per-class random holdout of recordings (20% by default):

```powershell
python test_sign_language_model.py
```

Set the holdout fraction or dataset directory:

```powershell
python test_sign_language_model.py --data .\Datasets\data --test-fraction 0.2
```

The script reports accuracy and a per-class precision/recall/F1 report. This is a recording-level random split; it should not be interpreted as a signer-independent or real-world performance guarantee.

## Use a sensor CSV recording

Train the model first, then run:

```powershell
python sign_language_model.py predict .\path\to\recording.csv
```

The recording must contain the expected sensor columns. The command prints the predicted label and the classifier's predicted class probability.

## Translate an Excel workbook

The workbook's selected sheet must contain the 15 sensor columns and one sensor sample per row. Additional columns are ignored.

```powershell
python excel_sign_translator.py .\hello_test_input.xlsx
```

The translator uses overlapping windows (128 rows per window, advancing 32 rows by default), predicts each window, removes short low-confidence label runs, collapses repeated labels, then formats the resulting labels into readable text. Adjust the window, stride, sheet, or minimum class probability with:

```powershell
python excel_sign_translator.py .\recording.xlsx `
  --sheet 0 `
  --window-size 128 `
  --stride 32 `
  --min-confidence 0.4
```

The default minimum confidence is `0`, so no window is filtered by probability. The displayed confidence is the Random Forest's class probability and is not a separately calibrated measure of correctness.

Single-letter labels are joined into fingerspelled text and split into likely common English words where possible. Multi-character labels become words; calibration labels `CAL_BENT` and `CAL_FLAT` are omitted from the formatted sentence.

## Live predictions from an ESP32

Connect the glove's ESP32 to the computer, close any other serial monitor using that port, and start:

```powershell
python esp_comm.py --port COM8 --baud-rate 115200
```

The serial device must emit one JSON object per line at approximately 30 samples per second. Each object must include all 15 sensor column names as numeric fields; unrelated fields are ignored. For example:

```json
{"thumb_mV":1259,"index_mV":1070,"middle_mV":1531,"ring_mV":1181,"little_mV":1352,"accel_x":3137,"accel_y":3528,"accel_z":12503,"gyro_x":-3,"gyro_y":-6,"gyro_z":1,"quat_w":-0.8969,"quat_x":-0.0763,"quat_y":0.1542,"quat_z":0.4075}
```

The stream uses a rolling sensor window (128 samples by default) and refreshes predictions every 32 new samples. A two-of-three recent-prediction vote is used before adding a new label to the translated text. Change the port, baud rate, window size, stride, confidence threshold, or model path with the command-line options shown by:

```powershell
python esp_comm.py --help
```

Press **Ctrl+C** to stop.

## Limitations

- Model quality depends on the coverage, consistency, and labeling of the recordings; a confident prediction can still be wrong.
- Resampling preserves a gesture's relative progression but removes its original duration and absolute sample count.
- Results can vary with signer, glove fit, sensor calibration, device orientation, and sensor hardware.
- Sentence formatting is a lightweight label-to-text heuristic; it does not model grammar, continuous signing, or context.
- Excel and live translation rely on the trained model file. If it is missing, run the training command first.
