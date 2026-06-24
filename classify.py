import json
import os
import shutil
import traceback
import librosa
import numpy as np
import sklearn
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

MODEL_PATH = r"C:\Users\kolli\OneDrive\Documents\Desktop\onyx_classification\dog_emotion_model.tflite"
LABELS_PATH = (
    r"C:\Users\kolli\OneDrive\Documents\Desktop\onyx_classification\labels.json"
)
INPUT_FOLDER = (
    r"C:\Users\kolli\OneDrive\Documents\Desktop\onyx_classification\test_data"
)
OUTPUT_FOLDER = (
    r"C:\Users\kolli\OneDrive\Documents\Desktop\onyx_classification\classification2"
)


def output_to_probabilities(raw_output):
    """Restore valid probabilities after INT8 dequantization distortion."""
    x = np.asarray(raw_output, dtype=np.float64).reshape(-1)

    # Already probabilities
    if np.all(x >= 0) and 0.95 <= np.sum(x) <= 1.05:
        return x.astype(np.float32)

    # Normalize positive outputs
    if np.all(x >= 0) and np.sum(x) > 0:
        return (x / np.sum(x)).astype(np.float32)

    # Softmax fallback
    x = x - np.max(x)
    exp = np.exp(x)
    return (exp / np.sum(exp)).astype(np.float32)


print("\nLoading TensorFlow Lite model...")

# LOAD TFLITE MODEL
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# INPUT + OUTPUT DETAILS
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# LOAD AND SAFE-PARSED LABELS
with open(LABELS_PATH, encoding="utf-8") as f:
    labels_data = json.load(f)

# If JSON is a dictionary (e.g., {"0": "bark", "1": "whine"}), extract values
if isinstance(labels_data, dict):
    # Sort by key if keys are numeric strings to ensure correct index mapping
    try:
        classes = [
            labels_data[k] for k in sorted(labels_data.keys(), key=int)
        ]
    except ValueError:
        classes = list(labels_data.values())
else:
    classes = labels_data

print(f"Loaded classes: {classes}")

# CREATE OUTPUT FOLDERS
for c in classes:
    os.makedirs(os.path.join(OUTPUT_FOLDER, c), exist_ok=True)

mismatched_files = []

# ACCURACY VARIABLES
correct = 0
total = 0

all_actual = []
all_predicted = []

print("\nStarting classification...")

if not os.path.exists(INPUT_FOLDER):
    print(f"ERROR: INPUT_FOLDER does not exist at {INPUT_FOLDER}")
    exit()

for actual_label in os.listdir(INPUT_FOLDER):
    label_path = os.path.join(INPUT_FOLDER, actual_label)

    if os.path.isdir(label_path):
        # Match validation check
        if actual_label not in classes:
            print(
                f"Skipping folder '{actual_label}' because it is not listed in your labels JSON."
            )
            continue

        for file in os.listdir(label_path):
            if file.endswith(".wav"):
                try:
                    file_path = os.path.join(label_path, file)
                    print(f"\nClassifying: {file}")

                    # LOAD AUDIO
                    audio, sr = librosa.load(file_path, sr=None)

                    # EXTRACT MFCC FEATURES
                    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                    mfcc_mean = np.mean(mfcc.T, axis=0)
                    mfcc_std = np.std(mfcc.T, axis=0)

                    feature = np.concatenate([mfcc_mean, mfcc_std])
                    feature = feature.reshape(1, 26).astype(np.float32)

                    # INPUT QUANTIZATION INFO
                    input_scale, input_zero_point = input_details[0][
                        "quantization"
                    ]

                    # QUANTIZE INPUT
                    if input_scale != 0:
                        feature = (
                            feature / input_scale + input_zero_point
                        ).astype(np.int8)

                    # SET INPUT TENSOR
                    interpreter.set_tensor(input_details[0]["index"], feature)

                    # RUN MODEL
                    interpreter.invoke()

                    # GET OUTPUT
                    predictions = interpreter.get_tensor(
                        output_details[0]["index"]
                    )

                    # OUTPUT QUANTIZATION INFO
                    output_scale, output_zero_point = output_details[0][
                        "quantization"
                    ]

                    # DEQUANTIZE OUTPUT
                    if output_scale != 0:
                        predictions = (
                            predictions.astype(np.float32) - output_zero_point
                        ) * output_scale

                    # CONVERT TO PROBABILITIES
                    probs = output_to_probabilities(predictions)
                    predicted_index = int(np.argmax(probs))
                    confidence = float(np.max(probs))

                    prediction = classes[predicted_index]

                    all_actual.append(actual_label)
                    all_predicted.append(prediction)

                    print(f"Actual: {actual_label}")
                    print(f"Predicted: {prediction}")
                    print(f"Confidence: {confidence * 100:.2f}%")

                    # ACCURACY TRACKING
                    total += 1

                    if actual_label == prediction:
                        correct += 1
                        if total % 50 == 0:
                            print(f"Processed {total} files so far...")

                    # COPY FILE TO PREDICTED FOLDER
                    destination = os.path.join(OUTPUT_FOLDER, prediction, file)
                    shutil.copy(file_path, destination)

                    # TRACK MISMATCHES
                    if actual_label != prediction:
                        mismatched_files.append(
                            {
                                "file": file,
                                "actual": actual_label,
                                "predicted": prediction,
                                "confidence": confidence,
                            }
                        )

                except Exception as e:
                    print(f"\nError processing {file}: {e}")
                    traceback.print_exc()

# ==========================================
# FINAL RESULTS (AFTER ALL FILES PROCESSED)
# ==========================================

print("\nClassification complete!")
print(f"\nTotal Files Processed: {total}")
print(f"Actual labels stored: {len(all_actual)}")
print(f"Predicted labels stored: {len(all_predicted)}")

if total > 0:
    accuracy = (correct / total) * 100

    precision = precision_score(
        all_actual, all_predicted, average="weighted", zero_division=0
    )
    recall = recall_score(
        all_actual, all_predicted, average="weighted", zero_division=0
    )
    f1 = f1_score(
        all_actual, all_predicted, average="weighted", zero_division=0
    )

    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nConfusion Matrix:\n")
    cm = confusion_matrix(all_actual, all_predicted, labels=classes)
    print(cm)

    print("\nClassification Report:\n")
    print(
        classification_report(
            all_actual, all_predicted, labels=classes, zero_division=0
        )
    )

else:
    print(
        "\nNo files were processed. Please check if your target folder names match your labels configuration exactly."
    )

print("\nMISMATCHED FILES:\n")
if len(mismatched_files) == 0:
    print("No mismatches found!")
else:
    for item in mismatched_files:
        print(
            f"{item['file']} | "
            f"Actual: {item['actual']} | "
            f"Predicted: {item['predicted']} | "
            f"Confidence: {item['confidence'] * 100:.2f}%"
        )