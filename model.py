import json
import os
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)


TRAIN_FOLDER = r"C:\Users\kolli\OneDrive\Documents\Desktop\onyx_classification\train_data"
TEST_FOLDER = r"C:\Users\kolli\OneDrive\Documents\Desktop\onyx_classification\test_data"

classes = [
    "Angry",
    "Normal Barking"
]

X_train = []
y_train = []

print("\nLoading training data...")
for label in os.listdir(TRAIN_FOLDER):
    if label not in classes:
        continue

    label_path = os.path.join(TRAIN_FOLDER, label)
    if os.path.isdir(label_path):
        for file in os.listdir(label_path):
            if file.endswith(".wav"):
                try:
                    file_path = os.path.join(label_path, file)
                    audio, sr = librosa.load(file_path, sr=None)

                    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                    mfcc_mean = np.mean(mfcc.T, axis=0)
                    mfcc_std = np.std(mfcc.T, axis=0)

                    feature = np.concatenate([mfcc_mean, mfcc_std])
                    X_train.append(feature)
                    y_train.append(label)
                except Exception as e:
                    print(f"Error loading {file}: {e}")

X_test = []
y_test = []

print("\nLoading testing data...")
for label in os.listdir(TEST_FOLDER):
    if label not in classes:
        continue

    label_path = os.path.join(TEST_FOLDER, label)
    if os.path.isdir(label_path):
        for file in os.listdir(label_path):
            if file.endswith(".wav"):
                try:
                    file_path = os.path.join(label_path, file)
                    audio, sr = librosa.load(file_path, sr=None)

                    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                    mfcc_mean = np.mean(mfcc.T, axis=0)
                    mfcc_std = np.std(mfcc.T, axis=0)

                    feature = np.concatenate([mfcc_mean, mfcc_std])
                    X_test.append(feature)
                    y_test.append(label)
                except Exception as e:
                    print(f"Error loading {file}: {e}")

X_train = np.array(X_train)
X_test = np.array(X_test)

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# Save labels mapping file
with open("labels.json", "w", encoding="utf-8") as f:
    json.dump(encoder.classes_.tolist(), f, indent=2)

# Build Model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(26,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining TensorFlow model...")
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=16
)

print("\nEvaluating model...")
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# Save Model
model.save("dog_emotion_model.h5")
print("\nTensorFlow model saved!")

# ==========================================
# ADVANCED METRICS EVALUATION
# ==========================================
print("\n================ METRICS REPORT ================")
if len(y_test) > 0:
    accuracy = accuracy_score(y_test, predicted_labels) * 100
    precision = precision_score(y_test, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(y_test, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predicted_labels, average='weighted', zero_division=0)

    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Precision:        {precision:.4f}")
    print(f"Recall:           {recall:.4f}")
    print(f"F1 Score:         {f1:.4f}")

    print("\nConfusion Matrix:")
    # Map the text labels back onto the matrix indices
    cm = confusion_matrix(y_test, predicted_labels)

    disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)

    plt.title("Dog Emotion Classification Confusion Matrix")
    plt.tight_layout()

    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()
    
else:
    print("No evaluation data found to compute metrics.")
print("================================================")