import tensorflow as tf
import os

MODEL_PATH = "dog_emotion_model.h5"

OUTPUT_PATH = "dog_emotion_model.tflite"

print("\nLoading TensorFlow model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("\nCreating FULLY optimized TensorFlow Lite model...")

converter = tf.lite.TFLiteConverter.from_keras_model(
    model
)

# FULL OPTIMIZATION
converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

# FORCE FULL INT8 QUANTIZATION
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

# INT8 input for size; float32 output keeps softmax probabilities accurate
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.float32

# REPRESENTATIVE DATASET
# VERY IMPORTANT FOR FULL QUANTIZATION

def representative_dataset():

    import librosa
    import numpy as np

    train_folder = r"C:\Users\kolli\OneDrive\Documents\Desktop\onyx_classification\train_data"
    count = 0

    for label in os.listdir(train_folder):

        label_path = os.path.join(train_folder, label)

        if not os.path.isdir(label_path):
            continue

        for file in os.listdir(label_path):

            if not file.endswith(".wav"):
                continue

            try:

                audio, sr = librosa.load(
                    os.path.join(label_path, file),
                    sr=None
                )

                mfcc = librosa.feature.mfcc(
                    y=audio,
                    sr=sr,
                    n_mfcc=13
                )

                mfcc_mean = np.mean(mfcc.T, axis=0)
                mfcc_std = np.std(mfcc.T, axis=0)
                feature = np.concatenate([mfcc_mean, mfcc_std])

                yield [feature.reshape(1, 26).astype(np.float32)]

                count += 1

                if count >= 100:
                    return

            except Exception:
                continue

converter.representative_dataset = (
    representative_dataset
)

print("\nConverting model...")

tflite_model = converter.convert()

with open(OUTPUT_PATH, "wb") as f:

    f.write(tflite_model)

print("\nFully optimized TensorFlow Lite model saved!")

# FILE SIZE COMPARISON

original_size = os.path.getsize(
    MODEL_PATH
) / 1024

optimized_size = os.path.getsize(
    OUTPUT_PATH
) / 1024

print(
    f"\nOriginal Model Size: "
    f"{original_size:.2f} KB"
)

print(
    f"Optimized TFLite Size: "
    f"{optimized_size:.2f} KB"
)

compression_ratio = (
    optimized_size / original_size
) * 100

print(
    f"Compression Ratio: "
    f"{compression_ratio:.2f}%"
)