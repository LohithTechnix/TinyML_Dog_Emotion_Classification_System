import os
import random
import shutil

INPUT_FOLDER = r"C:\Users\kolli\OneDrive\Documents\Desktop\onyx_classification\after_split_2_final"

TRAIN_FOLDER = r"C:\Users\kolli\OneDrive\Documents\Desktop\onyx_classification\train_data_2"

TEST_FOLDER = r"C:\Users\kolli\OneDrive\Documents\Desktop\onyx_classification\test_data_2"

TEST_RATIO = 0.2

RANDOM_SEED = 42

SKIP_LABELS = [
    "Happy"
]


def copy_files(files, source_dir, dest_dir):
    for file in files:
        shutil.copy2(
            os.path.join(source_dir, file),
            os.path.join(dest_dir, file)
        )


print("\nSplitting dataset...")
print(f"Input:  {INPUT_FOLDER}")
print(f"Train:  {TRAIN_FOLDER}")
print(f"Test:   {TEST_FOLDER}")
print(f"Test ratio: {TEST_RATIO * 100:.0f}%\n")

random.seed(RANDOM_SEED)

for folder in (TRAIN_FOLDER, TEST_FOLDER):
    if os.path.exists(folder):
        shutil.rmtree(folder)

os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(TEST_FOLDER, exist_ok=True)

for label in os.listdir(INPUT_FOLDER):

    if label in SKIP_LABELS:
        print(f"Skipping: {label}")
        continue

    label_path = os.path.join(INPUT_FOLDER, label)

    if not os.path.isdir(label_path):
        continue

    wav_files = [
        f for f in os.listdir(label_path)
        if f.lower().endswith(".wav")
    ]

    if not wav_files:
        print(f"No .wav files in {label}, skipping.")
        continue

    random.shuffle(wav_files)

    if len(wav_files) == 1:
        train_files = wav_files
        test_files = []
    else:
        split_index = int(len(wav_files) * (1 - TEST_RATIO))
        split_index = max(1, min(split_index, len(wav_files) - 1))
        train_files = wav_files[:split_index]
        test_files = wav_files[split_index:]

    train_label_dir = os.path.join(TRAIN_FOLDER, label)
    test_label_dir = os.path.join(TEST_FOLDER, label)

    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)

    copy_files(train_files, label_path, train_label_dir)

    if test_files:
        copy_files(test_files, label_path, test_label_dir)

    print(
        f"{label}: {len(wav_files)} total -> "
        f"{len(train_files)} train, {len(test_files)} test"
    )

print("\nSplit complete!")
