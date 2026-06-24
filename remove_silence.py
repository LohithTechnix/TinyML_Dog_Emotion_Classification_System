import librosa
import soundfile as sf
import numpy as np
import os

INPUT_FOLDER = r"C:\Users\kolli\OneDrive\Documents\Desktop\onyx_classification\raw_audio\spc"

OUTPUT_FOLDER = r"C:\Users\kolli\OneDrive\Documents\Desktop\onyx_classification\after_silence"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for label in os.listdir(INPUT_FOLDER):

    label_path = os.path.join(INPUT_FOLDER, label)

    if os.path.isdir(label_path):

        output_label_folder = os.path.join(
            OUTPUT_FOLDER,
            label
        )

        os.makedirs(output_label_folder, exist_ok=True)

        for file in os.listdir(label_path):

            if file.lower().endswith((".wav", ".mp3")):

                try:

                    file_path = os.path.join(
                        label_path,
                        file
                    )

                    print(f"Processing: {file_path}")

                    y, sr = librosa.load(
                        file_path,
                        sr=None
                    )

                    intervals = librosa.effects.split(
                        y,
                        top_db=20
                    )

                    cleaned_audio = []

                    for start, end in intervals:
                        cleaned_audio.extend(y[start:end])

                    cleaned_audio = np.array(cleaned_audio)

                    output_path = os.path.join(
                        output_label_folder,
                        os.path.splitext(file)[0] + ".wav"
                    )

                    sf.write(output_path, cleaned_audio, sr)

                    print(f"Saved: {output_path}")

                except Exception as e:

                    print(f"ERROR: {file}")
                    print(e)

print("Silence removal complete!")