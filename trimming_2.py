import librosa
import soundfile as sf
import os
import shutil

INPUT_FOLDER = r"C:\Users\kolli\OneDrive\Documents\Desktop\onyx_classification\after_silence"

OUTPUT_FOLDER = r"C:\Users\kolli\OneDrive\Documents\Desktop\onyx_classification\after_split_2_final"

CLIP_DURATION = 2


def safe_print(message):
    try:
        print(message)
    except UnicodeEncodeError:
        print(
            message.encode("ascii", errors="replace").decode("ascii")
        )


if os.path.exists(OUTPUT_FOLDER):
    shutil.rmtree(OUTPUT_FOLDER)

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

            if file.lower().endswith(".wav"):

                try:

                    file_path = os.path.join(
                        label_path,
                        file
                    )

                    safe_print(f"Splitting: {file_path}")

                    audio, sr = librosa.load(
                        file_path,
                        sr=8000
                    )

                    samples_per_clip = sr * CLIP_DURATION

                    total_clips = len(audio) // samples_per_clip

                    base_name = os.path.splitext(file)[0]

                    for i in range(total_clips):

                        start = i * samples_per_clip
                        end = start + samples_per_clip

                        clip = audio[start:end]

                        output_path = os.path.join(
                            output_label_folder,
                            f"{base_name}_{i}.wav"
                        )

                        sf.write(output_path, clip, sr)

                        safe_print(f"Saved: {output_path}")

                except Exception as e:

                    safe_print(f"ERROR: {file}")
                    safe_print(str(e))

print("Audio splitting complete!")
