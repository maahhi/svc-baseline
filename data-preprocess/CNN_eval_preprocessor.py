import psutil
import time
import os
import numpy as np
import librosa
import pyworld as pw
from joblib import dump, load


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2  # return in MB

# Before processing
memory_before = get_memory_usage()
# Start the clock
start_time = time.time()


def extract_world_features(x, fs=16000):
    # Load the audio file
    #x, _ = librosa.load(wav_file, sr=fs, dtype=np.float64)

    # Use WORLD to extract F0, spectral envelope and aperiodicity
    _f0, timeaxis = pw.dio(x, fs)  # Raw pitch extraction
    f0 = pw.stonemask(x, _f0, timeaxis, fs)  # Pitch refinement
    sp = pw.cheaptrick(x, f0, timeaxis, fs)  # Spectral envelope estimation
    ap = pw.d4c(x, f0, timeaxis, fs)  # Aperiodicity estimation

    return f0, sp, ap


# Parameters
dataset_path = r'C:\Users\maahh\Downloads\Compressed\VocalSet1-2\data_by_singer'
sample_rate = 16000
n_mels = 128
hop_length = 512
time_steps = 216  # Number of time-steps per spectrogram (e.g., 3 seconds at 22050 Hz with a hop_length of 512)
limit_of_lables =2

# Load audio files and convert to mel spectrogram
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=sample_rate, dtype=np.float64)
    f0, sp, ap = extract_world_features(y, sr)
    features = np.vstack((np.transpose(f0), np.transpose(sp), np.transpose(ap)))
    return features


# Traverse dataset and extract features
X = []
Y = []
songs = []
labels = sorted(os.listdir(dataset_path))
print(labels)
for label_id, label in enumerate(labels):
    print(label_id,label)
    class_folder = os.path.join(dataset_path, label)
    for root, _, files in os.walk(class_folder):
        for file in files:
            if file.endswith('.wav'):
                audio_path = os.path.join(root, file)
                mel_db = extract_features(audio_path)
                song_name = audio_path.split("\\")[-1].split('.')[0][3:]
                #print(song_name)
                # Padding or truncating to ensure consistent shape
                extra = mel_db.shape[1] - time_steps
                if extra>0:
                    mel_db = mel_db[:, int(extra/2)-1:time_steps+int(extra/2)-1]
                else:
                    padding = np.zeros((n_mels, time_steps - mel_db.shape[1]))
                    mel_db = np.hstack((mel_db, padding))

                X.append(mel_db)
                Y.append(label_id)
                songs.append(song_name)
    if label_id>limit_of_lables:
        break

unique_strings = set(songs)
string_to_int = {string: i for i, string in enumerate(unique_strings)}

# Step 2: Convert the original list
converted_list = [string_to_int[s] for s in songs]

print(converted_list)  # This will be your list of integers
print(string_to_int)   # This will be your dictionary of string to integer mappings

X = np.array(X)
Y = np.array(Y)
songs = np.array(converted_list)
dump(X, './run3/data.joblib')
dump(Y, './run3/labels.joblib')
dump(songs,'./run3/labels.joblib')

# Split the data into training, validation, and test sets (e.g., 80%, 10%, 10%)
from sklearn.model_selection import train_test_split

X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=1/9, random_state=42)


print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

data_loaded = load('./run2/data.joblib')
labels_loaded = load('./run2/labels.joblib')
print(data_loaded == X)
print(labels_loaded == Y)

memory_after = get_memory_usage()

print(f"Memory used: {memory_after - memory_before} MB")
# End the clock
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time:.2f} seconds")