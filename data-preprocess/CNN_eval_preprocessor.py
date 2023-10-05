import psutil
import time
import os
import numpy as np
import librosa
import pyworld as pw
from joblib import dump, load
import soundfile as sf
import inspect


reconstruct = False

def extract_world_features(x, fs):
    # Load the audio file
    #x, _ = librosa.load(wav_file, sr=fs, dtype=np.float64)

    # Use WORLD to extract F0, spectral envelope and aperiodicity
    _f0, timeaxis = pw.dio(x, fs)  # Raw pitch extraction
    f0 = pw.stonemask(x, _f0, timeaxis, fs)  # Pitch refinement     (????,1)
    sp = pw.cheaptrick(x, f0, timeaxis, fs)  # Spectral envelope estimation (????,513)
    ap = pw.d4c(x, f0, timeaxis, fs)  # Aperiodicity estimation (????,513)
    if reconstruct:
        reconstructed_waveform = pw.synthesize(f0[0:216], sp[0:216,:], ap[0:216,:], fs)
        sf.write('./reconstructed_audio216.wav', reconstructed_waveform, fs)
    return f0, sp, ap

# Load audio files and convert to mel spectrogram
def extract_features(audio_path, sample_rate):
    y, sr = librosa.load(audio_path, sr=sample_rate, dtype=np.float64)
    f0, sp, ap = extract_world_features(y, sr)
    features = np.vstack((np.transpose(f0), np.transpose(sp), np.transpose(ap)))
    return features # (1027, ????)


def main():
    sample_rate = 16000
    path = os.getcwd()
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**2  # return in MB

    # Before processing
    memory_before = get_memory_usage()
    # Start the clock
    start_time = time.time()



    log = []
    # add runtime data and time to the log file
    log.append(time.ctime())
    # Parameters
    dataset_path = r'C:\Users\maahh\Downloads\Compressed\VocalSet1-2\data_by_singer'
    log.append(f'{dataset_path=}')

    log.append(f'{sample_rate=}')
    n_mels = 128
    log.append(f'{n_mels=}')
    hop_length = 512
    log.append(f'{hop_length=}')
    time_steps = 216  # Number of time-steps per spectrogram (e.g., 3 seconds at 22050 Hz with a hop_length of 512)
    log.append(f'{time_steps=}')
    overlap = 0 #int(time_steps/3)
    log.append(f'{overlap=}')
    limit_of_lables =1
    log.append(f'{limit_of_lables=}')
    lable_list = ['female1','female2','male1','male2']
    log.append(f'{lable_list=}')




    # Traverse dataset and extract features
    X = []
    Y = []
    songs = []
    labels = sorted(os.listdir(dataset_path))
    print(labels)
    for label_id, label in enumerate(labels):
        if len(lable_list) > 0 :
            if label not in lable_list:
                continue
        elif label_id>limit_of_lables-1:
            break
        print(label_id,label)
        class_folder = os.path.join(dataset_path, label)
        for root, _, files in os.walk(class_folder):
            for file in files:
                if file.endswith('.wav'):
                    audio_path = os.path.join(root, file)
                    mel_db = extract_features(audio_path,sample_rate)
                    song_name = audio_path.split("\\")[-1].split('.')[0][3:]
                    lenght = time_steps
                    b = np.array([mel_db[:,i:i+lenght] for i in range(0, int(mel_db.shape[1]-lenght), lenght-overlap)])
                    for i in range(b.shape[0]):
                        X.append(b[i])
                    for i in range(b.shape[0]):
                        Y.append(label_id)
                        songs.append(song_name)



    unique_strings = set(songs)
    string_to_int = {string: i for i, string in enumerate(unique_strings)}

    # Step 2: Convert the original list
    converted_list = [string_to_int[s] for s in songs]

    log.append(f'{converted_list=}')  # This will be your list of integers
    log.append(f'{string_to_int=}')   # This will be your dictionary of string to integer mappings

    X = np.array(X)
    Y = np.array(Y)

    # a function search in specific directory, find all directories start with "run+number" and make a new directory with the next number
    def make_new_dir():
        path = os.getcwd()
        print(path)
        dirs = os.listdir(path)
        run_dirs = []
        for dir in dirs:
            if dir.startswith("run"):
                run_dirs.append(dir)
        run_dirs.sort()
        new_dir = "run" + str(int(run_dirs[-1][3:]) + 1)
        os.mkdir(new_dir)
        return new_dir
    new_run_dir = make_new_dir()
    songs = np.array(converted_list)
    dump(X, './'+new_run_dir+'/data.joblib')
    dump(Y, './'+new_run_dir+'/labels.joblib')
    dump(songs,'./'+new_run_dir+'/songs.joblib')


    log.append(f"Training samples: {X.shape[0]}")

    #data_loaded = load('./'+new_run_dir+'/data.joblib')
    #labels_loaded = load('./'+new_run_dir+'/labels.joblib')
    # function return true if 2 ndarray are equal
    #log.append(np.array_equal(data_loaded, X))
    #log.append(np.array_equal(labels_loaded, Y))
    #log.append(Y == labels_loaded)
    #log.append(Y)
    #log.append(labels_loaded)

    memory_after = get_memory_usage()

    log.append(f"Memory used: {memory_after - memory_before} MB")
    # End the clock
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    log.append(f"Time taken: {elapsed_time:.2f} seconds")

    # Save the log file
    with open('./'+new_run_dir+'/log.txt', 'w') as f:
        for item in log:
            f.write("%s\n" % item)


if "__name__" == "__main__" :
    main()