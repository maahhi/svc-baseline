from CNN_eval_preprocessor import extract_features

import h5py
import numpy as np
from joblib import dump, load


time_steps = 216
audio_path = r'C:\Users\maahh\Downloads\Compressed\VocalSet1-2\data_by_singer\female1\arpeggios\belt\f1_arpeggios_belt_c_a.wav' #m1_arpeggios_belt_e.wav
X = []
Y = []
songs = []
label_id = 1
overlap = 0
sample_rate = 8000
mel_db = extract_features(audio_path,sample_rate)

new_run_dir = 'singel_audio_file_run'
output_data_path = './'+new_run_dir+'/data-f-8khz'
with h5py.File(output_data_path+'.h5', 'w') as f:
    f.create_dataset('mel_db', data=mel_db, dtype='float64')
dump(mel_db, output_data_path+'.joblib')