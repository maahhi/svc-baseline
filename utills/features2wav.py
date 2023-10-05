import numpy as np
import pyworld as pw
import soundfile as sf
import h5py
from joblib import dump, load
import matplotlib.pyplot as plt


sample_rate = 8000
# data loading
data_run = 'run8'
Xtrain_path= './../data-preprocess/'+data_run+'/data.joblib'
Ytest_path = './../inference/one-to-one-autoencoder/inf1/Y_test.joblib'
output_path = './../inference/one-to-one-autoencoder/inf1/outputs.joblib'
Xtest_path = './../inference/one-to-one-autoencoder/inf1/X_test.joblib'
singe_audio_path = './../data-preprocess/singel_audio_file_run/data-f-8khz.joblib'
singe_audio_H5_path = './../data-preprocess/singel_audio_file_run/data-f-8khz.h5'

Xtrain = load(singe_audio_path)
XtrainH5 = None
with h5py.File(singe_audio_H5_path, 'r') as f:
    XtrainH5 = f['mel_db'][:]
a = np.equal(Xtrain, XtrainH5)

sample = Xtrain.copy().astype(np.float64)
sample5 = XtrainH5.copy().astype(np.float64)
b = np.equal(sample, sample5)
del Xtrain
sample = sample5.T
cutsize8 = 258
cutsize16 = 514
f0 = sample[:, 0]
sp = sample[:, 1:cutsize8]
ap = sample[:, cutsize8:]
f0 = np.ascontiguousarray(f0)
sp = np.ascontiguousarray(sp)
ap = np.ascontiguousarray(ap)

print(sp)
fig, ax = plt.subplots()

# Display an image, i.e. data on a 2D regular raster.
cax = ax.imshow(ap.T, cmap='jet')

plt.show()

# Use the WORLD vocoder to synthesize a waveform from the features
reconstructed_waveform = pw.synthesize(f0, sp, ap, sample_rate)

# Save the waveform to a .wav file
sf.write('./../data-preprocess/singel_audio_file_run/data-f-8khz-H5.wav', reconstructed_waveform, sample_rate)