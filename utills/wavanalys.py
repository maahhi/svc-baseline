# this is a code which read a wav file and print the values, make a plot of its shape
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import soundfile as sf
import sys
wav_dir = './autoencoder_epoch_1.wav'
def main():

    # Load the audio file
    x, fs = librosa.load(wav_dir, sr=16000, dtype=np.float64)
    print(x.shape)
    print(x)

    # in array x, find unique values and first place that value occurs
    a = np.unique(x, return_index=True)
    print(a[0])
    print(a[1])
    print(x[:16000]==x[16000:])
    # Plot the signal read from wav file
    plt.figure(figsize=(14, 5))
    plt.plot(x)
    plt.show()

def moin():
    from scipy.io.wavfile import read
    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    input_data = read(wav_dir)
    audio = input_data[1]
    plt.plot(audio[0:1024])
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    plt.show()

if __name__ == '__main__':
    main()