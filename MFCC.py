#create the waveform 
from cmath import log
import librosa , librosa.display
import matplotlib.pyplot as plt
import numpy as np
file = 'duke.wav'

signal , sr = librosa.load(file , sr=22050)

# librosa.display.waveshow(signal , sr=sr)
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.show()


fft = np.fft.fft(signal)
magnitude = np.abs(fft)
frequency = np.linspace(0 , sr , len(magnitude))

left_freq = frequency[:int(len(frequency)/2)]
left_mag = magnitude[:int(len(magnitude)/2)]

# plt.plot( left_freq , left_mag )
# plt.xlabel("magnitude")
# plt.ylabel("frequency")
# plt.show()

# stft -> spectogram 

n_fft = 2048 
hop_length = 512

stft = librosa.core.stft(signal , n_fft=n_fft , hop_length=hop_length)
spectogram = np.abs(stft)
log_spec = librosa.amplitude_to_db(spectogram)

# librosa.display.specshow(log_spec , sr=sr , hop_length=hop_length)
# plt.xlabel("Time")
# plt.xlabel("Frequency")
# plt.colorbar()
# plt.show()



#MFCC

MFCCs = librosa.feature.mfcc(signal , n_fft=n_fft , hop_length=hop_length , n_mfcc=13)
librosa.display.specshow(MFCCs , sr=sr , hop_length=hop_length)
plt.xlabel("Time")
plt.xlabel("MFCC")
plt.colorbar()
plt.show()
