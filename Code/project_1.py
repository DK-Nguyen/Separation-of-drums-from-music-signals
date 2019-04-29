# Khoa Nguyen - 272580 
# Hai Luong - 268461 
# Hung Nguyen - 272585

#%%
from scipy.io.wavfile import read 
import matplotlib.pyplot as plt
import librosa
import scipy
import numpy as np
from copy import deepcopy
from scipy.io.wavfile import write

#%%
fs,ft = read('rhythm_birdland.wav')
ft = ft.astype('float64')
duration = ft.size / fs

#%% plot the first 1024 samples
plt.plot(ft[0:1024])
# label the axes
plt.ylabel("Amplitude")
plt.xlabel("Time (samples)")
# set the title
plt.title("Rhythm Birdland (first 1024 samples)")
# display the plot
plt.show()

#%% Step 1,2: STFT
win_length = int(2**np.ceil(np.log2(fs*0.02))) # window length in samples
hop_length = int(win_length/2)
_,_,F = scipy.signal.stft(ft, nfft=win_length, noverlap=hop_length, fs=fs,nperseg=win_length)

#%% Signal's Spectrogram
mag_F = np.abs(F)
number_frequencies, number_time_frames = F.shape
freq_scale = np.linspace(0, fs / 2, number_frequencies)
timeframe_scale = np.linspace(0, duration, number_time_frames)
plt.figure(figsize=(20, 22))
plt.pcolormesh(timeframe_scale, freq_scale, mag_F)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

#%% Calculate the range-compressed version of the power spectrogram
gamma = 0.3
W = np.abs(F)**(2*gamma)

#%% W's Spectrogram
# mag_W = np.abs(W)
# number_frequencies, number_time_frames = W.shape
# freq_scale = np.linspace(0, fs / 2, number_frequencies)
# timeframe_scale = np.linspace(0, duration, number_time_frames)
# plt.figure(figsize=(20, 22))
# plt.pcolormesh(timeframe_scale, freq_scale, mag_W)
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (Hz)')

#%% Step 3,4,5,6:
# pad W for later calculation
W_padded = np.pad(W, [(1, 1), (1, 1)], mode='constant')
H = P = W_padded/2
delta = np.zeros(W_padded.shape)
kmax = 10
for k in range(kmax-1):
    alpha = np.var(P)**2 / (np.var(P)**2 + np.var(H)**2)
    delta[1:-1, 1:-1] = alpha * (H[1:-1, 0:-2] - 2 * H[1:-1, 1:-1] + H[1:-1, 2:])/4 - (1 - alpha) * (P[0:-2, 1:-1] - 2 * P[1:-1, 1:-1] + P[2:, 1:-1])/4
    H = np.minimum(np.maximum(H + delta , 0), W_padded)
    P = W_padded - H

#%% Step 8. Binarize the separation result
H_kmax = np.zeros(W_padded.shape)
P_kmax = np.zeros(W_padded.shape)
mask = H<P
P_kmax[mask] = deepcopy(W_padded[mask])
H_kmax[~mask] = deepcopy(W_padded[~mask])
# Unpad H_kmax and P_kmax
H_kmax = H_kmax[1:-1, 1:-1]
P_kmax = P_kmax[1:-1, 1:-1]
print(np.linalg.norm(H_kmax + P_kmax - W))
# if the norm is 0, then the equation holds

#%% Plot the Spectrogram for H_kmax
mag_H = np.abs(H_kmax)
number_frequencies, number_time_frames = H_kmax.shape
freq_scale = np.linspace(0, fs / 2, number_frequencies)
timeframe_scale = np.linspace(0, duration, number_time_frames)
plt.figure(figsize=(20, 22))
plt.pcolormesh(timeframe_scale, freq_scale, mag_H)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

#%% Plot the Spectrogram for P_kmax
mag_P = np.abs(P_kmax)
number_frequencies, number_time_frames = P_kmax.shape
freq_scale = np.linspace(0, fs / 2, number_frequencies)
timeframe_scale = np.linspace(0, duration, number_time_frames)
plt.figure(figsize=(20, 22))
plt.pcolormesh(timeframe_scale, freq_scale, mag_P)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
    
#%% Step 8:
_,ht = scipy.signal.istft((H_kmax**(1/(2*gamma))) * np.exp(1j*np.angle(F) ) ,fs=fs,nperseg= win_length, noverlap=hop_length,input_onesided=True)
_,pt = scipy.signal.istft((P_kmax**(1/(2*gamma))) * np.exp(1j*np.angle(F) ),fs=fs,nperseg= win_length, noverlap=hop_length,input_onesided=True)

#%% 
write('harmonics.wav', fs, ht / np.max(np.abs(ht)))
write('percussive.wav', fs, pt / np.max(np.abs(pt)))





