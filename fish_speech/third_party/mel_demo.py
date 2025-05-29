import os
import numpy as np
import librosa


def get_melspectrogram(wav_path, fft_size=800, hop_size=160, win_length=800, window="hann", num_mels=160,
    fmin=0, fmax=8000, eps=1e-6, sample_rate=16000, center=False, mel_basis=None):
    
    # Load wav
    if isinstance(wav_path, str):
        wav, _ = librosa.core.load(wav_path, sr=sample_rate)
    else:
        wav = wav_path
    
    # Pad wav to the multiple of the win_length
    if len(wav) % win_length < win_length - 1:
        wav = np.pad(wav, (0, win_length - 1 - (len(wav) % win_length)), mode='constant', constant_values=0.0)
        
    # get amplitude spectrogram
    x_stft = librosa.stft(wav, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, center=center)
    linear_spc = np.abs(x_stft)  # (n_bins, T)

    # get mel basis
    fmin = 0 if fmin == -1 else fmin
    fmax = sample_rate / 2 if fmax == -1 else min(fmax, sample_rate // 2)
    if mel_basis is None:
        mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax)

    # calculate mel spec
    mel = mel_basis @ linear_spc
    mel = np.log10(np.maximum(eps, mel))  # (n_mel_bins, T)
    return mel.T


mel = get_melspectrogram('demo.wav')
    