import librosa
import numpy as np
import pywt

def fix_length(audio, target_length=16000):
    if len(audio) > target_length:
        return audio[:target_length]
    else:
        return np.pad(audio, (0, target_length - len(audio)))

def get_fft(waveform, n_fft=256, hop_length=128):
    stft = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(stft)
    # Resize to (128,128)
    mag = mag[:128, :128]
    if mag.shape[1] < 128:
        mag = np.pad(mag, ((0,0),(0,128 - mag.shape[1])))
    return mag

def get_wavelet(waveform):
    coeffs = pywt.wavedec(waveform, 'db4', level=4)
    cA4 = coeffs[0]  # Approx. coeffs at level 4
    # Resize to (64, 128)
    if len(cA4) < 64*128:
        cA4 = np.pad(cA4, (0, 64*128 - len(cA4)))
    else:
        cA4 = cA4[:64*128]
    wav_mat = cA4.reshape(64, 128)
    return wav_mat

def prepare_inputs_from_array(audio_array):
    # Fix length to 16000 samples (1 sec at 16kHz)
    audio_array = fix_length(audio_array, 16000)
    fft = get_fft(audio_array)
    wav = get_wavelet(audio_array)

    # Add channel dims for CNN: (C=1,...)
    x_raw = audio_array[np.newaxis, :]          # (1, 16000)
    x_fft = fft[np.newaxis, :, :]                # (1, 128, 128)
    x_wav = wav[np.newaxis, :, :]                # (1, 64, 128)

    return x_raw.astype(np.float32), x_fft.astype(np.float32), x_wav.astype(np.float32)