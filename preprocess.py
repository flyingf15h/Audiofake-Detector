import librosa
import numpy as np
import pywt

def get_fft(waveform):
    stft = librosa.stft(waveform, n_fft=256)
    mag = np.abs(stft)
    return mag[:128, :128]  

def get_wavelet(waveform):
    coeffs = pywt.wavedec(waveform, 'db4', level=4)
    cA4 = coeffs[0]
    return np.resize(cA4, (64, 128))  

def prepare_inputs_from_array(audio_array, sr=16000):
    # 1 second of audio (pad/truncate)
    audio_array = librosa.util.fix_length(audio_array, size=sr)

    x_raw = audio_array
    x_fft = get_fft(x_raw)
    x_wav = get_wavelet(x_raw)

    return (
        np.expand_dims(x_raw, axis=(0,)),    # (1, 16000)
        np.expand_dims(x_fft, axis=(0,)),    # (1, 128, 128)
        np.expand_dims(x_wav, axis=(0,))     # (1, 64, 128)
    )
