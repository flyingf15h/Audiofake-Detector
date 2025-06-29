import torch
import numpy as np
import librosa
import pywt
from model import MultiCNN

def prepInputArray(audio_array, sr=16000, fixed_length=16000):
    audio_array = librosa.util.fix_length(audio_array, size=fixed_length)

    # Normalize
    x_raw = (audio_array - np.mean(audio_array)) / (np.std(audio_array) + 1e-6)
    x_raw = x_raw[np.newaxis, :]  # Shape: (1, 16000)

    # Spectrogram
    stft = librosa.stft(audio_array, n_fft=512, hop_length=256)
    mag = np.abs(stft)
    mag = mag[:128, :128]  
    mag = (mag - np.mean(mag)) / (np.std(mag) + 1e-6)
    x_fft = mag[np.newaxis, :, :]  # Shape: (1, 128, 128)

    # Wavelet 
    coeffs = pywt.wavedec(audio_array, 'db4', level=4)
    cA4 = coeffs[0]
    cA4_resized = np.resize(cA4, (64, 128))
    cA4_resized = (cA4_resized - np.mean(cA4_resized)) / (np.std(cA4_resized) + 1e-6)
    x_wav = cA4_resized[np.newaxis, :, :]  # Shape: (1, 64, 128)

    return x_raw.astype(np.float32), x_fft.astype(np.float32), x_wav.astype(np.float32)


def predict(audio_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model audio preprocess
    model = MultiCNN().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    audio_array, _ = librosa.load(audio_path, sr=16000)
    x_raw, x_fft, x_wav = prepInputArray(audio_array)

    x_raw = torch.tensor(x_raw).unsqueeze(0).to(device)  # (1, 1, 16000)
    x_fft = torch.tensor(x_fft).unsqueeze(0).to(device)  # (1, 1, 128, 128)
    x_wav = torch.tensor(x_wav).unsqueeze(0).to(device)  # (1, 1, 64, 128)

    with torch.no_grad():
        prob = model(x_raw, x_fft, x_wav).item()
        label = "Fake" if prob >= 0.5 else "Real"
        print(f"Prediction: {label} (score: {prob:.4f})")

def prepInputFile(path):
    audio_array, _ = librosa.load(path, sr=16000)
    return prepInputArray(audio_array)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python predict.py path_to_audio.wav")
    else:
        predict(sys.argv[1])
