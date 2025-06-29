import torch
import numpy as np
import librosa
import pywt
from model import AudioMultiBranchCNN

# Same preprocessing as training
def prepare_inputs_from_array(audio_array, sr=16000, fixed_length=16000):
    audio_array = librosa.util.fix_length(audio_array, size=fixed_length)
    x_raw = np.expand_dims(audio_array, axis=0)
    stft = librosa.stft(audio_array, n_fft=512, hop_length=256)
    mag = np.abs(stft)
    mag = mag[:128, :128]
    x_fft = np.expand_dims(mag, axis=0)
    coeffs = pywt.wavedec(audio_array, 'db4', level=4)
    cA4 = coeffs[0]
    cA4_resized = np.resize(cA4, (64, 128))
    x_wav = np.expand_dims(cA4_resized, axis=0)
    return x_raw, x_fft, x_wav

def predict(audio_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioMultiBranchCNN().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    # Load and preprocess audio
    audio_array, sr = librosa.load(audio_path, sr=16000)
    x_raw, x_fft, x_wav = prepare_inputs_from_array(audio_array)

    # Convert to tensor and send to device
    x_raw = torch.tensor(x_raw).unsqueeze(0).float().to(device)
    x_fft = torch.tensor(x_fft).unsqueeze(0).float().to(device)
    x_wav = torch.tensor(x_wav).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = model(x_raw, x_fft, x_wav).item()
        label = "FAKE" if output >= 0.5 else "REAL"
        print(f"Prediction: {label} (score: {output:.4f})")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python predict.py path_to_audio.wav")
    else:
        predict(sys.argv[1])
