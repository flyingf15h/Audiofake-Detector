import torch
import numpy as np
import librosa
import pywt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                           precision_score, recall_score, classification_report, roc_curve)
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from pathlib import Path
from model import TBranchDetector
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
model = torch.nn.DataParallel(TBranchDetector()).to(device)
model.load_state_dict(
    torch.load(
        "/kaggle/input/audifake-detector/pytorch/default/1/best_model.pth",
        map_location=device
    )
)
model.eval()

def calculate_eer(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def preprocess(audio, sr=16000):
    # Match training exactly
    audio = librosa.util.fix_length(audio, size=16000)
    
    # Raw waveform (identical to training)
    x_raw = (audio - np.mean(audio)) / (np.std(audio) + 1e-8)
    
    # Spectrogram (training uses n_fft=512, hop=256)
    stft = librosa.stft(audio, n_fft=512, hop_length=256)
    mag = np.abs(stft)[:128, :128]  # Crop to match training
    x_fft = (mag - np.mean(mag)) / (np.std(mag) + 1e-8)
    
    # Wavelet (identical to training)
    coeffs = pywt.wavedec(audio, 'db4', level=4)
    cA4_resized = np.resize(coeffs[0], (64, 128))
    x_wav = (cA4_resized - np.mean(cA4_resized)) / (np.std(cA4_resized) + 1e-8)
    
    return (
        torch.tensor(x_raw).unsqueeze(0).float(),
        torch.tensor(x_fft).unsqueeze(0).float(),
        torch.tensor(x_wav).unsqueeze(0).float()
    )

class AudioDataset(Dataset):
    def __init__(self, file_label_pairs):
        self.data = file_label_pairs
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        path, label = self.data[idx]
        try:
            audio, _ = librosa.load(path, sr=16000)
            return *preprocess(audio), label
        except Exception as e:
            print(f"Skipping {path}: {str(e)}")
            return (
                torch.zeros(1, 16000), 
                torch.zeros(1, 128, 128),
                torch.zeros(1, 64, 128), 
                -1  # Mark invalid samples
            )

def load_for_original(max_samples=None):
    base = Path("/kaggle/input/the-fake-or-real-dataset/for-original/for-original/testing")
    data = []
    for label, name in [(0, 'real'), (1, 'fake')]:
        files = list((base / name).glob("*.wav"))
        if max_samples:
            files = files[:max_samples]
        data += [(str(f), label) for f in files]
    return data

def load_in_the_wild(max_samples=None):
    base = Path("/kaggle/input/in-the-wild-audio-deepfake/release_in_the_wild")
    data = []
    for label, name in [(0, 'real'), (1, 'fake')]:
        files = list((base / name).glob("*.wav"))
        if max_samples:
            files = files[:max_samples]
        data += [(str(f), label) for f in files]
    return data

def load_asvspoof(max_samples=None):
    base = Path("/kaggle/input/asvspoof-2021/ASVspoof2021_LA_eval")
    protocol_path = base / "protocols/ASVspoof2021.LA.cm.eval.trl.txt"
    flac_dir = base / "flac"
    
    file_labels = {}
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                file_id = parts[1]
                label = 0 if parts[3] == 'bonafide' else 1
                file_labels[file_id + ".flac"] = label
    
    files = []
    for flac_file in flac_dir.glob("*.flac"):
        if flac_file.name in file_labels:
            files.append((str(flac_file), file_labels[flac_file.name]))
    
    if max_samples:
        files = files[:max_samples]
    
    print(f"Loaded {len(files)} ASVspoof samples (Real: {sum(1 for x in files if x[1]==0)}, Fake: {sum(1 for x in files if x[1]==1)})")
    return files

def evaluate(dataloader, name="Dataset"):
    y_true, y_prob = [], []
    
    for x_raw, x_fft, x_wav, labels in tqdm(dataloader, desc=f"Evaluating {name}"):
        x_raw = x_raw.to(device)
        x_fft = x_fft.to(device)
        x_wav = x_wav.to(device)
        labels = labels.to(device)
        
        mask = labels != -1
        if mask.sum() == 0:
            continue
            
        with torch.no_grad():
            outputs = model(x_raw[mask], x_fft[mask], x_wav[mask])
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
        y_true.extend(labels[mask].cpu().tolist())
        y_prob.extend(probs.cpu().tolist())
    
    if not y_true:
        print(f"No valid samples in {name}!")
        return
    
    # Calculate metrics
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    optimal_idx = np.argmax(tpr - fpr)
    threshold = thresholds[optimal_idx]
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    eer = calculate_eer(y_true, y_prob)
    
    print(f"\nResults for {name}:")
    print(f"Optimal Threshold: {threshold:.4f}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1: {f1_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_true, y_prob):.4f}")
    print(f"EER: {eer:.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))

if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    # Load datasets
    datasets = {
        "FOR-Original": load_for_original(max_samples=1000),
        "In-the-Wild": load_in_the_wild(max_samples=300),
        "ASVspoof": load_asvspoof(max_samples=700),
    }
    
    # Evaluate each
    for name, data in datasets.items():
        loader = DataLoader(
            AudioDataset(data),
            batch_size=128,
            num_workers=4,
            collate_fn=lambda x: (
                torch.stack([item[0] for item in x]),
                torch.stack([item[1] for item in x]),
                torch.stack([item[2] for item in x]),
                torch.tensor([item[3] for item in x])
            )
        )
        evaluate(loader, name)
    
    torch.cuda.empty_cache()