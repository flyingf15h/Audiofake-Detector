import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import pywt
import torch.nn as nn
from model import TBranchDetector
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                           precision_score, recall_score, log_loss,
                           classification_report, roc_curve)
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
model = nn.DataParallel(TBranchDetector()).to(device)
model.load_state_dict(
    torch.load(
        "/kaggle/input/audifake-detector/pytorch/default/1/best_model.pth",
        map_location=device
    )
)
model.eval()

def preprocess(audio, sr=16000):
    audio = librosa.util.fix_length(audio, size=16000)
    
    # Raw waveform
    x_raw = (audio - np.mean(audio)) / (np.std(audio) + 1e-8)
    
    # Spectrogram (match training eval params)
    stft = librosa.stft(audio, n_fft=256, hop_length=128)
    mag = np.abs(stft)[:128, :128]
    x_fft = (mag - np.mean(mag)) / (np.std(mag) + 1e-8)
    
    # Wavelet
    coeffs = pywt.wavedec(audio, 'db4', level=4)
    cA4_resized = np.resize(coeffs[0], (64, 128))
    x_wav = (cA4_resized - np.mean(cA4_resized)) / (np.std(cA4_resized) + 1e-8)
    
    return (
        torch.tensor(x_raw).unsqueeze(0).float(),
        torch.tensor(x_fft).unsqueeze(0).float(),
        torch.tensor(x_wav).unsqueeze(0).float()
    )

class ReverseFilepathDataset(Dataset):
    def __init__(self, file_label_pairs, max_samples=None):
        # Reverse the order of files to prevent overlap with training
        self.data = list(reversed(file_label_pairs))
        if max_samples and len(self.data) > max_samples:
            self.data = self.data[:max_samples]  # Take from the end
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        path, label = self.data[idx]
        try:
            if isinstance(path, dict):  # HuggingFace audio dict
                audio = librosa.load(path["array"], sr=16000)[0]
            else:  # Regular file path
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

class ReverseWaveFakeDataset(Dataset):
    def __init__(self, partitions=["partition0"], max_samples=1000):
        # Load all samples first to properly reverse
        self.data = []
        for p in partitions:
            ds = load_dataset("Keerthana982/wavefake-audio", split=p)
            for sample in reversed(ds):  # Process in reverse order
                self.data.append(sample)
                if len(self.data) >= max_samples:
                    break
            if len(self.data) >= max_samples:
                break
        self.max_samples = min(max_samples, len(self.data))

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):
        sample = self.data[idx]
        try:
            audio = sample["audio"]["array"]
            audio = librosa.util.fix_length(audio, size=16000)
            return *preprocess(audio), sample["label"]
        except Exception as e:
            print(f"Skipping sample {idx}: {str(e)}")
            return (
                torch.zeros(1, 16000), 
                torch.zeros(1, 128, 128),
                torch.zeros(1, 64, 128), 
                -1
            )

def load_for_original(max_samples=1000):
    base = "/kaggle/input/the-fake-or-real-dataset/for-original/for-original/testing"
    data = []
    for label, name in [(0, 'real'), (1, 'fake')]:
        files = sorted(Path(f"{base}/{name}").glob("*.wav"))  # Sort first
        files = list(reversed(files))  # Then reverse
        if max_samples and len(files) > max_samples:
            files = files[:max_samples]
        data += [(str(f), label) for f in files]
    return data

def load_in_the_wild(max_samples=300):
    base = "/kaggle/input/in-the-wild-audio-deepfake/release_in_the_wild"
    data = []
    for label, name in [(0, 'real'), (1, 'fake')]:
        files = sorted(Path(f"{base}/{name}").glob("*.wav"))
        files = list(reversed(files))
        if max_samples and len(files) > max_samples:
            files = files[:max_samples]
        data += [(str(f), label) for f in files]
    return data

def load_asvspoof(max_samples=1000):
    base = "/kaggle/input/asvspoof-2021/LA"
    protocol_path = f"{base}/ASVspoof2021_LA_cm_protocols/ASVspoof2021.LA.cm.eval.trl.txt"
    flac_dir = Path(f"{base}/ASVspoof2021_LA_eval/flac")
    
    # Mapping from filename to label
    file_labels = {}
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:   
                file_id = parts[1]
                label = 0 if parts[3] == 'bonafide' else 1  
                file_labels[file_id] = label
    
    files = []
    for flac_file in flac_dir.glob("*.flac"):
        file_id = flac_file.stem
        if file_id in file_labels:
            files.append((str(flac_file), file_labels[file_id]))
        else:
            print(f"Warning: No label found for {file_id}")
    
    files = list(reversed(files))
    if max_samples and len(files) > max_samples:
        files = files[:max_samples]
    
    label_counts = defaultdict(int)
    for _, label in files:
        label_counts[label] += 1
    print(f"ASVspoof 2021 Eval - Real: {label_counts[0]}, Fake: {label_counts[1]}")
    
    return files

def load_mlaad(max_samples=200):
    dataset = list(reversed(load_dataset("mueller91/MLAAD", split="test")))
    if max_samples and len(dataset) > max_samples:
        dataset = dataset[:max_samples]
    return [
        (item["audio"], 0 if item["label"] == "real" else 1)
        for item in dataset
    ]

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
    
    print(f"\nResults for {name}:")
    print(f"Optimal Threshold: {threshold:.4f}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1: {f1_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_true, y_prob):.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))

if __name__ == "__main__":
    torch.cuda.empty_cache()
    datasets = {
        "FOR-Original": load_for_original(max_samples=1000),
        "In-the-Wild": load_in_the_wild(max_samples=300),
        "ASVspoof": load_asvspoof(max_samples=700),
        "MLAAD": load_mlaad(max_samples=200),
    }
    
    for name, data in datasets.items():
        loader = DataLoader(
            ReverseFilepathDataset(data),
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
    
    wavefake_loader = DataLoader(
        ReverseWaveFakeDataset(partitions=["partition0", "partition1", "partition2"], max_samples=1000),
        batch_size=128,
        collate_fn=lambda x: (
            torch.stack([item[0] for item in x]),
            torch.stack([item[1] for item in x]),
            torch.stack([item[2] for item in x]),
            torch.tensor([item[3] for item in x])
        )
    )
    evaluate(wavefake_loader, "WaveFake")
    torch.cuda.empty_cache()