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
import json
import matplotlib.pyplot as plt
from datetime import datetime
import os

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
    audio = librosa.util.fix_length(audio, size=16000)
    
    # Raw waveform
    x_raw = (audio - np.mean(audio)) / (np.std(audio) + 1e-8)
    
    # Spectrogram (match training params)
    stft = librosa.stft(audio, n_fft=512, hop_length=256)
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

def load_deep_voice(max_samples=None):
    base = Path("/kaggle/input/deep-voice-deepfake-voice-recognition/KAGGLE/AUDIO")
    data = []
    
    # Assuming structure: /AUDIO/real/... and /AUDIO/fake/...
    for label, name in [(0, 'real'), (1, 'fake')]:
        files = list((base / name).glob("*.wav")) + list((base / name).glob("*.mp3"))
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
        return None
    
    # Calculate metrics
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    optimal_idx = np.argmax(tpr - fpr)
    threshold = thresholds[optimal_idx]
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    eer = calculate_eer(y_true, y_prob)
    
    metrics = {
        "dataset": name,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "eer": float(eer),
        "classification_report": classification_report(y_true, y_pred, target_names=['Real', 'Fake'], output_dict=True),
        "num_samples": len(y_true),
        "timestamp": datetime.now().isoformat()
    }
    
    # Print results
    print(f"\nResults for {name}:")
    print(f"Optimal Threshold: {metrics['threshold']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"EER: {metrics['eer']:.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {metrics['auc']:.4f}\nEER = {metrics['eer']:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend(loc='lower right')
    roc_path = f"/kaggle/working/roc_{name.lower().replace('-', '_')}.png"
    plt.savefig(roc_path)
    plt.close()
    metrics["roc_curve_path"] = roc_path
    
    return metrics

if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    # Create output directory
    os.makedirs("/kaggle/working/benchmark_results", exist_ok=True)
    
    # Load datasets
    datasets = {
        "FOR-Original": load_for_original(max_samples=1000),
        "ASVspoof-2021": load_asvspoof(max_samples=700),
        "Deep-Voice": load_deep_voice(max_samples=1000),
    }
    
    all_metrics = {}
    
    # Evaluate each dataset
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
        metrics = evaluate(loader, name)
        if metrics:
            all_metrics[name] = metrics
    
    # Save all metrics to JSON
    metrics_path = "/kaggle/working/benchmark_results/metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved all metrics to {metrics_path}")
    
    # Generate summary report
    report_path = "/kaggle/working/benchmark_results/summary.txt"
    with open(report_path, 'w') as f:
        f.write("Audio Deepfake Benchmark Results\n")
        f.write("="*40 + "\n\n")
        f.write(f"Evaluation timestamp: {datetime.now().isoformat()}\n\n")
        
        for name, metrics in all_metrics.items():
            f.write(f"Dataset: {name}\n")
            f.write("-"*40 + "\n")
            f.write(f"Num Samples: {metrics['num_samples']}\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"F1 Score: {metrics['f1']:.4f}\n")
            f.write(f"AUC: {metrics['auc']:.4f}\n")
            f.write(f"EER: {metrics['eer']:.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(classification_report(
                y_true=None, y_pred=None,
                target_names=['Real', 'Fake'],
                output_dict=False,
                **metrics['classification_report']
            ))
            f.write("\n\n")
    
    print(f"Saved summary report to {report_path}")
    torch.cuda.empty_cache()