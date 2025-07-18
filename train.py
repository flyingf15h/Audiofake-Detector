import random
import numpy as np
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import (accuracy_score, roc_auc_score, 
                           classification_report, roc_curve, log_loss)
import librosa
import pywt
import multiprocessing
from model import TBranchDetector
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import gc
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Configuration
CONFIG = {
    "sample_rate": 16000,
    "batch_size": 32,
    "num_epochs": 50,
    "lr": 3e-5,
    "weight_decay": 0.05,
    "drop_rate": 0.1,
    "attn_drop_rate": 0.1,
    "patience": 5,
    "n_fft_train": 1024,
    "hop_length_train": 512,
    "n_fft_eval": 256,
    "hop_length_eval": 128,
    "data_splits": {
        "in_the_wild": 0.7,
    },
    "accumulation_steps": 4
}

def prep_input_array(audio_arr, is_training=False):
    # Audio preprocessing with train/eval consistency
    audio_arr = librosa.util.fix_length(audio_arr, size=16000)
    
    x_raw = (audio_arr - np.mean(audio_arr)) / (np.std(audio_arr) + 1e-8)
    
    n_fft = CONFIG["n_fft_train"] if is_training else CONFIG["n_fft_eval"]
    hop_length = CONFIG["hop_length_train"] if is_training else CONFIG["hop_length_eval"]
    
    stft = librosa.stft(audio_arr, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(stft)[:128, :128]

    if mag.shape[1] < 128:
        mag = np.pad(mag, ((0,0), (0,128-mag.shape[1])))
    elif mag.shape[1] > 128:
        mag = mag[:,:128]

    x_fft = (mag - np.mean(mag)) / (np.std(mag) + 1e-8)
    
    coeffs = pywt.wavedec(audio_arr, 'db4', level=4)
    cA4 = np.resize(coeffs[0], (64, 128))
    x_wav = (cA4 - np.mean(cA4)) / (np.std(cA4) + 1e-8)
    
    return (
        torch.tensor(x_raw).unsqueeze(0).float(),  
        torch.tensor(x_fft).unsqueeze(0).unsqueeze(0).float(),
        torch.tensor(x_wav).unsqueeze(0).unsqueeze(0).float()
    )

class AudioDataset(Dataset):
    def __init__(self, file_label_pairs, augment=False, is_training=False):
        self.files = file_label_pairs
        self.augment = augment
        self.is_training = is_training

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        try:
            if isinstance(path, (str, Path)):
                audio, _ = librosa.load(path, sr=CONFIG["sample_rate"])
            else: 
                audio = librosa.load(path["array"], sr=CONFIG["sample_rate"])[0]
                
            if self.augment:
                audio = self._augment(audio)
            
            x_raw, x_fft, x_wav = prep_input_array(audio, self.is_training)
            x_raw = x_raw.squeeze() 
            
            # [1, height, width]
            if x_fft.dim() == 4:
                x_fft = x_fft.squeeze(0)
            elif x_fft.dim() == 2:
                x_fft = x_fft.unsqueeze(0)
                
            # [1, height, width]
            if x_wav.dim() == 4:
                x_wav = x_wav.squeeze(0)
            elif x_wav.dim() == 2:
                x_wav = x_wav.unsqueeze(0)
                
            return x_raw, x_fft, x_wav, label
            
        except Exception as e:
            print(f"Error loading {path}: {str(e)}")
            return (
                torch.zeros(16000),        # [16000]
                torch.zeros(1, 128, 128),  # [1, 128, 128]
                torch.zeros(1, 64, 128),   # [1, 64, 128]
                0                         # label
            )

    def _augment(self, audio):
        # Time-domain augmentations# 
        if random.random() < 0.3:
            audio = np.roll(audio, random.randint(-1600, 1600))
        if random.random() < 0.3:
            audio *= random.uniform(0.8, 1.2)
        if random.random() < 0.2:
            audio += np.random.normal(0, 0.005, audio.shape)
        return np.clip(audio, -1.0, 1.0)

def load_fakeorreal():
    base_path = Path("/kaggle/input/the-fake-or-real-dataset")
    subdirs = ["for-2sec/for-2seconds", "for-rerec"]
    files = []

    for subdir in subdirs:
        dataset_path = base_path / subdir
        real_files = list((dataset_path / "training" / "real").glob("*.wav"))
        fake_files = list((dataset_path / "training" / "fake").glob("*.wav"))
        files += [(str(f), 0) for f in real_files]
        files += [(str(f), 1) for f in fake_files]

    return files

def load_inthewild(split_ratio=0.7):
    # Load in-the-wild dataset with split
    base_path = "/kaggle/input/in-the-wild-audio-deepfake/release_in_the_wild"
    realf = list(Path(f"{base_path}/real").glob("*.wav"))
    fakef = list(Path(f"{base_path}/fake").glob("*.wav"))
    
    train_real, _ = train_test_split(realf, train_size=split_ratio)
    train_fake, _ = train_test_split(fakef, train_size=split_ratio)
    
    return [(str(f), 0) for f in train_real] + [(str(f), 1) for f in train_fake]

def load_asvspoof():
    # Load ASVspoof 2019 
    base_path = "/kaggle/input/asvspoof-2019/LA"
    protocol_path = f"{base_path}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    flac_dir = Path(f"{base_path}/ASVspoof2019_LA_train/flac")
    

    print(f"Checking protocol at: {protocol_path}")
    print(f"Checking FLAC files at: {flac_dir}")


    # Read protocol file
    file_labels = {}
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5: 
                file_id = parts[1]  
                label = 0 if parts[4] == 'bonafide' else 1
                file_labels[file_id + ".flac"] = label 
    print(f"Found {len(file_labels)} entries in protocol")

    # Pair files with labels
    files = []
    missing_files = 0
    for flac_file in flac_dir.glob("*.flac"):
        if flac_file.name in file_labels:
            files.append((str(flac_file), file_labels[flac_file.name]))
        else:
            missing_files += 1
    
    print(f"Matched {len(files)} files (missing labels for {missing_files} files)")
    
    if not files:
        raise ValueError("No valid files found - check protocol/flac matching")
    
    return files

def get_valset():
    # Get FOR validation set
    base_path = "/kaggle/input/the-fake-or-real-dataset"
    val_data = []
    val_paths = [
        f"{base_path}/for-2sec/for-2seconds/validation",
        f"{base_path}/for-rerec/for-rerecorded/validation"
    ]

    for val_path in val_paths:
        val_path = Path(val_path)
        if val_path.exists():
            real_files = list((val_path / "real").glob("*.wav"))
            fake_files = list((val_path / "fake").glob("*.wav"))
            val_data += [(str(f), 0) for f in real_files] + [(str(f), 1) for f in fake_files]
    
    if not val_data:
        print("Warning: No validation files found")
    
    return val_data

class HybridLoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.gamma = 2
        self.class_weights = class_weights 
        
    def forward(self, logits, targets):
        ce_loss = self.ce_loss(logits, targets)

        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal_loss = ((1 - pt) ** self.gamma) * ce

        if self.class_weights is not None:
            focal_loss = focal_loss * self.class_weights[targets]
            
        focal_loss = focal_loss.mean()
        
        return 0.7 * ce_loss + 0.3 * focal_loss

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    scaler = GradScaler()
    
    for i, (x_raw, x_fft, x_wav, y) in enumerate(loader):
        if x_raw.dim() == 2:
            x_raw = x_raw.unsqueeze(1)

        x_raw, x_fft, x_wav = x_raw.to(device), x_fft.to(device), x_wav.to(device)
        y = y.to(device)
        
        # Forward pass with autocast
        with autocast(): 
            logits = model(x_raw, x_fft, x_wav)
            loss = criterion(logits, y) / CONFIG["accumulation_steps"]
        
        # Backward pass with scaler
        scaler.scale(loss).backward()
        
        # Update weights
        if (i + 1) % CONFIG["accumulation_steps"] == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * x_raw.size(0) * CONFIG["accumulation_steps"]
        
        # Memory cleanup
        del logits, loss, x_raw, x_fft, x_wav, y
        torch.cuda.empty_cache()
        gc.collect()
    
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    
    with torch.no_grad():
        for x_raw, x_fft, x_wav, y in loader:
            x_raw, x_fft, x_wav = x_raw.to(device), x_fft.to(device), x_wav.to(device)
            
            chunk_size = 8  
            for i in range(0, x_raw.size(0), chunk_size):
                chunk_raw = x_raw[i:i+chunk_size]
                chunk_fft = x_fft[i:i+chunk_size]
                chunk_wav = x_wav[i:i+chunk_size]
                
                logits = model(chunk_raw, chunk_fft, chunk_wav)
                prob = torch.softmax(logits, dim=1)[:, 1]
                y_true.extend(y[i:i+chunk_size].cpu().numpy())
                y_prob.extend(prob.cpu().numpy())
                
                del chunk_raw, chunk_fft, chunk_wav, logits, prob
                torch.cuda.empty_cache()
    
    # Calculate metrics
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    optimal_idx = np.argmax(tpr - fpr)
    threshold = thresholds[optimal_idx]
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    
    return {
        "loss": log_loss(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "report": classification_report(y_true, y_pred, target_names=['Real', 'Fake']),
        "threshold": threshold,
        "y_true": y_true,
        "y_prob": y_prob
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load all datasets
    print("Loading datasets")
    train_data = load_fakeorreal() + load_inthewild(CONFIG["data_splits"]["in_the_wild"]) + load_asvspoof()
    val_data = get_valset()
    
    # Count classes for weighting
    class_counts = defaultdict(int)
    for _, label in train_data:
        class_counts[label] += 1
    total = sum(class_counts.values())
    class_weights = torch.tensor([
        class_counts[0]/total, 
        class_counts[1]/total 
    ], device=device)
    
    train_ds = AudioDataset(train_data, augment=True, is_training=True)
    val_ds = AudioDataset(val_data, augment=False)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=multiprocessing.cpu_count()
    )
    
    model = TBranchDetector(
        drop_rate=CONFIG["drop_rate"],
        attn_drop_rate=CONFIG["attn_drop_rate"]
    ).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")
        
    # Optimization
    criterion = HybridLoss(class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=1e-5,
        max_lr=3e-4,
        step_size_up=500,
        mode='exp_range'
    )
    
    # Training
    best_auc = 0.0
    patience_counter = 0
    
    for epoch in range(1, CONFIG["num_epochs"] + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        
        print(f"\nEpoch {epoch}/{CONFIG['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Accuracy: {val_metrics['accuracy']:.4f} | AUC: {val_metrics['auc']:.4f}")
        print(f"Optimal Threshold: {val_metrics['threshold']:.4f}")
        print(val_metrics["report"])
        
        # Save best model
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
            print("*** New best model saved ***")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break
        
        torch.cuda.empty_cache()
        gc.collect()
    
    # Final evaluation
    model.load_state_dict(torch.load("best_model.pth"))
    final_metrics = evaluate(model, val_loader, device)
    
    print("\nFinal Evaluation")
    print(f"Best AUC: {best_auc:.4f}")
    print(f"Optimal Threshold: {final_metrics['threshold']:.4f}")
    print(final_metrics["report"])
    
    # Plot ROC
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(final_metrics["y_true"], final_metrics["y_prob"])
    plt.plot(fpr, tpr, label=f'AUC = {final_metrics["auc"]:.4f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('roc_curve.png')

if __name__ == "__main__":
    main()