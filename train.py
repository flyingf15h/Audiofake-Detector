import random
import numpy as np
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, roc_auc_score, 
                           classification_report, roc_curve)
import pywt
import json
from model import TBranchDetector
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import gc
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import os
import torch.backends.cudnn
import hashlib
from torch_lr_finder import LRFinder
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
mp.set_start_method('spawn', force=True)

# Configuration
CONFIG = {
    "sample_rate": 16000,
    "batch_size": 32,
    "num_epochs": 50,
    "lr": 3e-4,
    "weight_decay": 0.05,
    "drop_rate": 0.1,
    "attn_drop_rate": 0.1,
    "patience": 8,
    "n_fft": 512,
    "hop_length": 256,
    "data_splits": {
        "in_the_wild": 0.7,
    },
    "accumulation_steps": 4
}

class DeviceSpectrogram(nn.Module):
    def __init__(self, n_fft, hop_length, power, normalized):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power
        self.normalized = normalized
        self.register_buffer('window', torch.hann_window(n_fft))
        
    def forward(self, x):
        x = x.to(self.window.device)
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            normalized=self.normalized,
            return_complex=True
        ).abs().pow(2)  

        
    def to(self, device):
        super().to(device)
        self.window = self.window.to(device)
        return self

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPEC = DeviceSpectrogram(
    n_fft=CONFIG["n_fft"],
    hop_length=CONFIG["hop_length"],
    power=2.0,
    normalized=True
).to(device)

def prep_input_array(audio_tensor, is_training=False):
    device = audio_tensor.device
    
    # Ensure audio is 1D and exactly CONFIG["sample_rate"] length
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.squeeze()
    
    # Pad or truncate to exact sample rate
    target_length = CONFIG["sample_rate"]
    if audio_tensor.size(-1) < target_length:
        pad_size = target_length - audio_tensor.size(-1)
        audio_tensor = F.pad(audio_tensor, (0, pad_size))
    elif audio_tensor.size(-1) > target_length:
        audio_tensor = audio_tensor[:target_length]
    
    # Normalize
    x_raw = (audio_tensor - audio_tensor.mean()) / (audio_tensor.std() + 1e-8)
    
    # FFT processing - ensure consistent size
    x_fft = SPEC(x_raw)
    x_fft = torch.sqrt(x_fft + 1e-6)
    
    # Ensure FFT is exactly [257, 63] (n_fft//2+1, frames)
    if x_fft.size(0) > 128:
        x_fft = x_fft[:128, :]
    if x_fft.size(1) > 128:
        x_fft = x_fft[:, :128]
    
    # Pad if necessary
    if x_fft.size(0) < 128:
        pad_freq = 128 - x_fft.size(0)
        x_fft = F.pad(x_fft, (0, 0, 0, pad_freq))
    if x_fft.size(1) < 128:
        pad_time = 128 - x_fft.size(1)
        x_fft = F.pad(x_fft, (0, pad_time))
    
    # Normalize FFT
    x_fft = (x_fft - x_fft.mean()) / (x_fft.std() + 1e-8)
    
    # Wavelet processing
    audio_np = x_raw.cpu().numpy()
    coeffs = pywt.wavedec(audio_np, 'db4', level=4)
    cA4 = coeffs[0]
    
    # Ensure wavelet coefficients are exactly 64*128 = 8192
    target_wav_length = 64 * 128
    if len(cA4) < target_wav_length:
        cA4 = np.pad(cA4, (0, target_wav_length - len(cA4)))
    elif len(cA4) > target_wav_length:
        cA4 = cA4[:target_wav_length]
    
    x_wav = torch.tensor(cA4.reshape(64, 128), dtype=torch.float32).to(device)
    x_wav = (x_wav - x_wav.mean()) / (x_wav.std() + 1e-8)
    
    return x_raw, x_fft, x_wav

class CachedAudioDataset(Dataset):
    def __init__(self, data, cache_dir, augment=False, transform=None):
        self.data = data
        self.transform = transform
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.augment = augment
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __getitem__(self, idx):
        path, label = self.data[idx]
        uid = hashlib.md5(path.encode()).hexdigest()
        cache_path = self.cache_dir / f"{uid}.pt"

        if cache_path.exists():
            try:
                cached_data = torch.load(cache_path, map_location='cpu')
                return (
                    cached_data["x_raw"],
                    cached_data["x_fft"],
                    cached_data["x_wav"],
                    cached_data["label"]
                )
            except Exception as e:
                print(f"Error loading cached file {cache_path}: {e}")

        try:
            waveform, sr = torchaudio.load(path)
            if sr != CONFIG["sample_rate"]:
                waveform = torchaudio.functional.resample(waveform, sr, CONFIG["sample_rate"])

            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            waveform = waveform.squeeze()

            target_len = CONFIG["sample_rate"]
            if waveform.size(0) < target_len:
                waveform = F.pad(waveform, (0, target_len - waveform.size(0)))
            elif waveform.size(0) > target_len:
                waveform = waveform[:target_len]

            if self.augment:
                waveform = self._augment(waveform)

            x_raw, x_fft, x_wav = prep_input_array(waveform, is_training=self.augment)

            sample = {
                "x_raw": x_raw.cpu(),
                "x_fft": x_fft.cpu(),
                "x_wav": x_wav.cpu(),
                "label": torch.tensor(label, dtype=torch.long),
            }

            torch.save(sample, cache_path)

            return (
                sample["x_raw"],
                sample["x_fft"],
                sample["x_wav"],
                sample["label"]
            )

        except Exception as e:
            print(f"Error processing {path}: {e}")
            return (
                torch.zeros(CONFIG["sample_rate"]),
                torch.zeros(128, 128),
                torch.zeros(64, 128),
                torch.tensor(0, dtype=torch.long)
            )

    def __len__(self):
        return len(self.data)

    def _augment(self, audio: torch.Tensor) -> torch.Tensor:
        if isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio

        if random.random() < 0.2:
            audio_np = np.roll(audio_np, random.randint(-800, 800))
        if random.random() < 0.2:
            audio_np *= random.uniform(0.95, 1.05)
        if random.random() < 0.2:
            audio_np += np.random.normal(0, 0.005, audio_np.shape)

        return torch.tensor(np.clip(audio_np, -1.0, 1.0), dtype=torch.float32)

    
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
                audio, sr = torchaudio.load(path)
                if sr != CONFIG["sample_rate"]:
                    resampler = T.Resample(orig_freq=sr, new_freq=CONFIG["sample_rate"])
                    audio = resampler(audio)
            else: 
                audio, sr = torchaudio.load(path["array"])
                if sr != CONFIG["sample_rate"]:
                    resampler = T.Resample(orig_freq=sr, new_freq=CONFIG["sample_rate"])
                    audio = resampler(audio)
            
            # Convert to mono and remove batch dimension
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            audio = audio.squeeze()
                                
            if self.augment:
                audio = self._augment(audio)
            
            # Use standardized preprocessing
            x_raw, x_fft, x_wav = prep_input_array(audio, self.is_training)
            if x_raw.size(0) != CONFIG["sample_rate"]:
                if x_raw.size(0) < CONFIG["sample_rate"]:
                    x_raw = F.pad(x_raw, (0, CONFIG["sample_rate"] - x_raw.size(0)))
                else:
                    x_raw = x_raw[:CONFIG["sample_rate"]]
 
            return (
                x_raw.cpu(),      # [16000]
                x_fft.cpu(),      # [128, 128]
                x_wav.cpu(),      # [64, 128]
                label
            )
            
        except Exception as e:
            print(f"Error loading {path}: {str(e)}")
            return (
                torch.zeros(CONFIG["sample_rate"]),      # [16000]
                torch.zeros(128, 128),                   # [128, 128]
                torch.zeros(64, 128),                    # [64, 128]
                0                       
            )   

    def _augment(self, audio):
        audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else audio
        audio_np = (audio_np - np.mean(audio_np)) / (np.std(audio_np) + 1e-8)  
        
        if random.random() < 0.3:
            audio_np = np.roll(audio_np, random.randint(-1600, 1600))
        if random.random() < 0.3:
            audio_np *= random.uniform(0.8, 1.2)
        if random.random() < 0.2:
            audio_np += np.random.normal(0, 0.005, audio_np.shape)
            
        return torch.tensor(np.clip(audio_np, -1.0, 1.0), dtype=torch.float32)

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
    base_path = "/kaggle/input/in-the-wild-audio-deepfake/release_in_the_wild"
    realf = list(Path(f"{base_path}/real").glob("*.wav"))
    fakef = list(Path(f"{base_path}/fake").glob("*.wav"))
    
    train_real, _ = train_test_split(realf, train_size=split_ratio)
    train_fake, _ = train_test_split(fakef, train_size=split_ratio)
    
    return [(str(f), 0) for f in train_real] + [(str(f), 1) for f in train_fake]

def load_asvspoof():
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
        self.gamma = 1.5
        self.class_weights = class_weights 
        
    def forward(self, logits, targets):
        ce_loss = self.ce_loss(logits, targets)

        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal_loss = ((1 - pt) ** self.gamma) * ce
        focal_loss = focal_loss.mean()

        if self.class_weights is not None:
            focal_loss = focal_loss * self.class_weights[targets]
            
        focal_loss = focal_loss.mean()
        
        return 0.5 * ce_loss + 0.5 * focal_loss

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    scaler = GradScaler()
    
    for i, (x_raw, x_fft, x_wav, y) in enumerate(loader):
        x_raw = x_raw.float().to(device)
        x_fft = x_fft.float().to(device)
        x_wav = x_wav.float().to(device)
        y = y.to(device)
        
        # Ensure correct dimensions
        if x_raw.dim() == 2:  # [B, 16000] -> [B, 1, 16000]
            x_raw = x_raw.unsqueeze(1)
        if x_fft.dim() == 3:  # [B, 128, 128] -> [B, 1, 128, 128]
            x_fft = x_fft.unsqueeze(1)
        if x_wav.dim() == 3:  # [B, 64, 128] -> [B, 1, 64, 128]
            x_wav = x_wav.unsqueeze(1)
        
        # Forward pass with autocast
        amp_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        with autocast(device_type='cuda', dtype=amp_dtype):
            logits = model(x_raw, x_fft, x_wav)
            loss = criterion(logits, y)
        
        scaler.scale(loss).backward()
        
        # Update weights
        if (i + 1) % CONFIG["accumulation_steps"] == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * x_raw.size(0) * CONFIG["accumulation_steps"]
        
        # Memory cleanup
        del logits, loss, x_raw, x_fft, x_wav, y
        torch.cuda.empty_cache()
        if i % 10 == 0:
            gc.collect()
    
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    y_true, y_prob = [], []
    total_loss = 0.0
    
    amp_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    with torch.no_grad(), autocast(device_type='cuda', dtype=amp_dtype):
        for x_raw, x_fft, x_wav, y in loader:
            x_raw = x_raw.unsqueeze(1).to(device)
            x_fft = x_fft.unsqueeze(1).to(device)
            x_wav = x_wav.unsqueeze(1).to(device)
            y = y.to(device)
            
            logits = model(x_raw, x_fft, x_wav)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            
            prob = torch.softmax(logits, dim=1)[:, 1]
            y_true.extend(y.cpu().numpy())
            y_prob.extend(prob.cpu().numpy())
    
    # Calculate metrics
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    optimal_idx = np.argmax(tpr - fpr)
    threshold = thresholds[optimal_idx]
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "report": classification_report(y_true, y_pred, target_names=['Real', 'Fake']),
        "threshold": threshold,
        "y_true": y_true,
        "y_prob": y_prob
    }

def collate_fn(batch):
    try:
        return (
            torch.stack([item[0] for item in batch]),
            torch.stack([item[1] for item in batch]),
            torch.stack([item[2] for item in batch]),
            torch.tensor([item[3] for item in batch])
        )
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        raise

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True
    val_losses = []
    
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
    
    train_ds = CachedAudioDataset(train_data, cache_dir="train_cache", augment=True)
    val_ds = CachedAudioDataset(val_data, cache_dir="val_cache", augment=False)
     
    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        multiprocessing_context='spawn',
        persistent_workers=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=4,
        multiprocessing_context='spawn',
        collate_fn=collate_fn
    )

    model = TBranchDetector(
        drop_rate=CONFIG["drop_rate"],
        attn_drop_rate=CONFIG["attn_drop_rate"]
    ).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")
        
    print("\nVerifying batch shapes:")
    test_batch = next(iter(train_loader))
    print(f"Raw audio: {test_batch[0].shape}")
    print(f"FFT: {test_batch[1].shape}")
    print(f"Wavelet: {test_batch[2].shape}")
    
    with torch.no_grad():
        test_inputs = (
            test_batch[0].unsqueeze(1).to(device),  # [B,1,16000]
            test_batch[1].unsqueeze(1).to(device),  # [B,1,128,128]
            test_batch[2].unsqueeze(1).to(device)   # [B,1,64,128]
        )
        test_output = model(*test_inputs)
        print(f"Model test output shape: {test_output.shape}")

    # Optimization
    criterion = HybridLoss(class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )

    # Get optimized lr 
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=1, num_iter=100)
    lr_finder.plot() 
    plt.savefig('lr_finder.png')
    optimal_lr = lr_finder.suggestion()
    print(f"Suggested LR: {optimal_lr}")
    lr_finder.reset()
    CONFIG["lr"] = optimal_lr

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG["lr"],
        steps_per_epoch=len(train_loader),
        epochs=CONFIG["num_epochs"],
        pct_start=0.3
    )
    
    # Training
    best_auc = 0.0
    patience_counter = 0
    
    for epoch in range(1, CONFIG["num_epochs"] + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_metrics["loss"])
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
            print("**New best model saved**")
            torch.save(model.state_dict(), "/kaggle/working/best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break
        
        torch.cuda.empty_cache()
        gc.collect()
    
    # Final evaluation
    model.load_state_dict(torch.load("best_model.pth"))
    final_metrics = evaluate(model, val_loader, criterion, device)
    
    print("\nFinal Evaluation")
    print(f"Best AUC: {best_auc:.4f}")
    print(f"Optimal Threshold: {final_metrics['threshold']:.4f}")
    print(final_metrics["report"])
    
    metrics = {
        "Best AUC": best_auc,
        "Final Threshold": final_metrics["threshold"],
        "Classification Report": final_metrics["report"]
    }

    with open("/kaggle/working/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot ROC
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(final_metrics["y_true"], final_metrics["y_prob"])
    plt.plot(fpr, tpr, label=f'AUC = {final_metrics["auc"]:.4f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('roc_curve.png')
    plt.savefig("/kaggle/working/roc_curve.png")
    
    # Val loss plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Over Epochs")
    plt.grid(True)
    plt.savefig("val_loss_curve.png")
    plt.savefig("/kaggle/working/val_loss_curve.png")


if __name__ == "__main__":
    main()