import os
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
import pywt
from model import TBranchDetector
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from collections import Counter
import random
import glob
import warnings
import multiprocessing
from pathlib import Path
import json

try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except RuntimeError:
    pass

class CachedAudioDataset(Dataset):
    def __init__(self, cache_dir, augment=False):
        self.cache_dir = Path(cache_dir)
        self.augment = augment
        with open(self.cache_dir/'metadata.json') as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        entry = self.metadata[idx]
        features = np.load(self.cache_dir/entry['feature_path'], allow_pickle=True).item()
        
        x_raw = torch.tensor(features['raw'], dtype=torch.float32)
        x_fft = torch.tensor(features['stft'], dtype=torch.float32) 
        x_wav = torch.tensor(features['wav'], dtype=torch.float32)
        
        if self.augment:
            x_raw = self._augment(x_raw)
            
        return x_raw, x_fft, x_wav, torch.tensor(entry['label'], dtype=torch.long)

    def _augment(self, x_raw):
        # Your existing augmentation code
        if random.random() < 0.3:
            shift = random.randint(-1600, 1600)
            x_raw = torch.roll(x_raw, shifts=shift, dims=1)
        # ... rest of augmentation
        return x_raw
    
def loadfiles(data_dir, target_splits):
    all_files = []
    subdir = 'for-2sec/for-2sec'  
    
    for split in target_splits:
        fake_path = os.path.join(data_dir, subdir, split, 'fake')
        real_path = os.path.join(data_dir, subdir, split, 'real')
        
        if os.path.exists(fake_path):
            fake_files = []
            for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
                fake_files.extend(glob.glob(os.path.join(fake_path, ext)))
            print(f"Found {len(fake_files)} fake files in {fake_path}")
            all_files.extend([(f, 1) for f in fake_files])
        
        if os.path.exists(real_path):
            real_files = []
            for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
                real_files.extend(glob.glob(os.path.join(real_path, ext)))
            print(f"Found {len(real_files)} real files in {real_path}")
            all_files.extend([(f, 0) for f in real_files])
    
    print(f"Total files loaded from {target_splits}: {len(all_files)}")
    if len(all_files) > 0:
        fake_count = sum(1 for _, label in all_files if label == 1)
        real_count = sum(1 for _, label in all_files if label == 0)
        print(f"Fake files: {fake_count}, Real files: {real_count}")
    
    random.shuffle(all_files)
    return all_files

# Shape validation for DatasetFolder.__getitem__
class DatasetFolder(Dataset):
    def __init__(self, file_label_pairs, augment=False):
        self.data = file_label_pairs
        self.augment_flag = augment
        self.failed_files = set()

    def __getitem__(self, idx):
        path, label = self.data[idx]
        
        try:
            audioArr, sr = librosa.load(path, sr=16000)
            if audioArr is None or len(audioArr) == 0:
                raise ValueError("Empty audio array")
                
            x_raw, x_fft, x_wav = prepInputArray(audioArr)
            
            if x_raw.shape != (1, 16000):
                raise ValueError(f"Invalid raw shape: {x_raw.shape}")
            if x_fft.shape != (1, 128, 128):
                raise ValueError(f"Invalid FFT shape: {x_fft.shape}")
            if x_wav.shape != (1, 64, 128):
                raise ValueError(f"Invalid Wave shape: {x_wav.shape}")
                
        except Exception as e:
            if path not in self.failed_files:
                print(f"Error processing {path}: {e}")
                self.failed_files.add(path)
            x_raw = np.zeros((1, 16000), dtype=np.float32)
            x_fft = np.zeros((1, 128, 128), dtype=np.float32)
            x_wav = np.zeros((1, 64, 128), dtype=np.float32)
            
        return (
            torch.tensor(x_raw, dtype=torch.float32),
            torch.tensor(x_fft, dtype=torch.float32),
            torch.tensor(x_wav, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )

def train_1epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    valid_samples = 0
    
    for batch_idx, (x_raw, x_fft, x_wav, y) in enumerate(dataloader):
        try:
            # Validate batch shapes
            assert x_raw.shape[1:] == (1, 16000), f"Bad raw shape: {x_raw.shape}"
            assert x_fft.shape[1:] == (1, 128, 128), f"Bad FFT shape: {x_fft.shape}"
            assert x_wav.shape[1:] == (1, 64, 128), f"Bad Wave shape: {x_wav.shape}"
            
            x_raw, x_fft, x_wav, y = x_raw.to(device), x_fft.to(device), x_wav.to(device), y.to(device)
            
            # Forward pass with shape checks
            outputs = model(x_raw, x_fft, x_wav)
            loss = criterion(outputs, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * y.size(0)
            valid_samples += y.size(0)
            
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
            
    return running_loss / max(valid_samples, 1)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, (x_raw, x_fft, x_wav, y) in enumerate(dataloader):
            try:
                x_raw, x_fft, x_wav, y = x_raw.to(device), x_fft.to(device), x_wav.to(device), y.to(device)
                logits = model(x_raw, x_fft, x_wav)
                probs = torch.softmax(logits, dim=1)[:, 1]  # probability of class 1 (fake)
                all_preds.append(probs.cpu())
                all_labels.append(y.cpu())
            except Exception as e:
                print(f"Error in evaluation batch {batch_idx}: {e}")
                continue
    
    if len(all_preds) == 0:
        return 0.0, float('nan'), [], []
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    pred_labels = (all_preds >= 0.5).astype(int)
    acc = accuracy_score(all_labels, pred_labels)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = float('nan')
    
    return acc, auc, all_labels, pred_labels


class DatasetFolder(Dataset):
    def __init__(self, file_label_pairs, augment=False):
        self.data = file_label_pairs
        self.augment_flag = augment
        self.failed_files = set()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]

        try:
            audioArr, sr = librosa.load(path, sr=16000)
            if audioArr is None or len(audioArr) == 0:
                raise ValueError("Empty audio array")
        except Exception as e:
            if path not in self.failed_files:
                print(f"Failed to load {path}: {e}")
                self.failed_files.add(path)
            audioArr = np.zeros(16000, dtype=np.float32)

        if self.augment_flag:
            audioArr = self.augmentAudio(audioArr)

        try:
            x_raw, x_fft, x_wav = prepInputArray(audioArr)
        except Exception as e:
            if path not in self.failed_files:
                print(f"Error preprocessing {path}: {e}")
                self.failed_files.add(path)
            x_raw = np.zeros((1, 16000), dtype=np.float32)
            x_fft = np.zeros((1, 128, 128), dtype=np.float32)
            x_wav = np.zeros((1, 64, 128), dtype=np.float32)

        return (
            torch.tensor(x_raw, dtype=torch.float32),
            torch.tensor(x_fft, dtype=torch.float32),
            torch.tensor(x_wav, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )

    def augmentAudio(self, audio):
        try:
            if random.random() < 0.3:
                shift = random.randint(-1600, 1600)
                audio = np.roll(audio, shift)
            if random.random() < 0.3:
                scale = random.uniform(0.8, 1.2)
                audio *= scale
            if random.random() < 0.2:
                noise = np.random.normal(0, 0.005, audio.shape)
                audio += noise
        except Exception:
            pass
        return audio

def collate_fn(batch):
    raws, ffts, wavs, labels = zip(*batch)
    return torch.stack(raws), torch.stack(ffts), torch.stack(wavs), torch.tensor(labels, dtype=torch.long)

def prepInputArray(audioArr, sr=16000, fixed_length=16000):
    audioArr = librosa.util.fix_length(audioArr, fixed_length).astype(np.float32)
    x_raw = (audioArr - np.mean(audioArr)) / (np.std(audioArr) + 1e-8)
    x_raw = np.expand_dims(x_raw, axis=0)

    try:
        stft = librosa.stft(audioArr, n_fft=256, hop_length=128)
        mag = np.abs(stft)[:128, :128]
        x_fft = (mag - np.mean(mag)) / (np.std(mag) + 1e-8)
        x_fft = np.expand_dims(x_fft, axis=0)
    except Exception:
        x_fft = np.zeros((1, 128, 128), dtype=np.float32)

    try:
        coeffs = pywt.wavedec(audioArr, 'db4', level=4)
        cA4_resized = np.resize(coeffs[0], (64, 128))
        x_wav = (cA4_resized - np.mean(cA4_resized)) / (np.std(cA4_resized) + 1e-8)
        x_wav = np.expand_dims(x_wav, axis=0)
    except Exception:
        x_wav = np.zeros((1, 64, 128), dtype=np.float32)

    return x_raw, x_fft, x_wav

def main():
    warnings.filterwarnings("ignore")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Kaggle 
    kaggle_train = CachedAudioDataset("/kaggle/input/fake-or-real-preprocessed/train", augment=True)
    kaggle_val = CachedAudioDataset("/kaggle/input/fake-or-real-preprocessed/val", augment=False)
    
    # Load Hugging Face 
    hf_train = CachedAudioDataset("/kaggle/input/hf-dataset-cached/train", augment=True)
    hf_val = CachedAudioDataset("/kaggle/input/hf-dataset-cached/val", augment=False)
    
    train_ds = torch.utils.data.ConcatDataset([kaggle_train, hf_train])
    val_ds = torch.utils.data.ConcatDataset([kaggle_val, hf_val])
    
    # Calculate class weights based on combined data
    all_labels = [d[3] for d in train_ds] 
    class_counts = Counter(all_labels)
    total = sum(class_counts.values())
    class_weights = torch.tensor([
        total / (2 * class_counts[0]),
        total / (2 * class_counts[1])
    ], device=device)
    

    batch_size = 64
    num_workers = min(4, multiprocessing.cpu_count())  

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    model = TBranchDetector().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=(4e-4)*batch_size/16, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    best_loss, patience = float('inf'), 10
    epochs_noImprove = 0
    epochs = 109

    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss = train_1epoch(model, train_loader, criterion, optimizer, device)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for x_raw, x_fft, x_wav, y in val_loader:
                x_raw, x_fft, x_wav, y = x_raw.to(device), x_fft.to(device), x_wav.to(device), y.to(device)
                out = model(x_raw, x_fft, x_wav)
                val_loss += criterion(out, y).item() * y.size(0)
        val_loss /= len(val_loader.dataset)

        acc, auc, _, _ = evaluate(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f} | Acc: {acc:.4f} | AUC: {auc:.4f}")
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_noImprove = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("  *** New best model saved ***")
        else:
            epochs_noImprove += 1
            if epochs_noImprove >= patience:
                print("Early stopping triggered.")
                break

    print("\nFinal Evaluation")
    model.load_state_dict(torch.load("best_model.pth"))
    acc, auc, true_labels, pred_labels = evaluate(model, val_loader, device)
    print(f"Final Accuracy: {acc:.4f}, ROC AUC: {auc:.4f}")
    print(classification_report(true_labels, pred_labels, target_names=['Real', 'Fake']))

if __name__ == "__main__":
    main()