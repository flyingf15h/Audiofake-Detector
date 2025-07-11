import os
import random
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from model import TBranchDetector

class CachedAudioDataset(Dataset):
    def __init__(self, folder_path, augment=False):
        self.files = list(Path(folder_path).glob("*.npz"))
        self.augment_flag = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            data = np.load(self.files[idx], allow_pickle=True)
            x_raw = torch.tensor(data['raw'], dtype=torch.float32)
            x_fft = torch.tensor(data['stft'], dtype=torch.float32)
            x_wav = torch.tensor(data['wav'], dtype=torch.float32)
            label = torch.tensor(data['label'], dtype=torch.long)

            if self.augment_flag:
                x_raw = self.augment(x_raw)

            return x_raw, x_fft, x_wav, label
        except Exception as e:
            print(f"Failed to load {self.files[idx]}: {e}")
            return (
                torch.zeros((1, 16000)), 
                torch.zeros((1, 128, 128)),
                torch.zeros((1, 64, 128)), 
                torch.tensor(0)
            )

    def augment(self, x_raw):
        try:
            audio = x_raw.squeeze().numpy()
            if random.random() < 0.3:
                shift = random.randint(-1600, 1600)
                audio = np.roll(audio, shift)
            if random.random() < 0.3:
                audio *= random.uniform(0.8, 1.2)
            if random.random() < 0.2:
                audio += np.random.normal(0, 0.005, audio.shape)
            audio = np.clip(audio, -1.0, 1.0)
            x_raw = torch.tensor((audio - np.mean(audio)) / (np.std(audio) + 1e-8)).unsqueeze(0)
        except Exception:
            pass
        return x_raw


def collate_fn(batch):
    raws, ffts, wavs, labels = zip(*batch)
    return torch.stack(raws), torch.stack(ffts), torch.stack(wavs), torch.stack(labels)

def train_1epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, count = 0, 0
    for x_raw, x_fft, x_wav, y in dataloader:
        x_raw, x_fft, x_wav, y = x_raw.to(device), x_fft.to(device), x_wav.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x_raw, x_fft, x_wav)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        count += y.size(0)
    return total_loss / max(count, 1)

def evaluate(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x_raw, x_fft, x_wav, y in dataloader:
            x_raw, x_fft, x_wav = x_raw.to(device), x_fft.to(device), x_wav.to(device)
            logits = model(x_raw, x_fft, x_wav)
            prob = torch.softmax(logits, dim=1)[:, 1]
            preds.append(prob.cpu())
            labels.append(y.cpu())

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    acc = accuracy_score(labels, preds >= 0.5)
    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = float('nan')
    return acc, auc, labels, (preds >= 0.5).astype(int)

def main():
    warnings.filterwarnings("ignore")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datasets
    train_ds = CachedAudioDataset("/kaggle/input/fake-or-real-preprocessed/train", augment=True)
    val_ds = CachedAudioDataset("/kaggle/input/fake-or-real-preprocessed/val", augment=False)

    # Calculate class weights
    labels = []
    for file in Path("/kaggle/input/fake-or-real-preprocessed/train").glob("*.npz"):
        data = np.load(file, allow_pickle=True)
        labels.append(data['label'])
    class_counts = Counter(labels)
    total = sum(class_counts.values())
    class_weights = torch.tensor([
        total / (2 * class_counts[0]),
        total / (2 * class_counts[1])
    ], device=device)

    # Create dataloaders
    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          collate_fn=collate_fn, num_workers=2)

    # Model setup
    model = TBranchDetector().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=4e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    # Training loop
    best_loss = float("inf")
    patience = 10
    for epoch in range(1, 50):
        print(f"\nEpoch {epoch}/50")
        train_loss = train_1epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")

        # Validation
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
            torch.save(model.state_dict(), "best_model.pth")
            print("  *** New best model saved ***")
            patience = 10  # Reset patience
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping.")
                break

    # Final evaluation
    model.load_state_dict(torch.load("best_model.pth"))
    acc, auc, y_true, y_pred = evaluate(model, val_loader, device)
    print("\nFinal Evaluation")
    print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))

if __name__ == "__main__":
    main()