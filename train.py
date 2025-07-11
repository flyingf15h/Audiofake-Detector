import os, glob, random, warnings, numpy as np
from pathlib import Path
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import librosa, pywt
import multiprocessing
from model import TBranchDetector


def prepInputArray(audioArr, sr=16000, fixed_length=16000):
    audioArr = librosa.util.fix_length(audioArr, size=fixed_length).astype(np.float32)

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

def collate_fn(batch):
    raws, ffts, wavs, labels = zip(*batch)
    return torch.stack(raws), torch.stack(ffts), torch.stack(wavs), torch.tensor(labels, dtype=torch.long)

class FolderAudioDataset(Dataset):
    def __init__(self, folder_path, augment=False):
        real_files = list(Path(folder_path, "real").glob("*.wav"))
        fake_files = list(Path(folder_path, "fake").glob("*.wav"))
        self.files = [(f, 0) for f in real_files] + [(f, 1) for f in fake_files]
        self.augment_flag = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        try:
            audio, sr = librosa.load(path, sr=16000)
            if self.augment_flag:
                audio = self.augment(audio)
            x_raw, x_fft, x_wav = prepInputArray(audio)
            return torch.tensor(x_raw), torch.tensor(x_fft), torch.tensor(x_wav), torch.tensor(label)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return (
                torch.zeros((1,16000)), torch.zeros((1,128,128)),
                torch.zeros((1,64,128)), torch.tensor(0)
            )

    def augment(self, audio):
        try:
            if random.random() < 0.3:
                audio = np.roll(audio, random.randint(-1600, 1600))
            if random.random() < 0.3:
                audio *= random.uniform(0.8, 1.2)
            if random.random() < 0.2:
                audio += np.random.normal(0, 0.005, audio.shape)
            audio = np.clip(audio, -1.0, 1.0)
        except Exception:
            pass
        return audio


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
            labels.append(y)

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

    base = "/kaggle/input/the-fake-or-real-dataset/for-2sec/for-2seconds"
    train_ds = FolderAudioDataset(os.path.join(base, "training"), augment=True)
    val_ds = FolderAudioDataset(os.path.join(base, "validation"), augment=False)

    batch_size = 64
    num_workers = min(4, multiprocessing.cpu_count())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers)

    model = TBranchDetector().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    class_weights = torch.tensor([1.0, 1.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=(4e-4)*batch_size/16, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    best_loss = float("inf")
    patience = 10
    for epoch in range(1, 50):
        print(f"\nEpoch {epoch}/50")
        train_loss = train_1epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")

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
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load("best_model.pth"))
    acc, auc, y_true, y_pred = evaluate(model, val_loader, device)
    print("\nFinal Evaluation")
    print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))


if __name__ == "__main__":
    main()
