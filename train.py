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

# Set multiprocessing sharing strategy for safer large dataset processing
try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except RuntimeError:
    pass

def loadfiles(data_dir, target_splits):
    all_files = []
    subdirs = ['for-2sec/for-2sec', 'for-rerec/for-rerecorded']

    for subdir in subdirs:
        for split in target_splits:
            for cls in ['fake', 'real']:
                path = os.path.join(data_dir, subdir, split, cls)
                if os.path.exists(path):
                    files = []
                    for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
                        files.extend(glob.glob(os.path.join(path, ext)))
                    print(f"Found {len(files)} {cls} files in {path}")
                    all_files.extend([(f, 1 if cls == 'fake' else 0) for f in files])

    print(f"Total files loaded from {target_splits}: {len(all_files)}")
    fake_count = sum(1 for _, label in all_files if label == 1)
    real_count = sum(1 for _, label in all_files if label == 0)
    print(f"Fake files: {fake_count}, Real files: {real_count}")
    random.shuffle(all_files)
    return all_files

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
    print(f"Using device: {device}")

    data_dir = "/kaggle/input/the-fake-or-real-dataset"
    train_files = loadfiles(data_dir, ['training'])
    val_files = loadfiles(data_dir, ['testing', 'validation'])

    if not train_files or not val_files:
        print("Missing data.")
        return

    train_labels = [y for _, y in train_files]
    val_labels = [y for _, y in val_files]
    print(f"Training class distribution: {Counter(train_labels)}")
    print(f"Validation class distribution: {Counter(val_labels)}")

    total = len(train_labels)
    class_weights = torch.tensor([
        total / (2 * Counter(train_labels)[0]),
        total / (2 * Counter(train_labels)[1])
    ], device=device)

    print("\nCreating datasets...")
    train_ds = DatasetFolder(train_files, augment=True)
    val_ds = DatasetFolder(val_files, augment=False)

    batch_size = 16  # Increased batch size for better throughput
    num_workers = min(4, multiprocessing.cpu_count())  # Set safely for Kaggle

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    model = TBranchDetector().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=4e-4, weight_decay=1e-4)
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