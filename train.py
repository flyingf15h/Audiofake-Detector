import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
import numpy as np
import librosa
import pywt
from model import AudioMultiBranchCNN  
from sklearn.metrics import accuracy_score, roc_auc_score

# Dataset wrapper 
class AudioFakeDataset(Dataset):
    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        audio_array = sample['audio']['array'].astype(np.float32)
        x_raw, x_fft, x_wav = prepare_inputs_from_array(audio_array)
        label = float(sample['label'])
        return torch.tensor(x_raw), torch.tensor(x_fft), torch.tensor(x_wav), torch.tensor(label)

# batch and stack inputs
def collate_fn(batch):
    raws, ffts, wavs, labels = zip(*batch)
    x_raw = torch.stack(raws)
    x_fft = torch.stack(ffts)
    x_wav = torch.stack(wavs)
    y = torch.tensor(labels).float()
    return x_raw, x_fft, x_wav, y

# Audio preprocessing 
def prepare_inputs_from_array(audio_array, sr=16000, fixed_length=16000):
    audio_array = librosa.util.fix_length(audio_array, size=fixed_length)
    # Raw waveform shape (1, fixed_length)
    x_raw = np.expand_dims(audio_array, axis=0)

    # FFT magnitude spectrogram shape (1, 128, 128)
    stft = librosa.stft(audio_array, n_fft=256, hop_length=128)
    mag = np.abs(stft)
    mag = mag[:128, :128]  # crop or pad to fixed size
    x_fft = np.expand_dims(mag, axis=0)

    # Wavelet coefficients shape (1, 64, 128)
    coeffs = pywt.wavedec(audio_array, 'db4', level=4)
    cA4 = coeffs[0]
    cA4_resized = np.resize(cA4, (64, 128))
    x_wav = np.expand_dims(cA4_resized, axis=0)

    return x_raw, x_fft, x_wav

# Training for one epoch
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    for x_raw, x_fft, x_wav, y in dataloader:
        x_raw, x_fft, x_wav, y = x_raw.to(device), x_fft.to(device), x_wav.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x_raw, x_fft, x_wav).squeeze()
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * y.size(0)
    return running_loss / len(dataloader.dataset)

# Evaluation 
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_raw, x_fft, x_wav, y in dataloader:
            x_raw, x_fft, x_wav, y = x_raw.to(device), x_fft.to(device), x_wav.to(device), y.to(device)
            preds = model(x_raw, x_fft, x_wav).squeeze()
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    pred_labels = (all_preds >= 0.5).astype(int)
    acc = accuracy_score(all_labels, pred_labels)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = float('nan')  
    return acc, auc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading dataset...")
    dataset = load_dataset("Hemg/Deepfake-Audio-Dataset")['train']

    # Split train/test 80/20
    split_ds = dataset.train_test_split(test_size=0.2)
    train_ds = AudioFakeDataset(split_ds['train'])
    test_ds = AudioFakeDataset(split_ds['test'])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

    model = AudioMultiBranchCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Learning rate scheduler reduces LR if no improvement in val loss for 3 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Stop if no improvement for 7 epochs
    best_loss = float('inf')
    patience = 7  # 
    epochs_no_improve = 0
    epochs = 100

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate on test set each epoch to get val loss for scheduler and early stopping
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_raw, x_fft, x_wav, y in test_loader:
                x_raw, x_fft, x_wav, y = x_raw.to(device), x_fft.to(device), x_wav.to(device), y.to(device)
                outputs = model(x_raw, x_fft, x_wav).squeeze()
                loss = criterion(outputs, y)
                val_loss += loss.item() * y.size(0)
        val_loss /= len(test_loader.dataset)

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)  # adjust LR based on val loss

        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs with no improvement.")
                break

    acc, auc = evaluate(model, test_loader, device)
    print(f"\nFinal Evaluation:\nTest Accuracy: {acc:.4f}\nTest ROC AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
