import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
import numpy as np
import librosa
import pywt
from model import MultiCNN
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from collections import Counter
import random

class fakeDataset(Dataset):  
    def __init__(self, hf_dataset, doaugment=False):
        self.ds = hf_dataset
        self.doaugment = doaugment

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        audioArr = sample['audio']['array'].astype(np.float32)
        
        if self.doaugment:
            audioArr = self.augmentAudio(audioArr)      
            
        x_raw, x_fft, x_wav = prepInputArray(audioArr)
        label = float(sample['label'])
        return (
            torch.tensor(x_raw, dtype=torch.float32),
            torch.tensor(x_fft, dtype=torch.float32),
            torch.tensor(x_wav, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )

    
    def augmentAudio(self, audio):  
        # Random time shift, amplitude scaling, noise
        if random.random() < 0.3:
            shift = random.randint(-1600, 1600) 
            audio = np.roll(audio, shift)
        
        if random.random() < 0.3:
            scale = random.uniform(0.8, 1.2)
            audio = audio * scale
        
        if random.random() < 0.2:
            noise = np.random.normal(0, 0.005, audio.shape)
            audio = audio + noise
            
        return audio

def collate_fn(batch):
    raws, ffts, wavs, labels = zip(*batch)
    x_raw = torch.stack(raws)
    x_fft = torch.stack(ffts)
    x_wav = torch.stack(wavs)
    y = torch.tensor(labels, dtype=torch.float32)
    return x_raw, x_fft, x_wav, y

def prepInputArray(audioArr, sr=16000, fixed_length=16000):
    audioArr = librosa.util.fix_length(audioArr, size=fixed_length).astype(np.float32)
    
    # Raw waveform w normalization
    x_raw = (audioArr - np.mean(audioArr)) / (np.std(audioArr) + 1e-8)
    x_raw = np.expand_dims(x_raw, axis=0).astype(np.float32)

    # Spectrogram wn  
    stft = librosa.stft(audioArr, n_fft=256, hop_length=128)
    mag = np.abs(stft)
    mag = mag[:128, :128]
    mag = (mag - np.mean(mag)) / (np.std(mag) + 1e-8)
    x_fft = np.expand_dims(mag, axis=0).astype(np.float32)

    # Wavelet coeffs wn
    coeffs = pywt.wavedec(audioArr, 'db4', level=4)
    cA4 = coeffs[0]
    cA4_resized = np.resize(cA4, (64, 128))
    cA4_resized = (cA4_resized - np.mean(cA4_resized)) / (np.std(cA4_resized) + 1e-8)
    x_wav = np.expand_dims(cA4_resized, axis=0).astype(np.float32)

    return x_raw, x_fft, x_wav

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    for x_raw, x_fft, x_wav, y in dataloader:
        x_raw, x_fft, x_wav, y = x_raw.to(device), x_fft.to(device), x_wav.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x_raw, x_fft, x_wav)
        loss = criterion(output, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * y.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_raw, x_fft, x_wav, y in dataloader:
            x_raw, x_fft, x_wav, y = x_raw.to(device), x_fft.to(device), x_wav.to(device), y.to(device)
            logits = model(x_raw, x_fft, x_wav)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu())
            all_labels.append(y.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    pred_labels = (all_preds >= 0.5).astype(int)
    acc = accuracy_score(all_labels, pred_labels)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError: 
        auc = float('nan')
    return acc, auc, all_labels, pred_labels

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading dataset...")
    try:
        dataset = load_dataset("Hemg/Deepfake-Audio-Dataset")['train']
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Check class distribution
    try:
        label_counts = Counter([sample['label'] for sample in dataset])
        print(f"Dataset size: {len(dataset)}")
        print(f"Class distribution: {label_counts}")
        
        # Calculate class weights for balanced loss
        total = sum(label_counts.values())
        pos_weight = label_counts[0] / label_counts[1] if label_counts[1] > 0 else 1.0
        print(f"Positive weight (for class 1): {pos_weight:.2f}")
    except Exception as e:
        print(f"Error analyzing dataset: {e}")
        return

    try:
 
        split_ds = dataset.train_test_split(test_size=0.2, stratify_by_column='label')
        train_ds = fakeDataset(split_ds['train'], augment=True)
        test_ds = fakeDataset(split_ds['test'], augment=False)

        batch_size = 2
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        model = MultiCNN().to(device)
        print(f"Model loaded successfully")
        
        # Weighted loss to handle class imbalance
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
        
        optimizer = optim.Adam(model.parameters(), lr=0.0004, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3)

        best_loss = float('inf')
        patience = 10  
        epochs_no_improve = 6
        epochs = 109   

        print(f"\nStarting training for {epochs} epochs...")
        for epoch in range(epochs):
            print(f"Starting epoch {epoch + 1}/{epochs}...")
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_raw, x_fft, x_wav, y in test_loader:
                    x_raw, x_fft, x_wav, y = x_raw.to(device), x_fft.to(device), x_wav.to(device), y.to(device)
                    outputs = model(x_raw, x_fft, x_wav)
                    loss = criterion(outputs, y)
                    val_loss += loss.item() * y.size(0)
            val_loss /= len(test_loader.dataset)

            # Get validation accuracy for monitoring
            val_acc, val_auc, _, _ = evaluate(model, test_loader, device)

            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")

            scheduler.step(val_loss)

            # Save best model 
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), "best_model.pth")
                print(f"  ✓ New best model saved!")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs with no improvement.")
                    break

        print("\n" + "="*50)
        print("FINAL EVALUATION")
        print("="*50)
        
        # Load best model
        model.load_state_dict(torch.load("best_model.pth"))
        acc, auc, true_labels, pred_labels = evaluate(model, test_loader, device)
        
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test ROC AUC: {auc:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(true_labels, pred_labels, target_names=['Real', 'Fake']))
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()