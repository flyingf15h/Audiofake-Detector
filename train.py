import os
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
import pywt
from model import MultiCNN
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from collections import Counter
import random
import glob
import warnings

def loadfiles(data_dir, target_splits):
    all_files = []
    subdirs = ['for-2sec/for-2seconds', 'for-norm/for-norm', 'for-original/for-original', 'for-rerec/for-rerecorded']
    
    for subdir in subdirs:
        for split in target_splits:
            fake_path = os.path.join(data_dir, subdir, split, 'fake')
            if os.path.exists(fake_path):
                fake_files = []
                for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
                    fake_files.extend(glob.glob(os.path.join(fake_path, ext)))
                
                print(f"Found {len(fake_files)} fake files in {fake_path}")
                all_files.extend([(f, 1) for f in fake_files])  # 1 for fake
            
            real_path = os.path.join(data_dir, subdir, split, 'real')
            if os.path.exists(real_path):
                real_files = []
                for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
                    real_files.extend(glob.glob(os.path.join(real_path, ext)))
                
                print(f"Found {len(real_files)} real files in {real_path}")
                all_files.extend([(f, 0) for f in real_files])  # 0 for real
    
    print(f"Total files loaded from {target_splits}: {len(all_files)}")
    if len(all_files) > 0:
        fake_count = sum(1 for _, label in all_files if label == 1)
        real_count = sum(1 for _, label in all_files if label == 0)
        print(f"Fake files: {fake_count}, Real files: {real_count}")
    
    random.shuffle(all_files)
    return all_files

class DatasetFolder(Dataset):
    def __init__(self, file_label_pairs, augment=False):
        self.data = file_label_pairs
        self.augment_flag = augment
        self.failed_files = set()  # Track failed files to avoid repeated warnings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        
        # Load audio with robust error handling
        try:
            audioArr, sr = librosa.load(path, sr=16000)
            # Check if audio was loaded successfully
            if audioArr is None or len(audioArr) == 0:
                raise ValueError("Empty audio array")
        except Exception as e:
            # Only print warning once per file
            if path not in self.failed_files:
                print(f"Failed to load {path}: {e}")
                self.failed_files.add(path)
            # Return a silent audio array as fallback
            audioArr = np.zeros(16000, dtype=np.float32)

        if self.augment_flag and audioArr is not None:
            audioArr = self.augmentAudio(audioArr)

        try:
            x_raw, x_fft, x_wav = prepInputArray(audioArr)
        except Exception as e:
            if path not in self.failed_files:
                print(f"Error preprocessing {path}: {e}")
                self.failed_files.add(path)
            # Return zero arrays as fallback
            x_raw = np.zeros((1, 16000), dtype=np.float32)
            x_fft = np.zeros((1, 128, 128), dtype=np.float32)
            x_wav = np.zeros((1, 64, 128), dtype=np.float32)
            
        return (
            torch.tensor(x_raw, dtype=torch.float32),
            torch.tensor(x_fft, dtype=torch.float32),
            torch.tensor(x_wav, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )

    def augmentAudio(self, audio):  
        try:
            if random.random() < 0.3:
                shift = random.randint(-1600, 1600) 
                audio = np.roll(audio, shift)
            
            if random.random() < 0.3:
                scale = random.uniform(0.8, 1.2)
                audio = audio * scale
            
            if random.random() < 0.2:
                noise = np.random.normal(0, 0.005, audio.shape)
                audio = audio + noise
        except Exception:
            # If augmentation fails, return original audio
            pass
            
        return audio

def collate_fn(batch):
    raws, ffts, wavs, labels = zip(*batch)
    x_raw = torch.stack(raws)
    x_fft = torch.stack(ffts)
    x_wav = torch.stack(wavs)
    y = torch.tensor(labels, dtype=torch.float32)
    return x_raw, x_fft, x_wav, y

def prepInputArray(audioArr, sr=16000, fixed_length=16000):
    # Ensure we have a valid audio array
    if audioArr is None or len(audioArr) == 0:
        audioArr = np.zeros(fixed_length, dtype=np.float32)
    
    audioArr = librosa.util.fix_length(audioArr, size=fixed_length).astype(np.float32)
    
    # Handle edge case where audio might be all zeros
    if np.std(audioArr) == 0:
        x_raw = audioArr.astype(np.float32)
    else:
        x_raw = (audioArr - np.mean(audioArr)) / (np.std(audioArr) + 1e-8)
    x_raw = np.expand_dims(x_raw, axis=0).astype(np.float32)

    try:
        stft = librosa.stft(audioArr, n_fft=256, hop_length=128)
        mag = np.abs(stft)
        mag = mag[:128, :128]
        if np.std(mag) == 0:
            x_fft = mag.astype(np.float32)
        else:
            x_fft = (mag - np.mean(mag)) / (np.std(mag) + 1e-8)
        x_fft = np.expand_dims(x_fft, axis=0).astype(np.float32)
    except Exception:
        x_fft = np.zeros((1, 128, 128), dtype=np.float32)

    try:
        coeffs = pywt.wavedec(audioArr, 'db4', level=4)
        cA4 = coeffs[0]
        cA4_resized = np.resize(cA4, (64, 128))
        if np.std(cA4_resized) == 0:
            x_wav = cA4_resized.astype(np.float32)
        else:
            x_wav = (cA4_resized - np.mean(cA4_resized)) / (np.std(cA4_resized) + 1e-8)
        x_wav = np.expand_dims(x_wav, axis=0).astype(np.float32)
    except Exception:
        x_wav = np.zeros((1, 64, 128), dtype=np.float32)

    return x_raw, x_fft, x_wav

def train_1epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    valid_batches = 0
    
    for batch_idx, (x_raw, x_fft, x_wav, y) in enumerate(dataloader):
        try:
            x_raw, x_fft, x_wav, y = x_raw.to(device), x_fft.to(device), x_wav.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x_raw, x_fft, x_wav)
            loss = criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            valid_batches += y.size(0)
        except Exception as e:
            print(f"Error in training batch {batch_idx}: {e}")
            continue
        
        # Print progress every 1000 batches
        if batch_idx % 1000 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}")
    
    return running_loss / max(valid_batches, 1)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, (x_raw, x_fft, x_wav, y) in enumerate(dataloader):
            try:
                x_raw, x_fft, x_wav, y = x_raw.to(device), x_fft.to(device), x_wav.to(device), y.to(device)
                logits = model(x_raw, x_fft, x_wav)
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu())
                all_labels.append(y.cpu())
            except Exception as e:
                print(f"Error in evaluation batch {batch_idx}: {e}")
                continue
    
    if len(all_preds) == 0:
        return 0.0, float('nan'), np.array([]), np.array([])
    
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
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading dataset...")
    try:
        data_dir = "/kaggle/input/the-fake-or-real-dataset"
        
        print("\nLoading Training Data")
        train_files = loadfiles(data_dir, ['training'])
        
        print("\nLoading Testing Data") 
        val_test_files = loadfiles(data_dir, ['testing', 'validation'])
        
        if len(train_files) == 0:
            print("No training files found.")
            return
            
        if len(val_test_files) == 0:
            print("No validation or test files found.")
            return
        
        # Check class distribution for training data
        train_labels = [label for _, label in train_files]
        train_label_counts = Counter(train_labels)
        print(f"\nTraining dataset size: {len(train_files)}")
        print(f"Training class distribution: {train_label_counts}")
        
        # Check class distribution for validation/test data
        val_test_labels = [label for _, label in val_test_files]
        val_test_label_counts = Counter(val_test_labels)
        print(f"Validation/Test dataset size: {len(val_test_files)}")
        print(f"Validation/Test class distribution: {val_test_label_counts}")
        
        if len(train_label_counts) < 2 or len(val_test_label_counts) < 2:
            print("Error: fake or real data set not found")
            return
            
        pos_weight = train_label_counts[0] / train_label_counts[1] if train_label_counts[1] > 0 else 1.0
        print(f"Positive weight (for class 1): {pos_weight:.2f}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    try:
        print("\nCreating datasets...")
        train_ds = DatasetFolder(train_files, augment=True)
        val_test_ds = DatasetFolder(val_test_files, augment=False)
        batch_size = 16
        print(f"Using batch size: {batch_size}")
        
        # Set num_workers to 0 to avoid multiprocessing issues
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                                collate_fn=collate_fn, num_workers=0, pin_memory=True)
        val_test_loader = DataLoader(val_test_ds, batch_size=batch_size, shuffle=False, 
                                   collate_fn=collate_fn, num_workers=0, pin_memory=True)

        print("Loading model...")
        model = MultiCNN().to(device)
        print(f"Model loaded successfully")

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
        optimizer = optim.Adam(model.parameters(), lr=0.0004, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3)

        best_loss = float('inf')
        patience = 10
        epochs_noImprove = 0
        epochs = 109

        print(f"\nStarting training for {epochs} epochs...")
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            train_loss = train_1epoch(model, train_loader, criterion, optimizer, device)

            # Validation on the separate validation/test set
            print("Running validation...")
            model.eval()
            val_loss = 0
            valid_samples = 0
            with torch.no_grad():
                for x_raw, x_fft, x_wav, y in val_test_loader:
                    try:
                        x_raw, x_fft, x_wav, y = x_raw.to(device), x_fft.to(device), x_wav.to(device), y.to(device)
                        outputs = model(x_raw, x_fft, x_wav)
                        loss = criterion(outputs, y)
                        val_loss += loss.item() * y.size(0)
                        valid_samples += y.size(0)
                    except Exception as e:
                        continue
            
            val_loss = val_loss / max(valid_samples, 1)
            val_acc, val_auc, _, _ = evaluate(model, val_test_loader, device)

            print(f"Epoch {epoch + 1}/{epochs} Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")

            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                epochs_noImprove = 0
                torch.save(model.state_dict(), "best_model.pth")
                print(f"  *** New best model saved ***")
            else:
                epochs_noImprove += 1
                if epochs_noImprove >= patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs with no improvement.")
                    break

        print("\n" + "="*50)
        print("Final Evaluation")
        print("="*50)
        
        model.load_state_dict(torch.load("best_model.pth"))
        acc, auc, true_labels, pred_labels = evaluate(model, val_test_loader, device)

        print(f"Final Test Accuracy: {acc:.4f}")
        print(f"Final Test ROC AUC: {auc:.4f}")
        
        if len(true_labels) > 0 and len(pred_labels) > 0:
            print("\nDetailed Classification Report:")
            print(classification_report(true_labels, pred_labels, target_names=['Real', 'Fake']))
        else:
            print("No valid predictions to evaluate.")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()