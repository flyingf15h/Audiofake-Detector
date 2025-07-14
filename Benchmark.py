import os
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import librosa
import pywt
import torch.nn as nn
from model import TBranchDetector
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from datasets import load_dataset
import glob
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = nn.DataParallel(TBranchDetector()).to(device)

model.load_state_dict(
    torch.load(
        "/kaggle/input/audifake-detector/pytorch/default/1/best_model.pth",
        map_location=device
    )
)
model.eval()

def preprocess(audio, sr=16000):
    audio = librosa.util.fix_length(audio, size=16000)
    x_raw = (audio - np.mean(audio)) / (np.std(audio) + 1e-8)
    
    stft = librosa.stft(audio, n_fft=256, hop_length=128)
    mag = np.abs(stft)[:128, :128]
    x_fft = (mag - np.mean(mag)) / (np.std(mag) + 1e-8)
    
    coeffs = pywt.wavedec(audio, 'db4', level=4)
    cA4_resized = np.resize(coeffs[0], (64, 128))
    x_wav = (cA4_resized - np.mean(cA4_resized)) / (np.std(cA4_resized) + 1e-8)
    
    return (
        torch.tensor(x_raw).unsqueeze(0).float(),
        torch.tensor(x_fft).unsqueeze(0).float(),
        torch.tensor(x_wav).unsqueeze(0).float()
    )

class FilepathDataset(Dataset):
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
            return torch.zeros(1, 16000), torch.zeros(1, 128, 128), torch.zeros(1, 64, 128), -1

class WaveFakeDataset(Dataset):
    def __init__(self, split="test", max_samples=None):
        self.dataset = load_dataset("Keerthana982/wavefake-audio", split=split, streaming=True)
        self.max_samples = max_samples
        
    def __iter__(self):
        count = 0
        for sample in self.dataset:
            if self.max_samples and count >= self.max_samples:
                break
            try:
                audio = sample["audio"]["array"]
                audio = librosa.util.fix_length(audio, size=16000)
                yield *preprocess(audio), sample["label"]
                count += 1
            except Exception as e:
                print(f"Skipping WaveFake sample: {str(e)}")

def evaluate(dataloader, name="Dataset"):
    criterion = torch.nn.BCEWithLogitsLoss()
    y_true, y_pred, y_prob = [], [], []
    total_loss = 0.0
    
    for x_raw, x_fft, x_wav, labels in tqdm(dataloader, desc=f"Evaluating {name}"):
        x_raw = x_raw.to(device)
        x_fft = x_fft.to(device)
        x_wav = x_wav.to(device)
        labels = labels.float().to(device)
        
        mask = labels != -1
        if mask.sum() == 0:  
            continue
            
        with torch.no_grad():
            outputs = model(x_raw[mask], x_fft[mask], x_wav[mask]).squeeze(-1) 
            loss = criterion(outputs, labels[mask].unsqueeze(1))
            probs = torch.sigmoid(outputs).squeeze()
            
        total_loss += loss.item() * mask.sum()
        y_true.extend(labels[mask].cpu().tolist())
        y_prob.extend(probs.cpu().tolist())
        y_pred.extend((probs >= 0.5).long().cpu().tolist())
    
    if len(y_true) == 0:
        print(f"No valid samples found in {name}!")
        return
    
    print(f"\nResults for {name}:")
    print(f"Loss: {total_loss / len(y_true):.4f}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1: {f1_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_true, y_prob):.4f}")

if __name__ == "__main__":
    # FoR and In-the-Wild
    for name, root in [
        ("FoR-Test", "/kaggle/input/the-fake-or-real-dataset/for-original/for-original/testing"),
        ("In-the-Wild", "/kaggle/input/in-the-wild-audio-deepfake/release_in_the_wild")
    ]:
        data = []
        for label_name, label in [('real', 0), ('fake', 1)]:
            folder = os.path.join(root, label_name)
            for ext in ('*.wav', '*.mp3', '*.flac'):
                data.extend([(f, label) for f in glob.glob(os.path.join(folder, ext))])
        
        loader = DataLoader(
            FilepathDataset(data),
            batch_size=128,  
            num_workers=2,
            collate_fn=lambda x: (
                torch.stack([item[0] for item in x]),
                torch.stack([item[1] for item in x]),
                torch.stack([item[2] for item in x]),
                torch.tensor([item[3] for item in x])
            )
        )
        evaluate(loader, name)
    
    # WaveFake (streaming)
    wavefake_loader = DataLoader(
        WaveFakeDataset(max_samples=10000),
        batch_size=128, 
        collate_fn=lambda x: (
            torch.stack([item[0] for item in x]),
            torch.stack([item[1] for item in x]),
            torch.stack([item[2] for item in x]),
            torch.tensor([item[3] for item in x])
        )
    )
    evaluate(wavefake_loader, "WaveFake")