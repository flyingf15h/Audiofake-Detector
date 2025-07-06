import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import soundfile as sf
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class AudioDataset(Dataset):    
    def __init__(self, audio_paths, labels, sample_rate=16000, duration=1.0, 
                 augment=False, chunk_size=None):
        self.audio_paths = audio_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        self.augment = augment
        self.chunk_size = chunk_size or self.target_length
        
        self.chunks = self._build_chunks()
    
    def _build_chunks(self):
        chunks = []
        for i, (audio_path, label) in enumerate(zip(self.audio_paths, self.labels)):
            try:
                audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                
                # Create chunks
                num_chunks = len(audio) // self.chunk_size
                for j in range(num_chunks):
                    start = j * self.chunk_size
                    end = start + self.chunk_size
                    chunk = audio[start:end]
                    
                    if len(chunk) == self.chunk_size:
                        chunks.append((chunk, label, f"{audio_path}_{j}"))
                        
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
                continue
        
        return chunks
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk, label, chunk_id = self.chunks[idx]
        
        audio_tensor = torch.FloatTensor(chunk).unsqueeze(0)  # (1, T)
        
        if self.augment:
            audio_tensor = self._augment_audio(audio_tensor)
        
        # Ensure correct length
        if audio_tensor.shape[1] > self.target_length:
            audio_tensor = audio_tensor[:, :self.target_length]
        elif audio_tensor.shape[1] < self.target_length:
            pad_length = self.target_length - audio_tensor.shape[1]
            audio_tensor = F.pad(audio_tensor, (0, pad_length))
        
        return audio_tensor, torch.LongTensor([label])[0], chunk_id
    
    def _augment_audio(self, audio):
        if torch.rand(1) < 0.3:  # 30% chance noise
            noise = torch.randn_like(audio) * 0.005
            audio = audio + noise
        
        if torch.rand(1) < 0.2:  # 20% chance time shift
            shift = torch.randint(-1600, 1600, (1,)).item()  
            audio = torch.roll(audio, shift, dims=-1)
        
        return audio


def prepare_dataset(data_dir, sample_rate=16000, duration=1.0, test_size=0.2, val_size=0.1):
    # Assumes data_dir folder with real_dir and fake_dir folders with the files
    real_dir = Path(data_dir) / "real"
    fake_dir = Path(data_dir) / "fake"
    
    audio_paths = []
    labels = []
    
    if real_dir.exists():
        for audio_file in real_dir.glob("*.wav"):
            audio_paths.append(str(audio_file))
            labels.append(0) 
    
    if fake_dir.exists():
        for audio_file in fake_dir.glob("*.wav"):
            audio_paths.append(str(audio_file))
            labels.append(1)  
    
    print(f"Found {len([l for l in labels if l == 0])} real audio files")
    print(f"Found {len([l for l in labels if l == 1])} fake audio files")
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        audio_paths, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
    )
    
    # Create datasets
    train_dataset = AudioDataset(X_train, y_train, sample_rate, duration, augment=True)
    val_dataset = AudioDataset(X_val, y_val, sample_rate, duration, augment=False)
    test_dataset = AudioDataset(X_test, y_test, sample_rate, duration, augment=False)
    
    return train_dataset, val_dataset, test_dataset