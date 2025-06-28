from datasets import load_dataset
from model import HybridCNN
from preprocess import prepare_inputs_from_array
import torch.nn as nn
import torch.optim as optim
import torch

# Load dataset
ds = load_dataset("Hemg/Deepfake-Audio-Dataset")["train"]

# Convert one sample to model input
def load_sample(audio_array, label):
    raw, fft, wav = prepare_inputs_from_array(audio_array)
    return (
        torch.tensor(raw, dtype=torch.float32).unsqueeze(0),  # (1, 1, 16000)
        torch.tensor(fft, dtype=torch.float32).unsqueeze(0),  # (1, 1, 128, 128)
        torch.tensor(wav, dtype=torch.float32).unsqueeze(0),  # (1, 1, 64, 128)
        torch.tensor([label], dtype=torch.float32)            # (1,)
    )

model = HybridCNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(10):
    total_loss = 0.0
    for sample in ds.select(range(10)):  # Just use 10 samples for now
        audio = sample["audio"]["array"]
        label = sample["label"]

        x_raw, x_fft, x_wav, y = load_sample(audio, label)
        output = model(x_raw, x_fft, x_wav).view(-1)  
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}: Avg Loss = {total_loss / 10:.4f}")
