import torch
import torch.nn as nn

class SimpleBranch1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 5, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64)
        )

    def forward(self, x):
        return self.conv(x).view(x.shape[0], -1)

class SimpleBranch2D(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )

    def forward(self, x):
        return self.conv(x).view(x.shape[0], -1)

class HybridCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_raw = SimpleBranch1D()
        self.branch_fft = SimpleBranch2D((128, 128))
        self.branch_wav = SimpleBranch2D((64, 128))

        self.fc = nn.Sequential(
            nn.Linear(16*64 + 16*8*8 + 16*8*8, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_raw, x_fft, x_wav):
        b1 = self.branch_raw(x_raw)
        b2 = self.branch_fft(x_fft)
        b3 = self.branch_wav(x_wav)
        x = torch.cat([b1, b2, b3], dim=1)
        return torch.sigmoid(self.fc(x))
