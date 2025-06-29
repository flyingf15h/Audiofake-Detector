import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
    def forward(self, x):
        return self.block(x)

class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    def forward(self, x):
        return self.block(x)

class FusionBlock(nn.Module):
    def __init__(self, in_dim=128, branches=3):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(branches))
        self.fc = nn.Sequential(
            nn.Linear(in_dim * branches, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  
        )
    
    def forward(self, b1, b2, b3):
        w = F.softmax(self.weights, dim=0)
        fused = torch.cat([w[0]*b1, w[1]*b2, w[2]*b3], dim=1)
        return self.fc(fused).squeeze(-1)  

class MultiCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Branch 1: Raw waveform (1D Conv)
        self.branch_raw = nn.Sequential(
            ConvBlock1D(1, 32),
            ConvBlock1D(32, 64),
            ConvBlock1D(64, 128),
            nn.AdaptiveAvgPool1d(1)
        )
        # Branch 2: FFT spectrogram (2D Conv)
        self.branch_fft = nn.Sequential(
            ConvBlock2D(1, 32),
            ConvBlock2D(32, 64),
            ConvBlock2D(64, 128),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # Branch 3: Wavelet coeffs (2D Conv)
        self.branch_wav = nn.Sequential(
            ConvBlock2D(1, 32),
            ConvBlock2D(32, 64),
            ConvBlock2D(64, 128),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # Fusion block
        self.fusion = FusionBlock(in_dim=128, branches=3)
    
    def forward(self, x_raw, x_fft, x_wav):
        b1 = self.branch_raw(x_raw).squeeze(-1)
        b2 = self.branch_fft(x_fft).squeeze(-1).squeeze(-1)
        b3 = self.branch_wav(x_wav).squeeze(-1).squeeze(-1)
        return self.fusion(b1, b2, b3)

