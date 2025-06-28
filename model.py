import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class AudioMultiBranchCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Branch 1, raw waveform (1D conv)
        self.branch_raw = nn.Sequential(
            ConvBlock1D(1, 32, kernel_size=7, stride=2, padding=3),
            ConvBlock1D(32, 64, kernel_size=5, stride=2, padding=2),
            ConvBlock1D(64, 128, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool1d(1)  
            # Output shape (batch, 128, 1)
        )

        # Branch 2, FFT spectrogram (2D conv)
        self.branch_fft = nn.Sequential(
            ConvBlock2D(1, 32),
            ConvBlock2D(32, 64),
            ConvBlock2D(64, 128),
            nn.AdaptiveAvgPool2d((1, 1))  
            # Output shape (batch, 128, 1, 1)
        )

        # Branch 3, Wavelet coeffs (2D conv)
        self.branch_wav = nn.Sequential(
            ConvBlock2D(1, 32),
            ConvBlock2D(32, 64),
            ConvBlock2D(64, 128),
            nn.AdaptiveAvgPool2d((1, 1))  
            # Output shape (batch, 128, 1, 1)
        )

        # Fusion fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(128 + 128 + 128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x_raw, x_fft, x_wav):
        # Branch outputs (batch, 128)
        b1 = self.branch_raw(x_raw).squeeze(-1)          
        b2 = self.branch_fft(x_fft).squeeze(-1).squeeze(-1)  
        b3 = self.branch_wav(x_wav).squeeze(-1).squeeze(-1)  

        # Concatenate (batch, 384)
        x = torch.cat([b1, b2, b3], dim=1)  

        # FC layers
        x = self.fc(x)
        return torch.sigmoid(x).squeeze(-1)  