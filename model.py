import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import pywt

class PatchEmbed(nn.Module):
    # 2D image to patch embedding for spectrograms
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return x

class TransformerBlock(nn.Module):
    # Multi head attention and MLP
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        self.proj_drop = nn.Dropout(drop) 
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class AST(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768, depth=12, 
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]  # Return CLS token


class WaveletTransform(nn.Module):
    def __init__(self, wavelet='db4', level=5):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self._init_filters()
        
    def _init_filters(self):
        dec_lo, dec_hi, _, _ = pywt.Wavelet(self.wavelet).filter_bank
        self.register_buffer('dec_lo', torch.tensor(dec_lo).float())
        self.register_buffer('dec_hi', torch.tensor(dec_hi).float())
        
    def forward(self, x):
        # x: (B, 1, T)
        B, _, T = x.shape
        pad = len(self.dec_lo) // 2
        x = F.pad(x, (pad, pad), mode='reflect')
        
        coeffs = []
        for _ in range(self.level):
            # 1D convolution with wavelet filters
            low = F.conv1d(x, self.dec_lo.view(1, 1, -1), stride=2)
            high = F.conv1d(x, self.dec_hi.view(1, 1, -1), stride=2)
            coeffs.append(high)
            x = low
            
        # Stack and reshape coefficients
        return torch.stack(coeffs, dim=1)  # (B, level, T//2^level)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = F.relu(out)
        
        return out


class RawCNN(nn.Module):
    def __init__(self, input_length=16000, num_classes=2):
        super().__init__()
        
        # First conv layer to capture temporal patterns
        self.conv1 = nn.Conv1d(1, 64, kernel_size=80, stride=4, padding=38)
        self.bn1 = nn.BatchNorm1d(64)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(64, 64, kernel_size=3, stride=1),
            ResidualBlock(64, 128, kernel_size=3, stride=2),
            ResidualBlock(128, 128, kernel_size=3, stride=1),
            ResidualBlock(128, 256, kernel_size=3, stride=2),
            ResidualBlock(256, 256, kernel_size=3, stride=1),
            ResidualBlock(256, 512, kernel_size=3, stride=2),
        ])
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = 512
    
    def forward(self, x):
        # Args x: Raw waveform (B, 1, T)
        # Returns features (B, feature_dim)
        
        # First conv layer
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Apply residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)  # (B, feature_dim)
        return x


class SpectrogramExtractor(nn.Module):
    # Extract mel spectrograms from audio
    def __init__(self, sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512):
        super().__init__()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0
        )
        
    def forward(self, x):
        # Args x = Raw audio (B, 1, T)
        # Returns Mel spectrogram (B, 1, n_mels, time_frames)
        
        x = x.squeeze(1)  # (B, T)
        mel_spec = self.mel_spectrogram(x)
        mel_spec = torch.log(mel_spec + 1e-8) 
        return mel_spec.unsqueeze(1)  


class AttentionFusion(nn.Module):
    def __init__(self, feature_dims, hidden_dim=256):
        super().__init__()
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        
        # Project each feature to same dimension
        self.projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in feature_dims
        ])
        
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, features):
        # Args features = List of feature tensors [(B, D1), (B, D2), (B, D3)]
        # Returns fused features (B, hidden_dim)
    
        # Project features to same dimension 
        projected_features = []
        for i, feat in enumerate(features):
            projected = self.projections[i](feat)
            projected_features.append(projected)
        
        stacked_features = torch.stack(projected_features, dim=1)  # (B, 3, hidden_dim)
        attended_features, _ = self.attention(stacked_features, stacked_features, stacked_features)
        
        fused_features = attended_features.mean(dim=1)  # (B, hidden_dim)
        fused_features = self.norm(fused_features)
        
        return fused_features


class TBranchDetector(nn.Module):
    def __init__(self, 
                 sample_rate=16000,
                 input_length=16000,
                 num_classes=2,
                 ast_img_size=224,
                 ast_patch_size=16,
                 ast_embed_dim=768,
                 ast_depth=12,
                 ast_num_heads=12,
                 fusion_hidden_dim=256,
                 drop_rate=0.1,     
                 attn_drop_rate=0.1):
        super().__init__()
        
        self.spectrogram_extractor = SpectrogramExtractor(sample_rate=sample_rate)
        self.wavelet_transform = WaveletTransform()
        
        self.ast_spectrogram = AST(
            img_size=ast_img_size,
            patch_size=ast_patch_size,
            embed_dim=ast_embed_dim,
            depth=ast_depth,
            num_heads=ast_num_heads,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate
        )
        
        self.ast_wavelet = AST(
            img_size=ast_img_size,
            patch_size=ast_patch_size,
            embed_dim=ast_embed_dim,
            depth=ast_depth,
            num_heads=ast_num_heads,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate  
        )
        
        self.cnn_raw = RawCNN(input_length=input_length)
        
        feature_dims = [ast_embed_dim, ast_embed_dim, self.cnn_raw.feature_dim]
        self.fusion = AttentionFusion(feature_dims, fusion_hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fusion_hidden_dim // 2, num_classes)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def resize_spectrogram(self, spec, target_size=224):
    # Properly handles tensor dimensions for model input
        while spec.dim() > 4:
            squeezed = False
            for dim in reversed(range(spec.dim())):
                if spec.shape[dim] == 1:
                    spec = spec.squeeze(dim)
                    squeezed = True
                    break
            
            if not squeezed:
                raise ValueError(f"Cannot reduce shape {spec.shape} to 4D - no singleton dimensions found")
    
        if spec.dim() == 3:  # [B,H,W] -> [B,1,H,W]
            spec = spec.unsqueeze(1)
        elif spec.dim() == 2:  # [H,W] -> [1,1,H,W] 
            spec = spec.unsqueeze(0).unsqueeze(0)
        elif spec.dim() != 4:
            raise ValueError(f"Cannot handle tensor with {spec.dim()} dimensions, shape: {spec.shape}")
            
        return spec
    
    def forward(self, x_raw, x_fft, x_wav):
        assert x_raw.dim() == 3, f"Expected raw audio shape [B,1,L], got {x_raw.shape}"
        assert x_fft.dim() == 4, f"Expected FFT shape [B,1,H,W], got {x_fft.shape}"
        assert x_wav.dim() == 4, f"Expected wavelet shape [B,1,H,W], got {x_wav.shape}"
        
        if isinstance(self, nn.DataParallel):
            x_raw = x_raw.squeeze(0) if x_raw.dim() == 4 else x_raw
            x_fft = x_fft.squeeze(0) if x_fft.dim() == 5 else x_fft
            x_wav = x_wav.squeeze(0) if x_wav.dim() == 5 else x_wav

        fft_resized = self.resize_spectrogram(x_fft, target_size=224)  
        ast_specfeat = self.ast_spectrogram(fft_resized)  # (B, embed_dim)

        wav_resized = self.resize_spectrogram(x_wav, target_size=224)
        ast_wavefeat = self.ast_wavelet(wav_resized)  # (B, embed_dim)

        cnn_features = self.cnn_raw(x_raw)  # (B, 512)

        all_features = [ast_specfeat, ast_wavefeat, cnn_features]
        fused_features = self.fusion(all_features)  # (B, fusion_hidden_dim)

        logits = self.classifier(fused_features)  # (B, num_classes)
        return logits

    
    def getbranch_features(self, x_raw, x_fft, x_wav):
        if isinstance(self, nn.DataParallel):
            device = next(self.module.parameters()).device
        else:
            device = next(self.parameters()).device
            
        x_raw, x_fft, x_wav = x_raw.to(device), x_fft.to(device), x_wav.to(device)

        if isinstance(self, nn.DataParallel):
            return self.module.gbf(x_raw, x_fft, x_wav)
        else:
            return self.gbf(x_raw, x_fft, x_wav)
        
    def gbf(self, x_raw, x_fft, x_wav):
        fft_resized = self.resize_spectrogram(x_fft, target_size=224)
        ast_specfeat = self.ast_spectrogram(fft_resized)

        wav_resized = self.resize_spectrogram(x_wav, target_size=224)
        ast_wavefeat = self.ast_wavelet(wav_resized)

        cnn_features = self.cnn_raw(x_raw)
        
        return ast_specfeat, ast_wavefeat, cnn_features


def create_model(sample_rate=16000, input_length=16000, num_classes=2):
    model = TBranchDetector(
        sample_rate=sample_rate,
        input_length=input_length,
        num_classes=num_classes,
        ast_img_size=224,
        ast_patch_size=16,
        ast_embed_dim=768,
        ast_depth=12,
        ast_num_heads=12,
        fusion_hidden_dim=256
    )
    return model