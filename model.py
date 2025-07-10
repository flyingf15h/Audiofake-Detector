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
        x = x + self.attn(self.norm1(x))
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
    def __init__(self, wavelet='db4', max_level=5):
        super().__init__()
        self.wavelet = wavelet
        self.max_level = max_level
        
    def forward(self, x):
        # Applies wavelet packet transform to audio
        # Args x = Input audio tensor (B, 1, T)
        # Returns wavelet coefficients arranged as 2D tensor (B, 1, H, W)
        batch_size = x.shape[0]
        wavelets = []
        
        for i in range(batch_size):
            audio = x[i, 0].detach().cpu().numpy()
            
            # Wavelet packet decomposition before getting the coefficients
            wp = pywt.WaveletPacket(audio, self.wavelet, maxlevel=self.max_level)
            coeffs = []
            for node in wp.get_level(self.max_level):
                coeffs.append(node.data)
            
            # Arrange to 2D grid and make it square
            coeffs = np.array(coeffs)
            target_size = int(np.sqrt(coeffs.shape[0]) * coeffs.shape[1])
            if coeffs.size < target_size * target_size:
                coeffs = np.pad(coeffs.flatten(), (0, target_size * target_size - coeffs.size))
            else:
                coeffs = coeffs.flatten()[:target_size * target_size]
            
            coeffs = coeffs.reshape(target_size, target_size)
            wavelets.append(coeffs)
        
        wavelets = torch.tensor(np.array(wavelets), dtype=torch.float32).unsqueeze(1)
        return wavelets.to(x.device)


class RawCNN(nn.Module):
    def __init__(self, input_length=16000, num_classes=2):
        super().__init__()
        
        # kernel to capture temporal patterns for first conv layer
        self.conv1 = nn.Conv1d(1, 64, kernel_size=80, stride=4, padding=38)
        self.bn1 = nn.BatchNorm1d(64)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            self._make_res_block(64, 64, kernel_size=3, stride=1),
            self._make_res_block(64, 128, kernel_size=3, stride=2),
            self._make_res_block(128, 128, kernel_size=3, stride=1),
            self._make_res_block(128, 256, kernel_size=3, stride=2),
            self._make_res_block(256, 256, kernel_size=3, stride=1),
            self._make_res_block(256, 512, kernel_size=3, stride=2),
        ])
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = 512
        
    def _make_res_block(self, in_channels, out_channels, kernel_size, stride):
        # Residual block
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Args x: Raw waveform (B, 1, T)
        # Returns features (B, feature_dim) and x = F.relu(self.bn1(self.conv1(x)))
        
        # Add residual connection if dimensions match
        for res_block in self.res_blocks:
            residual = x
            x = res_block(x)
        
            if residual.shape[1] == x.shape[1] and residual.shape[2] == x.shape[2]:
                x = x + residual
        
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
                 fusion_hidden_dim=256):
        super().__init__()
        
        self.spectrogram_extractor = SpectrogramExtractor(sample_rate=sample_rate)
        self.wavelet_transform = WaveletTransform()
        
        self.ast_spectrogram = AST(
            img_size=ast_img_size,
            patch_size=ast_patch_size,
            embed_dim=ast_embed_dim,
            depth=ast_depth,
            num_heads=ast_num_heads
        )
        
        self.ast_wavelet = AST(
            img_size=ast_img_size,
            patch_size=ast_patch_size,
            embed_dim=ast_embed_dim,
            depth=ast_depth,
            num_heads=ast_num_heads
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
        return F.interpolate(spec, size=(target_size, target_size), mode='bilinear', align_corners=False)
    
    def forward(self, x_raw, x_fft, x_wav):
        fft_resized = self.resize_spectrogram(x_fft, target_size=224)  
        ast_spec_features = self.ast_spectrogram(fft_resized)  # (B, embed_dim)

        wav_resized = self.resize_spectrogram(x_wav, target_size=224)
        ast_wavelet_features = self.ast_wavelet(wav_resized)  # (B, embed_dim)

        cnn_features = self.cnn_raw(x_raw)  # (B, 512)

        all_features = [ast_spec_features, ast_wavelet_features, cnn_features]
        fused_features = self.fusion(all_features)  # (B, fusion_hidden_dim)

        logits = self.classifier(fused_features)  # (B, num_classes)
        return logits

    
    def getbranch_features(self, x_raw, x_fft, x_wav):
        fft_resized = self.resize_spectrogram(x_fft, target_size=224)
        ast_spec_features = self.ast_spectrogram(fft_resized)

        wav_resized = self.resize_spectrogram(x_wav, target_size=224)
        ast_wavelet_features = self.ast_wavelet(wav_resized)

        cnn_features = self.cnn_raw(x_raw)

        return ast_spec_features, ast_wavelet_features, cnn_features


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

