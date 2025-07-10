import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from pathlib import Path
import torchaudio
import numpy as np
import pywt
import json

class CachedAudioDataset(Dataset):
    def __init__(self, cache_dir, chunk_size=16000, augment=False):
        # cache_dir has npy files and json files
        self.cache_dir = Path(cache_dir)
        self.chunk_size = chunk_size
        self.augment = augment

        metadata_path = self.cache_dir / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata JSON not found in {self.cache_dir}")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)  # List of dicts: {'feature_path': str, 'label': int}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        entry = self.metadata[idx]
        feat_path = self.cache_dir / entry['feature_path']
        label = entry['label']

        features = np.load(feat_path, allow_pickle=True).item()

        x_raw = torch.tensor(features['raw'], dtype=torch.float32)
        x_fft = torch.tensor(features['stft'], dtype=torch.float32)
        x_wav = torch.tensor(features['wav'], dtype=torch.float32)

        if self.augment:
            x_raw = self._augment(x_raw)

        return x_raw, x_fft, x_wav, torch.tensor(label, dtype=torch.long)

    def _augment(self, x_raw):
        if torch.rand(1) < 0.3:
            shift = torch.randint(-1600, 1600, (1,)).item()
            x_raw = torch.roll(x_raw, shifts=shift, dims=1)
        if torch.rand(1) < 0.3:
            scale = torch.empty(1).uniform_(0.8, 1.2).item()
            x_raw = x_raw * scale
        if torch.rand(1) < 0.2:
            noise = torch.randn_like(x_raw) * 0.005
            x_raw = x_raw + noise
        return x_raw

def cache_features(data_dir, oc_dir, sample_rate=16000, chunk_size=16000, overwrite=False):
    #  Precompute and cache features for audio files in data_dir and save in oc_dir.

    from tqdm import tqdm
    import json

    data_dir = Path(data_dir)
    oc_dir = Path(oc_dir)
    oc_dir.mkdir(parents=True, exist_ok=True)

    metadata = []

    for cls in ['real', 'fake']:
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            print(f"Warning: Class folder {cls_dir} does not exist, skipping.")
            continue

        audio_files = list(cls_dir.glob('*.wav')) + list(cls_dir.glob('*.mp3')) + list(cls_dir.glob('*.flac')) + list(cls_dir.glob('*.m4a'))

        print(f"Caching {len(audio_files)} {cls} files from {cls_dir}")

        for audio_path in tqdm(audio_files):
            cache_filename = f"{audio_path.stem}_sr{sample_rate}_chunk{chunk_size}.npy"
            cache_path = oc_dir / cache_filename

            if cache_path.exists() and not overwrite:
                metadata.append({'feature_path': cache_filename, 'label': 1 if cls == 'fake' else 0})
                continue

            # Load audio with torchaudio
            try:
                waveform, sr_orig = torchaudio.load(str(audio_path))
                if sr_orig != sample_rate:
                    resampler = torchaudio.transforms.Resample(sr_orig, sample_rate)
                    waveform = resampler(waveform)
                waveform = waveform.mean(dim=0) 
                waveform = waveform.numpy()

                if len(waveform) < chunk_size:
                    pad_width = chunk_size - len(waveform)
                    waveform = np.pad(waveform, (0, pad_width), mode='constant')
                else:
                    waveform = waveform[:chunk_size]

                # Normalize waveform
                wav_mean, wav_std = waveform.mean(), waveform.std()
                if wav_std < 1e-6:
                    wav_std = 1.0
                raw_norm = (waveform - wav_mean) / wav_std
                raw_norm = np.expand_dims(raw_norm.astype(np.float32), axis=0)  # (1, chunk_size)

                # STFT
                try:
                    import librosa
                    stft = librosa.stft(waveform, n_fft=256, hop_length=128)
                    mag = np.abs(stft)[:128, :128]
                    mag_mean, mag_std = mag.mean(), mag.std()
                    if mag_std < 1e-6:
                        mag_std = 1.0
                    stft_norm = (mag - mag_mean) / mag_std
                    stft_norm = np.expand_dims(stft_norm.astype(np.float32), axis=0)  # (1,128,128)
                except Exception as e:
                    print(f"Failed STFT for {audio_path}: {e}")
                    stft_norm = np.zeros((1, 128, 128), dtype=np.float32)

                # Wavelet
                try:
                    coeffs = pywt.wavedec(waveform, 'db4', level=4)
                    cA4 = coeffs[0]
                    cA4_resized = np.resize(cA4, (64, 128))
                    wav_mean, wav_std = cA4_resized.mean(), cA4_resized.std()
                    if wav_std < 1e-6:
                        wav_std = 1.0
                    wav_norm = (cA4_resized - wav_mean) / wav_std
                    wav_norm = np.expand_dims(wav_norm.astype(np.float32), axis=0)  # (1,64,128)
                except Exception as e:
                    print(f"Failed Wavelet for {audio_path}: {e}")
                    wav_norm = np.zeros((1, 64, 128), dtype=np.float32)

                # Save dict
                np.save(cache_path, {'raw': raw_norm, 'stft': stft_norm, 'wav': wav_norm})

                metadata.append({'feature_path': cache_filename, 'label': 1 if cls == 'fake' else 0})

            except Exception as e:
                print(f"Failed processing {audio_path}: {e}")

    # Save metadata JSON
    with open(oc_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f)

    print(f"Caching complete. Cached features saved to {oc_dir}")
    print(f"Metadata saved to {oc_dir / 'metadata.json'}")

def prep_cachedata(cache_dir, test_size=0.2, val_size=0.1, augment_train=True):
    # Split cached metadata into train/val/test datasets.
    import json
    from sklearn.model_selection import train_test_split

    cache_dir = Path(cache_dir)
    metadata_path = cache_dir / 'metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Extract paths and labels
    paths = [m['feature_path'] for m in metadata]
    labels = [m['label'] for m in metadata]

    X_temp, X_test, y_temp, y_test = train_test_split(
        paths, labels, test_size=test_size, random_state=42, stratify=labels
    )

    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction, random_state=42, stratify=y_temp
    )

    # Helper to build dataset from lists
    def build_ds(paths_list, labels_list, augment):
        items = [{'feature_path': p, 'label': l} for p, l in zip(paths_list, labels_list)]
        tmp_dir = cache_dir / ('train' if augment else 'val_test')
        tmp_dir.mkdir(exist_ok=True)
        json_path = tmp_dir / 'metadata.json'
        with open(json_path, 'w') as f:
            json.dump(items, f)
        return CachedAudioDataset(tmp_dir, augment=augment)

    train_ds = build_ds(X_train, y_train, augment=True)
    val_ds = build_ds(X_val, y_val, augment=False)
    test_ds = build_ds(X_test, y_test, augment=False)

    return train_ds, val_ds, test_ds


if __name__ == '__main__':
    data_dir = "/kaggle/input/the-fake-or-real-dataset/for-2sec/for-2seconds/training"
    cache_dir = "/kaggle/working/cached_features/training"

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    cache_features(data_dir, cache_dir, sample_rate=16000, chunk_size=16000, overwrite=False)

    train_ds, val_ds, test_ds = prep_cachedata("/kaggle/working/cached_features/training")