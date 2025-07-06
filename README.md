# Basic-Audi-Detector-Test

Audio deepfake detector trained on Kaggle fake-or-real-audio and HuggingFace MLAAD datasets.

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd audio-deepfake-detector

# Run the installation script
chmod +x install.sh
./install.sh

# Windows:
install.bat
```

### 2. Manual Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data checkpoints results
```

### 3. Setup Kaggle API 

1. Go to https://www.kaggle.com/account
2. Create a new API token
3. Download `kaggle.json`
4. Place it in `~/.kaggle/kaggle.json` (Linux/Mac) or `%USERPROFILE%\.kaggle\kaggle.json` (Windows)
5. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

## Usage

### Interactive Mode

```bash
python main.py
```

This will open a menu:
1. Train model
2. Check audio files
3. Check system requirements
4. View training configuration

### Command Line Mode

```bash
# Train model
python main.py --mode train

# Run inference on single file
python main.py --mode inference --audio /path/to/audio.wav

# Check system requirements
python main.py --mode check
```

## Dataset Configuration

1. **Kaggle Dataset**: `sbhatti/the-fake-or-real-audio-dataset`
   - Automatically downloaded
   - `real/` and `fake/` directories

2. **MLAAD Dataset**: `mueller91/MLAAD`
   - Automatically downloaded

```
data/
├── the-fake-or-real-audio-dataset/
│   ├── real/
│   │   ├── audio1.wav
│   │   └── audio2.wav
│   └── fake/
│       ├── audio1.wav
│       └── audio2.wav
├── mlaad_cache/
│   └── (HuggingFace dataset cache)
└── ...
```

## Training Configuration

### Default Configuration

```python
config = {
    'data_dir': './data',
    'save_dir': './checkpoints',
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'sample_rate': 16000,
    'duration': 1.0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'use_wandb': False,
    'use_kaggle': True,
    'use_mlaad': True,
    'max_mlaad_samples': 5000,
    'test_size': 0.2,
    'val_size': 0.1
}
```

### Customization

You can modify the configuration by:
1. Editing the `setup_training_config()` function in `main.py`
2. Creating a custom config file (JSON format)
3. Using environment variables

## Model Architecture

Input (1, 16000) -> Conv1D Layers -> BatchNorm -> ReLU -> Dropout -> FC Layers -> Output (2)

## Data Augmentation

- **Noise Addition**: Gaussian noise (30% probability)
- **Time Shifting**: Random circular shifts (20% probability)
- **Speed Perturbation**: Slight speed changes (15% probability)

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of classification results


### Using Weights & Biases

Enable W&B logging:

```python
config['use_wandb'] = True
```

Make sure you have W&B configured:
```bash
wandb login
```
