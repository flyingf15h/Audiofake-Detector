# Basic-Audi-Detector-Test

Audiofake detector trained on Kaggle fake-or-real-audio and HuggingFace MLAAD datasets.

**Kaggle Setup**

1. https://www.kaggle.com/account
2. Create API token
3. Download `kaggle.json`
4. Put in `~/.kaggle/kaggle.json` or  `%USERPROFILE%\.kaggle\kaggle.json`
5. Set permissions `chmod 600 ~/.kaggle/kaggle.json`


**Default Config**

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

Input (1, 16000) -> Conv1D Layers -> BatchNorm -> ReLU -> Dropout -> FC Layers -> Output (2)

Data Augmentation

- Noise Addition (30%)
- Time Shifting (20%)
- Speed Changes (15%)

Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

**Weights & Biases**

Enable W&B logging:

```python
config['use_wandb'] = True
```

Have W&B configured:
```bash
wandb login
```
