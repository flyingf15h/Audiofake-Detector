#!/bin/bash

# Create virtual environment
python -m venv venv
source venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing PyTorch with CUDA support..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "No CUDA detected, installing CPU-only PyTorch..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
pip install -r requirements.txt

# Create directories
mkdir -p data
mkdir -p checkpoints
mkdir -p results

# Setup Kaggle API 
echo "Setting up Kaggle API..."
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Kaggle API not configured. Please:"
    echo "1. Go to https://www.kaggle.com/account"
    echo "2. Create a new API token"
    echo "3. Download kaggle.json"
    echo "4. Place it in ~/.kaggle/kaggle.json"
    echo "5. Run: chmod 600 ~/.kaggle/kaggle.json"
fi

echo "Setup complete!"
echo "To activate the environment: source venv/bin/activate"
echo "To run the system: python main.py"