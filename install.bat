
@echo off
echo Setting up...

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
pip install --upgrade pip

:: Install PyTorch (with CUDA support if available)
echo Installing PyTorch...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo CUDA detected, installing PyTorch with CUDA support...
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo No CUDA detected, installing CPU-only PyTorch...
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
)

:: Install other requirements
echo Installing other requirements...
pip install -r requirements.txt

:: Create necessary directories
echo Creating directories...
mkdir data 2>nul
mkdir checkpoints 2>nul
mkdir results 2>nul

:: Setup Kaggle API
echo Setting up Kaggle API...
if not exist "%USERPROFILE%\.kaggle\kaggle.json" (
    echo Kaggle API not configured. Please:
    echo 1. Go to https://www.kaggle.com/account
    echo 2. Create a new API token
    echo 3. Download kaggle.json
    echo 4. Place it in %USERPROFILE%\.kaggle\kaggle.json
)

echo Setup complete!
echo To activate the environment: venv\Scripts\activate.bat
echo To run the system: python main.py
pause