from setuptools import setup, find_packages

setup(
    name="audio-deepfake-detector",
    version="1.0.0",
    description="Multi-dataset audio deepfake detection system",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "torchaudio>=0.12.0",
        "librosa>=0.9.0",
        "soundfile>=0.10.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "datasets>=2.0.0",
        "transformers>=4.20.0",
        "kaggle>=1.5.0",
        "wandb>=0.13.0",
        "requests>=2.25.0",
    ],
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/audio-deepfake-detector",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)

