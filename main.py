import torch
from preprocess import prepare_dataset
from model import create_model, ThreeBranchDeepfakeDetector
from train import train_model, evaluate_model
from data import download_mlaad
import os
import json
import subprocess
from torch.utils.data import DataLoader

def download_kaggle_dataset():
    print("Downloading 'the-fake-or-real-dataset' from Kaggle...")
    os.makedirs('./data', exist_ok=True)
    try:
        subprocess.run([
            'kaggle', 'datasets', 'download',
            '-d', 'sbhatti/the-fake-or-real-audio-dataset',
            '--unzip', '-p', './data'
        ], check=True)
        print("Kaggle dataset download complete.")
        
        # Reorganize files if needed
        kaggle_dir = './data/the-fake-or-real-audio-dataset'
        if os.path.exists(kaggle_dir):
            # Move files to expected structure
            import shutil
            if not os.path.exists('./data/real'):
                os.makedirs('./data/real')
            if not os.path.exists('./data/fake'):
                os.makedirs('./data/fake')
            # You may need to adjust this based on actual Kaggle dataset structure
            print("Please manually organize files into ./data/real and ./data/fake folders")
        
    except Exception as e:
        print("Failed to download Kaggle dataset. Ensure kaggle API is set up.")
        print(f"Error: {e}")

def prepare_combined_dataset():
    """Prepare dataset from both sources"""
    # Download MLAAD dataset
    if not os.path.exists('./data/MLAAD'):
        download_mlaad('./data/MLAAD')
    
    # Download Kaggle dataset
    if not os.path.exists('./data/real') or not os.path.exists('./data/fake'):
        download_kaggle_dataset()
    
    # Combine datasets - copy MLAAD files to main data directory
    import shutil
    mlaad_real = './data/MLAAD/real'
    mlaad_fake = './data/MLAAD/fake'
    main_real = './data/real'
    main_fake = './data/fake'
    
    os.makedirs(main_real, exist_ok=True)
    os.makedirs(main_fake, exist_ok=True)
    
    # Copy MLAAD files
    if os.path.exists(mlaad_real):
        for file in os.listdir(mlaad_real):
            if file.endswith('.wav'):
                shutil.copy(os.path.join(mlaad_real, file), 
                          os.path.join(main_real, f"mlaad_{file}"))
    
    if os.path.exists(mlaad_fake):
        for file in os.listdir(mlaad_fake):
            if file.endswith('.wav'):
                shutil.copy(os.path.join(mlaad_fake, file), 
                          os.path.join(main_fake, f"mlaad_{file}"))

def train():
    config = {
        'data_dir': './data',
        'save_dir': './checkpoints',
        'batch_size': 16,  # Reduced for memory
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'sample_rate': 16000,
        'duration': 2.0,  # Increased duration
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'use_wandb': False
    }

    # Prepare combined dataset
    prepare_combined_dataset()

    print("Preparing datasets...")
    train_dataset, val_dataset, test_dataset = prepare_dataset(
        config['data_dir'],
        config['sample_rate'],
        config['duration']
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=2,  # Reduced for stability
        pin_memory=True if config['device'] == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=2,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=2,
        pin_memory=True if config['device'] == 'cuda' else False
    )

    print(f"Train samples: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    print("Creating model...")
    model = create_model(
        sample_rate=config['sample_rate'],
        input_length=int(config['sample_rate'] * config['duration'])
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("Training model...")
    model, history = train_model(
        model, train_loader, val_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        device=config['device'],
        save_dir=config['save_dir'],
        use_wandb=config['use_wandb']
    )

    print("Evaluating model...")
    results = evaluate_model(model, test_loader, config['device'], config['save_dir'])

    # Save results
    results_to_save = {
        'config': config,
        'results': {
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'confusion_matrix': results['confusion_matrix'].tolist()
        }
    }
    
    with open(os.path.join(config['save_dir'], 'results.json'), 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print("Training complete.")
    print(f"Model saved to: {config['save_dir']}/best_model.pt")
    print(f"Results saved to: {config['save_dir']}/results.json")


def inference_example():
    print("Running inference...")
    audio_file = input("Enter path to audio file: ").strip()

    if not os.path.exists(audio_file):
        print(f"File not found: {audio_file}")
        return

    from predict import AudioDeepfakeDetector
    
    detector = AudioDeepfakeDetector(
        model_path='./checkpoints/best_model.pt',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    result = detector.predict_single_file(audio_file)
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
        
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Real probability: {result['real_probability']:.4f}")
    print(f"Fake probability: {result['fake_probability']:.4f}")

    analysis = detector.analyze_audio(audio_file)
    if 'error' not in analysis:
        print("\nDetailed analysis:")
        print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    print("Audio Deepfake Detection System")
    print("=" * 40)
    print("1. Train model using combined datasets")
    print("2. Run inference on audio file")
    print("3. Download datasets only")

    choice = input("Enter choice (1-3): ").strip()

    if choice == '1':
        train()
    elif choice == '2':
        inference_example()
    elif choice == '3':
        prepare_combined_dataset()
        print("Datasets downloaded and prepared.")
    else:
        print("Invalid choice.")