from datasets import load_dataset
import os
import soundfile as sf
import numpy as np

def download_mlaad(save_path='./data/MLAAD'):
    print("Downloading MLAAD dataset from Hugging Face")
    dataset = load_dataset("mueller91/MLAAD", split="train")
    os.makedirs(save_path, exist_ok=True)

    for i, example in enumerate(dataset):
        audio = example["audio"]
        label = example["label"]  # 0 = real, 1 = fake
        label_folder = "real" if label == 0 else "fake"
        out_folder = os.path.join(save_path, label_folder)
        os.makedirs(out_folder, exist_ok=True)

        filename = os.path.join(out_folder, f"sample_{i}.wav")
        
        # Properly save audio file
        audio_array = np.array(audio["array"])
        sample_rate = audio["sampling_rate"]
        sf.write(filename, audio_array, sample_rate)
        
        if i % 100 == 0:
            print(f"Processed {i} samples...")
    
    print(f"MLAAD dataset downloaded and saved to {save_path}")

if __name__ == "__main__":
    download_mlaad()