import torch
import librosa
import numpy as np
from model import create_model, ThreeBranchDeepfakeDetector
import torch.nn.functional as F

class AudioDeepfakeDetector:
    # Predict other files
    
    def __init__(self, model_path, device='cuda', sample_rate=16000, duration=2.0):
        self.device = device
        self.sample_rate = sample_rate
        self.duration = duration
        self.input_length = int(sample_rate * duration)
        
        # Create model
        self.model = create_model(
            sample_rate=sample_rate,
            input_length=self.input_length
        )
        
        self.load_model(model_path)
        self.model.eval()
        
    def load_model(self, model_path):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            raise Exception(f"Error loading model: {e}")
        
    def preprocess_audio(self, audio_path):
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Split into overlapping chunks and pad to chunk size
            chunk_size = self.input_length
            chunks = []
            
            if len(audio) < chunk_size:
                audio = np.pad(audio, (0, chunk_size - len(audio)))
            

            stride = chunk_size // 2
            for i in range(0, len(audio) - chunk_size + 1, stride):
                chunk = audio[i:i + chunk_size]
                chunks.append(chunk)
            
            if not chunks:
                chunks = [audio[:chunk_size]]
            
            audio_tensor = torch.FloatTensor(np.array(chunks)).unsqueeze(1)  # (N, 1, T)
            
            return audio_tensor
            
        except Exception as e:
            raise Exception(f"Error preprocessing audio: {e}")
    
    def predict_single_file(self, audio_path, return_probabilities=True):
        try:
            # Preprocess
            audio_tensor = self.preprocess_audio(audio_path)
            audio_tensor = audio_tensor.to(self.device)
            
            # Predict in batches to avoid memory issues
            batch_size = 8
            all_outputs = []
            
            with torch.no_grad():
                for i in range(0, len(audio_tensor), batch_size):
                    batch = audio_tensor[i:i + batch_size]
                    outputs = self.model(batch)
                    all_outputs.append(outputs)
            
            # Concatenate all outputs
            outputs = torch.cat(all_outputs, dim=0)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            # Aggregate predictions (majority vote and average probabilities)
            chunk_predictions = predictions.cpu().numpy()
            chunk_probabilities = probabilities.cpu().numpy()
            
            # Final prediction based on average probabilities
            avg_probs = np.mean(chunk_probabilities, axis=0)
            final_prediction = np.argmax(avg_probs)
            final_confidence = avg_probs[final_prediction]
            
            result = {
                'prediction': 'AI Generated' if final_prediction == 1 else 'Real',
                'confidence': float(final_confidence),
                'label': int(final_prediction),
                'num_chunks': len(chunk_predictions),
                'chunk_predictions': chunk_predictions.tolist(),
                'chunk_probabilities': chunk_probabilities.tolist()
            }
            
            if return_probabilities:
                result['real_probability'] = float(avg_probs[0])
                result['fake_probability'] = float(avg_probs[1])
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    def predict_batch(self, audio_paths):
        """Predict multiple audio files"""
        results = []
        for audio_path in audio_paths:
            result = self.predict_single_file(audio_path)
            result['file_path'] = audio_path
            results.append(result)
        return results
    
    def analyze_audio(self, audio_path):
        try:
            # Get basic prediction
            result = self.predict_single_file(audio_path)
            
            if 'error' in result:
                return result
            
            # Get branch-specific features
            audio_tensor = self.preprocess_audio(audio_path)
            audio_tensor = audio_tensor.to(self.device)
            
            # Take first chunk for analysis
            sample_chunk = audio_tensor[:1]
            
            with torch.no_grad():
                branch_features = self.model.get_branch_features(sample_chunk)
            
            # Calculate feature statistics
            analysis = {
                'file_path': audio_path,
                'prediction': result,
                'branch_analysis': {
                    'spectrogram_features': {
                        'mean': float(torch.mean(branch_features['spectrogram'])),
                        'std': float(torch.std(branch_features['spectrogram'])),
                        'max': float(torch.max(branch_features['spectrogram'])),
                        'min': float(torch.min(branch_features['spectrogram']))
                    },
                    'wavelet_features': {
                        'mean': float(torch.mean(branch_features['wavelet'])),
                        'std': float(torch.std(branch_features['wavelet'])),
                        'max': float(torch.max(branch_features['wavelet'])),
                        'min': float(torch.min(branch_features['wavelet']))
                    },
                    'raw_features': {
                        'mean': float(torch.mean(branch_features['raw'])),
                        'std': float(torch.std(branch_features['raw'])),
                        'max': float(torch.max(branch_features['raw'])),
                        'min': float(torch.min(branch_features['raw']))
                    }
                }
            }
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}