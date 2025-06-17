import os
import librosa
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login

class Loader:
  def __init__(self):
    pass

  def load_data(self, min_duration: float = 3.0):
    """Load Afrikaans data from Common Voice dataset version 17.0"""
    print("Loading Common Voice Afrikaans dataset from Hugging Face...")
    
    try:
      # Log in to Hugging Face
      login(token="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")
      
      # Load dataset from Hugging Face
      dataset = load_dataset("mozilla-foundation/common_voice_17_0", "af", split="validated", trust_remote_code=True)
      
      # Convert to our format
      audio_data = []
      for item in tqdm(dataset, desc="Processing audio files"):
        audio = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]
        
        # Resample to 16000 Hz if needed
        if sr != 16000:
          audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
          sr = 16000
          
        audio_data.append({
          'audio': audio,
          'sampling_rate': sr,
          'text': item["sentence"]
        })
      
      print(f"Loaded {len(audio_data)} audio samples")
      return audio_data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

  def load_podcast_chunks(self, chunk_dir="podcast_chunks"):
    audio_data = []
    for fname in tqdm(os.listdir(chunk_dir), desc="Loading podcast chunks"):
        if fname.endswith(".mp3"):
            path = os.path.join(chunk_dir, fname)
            try:
                audio, sr = librosa.load(path, sr=16000)
                if audio is None or len(audio) == 0:
                    continue
                audio_data.append({'audio': audio, 'sampling_rate': 16000, 'text': ""})
            except Exception as e:
                print(f"Failed to load {fname}: {e}")

    # Select 10 random samples if more than 10
    # if len(audio_data) > 10:
    #     import random
    #     audio_data = random.sample(audio_data, 10)
    # if not audio_data:
    #     print("No audio data found in the specified directory.")
    #     return []

    return audio_data
