import os
import librosa
from tqdm import tqdm

class Loader:
  def __init__(self):
    pass

  def load_data(self, min_duration: float = 3.0):
    """Load Afrikaans data from Common Voice dataset"""
    print("Loading Common Voice Afrikaans dataset...")
    
    try:
      clips_dir = '/root/cv-corpus-21.0-2025-03-14/af/clips'
      tsv_files = [
        '/root/cv-corpus-21.0-2025-03-14/af/test.tsv',
        '/root/cv-corpus-21.0-2025-03-14/af/train.tsv',
        '/root/cv-corpus-21.0-2025-03-14/af/validated.tsv'
      ]
      
      # Load transcriptions from all TSV files
      transcriptions = {}
      for tsv_path in tsv_files:
        with open(tsv_path, 'r', encoding='utf-8') as f:
          next(f)  # Skip header
          for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
              filename = parts[1]
              text = parts[2]
              transcriptions[filename] = text
      
      # Load audio files
      audio_data = []
      for filename in os.listdir(clips_dir):
        if filename.endswith('.mp3'):
          if filename in transcriptions:
            audio_path = os.path.join(clips_dir, filename)
            # Load audio file and resample to 16000 Hz
            audio, sr = librosa.load(audio_path, sr=16000)  # Force 16000 Hz sampling rate
            audio_data.append({
              'audio': audio,
              'sampling_rate': 16000,  # Set sampling rate to 16000
              'text': transcriptions[filename]
            })
      
      print(f"Loaded {len(audio_data)} audio samples from all splits")
      return audio_data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

  def load_podcast_chunks(self, chunk_dir="podcast_chunks"):
    """Load audio chunks from a directory, no reference transcript."""
    import librosa, os
    audio_data = []
    for fname in tqdm(os.listdir(chunk_dir), desc="Loading podcast chunks"):
      if fname.endswith(".mp3"):
        path = os.path.join(chunk_dir, fname)
        audio, sr = librosa.load(path, sr=16000)
        audio_data.append({'audio': audio, 'sampling_rate': 16000, 'text': ""})
    print(f"Loaded {len(audio_data)} podcast chunks from {chunk_dir}")

    # Select 10 random samples if more than 10
    if len(audio_data) > 10:
      import random
      audio_data = random.sample(audio_data, 10)
    if not audio_data:
      print("No audio data found in the specified directory.")
      return []
    return audio_data