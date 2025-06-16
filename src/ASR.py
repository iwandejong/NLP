from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoProcessor, SeamlessM4Tv2ForSpeechToText
import torch
from jiwer import wer
from typing import List, Dict
from tqdm import tqdm
from datasets import Audio

class ASR:
  def __init__(self, model_name):
    self.whisper_processor = WhisperProcessor.from_pretrained(model_name)
    self.whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name)
    self.whisper_model.to("cuda" if torch.cuda.is_available() else "cpu")

    self.m4tv2_model_id = "facebook/seamless-m4t-v2-large"
    self.m4tv2_processor = AutoProcessor.from_pretrained(self.m4tv2_model_id)
    self.m4tv2_model = SeamlessM4Tv2ForSpeechToText.from_pretrained(self.m4tv2_model_id)
    self.m4tv2_model.to("cuda" if torch.cuda.is_available() else "cpu")

  def transcribe_whisper(self, audio_data: List[Dict]) -> List[str]:
    transcriptions = []
    print("Transcribing with Whisper...")
    
    for i, sample in tqdm(enumerate(audio_data), total=len(audio_data), desc="Transcribing"):
      try:
        # Process audio with attention mask
        input_features = self.whisper_processor(
          sample['audio'], 
          sampling_rate=sample['sampling_rate'],
          return_tensors="pt"
        ).input_features.to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate transcription with optimized parameters
        with torch.no_grad():
          # First get the forced decoder ids
          forced_decoder_ids = self.whisper_processor.get_decoder_prompt_ids(
            language="af", task="transcribe"
          )
            
          # Generate transcription
          predicted_ids = self.whisper_model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            temperature=0.0,  # Use deterministic generation
            max_length=448,   # Maximum sequence length
            num_beams=5,      # Beam search
            return_timestamps=False
          )
        
        transcription = self.whisper_processor.batch_decode(
          predicted_ids, skip_special_tokens=True
        )[0]
        
        transcriptions.append(transcription)
      except Exception as e:
        print(f"Error transcribing sample {i}: {e}")
        transcriptions.append("")
    
    return transcriptions
  
  def transcribe_m4tv2(self, audio_data: List[Dict]) -> List[str]:
    transcriptions = []

    for x in tqdm(audio_data, desc="Transcribing with M4T-V2", total=len(audio_data)):
      inputs = self.m4tv2_processor(
        audios=x['audio'],
        sampling_rate=16000,
        return_tensors="pt",
        src_lang="afr"
      ).to("cuda" if torch.cuda.is_available() else "cpu")

      output_tokens = self.m4tv2_model.generate(**inputs, tgt_lang="afr")
      transcription = self.m4tv2_processor.batch_decode(output_tokens, skip_special_tokens=True)[0]

      transcriptions.append(transcription)
    return transcriptions
  
  def calculate_wer(self, reference_texts: List[str], hypothesis_texts: List[str]) -> float:
    total_wer = 0
    valid_pairs = 0
    
    for ref, hyp in zip(reference_texts, hypothesis_texts):
      if ref.strip() and hyp.strip():
        try:
          total_wer += wer(ref, hyp)
          valid_pairs += 1
        except Exception as e:
          print(f"Error calculating WER: {e}")
          continue
    
    return total_wer / valid_pairs if valid_pairs > 0 else 1.0