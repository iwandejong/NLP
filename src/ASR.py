from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoProcessor, SeamlessM4Tv2ForSpeechToText, AutoModelForSpeechSeq2Seq, pipeline
import torch
import evaluate
from typing import List, Dict
from tqdm import tqdm
from datasets import Audio

class ASR:
  def __init__(self):
    self.whisper_processor = None
    self.whisper_model = None
    self.m4tv2_processor = None
    self.m4tv2_model = None
    self.whisper_small_processor = None
    self.whisper_small_model = None
    self.whisper_small_pipe = None

  def _load_whisper(self):
    if self.whisper_processor is None or self.whisper_model is None:
      self.model_name = "openai/whisper-large-v3"
      self.whisper_processor = WhisperProcessor.from_pretrained(self.model_name)
      self.whisper_model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
      self.whisper_model.to("cuda" if torch.cuda.is_available() else "cpu")

  def _load_m4tv2(self):
    if self.m4tv2_processor is None or self.m4tv2_model is None:
      self.m4tv2_model_id = "facebook/seamless-m4t-v2-large"
      self.m4tv2_processor = AutoProcessor.from_pretrained(self.m4tv2_model_id)
      self.m4tv2_model = SeamlessM4Tv2ForSpeechToText.from_pretrained(self.m4tv2_model_id)
      self.m4tv2_model.to("cuda" if torch.cuda.is_available() else "cpu")

  def _load_whisper_small(self):
    if self.whisper_small_pipe is None:
      torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
      model_id = "openai/whisper-small"

      model = AutoModelForSpeechSeq2Seq.from_pretrained(
          model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
      )
      model.to("cuda" if torch.cuda.is_available() else "cpu")

      processor = AutoProcessor.from_pretrained(model_id)

      self.whisper_small_pipe = pipeline(
          "automatic-speech-recognition",
          model=model,
          tokenizer=processor.tokenizer,
          feature_extractor=processor.feature_extractor,
          torch_dtype=torch_dtype,
          device="cuda" if torch.cuda.is_available() else "cpu",
          generate_kwargs={"language": "af"}
      )

  def transcribe_whisper(self, audio_data: List[Dict]) -> List[str]:
    self._load_whisper()
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
    self._load_m4tv2()
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

  def transcribe_whisper_small(self, audio_data: List[Dict]) -> List[str]:
    self._load_whisper_small()
    transcriptions = []
    
    for x in tqdm(audio_data, desc="Transcribing with Whisper Small", total=len(audio_data)):
      try:
        result = self.whisper_small_pipe(x["audio"])
        transcriptions.append(result["text"])
      except Exception as e:
        print(f"Error transcribing with Whisper Small: {e}")
        transcriptions.append("")
    
    return transcriptions
  
  def calculate_wer(self, reference_texts: List[str], hypothesis_texts: List[str]) -> float:
    wer_metric = evaluate.load("wer")
    return wer_metric.compute(predictions=hypothesis_texts, references=reference_texts)