from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, pipeline as hf_pipeline
import torch
from typing import List
from tqdm import tqdm

class Translate:
  def __init__(self):
    self.m2m_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    self.m2m_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    self.m2m_model.to("cuda" if torch.cuda.is_available() else "cpu")

    self.nllb_translator = hf_pipeline(
      "translation", 
      model="facebook/nllb-200-distilled-600M",
      device=0 if torch.cuda.is_available() else -1
    )

  def translate_m2m(self, texts: List[str]) -> List[str]:
    if self.m2m_model is None:
      self.load_translation_models()
    
    translations = []
    print("Translating with M2M-100...")
    
    self.m2m_tokenizer.src_lang = "af"
    
    for i, text in tqdm(enumerate(texts), total=len(texts), desc="Translating"):
      if not text.strip():
        translations.append("")
        continue
          
      try:
        encoded = self.m2m_tokenizer(text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        generated_tokens = self.m2m_model.generate(
          **encoded, 
          forced_bos_token_id=self.m2m_tokenizer.get_lang_id("en")
        )
        translation = self.m2m_tokenizer.batch_decode(
          generated_tokens, skip_special_tokens=True
        )[0]
        translations.append(translation)
        
        if (i + 1) % 10 == 0:
          print(f"Translated {i + 1}/{len(texts)} texts")
              
      except Exception as e:
        print(f"Error translating text {i}: {e}")
        translations.append("")
    
    return translations
  
  def translate_nllb(self, texts: List[str]) -> List[str]:
    if self.nllb_translator is None:
      self.load_translation_models()
    
    translations = []
    print("Translating with NLLB...")
    
    for i, text in tqdm(enumerate(texts), total=len(texts), desc="Translating with NLLB"):
      if not text.strip():
        translations.append("")
        continue
          
      try:
        result = self.nllb_translator(
          text, 
          src_lang="afr_Latn", 
          tgt_lang="eng_Latn"
        )

        translations.append(result[0]['translation_text'])
              
      except Exception as e:
        print(f"Error translating text {i}: {e}")
        translations.append("")
    
    return translations