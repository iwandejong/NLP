from nltk.corpus import stopwords
import re
from typing import List
from tqdm import tqdm

class Preprocess:
  def __init__(self):
    self.AFRIKAANS_STOPWORDS = set([
      'aangaande', 'aangesien', 'al', 'alhoewel', 'as', 'baie', 
      'behalwe', 'belangrikste', 'beperkte', 'by', 'daar', 
      'daardie', 'daarin', 'dan', 'dat', 'die', 'dit', 
      'dus', 'een', 'en', 'ens', 'geen', 'gewees', 'haar', 
      'het', 'hom', 'hulle', 'hy', 'in', 'is', 'jou', 
      'kan', 'kon', 'maw', 'met', 'moet', 'my', 'na', 
      'nie', 'om', 'ons', 'onses', 'op', 'saam', 'sal', 
      'se', 'self', 'sien', 'so', 'sonder', 'sou', 'sy', 
      'synde', 'teen', 'terwyl', 'toe', 'tussen', 'uit', 
      'van', 'vir', 'was', 'wat', 'wees', 'waar', 'wanneer', 
      'wat', 'wie', 'wil', 'word', 'worden'
    ])

  def preprocess_text(self, texts: List[str], language: str = "af", min_doc_length: int = 0) -> List[str]:
    processed_texts = []
    
    # Select stopwords and stemmer based on language
    if language == "af":
      stop_words = set(self.AFRIKAANS_STOPWORDS)
    else:
      try:
        stop_words = set(stopwords.words('english'))
      except:
        stop_words = set()
    
    for text in tqdm(texts, desc="Preprocessing texts"):
      if not text or not isinstance(text, str) or not text.strip():
          continue
      
      # Convert to lowercase
      text = text.lower()
      
      # Remove URLs
      text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
      
      # Remove special characters and numbers
      text = re.sub(r'[^a-zA-Z\s]', ' ', text)
      
      # Remove extra whitespace
      text = re.sub(r'\s+', ' ', text).strip()
      
      # Simple tokenization and remove stopwords
      tokens = text.split()
      tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
      
      # Only add if we have enough tokens after preprocessing
      if len(tokens) >= min_doc_length:
        processed_texts.append(' '.join(tokens))

    print(f"Number of valid documents after preprocessing: {len(processed_texts)}")
    return processed_texts