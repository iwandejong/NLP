from nltk.corpus import stopwords
import re
from typing import List
from tqdm import tqdm

class Preprocess:
  def __init__(self):
    self.afrikaans_stopwords = set([
        'die', 'van', 'en', 'in', 'is', 'het', 'dat', 'met', 'op', 
        'vir', 'te', 'by', 'as', 'aan', 'was', 'sy', 'nie', 'hy',
        'dit', 'haar', 'wat', 'word', 'sal', 'kan', 'ook', 'maar',
        'so', 'nog', 'tot', 'na', 'om', 'oor', 'uit', 'al', 'daar'
    ])

  def preprocess_text(self, texts: List[str], language: str = "af") -> List[str]:
    processed_texts = []
    
    # Select stopwords based on language
    if language == "af":
      stop_words = self.afrikaans_stopwords
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
      
      # Only add if we have tokens after preprocessing
      if tokens:
        processed_texts.append(' '.join(tokens))

    print(f"Number of valid documents after preprocessing: {len(processed_texts)}")
    return processed_texts