from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from bertopic import BERTopic
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import pipeline
import numpy as np
import random
from typing import List, Dict

class Topic:
  def __init__(self):
    pass

  def train_lda(self, texts: List[str], n_topics: int = None) -> Tuple[Optional[LatentDirichletAllocation], Optional[CountVectorizer]]:
    print("Training LDA...")
    
    # Remove empty texts and ensure we have valid strings
    texts = [text for text in texts if text and isinstance(text, str) and text.strip()]
    
    if len(texts) < 2:
      print("Not enough valid texts for topic modeling.")
      return None, None
    
    # Vectorize texts with improved parameters
    vectorizer = CountVectorizer(
      min_df=2,
      max_df=0.7,
      ngram_range=(1, 1),
      max_features=10000
    )
    
    try:
      doc_term_matrix = vectorizer.fit_transform(texts)
      
      # Train LDA with improved parameters
      lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=200,
        learning_method='batch',
        learning_offset=10.0,
        batch_size=4,
        n_jobs=-1,
        doc_topic_prior=0.1,
        topic_word_prior=0.01
      )
      lda.fit(doc_term_matrix)
      
      return lda, vectorizer
    except Exception as e:
      print(f"Error training LDA: {e}")
      return None, None
        
  def train_bertopic(self, texts: List[str]) -> Optional[BERTopic]:
    print("Training BERTopic...")
    
    # Remove empty texts
    texts = [text for text in texts if text.strip()]
    print(f"BERTopic: {len(texts)} valid documents to train on")

    if len(texts) == 0:
      print("No valid texts for BERTopic training")
      return None
    
    # Initialize BERTopic with improved configuration
    topic_model = BERTopic(
      embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
      min_topic_size=5,
      n_gram_range=(1, 1),
      verbose=True,
      calculate_probabilities=True,
      nr_topics="auto",
      language="multilingual"
    )
    
    try:
      topics, probs = topic_model.fit_transform(texts)
      return topic_model
    except Exception as e:
      print(f"Error training BERTopic: {e}")
      return None

  def calculate_lda_topic_coherence(self, texts: List[str], lda_model, vectorizer):
    if not texts or lda_model is None or vectorizer is None:
      return {'umass': 0, 'npmi': 0}
    try:
      # Tokenize texts with improved preprocessing
      tokenized_texts = [text.split() for text in texts if text.strip()]
      
      # Get feature names from vectorizer
      feature_names = vectorizer.get_feature_names_out()
      
      # Get top words for each topic with increased number
      topics = []
      for topic_idx, topic in enumerate(lda_model.components_):
        top_words_idx = topic.argsort()[:-20-1:-1]  # Increased to 20 words
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append(top_words)
      
      # Create dictionary and corpus for coherence calculation
      dictionary = Dictionary(tokenized_texts)
      corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
      
      # Calculate UMass coherence with improved parameters
      cm_umass = CoherenceModel(
        topics=topics,
        corpus=corpus,
        dictionary=dictionary,
        coherence='u_mass',
        processes=-1
      )
      umass_score = cm_umass.get_coherence()
      
      # Calculate NPMI coherence with improved parameters
      cm_npmi = CoherenceModel(
        topics=topics,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence='c_npmi',
        processes=-1
      )
      npmi_score = cm_npmi.get_coherence()
      
      return {'umass': umass_score, 'npmi': npmi_score}
    except Exception as e:
      print(f"Error calculating coherence: {e}")
      return {'umass': 0, 'npmi': 0}

  def calculate_bertopic_topic_coherence(self, texts: List[str], bertopic_model, top_n: int = 20) -> Dict[str, float]:
    if not texts or bertopic_model is None:
      return {'umass': 0, 'npmi': 0}

    try:
      # Tokenize texts
      tokenized_texts = [text.split() for text in texts if text.strip()]

      # Extract topics directly from BERTopic
      topics = []
      for topic_id in bertopic_model.get_topics():
        if topic_id != -1:  # Skip outlier topic
          words_scores = bertopic_model.get_topics()[topic_id][:top_n]
          topic_words = [word for word, _ in words_scores]
          topics.append(topic_words)

      # Prepare corpus for coherence model
      dictionary = Dictionary(tokenized_texts)
      corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

      # UMass coherence
      cm_umass = CoherenceModel(
        topics=topics,
        corpus=corpus,
        dictionary=dictionary,
        coherence='u_mass',
        processes=-1
      )
      umass_score = cm_umass.get_coherence()

      # NPMI coherence
      cm_npmi = CoherenceModel(
        topics=topics,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence='c_npmi',
        processes=-1
      )
      npmi_score = cm_npmi.get_coherence()

      return {'umass': umass_score, 'npmi': npmi_score}

    except Exception as e:
      print(f"Error calculating coherence: {e}")

  def calculate_lda_c_v_coherence(self, texts: List[str], lda_model, vectorizer):
    if not texts or lda_model is None or vectorizer is None:
      return {'c_v': 0}
    
    try:
      # Tokenize texts
      tokenized_texts = [text.split() for text in texts if text.strip()]
      
      # Get feature names from vectorizer
      feature_names = vectorizer.get_feature_names_out()
      
      # Get top words for each topic
      topics = []
      for topic_idx, topic in tqdm(lda_model.components_, desc="Extracting top words for topics"):
        top_words_idx = topic.argsort()[:-10-1:-1]  # Reduced to 10 words
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append(top_words)
      
      # Create dictionary and corpus for coherence calculation
      dictionary = Dictionary(tokenized_texts)
      corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
      
      # Calculate C_V coherence with improved parameters
      cm_cv = CoherenceModel(
        topics=topics,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence='c_v',
        processes=-1
      )
      cv_score = cm_cv.get_coherence()
      
      return {'c_v': cv_score}
    except Exception as e:
      print(f"Error calculating C_V coherence: {e}")
      return {'c_v': 0}

  def optimal_topics(self, texts: List[str], min_topics: int = 2, max_topics: int = 20) -> Dict[int, Dict[str, float]]:
    coherence_scores = {}
    
    for n_topics in tqdm(range(min_topics, max_topics + 1), desc="Evaluating topics"):
      lda_model, vectorizer = self.train_lda(texts, n_topics)
      if lda_model is None or vectorizer is None:
        continue
      
      coherence = self.calculate_lda_topic_coherence(texts, lda_model, vectorizer)
      coherence_scores[n_topics] = coherence
    
    # minimise both umass and npmi (close to 0 is better, so take abs)
    best_n_topics = min(coherence_scores, key=lambda k: (abs(coherence_scores[k]['umass']), abs(coherence_scores[k]['npmi'])))
    best_scores = coherence_scores[best_n_topics]

    print(f"Optimal number of topics: {best_n_topics} with UMass: {best_scores['umass']}, NPMI: {best_scores['npmi']}")

    return best_n_topics

  def print_lda_topics(self, lda_model, vectorizer, n_words=10):
    feature_names = vectorizer.get_feature_names_out()
    for idx, topic in enumerate(lda_model.components_):
      top_words = [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
      print(f"LDA Topic {idx + 1}: {', '.join(top_words)}")

  def print_bertopic_topics(self, bertopic_model, n_words=10):
    topics = bertopic_model.get_topics()
    for topic_id, words in topics.items():
      if topic_id != -1:
        top_words = [word for word, _ in words[:n_words]]
        print(f"BERTopic {topic_id}: {', '.join(top_words)}")
