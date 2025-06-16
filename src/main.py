#!/usr/bin/env python3
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# * Modal Server Section
import modal
import pathlib
app = modal.App("nlp-project")
gpu = "A100"

image = (
  modal.Image.debian_slim()
  .pip_install(
  "transformers",
  "torch",
  "torchaudio",
  "datasets",
  "librosa",
  "jiwer",
  "scikit-learn",
  "gensim",
  "bertopic",
  "nltk",
  "matplotlib",
  "seaborn",
  "plotly",
  "sentence-transformers",
  "umap-learn",
  "hdbscan",
  "sentencepiece",
  "tqdm"
)
  .run_commands(
    "python -c \"from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration; "
    "M2M100Tokenizer.from_pretrained('facebook/m2m100_418M'); "
    "M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418M')\""
  )
  .add_local_dir("cv-corpus-21.0-2025-03-14", remote_path="/root/cv-corpus-21.0-2025-03-14")
  .add_local_dir("podcast_chunks", remote_path="/root/podcast_chunks")
)


volume = modal.Volume.from_name("storage", create_if_missing=True)
VOL_MOUNT_PATH = pathlib.Path("/vol")

HOURS = 60 * 60

@app.function(
  gpu=gpu,
  image=image,
  timeout=int(3 * HOURS),
  volumes={VOL_MOUNT_PATH: volume}, 
)
def main():
  import warnings
  import torch
  import nltk

  from src.loader import Loader
  from src.preprocess import Preprocess
  from src.ASR import ASR
  from src.translate import Translate
  from src.topic import Topic

  warnings.filterwarnings('ignore')

  try:
      nltk.data.find('corpora/stopwords')
  except LookupError:
      nltk.download('stopwords')

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Using device: {device}")
  
  print("Loading Loader")
  loader = Loader()
  print("Loading Preprocess")
  preprocess = Preprocess()
  print("Loading ASR")
  asr = ASR(model_name="openai/whisper-large-v3")
  print("Loading Translate")
  translate = Translate()
  print("Loading Topic")
  topic = Topic()
  
  results = {
    'asr_results': {},
    'translation_results': {},
    'topic_results': {},
    'evaluation_metrics': {}
  }
            
  # ! Main pipeline
  print("Starting pipeline execution...")
  print("Loading podcast chunks...")
  audio_data = loader.load_podcast_chunks(chunk_dir="/root/podcast_chunks")

  print("Transcribing audio...")
  # transcripts = asr.transcribe_whisper(audio_data)
  transcripts = asr.transcribe_m4tv2(audio_data)

  results['asr_results'] = {
    'whisper': transcripts,
  }

  translations = translate.translate_nllb(transcripts)
  
  results['translation_results'] = {
    'nllb': translations,
  }

  processed_af = preprocess.preprocess_text(transcripts, language="af", min_doc_length=15)
  processed_en = preprocess.preprocess_text(translations, language="en", min_doc_length=15)

  if len(processed_af) == 0 or len(processed_en) == 0:
    print("No valid documents after preprocessing. Exiting pipeline.")
    return results

  avg_len_af = sum(len(doc.split()) for doc in processed_af) / len(processed_af)
  avg_len_en = sum(len(doc.split()) for doc in processed_en) / len(processed_en)
  
  print(f"Afrikaans: Average words per doc: {avg_len_af:.2f}")
  print(f"English: Average words per doc: {avg_len_en:.2f}")

  # Train LDA models with automatic topic number selection
  lda_af, vectorizer_af = topic.train_lda(processed_af)
  lda_en, vectorizer_en = topic.train_lda(processed_en)

  # Train BERTopic models with improved configuration
  bertopic_af = topic.train_bertopic(processed_af)
  bertopic_en = topic.train_bertopic(processed_en)

  # Print the topics for LDA models
  if lda_af:
    print("\nLDA Topics for Afrikaans:")
    lda_af_topics = []
    for idx, topics in enumerate(lda_af.components_):
      terms = [vectorizer_af.get_feature_names_out()[i] for i in topics.argsort()[-10:]]
      print(f"Topic {idx}: {', '.join(terms)}")
      lda_af_topics.append({
        'topic': idx,
        'terms': terms
      })

  if lda_en:
    print("\nLDA Topics for English:")
    lda_en_topics = []
    for idx, topics in enumerate(lda_en.components_):
      terms = [vectorizer_en.get_feature_names_out()[i] for i in topics.argsort()[-10:]]
      print(f"Topic {idx}: {', '.join(terms)}")
      lda_en_topics.append({
        'topic': idx,
        'terms': terms
      })

  if bertopic_af:
    print("\nBERTopic Topics for Afrikaans:")
    print(bertopic_af.get_topic_info())

  if bertopic_en:
    print("\nBERTopic Topics for English:")
    print(bertopic_en.get_topic_info())

  # Calculate coherence scores for LDA models
  coherence_af = topic.calculate_lda_topic_coherence(processed_af, lda_af, vectorizer_af)
  coherence_en = topic.calculate_lda_topic_coherence(processed_en, lda_en, vectorizer_en)

  # Calculate coherence scores for BERTopic models
  bertopic_coherence_af = topic.calculate_bertopic_topic_coherence(processed_af, bertopic_af)
  bertopic_coherence_en = topic.calculate_bertopic_topic_coherence(processed_en, bertopic_en)

  # Save to topic_results
  results['topic_results'] = {
    'lda': {
      'afrikaans': lda_af_topics,
      'english': lda_en_topics
    },
    'bertopic': {
      'afrikaans': bertopic_af.get_topic_info().to_dict(orient='records'),
      'english': bertopic_en.get_topic_info().to_dict(orient='records')
    }
  }
  
  # Store evaluation metrics
  results['evaluation_metrics'] = {
    'lda': {
      'afrikaans': {
        'umass': coherence_af['umass'],
        'npmi': coherence_af['npmi']
      },
      'english': {
        'umass': coherence_en['umass'],
        'npmi': coherence_en['npmi']
      }
    },
    'bertopic': {
      'afrikaans': {
        'umass': bertopic_coherence_af['umass'],
        'npmi': bertopic_coherence_af['npmi']
      },
      'english': {
        'umass': bertopic_coherence_en['umass'],
        'npmi': bertopic_coherence_en['npmi']
      }
    }
  }
  
  print("\n" + "="*60)
  print("AFRIKAANS ASR AND TOPIC MODELING - SUMMARY REPORT")
  print("="*60)
  
  # Topic Modeling Performance
  print("\nTOPIC MODELING COHERENCE")
  print("-" * 50)
  
  print("LDA Coherence:")
  print(f"    • Afrikaans: UMass={results['evaluation_metrics']['lda']['afrikaans']['umass']:.3f}, NPMI={results['evaluation_metrics']['lda']['afrikaans']['npmi']:.3f}")
  print(f"    • English:  UMass={results['evaluation_metrics']['lda']['english']['umass']:.3f}, NPMI={results['evaluation_metrics']['lda']['english']['npmi']:.3f}")
  print("BERTopic Coherence:")
  print(f"    • Afrikaans: UMass={results['evaluation_metrics']['bertopic']['afrikaans']['umass']:.3f}, NPMI={results['evaluation_metrics']['bertopic']['afrikaans']['npmi']:.3f}")
  print(f"    • English:  UMass={results['evaluation_metrics']['bertopic']['english']['umass']:.3f}, NPMI={results['evaluation_metrics']['bertopic']['english']['npmi']:.3f}")

  print("\n🎉 Pipeline execution completed successfully!")

  return results