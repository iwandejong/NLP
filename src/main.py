#!/usr/bin/env python3
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# * Modal Server Section
import modal
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

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
  "tqdm",
  "evaluate"
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

def plot_wer_npmi(wer_npmi_data: List[Dict], save_path: str = "wer_npmi_analysis.png"):
    """Plot WER vs NPMI for different models and topic modeling approaches"""
    if not wer_npmi_data:
        print("No WER-NPMI data available to plot")
        return
        
    plt.figure(figsize=(15, 10))
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame(wer_npmi_data)
    
    # Verify required columns exist
    required_columns = ['npmi_af_lda', 'npmi_en_lda', 'npmi_af_bertopic', 'npmi_en_bertopic', 'wer', 'model']
    if not all(col in df.columns for col in required_columns):
        print(f"Missing required columns. Available columns: {df.columns.tolist()}")
        return
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot Afrikaans LDA
    sns.scatterplot(data=df, x='npmi_af_lda', y='wer', hue='model', style='model', s=100, ax=ax1)
    ax1.set_title('Afrikaans LDA Coherence vs WER')
    ax1.set_xlabel('NPMI Coherence Score')
    ax1.set_ylabel('Word Error Rate (WER)')
    
    # Plot English LDA
    sns.scatterplot(data=df, x='npmi_en_lda', y='wer', hue='model', style='model', s=100, ax=ax2)
    ax2.set_title('English LDA Coherence vs WER')
    ax2.set_xlabel('NPMI Coherence Score')
    ax2.set_ylabel('Word Error Rate (WER)')
    
    # Plot Afrikaans BERTopic
    sns.scatterplot(data=df, x='npmi_af_bertopic', y='wer', hue='model', style='model', s=100, ax=ax3)
    ax3.set_title('Afrikaans BERTopic Coherence vs WER')
    ax3.set_xlabel('NPMI Coherence Score')
    ax3.set_ylabel('Word Error Rate (WER)')
    
    # Plot English BERTopic
    sns.scatterplot(data=df, x='npmi_en_bertopic', y='wer', hue='model', style='model', s=100, ax=ax4)
    ax4.set_title('English BERTopic Coherence vs WER')
    ax4.set_xlabel('NPMI Coherence Score')
    ax4.set_ylabel('Word Error Rate (WER)')
    
    # Add correlation coefficients
    for ax, col in [(ax1, 'npmi_af_lda'), (ax2, 'npmi_en_lda'), 
                    (ax3, 'npmi_af_bertopic'), (ax4, 'npmi_en_bertopic')]:
        correlation = df['wer'].corr(df[col])
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_translation_quality(translation_data: List[Dict], save_path: str = "translation_quality.png"):
    """Plot translation quality metrics"""
    plt.figure(figsize=(12, 6))
    
    # Create DataFrame
    df = pd.DataFrame(translation_data)
    
    # Create grouped bar plot
    df_melted = pd.melt(df, id_vars=['model'], 
                        value_vars=['wer', 'translation_wer'],
                        var_name='metric', value_name='score')
    
    sns.barplot(data=df_melted, x='model', y='score', hue='metric')
    
    plt.xlabel('ASR Model')
    plt.ylabel('Error Rate')
    plt.title('ASR and Translation Quality Comparison')
    plt.xticks(rotation=45)
    plt.legend(title='Metric', labels=['ASR WER', 'Translation WER'])
    
    # Save plot
    plt.savefig(save_path)
    plt.close()

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
  import json

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
  asr = ASR()
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
  print("Loading Common Voice dataset...")
  audio_data = loader.load_data()

  print("Transcribing audio with different models...")
  
  # Get reference texts
  reference_texts = [sample['text'] for sample in audio_data]
  
  # Transcribe with different models
  # whisper_transcripts = asr.transcribe_whisper(audio_data)
  # m4tv2_transcripts = asr.transcribe_m4tv2(audio_data)
  whisper_small_transcripts = asr.transcribe_whisper_small(audio_data)

  # Calculate WER for each model
  # whisper_wer = asr.calculate_wer(reference_texts, whisper_transcripts)
  # m4tv2_wer = asr.calculate_wer(reference_texts, m4tv2_transcripts)
  whisper_small_wer = asr.calculate_wer(reference_texts, whisper_small_transcripts)

  print("\nWord Error Rate (WER) Results:")
  # print(f"Whisper Large v3: {whisper_wer:.4f}")
  # print(f"Seamless M4T v2: {m4tv2_wer:.4f}")
  print(f"Whisper Small: {whisper_small_wer:.4f}")

  # Store results for each model
  model_results = {
    # 'whisper_large': {
    #   'transcripts': whisper_transcripts,
    #   'wer': whisper_wer
    # },
    # 'm4tv2': {
    #   'transcripts': m4tv2_transcripts,
    #   'wer': m4tv2_wer
    # },
    'whisper_small': {
      'transcripts': whisper_small_transcripts,
      'wer': whisper_small_wer
    }
  }

  # Process and analyze each model's results
  wer_npmi_data = []
  topic_results = []

  for model_name, model_data in model_results.items():
    print(f"\nProcessing {model_name}...")
    transcripts = model_data['transcripts']
    
    # Translate
    processed_translations = translate.translate_nllb(transcripts)
    reference_translations = translate.translate_nllb(reference_texts)
    
    # Preprocess
    processed_af = preprocess.preprocess_text(transcripts, language="af")
    processed_en = preprocess.preprocess_text(processed_translations, language="en")

    # Also preprocess the reference texts
    reference_af = preprocess.preprocess_text(reference_texts, language="af")
    reference_en = preprocess.preprocess_text(reference_translations, language="en")

    print("PROCESSED:", len(processed_af), processed_af[:5])  # Print first 5 processed Afrikaans texts
    print("PROCESSED:", len(processed_en), processed_en[:5])  # Print first 5 processed English texts
    print("REF:", len(reference_af), reference_af[:5])  # Print first 5 reference Afrikaans texts
    print("REF:", len(reference_en), reference_en[:5])  # Print first 5 reference English texts
    
    if len(processed_af) > 0 and len(processed_en) > 0:
      # Train topic models for Afrikaans
      lda_af, vectorizer_af = topic.train_lda(processed_af, n_topics=5)
      bertopic_af = topic.train_bertopic(processed_af)
      # Reference
      lda_af_reference, vectorizer_af_reference = topic.train_lda(reference_af, n_topics=5)
      bertopic_af_reference = topic.train_bertopic(reference_af)
      
      # Train topic models for English
      lda_en, vectorizer_en = topic.train_lda(processed_en, n_topics=5)
      bertopic_en = topic.train_bertopic(processed_en)
      # Reference
      lda_en_reference, vectorizer_en_reference = topic.train_lda(reference_en, n_topics=5)
      bertopic_en_reference = topic.train_bertopic(reference_en)
      
      # Calculate coherence scores for Afrikaans
      if lda_af is not None and vectorizer_af is not None:
        coherence_af_lda = topic.calculate_lda_topic_coherence(processed_af, lda_af, vectorizer_af)
        coherence_af_lda_reference = topic.calculate_lda_topic_coherence(reference_af, lda_af_reference, vectorizer_af_reference)
        print(f"LDA Coherence scores for Afrikaans ({model_name}): {coherence_af_lda}")
        print(f"LDA Coherence scores for Afrikaans (Reference): {coherence_af_lda_reference}")
      else:
        coherence_af_lda = {'umass': 0, 'npmi': 0}
        
      if bertopic_af is not None:
        coherence_af_bertopic = topic.calculate_bertopic_topic_coherence(processed_af, bertopic_af)
        coherence_af_bertopic_reference = topic.calculate_bertopic_topic_coherence(reference_af, bertopic_af_reference)
        print(f"BERTopic Coherence scores for Afrikaans ({model_name}): {coherence_af_bertopic}")
        print(f"BERTopic Coherence scores for Afrikaans (Reference): {coherence_af_bertopic_reference}")
      else:
        coherence_af_bertopic = {'umass': 0, 'npmi': 0}
      
      # Calculate coherence scores for English
      if lda_en is not None and vectorizer_en is not None:
        coherence_en_lda = topic.calculate_lda_topic_coherence(processed_en, lda_en, vectorizer_en)
        coherence_en_lda_reference = topic.calculate_lda_topic_coherence(reference_en, lda_en_reference, vectorizer_en_reference)
        print(f"LDA Coherence scores for English ({model_name}): {coherence_en_lda}")
        print(f"LDA Coherence scores for English (Reference): {coherence_en_lda_reference}")
      else:
        coherence_en_lda = {'umass': 0, 'npmi': 0}
        
      if bertopic_en is not None:
        coherence_en_bertopic = topic.calculate_bertopic_topic_coherence(processed_en, bertopic_en)
        coherence_en_bertopic_reference = topic.calculate_bertopic_topic_coherence(reference_en, bertopic_en_reference)
        print(f"BERTopic Coherence scores for English ({model_name}): {coherence_en_bertopic}")
        print(f"BERTopic Coherence scores for English (Reference): {coherence_en_bertopic_reference}")
      else:
        coherence_en_bertopic = {'umass': 0, 'npmi': 0}
      
      # Store WER and NPMI data
      wer_npmi_data.append({
        'model': model_name,
        'wer': model_data['wer'],
        'npmi_af_lda': coherence_af_lda['npmi'],
        'npmi_en_lda': coherence_en_lda['npmi'],
        'npmi_af_bertopic': coherence_af_bertopic['npmi'],
        'npmi_en_bertopic': coherence_en_bertopic['npmi'],
        'npmi_af_lda_reference': coherence_af_lda_reference['npmi'],
        'npmi_en_lda_reference': coherence_en_lda_reference['npmi'],
        'npmi_af_bertopic_reference': coherence_af_bertopic_reference['npmi'],
        'npmi_en_bertopic_reference': coherence_en_bertopic_reference['npmi']
      })
      
      # Store topic results
      topic_results.append({
        'model': model_name,
        'afrikaans': {
          'lda': coherence_af_lda,
          'bertopic': coherence_af_bertopic,
          'lda_reference': coherence_af_lda_reference,
          'bertopic_reference': coherence_af_bertopic_reference
        },
        'english': {
          'lda': coherence_en_lda,
          'bertopic': coherence_en_bertopic,
          'lda_reference': coherence_en_lda_reference,
          'bertopic_reference': coherence_en_bertopic_reference
        }
      })
    else:
      print(f"Not enough valid texts after preprocessing for {model_name}")
    
    # Store detailed results
    results['asr_results'][model_name] = {
      'transcripts': {
        'processed': transcripts[:5],
        'reference': reference_texts[:5]
      },
      'wer': model_data['wer'],
      'translations': {
        'processed': processed_translations[:5],
        'reference': reference_translations[:5]
      },
      'topic_metrics': {
        'afrikaans': {
          'lda': coherence_af_lda if 'coherence_af_lda' in locals() else {'umass': 0, 'npmi': 0},
          'bertopic': coherence_af_bertopic if 'coherence_af_bertopic' in locals() else {'umass': 0, 'npmi': 0},
          'lda_reference': coherence_af_lda_reference if 'coherence_af_lda_reference' in locals() else {'umass': 0, 'npmi': 0},
          'bertopic_reference': coherence_af_bertopic_reference if 'coherence_af_bertopic_reference' in locals() else {'umass': 0, 'npmi': 0}
        },
        'english': {
          'lda': coherence_en_lda if 'coherence_en_lda' in locals() else {'umass': 0, 'npmi': 0},
          'bertopic': coherence_en_bertopic if 'coherence_en_bertopic' in locals() else {'umass': 0, 'npmi': 0},
          'lda_reference': coherence_en_lda_reference if 'coherence_en_lda_reference' in locals() else {'umass': 0, 'npmi': 0},
          'bertopic_reference': coherence_en_bertopic_reference if 'coherence_en_bertopic_reference' in locals() else {'umass': 0, 'npmi': 0}
        }
      }
    }

  print(f"\nFinal wer_npmi_data: {wer_npmi_data}")
  
  print(json.dumps(results, indent=2, ensure_ascii=False))

  return results