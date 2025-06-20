#!/usr/bin/env python3
def main():
  import os
  os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
  whisper_transcripts = asr.transcribe_whisper(audio_data)
  m4tv2_transcripts = asr.transcribe_m4tv2(audio_data)
  whisper_small_transcripts = asr.transcribe_whisper_small(audio_data)

  # Calculate WER for each model
  whisper_wer = asr.calculate_wer(reference_texts, whisper_transcripts)
  m4tv2_wer = asr.calculate_wer(reference_texts, m4tv2_transcripts)
  whisper_small_wer = asr.calculate_wer(reference_texts, whisper_small_transcripts)

  print("\nWord Error Rate (WER) Results:")
  print(f"Whisper Large v3: {whisper_wer:.4f}")
  print(f"Seamless M4T v2: {m4tv2_wer:.4f}")
  print(f"Whisper Small: {whisper_small_wer:.4f}")

  # Store results for each model
  model_results = {
    'whisper_large': {
      'transcripts': whisper_transcripts,
      'wer': whisper_wer
    },
    'm4tv2': {
      'transcripts': m4tv2_transcripts,
      'wer': m4tv2_wer
    },
    'whisper_small': {
      'transcripts': whisper_small_transcripts,
      'wer': whisper_small_wer
    }
  }

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

    print("PROCESSED:", len(processed_af), processed_af[:5])
    print("PROCESSED:", len(processed_en), processed_en[:5])
    print("REF:", len(reference_af), reference_af[:5])
    print("REF:", len(reference_en), reference_en[:5])
    
    if len(processed_af) > 0 and len(processed_en) > 0:
      # Find optimal number of topics
      optimal_topics_af = topic.optimal_topics(processed_af)
      optimal_topics_af_ref = topic.optimal_topics(reference_af)
      optimal_topics_en = topic.optimal_topics(processed_en)
      optimal_topics_en_ref = topic.optimal_topics(reference_en)

      print(f"Optimal topics for Afrikaans ({model_name}): {optimal_topics_af}")
      print(f"Optimal topics for Afrikaans (Reference): {optimal_topics_af_ref}")
      print(f"Optimal topics for English ({model_name}): {optimal_topics_en}")
      print(f"Optimal topics for English (Reference): {optimal_topics_en_ref}")

      # Train topic models for Afrikaans
      lda_af, vectorizer_af = topic.train_lda(processed_af, n_topics=optimal_topics_af)
      bertopic_af = topic.train_bertopic(processed_af)
      # Reference
      lda_af_reference, vectorizer_af_reference = topic.train_lda(reference_af, n_topics=optimal_topics_af_ref)
      bertopic_af_reference = topic.train_bertopic(reference_af)
      
      # Train topic models for English
      lda_en, vectorizer_en = topic.train_lda(processed_en, n_topics=optimal_topics_en)
      bertopic_en = topic.train_bertopic(processed_en)
      # Reference
      lda_en_reference, vectorizer_en_reference = topic.train_lda(reference_en, n_topics=optimal_topics_en_ref)
      bertopic_en_reference = topic.train_bertopic(reference_en)

      print("== Afrikaans ==")
      print("\nLDA (main):")
      topic.print_lda_topics(lda_af, vectorizer_af)
      print("\nLDA (reference):")
      topic.print_lda_topics(lda_af_reference, vectorizer_af_reference)
      print("\nBERTopic (main):")
      topic.print_bertopic_topics(bertopic_af)
      print("\nBERTopic (reference):")
      topic.print_bertopic_topics(bertopic_af_reference)

      print("\n== English ==")
      print("\nLDA (main):")
      topic.print_lda_topics(lda_en, vectorizer_en)
      print("\nLDA (reference):")
      topic.print_lda_topics(lda_en_reference, vectorizer_en_reference)
      print("\nBERTopic (main):")
      topic.print_bertopic_topics(bertopic_en)
      print("\nBERTopic (reference):")
      topic.print_bertopic_topics(bertopic_en_reference)

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

if __name__ == "__main__":
  main()