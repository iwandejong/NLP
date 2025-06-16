#!/usr/bin/env python3
"""
Enhancing Automatic Speech Recognition (ASR) and Topic Modelling for
Afrikaans using Direct Translation

This script implements the complete pipeline described in the research paper:
1. ASR using Whisper
2. Translation using M2M-100 and NLLB
3. Topic modeling using LDA and BERTopic
4. Evaluation and visualization

Requirements: transformers torch torchaudio datasets librosa jiwer scikit-learn gensim bertopic nltk matplotlib seaborn plotly sentence-transformers umap-learn hdbscan
"""
# Set environment variable for tokenizers
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
    "sentencepiece"
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
    timeout=int(1 * HOURS),
    volumes={VOL_MOUNT_PATH: volume}, 
)
def main():
    import os
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from typing import List, Dict, Tuple, Optional
    import json

    # Audio processing
    import librosa
    import torch
    import torchaudio
    from transformers import (
        WhisperProcessor, WhisperForConditionalGeneration,
        M2M100ForConditionalGeneration, M2M100Tokenizer,
        pipeline as hf_pipeline
    )

    # Datasets
    from datasets import load_dataset, Audio

    # Evaluation
    from jiwer import wer

    # Topic modeling
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    from gensim.models.coherencemodel import CoherenceModel
    from gensim.corpora import Dictionary
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    # NLP preprocessing
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import re

    # Visualization
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px

    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    from huggingface_hub import login
    login("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")

    class AfrikaansASRPipeline:
        """
        Complete pipeline for Afrikaans ASR and Topic Modeling research
        """
        
        def __init__(self, device: str = "auto"):
            """Initialize the pipeline with models and configurations"""
            
            # Set device
            if device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
                
            print(f"Using device: {self.device}")
            
            # Initialize models (will be loaded lazily)
            self.whisper_model = None
            self.whisper_processor = None
            self.m2m_model = None
            self.m2m_tokenizer = None
            self.nllb_translator = None
            
            # Initialize topic models
            self.lda_model = None
            self.bertopic_model = None
            
            # Results storage
            self.results = {
                'asr_results': {},
                'translation_results': {},
                'topic_results': {},
                'evaluation_metrics': {}
            }
            
            # Afrikaans stopwords (extended list)
            self.afrikaans_stopwords = set([
                'die', 'van', 'en', 'in', 'is', 'het', 'dat', 'met', 'op', 
                'vir', 'te', 'by', 'as', 'aan', 'was', 'sy', 'nie', 'hy',
                'dit', 'haar', 'wat', 'word', 'sal', 'kan', 'ook', 'maar',
                'so', 'nog', 'tot', 'na', 'om', 'oor', 'uit', 'al', 'daar'
            ])
        
        def load_whisper_model(self, model_size: str = "large-v3"):
            """Load Whisper model"""
            print(f"Loading Whisper {model_size} model...")
            model_name = f"openai/whisper-{model_size}"
            self.whisper_processor = WhisperProcessor.from_pretrained(model_name)
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name)
            self.whisper_model.to(self.device)
            
        def load_translation_models(self):
            """Load translation models"""
            print("Loading M2M-100 model...")
            self.m2m_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
            self.m2m_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
            self.m2m_model.to(self.device)
            
            print("Loading NLLB model...")
            self.nllb_translator = hf_pipeline(
                "translation", 
                model="facebook/nllb-200-distilled-600M",
                device=0 if self.device == "cuda" else -1
            )
        
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
            for fname in os.listdir(chunk_dir):
                if fname.endswith(".mp3"):
                    path = os.path.join(chunk_dir, fname)
                    audio, sr = librosa.load(path, sr=16000)
                    audio_data.append({'audio': audio, 'sampling_rate': 16000, 'text': ""})
            print(f"Loaded {len(audio_data)} podcast chunks from {chunk_dir}")
            return audio_data

        def transcribe_whisper(self, audio_data: List[Dict]) -> List[str]:
            """Transcribe audio using Whisper"""
            if self.whisper_model is None:
                self.load_whisper_model()
            
            transcriptions = []
            print("Transcribing with Whisper...")
            
            for i, sample in enumerate(audio_data):
                try:
                    # Process audio with attention mask
                    input_features = self.whisper_processor(
                        sample['audio'], 
                        sampling_rate=sample['sampling_rate'],
                        return_tensors="pt"
                    ).input_features.to(self.device)
                    
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
                    
                    if (i + 1) % 10 == 0:
                        print(f"Processed {i + 1}/{len(audio_data)} samples")
                        
                except Exception as e:
                    print(f"Error transcribing sample {i}: {e}")
                    transcriptions.append("")
            
            return transcriptions
        
        def translate_m2m(self, texts: List[str]) -> List[str]:
            """Translate texts using M2M-100"""
            if self.m2m_model is None:
                self.load_translation_models()
            
            translations = []
            print("Translating with M2M-100...")
            
            self.m2m_tokenizer.src_lang = "af"
            
            for i, text in enumerate(texts):
                if not text.strip():
                    translations.append("")
                    continue
                    
                try:
                    encoded = self.m2m_tokenizer(text, return_tensors="pt").to(self.device)
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
            """Translate texts using NLLB"""
            if self.nllb_translator is None:
                self.load_translation_models()
            
            translations = []
            print("Translating with NLLB...")
            
            for i, text in enumerate(texts):
                if not text.strip():
                    translations.append("")
                    continue
                    
                try:
                    result = self.nllb_translator(
                        text, 
                        src_lang="afr_Latn", 
                        tgt_lang="eng_Latn"
                    )
                    # The pipeline returns a list of dictionaries, we need the first one
                    translations.append(result[0]['translation_text'])
                    
                    if (i + 1) % 10 == 0:
                        print(f"Translated {i + 1}/{len(texts)} texts")
                        
                except Exception as e:
                    print(f"Error translating text {i}: {e}")
                    translations.append("")
            
            return translations
        
        def preprocess_text(self, texts: List[str], language: str = "af") -> List[str]:
            """Preprocess texts for topic modeling with enhanced cleaning"""
            processed_texts = []
            
            # Select stopwords based on language
            if language == "af":
                stop_words = self.afrikaans_stopwords
            else:
                try:
                    stop_words = set(stopwords.words('english'))
                except:
                    stop_words = set()
            
            for text in texts:
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
        
        def train_lda(self, texts: List[str], n_topics: int = 5) -> Tuple[Optional[LatentDirichletAllocation], Optional[CountVectorizer]]:
            """Train LDA topic model with improved parameters"""
            print(f"Training LDA with {n_topics} topics...")
            
            # Remove empty texts and ensure we have valid strings
            texts = [text for text in texts if text and isinstance(text, str) and text.strip()]
            
            if len(texts) < n_topics:
                print(f"Not enough valid texts for {n_topics} topics. Need at least {n_topics} texts.")
                return None, None
            
            # Vectorize texts with improved parameters
            vectorizer = CountVectorizer(
                min_df=1,          # Increased from 1
                max_df=0.9,        # Decreased from 0.8
                ngram_range=(1, 2) # Increased from (1, 2)
            )
            
            try:
                doc_term_matrix = vectorizer.fit_transform(texts)
                
                # Check if we have enough features
                if doc_term_matrix.shape[1] < n_topics:
                    print(f"Not enough features for {n_topics} topics. Need at least {n_topics} features.")
                    return None, None
                
                # Train LDA with improved parameters
                lda = LatentDirichletAllocation(
                    n_components=n_topics,
                    random_state=42,
                    max_iter=100,
                    learning_method='online',
                    learning_offset=50.0,
                    batch_size=32,
                    n_jobs=-1
                )
                lda.fit(doc_term_matrix)
                
                return lda, vectorizer
            except Exception as e:
                print(f"Error training LDA: {e}")
                return None, None
        
        def train_bertopic(self, texts: List[str]) -> Optional[BERTopic]:
            """Train BERTopic model"""
            print("Training BERTopic...")
            
            # Remove empty texts
            texts = [text for text in texts if text.strip()]
            print(f"BERTopic: {len(texts)} valid documents to train on")

            if len(texts) == 0:
                print("No valid texts for BERTopic training")
                return None
            
            # Initialize BERTopic with smaller embedding model for efficiency
            topic_model = BERTopic(
                embedding_model="all-MiniLM-L6-v2",
                min_topic_size=2,
                verbose=True
            )
            
            try:
                topics, probs = topic_model.fit_transform(texts)
                return topic_model
            except Exception as e:
                print(f"Error training BERTopic: {e}")
                return None
        
        def calculate_wer(self, reference_texts: List[str], hypothesis_texts: List[str]) -> float:
            """Calculate Word Error Rate"""
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
        
        def calculate_topic_coherence(self, texts: List[str], lda_model, vectorizer, topic_model_type: str = "lda"):
            """Calculate topic coherence using UMass and NPMI with improved parameters"""
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
                    top_words_idx = topic.argsort()[:-15-1:-1]  # Increased from 10 to 15
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
                    processes=-1  # Use all available cores
                )
                umass_score = cm_umass.get_coherence()
                
                # Calculate NPMI coherence with improved parameters
                cm_npmi = CoherenceModel(
                    topics=topics,
                    texts=tokenized_texts,
                    dictionary=dictionary,
                    coherence='c_npmi',
                    processes=-1  # Use all available cores
                )
                npmi_score = cm_npmi.get_coherence()
                
                return {'umass': umass_score, 'npmi': npmi_score}
            except Exception as e:
                print(f"Error calculating coherence: {e}")
                return {'umass': 0, 'npmi': 0}
        
        def run_full_pipeline(self):
            """Run the complete ASR and topic modeling pipeline"""
            print("=" * 60)
            print("AFRIKAANS ASR AND TOPIC MODELING PIPELINE")
            print("=" * 60)
            
            # Uncomment the one you want to use:

            # For Common Voice:
            # audio_data = self.load_data()
            # reference_texts = [sample['text'] for sample in audio_data]

            # For Podcast Chunks:
            audio_data = self.load_podcast_chunks(chunk_dir="/root/podcast_chunks")
            reference_texts = [""] * len(audio_data)  # No reference transcripts, so WER will be skipped/meaningless
            
            # 2. ASR Transcription
            print("\n" + "=" * 40)
            print("STEP 1: ASR TRANSCRIPTION")
            print("=" * 40)
            
            # Use actual ASR models for transcription
            whisper_transcripts = self.transcribe_whisper(audio_data)
            
            # Store ASR results
            self.results['asr_results'] = {
                'whisper': whisper_transcripts,
                'reference': reference_texts
            }
            
            # Calculate WER
            whisper_wer = self.calculate_wer(reference_texts, whisper_transcripts)
            
            print(f"Whisper WER: {whisper_wer:.3f}")
            
            # 3. Translation
            print("\n" + "=" * 40)
            print("STEP 2: TRANSLATION")
            print("=" * 40)
            
            # Use actual translation models
            m2m_translations = self.translate_m2m(whisper_transcripts)
            nllb_translations = self.translate_nllb(whisper_transcripts)
            
            self.results['translation_results'] = {
                'm2m': m2m_translations,
                'nllb': nllb_translations
            }
            
            # 4. Topic Modeling
            print("\n" + "=" * 40)
            print("STEP 3: TOPIC MODELING")
            print("=" * 40)
            
            # Preprocess texts
            processed_af = self.preprocess_text(whisper_transcripts, language="af")
            processed_en = self.preprocess_text(nllb_translations, language="en")
            # Diagnostics: Check average document length
            avg_len_af = sum(len(doc.split()) for doc in processed_af) / len(processed_af)
            avg_len_en = sum(len(doc.split()) for doc in processed_en) / len(processed_en)
            print(f"Afrikaans: Average words per doc: {avg_len_af:.2f}")
            print(f"English: Average words per doc: {avg_len_en:.2f}")

            # Train models
            lda_af, vectorizer_af = self.train_lda(processed_af, n_topics=2)
            lda_en, vectorizer_en = self.train_lda(processed_en, n_topics=2)
            
            # Calculate coherence scores
            coherence_af = self.calculate_topic_coherence(processed_af, lda_af, vectorizer_af)
            coherence_en = self.calculate_topic_coherence(processed_en, lda_en, vectorizer_en)
            
            # Store results
            self.results['topic_results'] = {
                'afrikaans': {'coherence': coherence_af, 'model': lda_af},
                'english': {'coherence': coherence_en, 'model': lda_en}
            }
            
            # Store evaluation metrics
            self.results['evaluation_metrics'] = {
                'wer': {'whisper': whisper_wer},
                'coherence': {
                    'afrikaans': coherence_af,
                    'english': coherence_en
                }
            }
            
            print("\nPipeline completed successfully!")
            return self.results
        
        def generate_visualizations(self):
            """Generate all visualizations from the paper"""
            print("\n" + "=" * 40)
            print("GENERATING VISUALIZATIONS")
            print("=" * 40)
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create output directory if it doesn't exist
            os.makedirs('/vol/figures', exist_ok=True)
            
            # Figure 2: ASR WER Comparison
            self._plot_asr_comparison()
            
            # Figure 3: Topic Coherence Comparison
            self._plot_topic_coherence()
            
            # Figure 4: BERTopic Visualization (simulated)
            self._plot_bertopic_visualization()
            
            # Figure 5: LDA Visualization (simulated)
            self._plot_lda_visualization()
            
            # Figure 6: WER vs Topic Coherence
            self._plot_wer_vs_coherence()
            
            print("All visualizations generated!")
        
        def _plot_asr_comparison(self):
            """Plot ASR WER comparison (Figure 2)"""
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            models = ['Whisper']
            wer_scores = [
                self.results['evaluation_metrics']['wer']['whisper']
            ]
            
            bars = ax.bar(models, wer_scores, color=['#1f77b4'], alpha=0.8)
            ax.set_ylabel('Word Error Rate (WER)')
            ax.set_title('ASR Performance on Afrikaans Speech')
            ax.set_ylim(0, max(wer_scores) * 1.2)
            
            # Add value labels on bars
            for bar, score in zip(bars, wer_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{score:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('/vol/figures/figure2_asr_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved Figure 2: ASR Performance")
        
        def _plot_topic_coherence(self):
            """Plot topic coherence comparison (Figure 3)"""
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            conditions = ['Afrikaans']
            umass_scores = [
                self.results['evaluation_metrics']['coherence']['afrikaans']['umass']
            ]
            npmi_scores = [
                self.results['evaluation_metrics']['coherence']['afrikaans']['npmi']
            ]
            
            x = np.arange(len(conditions))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, umass_scores, width, label='UMass', color='green', alpha=0.7)
            bars2 = ax.bar(x + width/2, npmi_scores, width, label='NPMI', color='red', alpha=0.7)
            
            ax.set_ylabel('Coherence Score')
            ax.set_title('Topic Modeling Coherence: Afrikaans')
            ax.set_xticks(x)
            ax.set_xticklabels(conditions)
            ax.legend()
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                          f'{height:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('/vol/figures/figure3_topic_coherence.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved Figure 3: Topic Coherence")
        
        def _plot_bertopic_visualization(self):
            """Create BERTopic visualization (Figure 4)"""
            if not hasattr(self, 'bertopic_model') or self.bertopic_model is None:
                print("BERTopic model not available for visualization")
                return

            # Get topic embeddings and visualize
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            # Get topic embeddings from BERTopic model
            topic_embeddings = self.bertopic_model.topic_embeddings_
            topic_labels = self.bertopic_model.topic_labels_
            
            # Plot topics
            scatter = ax.scatter(topic_embeddings[:, 0], topic_embeddings[:, 1], 
                               c=range(len(topic_embeddings)), cmap='viridis',
                               s=100, alpha=0.6)
            
            # Add topic labels
            for i, label in enumerate(topic_labels):
                ax.annotate(f'Topic {label}', 
                          (topic_embeddings[i, 0], topic_embeddings[i, 1]),
                          xytext=(5, 5), textcoords='offset points')
            
            ax.set_title('BERTopic Visualization - Afrikaans')
            ax.set_xlabel('UMAP Dimension 1')
            ax.set_ylabel('UMAP Dimension 2')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('/vol/figures/figure4_bertopic_visualization.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved Figure 4: BERTopic Visualization")
        
        def _plot_lda_visualization(self):
            """Create LDA visualization (Figure 5)"""
            if not hasattr(self, 'lda_model') or self.lda_model is None:
                print("LDA model not available for visualization")
                return

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            # Get topic distributions from LDA model
            topic_distributions = self.lda_model.components_
            n_topics = topic_distributions.shape[0]
            
            # Calculate average topic distribution across documents
            avg_topic_dist = np.mean(topic_distributions, axis=1)
            avg_topic_dist = avg_topic_dist / np.sum(avg_topic_dist)  # Normalize
            
            # Create pie chart
            topics = [f'Topic {i+1}' for i in range(n_topics)]
            colors = plt.cm.viridis(np.linspace(0, 1, n_topics))
            
            wedges, texts, autotexts = ax.pie(avg_topic_dist, labels=topics, colors=colors,
                                            autopct='%1.1f%%', startangle=90)
            ax.set_title('LDA Topic Distribution - Afrikaans')
            
            plt.tight_layout()
            plt.savefig('/vol/figures/figure5_lda_visualization.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved Figure 5: LDA Visualization")
        
        def _plot_wer_vs_coherence(self):
            """Plot WER vs Topic Coherence scatter plot (Figure 6)"""
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Get actual WER and coherence values from results
            wer_values = [self.results['evaluation_metrics']['wer']['whisper']]
            npmi_values = [self.results['evaluation_metrics']['coherence']['afrikaans']['npmi']]
            
            # Create scatter plot
            scatter = ax.scatter(wer_values, npmi_values, c=wer_values, cmap='viridis',
                              s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Customize plot
            ax.set_xlabel('Word Error Rate (WER)')
            ax.set_ylabel('Topic Coherence (NPMI)')
            ax.set_title('Relationship between ASR Quality and Topic Modeling Performance')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('WER', rotation=270, labelpad=15)
            
            plt.tight_layout()
            plt.savefig('/vol/figures/figure6_wer_vs_coherence.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved Figure 6: WER vs Coherence")
        
        def print_summary_report(self):
            """Print a comprehensive summary report"""
            print("\n" + "="*60)
            print("AFRIKAANS ASR AND TOPIC MODELING - SUMMARY REPORT")
            print("="*60)
            
            # ASR Performance
            print("\n1. AUTOMATIC SPEECH RECOGNITION PERFORMANCE")
            print("-" * 50)
            metrics = self.results['evaluation_metrics']
            print(f"Whisper WER:  {metrics['wer']['whisper']:.3f}")
            
            # Topic Modeling Performance
            print("\n2. TOPIC MODELING COHERENCE")
            print("-" * 50)
            af_coherence = metrics['coherence']['afrikaans']
            en_coherence = metrics['coherence']['english']
            
            print("Afrikaans (Direct):")
            print(f"  UMass: {af_coherence['umass']:.3f}")
            print(f"  NPMI:  {af_coherence['npmi']:.3f}")
            
            print("\nEnglish (Translated):")
            print(f"  UMass: {en_coherence['umass']:.3f}")
            print(f"  NPMI:  {en_coherence['npmi']:.3f}")
            
            # Research Questions Answers
            print("\n3. RESEARCH QUESTIONS ANSWERS")
            print("-" * 50)
            
            print("Q1: How well do existing ASR models transcribe Afrikaans speech?")
            print(f"    • Whisper achieves {metrics['wer']['whisper']:.1%} WER")
            print(f"    • Performance is {'good' if metrics['wer']['whisper'] < 0.3 else 'moderate'} for low-resource language")
            
            print("\nQ2: How does topic modeling performance compare between original and translated texts?")
            print(f"    • Afrikaans NPMI: {af_coherence['npmi']:.3f}")
            print(f"    • English NPMI:  {en_coherence['npmi']:.3f}")
            print(f"    • {'Better' if af_coherence['npmi'] > en_coherence['npmi'] else 'Worse'} coherence in original language")
            
            print("\n4. RECOMMENDATIONS")
            print("-" * 50)
            print("• For production ASR systems: Use Whisper")
            print("• Future work: Collect more Afrikaans training data")
            print("• Consider domain-specific fine-tuning for better performance")
            print("• Topic modeling works better on original language texts")
            
            print("\n" + "="*60)
        
        def save_results(self, filename: str = "afrikaans_asr_results.json"):
            """Save results to JSON file"""
            # Prepare results for JSON serialization
            json_results = {
                'evaluation_metrics': self.results['evaluation_metrics'],
                'summary': {
                    'total_samples': len(self.results['asr_results']['reference']),
                    'best_asr_model': 'Whisper'
                }
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            
            print(f"Results saved to {filename}")

    """Main function to run the complete pipeline"""
    print("Initializing Afrikaans ASR and Topic Modeling Pipeline...")
    
    # Initialize pipeline
    pipeline = AfrikaansASRPipeline()
    
    # Run the complete pipeline
    results = pipeline.run_full_pipeline()
    
    # Generate visualizations
    pipeline.generate_visualizations()
    
    # Print summary report
    pipeline.print_summary_report()
    
    # Save results
    pipeline.save_results()
    
    print("\n🎉 Pipeline execution completed successfully!")
    print("Check the generated PNG files for visualizations.")

    return results