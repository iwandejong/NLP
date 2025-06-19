# NLP Project: ASR, Translation, and Topic Modeling

This project evaluates automatic speech recognition (ASR) models, translation quality, and topic modeling coherence on Afrikaans and English audio data. It uses Common Voice datasets and podcast audio, applying state-of-the-art ASR and translation models, and analyzes topic coherence using LDA and BERTopic.

## Features
- Transcribes audio using Whisper and Seamless M4T models
- Translates transcripts with NLLB
- Evaluates ASR and translation quality (WER)
- Performs topic modeling (LDA, BERTopic) and coherence analysis (NPMI, UMass)
- Visualizes results with plots

## Structure
- `src/`: Main pipeline and modules for ASR, translation, topic modeling, and preprocessing
- `podcasts/`, `podcast_chunks/`: Audio data
- `cv-corpus-21.0-2025-03-14/`: Common Voice dataset
- `requirements.txt`: Python dependencies

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the main pipeline (locally or with Modal):
   ```bash
   python src/main.py
   ```