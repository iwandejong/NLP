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
1. Set up a virtual environment (recommended)
  - macOS / Linux:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
  - Windows (Command Prompt):
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
  - Windows (PowerShell):
    ```bash
    python -m venv venv
    venv\Scripts\Activate.ps1
    ```
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main pipeline:
   ```bash
   python3 -m src.main-non-modal # local (may require SentencePiece to be installed in order for M2M-100 to work)
   python3 -m src.main # on Modal [recommended!] (may need to create an account)
   ```