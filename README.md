# Zapptain America - Audio Fingerprinting System

This repository contains the final submission for the **EE200 (Signals, Systems, and Networks) Course Project** at IIT Kanpur (Summer 2026), instructed by Dr. Tushar Sandhan.

## Project Overview

This project implements an industry-standard Acoustic Fingerprinting system (similar to Shazam) capable of identifying highly corrupted, short audio samples against a database of 50 full-length tracks with near-perfect accuracy and constant-time $O(1)$ lookup complexity.

The system is deployed as an interactive, fully-featured web application using **Streamlit**.

### Key Features
1. **Library View**: Browse the 50-song database and view pre-computed Acoustic Fingerprints (Constellation Maps and Spectrograms).
2. **Generate & Auto-Identify**: Test the algorithm against pre-generated 15-second samples or upload your own degraded/corrupted audio files. 
3. **Batch Benchmarking**: Run automated internal benchmarks across the entire database to evaluate hash-matching accuracy and true-positive rate.
4. **Technical Deep-Dive**: In-app documentation detailing the mathematical foundation (STFT, peak extraction, anchor-target hashing, and offset histograms).

## Running the Application Locally

1. Ensure you have Python 3 installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit web application:
   ```bash
   streamlit run app.py
   ```

## Cloud Deployment Architecture

To comply with GitHub/Streamlit Cloud storage limitations (100MB max per file/repo), the raw 394MB `.mp3` database is excluded from this repository. Instead, the application intelligently falls back to:
- **`library_images/`**: Pre-computed spectrogram and constellation map visualisations for all 50 songs.
- **`sample_audio/`**: Pre-computed 15-second `.wav` slices for dynamic testing.
- **`song_db.pkl`**: The serialised hash-map database containing over 3 million unique acoustic hashes.

Because of this hybrid architecture, the app runs flawlessly on Streamlit Cloud while maintaining full algorithmic integrity.

## Author
- Kartikeya (EE200, Summer 2026)
