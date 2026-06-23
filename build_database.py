import os
import glob
from audio_recognizer import AudioRecognizer, SongDatabase

import matplotlib.pyplot as plt
import base64
from io import BytesIO

def get_constellation_b64(times, freqs):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(3, 1.5), dpi=80)
    ax.scatter(times, freqs, c='#00ff9d', s=0.5, alpha=0.8)
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def build_db(db_dir, output_pkl="c:\\Users\\karti\\OneDrive\\Desktop\\EE200 Project\\song_db.pkl"):
    print("Initializing Audio Recognizer and Database...")
    recognizer = AudioRecognizer(sr=11025) # Lower sample rate for faster processing
    db = SongDatabase(db_path=output_pkl)
    
    mp3_files = glob.glob(os.path.join(db_dir, "*.mp3"))
    print(f"Found {len(mp3_files)} MP3 files. Starting indexing...")
    
    for i, file_path in enumerate(mp3_files):
        song_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"[{i+1}/{len(mp3_files)}] Processing: {song_name}")
        
        try:
            # Index the entire song to allow matching any clip
            spectrogram, _ = recognizer.get_spectrogram(file_path)
            
            # Extract peaks
            freqs, times = recognizer.extract_peaks(spectrogram, percentile=90)
            
            # Generate hashes
            hashes = recognizer.generate_hashes(freqs, times)
            
            # Generate constellation thumbnail
            img_b64 = get_constellation_b64(times, freqs)
            
            # Add to database
            db.add_song(song_name, hashes, image_b64=img_b64)
            print(f"   -> Added {len(hashes)} hashes.")
        except Exception as e:
            print(f"   -> Error processing {song_name}: {e}")
            
    print(f"Saving database to {output_pkl}...")
    db.save()
    print("Done! Database built successfully.")

if __name__ == "__main__":
    q3_db_dir = r"c:\Users\karti\OneDrive\Desktop\EE200 Project\EE200_course_project_data_2026\Q3_database"
    build_db(q3_db_dir)
