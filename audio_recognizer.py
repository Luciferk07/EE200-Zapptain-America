import os
import librosa
import numpy as np
import pickle
from scipy.ndimage import maximum_filter
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class AudioRecognizer:
    def __init__(self, sr=22050, n_fft=2048, hop_length=512):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        # Parameters for peak extraction
        self.neighborhood_size = (15, 15)
        self.amp_threshold = 10.0 # Will dynamically calculate based on percentiles if needed
        # Parameters for hashing
        self.target_zone_time = (1, 15)  # look ahead 1 to 15 time frames
        self.target_zone_freq = (-50, 50) # look up/down 50 freq bins
        
    def get_spectrogram(self, file_path, duration=None, offset=0.0):
        """Returns the log-magnitude spectrogram of the audio file."""
        y, _ = librosa.load(file_path, sr=self.sr, duration=duration, offset=offset, mono=True)
        # Compute STFT
        S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        # Convert to dB
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        return S_db, y

    def extract_peaks(self, spectrogram, percentile=85):
        """Find local maxima in the spectrogram."""
        # Find local maxima using a maximum filter
        local_max = maximum_filter(spectrogram, size=self.neighborhood_size) == spectrogram
        
        # Calculate dynamic threshold based on non-zero magnitude to avoid noise floor
        threshold = np.percentile(spectrogram, percentile)
        
        # Boolean mask for peaks above threshold
        peak_mask = local_max & (spectrogram > threshold)
        
        # Get frequencies (rows) and times (cols) of the peaks
        frequencies, times = np.where(peak_mask)
        
        # Sort by time for easier hashing
        sort_idx = np.argsort(times)
        return frequencies[sort_idx], times[sort_idx]

    def generate_hashes(self, frequencies, times):
        """Generate (f1, f2, delta_t) hashes from the constellation of peaks."""
        hashes = []
        num_peaks = len(times)
        
        for i in range(num_peaks):
            f_anchor = frequencies[i]
            t_anchor = times[i]
            
            # Look ahead in the target zone
            for j in range(1, 20): # limit to nearest 20 peaks to save computation
                if i + j >= num_peaks:
                    break
                    
                f_target = frequencies[i + j]
                t_target = times[i + j]
                delta_t = t_target - t_anchor
                
                # Check if target is within the defined target zone
                if self.target_zone_time[0] <= delta_t <= self.target_zone_time[1]:
                    if self.target_zone_freq[0] <= (f_target - f_anchor) <= self.target_zone_freq[1]:
                        # Create hash tuple
                        hash_val = (int(f_anchor), int(f_target), int(delta_t))
                        hashes.append((hash_val, int(t_anchor)))
        return hashes

class SongDatabase:
    def __init__(self, db_path="c:\\Users\\karti\\OneDrive\\Desktop\\EE200 Project\\song_db.pkl"):
        self.db_path = db_path
        # hash_dict: {hash_val: [(song_id, anchor_time), ...]}
        self.hash_dict = defaultdict(list)
        self.song_names = {}
        self.song_images = {}
        self.next_song_id = 0
        
    def add_song(self, song_name, hashes, image_b64=None):
        """Add a song's hashes and optional image to the database."""
        song_id = self.next_song_id
        self.song_names[song_id] = song_name
        if image_b64:
            self.song_images[song_id] = image_b64
        self.next_song_id += 1
        
        for hash_val, anchor_time in hashes:
            self.hash_dict[hash_val].append((song_id, anchor_time))
            
    def save(self):
        """Serialize database to disk."""
        with open(self.db_path, 'wb') as f:
            pickle.dump({
                'hash_dict': self.hash_dict, 
                'song_names': self.song_names, 
                'song_images': self.song_images,
                'next_song_id': self.next_song_id
            }, f)
            
    def load(self):
        """Load database from disk."""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                data = pickle.load(f)
                self.hash_dict = data.get('hash_dict', defaultdict(list))
                self.song_names = data.get('song_names', {})
                self.song_images = data.get('song_images', {})
                self.next_song_id = data.get('next_song_id', 0)
            return True
        return False

    def match_hashes(self, query_hashes):
        """Match query hashes against the database and score based on offset histograms."""
        # matches per song: {song_id: {offset: count}}
        song_offsets = defaultdict(lambda: defaultdict(int))
        
        for q_hash, q_time in query_hashes:
            if q_hash in self.hash_dict:
                db_matches = self.hash_dict[q_hash]
                for song_id, db_time in db_matches:
                    offset = db_time - q_time
                    song_offsets[song_id][offset] += 1
                    
        # Find the best match
        best_song_id = None
        best_offset = None
        max_matches = 0
        
        results = []
        for song_id, offsets in song_offsets.items():
            if not offsets:
                continue
            # The score for a song is the highest peak in its offset histogram
            best_song_offset = max(offsets.items(), key=lambda x: x[1])
            score = best_song_offset[1]
            results.append({
                'song_id': song_id,
                'song_name': self.song_names[song_id],
                'best_offset': best_song_offset[0],
                'score': score,
                'histogram': offsets
            })
            
            if score > max_matches:
                max_matches = score
                best_song_id = song_id
                best_offset = best_song_offset[0]
                
        # Sort results by score descending
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
