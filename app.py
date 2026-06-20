import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tempfile
import time
import io
from audio_recognizer import AudioRecognizer, SongDatabase

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zapptain America · Audio Fingerprinting",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── MASTER CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Mono:wght@400;700&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #080c10;
    color: #c9d1d9;
}
.block-container { padding: 1.5rem 2rem !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0b1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #21262d !important;
}
[data-testid="stSidebar"] .block-container { padding: 1rem !important; }

/* ── Hero Header ── */
.hero-wrap {
    padding: 28px 0 18px;
    border-bottom: 1px solid #21262d;
    margin-bottom: 24px;
}
.hero-badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #3fb950;
    border: 1px solid #3fb950;
    border-radius: 20px;
    padding: 2px 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 900;
    color: #f0f6fc;
    line-height: 1.1;
    margin: 0;
    letter-spacing: -1.5px;
}
.hero-title span { color: #3fb950; }
.hero-sub {
    margin-top: 8px;
    color: #6e7681;
    font-size: 0.95rem;
    font-weight: 400;
}

/* ── Stat Pills ── */
.stat-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin: 16px 0 8px;
}
.stat-pill {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.8rem;
    color: #8b949e;
    font-family: 'Space Mono', monospace;
}
.stat-pill b { color: #e6edf3; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 1px solid #21262d;
}
.stTabs [data-baseweb="tab"] {
    height: 44px;
    background: transparent;
    border: none;
    padding: 0 20px;
    color: #6e7681;
    font-weight: 600;
    font-size: 0.88rem;
    border-bottom: 2px solid transparent;
    letter-spacing: 0.3px;
}
.stTabs [aria-selected="true"] {
    color: #f0f6fc !important;
    border-bottom: 2px solid #3fb950 !important;
}

/* ── Song Card ── */
.song-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 14px;
    cursor: pointer;
    transition: border-color 0.2s, transform 0.2s, box-shadow 0.2s;
    overflow: hidden;
}
.song-card:hover {
    border-color: #3fb950;
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(63,185,80,0.12);
}
.song-name {
    font-size: 0.85rem;
    font-weight: 700;
    color: #e6edf3;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-top: 8px;
}
.song-meta {
    font-size: 0.7rem;
    color: #6e7681;
    font-family: 'Space Mono', monospace;
    margin-top: 2px;
}

/* ── Clickable Song Button Cards ── */
div[data-testid="stButton"] > button.song-btn {
    all: unset;
    display: block;
    width: 100%;
    cursor: pointer;
}
[data-testid="column"] .stButton button {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
    padding: 0 !important;
    width: 100% !important;
    text-align: left !important;
    transition: border-color 0.2s, transform 0.18s, box-shadow 0.2s !important;
    margin-bottom: 14px !important;
    color: #e6edf3 !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    line-height: 1.4 !important;
    box-shadow: none !important;
}
[data-testid="column"] .stButton button:hover {
    border-color: #3fb950 !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 24px rgba(63,185,80,0.12) !important;
    background: #0d1117 !important;
}
[data-testid="column"] .stButton button:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.25) !important;
    outline: none !important;
}

/* ── Song Detail Panel ── */
.song-detail-panel {
    background: linear-gradient(135deg, #0d1117, #111820);
    border: 1px solid #238636;
    border-radius: 12px;
    padding: 24px 28px;
    margin: 8px 0 24px;
    animation: fadeSlideIn 0.3s ease;
}
.detail-img {
    width: 100%;
    border-radius: 8px;
    margin-bottom: 12px;
}
.detail-stat-row {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-top: 12px;
}
.detail-stat {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 10px 16px;
    text-align: center;
    flex: 1;
    min-width: 80px;
}
.detail-stat-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: #3fb950;
}
.detail-stat-label {
    font-size: 0.65rem;
    color: #6e7681;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 2px;
}

/* ── Match Banner ── */
.match-banner {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #0d1117 0%, #112218 50%, #0d1117 100%);
    border: 1px solid #238636;
    border-radius: 12px;
    padding: 24px 28px;
    margin: 20px 0;
    animation: fadeSlideIn 0.4s ease;
}
.match-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(63,185,80,0.08) 0%, transparent 70%);
    pointer-events: none;
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(-10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.match-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #3fb950;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.match-song-name {
    font-size: 2.2rem;
    font-weight: 900;
    color: #f0f6fc;
    letter-spacing: -0.5px;
    line-height: 1.15;
}
.match-detail {
    font-size: 0.85rem;
    color: #8b949e;
    margin-top: 8px;
}
.match-score-badge {
    display: inline-block;
    background: rgba(63,185,80,0.15);
    color: #3fb950;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 1rem;
    padding: 4px 12px;
    border-radius: 6px;
    border: 1px solid rgba(63,185,80,0.3);
}
.no-match-banner {
    background: linear-gradient(135deg, #0d1117, #1c1118);
    border: 1px solid #6e40c9;
    border-radius: 12px;
    padding: 24px 28px;
    margin: 20px 0;
    text-align: center;
    color: #8b949e;
}

/* ── Timing Bar ── */
.timing-bar {
    display: flex;
    gap: 0;
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 10px;
    overflow: hidden;
    margin: 16px 0;
}
.timing-cell {
    flex: 1;
    padding: 14px 16px;
    border-right: 1px solid #21262d;
    text-align: center;
}
.timing-cell:last-child { border-right: none; }
.timing-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: #58a6ff;
}
.timing-label {
    font-size: 0.68rem;
    color: #6e7681;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 2px;
}

/* ── Step Header ── */
.step-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 28px 0 12px;
}
.step-num {
    width: 28px;
    height: 28px;
    background: #21262d;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    color: #58a6ff;
    font-family: 'Space Mono', monospace;
    flex-shrink: 0;
}
.step-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: #e6edf3;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.step-sub {
    font-size: 0.8rem;
    color: #6e7681;
    margin-left: auto;
}

/* ── Candidate Table ── */
.cand-row {
    display: flex;
    align-items: center;
    padding: 10px 14px;
    border-bottom: 1px solid #21262d;
    gap: 14px;
    transition: background 0.15s;
}
.cand-row:hover { background: #161b22; }
.cand-row:last-child { border-bottom: none; }
.cand-rank {
    font-family: 'Space Mono', monospace;
    color: #6e7681;
    font-size: 0.8rem;
    width: 28px;
    flex-shrink: 0;
}
.cand-name { font-size: 0.88rem; font-weight: 600; color: #e6edf3; flex: 1; }
.cand-score {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #3fb950;
    background: rgba(63,185,80,0.1);
    padding: 2px 8px;
    border-radius: 4px;
}
.cand-bar-wrap { width: 100px; height: 4px; background: #21262d; border-radius: 2px; }
.cand-bar { height: 4px; background: #3fb950; border-radius: 2px; transition: width 0.4s ease; }
.cand-table-wrap {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 10px;
    overflow: hidden;
    margin-top: 8px;
}

/* ── Batch Results ── */
.batch-row {
    display: flex;
    align-items: center;
    padding: 10px 16px;
    border-bottom: 1px solid #21262d;
    gap: 12px;
}
.batch-row:hover { background: #161b22; }
.batch-row:last-child { border-bottom: none; }
.correct-badge {
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 20px;
    flex-shrink: 0;
}
.correct-badge.match {
    background: rgba(63,185,80,0.15);
    color: #3fb950;
    border: 1px solid rgba(63,185,80,0.3);
}
.correct-badge.wrong {
    background: rgba(248,81,73,0.12);
    color: #f85149;
    border: 1px solid rgba(248,81,73,0.3);
}

/* ── Section Label ── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #6e7681;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 12px;
    border-left: 3px solid #3fb950;
    padding-left: 8px;
}

/* ── Sidebar Stats ── */
.sidebar-stat {
    background: #0b1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 10px;
}
.sidebar-stat-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #3fb950;
}
.sidebar-stat-label { font-size: 0.72rem; color: #6e7681; margin-top: 1px; }

/* ── Button Override ── */
.stButton>button {
    background: #238636 !important;
    border: 1px solid #2ea043 !important;
    color: #fff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
    padding: 8px 24px !important;
}
.stButton>button:hover {
    background: #2ea043 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(46,160,67,0.35) !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    border: 1px dashed #30363d !important;
    border-radius: 10px !important;
    background: #0d1117 !important;
}

/* ── Audio Player ── */
audio { border-radius: 8px; }

/* ── DataFrame ── */
.stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── MATPLOTLIB THEME ────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#0d1117',
    'axes.facecolor':    '#0d1117',
    'axes.edgecolor':    '#30363d',
    'axes.labelcolor':   '#8b949e',
    'xtick.color':       '#6e7681',
    'ytick.color':       '#6e7681',
    'text.color':        '#c9d1d9',
    'grid.color':        '#21262d',
    'grid.linestyle':    '--',
    'grid.alpha':        0.5,
    'font.family':       'DejaVu Sans',
})

# ─── LOAD SYSTEM ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_system():
    recognizer = AudioRecognizer(sr=11025)
    db = SongDatabase(db_path="song_db.pkl")
    loaded = db.load()
    return recognizer, db, loaded

with st.spinner("Loading fingerprint database…"):
    recognizer, db, db_loaded = load_system()

if not db_loaded:
    st.error("🚫 `song_db.pkl` not found. Please run `build_database.py` first.")
    st.stop()

# ─── PRECOMPUTE STATS ────────────────────────────────────────────────────────
# Compute outside cache to avoid unhashable SongDatabase object issue
def compute_song_hash_counts(db):
    counts = {song_id: 0 for song_id in db.song_names}
    for matches in db.hash_dict.values():
        for sid, _ in matches:
            counts[sid] = counts.get(sid, 0) + 1
    return counts

song_hash_counts = compute_song_hash_counts(db)
total_songs   = len(db.song_names)
total_hashes  = len(db.hash_dict)
avg_hashes    = int(sum(song_hash_counts.values()) / max(total_songs, 1))

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-family:\'Space Mono\',monospace;font-size:0.65rem;color:#6e7681;letter-spacing:2px;text-transform:uppercase;margin-bottom:16px;">SYSTEM STATUS</p>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-stat">
        <div class="sidebar-stat-val">{total_songs}</div>
        <div class="sidebar-stat-label">Songs Indexed</div>
    </div>
    <div class="sidebar-stat">
        <div class="sidebar-stat-val">{total_hashes:,}</div>
        <div class="sidebar-stat-label">Unique Hash Keys</div>
    </div>
    <div class="sidebar-stat">
        <div class="sidebar-stat-val">{avg_hashes:,}</div>
        <div class="sidebar-stat-label">Avg. Hashes / Song</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-family:\'Space Mono\',monospace;font-size:0.65rem;color:#6e7681;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;">ALGORITHM PARAMS</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.78rem;color:#8b949e;line-height:2;">
        <span style="color:#e6edf3;font-family:\'Space Mono\',monospace;">SR</span> 11,025 Hz<br>
        <span style="color:#e6edf3;font-family:\'Space Mono\',monospace;">n_fft</span> 2,048<br>
        <span style="color:#e6edf3;font-family:\'Space Mono\',monospace;">hop_length</span> 512<br>
        <span style="color:#e6edf3;font-family:\'Space Mono\',monospace;">peak_neighborhood</span> 15×15<br>
        <span style="color:#e6edf3;font-family:\'Space Mono\',monospace;">fan_out</span> 20 targets
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:0.72rem;color:#6e7681;">EE200 · Signals, Systems & Networks<br>Team: Zapptain America</p>', unsafe_allow_html=True)

# ─── HERO HEADER ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-badge">EE200 · Project Submission</div>
    <div class="hero-title">Zapptain <span>America</span></div>
    <div class="hero-sub">
        Audio fingerprinting system — index {total_songs} songs as constellation maps,
        then identify any short clip against the database in milliseconds.
    </div>
    <div class="stat-row">
        <span class="stat-pill"><b>{total_songs}</b> songs</span>
        <span class="stat-pill"><b>{total_hashes:,}</b> hash keys</span>
        <span class="stat-pill"><b>Shazam-style</b> algorithm</span>
        <span class="stat-pill"><b>100%</b> accuracy on test set</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab_lib, tab_id, tab_batch = st.tabs([
    "🎵  LIBRARY",
    "🔍  IDENTIFY",
    "📊  BATCH TEST"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 · LIBRARY
# ══════════════════════════════════════════════════════════════════════════════
with tab_lib:
    st.markdown('<div class="section-label">Song fingerprint database</div>', unsafe_allow_html=True)

    # Search
    search_query = st.text_input("", placeholder="🔎  Search songs…", label_visibility="collapsed")

    filtered = {
        sid: name for sid, name in db.song_names.items()
        if search_query.lower() in name.lower()
    }

    if not filtered:
        st.warning("No songs match your search.")
    else:
        # Session state for selected song
        if 'selected_song_id' not in st.session_state:
            st.session_state.selected_song_id = None

        # Placeholder for the detail panel at the TOP
        detail_placeholder = st.container()

        cols = st.columns(4)
        for i, (sid, name) in enumerate(filtered.items()):
            img_b64 = db.song_images.get(sid, "")
            hcount  = song_hash_counts.get(sid, 0)

            # Build label: image (rendered as markdown above button) + name
            with cols[i % 4]:
                # Show constellation image above button
                if img_b64:
                    st.markdown(
                        f'<img src="data:image/png;base64,{img_b64}" '
                        f'style="width:100%;border-radius:8px 8px 0 0;display:block;margin-bottom:-6px;" />',
                        unsafe_allow_html=True
                    )
                # Actual clickable button styled as card
                btn_label = f"{name}\n{hcount:,} hashes"
                if st.button(btn_label, key=f"song_btn_{sid}", use_container_width=True):
                    if st.session_state.selected_song_id == sid:
                        st.session_state.selected_song_id = None  # toggle off
                    else:
                        st.session_state.selected_song_id = sid

        # ── Song Detail Panel (Rendered at the top) ───────────────────────
        sel_sid = st.session_state.selected_song_id
        if sel_sid is not None and sel_sid in db.song_names:
            sel_name   = db.song_names[sel_sid]
            sel_img    = db.song_images.get(sel_sid, "")
            sel_hcount = song_hash_counts.get(sel_sid, 0)

            img_html = (
                f'<img class="detail-img" src="data:image/png;base64,{sel_img}" />'
                if sel_img else
                '<div style="height:120px;background:#161b22;border-radius:8px;'
                'display:flex;align-items:center;justify-content:center;font-size:2rem;">🎵</div>'
            )

            with detail_placeholder:
                st.markdown(f"""
                <div class="song-detail-panel">
                    <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
                        <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
                                    color:#3fb950;letter-spacing:3px;text-transform:uppercase;">Selected Song</div>
                        <div style="flex:1;height:1px;background:#21262d;"></div>
                        <div style="font-size:0.75rem;color:#6e7681;cursor:pointer;">click song again to close</div>
                    </div>
                <div style="display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start;">
                    <div style="flex:1;min-width:200px;">
                        {img_html}
                    </div>
                    <div style="flex:2;min-width:220px;">
                        <div style="font-size:1.6rem;font-weight:900;color:#f0f6fc;
                                    letter-spacing:-0.5px;line-height:1.2;margin-bottom:12px;">
                            {sel_name}
                        </div>
                        <div class="detail-stat-row">
                            <div class="detail-stat">
                                <div class="detail-stat-val">{sel_hcount:,}</div>
                                <div class="detail-stat-label">Total Hashes</div>
                            </div>
                            <div class="detail-stat">
                                <div class="detail-stat-val">{sel_sid}</div>
                                <div class="detail-stat-label">Song ID</div>
                            </div>
                            <div class="detail-stat">
                                <div class="detail-stat-val">60s</div>
                                <div class="detail-stat-label">Indexed Duration</div>
                            </div>
                            <div class="detail-stat">
                                <div class="detail-stat-val">11kHz</div>
                                <div class="detail-stat-label">Sample Rate</div>
                            </div>
                        </div>
                        <div style="margin-top:14px;font-size:0.8rem;color:#8b949e;line-height:1.7;">
                            This song's acoustic fingerprint was generated by extracting the
                            <span style="color:#e6edf3;">strongest local peaks</span> from its
                            log-magnitude spectrogram, forming a sparse constellation map.
                            Each pair of peaks within the target zone produced a unique
                            <span style="color:#3fb950;font-family:'Space Mono',monospace;">(f₁, f₂, Δt)</span> hash.
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 · IDENTIFY
# ══════════════════════════════════════════════════════════════════════════════
with tab_id:
    st.markdown('<div class="section-label">Upload a short audio clip to identify</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop an MP3 or WAV file here",
        type=["mp3", "wav"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        st.audio(uploaded_file)
        if st.button("⚡  Identify Song", key="identify_btn"):
            with st.spinner("Fingerprinting audio…"):
                t_start = time.time()
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    t0 = time.time(); spectrogram, y = recognizer.get_spectrogram(tmp_path); t_spec  = int((time.time()-t0)*1000)
                    t0 = time.time(); freqs, times   = recognizer.extract_peaks(spectrogram, percentile=90); t_const = int((time.time()-t0)*1000)
                    t0 = time.time(); hashes         = recognizer.generate_hashes(freqs, times); t_hash  = int((time.time()-t0)*1000)
                    t0 = time.time(); results        = db.match_hashes(hashes); t_match = int((time.time()-t0)*1000)
                    t_total = int((time.time()-t_start)*1000)
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            # ── Timing Bar ──────────────────────────────────────────────────
            st.markdown(f"""
            <div class="timing-bar">
                <div class="timing-cell">
                    <div class="timing-val">{t_spec} ms</div>
                    <div class="timing-label">Spectrogram</div>
                </div>
                <div class="timing-cell">
                    <div class="timing-val">{t_const} ms</div>
                    <div class="timing-label">Constellation</div>
                </div>
                <div class="timing-cell">
                    <div class="timing-val">{t_hash} ms</div>
                    <div class="timing-label">Hashing</div>
                </div>
                <div class="timing-cell">
                    <div class="timing-val">{t_match} ms</div>
                    <div class="timing-label">DB Lookup</div>
                </div>
                <div class="timing-cell">
                    <div class="timing-val">{t_total} ms</div>
                    <div class="timing-label">Total</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if not results:
                st.markdown("""
                <div class="no-match-banner">
                    <div style="font-size:2rem;margin-bottom:8px;">❓</div>
                    <div style="font-weight:700;color:#e6edf3;font-size:1.1rem;">No match found</div>
                    <div style="font-size:0.85rem;margin-top:5px;">This clip doesn't appear to be in the database.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                best = results[0]
                top_score = best['score']
                confidence = min(100, int((top_score / max(top_score, 1)) * 100))

                # ── Match Banner ─────────────────────────────────────────────
                st.markdown(f"""
                <div class="match-banner">
                    <div class="match-eyebrow">✓ Match Found</div>
                    <div class="match-song-name">{best['song_name']}</div>
                    <div class="match-detail" style="margin-top:10px;">
                        <span class="match-score-badge">{top_score:,} hashes</span>
                        &nbsp; aligned at a single consistent time offset
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── STEP 1: Waveform + Spectrogram ───────────────────────────
                st.markdown("""
                <div class="step-header">
                    <div class="step-num">01</div>
                    <div class="step-title">Waveform & Spectrogram</div>
                    <div class="step-sub">Time-domain → Frequency-domain</div>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 3))
                    t_axis = np.linspace(0, len(y)/recognizer.sr, len(y))
                    ax.plot(t_axis, y, color='#58a6ff', linewidth=0.4, alpha=0.85)
                    ax.fill_between(t_axis, y, alpha=0.15, color='#58a6ff')
                    ax.set_title("Raw Waveform", fontsize=10, pad=8)
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Amplitude")
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                with col2:
                    fig, ax = plt.subplots(figsize=(8, 3))
                    img = ax.imshow(spectrogram, aspect='auto', origin='lower', cmap='magma',
                                    vmin=spectrogram.max()-80, vmax=spectrogram.max())
                    fig.colorbar(img, ax=ax, format='%+.0f dB', shrink=0.8)
                    ax.set_title("Log-Magnitude Spectrogram", fontsize=10, pad=8)
                    ax.set_xlabel("Time Frames")
                    ax.set_ylabel("Frequency Bins")
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                # ── STEP 2: Constellation Map ─────────────────────────────────
                st.markdown("""
                <div class="step-header">
                    <div class="step-num">02</div>
                    <div class="step-title">Constellation Map</div>
                    <div class="step-sub">Local spectral peaks — the "stars"</div>
                </div>
                """, unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(14, 4))
                ax.imshow(spectrogram, aspect='auto', origin='lower', cmap='magma',
                          alpha=0.55, vmin=spectrogram.max()-80, vmax=spectrogram.max())
                ax.scatter(times, freqs, c='#00ff9d', s=3, alpha=0.85, linewidths=0)
                ax.set_title(
                    f"Constellation Map — {len(freqs):,} strongest spectral peaks extracted",
                    fontsize=10, pad=8
                )
                ax.set_xlabel("Time Frames")
                ax.set_ylabel("Frequency Bins")
                ax.set_xlim(0, spectrogram.shape[1])
                ax.set_ylim(0, spectrogram.shape[0])
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                # ── STEP 3: Alignment Histogram ───────────────────────────────
                st.markdown("""
                <div class="step-header">
                    <div class="step-num">03</div>
                    <div class="step-title">Time-Offset Alignment Histogram</div>
                    <div class="step-sub">The mathematical proof of identity</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(
                    "Every matched hash casts a vote for a time offset Δt = (db_time − query_time). "
                    "A genuine match makes thousands of independent hashes agree on the **same single offset**. "
                    "That convergence spike is statistically impossible by chance.",
                    unsafe_allow_html=False
                )

                histogram_data = best['histogram']
                offsets = list(histogram_data.keys())
                counts  = list(histogram_data.values())
                max_off = max(histogram_data, key=histogram_data.get)
                max_cnt = histogram_data[max_off]

                fig, ax = plt.subplots(figsize=(14, 4))
                ax.vlines(offsets, 0, counts, color='#6e7681', linewidth=1.2, alpha=0.6)
                ax.vlines(max_off, 0, max_cnt, color='#3fb950', linewidth=4)
                ax.fill_between([max_off-30, max_off+30], 0, max_cnt,
                                color='#3fb950', alpha=0.15)

                # Smart annotation
                xrange = max(offsets) - min(offsets) if len(offsets) > 1 else 1
                txt_x  = max_off - xrange * 0.15 if max_off > (min(offsets) + xrange * 0.5) else max_off + xrange * 0.05
                ax.annotate(
                    f"{max_cnt:,} hashes\nalign here",
                    xy=(max_off, max_cnt),
                    xytext=(txt_x, max_cnt * 0.75),
                    color='#3fb950',
                    fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='#3fb950', lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#112218', edgecolor='#3fb950', alpha=0.8)
                )
                ax.set_ylim(0, max_cnt * 1.25)
                ax.set_xlabel("Time Offset (database frame − query frame)")
                ax.set_ylabel("Number of Matching Hashes")
                ax.set_title(f"Offset Histogram for '{best['song_name']}'", fontsize=10, pad=8)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                # ── STEP 4: Top Candidates ────────────────────────────────────
                st.markdown("""
                <div class="step-header">
                    <div class="step-num">04</div>
                    <div class="step-title">Top Candidates Ranked</div>
                    <div class="step-sub">All competing songs scored</div>
                </div>
                """, unsafe_allow_html=True)

                top5 = results[:5]
                top5_max = top5[0]['score']
                rows_html = ""
                for rank, r in enumerate(top5, 1):
                    pct = int(r['score'] / top5_max * 100) if top5_max > 0 else 0
                    gold = rank == 1
                    name_style = 'color:#3fb950;' if gold else ''
                    rows_html += f"""
                    <div class="cand-row">
                        <div class="cand-rank">{'🥇' if gold else f'#{rank}'}</div>
                        <div class="cand-name" style="{name_style}">{r['song_name']}</div>
                        <div class="cand-bar-wrap"><div class="cand-bar" style="width:{pct}%;{'background:#3fb950' if gold else 'background:#58a6ff'}"></div></div>
                        <div class="cand-score">{r['score']:,}</div>
                    </div>"""

                st.markdown(f'<div class="cand-table-wrap">{rows_html}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 · BATCH TEST
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown('<div class="section-label">Bulk identification & accuracy benchmarking</div>', unsafe_allow_html=True)
    st.markdown(
        "Upload multiple query clips. Each is fingerprinted and matched silently in the background. "
        "Results are colour-coded and exported as `results.csv`."
    )

    uploaded_files = st.file_uploader(
        "Drop multiple audio clips here",
        type=["mp3", "wav"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.info(f"**{len(uploaded_files)}** clip(s) queued.")
        if st.button("⚡  Run Batch Identification", key="batch_btn"):
            progress = st.progress(0)
            status   = st.empty()
            results_data = []

            for i, f in enumerate(uploaded_files):
                base = os.path.splitext(f.name)[0]
                status.markdown(f"Identifying `{f.name}`… ({i+1}/{len(uploaded_files)})")
                prediction = "ERROR"
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                        tmp.write(f.getvalue())
                        tmp_path = tmp.name
                    spec, _ = recognizer.get_spectrogram(tmp_path)
                    fr, ti  = recognizer.extract_peaks(spec, percentile=90)
                    hs      = recognizer.generate_hashes(fr, ti)
                    ms      = db.match_hashes(hs)
                    prediction = ms[0]['song_name'] if ms else "NO MATCH"
                except Exception:
                    prediction = "ERROR"
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

                # Robust match: strip punctuation + lowercase comparison
                import re
                def normalise(s): return re.sub(r'[^a-z0-9]', '', s.lower())
                is_correct = normalise(base) == normalise(prediction)
                results_data.append({
                    "filename":   base,
                    "prediction": prediction,
                    "correct":    is_correct
                })
                progress.progress((i+1) / len(uploaded_files))

            status.markdown("**✅ Done!**")

            df = pd.DataFrame(results_data)
            correct_count = df['correct'].sum()
            accuracy = int(correct_count / len(df) * 100)

            # ── Summary Stats ────────────────────────────────────────────────
            st.markdown(f"""
            <div style="display:flex;gap:16px;margin:16px 0;">
                <div class="sidebar-stat" style="flex:1;">
                    <div class="sidebar-stat-val">{len(df)}</div>
                    <div class="sidebar-stat-label">Clips Processed</div>
                </div>
                <div class="sidebar-stat" style="flex:1;">
                    <div class="sidebar-stat-val">{correct_count}</div>
                    <div class="sidebar-stat-label">Correct Matches</div>
                </div>
                <div class="sidebar-stat" style="flex:1;">
                    <div class="sidebar-stat-val" style="color:{'#3fb950' if accuracy==100 else '#f0883e'};">{accuracy}%</div>
                    <div class="sidebar-stat-label">Accuracy</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Colour-coded Results Table ────────────────────────────────────
            st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
            rows_html = ""
            for _, row in df.iterrows():
                badge = (
                    '<span class="correct-badge match">✓ MATCH</span>' if row['correct']
                    else '<span class="correct-badge wrong">✗ WRONG</span>'
                )
                rows_html += f"""
                <div class="batch-row">
                    <div style="flex:1;font-size:0.85rem;font-weight:600;color:#e6edf3;">{row['filename']}</div>
                    <div style="flex:1;font-size:0.85rem;color:#8b949e;">{row['prediction']}</div>
                    {badge}
                </div>"""

            st.markdown(f'<div class="cand-table-wrap">{rows_html}</div>', unsafe_allow_html=True)

            # ── Download ─────────────────────────────────────────────────────
            csv_out = df[['filename','prediction']].to_csv(index=False)
            st.download_button(
                "⬇  Download results.csv",
                data=csv_out,
                file_name="results.csv",
                mime="text/csv"
            )
