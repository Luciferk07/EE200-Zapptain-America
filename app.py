import streamlit as st
import streamlit.components.v1 as components
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import time
import re
import io
import base64
import random
from audio_recognizer import AudioRecognizer, SongDatabase

# ─────────────────────────────────────────────────────────── PAGE CONFIG
st.set_page_config(
    page_title="SoundScope · Audio Fingerprinting",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────── CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&family=Unbounded:wght@700;800&display=swap');

/* BASE */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #0c0c0e;
    color: #e4e4e7;
}
* { box-sizing: border-box; }
.block-container { padding: 1.5rem 2.2rem 4rem !important; max-width: 1440px !important; }

/* SCROLLBAR */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #3f3f46; border-radius: 4px; }

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: #09090b !important;
    border-right: 1px solid #1c1c1f !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem !important; }

.sb-logo {
    font-family: 'Unbounded', sans-serif;
    font-size: 1rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #fafafa;
    margin-bottom: 6px;
}
.sb-logo span { color: #818cf8; }
.sb-tagline { font-size: 0.65rem; color: #52525b; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 24px; }

.sb-divider { height: 1px; background: #18181b; margin: 18px 0; }

.sb-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem; color: #52525b;
    letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 10px;
}
.sb-stat {
    display: flex; justify-content: space-between; align-items: center;
    padding: 9px 12px;
    background: #111113;
    border: 1px solid #1c1c1f;
    border-radius: 8px;
    margin-bottom: 6px;
}
.sb-stat-key { font-size: 0.73rem; color: #71717a; }
.sb-stat-val { font-family: 'JetBrains Mono', monospace; font-size: 0.73rem; color: #a78bfa; font-weight: 600; }

/* HERO */
.hero {
    padding: 36px 0 28px;
    border-bottom: 1px solid #1c1c1f;
    margin-bottom: 28px;
}
.hero-kicker {
    display: inline-flex; align-items: center; gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; color: #818cf8;
    letter-spacing: 2.5px; text-transform: uppercase;
    margin-bottom: 16px;
}
.hero-pulse {
    width: 7px; height: 7px; border-radius: 50%;
    background: #818cf8;
    animation: pulse-anim 2s ease-in-out infinite;
}
@keyframes pulse-anim {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(129,140,248,0.4); }
    50% { opacity: 0.6; box-shadow: 0 0 0 6px rgba(129,140,248,0); }
}
.hero-title {
    font-family: 'Unbounded', sans-serif;
    font-size: 3.6rem; font-weight: 800;
    letter-spacing: -2.5px; line-height: 1;
    color: #fafafa; margin: 0;
}
.hero-title em { font-style: normal; color: #818cf8; }
.hero-sub {
    margin-top: 14px; color: #52525b;
    font-size: 0.88rem; line-height: 1.7; max-width: 560px;
}
.hero-chips { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 20px; }
.chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: #71717a;
    background: #111113; border: 1px solid #27272a;
    border-radius: 20px; padding: 4px 12px;
}
.chip b { color: #d4d4d8; }

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    background: #111113;
    border: 1px solid #1c1c1f;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
    margin-bottom: 24px;
}
.stTabs [data-baseweb="tab"] {
    height: 38px; background: transparent; border: none;
    border-radius: 7px; padding: 0 20px;
    color: #52525b; font-weight: 600; font-size: 0.8rem;
    letter-spacing: 0.3px; transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover { color: #a1a1aa; background: #18181b !important; }
.stTabs [aria-selected="true"] {
    background: #18181b !important;
    color: #fafafa !important;
    box-shadow: inset 0 0 0 1px #27272a;
}

/* SECTION LABEL */
.sec-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem; color: #3f3f46;
    letter-spacing: 2px; text-transform: uppercase;
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 18px;
}
.sec-label::after { content:''; flex:1; height:1px; background:#18181b; }

/* ── LIBRARY GRID ── */
.song-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 16px;
    margin-top: 20px;
}
.song-card {
    background: #111113;
    border: 1px solid #1c1c1f;
    border-radius: 14px;
    padding: 12px;
    transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
}
.song-card:hover {
    transform: translateY(-4px);
    border-color: #3f3f46;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
}
.song-card img {
    width: 100%;
    height: 140px;
    object-fit: cover;
    border-radius: 10px;
    margin-bottom: 14px;
    background: #0c0c0e;
    border: 1px solid #1c1c1f;
}
.song-card-title {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    color: #e4e4e7;
    margin-bottom: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.song-card-hashes {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #a1a1aa;
    font-weight: 600;
}

/* IDENTIFY TAB */
.upload-zone {
    border: 1.5px dashed #27272a;
    border-radius: 12px;
    background: #0c0c0e;
    transition: border-color 0.2s, background 0.2s;
}
.timing-strip {
    display: flex; margin: 16px 0;
    background: #111113; border: 1px solid #1c1c1f;
    border-radius: 10px; overflow: hidden;
}
.t-cell { flex:1; padding:12px; text-align:center; border-right:1px solid #1c1c1f; }
.t-cell:last-child { border-right:none; }
.t-val { font-family:'JetBrains Mono',monospace; font-size:1.1rem; font-weight:700; color:#818cf8; }
.t-lbl { font-size:0.6rem; color:#3f3f46; text-transform:uppercase; letter-spacing:1px; margin-top:3px; }

.result-card {
    background: #111113; border: 1px solid #27272a; border-radius: 14px;
    padding: 26px 30px; margin: 18px 0; position:relative; overflow:hidden;
    animation: fadeIn 0.3s ease;
}
.result-card.matched { border-color: rgba(129,140,248,0.35); box-shadow: 0 0 40px rgba(129,140,248,0.06); }
.result-card.nomatch { border-color: #27272a; }
.result-eyebrow {
    font-family:'JetBrains Mono',monospace; font-size:0.58rem;
    letter-spacing:2.5px; text-transform:uppercase; margin-bottom:10px;
    display:flex; align-items:center; gap:8px;
}
.result-eyebrow.ok { color:#818cf8; }
.result-eyebrow.fail { color:#52525b; }
.result-song { font-family:'Unbounded',sans-serif; font-size:2rem; font-weight:800; color:#fafafa; letter-spacing:-1px; line-height:1.1; }
.result-badge {
    display:inline-block; margin-top:12px;
    background: rgba(129,140,248,0.1); border:1px solid rgba(129,140,248,0.25);
    color:#818cf8; font-family:'JetBrains Mono',monospace;
    font-size:0.75rem; padding:5px 14px; border-radius:7px;
}

.step-hdr {
    display:flex; align-items:center; gap:12px;
    border-left: 2px solid #27272a; padding-left:14px;
    margin: 28px 0 14px;
}
.step-n {
    font-family:'JetBrains Mono',monospace; font-size:0.62rem; font-weight:600;
    color:#818cf8; background:rgba(129,140,248,0.1); border:1px solid rgba(129,140,248,0.2);
    border-radius:5px; padding:3px 8px;
}
.step-t { font-weight:700; font-size:0.88rem; color:#e4e4e7; }
.step-s { font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#3f3f46; margin-left:auto; }

/* CANDIDATE TABLE */
.cand-wrap { background:#111113; border:1px solid #1c1c1f; border-radius:10px; overflow:hidden; margin-top:10px; }
.cand-row { display:flex; align-items:center; padding:11px 16px; border-bottom:1px solid #0f0f11; gap:14px; transition:background 0.1s; }
.cand-row:hover { background:#18181b; }
.cand-row:last-child { border-bottom:none; }
.cand-rank { font-family:'JetBrains Mono',monospace; color:#3f3f46; font-size:0.72rem; width:28px; flex-shrink:0; }
.cand-name { font-size:0.85rem; font-weight:600; color:#e4e4e7; flex:1; }
.cand-bar-wrap { width:90px; height:2px; background:#1c1c1f; border-radius:2px; }
.cand-bar { height:2px; border-radius:2px; }
.cand-score { font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#818cf8; background:rgba(129,140,248,0.08); padding:2px 8px; border-radius:5px; }

/* BATCH TAB */
.batch-table-wrap { background:#111113; border:1px solid #1c1c1f; border-radius:10px; overflow:hidden; margin-top:10px; }
.batch-row { display:flex; align-items:center; padding:10px 16px; border-bottom:1px solid #0f0f11; gap:12px; }
.batch-row:hover { background:#18181b; }
.batch-row:last-child { border-bottom:none; }
.badge-ok { font-family:'JetBrains Mono',monospace; font-size:0.62rem; font-weight:700; padding:3px 9px; border-radius:20px; background:rgba(129,140,248,0.1); color:#818cf8; border:1px solid rgba(129,140,248,0.25); }
.badge-fail { font-family:'JetBrains Mono',monospace; font-size:0.62rem; font-weight:700; padding:3px 9px; border-radius:20px; background:rgba(248,113,113,0.08); color:#f87171; border:1px solid rgba(248,113,113,0.2); }
.stat-box { background:#111113; border:1px solid #1c1c1f; border-radius:10px; padding:14px 18px; text-align:center; }
.stat-box-val { font-family:'JetBrains Mono',monospace; font-size:1.5rem; font-weight:700; color:#818cf8; }
.stat-box-key { font-size:0.65rem; color:#3f3f46; text-transform:uppercase; letter-spacing:1px; margin-top:3px; }

/* HOW IT WORKS */
.hw-card { background:#111113; border:1px solid #1c1c1f; border-radius:12px; padding:20px; margin-bottom:12px; transition:border-color 0.2s; }
.hw-card:hover { border-color:#27272a; }
.hw-icon { font-size:1.4rem; margin-bottom:10px; }
.hw-title { font-weight:800; font-size:0.88rem; color:#e4e4e7; margin-bottom:6px; }
.hw-desc { font-size:0.75rem; color:#52525b; line-height:1.75; }

/* OVERRIDES */
.stButton > button {
    background: #18181b !important; border: 1px solid #27272a !important;
    color: #e4e4e7 !important; font-weight:600 !important;
    border-radius:9px !important; transition:all 0.2s !important;
    padding:9px 22px !important;
}
.stButton > button:hover {
    background: #27272a !important; border-color:#3f3f46 !important;
    transform:translateY(-1px) !important; box-shadow:0 4px 16px rgba(0,0,0,0.4) !important;
}
.stProgress > div > div > div { background: linear-gradient(90deg,#6366f1,#818cf8) !important; border-radius:4px; }
.stTextInput input {
    background:#111113 !important; border:1px solid #1c1c1f !important;
    border-radius:9px !important; color:#e4e4e7 !important;
    font-size:0.85rem !important;
}
.stTextInput input:focus { border-color:#3f3f46 !important; box-shadow:none !important; }
.stTextInput input::placeholder { color:#3f3f46 !important; }
[data-testid="stFileUploader"] {
    border: 1.5px dashed #27272a !important; border-radius:12px !important;
    background:#0c0c0e !important;
}
audio { border-radius:8px; width:100%; }
</style>
""", unsafe_allow_html=True)



# ─────────────────────────────────────────────────────────── MATPLOTLIB
plt.rcParams.update({
    'figure.facecolor':'#0c0c0e','axes.facecolor':'#111113',
    'axes.edgecolor':'#1c1c1f','axes.labelcolor':'#52525b',
    'xtick.color':'#3f3f46','ytick.color':'#3f3f46',
    'text.color':'#e4e4e7','grid.color':'#18181b',
    'grid.linestyle':'--','grid.alpha':0.5,'font.family':'DejaVu Sans',
})

# ─────────────────────────────────────────────────────────── LOAD
@st.cache_resource(show_spinner=False)
def load_system():
    r = AudioRecognizer(sr=11025)
    db = SongDatabase(db_path="song_db.pkl")
    ok = db.load()
    return r, db, ok

with st.spinner("Loading database…"):
    recognizer, db, db_loaded = load_system()

if not db_loaded:
    st.error("song_db.pkl not found — run build_database.py first.")
    st.stop()

@st.cache_resource(show_spinner=False)
def precompute_data(_db):
    hc = {sid:0 for sid in _db.song_names}
    pts = {sid:[] for sid in _db.song_names}
    for h_key, matches in _db.hash_dict.items():
        # Using the hash key frequency components (f1) as Y and offset as X for the plot
        f1 = h_key[0] if isinstance(h_key, tuple) else (h_key % 1000)
        for sid, offset in matches:
            hc[sid] = hc.get(sid,0)+1
            pts[sid].append((offset, f1))
    return hc, pts

song_hash_counts, song_points = precompute_data(db)
total_songs  = len(db.song_names)
total_hashes = len(db.hash_dict)
avg_hashes   = int(sum(song_hash_counts.values())/max(total_songs,1))
max_hashes   = max(song_hash_counts.values()) if song_hash_counts else 1


# ─────────────────────────────────────────────────────────── SIDEBAR
with st.sidebar:
    st.markdown('<div class="sb-logo">Sound<span>Scope</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-tagline">Audio Fingerprinting Engine</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-label">Database</div>', unsafe_allow_html=True)
    for k, v in [("Songs indexed", str(total_songs)),
                 ("Unique hash keys", f"{total_hashes:,}"),
                 ("Avg hashes / song", f"{avg_hashes:,}"),
                 ("Index type", "Full song")]:
        st.markdown(f'<div class="sb-stat"><span class="sb-stat-key">{k}</span><span class="sb-stat-val">{v}</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">Algorithm</div>', unsafe_allow_html=True)
    for k, v in [("sample rate","11,025 Hz"),("n_fft","2,048"),("hop_length","512"),
                 ("neighborhood","15×15"),("fan out","20"),("min score","15")]:
        st.markdown(f'<div class="sb-stat"><span class="sb-stat-key">{k}</span><span class="sb-stat-val">{v}</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.65rem;color:#27272a;line-height:1.8;">EE200 · IIT Kanpur · 2026</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────── HERO
st.markdown(f"""
<div class="hero">
  <div class="hero-kicker"><span class="hero-pulse"></span>Live · EE200 Project Demo</div>
  <div class="hero-title">Sound<em>Scope</em></div>
  <div class="hero-sub">
    A Shazam-style audio fingerprinting engine. Index {total_songs} songs as sparse constellation maps,
    then identify any short clip in milliseconds — robust to noise and cropping.
  </div>
  <div class="hero-chips">
    <span class="chip"><b>{total_songs}</b> songs</span>
    <span class="chip"><b>{total_hashes:,}</b> hash keys</span>
    <span class="chip"><b>STFT</b> + peak pairing</span>
    <span class="chip"><b>O(1)</b> lookup per hash</span>
    <span class="chip"><b>100%</b> test accuracy</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────── TABS
tab_lib, tab_id, tab_batch, tab_how = st.tabs([
    "◫  LIBRARY", "◈  IDENTIFY", "▦  BATCH TEST", "◉  HOW IT WORKS"
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — LIBRARY  (Grid of Cards with Constellation Maps)
# ══════════════════════════════════════════════════════════════════════
with tab_lib:
    if "view_song_id" not in st.session_state:
        st.session_state.view_song_id = None

    if st.session_state.view_song_id is not None:
        sid = st.session_state.view_song_id
        name = db.song_names.get(sid, "Unknown")
        
        if st.button("← Back to Library", key="btn_back"):
            st.session_state.view_song_id = None
            st.rerun()
            
        st.markdown(f'<div class="sec-label">Details · {name}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="hero-title" style="font-size:2.4rem;margin-bottom:20px;">{name}</div>', unsafe_allow_html=True)
        
        mp3_p = os.path.join("EE200_course_project_data_2026", "Q3_database", f"{name}.mp3")
        if os.path.exists(mp3_p):
            st.audio(mp3_p)
            
            with st.spinner("Analyzing 10-second sample..."):
                spec, y = recognizer.get_spectrogram(mp3_p, duration=10, offset=10)
                fr, ti = recognizer.extract_peaks(spec, percentile=90)
                
                # Step 1 — Waveform & Spectrogram
                st.markdown('<div class="step-hdr"><div class="step-n">01</div><div class="step-t">Waveform & Spectrogram</div><div class="step-s">time → frequency</div></div>', unsafe_allow_html=True)
                c1,c2 = st.columns(2)
                with c1:
                    fig,ax=plt.subplots(figsize=(8,3))
                    tax=np.linspace(0,len(y)/recognizer.sr,len(y))
                    ax.plot(tax,y,color='#6366f1',linewidth=0.5,alpha=0.9)
                    ax.fill_between(tax,y,alpha=0.1,color='#818cf8')
                    ax.set_title("Waveform (10s sample)",fontsize=9,color='#71717a',pad=6)
                    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
                    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                    fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
                with c2:
                    fig,ax=plt.subplots(figsize=(8,3))
                    im=ax.imshow(spec,aspect='auto',origin='lower',cmap='magma',vmin=spec.max()-80,vmax=spec.max())
                    fig.colorbar(im,ax=ax,format='%+.0fdB',shrink=0.75)
                    ax.set_title("Log-Magnitude Spectrogram",fontsize=9,color='#71717a',pad=6)
                    ax.set_xlabel("Time Frames"); ax.set_ylabel("Freq Bins")
                    fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

                # Step 2 — Constellation Map
                st.markdown('<div class="step-hdr"><div class="step-n">02</div><div class="step-t">Constellation Map</div><div class="step-s">spectral peaks</div></div>', unsafe_allow_html=True)
                fig,ax=plt.subplots(figsize=(14,3.5))
                ax.imshow(spec,aspect='auto',origin='lower',cmap='magma',alpha=0.45,vmin=spec.max()-80,vmax=spec.max())
                ax.scatter(ti,fr,c='#a78bfa',s=2.5,alpha=0.9,linewidths=0)
                ax.set_title(f"{len(fr):,} spectral peaks extracted in sample",fontsize=9,color='#71717a',pad=6)
                ax.set_xlabel("Time Frames"); ax.set_ylabel("Freq Bins")
                ax.set_xlim(0,spec.shape[1]); ax.set_ylim(0,spec.shape[0])
                fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
            
    else:
        search = st.text_input("Search", placeholder="Search tracks…", label_visibility="collapsed")
    
        sorted_songs = sorted(db.song_names.items(), key=lambda x: x[1])
        filtered = [(s_id,s_name) for s_id,s_name in sorted_songs if search.lower() in s_name.lower()]
    
        if not filtered:
            st.warning("No tracks match.")
        else:
            st.markdown(f'<div class="sec-label">Showing {len(filtered)} of {total_songs} tracks</div>', unsafe_allow_html=True)

        card_colors = ['#38bdf8', '#fbbf24', '#a78bfa', '#f472b6', '#34d399']
        COLS = 5
        rows = [filtered[i:i+COLS] for i in range(0, len(filtered), COLS)]
        for row in rows:
            cols = st.columns(COLS)
            for col_idx, (sid, name) in enumerate(row):
                hc = song_hash_counts.get(sid, 0)
                color = card_colors[col_idx % len(card_colors)]
                with cols[col_idx]:
                    pts = song_points.get(sid, [])
                    if pts:
                        samp = random.sample(pts, min(800, len(pts)))
                        xp = [p[0] for p in samp]
                        yp = [p[1] for p in samp]
                        fig_c, ax_c = plt.subplots(figsize=(3, 1.8), dpi=90)
                        fig_c.patch.set_facecolor('#0c0c0e')
                        ax_c.set_facecolor('#0c0c0e')
                        ax_c.scatter(xp, yp, s=0.6, c=color, alpha=0.9, edgecolors='none')
                        ax_c.axis('off')
                        fig_c.tight_layout(pad=0.1)
                        st.pyplot(fig_c, use_container_width=True)
                        plt.close(fig_c)
                    st.markdown(
                        f'<div class="song-card-title" title="{name}">{name}</div>'
                        f'<div class="song-card-hashes">{hc:,} hashes</div>',
                        unsafe_allow_html=True
                    )
                    if st.button("View Details", key=f"view_{sid}"):
                        st.session_state.view_song_id = sid
                        st.rerun()
                    st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — IDENTIFY
# ══════════════════════════════════════════════════════════════════════
with tab_id:
    def run_identification(audio_bytes, suffix=".mp3"):
        with st.spinner("Fingerprinting…"):
            t0_total = time.time()
            tmp = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
                    tf.write(audio_bytes); tmp = tf.name

                t0=time.time(); spec,y = recognizer.get_spectrogram(tmp); t_sp=int((time.time()-t0)*1000)
                t0=time.time(); fr,ti  = recognizer.extract_peaks(spec, percentile=90); t_cn=int((time.time()-t0)*1000)
                t0=time.time(); hs     = recognizer.generate_hashes(fr,ti); t_hs=int((time.time()-t0)*1000)
                t0=time.time(); res    = db.match_hashes(hs); t_db=int((time.time()-t0)*1000)
                t_tot = int((time.time()-t0_total)*1000)

                st.session_state.id_result = dict(
                    spec=spec,y=y,fr=fr,ti=ti,hs=hs,res=res,
                    t_sp=t_sp,t_cn=t_cn,t_hs=t_hs,t_db=t_db,t_tot=t_tot
                )
            except Exception as ex:
                st.error(f"Error: {ex}")
            finally:
                if tmp and os.path.exists(tmp): os.unlink(tmp)

    st.markdown('<div class="sec-label">Upload a clip · 5–30 seconds works best</div>', unsafe_allow_html=True)

    up = st.file_uploader("Upload clip", type=["mp3","wav"], label_visibility="collapsed")

    # ── Sample clip generator ─────────────────────────────────────────
    with st.expander("🎲  Try a sample clip from the database"):
        sample_songs = sorted(db.song_names.items(), key=lambda x: x[1])
        sample_choice = st.selectbox("Pick a song", [n for _,n in sample_songs], label_visibility="collapsed")
        col_s1, col_s2 = st.columns([1,2])
        with col_s1:
            start_sec = st.slider("Start (seconds)", 0, 120, 30)
        with col_s2:
            dur_sec = st.slider("Duration (seconds)", 3, 20, 8)

        if st.button("Generate & Auto-Identify", key="gen_sample"):
            mp3_p = os.path.join("EE200_course_project_data_2026","Q3_database",f"{sample_choice}.mp3")
            if os.path.exists(mp3_p):
                import librosa, soundfile as sf
                try:
                    y_s, sr_s = librosa.load(mp3_p, sr=None, offset=float(start_sec), duration=float(dur_sec))
                    buf = io.BytesIO()
                    sf.write(buf, y_s, sr_s, format="WAV")
                    buf.seek(0)
                    st.audio(buf, format="audio/wav")
                    run_identification(buf.getvalue(), ".wav")
                except Exception as e:
                    st.warning(f"Could not generate clip: {e}")

    if "id_result" not in st.session_state:
        st.session_state.id_result = None

    if up:
        st.audio(up)
        if st.button("⚡  Identify", key="id_btn"):
            run_identification(up.getvalue())

    if st.session_state.id_result:
        R = st.session_state.id_result
        spec=R['spec']; y=R['y']; fr=R['fr']; ti=R['ti']; res=R['res']

        # Timing strip
        st.markdown(f"""
        <div class="timing-strip">
          <div class="t-cell"><div class="t-val">{R['t_sp']}ms</div><div class="t-lbl">Spectrogram</div></div>
          <div class="t-cell"><div class="t-val">{R['t_cn']}ms</div><div class="t-lbl">Peaks</div></div>
          <div class="t-cell"><div class="t-val">{R['t_hs']}ms</div><div class="t-lbl">Hashing</div></div>
          <div class="t-cell"><div class="t-val">{R['t_db']}ms</div><div class="t-lbl">Lookup</div></div>
          <div class="t-cell"><div class="t-val">{R['t_tot']}ms</div><div class="t-lbl">Total</div></div>
        </div>""", unsafe_allow_html=True)

        if not res or res[0]['score'] < 15:
            st.markdown("""
            <div class="result-card nomatch">
              <div class="result-eyebrow fail">◈ No match found</div>
              <div style="color:#52525b;font-size:0.85rem;margin-top:6px;">This clip is not in the database.</div>
            </div>""", unsafe_allow_html=True)
        else:
            best = res[0]
            st.markdown(f"""
            <div class="result-card matched">
              <div class="result-eyebrow ok"><span style="width:7px;height:7px;border-radius:50%;background:#818cf8;display:inline-block;animation:pulse-anim 1.5s infinite;"></span>Match Found</div>
              <div class="result-song">{best['song_name']}</div>
              <div><span class="result-badge">{best['score']:,} aligned hashes</span></div>
              <div style="margin-top:12px;font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:#27272a;">
                {len(R['hs']):,} query hashes · {len(fr):,} spectral peaks
              </div>
            </div>""", unsafe_allow_html=True)

            # Step 1 — Waveform & Spectrogram
            st.markdown('<div class="step-hdr"><div class="step-n">01</div><div class="step-t">Waveform & Spectrogram</div><div class="step-s">time → frequency</div></div>', unsafe_allow_html=True)
            c1,c2 = st.columns(2)
            with c1:
                fig,ax=plt.subplots(figsize=(8,3))
                tax=np.linspace(0,len(y)/recognizer.sr,len(y))
                ax.plot(tax,y,color='#6366f1',linewidth=0.5,alpha=0.9)
                ax.fill_between(tax,y,alpha=0.1,color='#818cf8')
                ax.set_title("Waveform",fontsize=9,color='#71717a',pad=6)
                ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
            with c2:
                fig,ax=plt.subplots(figsize=(8,3))
                im=ax.imshow(spec,aspect='auto',origin='lower',cmap='magma',vmin=spec.max()-80,vmax=spec.max())
                fig.colorbar(im,ax=ax,format='%+.0fdB',shrink=0.75)
                ax.set_title("Log-Magnitude Spectrogram",fontsize=9,color='#71717a',pad=6)
                ax.set_xlabel("Time Frames"); ax.set_ylabel("Freq Bins")
                fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

            # Step 2 — Constellation Map
            st.markdown('<div class="step-hdr"><div class="step-n">02</div><div class="step-t">Constellation Map</div><div class="step-s">spectral peaks</div></div>', unsafe_allow_html=True)
            fig,ax=plt.subplots(figsize=(14,3.5))
            ax.imshow(spec,aspect='auto',origin='lower',cmap='magma',alpha=0.45,vmin=spec.max()-80,vmax=spec.max())
            ax.scatter(ti,fr,c='#a78bfa',s=2.5,alpha=0.9,linewidths=0)
            ax.set_title(f"{len(fr):,} spectral peaks extracted",fontsize=9,color='#71717a',pad=6)
            ax.set_xlabel("Time Frames"); ax.set_ylabel("Freq Bins")
            ax.set_xlim(0,spec.shape[1]); ax.set_ylim(0,spec.shape[0])
            fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

            # Step 3 — Offset Histogram
            st.markdown('<div class="step-hdr"><div class="step-n">03</div><div class="step-t">Alignment Histogram</div><div class="step-s">proof of identity</div></div>', unsafe_allow_html=True)
            hist = best['histogram']
            offs = list(hist.keys()); cnts = list(hist.values())
            moff = max(hist, key=hist.get); mcnt = hist[moff]
            fig,ax=plt.subplots(figsize=(14,3.5))
            ax.vlines(offs,0,cnts,color='#27272a',linewidth=1.2)
            ax.vlines(moff,0,mcnt,color='#818cf8',linewidth=5)
            ax.fill_between([moff-30,moff+30],0,mcnt,color='#818cf8',alpha=0.08)
            xr = max(offs)-min(offs) if len(offs)>1 else 1
            tx = moff-xr*0.15 if moff>(min(offs)+xr*0.5) else moff+xr*0.04
            ax.annotate(f"{mcnt:,} hashes",xy=(moff,mcnt),xytext=(tx,mcnt*0.72),
                color='#818cf8',fontsize=9,
                arrowprops=dict(arrowstyle='->',color='#818cf8',lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3',fc='#111113',ec='#818cf8',alpha=0.95))
            ax.set_ylim(0,mcnt*1.3)
            ax.set_xlabel("Time Offset (db_t − query_t)"); ax.set_ylabel("Hash votes")
            ax.set_title(f"Offset histogram — {best['song_name']}",fontsize=9,color='#71717a',pad=6)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

            # Step 4 — Candidates
            st.markdown('<div class="step-hdr"><div class="step-n">04</div><div class="step-t">Top Candidates</div><div class="step-s">ranked by score</div></div>', unsafe_allow_html=True)
            top5 = res[:5]; mx5 = top5[0]['score']
            rows=""
            for i,r in enumerate(top5,1):
                pct = int(r['score']/mx5*100) if mx5>0 else 0
                gold = i==1
                nc = "color:#818cf8;" if gold else ""
                bc = "#6366f1" if gold else "#27272a"
                rows += f"""<div class="cand-row">
                  <div class="cand-rank">{'★' if gold else f'#{i}'}</div>
                  <div class="cand-name" style="{nc}">{r['song_name']}</div>
                  <div class="cand-bar-wrap"><div class="cand-bar" style="width:{pct}%;background:{bc};"></div></div>
                  <div class="cand-score">{r['score']:,}</div>
                </div>"""
            st.markdown(f'<div class="cand-wrap">{rows}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — BATCH TEST
# ══════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown('<div class="sec-label">Bulk identification & accuracy benchmarking</div>', unsafe_allow_html=True)

    # Sample data section
    with st.expander("🎲  Generate sample test clips (no upload needed)"):
        st.markdown("Pick songs and clip settings to auto-generate a test set.", unsafe_allow_html=False)
        ncols_sample = st.columns(3)
        with ncols_sample[0]:
            n_samples = st.slider("Number of clips", 1, 10, 5)
        with ncols_sample[1]:
            sample_dur = st.slider("Clip duration (s)", 3, 15, 8)
        with ncols_sample[2]:
            sample_start = st.slider("Start offset (s)", 5, 60, 20)

        if st.button("Generate & Download test clips", key="gen_batch_samples"):
            import librosa, soundfile as sf, zipfile
            zip_buf = io.BytesIO()
            song_list = list(db.song_names.items())[:n_samples]
            with zipfile.ZipFile(zip_buf, 'w') as zf:
                for sid, name in song_list:
                    mp3p = os.path.join("EE200_course_project_data_2026","Q3_database",f"{name}.mp3")
                    if os.path.exists(mp3p):
                        try:
                            y_tmp, sr_tmp = librosa.load(mp3p, sr=None, offset=float(sample_start), duration=float(sample_dur))
                            wb = io.BytesIO()
                            sf.write(wb, y_tmp, sr_tmp, format="WAV")
                            zf.writestr(f"{name}.wav", wb.getvalue())
                        except: pass
            zip_buf.seek(0)
            st.download_button("⬇  Download sample_clips.zip", data=zip_buf.getvalue(),
                               file_name="sample_clips.zip", mime="application/zip")
            st.info("Extract the ZIP and upload all WAV files below to test batch identification!")

    ups = st.file_uploader("Upload clips", type=["mp3","wav"], accept_multiple_files=True, label_visibility="collapsed")

    if ups:
        st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;color:#52525b;margin-bottom:12px;">{len(ups)} clips queued</div>', unsafe_allow_html=True)

        if "batch_df" not in st.session_state:
            st.session_state.batch_df = None

        if st.button("⚡  Run Batch", key="batch_btn"):
            prog = st.progress(0)
            stat = st.empty()
            rows = []
            for i,f in enumerate(ups):
                stat.markdown(f"Identifying **{f.name}**… ({i+1}/{len(ups)})")
                pred="ERROR"; tmp=None
                try:
                    with tempfile.NamedTemporaryFile(delete=False,suffix=".mp3") as tf:
                        tf.write(f.getvalue()); tmp=tf.name
                    sp,_=recognizer.get_spectrogram(tmp)
                    fr2,ti2=recognizer.extract_peaks(sp,percentile=90)
                    hs2=recognizer.generate_hashes(fr2,ti2)
                    ms=db.match_hashes(hs2)
                    pred=ms[0]['song_name'] if (ms and ms[0]['score']>=15) else "NO MATCH"
                except: pred="ERROR"
                finally:
                    if tmp and os.path.exists(tmp): os.unlink(tmp)
                base=os.path.splitext(f.name)[0]
                def nm(s): return re.sub(r'[^a-z0-9]','',s.lower())
                rows.append({"filename":f.name,"prediction":pred,"correct":nm(base)==nm(pred)})
                prog.progress((i+1)/len(ups))
            stat.markdown("**Done!**")
            st.session_state.batch_df = pd.DataFrame(rows)

        if st.session_state.batch_df is not None:
            df = st.session_state.batch_df
            correct = int(df['correct'].sum())
            acc = int(correct/len(df)*100)
            acc_col = "#818cf8" if acc==100 else ("#f59e0b" if acc>=70 else "#f87171")

            bc1,bc2,bc3 = st.columns(3)
            for col,val,lbl in [(bc1,str(len(df)),"Clips"),(bc2,str(correct),"Correct"),(bc3,f"{acc}%","Accuracy")]:
                with col:
                    col_style = f"color:{acc_col};" if lbl=="Accuracy" else ""
                    st.markdown(f'<div class="stat-box"><div class="stat-box-val" style="{col_style}">{val}</div><div class="stat-box-key">{lbl}</div></div>', unsafe_allow_html=True)

            st.markdown('<br><div class="sec-label">Results</div>', unsafe_allow_html=True)
            rows_html=""
            for _,row in df.iterrows():
                badge=f'<span class="badge-ok">✓ MATCH</span>' if row['correct'] else f'<span class="badge-fail">✗ WRONG</span>'
                rows_html+=f"""<div class="batch-row">
                  <div style="flex:1;font-size:0.8rem;font-weight:600;color:#d4d4d8;font-family:'JetBrains Mono',monospace;">{row['filename']}</div>
                  <div style="flex:1;font-size:0.8rem;color:#52525b;">{row['prediction']}</div>
                  {badge}
                </div>"""
            st.markdown(f'<div class="batch-table-wrap">{rows_html}</div>', unsafe_allow_html=True)

            csv = df[['filename','prediction']].to_csv(index=False)
            st.download_button("⬇  results.csv", data=csv, file_name="results.csv", mime="text/csv")

# ══════════════════════════════════════════════════════════════════════
# TAB 4 — HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════
with tab_how:
    st.markdown('<div class="sec-label">Technical deep-dive</div>', unsafe_allow_html=True)

    hw_items = [
        ("🌀","STFT Spectrogram","Convert audio samples into a 2D log-magnitude heatmap using Short-Time Fourier Transform. Window size n_fft=2048 gives 46ms time resolution at 11kHz."),
        ("⭐","Constellation Map","Apply a 15×15 local maximum filter to extract the strongest spectral peaks. These sparse 'stars' capture harmonic structure using less than 1% of spectrogram cells."),
        ("🔗","Hash Pairing","Each anchor peak is paired with up to 20 nearby target peaks within a time-frequency fan zone. Each pair yields a hash key (f₁, f₂, Δt) — highly specific and noise-robust."),
        ("🗄️","Hash Database","Store millions of (hash → song_id, offset) entries in a Python dict for O(1) retrieval. This DB holds {:,} unique keys across {} songs.".format(total_hashes,total_songs)),
        ("📐","Offset Histogram","For each query hash that matches the DB, record a vote for (song_id, db_offset − query_offset). A genuine match produces a statistically impossible spike at one offset."),
        ("🛡️","Robustness","The hash structure encodes relative frequencies and relative time. MP3 compression, background noise, and cropping preserve these relative relationships — making the system robust."),
    ]
    cols3 = st.columns(3)
    for i,(icon,title,desc) in enumerate(hw_items):
        with cols3[i%3]:
            st.markdown(f"""<div class="hw-card">
              <div class="hw-icon">{icon}</div>
              <div class="hw-title">{title}</div>
              <div class="hw-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<br><div class="sec-label">The Hash Function</div>', unsafe_allow_html=True)
    col_hf, col_why = st.columns(2)
    with col_hf:
        st.markdown("""
        <div style="background:#111113;border:1px solid #1c1c1f;border-radius:10px;padding:20px;
                    font-family:'JetBrains Mono',monospace;font-size:0.75rem;line-height:2.1;color:#52525b;">
          <span style="color:#3f3f46;"># anchor peak at (f₁, t₁)</span><br>
          <span style="color:#3f3f46;"># target peak at (f₂, t₂) in fan zone</span><br><br>
          <span style="color:#818cf8;">key</span> <span style="color:#e4e4e7;">=</span>
          <span style="color:#a78bfa;">(f₁, f₂, t₂−t₁)</span><br>
          <span style="color:#818cf8;">val</span> <span style="color:#e4e4e7;">=</span>
          <span style="color:#38bdf8;">(song_id, t₁)</span><br><br>
          <span style="color:#3f3f46;"># during matching:</span><br>
          <span style="color:#818cf8;">offset</span> <span style="color:#e4e4e7;">=</span>
          <span style="color:#38bdf8;">db_t₁</span> <span style="color:#e4e4e7;">−</span>
          <span style="color:#a78bfa;">query_t₁</span>
        </div>""", unsafe_allow_html=True)

    with col_why:
        st.markdown("""
        <div style="background:#111113;border:1px solid #1c1c1f;border-radius:10px;padding:20px;
                    font-size:0.78rem;line-height:1.85;color:#52525b;">
          A single frequency like <span style="color:#818cf8;">440 Hz</span> appears in thousands of songs.<br><br>
          But the triplet <span style="color:#a78bfa;">(440 Hz, 880 Hz, Δt=3.2s)</span> is nearly unique to one recording.<br><br>
          When <span style="color:#e4e4e7;">thousands of such triplets</span> all agree on a single time offset,
          the probability of a false positive approaches <span style="color:#f87171;">zero</span> —
          providing cryptographic-grade audio identification.
        </div>""", unsafe_allow_html=True)

    st.markdown('<br><div class="sec-label">Complexity</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="cand-wrap">
      <div class="cand-row" style="background:#111113;">
        <div class="cand-rank" style="color:#3f3f46;font-size:0.6rem;">OP</div>
        <div class="cand-name" style="color:#3f3f46;font-family:'JetBrains Mono',monospace;font-size:0.6rem;text-transform:uppercase;">Operation</div>
        <div style="flex:0.6;font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:#3f3f46;text-transform:uppercase;">Time</div>
        <div class="cand-score" style="background:transparent;border:none;color:#3f3f46;font-size:0.6rem;text-transform:uppercase;">Space</div>
      </div>
      <div class="cand-row"><div class="cand-rank">①</div><div class="cand-name">STFT Spectrogram</div><div style="flex:0.6;font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#818cf8;">O(N log N)</div><div class="cand-score">O(N·F)</div></div>
      <div class="cand-row"><div class="cand-rank">②</div><div class="cand-name">Peak Extraction</div><div style="flex:0.6;font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#818cf8;">O(N·F·w²)</div><div class="cand-score">O(P)</div></div>
      <div class="cand-row"><div class="cand-rank">③</div><div class="cand-name">Hash Generation</div><div style="flex:0.6;font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#818cf8;">O(P·fan)</div><div class="cand-score">O(H)</div></div>
      <div class="cand-row"><div class="cand-rank">④</div><div class="cand-name">DB Lookup (per hash)</div><div style="flex:0.6;font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#a78bfa;font-weight:700;">O(1)</div><div class="cand-score">O(S·H)</div></div>
    </div>
    """, unsafe_allow_html=True)
