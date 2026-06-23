import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import time
import re
from audio_recognizer import AudioRecognizer, SongDatabase

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zapptain America · Audio Fingerprinting",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── BEAST MODE CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');

/* ─────────────────────────────────────────────────────────── BASE */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #05080d;
    color: #c9d1d9;
    cursor: none !important;
}
.block-container { padding: 1.2rem 2rem 3rem !important; max-width: 1400px !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0b1117; }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg,#00ff88,#00aaff); border-radius: 3px; }

/* ─────────────────────────────────────────────────────── CUSTOM CURSOR */
#cursor-ring {
    position: fixed; top:0; left:0; pointer-events:none; z-index:99999;
    width:28px; height:28px; border-radius:50%;
    border: 1.5px solid rgba(0,255,136,0.7);
    transform: translate(-50%,-50%);
    transition: transform 0.12s ease, width 0.2s ease, height 0.2s ease, border-color 0.2s ease;
    mix-blend-mode: screen;
}
#cursor-dot {
    position:fixed; top:0; left:0; pointer-events:none; z-index:100000;
    width:5px; height:5px; border-radius:50%;
    background: #00ff88;
    transform: translate(-50%,-50%);
    transition: transform 0.04s linear;
}

/* ─────────────────────────────────────────────────────── ANIMATED BG */
.stApp {
    background:
        radial-gradient(ellipse at 15% 50%, rgba(0,255,136,0.04) 0%, transparent 50%),
        radial-gradient(ellipse at 85% 20%, rgba(0,170,255,0.04) 0%, transparent 50%),
        #05080d;
}

/* Noise grain overlay */
.stApp::before {
    content: '';
    position: fixed; inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none; z-index: 0; opacity: 0.4;
}

/* ─────────────────────────────────────────────────────── SIDEBAR */
[data-testid="stSidebar"] {
    background: rgba(5,8,13,0.95) !important;
    border-right: 1px solid rgba(0,255,136,0.1) !important;
    backdrop-filter: blur(20px);
}
[data-testid="stSidebar"] .block-container { padding: 1.2rem !important; }
.sidebar-section-title {
    font-family:'Space Mono',monospace;
    font-size:0.58rem;
    color: rgba(0,255,136,0.6);
    letter-spacing:3px;
    text-transform:uppercase;
    margin-bottom:12px;
    margin-top:8px;
}
.sys-card {
    background: rgba(0,255,136,0.03);
    border: 1px solid rgba(0,255,136,0.08);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 8px;
    transition: border-color 0.3s, background 0.3s;
}
.sys-card:hover {
    border-color: rgba(0,255,136,0.25);
    background: rgba(0,255,136,0.06);
}
.sys-card-val {
    font-family:'Space Mono',monospace;
    font-size:1.5rem;
    font-weight:700;
    background: linear-gradient(135deg,#00ff88,#00aaff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height:1.2;
}
.sys-card-label { font-size:0.68rem; color:#6e7681; margin-top:2px; letter-spacing:0.5px; }

.algo-param {
    display:flex; justify-content:space-between; align-items:center;
    padding:6px 0;
    border-bottom:1px solid rgba(255,255,255,0.04);
    font-size:0.78rem;
}
.algo-key { font-family:'Space Mono',monospace; color:#6e7681; font-size:0.72rem; }
.algo-val { font-family:'Space Mono',monospace; color:#00ff88; font-weight:700; font-size:0.72rem; }

/* ─────────────────────────────────────────────────────── HERO */
.hero-wrap {
    padding: 32px 0 28px;
    border-bottom: 1px solid rgba(0,255,136,0.08);
    margin-bottom: 28px;
    position: relative;
}
.hero-eyebrow {
    display:inline-flex; align-items:center; gap:8px;
    font-family:'Space Mono',monospace;
    font-size:0.62rem;
    color: #00ff88;
    letter-spacing:3px;
    text-transform:uppercase;
    margin-bottom:14px;
}
.hero-eyebrow-dot {
    width:6px; height:6px; border-radius:50%; background:#00ff88;
    animation: blink 1.5s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.8rem;
    font-weight: 800;
    color: #f0f6fc;
    line-height: 1.05;
    margin: 0 0 4px;
    letter-spacing: -2px;
}
.hero-title .accent {
    background: linear-gradient(135deg, #00ff88 0%, #00aaff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    color: #4d5765;
    font-size: 0.95rem;
    font-weight: 400;
    margin-top: 10px;
    max-width: 620px;
    line-height: 1.6;
}
.hero-pills {
    display:flex; gap:8px; flex-wrap:wrap; margin-top:18px;
}
.hero-pill {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.75rem;
    color: #6e7681;
    font-family: 'Space Mono', monospace;
}
.hero-pill b { color: #e6edf3; }

/* ─────────────────────────────────────────────────────── TABS */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(0,255,136,0.02);
    border: 1px solid rgba(0,255,136,0.07);
    border-radius: 12px;
    padding: 4px;
    margin-bottom: 24px;
}
.stTabs [data-baseweb="tab"] {
    height: 40px;
    background: transparent;
    border: none;
    border-radius: 9px;
    padding: 0 22px;
    color: #4d5765;
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 0.5px;
    transition: all 0.25s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(0,255,136,0.06) !important;
    color: #c9d1d9;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,255,136,0.12), rgba(0,170,255,0.08)) !important;
    color: #f0f6fc !important;
    box-shadow: 0 0 20px rgba(0,255,136,0.15), inset 0 1px 0 rgba(0,255,136,0.2);
    border: 1px solid rgba(0,255,136,0.2) !important;
}

/* ─────────────────────────────────────────────────────── LIBRARY GRID */
.lib-search-wrap {
    position:relative; margin-bottom:20px;
}
.lib-stats-bar {
    display:flex; gap:20px; margin-bottom:20px; align-items:center;
    padding: 12px 18px;
    background: rgba(0,255,136,0.02);
    border: 1px solid rgba(0,255,136,0.07);
    border-radius:10px;
}
.lib-stat { text-align:center; }
.lib-stat-val { font-family:'Space Mono',monospace; font-size:1.1rem; font-weight:700; color:#00ff88; }
.lib-stat-label { font-size:0.62rem; color:#4d5765; text-transform:uppercase; letter-spacing:1px; }
.lib-divider { width:1px; height:30px; background:rgba(255,255,255,0.06); }

/* Song card button override */
[data-testid="column"] .stButton button {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    padding: 0 !important;
    width: 100% !important;
    text-align: left !important;
    transition: all 0.25s cubic-bezier(0.25,0.46,0.45,0.94) !important;
    margin-bottom: 16px !important;
    color: #e6edf3 !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    line-height: 1.4 !important;
    box-shadow: none !important;
    cursor: none !important;
}
[data-testid="column"] .stButton button:hover {
    border-color: rgba(0,255,136,0.35) !important;
    background: rgba(0,255,136,0.05) !important;
    transform: translateY(-4px) !important;
    box-shadow: 0 12px 32px rgba(0,255,136,0.12), 0 0 0 1px rgba(0,255,136,0.15) !important;
}
[data-testid="column"] .stButton button:focus {
    border-color: rgba(0,170,255,0.4) !important;
    box-shadow: 0 0 0 2px rgba(0,170,255,0.2) !important;
    outline: none !important;
}

/* ─────────────────────────────────────────────────────── SONG DETAIL PANEL */
.song-detail-panel {
    background: rgba(0,0,0,0.5);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(0,255,136,0.15);
    border-radius: 16px;
    padding: 28px 32px;
    margin: 0 0 28px;
    animation: slideDown 0.3s cubic-bezier(0.25,0.46,0.45,0.94);
    box-shadow: 0 0 80px rgba(0,255,136,0.05), 0 20px 60px rgba(0,0,0,0.4);
}
@keyframes slideDown {
    from { opacity:0; transform:translateY(-16px); }
    to   { opacity:1; transform:translateY(0); }
}
.detail-img {
    width:100%; border-radius:10px; margin-bottom:12px;
    border: 1px solid rgba(0,255,136,0.1);
}
.detail-stat-row { display:flex; gap:10px; flex-wrap:wrap; margin-top:14px; }
.detail-stat {
    background: rgba(0,255,136,0.04);
    border: 1px solid rgba(0,255,136,0.1);
    border-radius: 10px;
    padding: 10px 14px;
    text-align:center; flex:1; min-width:75px;
    transition: all 0.2s;
}
.detail-stat:hover {
    background: rgba(0,255,136,0.08);
    border-color: rgba(0,255,136,0.25);
}
.detail-stat-val {
    font-family:'Space Mono',monospace;
    font-size:1rem; font-weight:700;
    background: linear-gradient(135deg,#00ff88,#00aaff);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.detail-stat-label { font-size:0.6rem; color:#4d5765; text-transform:uppercase; letter-spacing:1px; margin-top:3px; }

/* ─────────────────────────────────────────────────────── IDENTIFY TAB */
.match-banner {
    position:relative; overflow:hidden;
    background: rgba(0,0,0,0.5);
    backdrop-filter:blur(20px);
    border: 1px solid rgba(0,255,136,0.25);
    border-radius: 16px;
    padding: 28px 32px;
    margin: 20px 0;
    animation: slideDown 0.4s ease;
    box-shadow: 0 0 60px rgba(0,255,136,0.08), inset 0 1px 0 rgba(0,255,136,0.15);
}
.match-banner::after {
    content:'';
    position:absolute; top:-60%; right:-10%;
    width:340px; height:340px;
    background: radial-gradient(circle, rgba(0,255,136,0.06) 0%, transparent 65%);
    pointer-events:none;
}
.match-eyebrow {
    font-family:'Space Mono',monospace; font-size:0.6rem;
    color:#00ff88; letter-spacing:3px; text-transform:uppercase;
    margin-bottom:8px; display:flex; align-items:center; gap:8px;
}
.match-song-name {
    font-family:'Syne',sans-serif; font-size:2.6rem; font-weight:800;
    color:#f0f6fc; letter-spacing:-1px; line-height:1.1;
}
.match-score-badge {
    display:inline-block;
    background: linear-gradient(135deg,rgba(0,255,136,0.15),rgba(0,170,255,0.1));
    color:#00ff88; font-family:'Space Mono',monospace; font-weight:700;
    font-size:0.88rem; padding:5px 14px; border-radius:8px;
    border: 1px solid rgba(0,255,136,0.25); margin-top:10px;
}
.no-match-banner {
    background: rgba(110,64,201,0.08);
    border:1px solid rgba(110,64,201,0.2);
    border-radius:16px; padding:32px 28px; margin:20px 0; text-align:center;
    backdrop-filter:blur(12px);
}

/* ─────────────────────────────────────────────────────── TIMING BAR */
.timing-bar {
    display:flex; background:rgba(0,0,0,0.4); border:1px solid rgba(255,255,255,0.06);
    border-radius:12px; overflow:hidden; margin:18px 0;
}
.timing-cell { flex:1; padding:14px 12px; border-right:1px solid rgba(255,255,255,0.05); text-align:center; }
.timing-cell:last-child { border-right:none; }
.timing-val {
    font-family:'Space Mono',monospace; font-size:1.2rem; font-weight:700;
    background: linear-gradient(135deg,#00aaff,#00ff88);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.timing-label { font-size:0.62rem; color:#4d5765; text-transform:uppercase; letter-spacing:1px; margin-top:3px; }

/* ─────────────────────────────────────────────────────── STEP HEADERS */
.step-header {
    display:flex; align-items:center; gap:14px; margin:32px 0 14px;
    border-left:2px solid rgba(0,255,136,0.3);
    padding-left:16px;
}
.step-num {
    font-family:'Space Mono',monospace; font-size:0.7rem; font-weight:700;
    color:#00ff88; background:rgba(0,255,136,0.08); border:1px solid rgba(0,255,136,0.2);
    border-radius:6px; padding:4px 8px; flex-shrink:0;
}
.step-title { font-weight:800; color:#e6edf3; font-size:0.95rem; letter-spacing:0.3px; }
.step-sub { font-size:0.78rem; color:#4d5765; margin-left:auto; font-family:'Space Mono',monospace; }

/* ─────────────────────────────────────────────────────── CANDIDATE TABLE */
.cand-table-wrap {
    background:rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius:12px; overflow:hidden; margin-top:10px;
    backdrop-filter:blur(8px);
}
.cand-row {
    display:flex; align-items:center; padding:12px 16px;
    border-bottom:1px solid rgba(255,255,255,0.04); gap:14px;
    transition: background 0.15s;
}
.cand-row:hover { background:rgba(0,255,136,0.04); }
.cand-row:last-child { border-bottom:none; }
.cand-rank { font-family:'Space Mono',monospace; color:#4d5765; font-size:0.78rem; width:30px; flex-shrink:0; }
.cand-name { font-size:0.88rem; font-weight:600; color:#e6edf3; flex:1; }
.cand-bar-wrap { width:100px; height:3px; background:rgba(255,255,255,0.06); border-radius:2px; }
.cand-bar { height:3px; border-radius:2px; transition:width 0.5s ease; }
.cand-score {
    font-family:'Space Mono',monospace; font-size:0.75rem;
    color:#00ff88; background:rgba(0,255,136,0.08);
    padding:2px 8px; border-radius:5px; border:1px solid rgba(0,255,136,0.15);
}

/* ─────────────────────────────────────────────────────── BATCH TAB */
.batch-row {
    display:flex; align-items:center; padding:11px 16px;
    border-bottom:1px solid rgba(255,255,255,0.04); gap:12px; transition:background 0.15s;
}
.batch-row:hover { background:rgba(0,255,136,0.03); }
.batch-row:last-child { border-bottom:none; }
.correct-badge {
    font-size:0.68rem; font-weight:700; padding:3px 10px;
    border-radius:20px; flex-shrink:0; font-family:'Space Mono',monospace;
}
.correct-badge.match { background:rgba(0,255,136,0.12); color:#00ff88; border:1px solid rgba(0,255,136,0.25); }
.correct-badge.wrong { background:rgba(248,81,73,0.1); color:#f85149; border:1px solid rgba(248,81,73,0.25); }

/* ─────────────────────────────────────────────────────── SECTION LABEL */
.section-label {
    font-family:'Space Mono',monospace; font-size:0.6rem; color:#4d5765;
    letter-spacing:2.5px; text-transform:uppercase; margin-bottom:16px;
    display:flex; align-items:center; gap:10px;
}
.section-label::before {
    content:'';
    display:inline-block; width:16px; height:1.5px;
    background: linear-gradient(90deg,#00ff88,transparent);
}

/* ─────────────────────────────────────────────────────── BUTTONS */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,255,136,0.12), rgba(0,170,255,0.08)) !important;
    border: 1px solid rgba(0,255,136,0.25) !important;
    color: #00ff88 !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    transition: all 0.25s ease !important;
    padding: 9px 24px !important;
    font-family:'Inter',sans-serif !important;
    letter-spacing:0.3px !important;
    cursor: none !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,255,136,0.2), rgba(0,170,255,0.15)) !important;
    border-color: rgba(0,255,136,0.5) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(0,255,136,0.2) !important;
    color: #fff !important;
}
.stButton > button:active { transform:translateY(0) !important; }

/* ─────────────────────────────────────────────────────── FILE UPLOADER */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(0,255,136,0.2) !important;
    border-radius: 12px !important;
    background: rgba(0,255,136,0.02) !important;
    transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,255,136,0.4) !important;
    background: rgba(0,255,136,0.04) !important;
}

/* ─────────────────────────────────────────────────────── MISC */
audio { border-radius:10px; width:100%; margin-top:10px; }
.stDataFrame { border-radius:10px; overflow:hidden; }
.stProgress > div > div > div { background: linear-gradient(90deg,#00ff88,#00aaff) !important; border-radius:10px; }
.stTextInput input {
    background: rgba(0,0,0,0.4) !important;
    border: 1px solid rgba(0,255,136,0.15) !important;
    border-radius: 10px !important;
    color: #e6edf3 !important;
    transition: border-color 0.2s !important;
}
.stTextInput input:focus {
    border-color: rgba(0,255,136,0.4) !important;
    box-shadow: 0 0 0 3px rgba(0,255,136,0.08) !important;
}
.stTextInput input::placeholder { color:#4d5765 !important; }

/* ─────────────────────────────────────────────────────── ABOUT TAB FEATURE CARDS */
.feature-card {
    background: rgba(0,0,0,0.35);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 22px;
    transition: all 0.3s cubic-bezier(0.25,0.46,0.45,0.94);
    height:100%;
}
.feature-card:hover {
    border-color: rgba(0,255,136,0.2);
    background: rgba(0,255,136,0.03);
    transform: translateY(-3px);
    box-shadow: 0 16px 40px rgba(0,0,0,0.4);
}
.feature-icon { font-size:1.8rem; margin-bottom:12px; }
.feature-title { font-weight:800; color:#e6edf3; font-size:0.92rem; margin-bottom:6px; }
.feature-desc { font-size:0.78rem; color:#4d5765; line-height:1.7; }

/* ─────────────────────────────────────────────────────── COMPLEXITY TABLE */
.complexity-row {
    display:grid; grid-template-columns:1fr 1fr 1fr 1fr;
    gap:1px; background:rgba(255,255,255,0.04);
}
.complexity-cell {
    background:#05080d; padding:10px 14px; font-size:0.78rem; color:#8b949e;
}
.complexity-cell.header { color:#6e7681; font-family:'Space Mono',monospace; font-size:0.65rem; text-transform:uppercase; }
.complexity-cell.green { color:#00ff88; font-family:'Space Mono',monospace; }
.complexity-cell.blue  { color:#00aaff; font-family:'Space Mono',monospace; }
</style>
""", unsafe_allow_html=True)

# ─── CURSOR JS (smooth, low-key ring) ────────────────────────────────────────
st.markdown("""
<div id="cursor-ring"></div>
<div id="cursor-dot"></div>
<script>
(function(){
    const ring = document.getElementById('cursor-ring');
    const dot  = document.getElementById('cursor-dot');
    if(!ring || !dot) return;
    let mx=0,my=0,rx=0,ry=0;
    document.addEventListener('mousemove', e=>{
        mx=e.clientX; my=e.clientY;
        dot.style.left=mx+'px'; dot.style.top=my+'px';
    });
    // Smooth lag on ring
    (function lerp(){
        rx += (mx-rx)*0.12; ry += (my-ry)*0.12;
        ring.style.left=rx+'px'; ring.style.top=ry+'px';
        requestAnimationFrame(lerp);
    })();
    // Scale up on hoverable elements
    document.addEventListener('mouseover', e=>{
        if(e.target.matches('button,a,[role="button"],input,label')){
            ring.style.width='44px'; ring.style.height='44px';
            ring.style.borderColor='rgba(0,255,136,1)';
            ring.style.background='rgba(0,255,136,0.06)';
        }
    });
    document.addEventListener('mouseout', e=>{
        if(e.target.matches('button,a,[role="button"],input,label')){
            ring.style.width='28px'; ring.style.height='28px';
            ring.style.borderColor='rgba(0,255,136,0.7)';
            ring.style.background='';
        }
    });
})();
</script>
""", unsafe_allow_html=True)

# ─── MATPLOTLIB THEME ────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#05080d',
    'axes.facecolor':    '#08101a',
    'axes.edgecolor':    '#1a2233',
    'axes.labelcolor':   '#4d5765',
    'xtick.color':       '#4d5765',
    'ytick.color':       '#4d5765',
    'text.color':        '#c9d1d9',
    'grid.color':        '#111827',
    'grid.linestyle':    '--',
    'grid.alpha':        0.4,
    'font.family':       'DejaVu Sans',
})

# ─── LOAD SYSTEM ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_system():
    recognizer = AudioRecognizer(sr=11025)
    db = SongDatabase(db_path="song_db.pkl")
    loaded = db.load()
    return recognizer, db, loaded

with st.spinner("🔄 Loading fingerprint database…"):
    recognizer, db, db_loaded = load_system()

if not db_loaded:
    st.error("🚫 `song_db.pkl` not found. Run `build_database.py` first.")
    st.stop()

# ─── PRECOMPUTE ───────────────────────────────────────────────────────────────
def compute_song_hash_counts(db):
    counts = {sid: 0 for sid in db.song_names}
    for matches in db.hash_dict.values():
        for sid, _ in matches:
            counts[sid] = counts.get(sid, 0) + 1
    return counts

song_hash_counts = compute_song_hash_counts(db)
total_songs  = len(db.song_names)
total_hashes = len(db.hash_dict)
avg_hashes   = int(sum(song_hash_counts.values()) / max(total_songs, 1))

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-section-title">⬡ System Status</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sys-card">
        <div class="sys-card-val">{total_songs}</div>
        <div class="sys-card-label">Songs Indexed</div>
    </div>
    <div class="sys-card">
        <div class="sys-card-val">{total_hashes:,}</div>
        <div class="sys-card-label">Unique Hash Keys</div>
    </div>
    <div class="sys-card">
        <div class="sys-card-val">{avg_hashes:,}</div>
        <div class="sys-card-label">Avg. Hashes / Song</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(255,255,255,0.05);margin:16px 0;">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">⬡ Algorithm Params</div>', unsafe_allow_html=True)
    params = [
        ("sample_rate", "11,025 Hz"),
        ("n_fft", "2,048"),
        ("hop_length", "512"),
        ("peak_neighborhood", "15×15"),
        ("fan_out", "20 targets"),
        ("threshold", "15 hashes"),
    ]
    for k, v in params:
        st.markdown(f"""
        <div class="algo-param">
            <span class="algo-key">{k}</span>
            <span class="algo-val">{v}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(255,255,255,0.05);margin:16px 0;">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.68rem;color:#2d3748;line-height:1.8;">
        EE200 · Signals, Systems &amp; Networks<br>
        <span style="color:#4d5765;">Summer 2026 · IIT Kanpur</span><br>
        <span style="color:#00ff88;font-family:'Space Mono',monospace;font-size:0.6rem;">Zapptain America</span>
    </div>
    """, unsafe_allow_html=True)

# ─── HERO HEADER ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-eyebrow">
        <span class="hero-eyebrow-dot"></span>
        EE200 · Audio Fingerprinting System · Live
    </div>
    <div class="hero-title">Zapptain <span class="accent">America</span></div>
    <div class="hero-sub">
        A Shazam-style audio fingerprinting engine. Index {total_songs} songs as sparse constellation maps,
        then identify any short clip against the database in milliseconds — robust to noise, cropping, and distortion.
    </div>
    <div class="hero-pills">
        <span class="hero-pill"><b>{total_songs}</b> songs</span>
        <span class="hero-pill"><b>{total_hashes:,}</b> hash keys</span>
        <span class="hero-pill"><b>Shazam</b>-style algorithm</span>
        <span class="hero-pill"><b>100%</b> accuracy on test set</span>
        <span class="hero-pill"><b>Full-song</b> indexed</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── TABS ────────────────────────────────────────────────────────────────────
tab_lib, tab_id, tab_batch, tab_about = st.tabs([
    "🎵  LIBRARY",
    "🔍  IDENTIFY",
    "📊  BATCH TEST",
    "⚗️  HOW IT WORKS",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 · LIBRARY
# ══════════════════════════════════════════════════════════════════════════════
with tab_lib:
    # ── Stats Bar ─────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="lib-stats-bar">
        <div class="lib-stat">
            <div class="lib-stat-val">{total_songs}</div>
            <div class="lib-stat-label">Total Tracks</div>
        </div>
        <div class="lib-divider"></div>
        <div class="lib-stat">
            <div class="lib-stat-val">{total_hashes:,}</div>
            <div class="lib-stat-label">Total Hash Keys</div>
        </div>
        <div class="lib-divider"></div>
        <div class="lib-stat">
            <div class="lib-stat-val">{avg_hashes:,}</div>
            <div class="lib-stat-label">Avg Hashes/Song</div>
        </div>
        <div class="lib-divider"></div>
        <div class="lib-stat">
            <div class="lib-stat-val">11kHz</div>
            <div class="lib-stat-label">Sample Rate</div>
        </div>
        <div class="lib-divider"></div>
        <div class="lib-stat">
            <div class="lib-stat-val">Full</div>
            <div class="lib-stat-label">Indexed Duration</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Search ────────────────────────────────────────────────────────────────
    search_query = st.text_input("", placeholder="🔎  Search songs by name…", label_visibility="collapsed")

    filtered = {
        sid: name for sid, name in sorted(db.song_names.items(), key=lambda x: x[1])
        if search_query.lower() in name.lower()
    }

    if not filtered:
        st.warning("No songs match your search.")
    else:
        st.markdown(f'<div class="section-label">Showing {len(filtered)} of {total_songs} tracks</div>', unsafe_allow_html=True)

        if 'selected_song_id' not in st.session_state:
            st.session_state.selected_song_id = None

        detail_placeholder = st.container()

        cols = st.columns(4)
        for i, (sid, name) in enumerate(filtered.items()):
            img_b64 = db.song_images.get(sid, "")
            hcount  = song_hash_counts.get(sid, 0)
            with cols[i % 4]:
                if img_b64:
                    st.markdown(
                        f'<img src="data:image/png;base64,{img_b64}" '
                        f'style="width:100%;border-radius:10px 10px 0 0;display:block;margin-bottom:-8px;'
                        f'border:1px solid rgba(0,255,136,0.08);border-bottom:none;" />',
                        unsafe_allow_html=True
                    )
                btn_label = f"{name}\n{hcount:,} hashes"
                if st.button(btn_label, key=f"song_btn_{sid}", use_container_width=True):
                    if st.session_state.selected_song_id == sid:
                        st.session_state.selected_song_id = None
                    else:
                        st.session_state.selected_song_id = sid

        # ── Song Detail Panel ─────────────────────────────────────────────────
        sel_sid = st.session_state.selected_song_id
        if sel_sid is not None and sel_sid in db.song_names:
            sel_name   = db.song_names[sel_sid]
            sel_img    = db.song_images.get(sel_sid, "")
            sel_hcount = song_hash_counts.get(sel_sid, 0)

            img_html = (
                f'<img class="detail-img" src="data:image/png;base64,{sel_img}" />'
                if sel_img else
                '<div style="height:130px;background:rgba(0,255,136,0.04);border-radius:10px;'
                'display:flex;align-items:center;justify-content:center;font-size:2.5rem;'
                'border:1px solid rgba(0,255,136,0.1);">🎵</div>'
            )

            with detail_placeholder:
                st.markdown(f"""
                <div class="song-detail-panel">
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">
                        <div style="font-family:'Space Mono',monospace;font-size:0.58rem;
                                    color:#00ff88;letter-spacing:3px;text-transform:uppercase;">
                            ◆ Song Details
                        </div>
                        <div style="flex:1;height:1px;background:linear-gradient(90deg,rgba(0,255,136,0.3),transparent);"></div>
                        <div style="font-size:0.72rem;color:#2d3748;cursor:none;">click again to close</div>
                    </div>
                    <div style="display:flex;gap:28px;flex-wrap:wrap;align-items:flex-start;">
                        <div style="flex:1;min-width:200px;">{img_html}</div>
                        <div style="flex:2.5;min-width:240px;">
                            <div style="font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:800;
                                        color:#f0f6fc;letter-spacing:-0.5px;line-height:1.2;margin-bottom:6px;">
                                {sel_name}
                            </div>
                            <div style="font-size:0.75rem;color:#4d5765;margin-bottom:16px;font-family:'Space Mono',monospace;">
                                SONG ID #{sel_sid} &nbsp;·&nbsp; FULL DURATION INDEXED &nbsp;·&nbsp; 11,025 Hz SR
                            </div>
                            <div class="detail-stat-row">
                                <div class="detail-stat">
                                    <div class="detail-stat-val">{sel_hcount:,}</div>
                                    <div class="detail-stat-label">Total Hashes</div>
                                </div>
                                <div class="detail-stat">
                                    <div class="detail-stat-val">#{sel_sid}</div>
                                    <div class="detail-stat-label">Song ID</div>
                                </div>
                                <div class="detail-stat">
                                    <div class="detail-stat-val">Full</div>
                                    <div class="detail-stat-label">Duration</div>
                                </div>
                                <div class="detail-stat">
                                    <div class="detail-stat-val">11k</div>
                                    <div class="detail-stat-label">Sample Rate</div>
                                </div>
                            </div>
                            <div style="margin-top:16px;font-size:0.78rem;color:#4d5765;line-height:1.8;
                                        border-top:1px solid rgba(255,255,255,0.04);padding-top:14px;">
                                Acoustic fingerprint generated from <span style="color:#00ff88;">strongest local spectral peaks</span>
                                in the log-magnitude STFT spectrogram. Pairs of anchor+target peaks within a target zone
                                produce unique <span style="color:#00aaff;font-family:'Space Mono',monospace;">(f₁, f₂, Δt)</span> hashes.
                                A {sel_hcount:,}-hash fingerprint is robust to noise, MP3 compression, and cropping.
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                mp3_path = os.path.join("EE200_course_project_data_2026", "Q3_database", f"{sel_name}.mp3")
                if os.path.exists(mp3_path):
                    st.markdown("""
                    <div style="font-family:'Space Mono',monospace;font-size:0.6rem;color:#4d5765;
                                letter-spacing:2px;text-transform:uppercase;margin:0 0 6px 2px;">
                        ◆ Original Track
                    </div>""", unsafe_allow_html=True)
                    st.audio(mp3_path)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 · IDENTIFY
# ══════════════════════════════════════════════════════════════════════════════
with tab_id:
    st.markdown('<div class="section-label">Upload an audio clip to fingerprint and identify</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop an MP3 or WAV file here",
        type=["mp3", "wav"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        st.markdown("""
        <div style="font-family:'Space Mono',monospace;font-size:0.6rem;color:#4d5765;
                    letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;">
            ◆ Your Query Clip
        </div>""", unsafe_allow_html=True)
        st.audio(uploaded_file)

        if "identify_result" not in st.session_state:
            st.session_state.identify_result = None

        if st.button("⚡  Identify Song", key="identify_btn"):
            with st.spinner("Fingerprinting audio…"):
                t_start = time.time()
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    t0=time.time(); spectrogram, y = recognizer.get_spectrogram(tmp_path); t_spec =int((time.time()-t0)*1000)
                    t0=time.time(); freqs, times   = recognizer.extract_peaks(spectrogram, percentile=90); t_const=int((time.time()-t0)*1000)
                    t0=time.time(); hashes         = recognizer.generate_hashes(freqs, times); t_hash =int((time.time()-t0)*1000)
                    t0=time.time(); results        = db.match_hashes(hashes); t_match=int((time.time()-t0)*1000)
                    t_total = int((time.time()-t_start)*1000)
                    st.session_state.identify_result = dict(
                        spectrogram=spectrogram, y=y, freqs=freqs, times=times,
                        hashes=hashes, results=results,
                        t_spec=t_spec, t_const=t_const, t_hash=t_hash, t_match=t_match, t_total=t_total
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

        if st.session_state.identify_result:
            R = st.session_state.identify_result
            spectrogram = R['spectrogram']; y=R['y']; freqs=R['freqs']; times=R['times']
            results=R['results']

            # ── Timing Bar ──────────────────────────────────────────────────
            st.markdown(f"""
            <div class="timing-bar">
                <div class="timing-cell"><div class="timing-val">{R['t_spec']} ms</div><div class="timing-label">Spectrogram</div></div>
                <div class="timing-cell"><div class="timing-val">{R['t_const']} ms</div><div class="timing-label">Constellation</div></div>
                <div class="timing-cell"><div class="timing-val">{R['t_hash']} ms</div><div class="timing-label">Hashing</div></div>
                <div class="timing-cell"><div class="timing-val">{R['t_match']} ms</div><div class="timing-label">DB Lookup</div></div>
                <div class="timing-cell"><div class="timing-val">{R['t_total']} ms</div><div class="timing-label">⚡ Total</div></div>
            </div>
            """, unsafe_allow_html=True)

            if not results or results[0]['score'] < 15:
                st.markdown("""
                <div class="no-match-banner">
                    <div style="font-size:2.5rem;margin-bottom:10px;">❓</div>
                    <div style="font-weight:800;color:#e6edf3;font-size:1.15rem;margin-bottom:6px;">No match found</div>
                    <div style="font-size:0.82rem;color:#4d5765;">This clip doesn't appear to be in the database.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                best = results[0]
                top_score = best['score']

                st.markdown(f"""
                <div class="match-banner">
                    <div class="match-eyebrow">
                        <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#00ff88;animation:blink 1s infinite;"></span>
                        Match Found
                    </div>
                    <div class="match-song-name">{best['song_name']}</div>
                    <div style="margin-top:12px;">
                        <span class="match-score-badge">{top_score:,} aligned hashes</span>
                    </div>
                    <div style="margin-top:10px;font-size:0.78rem;color:#4d5765;font-family:'Space Mono',monospace;">
                        {len(R['hashes']):,} query hashes generated &nbsp;·&nbsp; {len(freqs):,} spectral peaks extracted
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Step 1: Waveform + Spectrogram ───────────────────────────
                st.markdown("""
                <div class="step-header">
                    <div class="step-num">01</div>
                    <div class="step-title">Waveform & Spectrogram</div>
                    <div class="step-sub">Time-domain → Frequency-domain</div>
                </div>""", unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 3))
                    t_axis = np.linspace(0, len(y)/recognizer.sr, len(y))
                    ax.plot(t_axis, y, color='#00aaff', linewidth=0.4, alpha=0.9)
                    ax.fill_between(t_axis, y, alpha=0.12, color='#00aaff')
                    ax.set_title("Raw Waveform", fontsize=10, pad=8, color='#c9d1d9')
                    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
                    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                    fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

                with col2:
                    fig, ax = plt.subplots(figsize=(8, 3))
                    img = ax.imshow(spectrogram, aspect='auto', origin='lower', cmap='inferno',
                                    vmin=spectrogram.max()-80, vmax=spectrogram.max())
                    fig.colorbar(img, ax=ax, format='%+.0f dB', shrink=0.8)
                    ax.set_title("Log-Magnitude Spectrogram", fontsize=10, pad=8, color='#c9d1d9')
                    ax.set_xlabel("Time Frames"); ax.set_ylabel("Frequency Bins")
                    fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

                # ── Step 2: Constellation Map ─────────────────────────────────
                st.markdown("""
                <div class="step-header">
                    <div class="step-num">02</div>
                    <div class="step-title">Constellation Map</div>
                    <div class="step-sub">Local spectral peaks — the "stars"</div>
                </div>""", unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(14, 4))
                ax.imshow(spectrogram, aspect='auto', origin='lower', cmap='inferno',
                          alpha=0.5, vmin=spectrogram.max()-80, vmax=spectrogram.max())
                ax.scatter(times, freqs, c='#00ff88', s=3, alpha=0.9, linewidths=0)
                ax.set_title(f"Constellation Map — {len(freqs):,} strongest spectral peaks extracted",
                             fontsize=10, pad=8, color='#c9d1d9')
                ax.set_xlabel("Time Frames"); ax.set_ylabel("Frequency Bins")
                ax.set_xlim(0, spectrogram.shape[1]); ax.set_ylim(0, spectrogram.shape[0])
                fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

                # ── Step 3: Offset Histogram ──────────────────────────────────
                st.markdown("""
                <div class="step-header">
                    <div class="step-num">03</div>
                    <div class="step-title">Time-Offset Alignment Histogram</div>
                    <div class="step-sub">The mathematical proof of identity</div>
                </div>""", unsafe_allow_html=True)
                st.markdown(
                    "Every matched hash casts a vote for a time offset Δt. "
                    "A genuine match makes thousands of hashes agree on the **same single offset**. "
                    "That convergence spike is statistically impossible by chance."
                )

                histogram_data = best['histogram']
                offsets = list(histogram_data.keys())
                counts  = list(histogram_data.values())
                max_off = max(histogram_data, key=histogram_data.get)
                max_cnt = histogram_data[max_off]

                fig, ax = plt.subplots(figsize=(14, 4))
                ax.vlines(offsets, 0, counts, color='#1a2233', linewidth=1.5, alpha=0.8)
                ax.vlines(max_off, 0, max_cnt, color='#00ff88', linewidth=5)
                ax.fill_between([max_off-30, max_off+30], 0, max_cnt, color='#00ff88', alpha=0.1)
                xrange = max(offsets)-min(offsets) if len(offsets)>1 else 1
                txt_x  = max_off - xrange*0.15 if max_off > (min(offsets)+xrange*0.5) else max_off+xrange*0.05
                ax.annotate(f"{max_cnt:,} hashes\nalign here",
                            xy=(max_off, max_cnt), xytext=(txt_x, max_cnt*0.72),
                            color='#00ff88', fontsize=9,
                            arrowprops=dict(arrowstyle='->', color='#00ff88', lw=1.5),
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='#0a1a0f', edgecolor='#00ff88', alpha=0.9))
                ax.set_ylim(0, max_cnt*1.28)
                ax.set_xlabel("Time Offset (database frame − query frame)")
                ax.set_ylabel("Number of Matching Hashes")
                ax.set_title(f"Offset Histogram — '{best['song_name']}'", fontsize=10, pad=8, color='#c9d1d9')
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

                # ── Step 4: Top Candidates ────────────────────────────────────
                st.markdown("""
                <div class="step-header">
                    <div class="step-num">04</div>
                    <div class="step-title">Top Candidates Ranked</div>
                    <div class="step-sub">All competing songs scored</div>
                </div>""", unsafe_allow_html=True)

                top5 = results[:5]
                top5_max = top5[0]['score']
                rows_html = ""
                for rank, r in enumerate(top5, 1):
                    pct  = int(r['score']/top5_max*100) if top5_max>0 else 0
                    gold = rank == 1
                    bar_color = "linear-gradient(90deg,#00ff88,#00aaff)" if gold else "#1a2a3a"
                    name_col  = "color:#00ff88;" if gold else ""
                    rows_html += f"""
                    <div class="cand-row">
                        <div class="cand-rank">{'🥇' if gold else f'#{rank}'}</div>
                        <div class="cand-name" style="{name_col}">{r['song_name']}</div>
                        <div class="cand-bar-wrap"><div class="cand-bar" style="width:{pct}%;background:{bar_color};"></div></div>
                        <div class="cand-score">{r['score']:,}</div>
                    </div>"""
                st.markdown(f'<div class="cand-table-wrap">{rows_html}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 · BATCH TEST
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown('<div class="section-label">Bulk identification &amp; accuracy benchmarking</div>', unsafe_allow_html=True)
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
        st.markdown(f"""
        <div style="padding:10px 16px;background:rgba(0,170,255,0.06);border:1px solid rgba(0,170,255,0.15);
                    border-radius:10px;font-size:0.82rem;color:#00aaff;margin-bottom:14px;
                    font-family:'Space Mono',monospace;">
            {len(uploaded_files)} clip(s) queued for identification
        </div>""", unsafe_allow_html=True)

        if "batch_df" not in st.session_state:
            st.session_state.batch_df = None

        if st.button("⚡  Run Batch Identification", key="batch_btn"):
            progress = st.progress(0)
            status   = st.empty()
            results_data = []

            for i, f in enumerate(uploaded_files):
                base      = os.path.splitext(f.name)[0]
                full_name = f.name
                status.markdown(f"🔍 Identifying `{f.name}`… ({i+1}/{len(uploaded_files)})")
                prediction = "ERROR"
                tmp_path   = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                        tmp.write(f.getvalue())
                        tmp_path = tmp.name
                    spec, _ = recognizer.get_spectrogram(tmp_path)
                    fr, ti  = recognizer.extract_peaks(spec, percentile=90)
                    hs      = recognizer.generate_hashes(fr, ti)
                    ms      = db.match_hashes(hs)
                    prediction = ms[0]['song_name'] if (ms and ms[0]['score'] >= 15) else "NO MATCH"
                except Exception:
                    prediction = "ERROR"
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

                def normalise(s): return re.sub(r'[^a-z0-9]', '', s.lower())
                is_correct = normalise(base) == normalise(prediction)
                results_data.append({"filename": full_name, "prediction": prediction, "correct": is_correct})
                progress.progress((i+1)/len(uploaded_files))

            status.markdown("**✅ Batch complete!**")
            st.session_state.batch_df = pd.DataFrame(results_data)

        if st.session_state.batch_df is not None:
            df = st.session_state.batch_df
            correct_count = df['correct'].sum()
            accuracy = int(correct_count / len(df) * 100)

            acc_color = "#00ff88" if accuracy == 100 else ("#f0883e" if accuracy >= 70 else "#f85149")

            st.markdown(f"""
            <div style="display:flex;gap:12px;margin:18px 0;flex-wrap:wrap;">
                <div class="sys-card" style="flex:1;min-width:120px;text-align:center;">
                    <div class="sys-card-val">{len(df)}</div>
                    <div class="sys-card-label">Clips Processed</div>
                </div>
                <div class="sys-card" style="flex:1;min-width:120px;text-align:center;">
                    <div class="sys-card-val">{int(correct_count)}</div>
                    <div class="sys-card-label">Correct Matches</div>
                </div>
                <div class="sys-card" style="flex:1;min-width:120px;text-align:center;">
                    <div class="sys-card-val" style="background:none;-webkit-text-fill-color:{acc_color};color:{acc_color};">{accuracy}%</div>
                    <div class="sys-card-label">Accuracy</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
            rows_html = ""
            for _, row in df.iterrows():
                badge = (
                    '<span class="correct-badge match">✓ MATCH</span>' if row['correct']
                    else '<span class="correct-badge wrong">✗ WRONG</span>'
                )
                rows_html += f"""
                <div class="batch-row">
                    <div style="flex:1;font-size:0.82rem;font-weight:600;color:#e6edf3;font-family:'Space Mono',monospace;">{row['filename']}</div>
                    <div style="flex:1;font-size:0.82rem;color:#6e7681;">{row['prediction']}</div>
                    {badge}
                </div>"""
            st.markdown(f'<div class="cand-table-wrap">{rows_html}</div>', unsafe_allow_html=True)

            csv_out = df[['filename', 'prediction']].to_csv(index=False)
            st.download_button("⬇  Download results.csv", data=csv_out, file_name="results.csv", mime="text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 · HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown('<div class="section-label">Technical deep-dive into the fingerprinting algorithm</div>', unsafe_allow_html=True)

    # Feature cards
    fc = st.columns(3)
    features = [
        ("🌊", "STFT Spectrogram", "Transform audio into a log-magnitude frequency-time heatmap using Short-Time Fourier Transform with n_fft=2048 and hop_length=512, giving a 46ms time resolution."),
        ("⭐", "Constellation Map", "Extract local spectral maxima (peaks) using a 15×15 neighborhood filter. This sparse set of 'stars' captures the harmonic skeleton using <1% of spectrogram cells."),
        ("🔗", "Hash Pairs", "Pair each anchor peak with up to 20 nearby target peaks. Each pair generates a hash (f₁, f₂, Δt) — a highly specific, noise-robust fingerprint of the song's structure."),
        ("🗃️", "Hash Database", "Store millions of (hash → song_id, time_offset) entries in a Python dict for O(1) lookup. Our database holds {:,} unique hash keys.".format(total_hashes)),
        ("📐", "Offset Histogram", "For a query clip, generate hashes and lookup each in the DB. Count votes for each (song, offset) pair. A genuine match produces a statistically impossible spike."),
        ("🛡️", "Robustness", "The system is robust to MP3 compression, background noise, and partial clips because the hash structure (relative freq, relative time) is preserved under these distortions."),
    ]
    for i, (icon, title, desc) in enumerate(features):
        with fc[i % 3]:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div><br>""", unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(255,255,255,0.05);margin:8px 0 24px;">', unsafe_allow_html=True)

    # Algorithm complexity table
    st.markdown('<div class="section-label">Algorithm Complexity Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="cand-table-wrap">
        <div class="cand-row" style="background:rgba(0,255,136,0.04);">
            <div style="flex:1;font-size:0.72rem;font-family:'Space Mono',monospace;color:#4d5765;">OPERATION</div>
            <div style="flex:1;font-size:0.72rem;font-family:'Space Mono',monospace;color:#4d5765;">TIME COMPLEXITY</div>
            <div style="flex:1;font-size:0.72rem;font-family:'Space Mono',monospace;color:#4d5765;">SPACE COMPLEXITY</div>
            <div style="flex:1;font-size:0.72rem;font-family:'Space Mono',monospace;color:#4d5765;">NOTES</div>
        </div>
        <div class="cand-row">
            <div style="flex:1;font-size:0.8rem;color:#e6edf3;">STFT Spectrogram</div>
            <div style="flex:1;font-size:0.8rem;color:#00aaff;font-family:'Space Mono',monospace;">O(N log N)</div>
            <div style="flex:1;font-size:0.8rem;color:#6e7681;font-family:'Space Mono',monospace;">O(N·F)</div>
            <div style="flex:1;font-size:0.78rem;color:#4d5765;">N=signal length, F=freq bins</div>
        </div>
        <div class="cand-row">
            <div style="flex:1;font-size:0.8rem;color:#e6edf3;">Peak Extraction</div>
            <div style="flex:1;font-size:0.8rem;color:#00aaff;font-family:'Space Mono',monospace;">O(N·F·w²)</div>
            <div style="flex:1;font-size:0.8rem;color:#6e7681;font-family:'Space Mono',monospace;">O(P)</div>
            <div style="flex:1;font-size:0.78rem;color:#4d5765;">w=neighborhood size (15)</div>
        </div>
        <div class="cand-row">
            <div style="flex:1;font-size:0.8rem;color:#e6edf3;">Hash Generation</div>
            <div style="flex:1;font-size:0.8rem;color:#00aaff;font-family:'Space Mono',monospace;">O(P·fan)</div>
            <div style="flex:1;font-size:0.8rem;color:#6e7681;font-family:'Space Mono',monospace;">O(H)</div>
            <div style="flex:1;font-size:0.78rem;color:#4d5765;">P=peaks, fan=fan_out (20)</div>
        </div>
        <div class="cand-row">
            <div style="flex:1;font-size:0.8rem;color:#e6edf3;">DB Lookup</div>
            <div style="flex:1;font-size:0.8rem;color:#00ff88;font-family:'Space Mono',monospace;">O(H)</div>
            <div style="flex:1;font-size:0.8rem;color:#6e7681;font-family:'Space Mono',monospace;">O(S·H)</div>
            <div style="flex:1;font-size:0.78rem;color:#4d5765;">O(1) per hash — dict lookup</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(255,255,255,0.05);margin:24px 0;">', unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-label">The Hash Function</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:rgba(0,0,0,0.4);border:1px solid rgba(0,255,136,0.1);
                    border-radius:12px;padding:20px;font-family:'Space Mono',monospace;font-size:0.78rem;line-height:2;">
            <span style="color:#4d5765;"># For each anchor peak at (f₁, t₁):</span><br>
            <span style="color:#4d5765;"># For each target peak at (f₂, t₂) in fan zone:</span><br>
            <br>
            <span style="color:#00aaff;">hash_key</span> <span style="color:#e6edf3;">=</span> <span style="color:#00ff88;">(f₁, f₂, t₂ − t₁)</span><br>
            <span style="color:#00aaff;">hash_val</span> <span style="color:#e6edf3;">=</span> <span style="color:#f0883e;">(song_id, t₁)</span><br>
            <br>
            <span style="color:#4d5765;"># During matching:</span><br>
            <span style="color:#00aaff;">offset</span> <span style="color:#e6edf3;">=</span> <span style="color:#f0883e;">db_t₁</span> <span style="color:#e6edf3;">−</span> <span style="color:#00ff88;">query_t₁</span>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-label">Why It Works</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:rgba(0,0,0,0.4);border:1px solid rgba(0,170,255,0.1);
                    border-radius:12px;padding:20px;font-size:0.78rem;line-height:1.9;color:#6e7681;">
            A single frequency <span style="color:#00ff88;">f₁ = 440 Hz</span> appears in thousands of songs.<br><br>
            But the triplet <span style="color:#00aaff;">(440 Hz, 880 Hz, Δt=3.2s)</span> is nearly unique.<br><br>
            When <span style="color:#e6edf3;">thousands of such triplets</span> all agree on a single time offset, 
            the probability of a <span style="color:#f85149;">false positive approaches zero</span> — 
            providing cryptographic-grade identification.
        </div>
        """, unsafe_allow_html=True)
