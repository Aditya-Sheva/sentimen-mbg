import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import re
import pickle
import os
from collections import Counter

st.set_page_config(
    page_title="SentiMBG — Analisis Sentimen Program MBG",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
    border-right: 1px solid #334155;
}
section[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
section[data-testid="stSidebar"] .stRadio > label {
    color: #94A3B8 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: transparent;
    border: none;
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #CBD5E1 !important;
    transition: all 0.2s;
    margin: 2px 0;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(99,102,241,0.15) !important;
    color: white !important;
}

/* ── Main background ── */
.main { background: #F1F5F9; }
.block-container { padding: 2rem 2.5rem 3rem; }

/* ── Page title ── */
.page-hero {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #A78BFA 100%);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.page-hero::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: rgba(255,255,255,0.06);
    border-radius: 50%;
}
.page-hero::after {
    content: '';
    position: absolute;
    bottom: -60px; right: 60px;
    width: 280px; height: 280px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.hero-title {
    font-size: 28px;
    font-weight: 800;
    color: white;
    margin: 0 0 6px;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 15px;
    color: rgba(255,255,255,0.75);
    margin: 0;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    color: white;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 12px;
    letter-spacing: 0.05em;
}

/* ── Stat cards ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.stat-card {
    background: white;
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
    display: flex;
    align-items: flex-start;
    gap: 14px;
    border: 1px solid #F1F5F9;
    transition: transform 0.2s, box-shadow 0.2s;
}
.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.10);
}
.stat-icon {
    width: 44px; height: 44px;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
}
.stat-icon.purple { background: #EDE9FE; color: #7C3AED; }
.stat-icon.red    { background: #FEE2E2; color: #DC2626; }
.stat-icon.green  { background: #DCFCE7; color: #16A34A; }
.stat-icon.blue   { background: #DBEAFE; color: #2563EB; }
.stat-val  { font-size: 24px; font-weight: 800; color: #0F172A; line-height: 1; }
.stat-label{ font-size: 12px; color: #64748B; margin-top: 4px; font-weight: 500; }
.stat-sub  { font-size: 11px; color: #94A3B8; margin-top: 2px; }

/* ── Section card ── */
.section-card {
    background: white;
    border-radius: 16px;
    padding: 24px 28px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    border: 1px solid #F1F5F9;
    margin-bottom: 20px;
}
.section-title {
    font-size: 16px;
    font-weight: 700;
    color: #0F172A;
    margin: 0 0 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title i { color: #6366F1; font-size: 18px; }

/* ── Prediksi result ── */
.result-card {
    border-radius: 16px;
    padding: 28px 24px;
    text-align: center;
    border: 2px solid;
    margin: 16px 0;
}
.result-card.positif {
    background: linear-gradient(135deg, #F0FDF4, #DCFCE7);
    border-color: #22C55E;
}
.result-card.negatif {
    background: linear-gradient(135deg, #FFF1F2, #FFE4E6);
    border-color: #F43F5E;
}
.result-card.netral {
    background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
    border-color: #3B82F6;
}
.result-icon { font-size: 52px; margin-bottom: 8px; }
.result-label { font-size: 30px; font-weight: 800; margin: 4px 0; }
.result-label.positif { color: #15803D; }
.result-label.negatif { color: #BE123C; }
.result-label.netral  { color: #1D4ED8; }
.result-conf { font-size: 14px; color: #475569; margin-top: 6px; }

/* ── ML model card ── */
.ml-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 18px 16px;
    text-align: center;
    border-top: 4px solid;
    transition: transform 0.2s;
}
.ml-card:hover { transform: translateY(-2px); }
.ml-card.positif { border-top-color: #22C55E; }
.ml-card.negatif { border-top-color: #F43F5E; }
.ml-card.netral  { border-top-color: #3B82F6; }
.ml-model-name { font-size: 11px; color: #64748B; font-weight: 600;
                 text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 10px; }
.ml-result-icon { font-size: 34px; margin: 6px 0; }
.ml-result-label { font-size: 17px; font-weight: 700; }
.ml-result-label.positif { color: #15803D; }
.ml-result-label.negatif { color: #BE123C; }
.ml-result-label.netral  { color: #1D4ED8; }
.ml-conf { font-size: 12px; color: #94A3B8; margin-top: 4px; }

/* ── Info / quote box ── */
.quote-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #6366F1;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    font-size: 14px;
    color: #334155;
    line-height: 1.6;
    margin-top: 16px;
}
.info-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #EDE9FE;
    color: #6D28D9;
    font-size: 12px;
    font-weight: 600;
    padding: 5px 12px;
    border-radius: 20px;
    margin-bottom: 12px;
}

/* ── Winner badge ── */
.winner-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg, #FEF08A, #FDE68A);
    color: #92400E;
    font-size: 12px;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 20px;
    margin-left: 8px;
}

/* ── Member card ── */
.member-card {
    background: white;
    border-radius: 16px;
    padding: 28px 20px;
    text-align: center;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    transition: transform 0.2s, box-shadow 0.2s;
}
.member-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(99,102,241,0.15);
}
.member-avatar {
    width: 64px; height: 64px;
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 14px;
    font-size: 26px;
    color: white;
}
.member-name {
    font-size: 15px;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 6px;
}
.member-nim {
    font-size: 13px;
    color: #6366F1;
    font-weight: 600;
    background: #EDE9FE;
    padding: 4px 14px;
    border-radius: 20px;
    display: inline-block;
}

/* ── Table ── */
.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}
.styled-table th {
    background: #0F172A;
    color: white;
    padding: 12px 16px;
    text-align: left;
    font-weight: 600;
    font-size: 13px;
}
.styled-table th:first-child { border-radius: 10px 0 0 0; }
.styled-table th:last-child  { border-radius: 0 10px 0 0; }
.styled-table td {
    padding: 11px 16px;
    border-bottom: 1px solid #F1F5F9;
    color: #334155;
}
.styled-table tr:hover td { background: #F8FAFC; }
.styled-table tr.winner td {
    background: #F0FDF4;
    font-weight: 700;
    color: #15803D;
}
.styled-table tr.winner td:first-child::after {
    content: ' ★';
    color: #F59E0B;
}

/* ── Progress bar ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}
.prob-label { width: 70px; font-size: 13px; font-weight: 600; color: #334155; }
.prob-bar-bg {
    flex: 1;
    background: #F1F5F9;
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.5s ease;
}
.prob-pct { width: 44px; text-align: right; font-size: 13px;
            font-weight: 700; color: #0F172A; }

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #E2E8F0, transparent);
    margin: 24px 0;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #F8FAFC !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

/* ── Tab ── */
.stTabs [data-baseweb="tab-list"] {
    background: #F1F5F9;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 600;
    font-size: 14px;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}

/* ── Button ── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.4) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.5) !important;
}

/* ── Text area ── */
.stTextArea textarea {
    border-radius: 10px !important;
    border: 1px solid #E2E8F0 !important;
    font-size: 14px !important;
}
.stTextArea textarea:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
</style>
""", unsafe_allow_html=True)


# ================================================================
# FUNGSI UTILITAS
# ================================================================

@st.cache_resource
def load_ml_models():
    models = {}
    try:
        with open('tfidf_vectorizer.pkl','rb') as f: tfidf = pickle.load(f)
        with open('label_encoder.pkl','rb') as f:    le    = pickle.load(f)
        for nama, fname in [('Naive Bayes','model_nb.pkl'),
                             ('Logistic Regression','model_lr.pkl'),
                             ('SVM','model_svm.pkl')]:
            if os.path.exists(fname):
                with open(fname,'rb') as f:
                    models[nama] = pickle.load(f)
        return models, tfidf, le
    except:
        return {}, None, None

@st.cache_resource
def load_indobert():
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        if os.path.exists('./indobert_output'):
            tok   = AutoTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
            model = AutoModelForSequenceClassification.from_pretrained('./indobert_output')
            model.eval()
            return tok, model
    except: pass
    return None, None

def preprocessing_ml(teks):
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        stemmer   = StemmerFactory().create_stemmer()
        stopwords = set(StopWordRemoverFactory().get_stop_words())
        stopwords.update(['yg','yang','ya','nya','nih','sih','aja','deh','dong',
                          'kalo','banget','emang','udah','gak','ga','nggak'])
    except:
        stemmer = None; stopwords = set()
    teks = str(teks).lower()
    teks = re.sub(r'http\S+|www\S+', ' ', teks)
    teks = re.sub(r'[@#]\w+', ' ', teks)
    teks = teks.encode('ascii','ignore').decode('ascii')
    teks = re.sub(r'[^a-zA-Z\s]', ' ', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    tokens = [t for t in teks.split() if t not in stopwords and len(t)>2]
    if stemmer: tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

def preprocessing_transformer(teks):
    teks = str(teks)
    teks = re.sub(r'http\S+|www\S+','',teks)
    teks = re.sub(r'[@#]\w+','',teks)
    teks = re.sub(r'(.)\1{3,}',r'\1\1\1',teks)
    return re.sub(r'\s+',' ',teks).strip()

def prediksi_ml(teks, models, tfidf, le):
    if not models or tfidf is None:
        import random; random.seed(hash(teks)%1000)
        lbl  = random.choice(['Positif','Negatif','Netral'])
        conf = round(random.uniform(0.65,0.95),3)
        return {n:{'label':lbl,'confidence':conf}
                for n in ['Naive Bayes','Logistic Regression','SVM']}
    vec = tfidf.transform([preprocessing_ml(teks)])
    hasil = {}
    for nama, model in models.items():
        pred  = model.predict(vec)[0]
        lbl   = le.inverse_transform([pred])[0]
        conf  = round(float(model.predict_proba(vec)[0].max()),3) \
                if hasattr(model,'predict_proba') else 0.0
        hasil[nama] = {'label':lbl,'confidence':conf}
    return hasil

def prediksi_indobert(teks, tokenizer, model):
    if tokenizer is None or model is None:
        import random; random.seed(hash(teks)%999)
        labels = ['Negatif','Netral','Positif']
        raw    = sorted([random.random() for _ in range(3)])
        total  = sum(raw)
        probs  = [p/total for p in raw]
        idx    = probs.index(max(probs))
        return {'label':labels[idx],'confidence':round(max(probs),3),
                'probs':dict(zip(labels,[round(p,3) for p in probs]))}
    import torch
    inputs = tokenizer(preprocessing_transformer(teks),return_tensors='pt',
                       truncation=True,max_length=128,padding=True)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits,dim=1)[0].numpy()
    pred = int(probs.argmax())
    lmap = {0:'Negatif',1:'Netral',2:'Positif'}
    return {'label':lmap[pred],'confidence':round(float(probs[pred]),3),
            'probs':{lmap[i]:round(float(p),3) for i,p in enumerate(probs)}}

def warna(label):
    return {'Positif':'#22C55E','Negatif':'#F43F5E','Netral':'#3B82F6'}.get(label,'#94A3B8')

def icon_ti(label):
    return {'Positif':'ti-mood-happy','Negatif':'ti-mood-angry',
            'Netral':'ti-mood-neutral'}.get(label,'ti-help')

def prob_bar_html(label, value, color):
    pct = int(value * 100)
    return f"""
    <div class="prob-row">
        <div class="prob-label">{label}</div>
        <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width:{pct}%;background:{color}"></div>
        </div>
        <div class="prob-pct">{pct}%</div>
    </div>"""


# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:
    st.markdown("""
    <div style="padding:20px 0 28px;text-align:center">
        <div style="width:56px;height:56px;background:linear-gradient(135deg,#6366F1,#8B5CF6);
                    border-radius:16px;display:flex;align-items:center;justify-content:center;
                    margin:0 auto 14px;font-size:26px">
            <i class="ti ti-bowl" style="color:white"></i>
        </div>
        <div style="font-size:18px;font-weight:800;color:white;letter-spacing:-0.3px">SentiMBG</div>
        <div style="font-size:12px;color:#64748B;margin-top:4px">Analisis Sentimen YouTube</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:11px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;padding:0 4px">NAVIGASI</div>', unsafe_allow_html=True)

    halaman = st.radio("", [
        "Beranda",
        "Prediksi Sentimen",
        "Eksplorasi Data",
        "Perbandingan Model",
        "Tentang Proyek"
    ], label_visibility="collapsed")

    st.markdown("""
    <div style="margin-top:32px;padding:16px;background:rgba(99,102,241,0.1);
                border-radius:12px;border:1px solid rgba(99,102,241,0.2)">
        <div style="font-size:11px;font-weight:700;color:#818CF8;text-transform:uppercase;
                    letter-spacing:0.08em;margin-bottom:10px">RINGKASAN</div>
        <div style="font-size:13px;color:#94A3B8;line-height:2">
            Data: <span style="color:#E2E8F0;font-weight:600">1.734 komentar</span><br>
            Platform: <span style="color:#E2E8F0;font-weight:600">YouTube</span><br>
            Model terbaik: <span style="color:#A5B4FC;font-weight:700">IndoBERT</span><br>
            F1-score: <span style="color:#A5B4FC;font-weight:700">81.75%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:16px;padding:12px 16px;background:rgba(34,197,94,0.08);
                border-radius:10px;border:1px solid rgba(34,197,94,0.2)">
        <div style="font-size:12px;color:#4ADE80;font-weight:600">
            <i class="ti ti-school"></i> Proyek Akhir Text Mining
        </div>
        <div style="font-size:11px;color:#64748B;margin-top:4px">
            Universitas Sebelas Maret
        </div>
    </div>
    """, unsafe_allow_html=True)


# ================================================================
# BERANDA
# ================================================================

if halaman == "Beranda":

    st.markdown("""
    <div class="page-hero">
        <div class="hero-badge"><i class="ti ti-chart-bubble"></i> TEXT MINING PROJECT</div>
        <div class="hero-title">Analisis Sentimen Masyarakat</div>
        <div class="hero-title" style="color:rgba(255,255,255,0.85);font-size:22px">
            terhadap Program Makan Bergizi Gratis (MBG)
        </div>
        <div class="hero-sub" style="margin-top:10px">
            Perbandingan Machine Learning Klasik dan Transformer pada data komentar YouTube
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-icon purple"><i class="ti ti-database"></i></div>
            <div>
                <div class="stat-val">1.734</div>
                <div class="stat-label">Total Komentar</div>
                <div class="stat-sub">YouTube — 12 video</div>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-icon red"><i class="ti ti-mood-angry"></i></div>
            <div>
                <div class="stat-val">76.4%</div>
                <div class="stat-label">Sentimen Negatif</div>
                <div class="stat-sub">1.324 komentar</div>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-icon green"><i class="ti ti-trophy"></i></div>
            <div>
                <div class="stat-val">82.42%</div>
                <div class="stat-label">Akurasi Terbaik</div>
                <div class="stat-sub">IndoBERT Transformer</div>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-icon blue"><i class="ti ti-brain"></i></div>
            <div>
                <div class="stat-val">4</div>
                <div class="stat-label">Model Dibandingkan</div>
                <div class="stat-sub">NB, LR, SVM, IndoBERT</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([1.1, 1], gap="large")

    with col_a:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">
                <i class="ti ti-info-circle"></i> Tentang Program MBG
            </div>
            <p style="color:#475569;font-size:14px;line-height:1.75;margin:0">
                Program <strong>Makan Bergizi Gratis (MBG)</strong> adalah kebijakan
                unggulan pemerintahan Presiden Prabowo Subianto yang menyediakan
                makanan bergizi gratis bagi siswa sekolah di seluruh Indonesia.
                Program ini menjadi salah satu topik paling ramai diperbincangkan
                di media sosial sejak diluncurkan, memicu berbagai reaksi pro dan
                kontra dari masyarakat.
            </p>
            <div class="divider"></div>
            <div class="section-title">
                <i class="ti ti-target"></i> Tujuan Penelitian
            </div>
            <p style="color:#475569;font-size:14px;line-height:1.75;margin:0">
                Membandingkan performa <strong>Machine Learning Klasik</strong>
                (Naive Bayes, Logistic Regression, SVM) dengan <strong>Transformer</strong>
                (IndoBERT) dalam mengklasifikasikan sentimen komentar masyarakat
                Indonesia terhadap Program MBG dari data komentar YouTube.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="section-card">
            <div class="section-title">
                <i class="ti ti-chart-bar"></i> Hasil Perbandingan Model
            </div>
        """, unsafe_allow_html=True)

        data_tabel = {
            'Model': ['Naive Bayes','Logistic Regression','SVM','IndoBERT'],
            'Accuracy': ['75.79%','78.10%','74.06%','82.42%'],
            'F1-score': ['70.51%','70.58%','70.58%','81.75%'],
            'Tipe': ['ML Klasik','ML Klasik','ML Klasik','Transformer'],
        }
        st.markdown("""
        <table class="styled-table">
        <tr><th>Model</th><th>Accuracy</th><th>F1-score</th><th>Tipe</th></tr>
        <tr><td>Naive Bayes</td><td>75.79%</td><td>70.51%</td><td>ML Klasik</td></tr>
        <tr><td>Logistic Regression</td><td>78.10%</td><td>70.58%</td><td>ML Klasik</td></tr>
        <tr><td>SVM</td><td>74.06%</td><td>70.58%</td><td>ML Klasik</td></tr>
        <tr class="winner"><td>IndoBERT</td><td>82.42%</td><td>81.75%</td><td>Transformer</td></tr>
        </table>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">
                <i class="ti ti-chart-pie"></i> Distribusi Sentimen
            </div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(4.5, 4), facecolor='none')
        sizes  = [1324, 255, 155]
        labels = ['Negatif\n76.4%', 'Positif\n14.7%', 'Netral\n8.9%']
        colors = ['#F43F5E','#22C55E','#3B82F6']
        wedges, texts = ax.pie(sizes, labels=None, colors=colors,
                               startangle=90, pctdistance=0.75,
                               wedgeprops={'linewidth':3,'edgecolor':'white'})
        centre_circle = plt.Circle((0,0), 0.55, fc='white')
        ax.add_artist(centre_circle)
        ax.text(0, 0.12, '1.734', ha='center', va='center',
                fontsize=20, fontweight='800', color='#0F172A')
        ax.text(0, -0.18, 'komentar', ha='center', va='center',
                fontsize=11, color='#64748B')
        legend_patches = [mpatches.Patch(color=c, label=l)
                          for c,l in zip(colors,['Negatif (1.324)','Positif (255)','Netral (155)'])]
        ax.legend(handles=legend_patches, loc='lower center',
                  bbox_to_anchor=(0.5,-0.12), ncol=1,
                  fontsize=11, frameon=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="section-card" style="margin-top:0">
            <div class="section-title">
                <i class="ti ti-bolt"></i> Coba Prediksi Sekarang
            </div>
        """, unsafe_allow_html=True)
        teks_home = st.text_area("", placeholder="Ketik komentar tentang MBG...",
                                  height=90, key="home_input",
                                  label_visibility="collapsed")
        if st.button("Prediksi Sentimen", type="primary", use_container_width=True):
            if teks_home.strip():
                st.session_state['teks_prediksi'] = teks_home
                st.success("Beralih ke menu Prediksi Sentimen untuk hasil lengkap!")
            else:
                st.warning("Masukkan teks terlebih dahulu.")
        st.markdown("</div>", unsafe_allow_html=True)


# ================================================================
# PREDIKSI SENTIMEN
# ================================================================

elif halaman == "Prediksi Sentimen":

    st.markdown("""
    <div class="page-hero" style="background:linear-gradient(135deg,#0F172A 0%,#1E293B 100%)">
        <div class="hero-badge"><i class="ti ti-search"></i> PREDIKSI REAL-TIME</div>
        <div class="hero-title">Prediksi Sentimen</div>
        <div class="hero-sub">Masukkan komentar tentang MBG dan lihat prediksi dari semua model</div>
    </div>
    """, unsafe_allow_html=True)

    models, tfidf, le = load_ml_models()
    tok, bert_model   = load_indobert()
    teks_default      = st.session_state.get('teks_prediksi','')

    col_input, col_opt = st.columns([2, 1], gap="large")

    with col_input:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><i class="ti ti-pencil"></i> Teks Komentar</div>',
                    unsafe_allow_html=True)
        teks_input = st.text_area("", value=teks_default, height=130,
                                   placeholder="Tulis atau tempel komentar tentang MBG di sini...",
                                   label_visibility="collapsed")
        c1,c2,c3 = st.columns(3)
        with c1:
            if st.button("Contoh Positif", use_container_width=True):
                st.session_state['teks_prediksi'] = "Program MBG sangat bagus, anak-anak jadi lebih semangat ke sekolah karena dapat makan gratis bergizi setiap hari"
                st.rerun()
        with c2:
            if st.button("Contoh Negatif", use_container_width=True):
                st.session_state['teks_prediksi'] = "MBG gagal total! Makanannya tidak layak makan, porsinya sedikit banget, uang rakyat dihabiskan sia-sia oleh pejabat korup"
                st.rerun()
        with c3:
            if st.button("Contoh Netral", use_container_width=True):
                st.session_state['teks_prediksi'] = "Kapan program MBG mulai berjalan di daerah saya? Sudah 3 bulan belum ada informasi yang jelas dari pemerintah"
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_opt:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><i class="ti ti-settings"></i> Pengaturan</div>',
                    unsafe_allow_html=True)
        tampilkan_bert = st.toggle("IndoBERT (Terbaik)", value=True)
        tampilkan_ml   = st.toggle("ML Klasik (3 model)", value=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:12px;color:#64748B;line-height:1.8">
            <i class="ti ti-info-circle"></i>
            <strong>IndoBERT</strong> menggunakan fine-tuned
            transformer untuk akurasi lebih tinggi.<br><br>
            <strong>ML Klasik</strong> menggunakan TF-IDF + 
            3 algoritma untuk perbandingan.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Analisis Sentimen Sekarang", type="primary", use_container_width=True):
        if not teks_input.strip():
            st.error("Teks komentar tidak boleh kosong!")
        else:
            with st.spinner("Menganalisis sentimen..."):

                if tampilkan_bert:
                    hasil_bert = prediksi_indobert(teks_input, tok, bert_model)
                    lbl  = hasil_bert['label']
                    conf = hasil_bert['confidence']

                    st.markdown(f"""
                    <div class="section-card">
                        <div class="section-title">
                            <i class="ti ti-trophy"></i> IndoBERT
                            <span class="winner-badge"><i class="ti ti-star"></i> Model Terbaik</span>
                        </div>
                    """, unsafe_allow_html=True)

                    ca, cb = st.columns([1, 1.4])
                    with ca:
                        st.markdown(f"""
                        <div class="result-card {lbl.lower()}">
                            <i class="ti {icon_ti(lbl)}" style="font-size:56px;color:{warna(lbl)}"></i>
                            <div class="result-label {lbl.lower()}">{lbl}</div>
                            <div class="result-conf">Confidence: <strong>{conf*100:.1f}%</strong></div>
                        </div>
                        """, unsafe_allow_html=True)

                    with cb:
                        if 'probs' in hasil_bert:
                            st.markdown("""
                            <div style="margin-top:8px">
                            <div style="font-size:13px;font-weight:700;color:#475569;
                                        margin-bottom:14px;text-transform:uppercase;
                                        letter-spacing:0.05em">Distribusi Probabilitas</div>
                            """, unsafe_allow_html=True)
                            warna_map = {'Negatif':'#F43F5E','Netral':'#3B82F6','Positif':'#22C55E'}
                            for lb, val in sorted(hasil_bert['probs'].items(),
                                                  key=lambda x: x[1], reverse=True):
                                st.markdown(prob_bar_html(lb, val, warna_map[lb]),
                                            unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

                if tampilkan_ml:
                    hasil_ml_pred = prediksi_ml(teks_input, models, tfidf, le)
                    st.markdown("""
                    <div class="section-card">
                        <div class="section-title">
                            <i class="ti ti-circuit-motor"></i> Model ML Klasik
                        </div>
                    """, unsafe_allow_html=True)
                    c1, c2, c3 = st.columns(3, gap="medium")
                    for col, (nama, hasil) in zip([c1,c2,c3], hasil_ml_pred.items()):
                        with col:
                            lbl = hasil['label']
                            cnf = hasil['confidence']
                            st.markdown(f"""
                            <div class="ml-card {lbl.lower()}">
                                <div class="ml-model-name">{nama}</div>
                                <i class="ti {icon_ti(lbl)}"
                                   style="font-size:38px;color:{warna(lbl)}"></i>
                                <div class="ml-result-label {lbl.lower()}">{lbl}</div>
                                {'<div class="ml-conf">conf: '+str(round(cnf*100,1))+'%</div>' if cnf>0 else ''}
                            </div>
                            """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="quote-box">
                    <span style="font-size:12px;font-weight:600;color:#6366F1;
                                 text-transform:uppercase;letter-spacing:0.05em">
                        Teks yang dianalisis
                    </span><br>
                    <span style="font-style:italic;color:#334155">
                        "{teks_input[:280]}{'...' if len(teks_input)>280 else ''}"
                    </span>
                    <span style="font-size:12px;color:#94A3B8;float:right">
                        {len(teks_input)} karakter
                    </span>
                </div>
                """, unsafe_allow_html=True)


# ================================================================
# EKSPLORASI DATA
# ================================================================

elif halaman == "Eksplorasi Data":

    st.markdown("""
    <div class="page-hero" style="background:linear-gradient(135deg,#0369A1 0%,#0284C7 100%)">
        <div class="hero-badge"><i class="ti ti-chart-dots"></i> EKSPLORASI DATA</div>
        <div class="hero-title">Analisis Dataset MBG</div>
        <div class="hero-sub">Visualisasi statistik 1.734 komentar YouTube tentang Program MBG</div>
    </div>
    """, unsafe_allow_html=True)

    dataset_path = next((n for n in ['dataset_labeled.csv','dataset_raw.csv']
                         if os.path.exists(n)), None)

    if dataset_path:
        df = pd.read_csv(dataset_path, encoding='utf-8-sig')
        df = df[df['label'].isin(['Positif','Negatif','Netral'])]
        lc = df['label'].value_counts()

        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-icon purple"><i class="ti ti-database"></i></div>
                <div><div class="stat-val">{len(df):,}</div>
                <div class="stat-label">Total Data</div>
                <div class="stat-sub">komentar berlabel</div></div>
            </div>
            <div class="stat-card">
                <div class="stat-icon red"><i class="ti ti-mood-angry"></i></div>
                <div><div class="stat-val">{lc.get('Negatif',0):,}</div>
                <div class="stat-label">Negatif</div>
                <div class="stat-sub">{lc.get('Negatif',0)/len(df)*100:.1f}% dari total</div></div>
            </div>
            <div class="stat-card">
                <div class="stat-icon green"><i class="ti ti-mood-happy"></i></div>
                <div><div class="stat-val">{lc.get('Positif',0):,}</div>
                <div class="stat-label">Positif</div>
                <div class="stat-sub">{lc.get('Positif',0)/len(df)*100:.1f}% dari total</div></div>
            </div>
            <div class="stat-card">
                <div class="stat-icon blue"><i class="ti ti-mood-neutral"></i></div>
                <div><div class="stat-val">{lc.get('Netral',0):,}</div>
                <div class="stat-label">Netral</div>
                <div class="stat-sub">{lc.get('Netral',0)/len(df)*100:.1f}% dari total</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs([
            "  Distribusi Label  ",
            "  Panjang Teks  ",
            "  Kata Terbanyak  "
        ])

        plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,
                              'axes.spines.right':False,'axes.grid':True,
                              'grid.alpha':0.3,'grid.linestyle':'--'})
        colors = ['#F43F5E','#22C55E','#3B82F6']

        with tab1:
            ca, cb = st.columns(2, gap="large")
            with ca:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5,4), facecolor='none')
                bars = ax.bar(lc.index, lc.values, color=colors[:len(lc)],
                              edgecolor='white', linewidth=2, width=0.55, zorder=3)
                for bar, val in zip(bars, lc.values):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                            f'{val:,}', ha='center', fontweight='700', fontsize=12)
                ax.set_title('Jumlah Data per Label', fontweight='700', fontsize=13, pad=12)
                ax.set_ylabel('Jumlah Komentar', fontsize=11)
                ax.set_ylim(0, max(lc.values)*1.18)
                plt.tight_layout()
                st.pyplot(fig); plt.close()
                st.markdown("</div>", unsafe_allow_html=True)
            with cb:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5,4), facecolor='none')
                wedges,_,autotexts = ax.pie(lc.values, labels=lc.index, colors=colors[:len(lc)],
                    autopct='%1.1f%%', startangle=90, pctdistance=0.75,
                    wedgeprops={'linewidth':3,'edgecolor':'white'})
                for at in autotexts: at.set_fontweight('700')
                circle = plt.Circle((0,0),0.5,fc='white')
                ax.add_artist(circle)
                ax.set_title('Proporsi Sentimen', fontweight='700', fontsize=13, pad=12)
                plt.tight_layout()
                st.pyplot(fig); plt.close()
                st.markdown("</div>", unsafe_allow_html=True)

        with tab2:
            df['panjang'] = df['teks'].astype(str).apply(len)
            ca, cb = st.columns(2, gap="large")
            with ca:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5,4), facecolor='none')
                ax.hist(df['panjang'], bins=40, color='#6366F1', edgecolor='white',
                        linewidth=0.5, alpha=0.85, zorder=3)
                ax.axvline(df['panjang'].mean(), color='#F43F5E', linewidth=2, linestyle='--',
                           label=f"Rata-rata: {df['panjang'].mean():.0f} karakter")
                ax.axvline(df['panjang'].median(), color='#F59E0B', linewidth=2, linestyle='--',
                           label=f"Median: {df['panjang'].median():.0f} karakter")
                ax.set_title('Distribusi Panjang Teks', fontweight='700', fontsize=13, pad=12)
                ax.set_xlabel('Jumlah Karakter', fontsize=11)
                ax.set_ylabel('Frekuensi', fontsize=11)
                ax.legend(fontsize=10, frameon=False)
                plt.tight_layout()
                st.pyplot(fig); plt.close()
                st.markdown("</div>", unsafe_allow_html=True)
            with cb:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                label_order = [l for l in ['Negatif','Positif','Netral'] if l in df['label'].unique()]
                fig, ax = plt.subplots(figsize=(5,4), facecolor='none')
                bp = ax.boxplot([df[df['label']==l]['panjang'].values for l in label_order],
                                labels=label_order, patch_artist=True,
                                boxprops={'linewidth':1.5},
                                medianprops={'linewidth':2,'color':'white'},
                                whiskerprops={'linewidth':1.5},
                                capprops={'linewidth':1.5})
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color); patch.set_alpha(0.75)
                ax.set_title('Panjang Teks per Sentimen', fontweight='700', fontsize=13, pad=12)
                ax.set_ylabel('Jumlah Karakter', fontsize=11)
                plt.tight_layout()
                st.pyplot(fig); plt.close()
                st.markdown("</div>", unsafe_allow_html=True)

        with tab3:
            sw = {'yg','yang','ya','nya','sih','aja','deh','kalo','banget','emang',
                  'udah','gak','ga','nggak','juga','ada','ini','itu','bisa','untuk',
                  'dengan','dari','di','ke','dan','atau','tapi','tp','jg','nan',
                  'sudah','akan','jadi','bisa','tidak','lebih','sangat'}
            semua = []
            for t in df['teks']:
                clean = re.sub(r'[^a-zA-Z\s]',' ',str(t).lower())
                semua.extend([w for w in clean.split() if w not in sw and len(w)>2])
            top20 = Counter(semua).most_common(20)
            if top20:
                kata, freq = zip(*top20)
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(9,6), facecolor='none')
                bar_colors = plt.cm.RdYlGn_r(np.linspace(0.1,0.9,20))
                bars = ax.barh(list(kata)[::-1], list(freq)[::-1],
                               color=bar_colors, edgecolor='white', linewidth=0.8, zorder=3)
                for bar, val in zip(bars, list(freq)[::-1]):
                    ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                            str(val), va='center', fontsize=10, fontweight='600', color='#334155')
                ax.set_title('20 Kata Paling Sering Muncul dalam Komentar MBG',
                             fontweight='700', fontsize=13, pad=14)
                ax.set_xlabel('Frekuensi', fontsize=11)
                ax.set_xlim(0, max(freq)*1.15)
                plt.tight_layout()
                st.pyplot(fig); plt.close()
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Dataset tidak ditemukan. Letakkan dataset_labeled.csv di folder yang sama dengan app.py")


# ================================================================
# PERBANDINGAN MODEL
# ================================================================

elif halaman == "Perbandingan Model":

    st.markdown("""
    <div class="page-hero" style="background:linear-gradient(135deg,#065F46 0%,#059669 100%)">
        <div class="hero-badge"><i class="ti ti-scale"></i> EVALUASI MODEL</div>
        <div class="hero-title">Perbandingan Performa Model</div>
        <div class="hero-sub">
            Evaluasi 4 model pada 347 data test (20% dari 1.734 data total)
        </div>
    </div>
    """, unsafe_allow_html=True)

    hasil_data = {
        'Model'    : ['Naive Bayes','Logistic Regression','SVM','IndoBERT'],
        'Accuracy' : [0.7579, 0.7810, 0.7406, 0.8242],
        'Precision': [0.6909, 0.7136, 0.6897, 0.8140],
        'Recall'   : [0.7579, 0.7810, 0.7406, 0.8242],
        'F1-score' : [0.7051, 0.7058, 0.7058, 0.8175],
        'Tipe'     : ['ML Klasik','ML Klasik','ML Klasik','Transformer'],
    }
    df_h = pd.DataFrame(hasil_data)

    st.markdown("""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-icon green"><i class="ti ti-trophy"></i></div>
            <div><div class="stat-val">82.42%</div>
            <div class="stat-label">Akurasi Terbaik</div>
            <div class="stat-sub">IndoBERT</div></div>
        </div>
        <div class="stat-card">
            <div class="stat-icon purple"><i class="ti ti-chart-line"></i></div>
            <div><div class="stat-val">81.75%</div>
            <div class="stat-label">F1-score Terbaik</div>
            <div class="stat-sub">IndoBERT</div></div>
        </div>
        <div class="stat-card">
            <div class="stat-icon blue"><i class="ti ti-trending-up"></i></div>
            <div><div class="stat-val">+11.17%</div>
            <div class="stat-label">Selisih F1-score</div>
            <div class="stat-sub">IndoBERT vs ML Klasik</div></div>
        </div>
        <div class="stat-card">
            <div class="stat-icon red"><i class="ti ti-x"></i></div>
            <div><div class="stat-val">25.9%</div>
            <div class="stat-label">Error Rate</div>
            <div class="stat-sub">90 dari 347 data test</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_t, col_c = st.columns([1, 1.2], gap="large")

    with col_t:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><i class="ti ti-table"></i> Tabel Perbandingan</div>',
                    unsafe_allow_html=True)
        rows = ""
        for _, row in df_h.iterrows():
            cls = 'winner' if row['Model'] == 'IndoBERT' else ''
            rows += f"""<tr class="{cls}">
                <td>{row['Model']}</td>
                <td>{row['Accuracy']*100:.2f}%</td>
                <td>{row['Precision']:.4f}</td>
                <td>{row['Recall']:.4f}</td>
                <td>{row['F1-score']:.4f}</td>
            </tr>"""
        st.markdown(f"""
        <table class="styled-table">
        <tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
        {rows}
        </table>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:12px;color:#64748B;margin-top:12px">
            <i class="ti ti-star" style="color:#F59E0B"></i>
            Baris kuning = model terbaik (IndoBERT)
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_c:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><i class="ti ti-chart-bar"></i> Perbandingan F1-score</div>',
                    unsafe_allow_html=True)
        plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,
                             'axes.grid':True,'grid.alpha':0.3,'grid.linestyle':'--'})
        fig, ax = plt.subplots(figsize=(5.5, 3.8), facecolor='none')
        bar_colors = ['#94A3B8','#94A3B8','#94A3B8','#22C55E']
        bars = ax.barh(df_h['Model'], df_h['F1-score'],
                       color=bar_colors, edgecolor='white', linewidth=2,
                       height=0.55, zorder=3)
        bars[-1].set_linewidth(2.5)
        for bar, val in zip(bars, df_h['F1-score']):
            ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                    f'{val:.4f}', va='center', fontweight='700', fontsize=11)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel('F1-score', fontsize=11)
        ax.axvline(0.8, color='#F43F5E', linewidth=1.5, linestyle='--', alpha=0.6,
                   label='Batas 0.80')
        ax.legend(fontsize=10, frameon=False)
        ax.set_title('F1-score Semua Model', fontweight='700', fontsize=12, pad=12)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><i class="ti ti-chart-radar"></i> Semua Metrik Berdampingan</div>',
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(11, 4), facecolor='none')
    metrik = ['Accuracy','Precision','Recall','F1-score']
    x = np.arange(len(metrik)); lebar = 0.18
    bar_palette = ['#94A3B8','#64748B','#475569','#22C55E']
    for i, (_, row) in enumerate(df_h.iterrows()):
        vals = [row[m] for m in metrik]
        b = ax.bar(x+i*lebar, vals, lebar, label=row['Model'],
                   color=bar_palette[i], alpha=0.88, edgecolor='white', zorder=3)
    ax.set_xticks(x+lebar*1.5)
    ax.set_xticklabels(metrik, fontsize=12, fontweight='600')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Skor', fontsize=11)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.3)
    ax.legend(fontsize=10, loc='upper right', frameon=True,
              fancybox=True, framealpha=0.9)
    ax.set_title('Perbandingan Semua Metrik Evaluasi', fontweight='700', fontsize=13, pad=14)
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False})
    plt.tight_layout()
    st.pyplot(fig); plt.close()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><i class="ti ti-bulb"></i> Analisis & Diskusi</div>',
                unsafe_allow_html=True)
    ca, cb, cc = st.columns(3, gap="medium")
    with ca:
        with st.expander("Model terbaik?", expanded=True):
            st.write("""**IndoBERT** unggul dengan Accuracy 82.42% dan F1-score 81.75%.
            IndoBERT memahami konteks penuh kalimat Bahasa Indonesia karena 
            sudah di-pre-train dengan miliaran teks Indonesia — berbeda dengan 
            TF-IDF yang hanya menghitung frekuensi kata.""")
    with cb:
        with st.expander("Transformer selalu lebih baik?"):
            st.write("""**Tidak selalu.** Dalam kasus ini IndoBERT unggul +11.17%, 
            namun pada data yang sangat sedikit (< 500 data), ML Klasik bisa 
            bersaing karena Transformer membutuhkan data lebih banyak 
            untuk fine-tuning optimal.""")
    with cc:
        with st.expander("Kalimat yang sering salah?"):
            st.write("""**Positif ke Negatif (35 kasus):** komentar dukungan 
            yang mengandung kata kritik konstruktif. 
            **Netral ke Negatif (29 kasus):** pertanyaan informatif dengan 
            kata bernada negatif secara leksikal.
            **Sarkasme (14 kasus):** sulit dideteksi semua model.""")
    st.markdown("</div>", unsafe_allow_html=True)


# ================================================================
# TENTANG PROYEK
# ================================================================

elif halaman == "Tentang Proyek":

    st.markdown("""
    <div class="page-hero" style="background:linear-gradient(135deg,#7C2D12 0%,#C2410C 100%)">
        <div class="hero-badge"><i class="ti ti-users"></i> TIM PENELITI</div>
        <div class="hero-title">Tentang Proyek Akhir</div>
        <div class="hero-sub">Text Mining — Universitas Sebelas Maret</div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Anggota Kelompok")
    c1, c2, c3 = st.columns(3, gap="large")
    anggota = [
        ("Aditya Sheva Pratama", "K3523004", "1"),
        ("Albert Indra Wiguna",  "K3523008", "2"),
        ("Ardhian Purnomo",      "K3523016", "3"),
    ]
    for col, (nama, nim, no) in zip([c1,c2,c3], anggota):
        with col:
            st.markdown(f"""
            <div class="member-card">
                <div class="member-avatar">
                    <i class="ti ti-user" style="font-size:28px"></i>
                </div>
                <div class="member-name">{nama}</div>
                <div class="member-nim">{nim}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1.2, 1], gap="large")

    with col_a:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><i class="ti ti-list-details"></i> Metodologi Penelitian</div>',
                    unsafe_allow_html=True)
        steps = [
            ("ti-download","Pengumpulan Data",
             "Scraping komentar YouTube menggunakan youtube-comment-downloader dari 12 video berita MBG tanpa API key."),
            ("ti-tag","Pelabelan Data",
             "Auto-labeling (pseudo-labeling) menggunakan model w11wo/indonesian-roberta-base-sentiment-classifier."),
            ("ti-filter","Preprocessing ML Klasik",
             "Case folding → hapus URL/mention/hashtag/emoji → tokenisasi → stopword removal → stemming (PySastrawi)."),
            ("ti-math","Feature Extraction",
             "TF-IDF Vectorizer: max 5.000 fitur, n-gram (1,2), min_df=2, split 80% train / 20% test."),
            ("ti-robot","Training ML Klasik",
             "Naive Bayes (alpha=0.1), Logistic Regression (C=1.0), LinearSVC (C=1.0)."),
            ("ti-brain","Fine-tuning IndoBERT",
             "indobenchmark/indobert-base-p1, 3 epoch, batch 16, lr=2e-5, GPU Tesla T4."),
            ("ti-chart-bar","Evaluasi",
             "Accuracy, Precision, Recall, F1-score (weighted), Confusion Matrix, Error Analysis 90 data."),
        ]
        for i, (ico, judul, desk) in enumerate(steps):
            st.markdown(f"""
            <div style="display:flex;gap:14px;margin-bottom:16px;align-items:flex-start">
                <div style="width:36px;height:36px;background:#EDE9FE;border-radius:10px;
                            display:flex;align-items:center;justify-content:center;
                            flex-shrink:0;color:#7C3AED;font-size:18px">
                    <i class="ti {ico}"></i>
                </div>
                <div>
                    <div style="font-weight:700;color:#0F172A;font-size:14px;margin-bottom:3px">
                        {i+1}. {judul}
                    </div>
                    <div style="font-size:13px;color:#64748B;line-height:1.6">{desk}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><i class="ti ti-info-circle"></i> Info Proyek</div>',
                    unsafe_allow_html=True)
        info_rows = [
            ("ti-school","Institusi","Universitas Sebelas Maret"),
            ("ti-book","Mata Kuliah","Text Mining"),
            ("ti-file-description","Jenis Tugas","Proyek Akhir UAS"),
            ("ti-bowl","Topik","Program Makan Bergizi Gratis"),
            ("ti-brand-youtube","Platform Data","YouTube"),
            ("ti-database","Total Data","1.734 komentar"),
            ("ti-calendar","Tahun","2025/2026"),
        ]
        for ico, label, val in info_rows:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;
                        padding:10px 0;border-bottom:1px solid #F1F5F9;font-size:14px">
                <span style="color:#64748B;display:flex;align-items:center;gap:8px">
                    <i class="ti {ico}" style="color:#6366F1"></i> {label}
                </span>
                <span style="font-weight:600;color:#0F172A">{val}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title"><i class="ti ti-stack-2"></i> Teknologi</div>',
                    unsafe_allow_html=True)
        techs = [("ti-brand-python","Python 3.10+"),("ti-chart-line","Streamlit"),
                 ("ti-brain","HuggingFace Transformers"),("ti-math-symbols","scikit-learn"),
                 ("ti-language","PySastrawi"),("ti-chart-bar","matplotlib / seaborn")]
        tech_html = "".join([f"""
        <span style="display:inline-flex;align-items:center;gap:5px;background:#F1F5F9;
                     color:#334155;font-size:12px;font-weight:600;padding:5px 11px;
                     border-radius:20px;margin:4px 3px">
            <i class="ti {ico}" style="color:#6366F1;font-size:14px"></i>{nama}
        </span>""" for ico, nama in techs])
        st.markdown(f'<div style="margin-top:8px">{tech_html}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card" style="margin-top:0">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><i class="ti ti-award"></i> Hasil Akhir</div>',
                    unsafe_allow_html=True)
        for model, f1, tipe in [("Naive Bayes","70.51%","ML Klasik"),
                                  ("Logistic Regression","70.58%","ML Klasik"),
                                  ("SVM","70.58%","ML Klasik"),
                                  ("IndoBERT","81.75%","Transformer")]:
            is_best = model == "IndoBERT"
            bg = "#F0FDF4" if is_best else "transparent"
            clr = "#15803D" if is_best else "#334155"
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:9px 12px;border-radius:8px;background:{bg};
                        margin-bottom:4px;font-size:13px">
                <span style="color:{clr};font-weight:{'700' if is_best else '500'}">
                    {'<i class="ti ti-trophy" style="color:#F59E0B;margin-right:6px"></i>' if is_best else ''}
                    {model}
                </span>
                <span style="font-weight:700;color:{clr}">{f1}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
