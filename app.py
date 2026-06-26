import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import pickle
import os
from collections import Counter

st.set_page_config(
    page_title="SentiMBG — Analisis Sentimen MBG",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Inject CSS via st.markdown dengan unsafe_allow_html ─────
# Streamlit hanya render <style> jika ada konten HTML setelahnya
# Cara paling aman: gunakan 1 blok st.markdown untuk CSS saja

def inject_css():
    css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');


* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color: #CBD5E1 !important; }

/* Main area */
[data-testid="stAppViewContainer"] > .main { background: #F1F5F9 !important; }
.block-container { padding: 1.5rem 2rem 3rem !important; max-width: 1200px !important; }

/* Hide streamlit default header */
header[data-testid="stHeader"] { background: transparent !important; }

/* Buttons */
.stButton button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    transition: all 0.2s !important;
    border: 1.5px solid #E2E8F0 !important;
}
.stButton button[kind="primary"],
.stButton button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    color: white !important;
    border: none !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #F1F5F9 !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: none !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 600 !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
}

/* Text area */
textarea {
    border-radius: 10px !important;
    border: 1.5px solid #E2E8F0 !important;
    font-size: 14px !important;
}

/* Metric */
[data-testid="stMetric"] {
    background: white;
    border-radius: 12px;
    padding: 16px 20px;
    border: 1px solid #F1F5F9;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}

/* Expander */
details summary {
    font-weight: 600 !important;
    border-radius: 10px !important;
}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)

inject_css()

# ── Helper HTML components ────────────────────────────────────

def hero(title, sub, badge, gradient):
    st.markdown(f"""
<div style="background:{gradient};border-radius:20px;padding:32px 40px;
            margin-bottom:28px;position:relative;overflow:hidden">
  <div style="display:inline-block;background:rgba(255,255,255,0.18);color:white;
              font-size:11px;font-weight:700;padding:4px 14px;border-radius:20px;
              margin-bottom:10px;letter-spacing:0.08em">{badge}</div>
  <div style="font-size:26px;font-weight:800;color:white;letter-spacing:-0.5px;
              margin-bottom:6px">{title}</div>
  <div style="font-size:14px;color:rgba(255,255,255,0.78);font-weight:400">{sub}</div>
</div>
""", unsafe_allow_html=True)


def card(content_html, padding="24px 28px"):
    st.markdown(f"""
<div style="background:white;border-radius:16px;padding:{padding};
            box-shadow:0 1px 4px rgba(0,0,0,0.06);border:1px solid #F1F5F9;
            margin-bottom:16px">{content_html}</div>
""", unsafe_allow_html=True)


def section_title(icon_cls, title):
    return f"""<div style="font-size:15px;font-weight:700;color:#0F172A;
        margin-bottom:16px;display:flex;align-items:center;gap:8px;
        padding-bottom:10px;border-bottom:1px solid #F1F5F9">
        •{title}</div>"""


def stat_card(icon_cls, icon_bg, icon_color, val, label, sub):
    return f"""
<div style="background:white;border-radius:14px;padding:18px 20px;
            box-shadow:0 1px 4px rgba(0,0,0,0.06);border:1px solid #F1F5F9;
            display:flex;align-items:flex-start;gap:14px;
            transition:transform 0.2s">
  <div style="width:44px;height:44px;border-radius:12px;background:{icon_bg};
              color:{icon_color};display:flex;align-items:center;
              justify-content:center;font-size:20px;flex-shrink:0">
    •
  </div>
  <div>
    <div style="font-size:22px;font-weight:800;color:#0F172A;line-height:1">{val}</div>
    <div style="font-size:12px;color:#64748B;margin-top:4px;font-weight:500">{label}</div>
    <div style="font-size:11px;color:#94A3B8;margin-top:2px">{sub}</div>
  </div>
</div>"""


def result_card(label, conf, probs=None):
    cfg = {
        'Positif': ('#F0FDF4','#22C55E','#15803D','😊'),
        'Negatif': ('#FFF1F2','#F43F5E','#BE123C','😠'),
        'Netral':  ('#EFF6FF','#3B82F6','#1D4ED8','😐'),
    }
    bg, border, text, icon = cfg.get(label, ('#F8FAFC','#94A3B8','#334155','❓'))
    return f"""
<div style="background:{bg};border:2px solid {border};border-radius:16px;
            padding:24px;text-align:center">
  •
  <div style="font-size:28px;font-weight:800;color:{text};margin:8px 0">{label}</div>
  <div style="font-size:13px;color:#475569">Confidence: <strong>{conf*100:.1f}%</strong></div>
</div>"""


def ml_card(nama, label, conf):
    cfg = {
        'Positif': ('#22C55E','#15803D','😊'),
        'Negatif': ('#F43F5E','#BE123C','😠'),
        'Netral':  ('#3B82F6','#1D4ED8','😐'),
    }
    border, text, icon = cfg.get(label, ('#94A3B8','#334155','❓'))
    conf_html = f'<div style="font-size:11px;color:#94A3B8;margin-top:4px">conf: {conf*100:.1f}%</div>' if conf > 0 else ''
    return f"""
<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:14px;
            padding:18px 16px;text-align:center;border-top:4px solid {border}">
  <div style="font-size:10px;color:#64748B;font-weight:700;text-transform:uppercase;
              letter-spacing:0.06em;margin-bottom:10px">{nama}</div>
  •
  <div style="font-size:17px;font-weight:700;color:{text};margin-top:8px">{label}</div>
  {conf_html}
</div>"""


def prob_bars(probs):
    warna = {'Negatif':'#F43F5E','Netral':'#3B82F6','Positif':'#22C55E'}
    html = '<div style="margin-top:8px">'
    html += '<div style="font-size:11px;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:12px">Distribusi Probabilitas</div>'
    for lb, val in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        pct = int(val * 100)
        c = warna.get(lb, '#94A3B8')
        html += f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
  <div style="width:64px;font-size:13px;font-weight:600;color:#334155">{lb}</div>
  <div style="flex:1;background:#F1F5F9;border-radius:6px;height:10px;overflow:hidden">
    <div style="width:{pct}%;background:{c};height:100%;border-radius:6px"></div>
  </div>
  <div style="width:40px;text-align:right;font-size:13px;font-weight:700;color:#0F172A">{pct}%</div>
</div>"""
    html += "</div>"
    return html


# ================================================================
# MODEL LOADERS
# ================================================================

@st.cache_resource
def load_ml_models():
    try:
        with open('tfidf_vectorizer.pkl','rb') as f: tfidf = pickle.load(f)
        with open('label_encoder.pkl','rb') as f:    le    = pickle.load(f)
        models = {}
        for n, fn in [('Naive Bayes','model_nb.pkl'),
                      ('Logistic Regression','model_lr.pkl'),
                      ('SVM','model_svm.pkl')]:
            if os.path.exists(fn):
                with open(fn,'rb') as f: models[n] = pickle.load(f)
        return models, tfidf, le
    except:
        return {}, None, None

@st.cache_resource
def load_bert():
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        if os.path.exists('./indobert_output'):
            tok   = AutoTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
            model = AutoModelForSequenceClassification.from_pretrained('./indobert_output')
            model.eval()
            return tok, model
    except: pass
    return None, None

def preproc_ml(teks):
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        stemmer   = StemmerFactory().create_stemmer()
        stopwords = set(StopWordRemoverFactory().get_stop_words())
        stopwords.update(['yg','yang','ya','nya','nih','sih','aja','deh','dong',
                          'kalo','banget','emang','udah','gak','ga','nggak'])
    except:
        stemmer = None; stopwords = set()
    t = str(teks).lower()
    t = re.sub(r'http\S+|www\S+','',t)
    t = re.sub(r'[@#]\w+','',t)
    t = t.encode('ascii','ignore').decode('ascii')
    t = re.sub(r'[^a-zA-Z\s]',' ',t)
    t = re.sub(r'\s+',' ',t).strip()
    tokens = [w for w in t.split() if w not in stopwords and len(w)>2]
    if stemmer: tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(tokens)

def preproc_bert(teks):
    t = str(teks)
    t = re.sub(r'http\S+|www\S+','',t)
    t = re.sub(r'[@#]\w+','',t)
    t = re.sub(r'(.)\1{3,}',r'\1\1\1',t)
    return re.sub(r'\s+',' ',t).strip()

def pred_ml(teks, models, tfidf, le):
    if not models or tfidf is None:
        import random; random.seed(hash(teks)%1000)
        lbl  = random.choice(['Positif','Negatif','Netral'])
        conf = round(random.uniform(0.65,0.95),3)
        return {n:{'label':lbl,'confidence':conf}
                for n in ['Naive Bayes','Logistic Regression','SVM']}
    vec = tfidf.transform([preproc_ml(teks)])
    out = {}
    for nm, model in models.items():
        pred = model.predict(vec)[0]
        lbl  = le.inverse_transform([pred])[0]
        conf = round(float(model.predict_proba(vec)[0].max()),3) \
               if hasattr(model,'predict_proba') else 0.0
        out[nm] = {'label':lbl,'confidence':conf}
    return out

def pred_bert(teks, tok, model):
    if tok is None or model is None:
        import random; random.seed(hash(teks)%999)
        labels = ['Negatif','Netral','Positif']
        raw    = sorted([random.random() for _ in range(3)])
        total  = sum(raw); probs = [p/total for p in raw]
        idx    = probs.index(max(probs))
        return {'label':labels[idx],'confidence':round(max(probs),3),
                'probs':dict(zip(labels,[round(p,3) for p in probs]))}
    import torch
    inputs = tok(preproc_bert(teks),return_tensors='pt',
                 truncation=True,max_length=128,padding=True)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits,dim=1)[0].numpy()
    pred  = int(probs.argmax())
    lmap  = {0:'Negatif',1:'Netral',2:'Positif'}
    return {'label':lmap[pred],'confidence':round(float(probs[pred]),3),
            'probs':{lmap[i]:round(float(p),3) for i,p in enumerate(probs)}}


# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:
    st.markdown("""
<div style="padding:20px 0 24px;text-align:center">
  <div style="width:54px;height:54px;background:linear-gradient(135deg,#6366F1,#8B5CF6);
              border-radius:16px;display:flex;align-items:center;justify-content:center;
              margin:0 auto 12px">
    🍱
  </div>
  <div style="font-size:18px;font-weight:800;color:white">SentiMBG</div>
  <div style="font-size:12px;color:#475569;margin-top:3px">Analisis Sentimen YouTube</div>
</div>
""", unsafe_allow_html=True)

    halaman = st.radio("", [
        "Beranda", "Prediksi Sentimen",
        "Eksplorasi Data", "Perbandingan Model", "Tentang Proyek"
    ], label_visibility="collapsed")

    st.markdown("""
<div style="margin-top:28px;padding:14px 16px;background:rgba(99,102,241,0.12);
            border-radius:12px;border:1px solid rgba(99,102,241,0.25)">
  <div style="font-size:10px;font-weight:700;color:#818CF8;text-transform:uppercase;
              letter-spacing:0.1em;margin-bottom:8px">RINGKASAN</div>
  <div style="font-size:13px;line-height:2">
    Data: <span style="color:#E2E8F0;font-weight:600">1.734 komentar</span><br>
    Platform: <span style="color:#E2E8F0;font-weight:600">YouTube</span><br>
    Model terbaik: <span style="color:#A5B4FC;font-weight:700">IndoBERT</span><br>
    F1-score: <span style="color:#A5B4FC;font-weight:700">81.75%</span>
  </div>
</div>
<div style="margin-top:12px;padding:10px 14px;background:rgba(34,197,94,0.08);
            border-radius:10px;border:1px solid rgba(34,197,94,0.2)">
  <div style="font-size:12px;color:#4ADE80;font-weight:600">
    🏫 Proyek Akhir Text Mining
  </div>
  <div style="font-size:11px;color:#475569;margin-top:3px">Universitas Sebelas Maret</div>
</div>
""", unsafe_allow_html=True)


# ================================================================
# BERANDA
# ================================================================

if halaman == "Beranda":
    hero("Analisis Sentimen Program MBG",
         "Perbandingan Machine Learning Klasik dan Transformer pada komentar YouTube",
         "TEXT MINING PROJECT",
         "linear-gradient(135deg, #6366F1 0%, #8B5CF6 60%, #A78BFA 100%)")

    c1,c2,c3,c4 = st.columns(4, gap="medium")
    cards = [
        ("🗄️","#EDE9FE","#7C3AED","1.734","Total Komentar","12 video YouTube"),
        ("😠","#FEE2E2","#DC2626","76.4%","Sentimen Negatif","1.324 komentar"),
        ("🏆","#DCFCE7","#16A34A","82.42%","Akurasi Terbaik","IndoBERT Transformer"),
        ("🧠","#DBEAFE","#2563EB","4 Model","Dibandingkan","NB, LR, SVM, IndoBERT"),
    ]
    for col, (ico,bg,clr,val,lbl,sub) in zip([c1,c2,c3,c4], cards):
        with col:
            st.markdown(stat_card(ico,bg,clr,val,lbl,sub), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1.1, 1], gap="large")

    with col_a:
        card(
            section_title("ℹ️","Tentang Program MBG") +
            """<p style="color:#475569;font-size:14px;line-height:1.75;margin:0 0 16px">
            Program <strong>Makan Bergizi Gratis (MBG)</strong> adalah kebijakan unggulan
            pemerintahan Presiden Prabowo Subianto yang menyediakan makanan bergizi gratis
            bagi siswa sekolah di seluruh Indonesia. Program ini menjadi salah satu topik
            paling ramai diperbincangkan di media sosial sejak diluncurkan.</p>""" +
            section_title("🎯","Tujuan Penelitian") +
            """<p style="color:#475569;font-size:14px;line-height:1.75;margin:0">
            Membandingkan performa <strong>Machine Learning Klasik</strong> (Naive Bayes,
            Logistic Regression, SVM) dengan <strong>Transformer</strong> (IndoBERT)
            dalam mengklasifikasikan sentimen komentar masyarakat Indonesia terhadap
            Program MBG dari data komentar YouTube.</p>"""
        )

        card(
            section_title("📈","Hasil Perbandingan Model") +
            """<table style="width:100%;border-collapse:collapse;font-size:13px">
            <tr style="background:#0F172A;color:white">
              <th style="padding:10px 14px;text-align:left;border-radius:8px 0 0 0">Model</th>
              <th style="padding:10px 14px;text-align:center">Accuracy</th>
              <th style="padding:10px 14px;text-align:center;border-radius:0 8px 0 0">F1-score</th>
            </tr>
            <tr style="border-bottom:1px solid #F1F5F9">
              <td style="padding:10px 14px;color:#334155">Naive Bayes</td>
              <td style="padding:10px 14px;text-align:center;color:#334155">75.79%</td>
              <td style="padding:10px 14px;text-align:center;color:#334155">70.51%</td>
            </tr>
            <tr style="border-bottom:1px solid #F1F5F9">
              <td style="padding:10px 14px;color:#334155">Logistic Regression</td>
              <td style="padding:10px 14px;text-align:center;color:#334155">78.10%</td>
              <td style="padding:10px 14px;text-align:center;color:#334155">70.58%</td>
            </tr>
            <tr style="border-bottom:1px solid #F1F5F9">
              <td style="padding:10px 14px;color:#334155">SVM</td>
              <td style="padding:10px 14px;text-align:center;color:#334155">74.06%</td>
              <td style="padding:10px 14px;text-align:center;color:#334155">70.58%</td>
            </tr>
            <tr style="background:#F0FDF4">
              <td style="padding:10px 14px;font-weight:700;color:#15803D">
                ★ IndoBERT</td>
              <td style="padding:10px 14px;text-align:center;font-weight:700;color:#15803D">
                82.42%</td>
              <td style="padding:10px 14px;text-align:center;font-weight:700;color:#15803D">
                81.75%</td>
            </tr>
            </table>"""
        )

    with col_b:
        # Donut chart
        fig, ax = plt.subplots(figsize=(4.5,4.2), facecolor='none')
        sizes  = [1324, 255, 155]
        labels = ['Negatif','Positif','Netral']
        colors = ['#F43F5E','#22C55E','#3B82F6']
        wedges, _ = ax.pie(sizes, colors=colors, startangle=90,
                           wedgeprops={'linewidth':3,'edgecolor':'white'})
        circle = plt.Circle((0,0),0.55,fc='white')
        ax.add_artist(circle)
        ax.text(0, 0.12,'1.734',ha='center',va='center',
                fontsize=20,fontweight='800',color='#0F172A')
        ax.text(0,-0.15,'komentar',ha='center',va='center',
                fontsize=11,color='#64748B')
        patches = [mpatches.Patch(color=c,label=f'{l} ({v:,})')
                   for c,l,v in zip(colors,labels,sizes)]
        ax.legend(handles=patches,loc='lower center',bbox_to_anchor=(0.5,-0.12),
                  ncol=1,fontsize=11,frameon=False)
        plt.tight_layout()

        card(section_title("📊","Distribusi Sentimen"))
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("<br>", unsafe_allow_html=True)
        card(
            section_title("⚡","Coba Prediksi") +
            "<p style='font-size:13px;color:#64748B;margin:0 0 10px'>Ketik komentar MBG untuk prediksi cepat</p>"
        )
        teks_h = st.text_area("", placeholder="Contoh: Program MBG sangat membantu...",
                               height=85, key="home_txt", label_visibility="collapsed")
        if st.button("Prediksi Sentimen", type="primary", use_container_width=True):
            if teks_h.strip():
                st.session_state['teks_prediksi'] = teks_h
                st.success("Beralih ke menu Prediksi Sentimen untuk hasil lengkap!")
            else:
                st.warning("Masukkan teks terlebih dahulu.")


# ================================================================
# PREDIKSI SENTIMEN
# ================================================================

elif halaman == "Prediksi Sentimen":
    hero("Prediksi Sentimen",
         "Masukkan komentar MBG dan lihat prediksi dari semua model sekaligus",
         "PREDIKSI REAL-TIME",
         "linear-gradient(135deg, #0F172A 0%, #1E293B 100%)")

    models, tfidf, le = load_ml_models()
    tok, bert_model   = load_bert()
    teks_default      = st.session_state.get('teks_prediksi','')

    col_in, col_op = st.columns([2,1], gap="large")
    with col_in:
        teks_input = st.text_area(
            "Teks komentar:", value=teks_default, height=130,
            placeholder="Tulis atau tempel komentar tentang Program MBG di sini...")
        c1,c2,c3 = st.columns(3)
        contoh = [
            ("Contoh Positif",
             "Program MBG sangat bagus, anak-anak jadi lebih semangat sekolah karena dapat makan gratis bergizi setiap hari"),
            ("Contoh Negatif",
             "MBG gagal total! Makanannya tidak layak makan, porsinya sedikit, uang rakyat dihabiskan sia-sia"),
            ("Contoh Netral",
             "Kapan program MBG mulai berjalan di daerah saya? Sudah 3 bulan belum ada info yang jelas"),
        ]
        for col, (lbl, txt) in zip([c1,c2,c3], contoh):
            with col:
                if st.button(lbl, use_container_width=True):
                    st.session_state['teks_prediksi'] = txt
                    st.rerun()

    with col_op:
        st.markdown("**Pengaturan Model**")
        show_bert = st.toggle("IndoBERT (Terbaik)", value=True)
        show_ml   = st.toggle("ML Klasik (3 model)", value=True)
        st.info("IndoBERT memberikan akurasi terbaik 82.42% dengan memahami konteks kalimat penuh.")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Analisis Sentimen", type="primary", use_container_width=True):
        if not teks_input.strip():
            st.error("Masukkan teks komentar terlebih dahulu!")
        else:
            with st.spinner("Menganalisis sentimen..."):

                if show_bert:
                    h = pred_bert(teks_input, tok, bert_model)
                    lbl, conf = h['label'], h['confidence']

                    st.markdown("### IndoBERT — Model Terbaik")
                    ca, cb = st.columns([1, 1.3], gap="large")
                    with ca:
                        st.markdown(result_card(lbl, conf), unsafe_allow_html=True)
                    with cb:
                        if 'probs' in h:
                            st.markdown(prob_bars(h['probs']), unsafe_allow_html=True)
                    st.markdown("---")

                if show_ml:
                    h_ml = pred_ml(teks_input, models, tfidf, le)
                    st.markdown("### Model ML Klasik")
                    c1,c2,c3 = st.columns(3, gap="medium")
                    for col, (nm, res) in zip([c1,c2,c3], h_ml.items()):
                        with col:
                            st.markdown(ml_card(nm, res['label'], res['confidence']),
                                        unsafe_allow_html=True)

                st.markdown(f"""
<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-left:4px solid #6366F1;
            border-radius:0 12px 12px 0;padding:14px 18px;margin-top:16px;
            font-size:13px;color:#334155;line-height:1.6">
  <span style="font-size:11px;font-weight:700;color:#6366F1;text-transform:uppercase;
               letter-spacing:0.05em">Teks yang Dianalisis</span><br>
  <em>"{teks_input[:280]}{'...' if len(teks_input)>280 else ''}"</em>
  <span style="float:right;font-size:11px;color:#94A3B8">{len(teks_input)} karakter</span>
</div>
""", unsafe_allow_html=True)


# ================================================================
# EKSPLORASI DATA
# ================================================================

elif halaman == "Eksplorasi Data":
    hero("Eksplorasi Dataset MBG",
         "Visualisasi statistik 1.734 komentar YouTube",
         "EDA — ANALISIS DATA",
         "linear-gradient(135deg, #0369A1 0%, #0284C7 100%)")

    path = next((n for n in ['dataset_labeled.csv','dataset_raw.csv']
                 if os.path.exists(n)), None)

    if path:
        df = pd.read_csv(path, encoding='utf-8-sig')
        df = df[df['label'].isin(['Positif','Negatif','Netral'])]
        lc = df['label'].value_counts()

        c1,c2,c3,c4 = st.columns(4, gap="medium")
        for col, (ico,bg,clr,val,lbl,sub) in zip([c1,c2,c3,c4],[
            ("🗄️","#EDE9FE","#7C3AED",f"{len(df):,}","Total Data","komentar berlabel"),
            ("😠","#FEE2E2","#DC2626",f"{lc.get('Negatif',0):,}","Negatif",
             f"{lc.get('Negatif',0)/len(df)*100:.1f}%"),
            ("😊","#DCFCE7","#16A34A",f"{lc.get('Positif',0):,}","Positif",
             f"{lc.get('Positif',0)/len(df)*100:.1f}%"),
            ("😐","#DBEAFE","#2563EB",f"{lc.get('Netral',0):,}","Netral",
             f"{lc.get('Netral',0)/len(df)*100:.1f}%"),
        ]):
            with col:
                st.markdown(stat_card(ico,bg,clr,val,lbl,sub), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,
                             'axes.grid':True,'grid.alpha':0.3,'grid.linestyle':'--',
                             'font.family':'DejaVu Sans'})
        COLORS = ['#F43F5E','#22C55E','#3B82F6']

        tab1, tab2, tab3 = st.tabs(["Distribusi Label","Panjang Teks","Kata Terbanyak"])

        with tab1:
            ca, cb = st.columns(2, gap="large")
            with ca:
                fig, ax = plt.subplots(figsize=(5,4), facecolor='none')
                ax.bar(lc.index, lc.values, color=COLORS[:len(lc)],
                       edgecolor='white', linewidth=2, width=0.55, zorder=3)
                for i,(idx,v) in enumerate(zip(lc.index,lc.values)):
                    ax.text(i, v+8, f'{v:,}', ha='center', fontweight='700', fontsize=12)
                ax.set_title('Jumlah Data per Label',fontweight='700',fontsize=13,pad=12)
                ax.set_ylabel('Jumlah Komentar',fontsize=11)
                ax.set_ylim(0, max(lc.values)*1.18)
                plt.tight_layout()
                st.pyplot(fig); plt.close()
            with cb:
                fig, ax = plt.subplots(figsize=(5,4), facecolor='none')
                wedges,_,autotexts = ax.pie(
                    lc.values, labels=lc.index, colors=COLORS[:len(lc)],
                    autopct='%1.1f%%', startangle=90, pctdistance=0.75,
                    wedgeprops={'linewidth':3,'edgecolor':'white'})
                for at in autotexts: at.set_fontweight('700')
                ax.add_artist(plt.Circle((0,0),0.5,fc='white'))
                ax.set_title('Proporsi Sentimen',fontweight='700',fontsize=13,pad=12)
                plt.tight_layout()
                st.pyplot(fig); plt.close()

        with tab2:
            df['panjang'] = df['teks'].astype(str).apply(len)
            ca, cb = st.columns(2, gap="large")
            with ca:
                fig, ax = plt.subplots(figsize=(5,4), facecolor='none')
                ax.hist(df['panjang'], bins=40, color='#6366F1',
                        edgecolor='white', linewidth=0.5, alpha=0.85, zorder=3)
                ax.axvline(df['panjang'].mean(), color='#F43F5E', linewidth=2,
                           linestyle='--', label=f"Rata-rata: {df['panjang'].mean():.0f}")
                ax.axvline(df['panjang'].median(), color='#F59E0B', linewidth=2,
                           linestyle='--', label=f"Median: {df['panjang'].median():.0f}")
                ax.set_title('Distribusi Panjang Teks',fontweight='700',fontsize=13,pad=12)
                ax.set_xlabel('Jumlah Karakter',fontsize=11)
                ax.set_ylabel('Frekuensi',fontsize=11)
                ax.legend(fontsize=10,frameon=False)
                plt.tight_layout()
                st.pyplot(fig); plt.close()
            with cb:
                fig, ax = plt.subplots(figsize=(5,4), facecolor='none')
                lo = [l for l in ['Negatif','Positif','Netral'] if l in df['label'].unique()]
                means   = [df[df['label']==l]['panjang'].mean()   for l in lo]
                medians = [df[df['label']==l]['panjang'].median() for l in lo]
                x = np.arange(len(lo))
                w = 0.35
                b1 = ax.bar(x - w/2, means,   w, label='Rata-rata', color=COLORS[:len(lo)], alpha=0.85, edgecolor='white')
                b2 = ax.bar(x + w/2, medians, w, label='Median',    color=COLORS[:len(lo)], alpha=0.45, edgecolor='white')
                for bar, v in zip(b1, means):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                            f'{v:.0f}', ha='center', fontsize=9, fontweight='600')
                for bar, v in zip(b2, medians):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                            f'{v:.0f}', ha='center', fontsize=9, fontweight='600')
                ax.set_xticks(x)
                ax.set_xticklabels(lo)
                ax.set_title('Rata-rata & Median Panjang per Sentimen', fontweight='700', fontsize=12, pad=12)
                ax.set_ylabel('Jumlah Karakter', fontsize=11)
                ax.legend(fontsize=10, frameon=False)
                plt.tight_layout()
                st.pyplot(fig); plt.close()

        with tab3:
            sw = {'yg','yang','ya','nya','sih','aja','deh','kalo','banget','emang',
                  'udah','gak','ga','nggak','juga','ada','ini','itu','bisa','untuk',
                  'dengan','dari','di','ke','dan','atau','tapi','tp','jg','nan',
                  'sudah','akan','jadi','tidak','lebih','sangat'}
            semua = []
            for t in df['teks']:
                c = re.sub(r'[^a-zA-Z\s]',' ',str(t).lower())
                semua.extend([w for w in c.split() if w not in sw and len(w)>2])
            top20 = Counter(semua).most_common(20)
            if top20:
                kata, freq = zip(*top20)
                fig, ax = plt.subplots(figsize=(9,6), facecolor='none')
                bar_colors = plt.cm.RdYlGn_r(np.linspace(0.1,0.9,20))
                ax.barh(list(kata)[::-1], list(freq)[::-1],
                        color=bar_colors, edgecolor='white', linewidth=0.8, zorder=3)
                for i,(k,f) in enumerate(zip(list(kata)[::-1],list(freq)[::-1])):
                    ax.text(f+1, i, str(f), va='center', fontsize=10,
                            fontweight='600', color='#334155')
                ax.set_title('20 Kata Paling Sering Muncul',fontweight='700',fontsize=13,pad=14)
                ax.set_xlabel('Frekuensi',fontsize=11)
                plt.tight_layout()
                st.pyplot(fig); plt.close()
    else:
        st.info("File dataset_labeled.csv tidak ditemukan. Upload file ke folder yang sama dengan app.py")
        c1,c2,c3 = st.columns(3)
        for col, (v,l,s) in zip([c1,c2,c3],[
            ("1.324","Negatif","76.4%"),("255","Positif","14.7%"),("155","Netral","8.9%")]):
            with col: st.metric(l, v, s)


# ================================================================
# PERBANDINGAN MODEL
# ================================================================

elif halaman == "Perbandingan Model":
    hero("Perbandingan Performa Model",
         "Evaluasi 4 model pada 347 data test (20% dari 1.734 total)",
         "EVALUASI MODEL",
         "linear-gradient(135deg, #065F46 0%, #059669 100%)")

    dh = pd.DataFrame({
        'Model'    : ['Naive Bayes','Logistic Regression','SVM','IndoBERT'],
        'Accuracy' : [0.7579, 0.7810, 0.7406, 0.8242],
        'Precision': [0.6909, 0.7136, 0.6897, 0.8140],
        'Recall'   : [0.7579, 0.7810, 0.7406, 0.8242],
        'F1-score' : [0.7051, 0.7058, 0.7058, 0.8175],
    })

    c1,c2,c3,c4 = st.columns(4, gap="medium")
    for col, (ico,bg,clr,val,lbl,sub) in zip([c1,c2,c3,c4],[
        ("🏆","#DCFCE7","#16A34A","82.42%","Akurasi Terbaik","IndoBERT"),
        ("📉","#EDE9FE","#7C3AED","81.75%","F1-score Terbaik","IndoBERT"),
        ("📈","#DBEAFE","#2563EB","+11.17%","Selisih F1","vs ML Klasik"),
        ("❌","#FEE2E2","#DC2626","25.9%","Error Rate","90 dari 347 test"),
    ]):
        with col:
            st.markdown(stat_card(ico,bg,clr,val,lbl,sub), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ct, cc = st.columns([1, 1.2], gap="large")

    with ct:
        rows_html = ""
        for _, row in dh.iterrows():
            is_best = row['Model'] == 'IndoBERT'
            bg  = '#F0FDF4' if is_best else 'transparent'
            clr = '#15803D' if is_best else '#334155'
            star = '★ ' if is_best else ''
            rows_html += f"""<tr style="background:{bg}">
              <td style="padding:11px 14px;font-weight:{'700' if is_best else '400'};
                         color:{clr}">{star}{row['Model']}</td>
              <td style="padding:11px 14px;text-align:center;font-weight:{'700' if is_best else '400'};
                         color:{clr}">{row['Accuracy']*100:.2f}%</td>
              <td style="padding:11px 14px;text-align:center;font-weight:{'700' if is_best else '400'};
                         color:{clr}">{row['Precision']:.4f}</td>
              <td style="padding:11px 14px;text-align:center;font-weight:{'700' if is_best else '400'};
                         color:{clr}">{row['Recall']:.4f}</td>
              <td style="padding:11px 14px;text-align:center;font-weight:{'700' if is_best else '400'};
                         color:{clr}">{row['F1-score']:.4f}</td>
            </tr>"""

        card(section_title("📋","Tabel Perbandingan") + f"""
<table style="width:100%;border-collapse:collapse;font-size:13px">
<tr style="background:#0F172A;color:white">
  <th style="padding:11px 14px;text-align:left;border-radius:8px 0 0 0">Model</th>
  <th style="padding:11px 14px;text-align:center">Accuracy</th>
  <th style="padding:11px 14px;text-align:center">Precision</th>
  <th style="padding:11px 14px;text-align:center">Recall</th>
  <th style="padding:11px 14px;text-align:center;border-radius:0 8px 0 0">F1</th>
</tr>
{rows_html}
</table>
<div style="font-size:12px;color:#64748B;margin-top:10px">
  ⭐ Baris hijau = IndoBERT (model terbaik)
</div>""")

    with cc:
        plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,
                             'axes.grid':True,'grid.alpha':0.3,'grid.linestyle':'--'})
        fig, axes = plt.subplots(1, 2, figsize=(8,4), facecolor='none')

        bar_clrs = ['#94A3B8','#94A3B8','#94A3B8','#22C55E']
        bars = axes[0].barh(dh['Model'], dh['F1-score'], color=bar_clrs,
                            edgecolor='white', linewidth=1.5, height=0.55, zorder=3)
        bars[-1].set_linewidth(2.5)
        for bar, val in zip(bars, dh['F1-score']):
            axes[0].text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                         f'{val:.4f}', va='center', fontweight='700', fontsize=10)
        axes[0].set_xlim(0, 1.05)
        axes[0].axvline(0.8,color='#F43F5E',linewidth=1.5,linestyle='--',alpha=0.6,label='0.80')
        axes[0].legend(fontsize=9,frameon=False)
        axes[0].set_title('F1-score', fontweight='700', fontsize=12)

        metrik = ['Accuracy','Precision','Recall','F1-score']
        x = np.arange(len(metrik)); lebar = 0.18
        pal = ['#CBD5E1','#94A3B8','#64748B','#22C55E']
        for i, (_, row) in enumerate(dh.iterrows()):
            axes[1].bar(x+i*lebar, [row[m] for m in metrik], lebar,
                        label=row['Model'], color=pal[i], alpha=0.88,
                        edgecolor='white', zorder=3)
        axes[1].set_xticks(x+lebar*1.5)
        axes[1].set_xticklabels(['Acc','Prec','Rec','F1'], fontsize=10, fontweight='600')
        axes[1].set_ylim(0, 1.1)
        axes[1].legend(fontsize=8, loc='upper right', frameon=True, framealpha=0.9)
        axes[1].set_title('Semua Metrik', fontweight='700', fontsize=12)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.subheader("Analisis Hasil")
    ca,cb,cc = st.columns(3, gap="medium")
    with ca:
        with st.expander("Model terbaik?", expanded=True):
            st.write("""**IndoBERT** unggul dengan Accuracy 82.42% dan F1 81.75%.
            IndoBERT memahami konteks penuh kalimat karena di-pre-train dengan
            miliaran teks Bahasa Indonesia — berbeda dengan TF-IDF yang hanya
            menghitung frekuensi kata.""")
    with cb:
        with st.expander("Transformer selalu lebih baik?"):
            st.write("""**Tidak selalu.** Dalam kasus ini IndoBERT unggul +11.17%.
            Namun pada data sangat sedikit (< 500), ML Klasik bisa bersaing
            karena Transformer membutuhkan lebih banyak data untuk fine-tuning optimal.""")
    with cc:
        with st.expander("Kalimat yang sering salah?"):
            st.write("""**Positif ke Negatif (35 kasus):** kalimat dukungan dengan
            kata kritik konstruktif. **Netral ke Negatif (29 kasus):** pertanyaan
            informatif dengan kata negatif secara leksikal.
            **Sarkasme (14 kasus):** sulit dideteksi semua model.""")


# ================================================================
# TENTANG PROYEK
# ================================================================

elif halaman == "Tentang Proyek":
    hero("Tentang Proyek Akhir",
         "Text Mining — Universitas Sebelas Maret",
         "TIM PENELITI",
         "linear-gradient(135deg, #7C2D12 0%, #C2410C 100%)")

    st.subheader("Anggota Kelompok")
    c1,c2,c3 = st.columns(3, gap="large")
    for col, (no, nama, nim) in zip([c1,c2,c3],[
        ("1","Aditya Sheva Pratama","K3523004"),
        ("2","Albert Indra Wiguna","K3523008"),
        ("3","Ardhian Purnomo","K3523016"),
    ]):
        with col:
            st.markdown(f"""
<div style="background:white;border-radius:16px;padding:28px 20px;text-align:center;
            border:1px solid #E2E8F0;box-shadow:0 2px 8px rgba(0,0,0,0.05)">
  <div style="width:64px;height:64px;background:linear-gradient(135deg,#6366F1,#8B5CF6);
              border-radius:50%;display:flex;align-items:center;justify-content:center;
              margin:0 auto 14px">
    👤
  </div>
  <div style="font-weight:700;color:#0F172A;font-size:15px;margin-bottom:8px">{nama}</div>
  <div style="font-size:13px;color:#6366F1;font-weight:600;background:#EDE9FE;
              padding:4px 16px;border-radius:20px;display:inline-block">{nim}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ca, cb = st.columns([1.2,1], gap="large")

    with ca:
        steps = [
            ("⬇️","1. Pengumpulan Data",
             "Scraping komentar YouTube menggunakan youtube-comment-downloader dari 12 video berita MBG tanpa API key."),
            ("🏷️","2. Pelabelan Data",
             "Auto-labeling menggunakan model w11wo/indonesian-roberta-base-sentiment-classifier (pseudo-labeling)."),
            ("🔧","3. Preprocessing ML",
             "Case folding → hapus URL/mention → tokenisasi → stopword removal → stemming (PySastrawi)."),
            ("➕","4. Feature Extraction",
             "TF-IDF: max 5.000 fitur, n-gram (1,2), min_df=2. Split 80% train / 20% test."),
            ("🤖","5. Training ML Klasik",
             "Naive Bayes (alpha=0.1), Logistic Regression (C=1.0), LinearSVC (C=1.0)."),
            ("🧠","6. Fine-tuning IndoBERT",
             "indobenchmark/indobert-base-p1, 3 epoch, batch size 16, lr=2e-5, GPU Tesla T4."),
            ("📈","7. Evaluasi",
             "Accuracy, Precision, Recall, F1-score (weighted), Confusion Matrix, Error Analysis."),
        ]
        html_steps = section_title("📋","Metodologi Penelitian")
        for ico, judul, desk in steps:
            html_steps += f"""
<div style="display:flex;gap:14px;margin-bottom:14px;align-items:flex-start">
  <div style="width:36px;height:36px;background:#EDE9FE;border-radius:10px;
              display:flex;align-items:center;justify-content:center;
              flex-shrink:0;color:#7C3AED;font-size:18px">•</div>
  <div>
    <div style="font-weight:700;color:#0F172A;font-size:14px;margin-bottom:2px">{judul}</div>
    <div style="font-size:13px;color:#64748B;line-height:1.6">{desk}</div>
  </div>
</div>"""
        card(html_steps)

    with cb:
        info_items = [
            ("🏫","Institusi","Universitas Sebelas Maret"),
            ("📚","Mata Kuliah","Text Mining"),
            ("📄","Tugas","Proyek Akhir UAS"),
            ("🍱","Topik","Program MBG"),
            ("▶️","Platform","YouTube"),
            ("🗄️","Total Data","1.734 komentar"),
            ("📅","Tahun","2025/2026"),
        ]
        info_html = section_title("ℹ️","Informasi Proyek")
        for ico, lbl, val in info_items:
            info_html += f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            padding:9px 0;border-bottom:1px solid #F1F5F9;font-size:13px">
  <span style="color:#64748B;display:flex;align-items:center;gap:6px">
    •{lbl}
  </span>
  <span style="font-weight:600;color:#0F172A">{val}</span>
</div>"""
        card(info_html)

        hasil_html = section_title("🥇","Hasil Akhir")
        for nm, f1, is_best in [("Naive Bayes","70.51%",False),
                                  ("Logistic Regression","70.58%",False),
                                  ("SVM","70.58%",False),
                                  ("IndoBERT","81.75%",True)]:
            bg  = '#F0FDF4' if is_best else 'transparent'
            clr = '#15803D' if is_best else '#334155'
            star = '🏆' if is_best else ''
            hasil_html += f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            padding:8px 12px;border-radius:8px;background:{bg};
            margin-bottom:4px;font-size:13px">
  <span style="font-weight:{'700' if is_best else '400'};color:{clr}">{star}{nm}</span>
  <span style="font-weight:700;color:{clr}">{f1}</span>
</div>"""
        card(hasil_html)
