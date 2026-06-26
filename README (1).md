# Aplikasi Analisis Sentimen MBG

Aplikasi Streamlit untuk klasifikasi sentimen komentar masyarakat terhadap Program Makan Bergizi Gratis (MBG).

## Cara Deploy ke Streamlit Community Cloud

### 1. Siapkan file di Colab (jalankan cell ini)

```python
# Export model ML dan TF-IDF dari notebook ke file pkl
import pickle

# Simpan TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Simpan Label Encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Simpan model ML klasik
with open('model_nb.pkl', 'wb') as f:
    pickle.dump(model_klasik['Naive Bayes'], f)

with open('model_lr.pkl', 'wb') as f:
    pickle.dump(model_klasik['Logistic Regression'], f)

with open('model_svm.pkl', 'wb') as f:
    pickle.dump(model_klasik['SVM'], f)

# Simpan IndoBERT (opsional - jika mau deploy dengan BERT)
trainer.model.save_pretrained('./indobert_output')
tokenizer.save_pretrained('./indobert_output')

# Download semua file pkl
from google.colab import files
for f in ['tfidf_vectorizer.pkl','label_encoder.pkl',
          'model_nb.pkl','model_lr.pkl','model_svm.pkl']:
    files.download(f)

# Download dataset berlabel
files.download('dataset_labeled.csv')
```

### 2. Struktur folder yang dibutuhkan

```
repo-github-kalian/
├── app.py                  ← file utama Streamlit
├── requirements.txt        ← library yang diperlukan
├── dataset_labeled.csv     ← dataset (opsional, untuk halaman EDA)
├── model_nb.pkl            ← Naive Bayes (opsional)
├── model_lr.pkl            ← Logistic Regression (opsional)
├── model_svm.pkl           ← SVM (opsional)
├── tfidf_vectorizer.pkl    ← TF-IDF vectorizer (opsional)
├── label_encoder.pkl       ← Label encoder (opsional)
└── indobert_output/        ← folder IndoBERT (opsional, besar ~400MB)
    ├── config.json
    ├── model.safetensors
    └── tokenizer files...
```

> **Catatan**: Jika file pkl tidak ada, aplikasi otomatis berjalan dalam
> mode demo dengan hasil prediksi acak untuk tujuan demonstrasi.

### 3. Upload ke GitHub

1. Buat repo baru di github.com (misal: `sentimen-mbg`)
2. Upload semua file di atas ke repo tersebut
3. Pastikan `app.py` dan `requirements.txt` ada di root repo

### 4. Deploy ke Streamlit Cloud

1. Buka https://streamlit.io/cloud
2. Login dengan akun GitHub
3. Klik **New app**
4. Pilih repo → branch: `main` → Main file: `app.py`
5. Klik **Deploy!**
6. Tunggu 2-5 menit → aplikasi live!

### 5. URL aplikasi

Setelah deploy, aplikasi bisa diakses di:
`https://[username]-[repo-name]-[random].streamlit.app`

Salin URL ini untuk dicantumkan di laporan dan video presentasi.
