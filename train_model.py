import os
import pandas as pd
import numpy as np
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

# Pastikan nltk_data ada
nltk.download('stopwords')

# Path dataset (ganti dengan path relatif kalau perlu)
email_path = r'C:\Users\LENOVO\projek akhir data science\email_spam_indo.csv'
sms_path = r'C:\Users\LENOVO\projek akhir data science\dataset_sms_spam_v1.csv'

# Load dataset
email_df = pd.read_csv(email_path)
sms_df = pd.read_csv(sms_path)

# Inisialisasi Stemmer dan Stopwords
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

# Fungsi preprocessing
def preprocess_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    cleaned_tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(cleaned_tokens)

# Preprocessing dengan tqdm
tqdm.pandas()
email_df['clean_text'] = email_df['Pesan'].progress_apply(preprocess_text)
sms_df['clean_text'] = sms_df['Teks'].progress_apply(preprocess_text)

# Standarisasi kolom
email_df = email_df.rename(columns={'Kategori': 'label', 'clean_text': 'text'})[['text', 'label']]
sms_df = sms_df.rename(columns={'Kategori': 'label', 'clean_text': 'text'})[['text', 'label']]
combined_df = pd.concat([email_df, sms_df], ignore_index=True)

# Mapping label ke angka
label_mapping = {'ham': 0, 'spam': 1}
combined_df['label'] = combined_df['label'].map(label_mapping)

# Hapus baris dengan NaN (label tidak dikenali)
combined_df = combined_df.dropna(subset=['label'])
combined_df['label'] = combined_df['label'].astype(int)
print(combined_df['label'].value_counts(dropna=False))

# TF-IDF dan Model
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(combined_df['text'])
y = combined_df['label']

# Split data dan training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluasi
from sklearn.metrics import accuracy_score, classification_report
y_pred = model.predict(X_test)
print("Accuracy (80% training):", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# === Latih ulang dengan 100% data untuk disimpan ===
model.fit(X, y)  # gunakan semua data

# Simpan model dan vectorizer ke file
joblib.dump(model, 'spam_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("Model FINAL dilatih dengan 100% data dan disimpan ke spam_model.pkl")