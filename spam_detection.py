# Manipulasi data
import pandas as pd
import numpy as np

# Preprocessing teks
import re
import string

# NLP Bahasa Indonesia
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

# Visualisasi (jika dibutuhkan)
import matplotlib.pyplot as plt
import seaborn as sns

# Untuk deteksi encoding saat load dataset (jika perlu)
import chardet

# Load dataset email
email_df = pd.read_csv(r'C:\Users\LENOVO\projek akhir data science\email_spam_indo.csv')

# Load dataset SMS
sms_df = pd.read_csv(r'C:\Users\LENOVO\projek akhir data science\dataset_sms_spam_v1.csv')

print("Email Dataset:")
print(email_df.head(), "\n")
print("SMS Dataset:")
print(sms_df.head())

# Import library
import pandas as pd
import numpy as np
import re

# NLP Bahasa Indonesia
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset email
email_df = pd.read_csv(r'C:\Users\LENOVO\projek akhir data science\email_spam_indo.csv')

# Load dataset sms
sms_df = pd.read_csv(r'C:\Users\LENOVO\projek akhir data science\dataset_sms_spam_v1.csv')

# Tampilkan 5 data pertama
print(email_df.head())
print(sms_df.head())

print(email_df.columns)
print(sms_df.columns)

# Inisialisasi stemmer dan stopwords
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

# Fungsi preprocessing
def preprocess_text(text):
    if pd.isnull(text):
        return ""

    # Case folding
    text = text.lower()
    
    # Hilangkan karakter non-huruf (angka, simbol, tanda baca)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Tokenizing
    tokens = text.split()
    
    # Stopword removal dan stemming
    cleaned_tokens = []
    for word in tokens:
        if word not in stop_words:
            stemmed = stemmer.stem(word)
            cleaned_tokens.append(stemmed)
    
    # Gabungkan kembali menjadi satu string
    return ' '.join(cleaned_tokens)

# Uji coba preprocessing untuk 5 baris pertama saja
email_df_test = email_df.head(5)
sms_df_test = sms_df.head(5)

email_df_test['clean_text'] = email_df_test['Pesan'].apply(preprocess_text)
sms_df_test['clean_text'] = sms_df_test['Teks'].apply(preprocess_text)

print(email_df_test[['Pesan', 'clean_text']])
print(sms_df_test[['Teks', 'clean_text']])

from tqdm import tqdm
import sys
tqdm.pandas()

# Cek apakah stdout tersedia (untuk mendeteksi console atau GUI)
tqdm_disable = not sys.stdout or not sys.stdout.isatty()
tqdm.pandas(disable=tqdm_disable)

if tqdm_disable:
    email_df['clean_text'] = email_df['Pesan'].apply(preprocess_text)
    sms_df['clean_text'] = sms_df['Teks'].apply(preprocess_text)
else:
    email_df['clean_text'] = email_df['Pesan'].progress_apply(preprocess_text)
    sms_df['clean_text'] = sms_df['Teks'].progress_apply(preprocess_text)

email_df['clean_text'] = email_df['Pesan'].progress_apply(preprocess_text)
sms_df['clean_text'] = sms_df['Teks'].progress_apply(preprocess_text)

# Standarisasi email
email_df = email_df.rename(columns={'Kategori': 'label', 'clean_text': 'text'})
email_df = email_df[['text', 'label']]

# Standarisasi SMS
sms_df = sms_df.rename(columns={'Kategori': 'label', 'clean_text': 'text'})
sms_df = sms_df[['text', 'label']]

combined_df = pd.concat([email_df, sms_df], ignore_index=True)
print(combined_df['label'].value_counts())
combined_df.head()

# Mapping label teks ke angka
label_mapping = {'ham': 0, 'spam': 1}
combined_df['label'] = combined_df['label'].map(label_mapping)

# Hapus baris yang tidak bisa dimapping (NaN)
combined_df = combined_df.dropna(subset=['label'])

# Ubah ke integer
combined_df['label'] = combined_df['label'].astype(int)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

import joblib

# Load model dan vectorizer yang sudah dilatih
model = joblib.load('spam_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

def predict_message(message):
    clean = preprocess_text(message)
    vector = tfidf.transform([clean])
    result = model.predict(vector)[0]
    return 'SPAM' if result == 1 else 'HAM (Bukan Spam)'

from colorama import Fore, Style, init
init(autoreset=True)

def predict_message(message):
    clean = preprocess_text(message)
    vector = tfidf.transform([clean])
    proba = model.predict_proba(vector)[0]  # probabilitas kelas [ham, spam]
    result = model.predict(vector)[0]
    
    label = 'SPAM' if result == 1 else 'HAM (Bukan Spam)'
    confidence = proba[result] * 100
    
    # Warna output: merah untuk spam, hijau untuk ham
    color = Fore.RED if result == 1 else Fore.GREEN
    
    return f"{color}{label} dengan confidence {confidence:.2f}%{Style.RESET_ALL}"

# Uji coba
contoh_pesan = "Selamat! Anda memenangkan hadiah Rp10 juta. Klik link ini untuk klaim."
hasil = predict_message(contoh_pesan)
print("Hasil Prediksi:", hasil)

import csv
from datetime import datetime

# Fungsi untuk simpan data prediksi ke file history.csv
def save_prediction(message, result, filename='history.csv'):
    # Jika file belum ada, header akan ditulis otomatis (mode 'a' = append)
    try:
        with open(filename, 'x', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Message', 'Prediction'])
    except FileExistsError:
        pass  # file sudah ada, lanjut append

    # Simpan data prediksi
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message, result])

# Fungsi untuk baca dan tampilkan isi history.csv
def show_history(filename='history.csv'):
    try:
        with open(filename, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                print(row)
    except FileNotFoundError:
        print("File history.csv belum ada. Belum ada prediksi yang disimpan.")

# Contoh fungsi prediksi yang memanggil save_prediction
def predict_message_with_history(message):
    # Simulasi prediksi (ganti ini dengan fungsi predict_message kamu)
    hasil_prediksi = predict_message(message)  # Pastikan fungsi predict_message sudah ada

    # Simpan ke history log
    save_prediction(message, hasil_prediksi)

    return hasil_prediksi

# Contoh pemakaian
pesan = "Selamat! Anda memenangkan hadiah Rp10 juta. Klik link ini untuk klaim."
hasil = predict_message_with_history(pesan)
print("Hasil Prediksi:", hasil)

print("\nRiwayat Prediksi:")
show_history()

import re

# List domain pendek/umum untuk dicurigai (bisa ditambah)
phishing_domains = [
    "bit.ly", "tinyurl.com", "adf.ly", "ow.ly", "shorte.st",
    "t.co", "goo.gl", "rb.gy", "rebrand.ly"
]


def contains_phishing_link(text):
    url_pattern = r"http[s]?://[^\s]+"
    urls = re.findall(url_pattern, text)

    suspicious_keywords = ['free', 'promo', 'claim', 'secure', 'login', 'update', 'verify']
    suspicious_extensions = ['.gq', '.tk', '.ml', '.cf', '.ga']

    for url in urls:
        # Cek domain mencurigakan
        for ext in suspicious_extensions:
            if ext in url:
                return True

        # Cek kata kunci mencurigakan dalam URL
        for keyword in suspicious_keywords:
            if keyword in url.lower():
                return True

    return False

import tkinter as tk
from tkinter import scrolledtext, messagebox
from datetime import datetime
import csv

# Fungsi prediksi (pastikan model, tfidf, preprocess_text sudah di-load)
def predict_message_with_confidence(message):
    clean = preprocess_text(message)
    vector = tfidf.transform([clean])
    
    prob = model.predict_proba(vector)[0]
    result = model.predict(vector)[0]
    confidence = prob[result] * 100
    label = "SPAM" if result == 1 else "HAM (Bukan Spam)"
    return label, confidence

# Simpan ke history dan tampilkan hasil
def predict_message_with_history(message):
    clean = preprocess_text(message)
    vector = tfidf.transform([clean])
    result = model.predict(vector)[0]
    confidence = max(model.predict_proba(vector)[0]) * 100
    label = 'SPAM' if result == 1 else 'HAM (Bukan Spam)'

    phishing_flag = contains_phishing_link(message)
    phishing_info = "⚠️ Phishing Link Terdeteksi!" if phishing_flag else "✅ Tidak ada tautan mencurigakan."

    # Simpan ke riwayat
    with open('history.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now(), message, label, f"{confidence:.2f}%", phishing_info])

    # Gabungkan hasil
    return f"{label} dengan confidence {confidence:.2f}%\n{phishing_info}"

# Fungsi saat klik "Prediksi"
def on_predict():
    pesan = text_input.get("1.0", tk.END).strip()
    if pesan == "":
        messagebox.showwarning("Warning", "Pesan tidak boleh kosong!")
        return

    hasil = predict_message_with_history(pesan)

    warna = "red" if "SPAM" in hasil else "green"
    result_label.config(text=hasil, fg=warna)

# Tampilkan isi file history.csv
def on_show_history():
    try:
        with open('history.csv', 'r', encoding='utf-8') as file:
            content = file.read()
        history_window = tk.Toplevel(root)
        history_window.title("Riwayat Prediksi")
        text = scrolledtext.ScrolledText(history_window, width=80, height=20)
        text.pack()
        text.insert(tk.END, content)
        text.config(state='disabled')
    except FileNotFoundError:
        messagebox.showinfo("Info", "Belum ada riwayat prediksi yang tersimpan.")

# ==== UI START ====
root = tk.Tk()
root.title("Aplikasi Deteksi Spam")
root.geometry("600x400")

tk.Label(root, text="Masukkan pesan email/SMS:", font=('Arial', 11)).pack(pady=5)

text_input = scrolledtext.ScrolledText(root, width=60, height=8, font=('Arial', 10))
text_input.pack(pady=5)

btn_predict = tk.Button(root, text="Prediksi", command=on_predict, font=('Arial', 10, 'bold'), bg='blue', fg='white')
btn_predict.pack(pady=5)

result_label = tk.Label(root, text="", font=('Arial', 12, 'bold'))
result_label.pack(pady=5)

btn_history = tk.Button(root, text="Tampilkan Riwayat Prediksi", command=on_show_history)
btn_history.pack(pady=10)

root.mainloop()