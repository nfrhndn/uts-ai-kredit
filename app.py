import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. JUDUL & DESKRIPSI APLIKASI UI
# ==========================================
st.set_page_config(page_title="Prediksi Risiko Kredit", layout="centered")
st.title("🏦 Aplikasi Prediksi Persetujuan Pinjaman")
st.write("Aplikasi sederhana menggunakan **Machine Learning (Random Forest)** untuk memprediksi kelayakan pengajuan kredit/pinjaman nasabah berdasarkan data riwayat finansial.")
st.write("---")

# ==========================================
# 2. LOAD DATA & PREPROCESSING
# ==========================================
@st.cache_data
def load_data():
    # Baca file dataset
    df = pd.read_csv("dataset_kredit.csv")
    
    # Memilih fitur yang penting saja agar cepat dan simpel
    features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Credit_History', 'Loan_Status']
    df = df[features].dropna() # Hapus baris yang ada data kosong (NaN)
    
    # Ubah target Y/N menjadi 1/0
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
    return df

df = load_data()

# Tampilkan sampel data di UI untuk validasi dosen (Syarat tugas)
with st.expander("Klik untuk melihat Sampel Dataset (Validasi Data)"):
    st.dataframe(df.head())
    st.write("**Sumber Data:** Kaggle - Loan Prediction Problem Dataset")

# ==========================================
# 3. TRAINING MODEL MACHINE LEARNING
# ==========================================
# Pemisahan fitur (X) dan target (y)
X = df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Credit_History']]
y = df['Loan_Status']

# Split data testing & training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gunakan model Random Forest (Bukan AI API)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 4. FORM INPUT PENGGUNA (SIDEBAR)
# ==========================================
st.sidebar.header("📝 Masukkan Data Nasabah")

income = st.sidebar.number_input("Pendapatan Pemohon (USD)", min_value=0, value=5000)
co_income = st.sidebar.number_input("Pendapatan Penjamin/Pasangan (USD)", min_value=0.0, value=0.0)
loan_amt = st.sidebar.number_input("Jumlah Pinjaman yang Diajukan", min_value=0.0, value=150.0)

# Credit history (1.0 = Punya riwayat baik, 0.0 = Buruk/Tidak punya)
credit_history_input = st.sidebar.selectbox("Riwayat Kredit Memenuhi Syarat?", ["Ya", "Tidak"])
credit_hist_val = 1.0 if credit_history_input == "Ya" else 0.0

# ==========================================
# 5. TOMBOL PREDIKSI & HASIL
# ==========================================
if st.sidebar.button("🔍 Cek Kelayakan Pinjaman"):
    # Lakukan Prediksi
    input_data = [[income, co_income, loan_amt, credit_hist_val]]
    prediksi = model.predict(input_data)
    
    st.subheader("💡 Hasil Prediksi Sistem:")
    if prediksi[0] == 1:
        st.success("🎉 STATUS: **DISETUJUI** (Risiko Rendah)")
        st.write("Berdasarkan algoritma kami, nasabah ini memiliki kelayakan finansial yang baik untuk diberikan pinjaman.")
    else:
        st.error("⚠️ STATUS: **DITOLAK** (Risiko Tinggi)")
        st.write("Berdasarkan algoritma kami, nasabah ini berisiko tinggi gagal bayar.")