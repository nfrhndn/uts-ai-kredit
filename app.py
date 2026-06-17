import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. JUDUL & DESKRIPSI APLIKASI UI
# ==========================================
st.set_page_config(page_title="Dashboard Prediksi Kredit", layout="centered")
st.title("🏦 Dashboard Internal: Analisis Risiko Kredit")
st.write("Sistem Pendukung Keputusan (DSS) bagi **Petugas Bank / Credit Analyst** untuk mengevaluasi kelayakan pengajuan pinjaman nasabah secara otomatis menggunakan Machine Learning (Random Forest).")
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
st.sidebar.header("📝 Form Input Petugas Bank")
st.sidebar.caption("Simulasi input data dari Core Banking System")

# Asumsi nilai 5000 di dataset = Rp 5.000.000 (Skala Ribuan Rupiah)
income = st.sidebar.number_input("Pendapatan Pemohon (dalam Ribuan Rp)", min_value=0, value=5000, help="Contoh: Ketik 5000 untuk Rp 5.000.000")
co_income = st.sidebar.number_input("Pendapatan Pasangan/Penjamin (dalam Ribuan Rp)", min_value=0, value=0)
loan_amt = st.sidebar.number_input("Plafon Pinjaman yang Diajukan", min_value=0, value=150)

st.sidebar.markdown("---")
st.sidebar.write("**Integrasi Data Eksternal (Simulasi API):**")
# Mengubah kesan "input manual" menjadi kesan "pengecekan sistem"
credit_history_input = st.sidebar.selectbox("Hasil Pengecekan SLIK OJK (BI Checking)", ["Aman (Skor 1-2 / Lancar)", "Bermasalah (Skor 3-5 / Nunggak)"])
credit_hist_val = 1.0 if credit_history_input == "Aman (Skor 1-2 / Lancar)" else 0.0

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