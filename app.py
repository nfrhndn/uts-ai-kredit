import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# 1. KONFIGURASI HALAMAN (Biar lebih lebar & rapi)
# ==========================================
st.set_page_config(page_title="Dashboard Risiko Kredit", page_icon="🏦", layout="wide")

st.title("🏦 Dashboard Internal: Analisis Risiko Kredit")
st.markdown("Sistem Pendukung Keputusan (DSS) berbasis **Machine Learning (Random Forest)** untuk Petugas Bank / Credit Analyst.")
st.markdown("---")

# ==========================================
# 2. LOAD DATA & PREPROCESSING
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_kredit.csv")
    features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Credit_History', 'Loan_Status']
    df = df[features].dropna()
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
    return df

df = load_data()

# ==========================================
# 3. TRAINING MODEL & EVALUASI
# ==========================================
X = df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Credit_History']]
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Menghitung akurasi model untuk dipamerkan ke dosen
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)

# ==========================================
# 4. MEMBUAT TAMPILAN TABS (UI/UX Flow Lebih Profesional)
# ==========================================
tab1, tab2 = st.tabs(["📝 Form Evaluasi Nasabah", "🧠 Analisis Model AI"])

# ---------- TAB 1: FORM UTAMA ----------
with tab1:
    st.subheader("Input Data Pengecekan")
    st.caption("Silakan masukkan data finansial nasabah dan hasil SLIK OJK.")
    
    # Membagi form menjadi 2 kolom agar tidak memanjang ke bawah
    col1, col2 = st.columns(2)
    
    with col1:
        income = st.number_input("Pendapatan Pemohon (Ribuan Rp)", min_value=0, value=5000, help="Contoh: 5000 = Rp 5.000.000")
        loan_amt = st.number_input("Plafon Pinjaman yang Diajukan", min_value=0, value=150)
        
    with col2:
        co_income = st.number_input("Pendapatan Pasangan/Penjamin (Ribuan Rp)", min_value=0, value=0)
        credit_history_input = st.selectbox("Hasil SLIK OJK (BI Checking)", ["Aman (Skor 1-2 / Lancar)", "Bermasalah (Skor 3-5 / Nunggak)"])
        credit_hist_val = 1.0 if credit_history_input == "Aman (Skor 1-2 / Lancar)" else 0.0

    st.markdown("<br>", unsafe_allow_html=True) # Memberi jarak
    
    # Tombol dibuat lebar penuh (use_container_width)
    if st.button("🔍 Analisis Kelayakan Pinjaman", type="primary", use_container_width=True):
        input_data = [[income, co_income, loan_amt, credit_hist_val]]
        prediksi = model.predict(input_data)
        
        st.markdown("### 💡 Keputusan Sistem:")
        if prediksi[0] == 1:
            st.success("🎉 **STATUS: DISETUJUI (Risiko Rendah)** - Nasabah memiliki rekam jejak dan kapasitas finansial yang layak untuk diberikan pinjaman.")
        else:
            st.error("⚠️ **STATUS: DITOLAK (Risiko Tinggi)** - Nasabah berisiko gagal bayar berdasarkan pola historis.")

# ---------- TAB 2: METRIK AI (Biar Dosen Yakin Ini Aplikasi AI) ----------
with tab2:
    st.subheader("Performa Model Machine Learning")
    st.info(f"**Tingkat Akurasi Model (Accuracy Score): {akurasi * 100:.2f}%**")
    st.write("Akurasi ini didapatkan dari pengujian algoritma Random Forest terhadap 20% data validasi (Test Set) dari total dataset.")
    
    st.markdown("---")
    st.markdown("#### Validasi Dataset (Syarat Ketentuan UTS)")
    st.dataframe(df.head(10))
    st.caption("Sumber Data Asli: Kaggle - Loan Prediction Problem Dataset")