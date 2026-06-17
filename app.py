import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. KONFIGURASI HALAMAN 
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
# 3. TRAINING MODEL
# ==========================================
X = df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Credit_History']]
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 4. TAMPILAN TABS UI
# ==========================================
tab1, tab2 = st.tabs(["📝 Form Evaluasi Nasabah", "🧠 Analisis Model AI"])

# ---------- TAB 1: FORM UTAMA ----------
with tab1:
    st.subheader("Input Data Pengecekan Analisis Risiko")
    st.caption("Silakan masukkan data finansial nasabah.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Menggunakan format bawaan Streamlit untuk pemisah ribuan otomatis di dalam kotak
        income_rp = st.number_input("Pendapatan Bulanan Pemohon (Format: Rp. Angka)", min_value=0, value=6000000, step=500000, format="%d")
        st.markdown(f"**Nilai Terkonfirmasi:** Rp {income_rp:,}".replace(",", "."))
        income = income_rp / 1000
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        loan_amt_rp = st.number_input("Plafon Pinjaman yang Diajukan (Format: Rp. Angka)", min_value=0, value=150000000, step=10000000, format="%d")
        st.markdown(f"**Nilai Terkonfirmasi:** Rp {loan_amt_rp:,}".replace(",", "."))
        loan_amt = loan_amt_rp / 1000000
        st.caption("ℹ️ *Plafon Pinjaman: Total nominal uang maksimal yang diajukan oleh nasabah.*")
        
    with col2:
        co_income_rp = st.number_input("Pendapatan Bulanan Pasangan / Penjamin (Format: Rp. Angka)", min_value=0, value=0, step=500000, format="%d")
        st.markdown(f"**Nilai Terkonfirmasi:** Rp {co_income_rp:,}".replace(",", "."))
        co_income = co_income_rp / 1000

        st.markdown("<br>", unsafe_allow_html=True)
        
        credit_history_input = st.selectbox("Hasil Pengecekan SLIK OJK (BI Checking)", ["Aman (Skor 1-2 / Lancar)", "Bermasalah (Skor 3-5 / Nunggak)"])
        credit_hist_val = 1.0 if credit_history_input == "Aman (Skor 1-2 / Lancar)" else 0.0

    st.markdown("---")
    
    if st.button("🔍 Analisis Kelayakan Pinjaman", type="primary", use_container_width=True):
        input_data = [[income, co_income, loan_amt, credit_hist_val]]
        prediksi = model.predict(input_data)
        
        formatted_loan = f"Rp{loan_amt_rp:,}".replace(",", ".")
        formatted_income = f"Rp{income_rp:,}".replace(",", ".")
        
        st.markdown(f"### 💡 Keputusan Sistem untuk Pengajuan {formatted_loan}:")
        if prediksi[0] == 1:
            st.success(f"🎉 **STATUS: DISETUJUI (Risiko Rendah)** - Nasabah dengan pendapatan {formatted_income} layak untuk diberikan pinjaman.")
        else:
            st.error(f"⚠️ **STATUS: DITOLAK (Risiko Tinggi)** - Nasabah dengan pendapatan {formatted_income} berisiko tinggi gagal bayar berdasarkan pola historis data.")

with tab2:
    st.subheader("Performa Model Machine Learning")
    st.dataframe(df.head(10))