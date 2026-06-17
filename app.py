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

with tab1:
    st.subheader("Input Data Pengecekan Analisis Risiko")
    st.caption("Silakan masukkan data finansial nasabah. Format Rp akan terkonversi otomatis di bawah kotak input.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # PENDAPATAN PEMOHON
        inc_raw = st.text_input("Pendapatan Bulanan Pemohon (Ketik Angka Saja)", value="6000000")
        # Logika pemformatan otomatis secara real-time di bawah kotak
        try:
            inc_val = int(inc_raw) if inc_raw else 0
            st.markdown(f"**Format Terkonversi:** `Rp{inc_val:,}`".replace(",", "."))
        except ValueError:
            st.error("Masukkan angka yang valid!")
            inc_val = 0
        income = inc_val / 1000

        st.markdown("<br>", unsafe_allow_html=True)
        
        # PLAFON PINJAMAN
        loan_raw = st.text_input("Plafon Pinjaman yang Diajukan (Ketik Angka Saja)", value="150000000")
        try:
            loan_val = int(loan_raw) if loan_raw else 0
            st.markdown(f"**Format Terkonversi:** `Rp{loan_val:,}`".replace(",", "."))
        except ValueError:
            st.error("Masukkan angka yang valid!")
            loan_val = 0
        loan_amt = loan_val / 1000000
        st.caption("ℹ️ *Plafon Pinjaman: Total nominal uang maksimal yang diajukan oleh nasabah.*")
        
    with col2:
        # PENDAPATAN PASANGAN
        co_raw = st.text_input("Pendapatan Bulanan Pasangan / Penjamin (Ketik Angka Saja)", value="0")
        try:
            co_val = int(co_raw) if co_raw else 0
            st.markdown(f"**Format Terkonversi:** `Rp{co_val:,}`".replace(",", "."))
        except ValueError:
            st.error("Masukkan angka yang valid!")
            co_val = 0
        co_income = co_val / 1000

        st.markdown("<br>", unsafe_allow_html=True)
        
        # SLIK OJK
        credit_history_input = st.selectbox("Hasil Pengecekan SLIK OJK (BI Checking)", ["Aman (Skor 1-2 / Lancar)", "Bermasalah (Skor 3-5 / Nunggak)"])
        credit_hist_val = 1.0 if credit_history_input == "Aman (Skor 1-2 / Lancar)" else 0.0
        st.caption("ℹ️ *Pengecekan SLIK OJK dilakukan manual oleh Petugas Bank via sistem OJK.*")

    st.markdown("---")
    
    if st.button("🔍 Analisis Kelayakan Pinjaman", type="primary", use_container_width=True):
        input_data = [[income, co_income, loan_amt, credit_hist_val]]
        prediksi = model.predict(input_data)
        
        formatted_loan = f"Rp{loan_val:,}".replace(",", ".")
        formatted_income = f"Rp{inc_val:,}".replace(",", ".")
        
        st.markdown(f"### 💡 Keputusan Sistem untuk Pengajuan {formatted_loan}:")
        if prediksi[0] == 1:
            st.success(f"🎉 **STATUS: DISETUJUI (Risiko Rendah)** - Nasabah dengan pendapatan {formatted_income} layak untuk diberikan pinjaman.")
        else:
            st.error(f"⚠️ **STATUS: DITOLAK (Risiko Tinggi)** - Nasabah dengan pendapatan {formatted_income} berisiko tinggi gagal bayar berdasarkan pola historis data.")

with tab2:
    st.subheader("Performa Model Machine Learning")
    st.dataframe(df.head(10))