import streamlit as st
import pandas as pd
import numpy as np
import pickle 
import os 
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB 
# Catatan: Library visualisasi (matplotlib, seaborn) dihapus untuk fokus pada deployment
# Jika ingin ada visualisasi di tab Eksplorasi, silakan tambahkan kembali

# --- KONSTANTA & KONFIGURASI ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Nama file model yang akan digunakan untuk PREDIKSI
MODEL_CLASSIFICATION_FILE = "model_lr_cls.pkl" 
DEFAULT_DATA_PATH = "Morning_Routine_Productivity_Dataset.csv" 

# Daftar fitur yang digunakan untuk pelatihan (Harus SAMA dengan urutan saat pelatihan!)
FEATURE_COLUMNS = [
    'Sleep Duration (hrs)', 'Meditation (mins)', 'Exercise (mins)', 
    'Breakfast Type_Light', 'Breakfast Type_Protein-rich', 
    'Breakfast Type_Skipped', 'Journaling (Y/N)_Yes', 
    'Mood_Neutral', 'Mood_Sad'
]

# --- INISIALISASI STATUS SESI ---
if 'fe_done' not in st.session_state:
    st.session_state.fe_done = False
if 'df_final_train' not in st.session_state:
    st.session_state.df_final_train = None
if 'df_fe_display' not in st.session_state:
    st.session_state.df_fe_display = None


# --- FUNGSI MEMUAT MODEL (Pre-trained) ---
@st.cache_resource
def load_pretrained_model(file_path):
    """Memuat model dari file path."""
    try:
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        return None
    except Exception as e:
        st.sidebar.caption(f"Status {os.path.basename(file_path)}: GAGAL dimuat ({type(e).__name__}).")
        st.sidebar.caption("Pastikan model sudah dilatih dan di-upload ke GitHub.")
        return None

# --- FUNGSI MEMUAT DATA DENGAN DUKUNGAN REPOSITORY ---
@st.cache_data
def load_df(file_source):
    """Memuat DataFrame dari file yang diunggah atau dari path default."""
    if isinstance(file_source, str):
        try:
            df = pd.read_csv(file_source)
            st.sidebar.success(f"Dataset berhasil dimuat dari repository: {os.path.basename(file_source)}")
            return df
        except Exception as e:
            st.error(f"Gagal memuat file default dari GitHub. Error: {e}")
            return None
    else:
        df = pd.read_csv(file_source)
        st.sidebar.success("Dataset berhasil dimuat dari file yang diunggah.")
        return df

# --- LOGIKA PEMROSESAN DATA (FE & Encoding) ---
def run_feature_engineering_and_encoding(df):
    """Melakukan imputasi, fitur baru, dan encoding."""
    df_fe = df.copy()
    
    # Imputasi sederhana (Median untuk numerik, Mode untuk kategorikal)
    df_fe["Sleep Duration (hrs)"] = df_fe["Sleep Duration (hrs)"].fillna(df_fe["Sleep Duration (hrs)"].median())
    df_fe["Meditation (mins)"] = df_fe["Meditation (mins)"].fillna(df_fe["Meditation (mins)"].median())
    df_fe["Exercise (mins)"] = df_fe["Exercise (mins)"].fillna(df_fe["Exercise (mins)"].median())
    df_fe["Breakfast Type"] = df_fe["Breakfast Type"].fillna(df_fe["Breakfast Type"].mode()[0])
    df_fe["Journaling (Y/N)"] = df_fe["Journaling (Y/N)"].fillna(df_fe["Journaling (Y/N)"].mode()[0])
    df_fe["Mood"] = df_fe["Mood"].fillna(df_fe["Mood"].mode()[0])
    
    # Target Klasifikasi: Produktif (1) jika Skor >= 7, Tidak Produktif (0) jika < 7
    df_fe['Is_Productive'] = (df_fe["Productivity_Score (1-10)"] >= 7).astype(int)
    
    # One-hot encoding
    df_encoded = pd.get_dummies(df_fe, columns=["Breakfast Type", "Journaling (Y/N)", "Mood"])
    
    # Seleksi kolom fitur yang relevan (termasuk OHE, tapi drop satu kategori per variabel)
    # Kita menggunakan "Breakfast Type_Heavy", "Journaling (Y/N)_No", "Mood_Happy" sebagai referensi yang DITINGGALKAN
    cols_to_keep = [col for col in FEATURE_COLUMNS if col in df_encoded.columns]
    
    # Pastikan semua kolom yang diperlukan ada, jika tidak, tambahkan dengan nilai 0 (penting untuk konsistensi OHE)
    for col in FEATURE_COLUMNS:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    # Pilih dan atur ulang kolom sesuai urutan FEATURE_COLUMNS
    df_final = df_encoded[FEATURE_COLUMNS + ["Productivity_Score (1-10)", "Is_Productive"]].copy()
    
    return df_final, df_fe

# --- APLIKASI UTAMA ---

# Muat Model yang Sudah Dilatih
lr_cls_model = load_pretrained_model(MODEL_CLASSIFICATION_FILE)

# --- JUDUL UTAMA & SIDEBAR ---
st.title("🚀 Morning Routine Productivity Predictor")
st.markdown("Aplikasi berbasis Streamlit untuk memprediksi tingkat produktivitas Anda menggunakan model **Logistic Regression**.")

st.sidebar.subheader("Status Model Prediksi")
if lr_cls_model:
    st.sidebar.success(f"✅ Model: Logistic Regression dimuat.")
else:
    st.sidebar.error(f"❌ Model: {MODEL_CLASSIFICATION_FILE} tidak ditemukan. Silakan *train* model di tab 'Eksplorasi & Training' dan *upload* hasilnya.")

st.sidebar.markdown("---")


# --- LOGIKA PEMUATAN DATA UTAMA ---
uploaded_file = st.file_uploader(f"1. Unggah file CSV (opsional, menggunakan data default jika kosong)", type=['csv'])

if uploaded_file is not None:
    data_source = uploaded_file
elif os.path.exists(DEFAULT_DATA_PATH):
    data_source = DEFAULT_DATA_PATH
else:
    data_source = None

if data_source is not None:
    df = load_df(data_source)
    
    # --- TABS: PREDIKSI vs TRAINING/EXPLORER ---
    tab1, tab2 = st.tabs(["⚡ Prediksi Produktivitas", "📊 Eksplorasi & Training"])

    with tab2: # Eksplorasi & Training
        st.header("Training Model (Logistic Regression)")
        st.info("Gunakan tab ini jika Anda ingin melatih model baru dari data yang diunggah.")
        
        # Logika Feature Engineering
        if st.button("Run feature engineering"):
            df_final_train, df_fe_display = run_feature_engineering_and_encoding(df)
            
            st.session_state.df_final_train = df_final_train
            st.session_state.df_fe_display = df_fe_display
            st.session_state.fe_done = True
            
            st.success("Feature engineering selesai. Data siap untuk Training.")
            st.markdown("---")
            st.dataframe(st.session_state.df_final_train.head())

        # Logika Training
        if st.session_state.fe_done and st.session_state.df_final_train is not None:
            st.subheader("Train Logistic Regression")
            if st.button("Train, Evaluate, and Save Logistic Regression Model"):
                
                df_train_data = st.session_state.df_final_train
                
                # Data Klasifikasi (Target: Is_Productive)
                X_cls = df_train_data.drop(columns=["Productivity_Score (1-10)", "Is_Productive"], errors='ignore')
                y_cls = df_train_data["Is_Productive"]

                # Split Data Klasifikasi
                X_train_cls, X_val_cls, y_train_cls, y_val_cls = train_test_split(X_cls, y_cls, test_size=0.2, stratify=y_cls, random_state=42)
                
                st.markdown("#### Logistic Regression (Predict Produktif/Tidak)")
                # Inisialisasi dan latih Logistic Regression
                lr_cls = LogisticRegression(random_state=42, max_iter=1000)
                lr_cls.fit(X_train_cls, y_train_cls)
                y_val_pred_lr = lr_cls.predict(X_val_cls)
                accuracy_lr = accuracy_score(y_val_cls, y_val_pred_lr)
                st.write(f"Akurasi Model (Logistic Regression): **{accuracy_lr:.4f}**")
                
                # Menyimpan model Klasifikasi
                with open(MODEL_CLASSIFICATION_FILE, "wb") as f:
                    pickle.dump(lr_cls, f)
                st.success(f"Model Klasifikasi (LR) disimpan sebagai **{MODEL_CLASSIFICATION_FILE}**")
                st.warning("PENTING: Agar prediksi bekerja, Anda harus mengunduh file **model_lr_cls.pkl** ini dan mengunggahnya ke GitHub Anda!")

    with tab1: # Prediksi Produktivitas
        st.header("Masukkan Rutin Pagi Anda")
        
        if lr_cls_model is None:
            st.warning("Model prediksi belum dimuat. Silakan cek sidebar atau latih model di tab 'Eksplorasi & Training'.")
        else:
            # --- UI INPUT ---
            colA, colB = st.columns(2)
            
            with colA:
                sleep_duration = st.slider("Durasi Tidur (Jam)", min_value=4.0, max_value=10.0, value=7.5, step=0.1)
                meditation_mins = st.slider("Meditasi (Menit)", min_value=0, max_value=60, value=15, step=5)
                exercise_mins = st.slider("Olahraga (Menit)", min_value=0, max_value=120, value=30, step=5)
            
            with colB:
                breakfast_type = st.selectbox("Jenis Sarapan", ["Heavy", "Light", "Protein-rich", "Carb-rich", "Skipped"], index=0)
                journaling_yn = st.radio("Jurnal (Ya/Tidak)", ["Yes", "No"], index=0)
                mood = st.selectbox("Mood Saat Bangun", ["Happy", "Neutral", "Sad"], index=0)
            
            st.markdown("---")
            
            if st.button("Lakukan Prediksi", type="primary"):
                # --- PRE-PROCESSING INPUT PENGGUNA ---
                input_data = {
                    'Sleep Duration (hrs)': [sleep_duration],
                    'Meditation (mins)': [meditation_mins],
                    'Exercise (mins)': [exercise_mins],
                    'Breakfast Type': [breakfast_type],
                    'Journaling (Y/N)': [journaling_yn],
                    'Mood': [mood]
                }
                
                input_df = pd.DataFrame(input_data)
                
                # Lakukan One-Hot Encoding pada input
                # NOTE: kita harus membuat semua kolom yang diperlukan oleh model, meskipun nilainya 0
                
                # Inisialisasi DataFrame dummy dengan kolom yang diharapkan model
                df_pred = pd.DataFrame(0, index=[0], columns=FEATURE_COLUMNS)
                
                # Isi nilai numerik
                df_pred['Sleep Duration (hrs)'] = input_df['Sleep Duration (hrs)'][0]
                df_pred['Meditation (mins)'] = input_df['Meditation (mins)'][0]
                df_pred['Exercise (mins)'] = input_df['Exercise (mins)'][0]
                
                # Isi nilai kategorikal (Hanya yang TIDAK menjadi kolom referensi)
                # Breakfast Type
                bt_col = f'Breakfast Type_{breakfast_type}'
                if bt_col in FEATURE_COLUMNS:
                    df_pred[bt_col] = 1
                
                # Journaling
                if journaling_yn == 'Yes':
                    df_pred['Journaling (Y/N)_Yes'] = 1
                    
                # Mood
                mood_col = f'Mood_{mood}'
                if mood_col in FEATURE_COLUMNS:
                    df_pred[mood_col] = 1

                
                # --- PREDIKSI ---
                try:
                    prediction_raw = lr_cls_model.predict(df_pred[FEATURE_COLUMNS])[0]
                    prediction_prob = lr_cls_model.predict_proba(df_pred[FEATURE_COLUMNS])[0]
                    
                    # Konversi hasil
                    if prediction_raw == 1:
                        result_text = "Produktif"
                        result_color = "green"
                    else:
                        result_text = "Tidak Produktif"
                        result_color = "red"
                        
                    prob_productive = prediction_prob[1] * 100
                    
                    # --- TAMPILKAN HASIL ---
                    st.subheader("Hasil Prediksi")
                    st.markdown(f"""
                        <div style="background-color: {result_color}; color: white; padding: 15px; border-radius: 10px; text-align: center;">
                            <h1 style="margin: 0;">{result_text}</h1>
                            <p style="margin: 5px 0 0 0;">(Probabilitas Produktif: {prob_productive:.2f}%)</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.caption("Prediksi 'Produktif' berarti Skor Produktivitas diperkirakan 7 atau lebih.")
                    
                except Exception as e:
                    st.error("Terjadi error saat melakukan prediksi. Pastikan semua fitur yang diperlukan model sudah tersedia.")
                    st.exception(e)


# --- PENUTUP ---
else:
    st.info("Silakan unggah dataset CSV Anda atau letakkan file `Morning_Routine_Productivity_Dataset.csv` di root repository Anda untuk memulai.")
