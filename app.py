import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
import pickle 
import os 
import warnings

# --- KONSTANTA & KONFIGURASI ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (8,5)

MODEL_CLASSIFICATION_FILE = "model_lr_cls.pkl" 
DEFAULT_DATA_PATH = "Morning_Routine_Productivity_Dataset.csv" 

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


# --- 1. FUNGSI MEMUAT MODEL (Pre-trained) ---
# Menggunakan st.cache_data agar Streamlit tidak mencoba load saat startup
@st.cache_data
def load_pretrained_model(file_path):
    """Memuat model dari file path."""
    try:
        # PENTING: Membaca file model biner (rb)
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        # Mengembalikan None jika gagal
        return None

# --- 2. FUNGSI MEMUAT DATA DENGAN DUKUNGAN REPOSITORY ---
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

# --- 3. LOGIKA PEMROSESAN DATA (FE & Encoding) ---
def run_feature_engineering_and_encoding(df):
    """Melakukan imputasi, fitur baru, dan encoding."""
    df_fe = df.copy()
    
    # Imputasi
    df_fe["Sleep Duration (hrs)"] = df_fe["Sleep Duration (hrs)"].fillna(df_fe["Sleep Duration (hrs)"].median())
    df_fe["Meditation (mins)"] = df_fe["Meditation (mins)"].fillna(df_fe["Meditation (mins)"].median())
    df_fe["Exercise (mins)"] = df_fe["Exercise (mins)"].fillna(df_fe["Exercise (mins)"].median())
    df_fe["Breakfast Type"] = df_fe["Breakfast Type"].fillna(df_fe["Breakfast Type"].mode()[0])
    df_fe["Journaling (Y/N)"] = df_fe["Journaling (Y/N)"].fillna(df_fe["Journaling (Y/N)"].mode()[0])
    df_fe["Mood"] = df_fe["Mood"].fillna(df_fe["Mood"].mode()[0])
    
    # Fitur Baru
    df_fe["Productivity_Level"] = df_fe["Productivity_Score (1-10)"].apply(lambda x: "Produktif" if x >= 7 else "Tidak Produktif")
    df_fe["Healthy_Morning"] = np.where(
        (df_fe["Sleep Duration (hrs)"] >= 7) &
        (df_fe["Meditation (mins)"] >= 10) &
        (df_fe["Exercise (mins)"] >= 30) &
        (df_fe["Breakfast Type"] != "Skipped") &
        (df_fe["Journaling (Y/N)"] == "Yes"),
        "Yes", "No"
    )
    df_fe["Sleep_Category"] = pd.cut(df_fe["Sleep Duration (hrs)"], bins=[0,5,7,9,12], labels=["Kurang Tidur","Cukup Tidur","Ideal","Berlebihan"])
    
    # One-hot encoding dan seleksi fitur untuk Training
    df_fe['Is_Productive'] = (df_fe["Productivity_Score (1-10)"] >= 7).astype(int)
    df_encoded = pd.get_dummies(df_fe, columns=["Breakfast Type", "Journaling (Y/N)", "Mood"])
    
    cols_to_drop = ["Date", "Wake-up Time", "Work Start Time", "Notes", "Productivity_Level", "Healthy_Morning", "Sleep_Category"]
    df_final = df_encoded.drop(columns=[c for c in cols_to_drop if c in df_encoded.columns], errors='ignore')
    
    return df_final, df_fe 

# --- 4. JUDUL UTAMA & SIDEBAR ---
st.title("🚀 Morning Routine Productivity Predictor")
st.markdown("Aplikasi berbasis Streamlit untuk memprediksi tingkat produktivitas Anda menggunakan model **Logistic Regression**.")

st.sidebar.subheader("Status Model Prediksi")
# TIDAK MEMANGGIL load_pretrained_model di sini agar tidak langsung error saat startup
st.sidebar.caption(f"Model Klasifikasi (LR): {MODEL_CLASSIFICATION_FILE}")
st.sidebar.info("Model akan dimuat saat tombol 'Lakukan Prediksi' ditekan.")
st.sidebar.markdown("---")

# --- 5. LOGIKA PEMUATAN DATA UTAMA ---
uploaded_file = st.file_uploader(f"1. Unggah file CSV (opsional, menggunakan data default jika kosong)", type=['csv'])

if uploaded_file is not None:
    data_source = uploaded_file
    source_info = "File Upload"
elif os.path.exists(DEFAULT_DATA_PATH):
    data_source = DEFAULT_DATA_PATH
    source_info = "Default Repository"
else:
    data_source = None
    st.warning(f"Unggah file CSV ({DEFAULT_DATA_PATH}) untuk memulai, atau pastikan file tersebut ada di root repository Anda.")


if data_source is not None:
    df = load_df(data_source)
    
    if df is not None:
        # --- TABS: PREDIKSI vs TRAINING/EXPLORER ---
        tab1, tab2 = st.tabs(["⚡ Prediksi Produktivitas", "📊 Eksplorasi & Training"])

        with tab2: # Eksplorasi & Training
            st.header("Training Model (Logistic Regression)")
            st.info("Gunakan tab ini jika Anda ingin melatih model baru dari data yang diunggah dan menyimpannya.")
            
            # Logika Feature Engineering
            if st.button("Run feature engineering"):
                df_final_train, df_fe_display = run_feature_engineering_and_encoding(df)
                
                # SIMPAN HASIL KE SESSION STATE
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
            
            # --- PEMUATAN MODEL DI DALAM BLOK PREDIKSI ---
            lr_cls_model = load_pretrained_model(MODEL_CLASSIFICATION_FILE)
            
            if lr_cls_model is None:
                st.warning("Model prediksi belum dimuat. Silakan latih dan unggah model di tab 'Eksplorasi & Training'.")
            else:
                # --- UI INPUT ---
                colA, colB = st.columns(2)
                
                with colA:
                    sleep_duration = st.slider("Durasi Tidur (Jam)", min_value=4.0, max_value=10.0, value=7.5, step=0.1)
                    meditation_mins = st.slider("Meditasi (Menit)", min_value=0, max_value=60, value=15, step=5)
                    exercise_mins = st.slider("Olahraga (Menit)", min_value=0, max_value=120, value=30, step=5)
                
                with colB:
                    # NOTE: Options harus sama dengan kategori di data training
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
                    
                    # Inisialisasi DataFrame dummy dengan kolom yang diharapkan model
                    df_pred = pd.DataFrame(0, index=[0], columns=FEATURE_COLUMNS)
                    
                    # Isi nilai numerik
                    df_pred['Sleep Duration (hrs)'] = input_df['Sleep Duration (hrs)'][0]
                    df_pred['Meditation (mins)'] = input_df['Meditation (mins)'][0]
                    df_pred['Exercise (mins)'] = input_df['Exercise (mins)'][0]
                    
                    # Isi nilai kategorikal (sesuai OHE)
                    bt_col = f'Breakfast Type_{breakfast_type}'
                    if bt_col in FEATURE_COLUMNS:
                        df_pred[bt_col] = 1
                    
                    if journaling_yn == 'Yes':
                        df_pred['Journaling (Y/N)_Yes'] = 1
                        
                    mood_col = f'Mood_{mood}'
                    if mood_col in FEATURE_COLUMNS:
                        df_pred[mood_col] = 1

                    
                    # --- PREDIKSI ---
                    try:
                        prediction_raw = lr_cls_model.predict(df_pred[FEATURE_COLUMNS])[0]
                        prediction_prob = lr_cls_model.predict_proba(df_pred[FEATURE_COLUMNS])[0]
                        
                        if prediction_raw == 1:
                            result_text = "Produktif"
                            result_color = "green"
                            prob_value = prediction_prob[1] * 100
                        else:
                            result_text = "Tidak Produktif"
                            result_color = "red"
                            prob_value = prediction_prob[0] * 100
                            
                        
                        # --- TAMPILKAN HASIL ---
                        st.subheader("Hasil Prediksi")
                        st.markdown(f"""
                            <div style="background-color: {result_color}; color: white; padding: 15px; border-radius: 10px; text-align: center;">
                                <h1 style="margin: 0;">{result_text}</h1>
                                <p style="margin: 5px 0 0 0;">(Tingkat Keyakinan: {prob_value:.2f}%)</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.caption("Prediksi 'Produktif' berarti Skor Produktivitas diperkirakan 7 atau lebih.")
                        
                    except Exception as e:
                        st.error("Terjadi error saat melakukan prediksi. Pastikan model yang dimuat cocok dengan input fitur.")
                        st.exception(e)
