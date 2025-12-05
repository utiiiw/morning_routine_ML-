import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import pickle 
import os 
import warnings

# Mengabaikan warning KMeans Inisialisasi
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- KONSTANTA ---
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (8,5)
MODEL_FILE_NAME = "model_rf.pkl" 
DEFAULT_DATA_PATH = "Morning_Routine_Productivity_Dataset.csv" 

# --- 1. FUNGSI MEMUAT MODEL (Pre-trained) ---
@st.cache_resource
def load_pretrained_model(file_path):
    """Memuat model Random Forest Regressor yang sudah dilatih (pkl)."""
    try:
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.sidebar.error(f"Error: File model '{file_path}' tidak ditemukan. Anda dapat melatih model baru di bawah.")
        return None
    except Exception as e:
        st.sidebar.error(f"Error saat memuat model: {e}")
        return None

# --- 2. FUNGSI MEMUAT DATA DENGAN DUKUNGAN REPOSITORY ---
@st.cache_data
def load_df(file_source):
    """Memuat DataFrame dari file yang diunggah atau dari path default."""
    if isinstance(file_source, str):
        try:
            df = pd.read_csv(file_source)
            st.sidebar.success(f"Dataset berhasil dimuat dari repository: {file_source}")
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
    
    # fitur baru (untuk tampilan)
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
    df_encoded = pd.get_dummies(df_fe, columns=["Breakfast Type", "Journaling (Y/N)", "Mood"], drop_first=True)
    drop_cols_train = ["Date", "Wake-up Time", "Work Start Time", "Notes"]
    
    # Menghapus kolom
    df_final = df_encoded.drop(columns=[c for c in drop_cols_train if c in df_encoded.columns], errors='ignore')
    
    return df_final, df_fe # Mengembalikan data FE untuk training dan data FE untuk display

# --- 4. JUDUL UTAMA & SIDEBAR (Menampilkan status model) ---
st.title("Morning Routine Productivity — Explorer & Model Demo")
st.markdown("Aplikasi untuk eksplorasi data, melatih model, dan memuat model yang sudah ada.")

rf_reg_model = load_pretrained_model(MODEL_FILE_NAME)
if rf_reg_model:
    st.sidebar.success(f"Model {MODEL_FILE_NAME} siap digunakan.")
    st.sidebar.subheader("Info Model")
    st.sidebar.caption(f"Estimator: {rf_reg_model.n_estimators}")
else:
    st.sidebar.info("Model tidak dimuat. Silakan latih model baru di bawah.")

st.sidebar.markdown("---")

# --- 5. LOGIKA PEMUATAN DATA UTAMA ---
uploaded_file = st.file_uploader(f"1. Upload file CSV (opsional, menggunakan data default jika kosong)", type=['csv'])

# Tentukan sumber data
if uploaded_file is not None:
    data_source = uploaded_file
    source_info = "File Upload"
elif os.path.exists(DEFAULT_DATA_PATH):
    data_source = DEFAULT_DATA_PATH
    source_info = "Default Repository"
else:
    data_source = None
    source_info = None

# --- MEMPROSES DAN MENAMPILKAN DATA ---
if data_source is not None:
    df = load_df(data_source)
    
    if df is not None:
        st.subheader("Sample data (first 10 rows)")
        st.dataframe(df.head(10))

        with st.expander("Dataset info & missing values"):
            st.text("Columns and dtypes:")
            st.write(df.dtypes)
            st.write("Missing values per column:")
            st.write(df.isnull().sum())

        st.subheader("Exploratory plots")
        # Plotting code
        st.markdown("### Distribusi Tingkat Produktivitas (1–10)")
        fig1 = plt.figure()
        sns.countplot(x="Productivity_Score (1-10)", data=df, palette="pastel")
        plt.title("Distribusi Tingkat Produktivitas (1–10)")
        plt.xlabel("Skor Produktivitas")
        st.pyplot(fig1)

        st.markdown("### Distribusi Durasi Tidur (Jam)")
        fig2 = plt.figure()
        sns.histplot(df["Sleep Duration (hrs)"].dropna(), bins=20, kde=True)
        st.title("Distribusi Durasi Tidur (Jam)")
        st.pyplot(fig2)

        st.markdown("### Boxplot Durasi Tidur per Tingkat Produktivitas")
        fig3 = plt.figure()
        sns.boxplot(x="Productivity_Score (1-10)", y="Sleep Duration (hrs)", data=df)
        st.title("Boxplot Durasi Tidur per Tingkat Produktivitas")
        st.pyplot(fig3)

        st.markdown("### Scatter: Durasi Tidur vs Durasi Olahraga (hue: Productivity Score)")
        fig4 = plt.figure(figsize=(8,6))
        x_jitter = df["Exercise (mins)"] + np.random.uniform(-2, 2, size=len(df))
        y_jitter = df["Sleep Duration (hrs)"] + np.random.uniform(-0.1, 0.1, size=len(df))
        sns.scatterplot(x=x_jitter, y=y_jitter, hue=df["Productivity_Score (1-10)"], palette="coolwarm")
        plt.xlabel("Durasi Olahraga (menit)")
        plt.ylabel("Durasi Tidur (jam)")
        plt.legend(title="Skor Produktivitas", bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig4)

        st.markdown("### Heatmap Korelasi Antar Variabel Numerik")
        numeric_df = df.select_dtypes(include=np.number)
        fig5 = plt.figure(figsize=(10,8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        st.title("Heatmap Korelasi Antar Variabel Numerik pada Morning Routine Dataset")
        st.pyplot(fig5)

        # --- BAGIAN TRAINING MODEL ---
        st.subheader("Feature engineering (imputasi & fitur baru)")
        if st.button("Run feature engineering"):
            
            df_final_train, df_fe_display = run_feature_engineering_and_encoding(df)
            
            st.success("Feature engineering selesai.")
            
            st.dataframe(df_fe_display.head()) # Tampilkan hasil FE dengan fitur baru
            
            st.markdown("### One-hot encoding dan seleksi fitur (contoh)")
            st.write("Shape setelah encoding & selection:", df_final_train.shape)
            st.dataframe(df_final_train.head())

            st.subheader("Train models (regresi)")
            if st.button("Train models (Linear + RF)"):
                
                # Split Data
                X = df_final_train.drop("Productivity_Score (1-10)", axis=1)
                y = df_final_train["Productivity_Score (1-10)"]
                X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
                X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

                # Linear Regression
                lin = LinearRegression()
                lin.fit(X_train, y_train)
                y_val_pred_lin = lin.predict(X_val)
                mse_lin = mean_squared_error(y_val, y_val_pred_lin)

                # Random Forest
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X_train, y_train)
                y_val_pred_rf = rf.predict(X_val)
                mse_rf = mean_squared_error(y_val, y_val_pred_rf)

                st.write("MSE (Linear Regression):", mse_lin)
                st.write("MSE (Random Forest):", mse_rf)

                # --- Menyimpan model ke MODEL_FILE_NAME (model_rf.pkl) ---
                with open(MODEL_FILE_NAME, "wb") as f:
                    pickle.dump(rf, f)
                st.success(f"Model Random Forest disimpan sebagai {MODEL_FILE_NAME}")
                # --------------------------------------------------------

            st.subheader("Clustering (KMeans)")
            if st.button("Run KMeans (n_clusters=3)"):
                # Kita harus memastikan df_final_train digunakan untuk Clustering
                X_unsup = df_final_train.select_dtypes(include=np.number).drop(columns=["Productivity_Score (1-10)"], errors='ignore')
                X_scaled_unsup = StandardScaler().fit_transform(X_unsup)
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled_unsup)
                df_fe_display["Cluster"] = clusters
                st.write(df_fe_display[["Sleep Duration (hrs)", "Exercise (mins)", "Productivity_Score (1-10)", "Cluster"]].head())
                
                figc = plt.figure(figsize=(6,5))
                sns.scatterplot(x=df_fe_display["Sleep Duration (hrs)"], y=df_fe_display["Productivity_Score (1-10)"], hue=df_fe_display["Cluster"], palette="Set2")
                plt.title("Clustering: Durasi Tidur vs Skor Produktivitas")
                st.pyplot(figc)

        st.markdown("---")
        st.info(f"Dataset saat ini dimuat dari: {source_info}. File model Anda adalah `{MODEL_FILE_NAME}`.")
        
else:
    st.warning(f"Unggah file CSV ({DEFAULT_DATA_PATH}) untuk memulai, atau pastikan file tersebut ada di root repository Anda.")
