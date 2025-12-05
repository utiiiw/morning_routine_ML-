
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

st.set_option('deprecation.showPyplotGlobalUse', False)
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (8,5)

st.title("Morning Routine Productivity — Explorer & Model Demo")
st.markdown("Streamlit app generated from `morningdataset-1.pdf` content. Upload the CSV `Morning_Routine_Productivity_Dataset.csv` or use an uploaded file.")

uploaded = st.file_uploader("Upload Morning_Routine_Productivity_Dataset.csv", type=['csv'])

@st.cache_data
def load_df(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

if uploaded is not None:
    df = load_df(uploaded)
    st.subheader("Sample data (first 10 rows)")
    st.dataframe(df.head(10))

    with st.expander("Dataset info & missing values"):
        buffer = []
        df_info = df.info()
        st.text("Columns and dtypes:")
        st.write(df.dtypes)
        st.write("Missing values per column:")
        st.write(df.isnull().sum())

    st.subheader("Exploratory plots")
    st.markdown("### Distribusi Tingkat Produktivitas (1–10)")
    fig1 = plt.figure()
    sns.countplot(x="Productivity_Score (1-10)", data=df, palette="pastel")
    plt.title("Distribusi Tingkat Produktivitas (1–10)")
    plt.xlabel("Skor Produktivitas")
    plt.ylabel("Jumlah Hari")
    st.pyplot(fig1)

    st.markdown("### Distribusi Durasi Tidur (Jam)")
    fig2 = plt.figure()
    sns.histplot(df["Sleep Duration (hrs)"].dropna(), bins=20, kde=True)
    plt.title("Distribusi Durasi Tidur (Jam)")
    st.pyplot(fig2)

    st.markdown("### Boxplot Durasi Tidur per Tingkat Produktivitas")
    fig3 = plt.figure()
    sns.boxplot(x="Productivity_Score (1-10)", y="Sleep Duration (hrs)", data=df)
    plt.title("Boxplot Durasi Tidur per Tingkat Produktivitas")
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
    plt.title("Heatmap Korelasi Antar Variabel Numerik pada Morning Routine Dataset")
    st.pyplot(fig5)

    st.subheader("Feature engineering (imputasi & fitur baru)")
    if st.button("Run feature engineering"):
        df_fe = df.copy()
        # Imputasi
        df_fe["Sleep Duration (hrs)"] = df_fe["Sleep Duration (hrs)"].fillna(df_fe["Sleep Duration (hrs)"].median())
        df_fe["Meditation (mins)"] = df_fe["Meditation (mins)"].fillna(df_fe["Meditation (mins)"].median())
        df_fe["Exercise (mins)"] = df_fe["Exercise (mins)"].fillna(df_fe["Exercise (mins)"].median())
        df_fe["Breakfast Type"] = df_fe["Breakfast Type"].fillna(df_fe["Breakfast Type"].mode()[0])
        df_fe["Journaling (Y/N)"] = df_fe["Journaling (Y/N)"].fillna(df_fe["Journaling (Y/N)"].mode()[0])
        df_fe["Mood"] = df_fe["Mood"].fillna(df_fe["Mood"].mode()[0])

        # fitur baru
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
        st.success("Feature engineering selesai.")
        st.dataframe(df_fe.head())

        st.markdown("### One-hot encoding dan seleksi fitur (contoh)")
        df_encoded = pd.get_dummies(df_fe, columns=["Breakfast Type", "Journaling (Y/N)", "Mood"], drop_first=True)
        drop_cols = ["Date", "Wake-up Time", "Work Start Time", "Notes"]
        df_final = df_encoded.drop(columns=[c for c in drop_cols if c in df_encoded.columns])
        st.write("Shape setelah encoding & selection:", df_final.shape)
        st.dataframe(df_final.head())

        st.subheader("Train models (regresi)")
        if st.button("Train models (Linear + RF)"):
            X = df_final.drop("Productivity_Score (1-10)", axis=1)
            y = df_final["Productivity_Score (1-10)"]
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

            # save rf model as example
            with open("model_rf.pkl", "wb") as f:
                pickle.dump(rf, f)
            st.success("Model Random Forest disimpan sebagai model_rf.pkl")

        st.subheader("Clustering (KMeans)")
        if st.button("Run KMeans (n_clusters=3)"):
            X_unsup = df_fe.select_dtypes(include=np.number).drop(columns=["Productivity_Score (1-10)"], errors='ignore')
            X_scaled_unsup = StandardScaler().fit_transform(X_unsup)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled_unsup)
            df_fe["Cluster"] = clusters
            st.write(df_fe[["Sleep Duration (hrs)", "Exercise (mins)", "Productivity_Score (1-10)", "Cluster"]].head())
            figc = plt.figure(figsize=(6,5))
            sns.scatterplot(x=df_fe["Sleep Duration (hrs)"], y=df_fe["Productivity_Score (1-10)"], hue=df_fe["Cluster"], palette="Set2")
            plt.title("Clustering: Durasi Tidur vs Skor Produktivitas")
            st.pyplot(figc)

    st.markdown("---")
    st.info("Selesai. Kamu bisa mengunduh `model_rf.pkl` jika dibuat, atau mendownload file ini dari direktori kerja `/content` ketika menjalankan locally.")

else:
    st.warning("Unggah file CSV untuk memulai. Dataset yang diperlukan: `Morning_Routine_Productivity_Dataset.csv`.")

