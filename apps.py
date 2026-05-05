import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Harga Rumah", layout="wide")

# --- DATASET ---
@st.cache_data
def load_data():
    data_rumah = {
        'luas_bangunan': [45,  60,  75,  90,  100, 120, 55,  80,  110, 130, 70,  95],
        'jml_kamar':     [2,   2,   3,   3,   4,   4,   2,   3,   4,   5,   3,   4 ],
        'jarak_kota_km': [15,  10,  8,   5,   3,   2,   12,  6,   4,   1,   9,   7 ],
        'harga_juta':    [280, 370, 460, 580, 700, 880, 310, 510, 730, 1100, 420, 640]
    }
    return pd.DataFrame(data_rumah)

# --- TRAINING MODEL ---
def train_model(df):
    X = df[['luas_bangunan', 'jml_kamar', 'jarak_kota_km']]
    y = df['harga_juta']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'y_test': y_test,
        'y_pred': y_pred
    }
    return model, metrics

# --- MAIN APP ---
def main():
    st.title("🏠 Aplikasi Prediksi Harga Rumah")
    st.write("Aplikasi ini menggunakan **Multiple Linear Regression** untuk mengestimasi harga rumah berdasarkan spesifikasi tertentu.")

    df_rumah = load_data()
    model, metrics = train_model(df_rumah)

    # Sidebar untuk Input Pengguna
    st.sidebar.header("📝 Input Spesifikasi Rumah")
    luas = st.sidebar.slider("Luas Bangunan (m²)", 30, 200, 85)
    kamar = st.sidebar.number_input("Jumlah Kamar", 1, 10, 3)
    jarak = st.sidebar.slider("Jarak ke Kota (km)", 0, 30, 4)

    # Layout Utama: 2 Kolom
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🔮 Hasil Prediksi")
        input_data = [[luas, kamar, jarak]]
        prediksi = model.predict(input_data)[0]
        
        st.metric(label="Estimasi Harga Rumah", value=f"Rp {prediksi:.2f} Juta")
        
        st.write("---")
        st.subheader("📊 Evaluasi Model")
        st.write(f"**MAE:** {metrics['mae']:.2f}")
        st.write(f"**RMSE:** {metrics['rmse']:.2f}")
        st.write(f"**R² Score:** {metrics['r2']:.4f}")

    with col2:
        st.subheader("📈 Grafik Aktual vs Prediksi")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(metrics['y_test'], metrics['y_pred'], color='mediumseagreen', edgecolors='k', s=80)
        ax.plot([metrics['y_test'].min(), metrics['y_test'].max()],
                [metrics['y_test'].min(), metrics['y_test'].max()], 'r--', label='Prediksi Sempurna')
        ax.set_xlabel('Harga Aktual (Juta)')
        ax.set_ylabel('Harga Prediksi (Juta)')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    # Bagian Dataset
    if st.checkbox("Tampilkan Dataset Referensi"):
        st.dataframe(df_rumah, use_container_width=True)

if __name__ == "__main__":
    main()
