# Heart Disease Cleveland Classification — Kelompok 2

Aplikasi web interaktif berbasis **Streamlit** untuk eksplorasi data klinis dan prediksi risiko penyakit jantung menggunakan dataset **Heart Disease Cleveland** (303 sampel, 14 atribut).

## Metodologi & Fitur Utama

- **Preprocessing Pipeline (CRISP-DM):** Menggunakan `KNNImputer` dan `StandardScaler` di dalam `sklearn.pipeline.Pipeline` untuk mencegah _data leakage_.
- **Komparasi 3 Algoritma:** Membandingkan performa **Random Forest** (model utama), **Logistic Regression**, dan **K-Nearest Neighbors (KNN)** lengkap dengan metrik Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, dan Feature Importance.
- **Eksplorasi & Simulasi Prediksi:** Filter klinis interaktif di sidebar, visualisasi distribusi pasien, ekspor CSV, serta formulir prediksi pasien beserta probabilitasnya.

## Cara Menjalankan Secara Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```
