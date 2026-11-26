import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

# ==========================================================
# KONFIGURASI HALAMAN UTAMA
# ==========================================================
st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# CUSTOM CSS FULL PINK & PUTIH
# ==========================================================
st.markdown(
    """
    <style>
    /* Latar Utama Putih Kemerahan Lembut */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #FFF8F7 !important;
        color: #4A232D !important;
    }

    /* Sidebar Pink Pastel Lembut */
    section[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #FDECEF !important;
        border-right: 1px solid #F8D7DA;
    }

    /* Pastikan Teks Umum Selalu Gelap Kontras */
    h1, h2, h3, h4, p, li, label,
    [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] div {
        color: #4A232D !important;
    }

    /* Hero Banner Pink Gradasi Lembut */
    .hero-banner {
        background: linear-gradient(135deg, #E88C9B 0%, #D9778A 100%);
        padding: 26px 30px;
        border-radius: 16px;
        margin-bottom: 22px;
        box-shadow: 0 6px 20px rgba(217, 119, 138, 0.20);
    }
    .hero-banner h1 {
        color: #FFFFFF !important;
        margin: 0 0 6px 0;
        font-size: 2rem;
        font-weight: 800;
    }
    .hero-banner .hero-sub {
        color: #FFF8F7 !important;
        font-size: 0.95rem;
        font-weight: 500;
        margin-bottom: 10px;
        opacity: 0.95;
    }
    .hero-banner p {
        color: #FFFFFF !important;
        margin: 0;
        font-size: 0.95rem;
        line-height: 1.55;
    }

    /* Header Halaman Sekunder */
    .page-header {
        background-color: #FFFFFF;
        border: 1px solid #F8D7DA;
        border-left: 6px solid #E88C9B;
        padding: 18px 22px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 3px 10px rgba(217, 119, 138, 0.07);
    }
    .page-header h2 {
        color: #5C2B35 !important;
        margin: 0 0 4px 0;
        font-size: 1.6rem;
    }
    .page-header p {
        color: #7A4B56 !important;
        margin: 0;
        font-size: 0.92rem;
    }

    /* Kartu KPI Putih dengan Aksen Pink */
    .kpi-card {
        background-color: #FFFFFF;
        border: 1.5px solid #F8D7DA;
        border-top: 5px solid #E88C9B;
        border-radius: 14px;
        padding: 18px 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(217, 119, 138, 0.08);
        text-align: center;
    }
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #7A4B56 !important;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #D9778A !important;
    }
    .kpi-value-sm {
        font-size: 1.35rem;
        font-weight: 800;
        color: #D9778A !important;
    }

    /* Kartu Alur Pengerjaan */
    .step-card {
        background-color: #FFFFFF;
        border: 1.5px solid #F8D7DA;
        border-radius: 14px;
        padding: 16px 18px;
        height: 100%;
        margin-bottom: 10px;
        box-shadow: 0 3px 10px rgba(217, 119, 138, 0.06);
    }
    .step-badge {
        display: inline-block;
        background-color: #E88C9B;
        color: #FFFFFF !important;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 10px;
    }

    /* Kotak Konten Umum */
    .content-box {
        background-color: #FFFFFF;
        border: 1.5px solid #F8D7DA;
        border-radius: 14px;
        padding: 20px 22px;
        margin-bottom: 14px;
        box-shadow: 0 3px 10px rgba(217, 119, 138, 0.06);
    }

    /* Sidebar Brand & Footer */
    .sidebar-brand {
        background: linear-gradient(135deg, #E88C9B, #D9778A);
        padding: 14px 16px;
        border-radius: 12px;
        margin-bottom: 12px;
    }
    .sidebar-brand h3 {
        color: #FFFFFF !important;
        margin: 0;
        font-size: 1.1rem;
    }
    .sidebar-brand p {
        color: #FFF8F7 !important;
        margin: 4px 0 0 0;
        font-size: 0.78rem;
    }
    .sidebar-footer {
        background-color: #FFFFFF;
        border: 1.5px solid #F8D7DA;
        border-top: 4px solid #E88C9B;
        border-radius: 12px;
        padding: 14px 14px;
        font-size: 0.80rem;
        line-height: 1.5;
        color: #4A232D !important;
        box-shadow: 0 2px 8px rgba(217, 119, 138, 0.07);
    }
    .sidebar-footer .team-title {
        font-weight: 700;
        color: #5C2B35 !important;
        font-size: 0.85rem;
    }
    .sidebar-footer .team-sub {
        color: #7A4B56 !important;
        font-size: 0.78rem;
        margin-bottom: 8px;
    }
    .sidebar-footer .section-label {
        font-weight: 700;
        color: #D9778A !important;
        margin-top: 8px;
        margin-bottom: 3px;
        font-size: 0.79rem;
    }

    /* Tombol Pink */
    div.stButton > button, div.stFormSubmitButton > button, div.stDownloadButton > button {
        background-color: #E88C9B !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.55rem 1.2rem !important;
        font-weight: 600 !important;
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover, div.stDownloadButton > button:hover {
        background-color: #D9778A !important;
        color: #FFFFFF !important;
    }
    div.stButton > button p, div.stFormSubmitButton > button p, div.stDownloadButton > button p {
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# KAMUS ATRIBUT DATASET HEART DISEASE CLEVELAND
# ==========================================================
FEATURE_INFO = {
    "age": "Usia pasien (tahun)",
    "sex": "Jenis kelamin (1 = Laki-laki, 0 = Perempuan)",
    "cp": "Tipe nyeri dada (0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal Pain, 3 = Asymptomatic)",
    "trestbps": "Tekanan darah saat istirahat (mm Hg)",
    "chol": "Kadar kolesterol serum (mg/dl)",
    "fbs": "Gula darah puasa > 120 mg/dl (1 = Ya, 0 = Tidak)",
    "restecg": "Hasil EKG istirahat (0 = Normal, 1 = Kelainan ST-T, 2 = Hipertrofi ventrikel kiri)",
    "thalach": "Detak jantung maksimum yang dicapai (bpm)",
    "exang": "Angina akibat aktivitas olahraga (1 = Ya, 0 = Tidak)",
    "oldpeak": "Depresi segmen ST akibat olahraga relatif terhadap istirahat",
    "slope": "Kemiringan segmen ST puncak olahraga (0 = Upsloping, 1 = Flat, 2 = Downsloping)",
    "ca": "Jumlah pembuluh darah utama (0–3) yang terdeteksi fluoroskopi",
    "thal": "Status thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)",
    "target": "Diagnosis penyakit jantung (1 = Terindikasi Penyakit Jantung, 0 = Sehat)",
}


# ==========================================================
# PEMUATAN DATA & PIPELINE MODELING (BAB 1 - BAB 3)
# ==========================================================
@st.cache_data
def load_data(filepath: str = "heart.csv") -> pd.DataFrame:
    df = pd.read_csv(filepath, na_values=["?", "NA", "null", ""])
    df.columns = [c.strip().lower() for c in df.columns]
    if "target" not in df.columns and "num" in df.columns:
        df["target"] = (df["num"] > 0).astype(int)
        df.drop(columns=["num"], inplace=True)
    return df


@st.cache_resource
def run_ml_pipeline(df: pd.DataFrame):
    df_clean = df.drop_duplicates().copy()
    X = df_clean.drop(columns=["target"])
    y = df_clean["target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    prep_only = Pipeline(
        steps=[
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler()),
        ]
    )
    X_train_scaled = prep_only.fit_transform(X_train)

    pca_full = PCA(random_state=42)
    X_train_pca_all = pca_full.fit_transform(X_train_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_90 = int(np.argmax(cum_var >= 0.90) + 1)

    classifiers = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=6, random_state=42
        ),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(
            kernel="rbf", probability=True, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    }

    pipelines = {}
    pca_pipelines = {}
    metrics_rows = []
    pca_metrics_rows = []
    predictions = {}
    probabilities = {}

    for name, clf in classifiers.items():
        pipe = Pipeline(
            steps=[
                ("imputer", KNNImputer(n_neighbors=5)),
                ("scaler", StandardScaler()),
                ("model", clf),
            ]
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        cv_score = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy").mean()

        pipelines[name] = pipe
        predictions[name] = y_pred
        probabilities[name] = y_prob

        metrics_rows.append(
            {
                "Algoritma": name,
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
                "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
                "CV 5-Fold": round(cv_score, 4),
            }
        )

        clf_pca = classifiers[name].__class__(**classifiers[name].get_params())
        pipe_pca = Pipeline(
            steps=[
                ("imputer", KNNImputer(n_neighbors=5)),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components_90, random_state=42)),
                ("model", clf_pca),
            ]
        )
        pipe_pca.fit(X_train, y_train)
        y_pred_pca = pipe_pca.predict(X_test)
        y_prob_pca = pipe_pca.predict_proba(X_test)[:, 1]
        pca_pipelines[name] = pipe_pca

        pca_metrics_rows.append(
            {
                "Algoritma": name,
                "Accuracy (PCA)": round(accuracy_score(y_test, y_pred_pca), 4),
                "F1-Score (PCA)": round(f1_score(y_test, y_pred_pca, zero_division=0), 4),
                "ROC-AUC (PCA)": round(roc_auc_score(y_test, y_prob_pca), 4),
            }
        )

    metrics_df = (
        pd.DataFrame(metrics_rows)
        .sort_values(by=["F1-Score", "ROC-AUC"], ascending=False)
        .reset_index(drop=True)
    )

    pca_metrics_df = (
        pd.DataFrame(pca_metrics_rows)
        .sort_values(by=["F1-Score (PCA)", "ROC-AUC (PCA)"], ascending=False)
        .reset_index(drop=True)
    )

    return {
        "df_clean": df_clean,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "pca_full": pca_full,
        "X_train_pca_all": X_train_pca_all,
        "cum_var": cum_var,
        "n_components_90": n_components_90,
        "pipelines": pipelines,
        "pca_pipelines": pca_pipelines,
        "metrics_df": metrics_df,
        "pca_metrics_df": pca_metrics_df,
        "predictions": predictions,
        "probabilities": probabilities,
    }


try:
    df_raw = load_data("heart.csv")
except FileNotFoundError:
    st.error("File `heart.csv` tidak ditemukan. Pastikan `heart.csv` satu folder dengan `app.py`.")
    st.stop()

res = run_ml_pipeline(df_raw)


# ==========================================================
# HALAMAN 1: BERANDA
# ==========================================================
def page_beranda():
    st.markdown(
        """
        <div class="hero-banner">
            <h1>🫀 Prediksi Penyakit Jantung</h1>
            <div class="hero-sub">Implementasi Machine Learning Berbasis Web dengan Streamlit</div>
            <p>
                Website ini memaparkan seluruh tahapan proyek <b>prediksi risiko penyakit jantung</b> pada dataset 
                <i>Heart Disease Cleveland</i>: mulai dari pengenalan data, eksplorasi, preprocessing, reduksi dimensi (PCA), 
                perbandingan lima algoritma klasifikasi, hingga prediksi langsung (satuan maupun massal CSV) yang bisa dicoba pengguna.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_pasien = len(df_raw)
    risk_rate = (df_raw["target"] == 1).mean() * 100
    best_row = res["metrics_df"].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Jumlah pasien</div>'
            f'<div class="kpi-value">{total_pasien:,}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Rasio positif penyakit</div>'
            f'<div class="kpi-value">{risk_rate:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Model terbaik (F1)</div>'
            f'<div class="kpi-value-sm">{best_row["Algoritma"]}</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">ROC-AUC model terbaik</div>'
            f'<div class="kpi-value">{best_row["ROC-AUC"]:.3f}</div></div>',
            unsafe_allow_html=True,
        )

    st.subheader("Alur Pengerjaan")

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(
            f"""
            <div class="step-card">
                <span class="step-badge">1. Dataset</span>
                <p style="font-size:0.88rem; margin:0;">
                Karakteristik data klinis <b>{total_pasien} pasien</b> dengan 13 variabel diagnostik dan 1 label target biner.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            """
            <div class="step-card">
                <span class="step-badge">2. EDA & Preprocessing</span>
                <p style="font-size:0.88rem; margin:0;">
                Menangani data hilang (<i>KNNImputer</i>), menghapus duplikat, dan standarisasi fitur (<i>StandardScaler</i>).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            f"""
            <div class="step-card">
                <span class="step-badge">3. Reduksi PCA</span>
                <p style="font-size:0.88rem; margin:0;">
                13 fitur dipadatkan menjadi <b>{res['n_components_90']} komponen utama</b> dengan mempertahankan ≥ 90% varians data.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s4:
        st.markdown(
            """
            <div class="step-card">
                <span class="step-badge">4. Modelling</span>
                <p style="font-size:0.88rem; margin:0;">
                Lima algoritma klasifikasi dievaluasi dengan metrik <i>Accuracy, F1-Score, ROC-AUC,</i> dan validasi silang.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Informasi Kelompok & Dosen Pengampu")

    col_team, col_lect = st.columns([1.4, 1])
    with col_team:
        st.markdown(
            """
            <div class="content-box" style="border-left: 5px solid #E88C9B;">
                <span class="step-badge">Anggota Kelompok 2</span>
                <ol style="margin: 4px 0 0 0; padding-left: 20px; line-height: 1.8;">
                    <li><b>Haifa Auliatul Pynkan</b> — NIM : <code>241011401848</code></li>
                    <li><b>Rahma Dika Yhustira</b> — NIM : <code>241011402308</code></li>
                    <li><b>Syahwa Rahma Andini</b> — NIM : <code>241011402529</code></li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_lect:
        st.markdown(
            """
            <div class="content-box" style="border-left: 5px solid #D9778A;">
                <span class="step-badge">Dosen Pengampu</span>
                <p style="margin: 6px 0 4px 0; font-size: 1.02rem;">
                    <b>Farizi Ilham S.Kom., M.Kom</b>
                </p>
                <p style="margin: 0; font-size: 0.88rem; color: #7A4B56 !important;">
                    Program Studi Teknik Informatika<br>Universitas Pamulang
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==========================================================
# HALAMAN 2: TENTANG DATASET
# ==========================================================
def page_tentang_dataset():
    st.markdown(
        """
        <div class="page-header">
            <h2>🗄️ Tentang Dataset</h2>
            <p>Deskripsi sumber data, struktur atribut klinis, dan statistik ringkasan Heart Disease Cleveland</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Total Baris</div>'
            f'<div class="kpi-value">{df_raw.shape[0]}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Jumlah Fitur</div>'
            f'<div class="kpi-value">{df_raw.shape[1] - 1}</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Nilai Hilang (NaN)</div>'
            f'<div class="kpi-value">{int(df_raw.isna().sum().sum())}</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Data Duplikat</div>'
            f'<div class="kpi-value">{int(df_raw.duplicated().sum())}</div></div>',
            unsafe_allow_html=True,
        )

    st.subheader("Pratinjau Data Mentah")
    st.dataframe(df_raw.head(15), use_container_width=True)

    col_left, col_right = st.columns([1.15, 1])
    with col_left:
        st.subheader("Kamus Atribut Klinis")
        kamus_df = pd.DataFrame(
            [
                {
                    "Atribut": col,
                    "Tipe Data": str(df_raw[col].dtype),
                    "Keterangan": FEATURE_INFO.get(col, "-"),
                }
                for col in df_raw.columns
            ]
        )
        st.dataframe(kamus_df, use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("Statistik Deskriptif")
        st.dataframe(df_raw.describe().T.round(2), use_container_width=True)


# ==========================================================
# HALAMAN 3: EKSPLORASI DATA (EDA)
# ==========================================================
def page_eda():
    st.markdown(
        """
        <div class="page-header">
            <h2>📊 Eksplorasi Data (EDA)</h2>
            <p>Visualisasi distribusi kelas target, sebaran variabel klinis, dan korelasi antar fitur</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(
        ["Distribusi Target & Numerik", "Perbandingan Fitur Kategorikal", "Heatmap Korelasi"]
    )

    with tab1:
        col_t, col_n = st.columns([1, 1.3])
        with col_t:
            st.markdown("**Proporsi Kelas Diagnosis**")
            fig_t, ax_t = plt.subplots(figsize=(4.5, 3.8))
            fig_t.patch.set_facecolor("white")
            counts = df_raw["target"].value_counts().sort_index()
            bars = ax_t.bar(
                ["Sehat (0)", "Penyakit Jantung (1)"],
                counts.values,
                color=["#F8D7DA", "#E88C9B"],
                edgecolor="#D9778A",
                width=0.5,
            )
            ax_t.set_ylabel("Jumlah Pasien", color="#4A232D")
            ax_t.tick_params(colors="#4A232D")
            ax_t.spines[["top", "right"]].set_visible(False)
            for b in bars:
                ax_t.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + 2,
                    f"{int(b.get_height())}",
                    ha="center",
                    fontweight="bold",
                    color="#5C2B35",
                )
            st.pyplot(fig_t)

        with col_n:
            num_choice = st.selectbox(
                "Pilih Fitur Numerik:",
                ["age", "trestbps", "chol", "thalach", "oldpeak"],
            )
            fig_n, ax_n = plt.subplots(figsize=(5.5, 3.6))
            fig_n.patch.set_facecolor("white")
            sns.histplot(
                data=df_raw,
                x=num_choice,
                hue="target",
                kde=True,
                palette=["#F4A6B8", "#C94A6A"],
                ax=ax_n,
            )
            ax_n.set_title(f"Distribusi '{num_choice}' Berdasarkan Diagnosis", color="#4A232D")
            ax_n.tick_params(colors="#4A232D")
            ax_n.spines[["top", "right"]].set_visible(False)
            ax_n.legend(title="Target", labels=["Penyakit Jantung (1)", "Sehat (0)"])
            st.pyplot(fig_n)

    with tab2:
        cat_choice = st.selectbox(
            "Pilih Fitur Kategorikal:",
            ["cp", "sex", "exang", "slope", "ca", "thal"],
        )
        fig_c, ax_c = plt.subplots(figsize=(8, 3.8))
        fig_c.patch.set_facecolor("white")
        sns.countplot(
            data=df_raw,
            x=cat_choice,
            hue="target",
            palette=["#F8D7DA", "#E88C9B"],
            edgecolor="#D9778A",
            ax=ax_c,
        )
        ax_c.set_title(f"Distribusi '{cat_choice}' Terhadap Diagnosis", color="#4A232D")
        ax_c.set_ylabel("Jumlah Pasien", color="#4A232D")
        ax_c.tick_params(colors="#4A232D")
        ax_c.spines[["top", "right"]].set_visible(False)
        ax_c.legend(title="Diagnosis", labels=["Sehat (0)", "Penyakit Jantung (1)"])
        st.pyplot(fig_c)
        st.caption(f"Deskripsi atribut: {FEATURE_INFO.get(cat_choice, '')}")

    with tab3:
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        fig_corr.patch.set_facecolor("white")
        sns.heatmap(
            df_raw.corr(numeric_only=True),
            annot=True,
            fmt=".2f",
            cmap="RdPu",
            ax=ax_corr,
            annot_kws={"size": 7.5},
        )
        ax_corr.set_title("Matriks Korelasi Pearson", color="#4A232D")
        st.pyplot(fig_corr)


# ==========================================================
# HALAMAN 4: PREPROCESSING
# ==========================================================
def page_preprocessing():
    st.markdown(
        """
        <div class="page-header">
            <h2>⚙️ Preprocessing</h2>
            <p>Pembersihan data, imputasi nilai hilang, pembagian data, dan standarisasi fitur tanpa data leakage</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Data Bersih Unik</div>'
            f'<div class="kpi-value">{len(res["df_clean"])}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Data Latih (80%)</div>'
            f'<div class="kpi-value">{len(res["X_train"])}</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Data Uji (20%)</div>'
            f'<div class="kpi-value">{len(res["X_test"])}</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-label">Metode Scaling</div>'
            '<div class="kpi-value-sm">StandardScaler</div></div>',
            unsafe_allow_html=True,
        )

    st.subheader("Tahapan Pipeline Preprocessing")
    audit_df = pd.DataFrame(
        {
            "Tahap": [
                "1. Deduplikasi",
                "2. Data Splitting",
                "3. Imputasi Missing Value",
                "4. Feature Scaling",
            ],
            "Metode": [
                "drop_duplicates()",
                "train_test_split (80:20, stratified)",
                "KNNImputer(n_neighbors=5)",
                "StandardScaler (mean=0, std=1)",
            ],
            "Tujuan": [
                "Mencegah baris identik masuk ke data latih dan data uji sekaligus.",
                "Menjaga proporsi kelas sehat dan sakit seimbang pada data latih & uji.",
                "Mengisi nilai kosong berdasarkan kemiripan 5 tetangga klinis terdekat.",
                "Menyamakan skala antar fitur agar optimal pada KNN, SVM, dan PCA.",
            ],
        }
    )
    st.dataframe(audit_df, use_container_width=True, hide_index=True)

    st.subheader("Perbandingan Data Sebelum & Sesudah Preprocessing (X_train)")
    t_before, t_after = st.tabs(["Sebelum Preprocessing", "Sesudah Imputasi & StandardScaler"])
    with t_before:
        st.dataframe(res["X_train"].head(8), use_container_width=True)
    with t_after:
        scaled_df = pd.DataFrame(
            res["X_train_scaled"],
            columns=res["X_train"].columns,
            index=res["X_train"].index,
        )
        st.dataframe(scaled_df.head(8).round(4), use_container_width=True)


# ==========================================================
# HALAMAN 5: REDUKSI DIMENSI (PCA)
# ==========================================================
def page_pca():
    st.markdown(
        """
        <div class="page-header">
            <h2>🧬 Reduksi Dimensi (PCA)</h2>
            <p>Analisis Principal Component Analysis untuk meringkas dimensi fitur tanpa kehilangan informasi utama</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pca_full = res["pca_full"]
    cum_var = res["cum_var"]
    n_opt = res["n_components_90"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-label">Dimensi Fitur Awal</div>'
            '<div class="kpi-value">13 Fitur</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Komponen Optimal (≥ 90% Varians)</div>'
            f'<div class="kpi-value">{n_opt} Komponen</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Varians Terjelaskan (PC1 + PC2)</div>'
            f'<div class="kpi-value">{cum_var[1] * 100:.2f}%</div></div>',
            unsafe_allow_html=True,
        )

    col_var, col_scatter = st.columns(2)
    with col_var:
        st.subheader("Cumulative Explained Variance")
        fig_v, ax_v = plt.subplots(figsize=(5.5, 4))
        fig_v.patch.set_facecolor("white")
        pcs = np.arange(1, len(cum_var) + 1)
        ax_v.bar(
            pcs,
            pca_full.explained_variance_ratio_ * 100,
            color="#F8D7DA",
            edgecolor="#E88C9B",
            label="Varians Individu (%)",
        )
        ax_v.plot(
            pcs,
            cum_var * 100,
            marker="o",
            color="#D9778A",
            linewidth=2.2,
            label="Varians Kumulatif (%)",
        )
        ax_v.axhline(90, color="#9D174D", linestyle="--", linewidth=1.2, label="Batas 90%")
        ax_v.set_xlabel("Principal Component (PC)", color="#4A232D")
        ax_v.set_ylabel("Persentase Varians (%)", color="#4A232D")
        ax_v.set_xticks(pcs)
        ax_v.tick_params(colors="#4A232D")
        ax_v.spines[["top", "right"]].set_visible(False)
        ax_v.legend(fontsize=8)
        st.pyplot(fig_v)

    with col_scatter:
        st.subheader("Proyeksi 2D (PC1 vs PC2)")
        fig_s, ax_s = plt.subplots(figsize=(5.5, 4))
        fig_s.patch.set_facecolor("white")
        pca_2d = res["X_train_pca_all"][:, :2]
        y_tr = res["y_train"].values
        ax_s.scatter(
            pca_2d[y_tr == 0, 0],
            pca_2d[y_tr == 0, 1],
            c="#F4A6B8",
            alpha=0.85,
            label="Sehat (0)",
            edgecolors="#D9778A",
            s=45,
        )
        ax_s.scatter(
            pca_2d[y_tr == 1, 0],
            pca_2d[y_tr == 1, 1],
            c="#BE185D",
            alpha=0.85,
            label="Penyakit Jantung (1)",
            edgecolors="w",
            s=45,
        )
        ax_s.set_xlabel(f"PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}%)", color="#4A232D")
        ax_s.set_ylabel(f"PC2 ({pca_full.explained_variance_ratio_[1]*100:.1f}%)", color="#4A232D")
        ax_s.tick_params(colors="#4A232D")
        ax_s.spines[["top", "right"]].set_visible(False)
        ax_s.legend()
        st.pyplot(fig_s)

    st.subheader("Perbandingan Performa Model: Tanpa PCA vs Dengan PCA")
    comp_pca = pd.merge(
        res["metrics_df"][["Algoritma", "Accuracy", "F1-Score", "ROC-AUC"]],
        res["pca_metrics_df"],
        on="Algoritma",
    )
    st.dataframe(comp_pca, use_container_width=True, hide_index=True)


# ==========================================================
# HALAMAN 6: MODELLING & EVALUASI
# ==========================================================
def page_modelling():
    st.markdown(
        """
        <div class="page-header">
            <h2>🧠 Modelling & Evaluasi</h2>
            <p>Perbandingan lima algoritma klasifikasi menggunakan metrik evaluasi lengkap dan visualisasi performa</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("1. Tabel Perbandingan Lima Algoritma")
    st.dataframe(
        res["metrics_df"],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Accuracy": st.column_config.ProgressColumn(
                "Accuracy", min_value=0.0, max_value=1.0, format="%.4f"
            ),
            "F1-Score": st.column_config.ProgressColumn(
                "F1-Score", min_value=0.0, max_value=1.0, format="%.4f"
            ),
            "ROC-AUC": st.column_config.ProgressColumn(
                "ROC-AUC", min_value=0.0, max_value=1.0, format="%.4f"
            ),
            "CV 5-Fold": st.column_config.ProgressColumn(
                "CV 5-Fold", min_value=0.0, max_value=1.0, format="%.4f"
            ),
        },
    )

    selected_algo = st.selectbox(
        "Pilih Algoritma untuk Analisis Detail:",
        list(res["pipelines"].keys()),
        index=0,
    )

    y_test = res["y_test"]
    y_pred = res["predictions"][selected_algo]

    col_cm, col_roc = st.columns(2)
    with col_cm:
        st.subheader(f"2. Confusion Matrix ({selected_algo})")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        fig_cm.patch.set_facecolor("white")
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="RdPu",
            xticklabels=["Prediksi Sehat (0)", "Prediksi Sakit (1)"],
            yticklabels=["Aktual Sehat (0)", "Aktual Sakit (1)"],
            ax=ax_cm,
        )
        ax_cm.set_xlabel("Prediksi", color="#4A232D")
        ax_cm.set_ylabel("Aktual", color="#4A232D")
        st.pyplot(fig_cm)

    with col_roc:
        st.subheader("3. Kurva ROC Semua Model")
        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
        fig_roc.patch.set_facecolor("white")
        pink_shades = ["#9D174D", "#DB2777", "#E88C9B", "#F472B6", "#FBCFE8"]
        for (m_name, y_prob), col_c in zip(res["probabilities"].items(), pink_shades):
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_val = roc_auc_score(y_test, y_prob)
            ax_roc.plot(fpr, tpr, lw=2.1, color=col_c, label=f"{m_name} ({auc_val:.3f})")
        ax_roc.plot([0, 1], [0, 1], "--", color="#7A4B56", lw=1)
        ax_roc.set_xlabel("False Positive Rate", color="#4A232D")
        ax_roc.set_ylabel("True Positive Rate", color="#4A232D")
        ax_roc.tick_params(colors="#4A232D")
        ax_roc.spines[["top", "right"]].set_visible(False)
        ax_roc.legend(loc="lower right", fontsize=8)
        st.pyplot(fig_roc)

    col_rep, col_fi = st.columns([1, 1.15])
    with col_rep:
        st.subheader(f"4. Classification Report ({selected_algo})")
        rep = pd.DataFrame(
            classification_report(
                y_test,
                y_pred,
                target_names=["Sehat (0)", "Penyakit Jantung (1)"],
                output_dict=True,
            )
        ).T.round(4)
        st.dataframe(rep, use_container_width=True)

    with col_fi:
        st.subheader("5. Kontribusi Fitur Klinis (Random Forest)")
        rf_clf = res["pipelines"]["Random Forest"].named_steps["model"]
        feat_imp = pd.Series(
            rf_clf.feature_importances_, index=res["X_train"].columns
        ).sort_values(ascending=True)
        fig_fi, ax_fi = plt.subplots(figsize=(5.5, 4))
        fig_fi.patch.set_facecolor("white")
        bar_colors = plt.cm.RdPu(np.linspace(0.35, 0.85, len(feat_imp)))
        feat_imp.plot(kind="barh", color=bar_colors, edgecolor="#D9778A", ax=ax_fi)
        ax_fi.set_xlabel("Skor Kepentingan Fitur", color="#4A232D")
        ax_fi.tick_params(colors="#4A232D")
        ax_fi.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig_fi)


# ==========================================================
# HALAMAN 7: PREDIKSI SATU PASIEN
# ==========================================================
def page_prediksi_satu():
    st.markdown(
        """
        <div class="page-header">
            <h2>🩺 Prediksi Satu Pasien</h2>
            <p>Simulasi prediksi risiko penyakit jantung secara langsung berdasarkan input parameter klinis</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    best_default = res["metrics_df"].iloc[0]["Algoritma"]
    algo_list = list(res["pipelines"].keys())
    default_idx = algo_list.index(best_default) if best_default in algo_list else 0

    model_name = st.selectbox("Pilih Model Prediksi:", algo_list, index=default_idx)

    with st.form("form_satu_pasien"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Usia (age)", 18, 100, 54)
            sex = st.selectbox(
                "Jenis Kelamin (sex)",
                [1, 0],
                format_func=lambda x: "1 - Laki-laki" if x == 1 else "0 - Perempuan",
            )
            cp = st.selectbox(
                "Tipe Nyeri Dada (cp)",
                [0, 1, 2, 3],
                format_func=lambda x: {
                    0: "0 - Typical Angina",
                    1: "1 - Atypical Angina",
                    2: "2 - Non-anginal Pain",
                    3: "3 - Asymptomatic",
                }[x],
            )
            trestbps = st.number_input("Tekanan Darah Istirahat (trestbps)", 80, 220, 130)
            chol = st.number_input("Kolesterol Serum (chol)", 100, 600, 246)

        with c2:
            fbs = st.selectbox(
                "Gula Darah Puasa > 120 mg/dl (fbs)",
                [0, 1],
                format_func=lambda x: "1 - Ya" if x == 1 else "0 - Tidak",
            )
            restecg = st.selectbox(
                "Hasil EKG Istirahat (restecg)",
                [0, 1, 2],
                format_func=lambda x: {
                    0: "0 - Normal",
                    1: "1 - Kelainan Gelombang ST-T",
                    2: "2 - Hipertrofi Ventrikel Kiri",
                }[x],
            )
            thalach = st.number_input("Detak Jantung Maksimum (thalach)", 60, 230, 150)
            exang = st.selectbox(
                "Angina Akibat Olahraga (exang)",
                [0, 1],
                format_func=lambda x: "1 - Ya" if x == 1 else "0 - Tidak",
            )

        with c3:
            oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
            slope = st.selectbox(
                "Kemiringan Segmen ST (slope)",
                [0, 1, 2],
                format_func=lambda x: {
                    0: "0 - Upsloping",
                    1: "1 - Flat",
                    2: "2 - Downsloping",
                }[x],
            )
            ca = st.selectbox("Jumlah Pembuluh Utama (ca)", [0, 1, 2, 3])
            thal = st.selectbox(
                "Thalassemia (thal)",
                [1, 2, 3],
                index=1,
                format_func=lambda x: {
                    1: "1 - Normal",
                    2: "2 - Fixed Defect",
                    3: "3 - Reversible Defect",
                }[x],
            )

        submitted = st.form_submit_button("🔍 Prediksi Risiko Pasien")

    if submitted:
        input_df = pd.DataFrame(
            [
                {
                    "age": age,
                    "sex": sex,
                    "cp": cp,
                    "trestbps": trestbps,
                    "chol": chol,
                    "fbs": fbs,
                    "restecg": restecg,
                    "thalach": thalach,
                    "exang": exang,
                    "oldpeak": oldpeak,
                    "slope": slope,
                    "ca": ca,
                    "thal": thal,
                }
            ]
        )[res["X_train"].columns]

        pipe = res["pipelines"][model_name]
        pred = int(pipe.predict(input_df)[0])
        proba = pipe.predict_proba(input_df)[0]

        st.markdown("---")
        r1, r2 = st.columns([1.4, 1])
        with r1:
            if pred == 1:
                st.markdown(
                    f"""
                    <div class="content-box" style="border-left: 6px solid #BE185D; background-color: #FDECEF;">
                        <h4 style="margin:0 0 6px 0; color:#9D174D !important;">⚠️ Hasil Diagnosis ({model_name}): TERINDIKASI PENYAKIT JANTUNG (1)</h4>
                        <p style="margin:0;">Probabilitas Risiko Positif: <b>{proba[1]*100:.2f}%</b> | Probabilitas Sehat: <b>{proba[0]*100:.2f}%</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="content-box" style="border-left: 6px solid #E88C9B; background-color: #FFFFFF;">
                        <h4 style="margin:0 0 6px 0; color:#5C2B35 !important;">✅ Hasil Diagnosis ({model_name}): TIDAK TERINDIKASI PENYAKIT JANTUNG (0)</h4>
                        <p style="margin:0;">Probabilitas Sehat: <b>{proba[0]*100:.2f}%</b> | Probabilitas Risiko: <b>{proba[1]*100:.2f}%</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with r2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Tingkat Probabilitas Risiko</div>
                    <div class="kpi-value">{proba[1]*100:.1f}%</div>
                    <div style="font-size:0.8rem; color:#7A4B56;">Sehat: {proba[0]*100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==========================================================
# HALAMAN 8: PREDIKSI MASSAL (CSV)
# ==========================================================
def page_prediksi_massal():
    st.markdown(
        """
        <div class="page-header">
            <h2>📄 Prediksi Massal (CSV)</h2>
            <p>Unggah file CSV data pasien untuk melakukan prediksi sekaligus dan mengunduh hasilnya</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    required_cols = list(res["X_train"].columns)

    with st.expander("📥 Unduh Template CSV Contoh (Siap Uji Coba)", expanded=True):
        st.write("Gunakan file contoh berikut jika belum menyiapkan file CSV data pasien baru:")
        sample_csv = res["X_test"].head(10).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Unduh Template Data Pasien (10 Baris).csv",
            data=sample_csv,
            file_name="template_pasien_heart.csv",
            mime="text/csv",
        )

    model_batch = st.selectbox(
        "Pilih Model untuk Prediksi Massal:",
        list(res["pipelines"].keys()),
        index=0,
    )

    uploaded_file = st.file_uploader("Unggah file CSV pasien:", type=["csv"])

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file, na_values=["?", "NA", "null", ""])
        df_upload.columns = [c.strip().lower() for c in df_upload.columns]

        missing_cols = [c for c in required_cols if c not in df_upload.columns]
        if missing_cols:
            st.error(f"Kolom wajib berikut tidak ditemukan pada CSV: {', '.join(missing_cols)}")
        else:
            X_batch = df_upload[required_cols]
            pipe = res["pipelines"][model_batch]
            preds = pipe.predict(X_batch)
            probs = pipe.predict_proba(X_batch)[:, 1]

            df_result = df_upload.copy()
            df_result["prediksi_target"] = preds
            df_result["keterangan_diagnosis"] = np.where(
                preds == 1, "Terindikasi Penyakit Jantung", "Sehat"
            )
            df_result["probabilitas_risiko (%)"] = (probs * 100).round(2)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-label">Total Data Diprediksi</div>'
                    f'<div class="kpi-value">{len(df_result)} pasien</div></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-label">Terindikasi Positif</div>'
                    f'<div class="kpi-value">{int((preds == 1).sum())} pasien</div></div>',
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-label">Tidak Terindikasi</div>'
                    f'<div class="kpi-value">{int((preds == 0).sum())} pasien</div></div>',
                    unsafe_allow_html=True,
                )

            st.subheader("Hasil Prediksi Massal")
            st.dataframe(
                df_result,
                use_container_width=True,
                column_config={
                    "probabilitas_risiko (%)": st.column_config.ProgressColumn(
                        "Probabilitas Risiko (%)",
                        min_value=0.0,
                        max_value=100.0,
                        format="%.2f%%",
                    )
                },
            )

            out_csv = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Unduh Hasil Prediksi (CSV)",
                data=out_csv,
                file_name="hasil_prediksi_heart_disease.csv",
                mime="text/csv",
            )


# ==========================================================
# HALAMAN 9: KESIMPULAN & SARAN
# ==========================================================
def page_kesimpulan():
    st.markdown(
        """
        <div class="page-header">
            <h2>✅ Kesimpulan & Saran</h2>
            <p>Rangkuman hasil pemodelan klasifikasi risiko penyakit jantung dan rekomendasi pengembangan</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    best = res["metrics_df"].iloc[0]

    st.markdown(
        f"""
        <div class="content-box">
            <h3 style="margin-top:0; color:#5C2B35 !important;">📌 Kesimpulan Utama</h3>
            <ol style="margin-bottom:0; line-height:1.7;">
                <li><b>Karakteristik Dataset:</b> Dataset <i>Heart Disease Cleveland</i> terdiri dari <b>{len(df_raw)} rekam medis</b> dengan 13 variabel klinis. Distribusi kelas relatif seimbang dengan proporsi pasien terindikasi penyakit jantung sebesar <b>{(df_raw['target']==1).mean()*100:.1f}%</b>.</li>
                <li><b>Efektivitas Preprocessing & PCA:</b> Penerapan <code>KNNImputer</code> dan <code>StandardScaler</code> di dalam <code>Pipeline</code> mencegah <i>data leakage</i>, sementara analisis PCA menunjukkan <b>{res['n_components_90']} komponen utama</b> sudah mampu merepresentasikan minimal <b>90% varians</b> informasi klinis.</li>
                <li><b>Performa Pemodelan:</b> Dari lima algoritma klasifikasi yang diuji, <b>{best['Algoritma']}</b> memberikan kinerja terbaik berdasarkan keseimbangan metrik dengan nilai <b>F1-Score sebesar {best['F1-Score']:.4f}</b>, <b>Accuracy {best['Accuracy']*100:.2f}%</b>, dan <b>ROC-AUC {best['ROC-AUC']:.4f}</b>.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="content-box">
            <h3 style="margin-top:0; color:#5C2B35 !important;">💡 Saran Pengembangan Lanjutan</h3>
            <ul style="margin-bottom:0; line-height:1.7;">
                <li><b>Optimasi Hyperparameter:</b> Melakukan <i>GridSearchCV</i> atau <i>RandomizedSearchCV</i> untuk menyetel parameter model secara lebih mendalam.</li>
                <li><b>Penambahan Sampel Multi-Rumah Sakit:</b> Menggabungkan dataset dari pusat medis lain (Hungarian, Switzerland, VA Long Beach) untuk meningkatkan generalisasi model.</li>
                <li><b>Interpretabilitas Klinis:</b> Menambahkan analisis SHAP (<i>SHapley Additive exPlanations</i>) untuk menjelaskan kontribusi tiap parameter pada setiap pasien secara individual.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# NAVIGASI SIDEBAR BERTINGKAT (SESUAI REFERENSI)
# ==========================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <h3>🫀 Heart Disease ML</h3>
            <p>Cleveland Dataset • CRISP-DM</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

pg = st.navigation(
    {
        "Pengantar": [
            st.Page(page_beranda, title="Beranda", icon="🏠", default=True),
            st.Page(page_tentang_dataset, title="Tentang Dataset", icon="🗄️"),
            st.Page(page_eda, title="Eksplorasi Data (EDA)", icon="📊"),
        ],
        "Pengolahan & Model": [
            st.Page(page_preprocessing, title="Preprocessing", icon="⚙️"),
            st.Page(page_pca, title="Reduksi Dimensi (PCA)", icon="🧬"),
            st.Page(page_modelling, title="Modelling & Evaluasi", icon="🧠"),
        ],
        "Penerapan": [
            st.Page(page_prediksi_satu, title="Prediksi Satu Pasien", icon="🩺"),
            st.Page(page_prediksi_massal, title="Prediksi Massal (CSV)", icon="📄"),
            st.Page(page_kesimpulan, title="Kesimpulan & Saran", icon="✅"),
        ],
    }
)

with st.sidebar:
    st.markdown("---")
    st.markdown(
        """
        <div class="sidebar-footer">
            <div class="team-title">Tugas Machine Learning - Kelompok 2</div>
            <div class="team-sub">Informatika, Universitas Pamulang</div>
            <div class="section-label">Anggota Kelompok:</div>
            <div>1. Haifa Auliatul Pynkan<br>&nbsp;&nbsp;&nbsp;&nbsp;NIM : 241011401848</div>
            <div>2. Rahma Dika Yhustira<br>&nbsp;&nbsp;&nbsp;&nbsp;NIM : 241011402308</div>
            <div>3. Syahwa Rahma Andini<br>&nbsp;&nbsp;&nbsp;&nbsp;NIM : 241011402529</div>
            <div class="section-label">Dosen Pengampu:</div>
            <div>Farizi Ilham S.Kom., M.Kom</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

pg.run()
