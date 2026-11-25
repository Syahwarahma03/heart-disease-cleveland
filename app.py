import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
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
# KONFIGURASI HALAMAN & STYLING
# ==========================================================
st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFF8F7;
        color: #3D1E26;
    }
    section[data-testid="stSidebar"] {
        background-color: #FDECEF;
        border-right: 1px solid #F8D7DA;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #4A232D !important;
    }
    .kpi-card {
        background-color: #FFFFFF;
        border: 1.5px solid #F8D7DA;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 12px rgba(217, 119, 138, 0.08);
        text-align: center;
        margin-bottom: 10px;
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #7A4B56 !important;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.65rem;
        color: #D9778A !important;
        font-weight: 700;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #9E6B76 !important;
        margin-top: 4px;
    }
    .info-banner {
        background-color: #FFFFFF;
        color: #3D1E26 !important;
        border-left: 5px solid #E88C9B;
        border-radius: 10px;
        padding: 15px 20px;
        margin-bottom: 18px;
        box-shadow: 0 2px 8px rgba(217, 119, 138, 0.06);
    }
    h1, h2, h3, [data-testid="stMarkdownContainer"] p, label {
        color: #4A232D !important;
    }
    div.stButton > button, div.stFormSubmitButton > button {
        background-color: #E88C9B;
        color: #FFFFFF !important;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.2rem;
        font-weight: 600;
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        background-color: #D9778A;
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
# LOAD DATA & PIPELINE MODELING
# ==========================================================
@st.cache_data
def load_data(path: str = "heart.csv") -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["?", "NA", "null", ""])
    df.columns = [c.strip().lower() for c in df.columns]
    if "target" not in df.columns and "num" in df.columns:
        df["target"] = (df["num"] > 0).astype(int)
        df.drop(columns=["num"], inplace=True)
    return df


@st.cache_resource
def build_models(
    df: pd.DataFrame,
    test_size: float = 0.20,
    n_estimators: int = 150,
    max_depth: int = 6,
    knn_k: int = 5,
    random_state: int = 42,
):
    df_clean = df.drop_duplicates().copy()
    X = df_clean.drop(columns=["target"])
    y = df_clean["target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=knn_k),
    }

    pipelines = {}
    metrics_list = []
    preds = {}
    probs = {}

    for name, estimator in models.items():
        pipe = Pipeline(
            steps=[
                ("imputer", KNNImputer(n_neighbors=5)),
                ("scaler", StandardScaler()),
                ("classifier", estimator),
            ]
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        cv_acc = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy").mean()

        pipelines[name] = pipe
        preds[name] = y_pred
        probs[name] = y_prob

        metrics_list.append(
            {
                "Model": name,
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
                "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
                "CV 5-Fold": round(cv_acc, 4),
            }
        )

    metrics_df = pd.DataFrame(metrics_list).sort_values(
        by=["F1-Score", "ROC-AUC"], ascending=False
    )

    return {
        "df_clean": df_clean,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "pipelines": pipelines,
        "metrics_df": metrics_df,
        "preds": preds,
        "probs": probs,
    }


try:
    df_raw = load_data("heart.csv")
except FileNotFoundError:
    st.error("File `heart.csv` tidak ditemukan. Letakkan `heart.csv` satu folder dengan `app.py`.")
    st.stop()

# ==========================================================
# SIDEBAR NAVIGATION (SESUAI STRUKTUR TELCO CHURN WEB)
# ==========================================================
st.sidebar.markdown("## 🫀 Heart Disease Analytics")
st.sidebar.caption("Cleveland Clinic Dataset • CRISP-DM")
st.sidebar.markdown("---")

nav_menu = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "EDA",
        "Preprocessing",
        "Modeling",
        "Prediction",
        "Dataset",
    ],
)

st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ Model Hyperparameters", expanded=False):
    test_size = st.slider("Test Split Ratio", 0.15, 0.35, 0.20, 0.05)
    n_estimators = st.slider("RF n_estimators", 50, 300, 150, 25)
    max_depth = st.slider("RF max_depth", 2, 15, 6, 1)
    knn_k = st.slider("KNN Neighbors (K)", 3, 15, 5, 2)

st.sidebar.info(
    "**Alur CRISP-DM:**\n"
    "1. Executive Overview\n"
    "2. Exploratory Data Analysis\n"
    "3. Preprocessing Pipeline\n"
    "4. Modeling & Evaluation\n"
    "5. Clinical Risk Prediction"
)

artifacts = build_models(
    df_raw,
    test_size=test_size,
    n_estimators=n_estimators,
    max_depth=max_depth,
    knn_k=knn_k,
)

# ==========================================================
# 1. EXECUTIVE OVERVIEW
# ==========================================================
if nav_menu == "Executive Overview":
    st.title("🫀 Executive Overview")
    st.markdown(
        """
        <div class="info-banner">
        <b>Ringkasan Proyek:</b> Aplikasi analitik prediktif berbasis Machine Learning untuk mendeteksi risiko 
        penyakit jantung pada dataset <i>Heart Disease Cleveland</i> menggunakan pendekatan klasifikasi terawasi 
        (<b>Random Forest</b>, <b>Logistic Regression</b>, dan <b>K-Nearest Neighbors</b>).
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_records = len(df_raw)
    pos_cases = int((df_raw["target"] == 1).sum())
    neg_cases = int((df_raw["target"] == 0).sum())
    best_model_row = artifacts["metrics_df"].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">Total Pasien</div>'
            f'<div class="kpi-value">{total_records}</div>'
            f'<div class="kpi-sub">13 Fitur Klinis + 1 Target</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">Positif Penyakit Jantung</div>'
            f'<div class="kpi-value">{pos_cases}</div>'
            f'<div class="kpi-sub">{pos_cases / total_records * 100:.1f}% dari total sampel</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">Model Terbaik</div>'
            f'<div class="kpi-value" style="font-size:1.25rem;">{best_model_row["Model"]}</div>'
            f'<div class="kpi-sub">ROC-AUC: {best_model_row["ROC-AUC"]:.4f}</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">Akurasi & F1-Score</div>'
            f'<div class="kpi-value">{best_model_row["Accuracy"] * 100:.1f}%</div>'
            f'<div class="kpi-sub">F1-Score: {best_model_row["F1-Score"]:.4f}</div></div>',
            unsafe_allow_html=True,
        )

    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        st.subheader("Ringkasan Performa Seluruh Model")
        st.dataframe(artifacts["metrics_df"], use_container_width=True, hide_index=True)

        st.subheader("Tahapan Metodologi yang Diterapkan")
        st.markdown(
            """
            - **Business & Data Understanding:** Audit 303+ rekam medis pasien dengan 13 variabel diagnostik non-invasif dan angiografi.
            - **Data Preparation:** Penghapusan duplikat, imputasi nilai hilang berbasis tetangga terdekat (`KNNImputer`), dan standardisasi (`StandardScaler`) di dalam `Pipeline`.
            - **Modeling & Evaluation:** Komparasi 3 algoritma klasifikasi menggunakan validasi silang *5-Fold* serta evaluasi metrik *Accuracy, Precision, Recall, F1-Score,* dan *ROC-AUC*.
            """
        )

    with col_right:
        st.subheader("Proporsi Diagnosis Pasien")
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df_raw["target"].value_counts().sort_index()
        bars = ax.bar(
            ["Sehat (0)", "Penyakit Jantung (1)"],
            counts.values,
            color=["#F8D7DA", "#E88C9B"],
            edgecolor="#D9778A",
            width=0.55,
        )
        ax.set_ylabel("Jumlah Pasien")
        ax.set_title("Distribusi Kelas Target")
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 2,
                f"{int(b.get_height())}",
                ha="center",
                fontweight="bold",
                color="#5C2B35",
            )
        st.pyplot(fig)


# ==========================================================
# 2. EDA (EXPLORATORY DATA ANALYSIS)
# ==========================================================
elif nav_menu == "EDA":
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.markdown(
        """
        <div class="info-banner">
        Eksplorasi pola distribusi variabel klinis, perbandingan antar kelompok pasien sehat vs terindikasi penyakit jantung, 
        serta korelasi antar fitur sebelum pemodelan.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(
        ["Distribusi Fitur Numerik", "Analisis Kategorikal Klinis", "Matriks Korelasi"]
    )

    with tab1:
        num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        selected_num = st.selectbox("Pilih Fitur Numerik:", num_cols, index=0)

        col_a, col_b = st.columns(2)
        with col_a:
            fig1, ax1 = plt.subplots(figsize=(5.5, 4))
            sns.histplot(
                data=df_raw,
                x=selected_num,
                hue="target",
                kde=True,
                palette=["#F4A6B8", "#C94A6A"],
                ax=ax1,
            )
            ax1.set_title(f"Distribusi {selected_num} Berdasarkan Diagnosis")
            ax1.legend(title="Target", labels=["Penyakit Jantung (1)", "Sehat (0)"])
            st.pyplot(fig1)

        with col_b:
            fig2, ax2 = plt.subplots(figsize=(5.5, 4))
            sns.boxplot(
                data=df_raw,
                x="target",
                y=selected_num,
                palette=["#F8D7DA", "#E88C9B"],
                ax=ax2,
            )
            ax2.set_xticklabels(["Sehat (0)", "Penyakit Jantung (1)"])
            ax2.set_title(f"Sebaran Kuartil {selected_num}")
            st.pyplot(fig2)

    with tab2:
        cat_cols = ["cp", "sex", "exang", "slope", "ca", "thal"]
        selected_cat = st.selectbox("Pilih Fitur Kategorikal:", cat_cols, index=0)

        fig3, ax3 = plt.subplots(figsize=(8, 4))
        sns.countplot(
            data=df_raw,
            x=selected_cat,
            hue="target",
            palette=["#F8D7DA", "#E88C9B"],
            ax=ax3,
        )
        ax3.set_title(f"Perbandingan Diagnosis Berdasarkan Fitur '{selected_cat}'")
        ax3.set_ylabel("Jumlah Pasien")
        ax3.legend(title="Diagnosis", labels=["Sehat (0)", "Penyakit Jantung (1)"])
        st.pyplot(fig3)
        st.caption(f"Keterangan atribut: {FEATURE_INFO.get(selected_cat, '')}")

    with tab3:
        fig4, ax4 = plt.subplots(figsize=(10, 6.5))
        corr = df_raw.corr(numeric_only=True)
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="RdPu",
            ax=ax4,
            annot_kws={"size": 7.5},
        )
        ax4.set_title("Heatmap Korelasi Pearson Antar Variabel")
        st.pyplot(fig4)


# ==========================================================
# 3. PREPROCESSING
# ==========================================================
elif nav_menu == "Preprocessing":
    st.title("🧹 Data Preprocessing & Pipeline")
    st.markdown(
        """
        <div class="info-banner">
        <b>Pencegahan Data Leakage:</b> Seluruh langkah imputasi dan normalisasi dijalankan melalui 
        <code>sklearn.pipeline.Pipeline</code> yang hanya di-<i>fit</i> pada data latih (<code>X_train</code>).
        </div>
        """,
        unsafe_allow_html=True,
    )

    dup_removed = int(df_raw.duplicated().sum())
    df_clean = artifacts["df_clean"]
    X_train = artifacts["X_train"]
    X_test = artifacts["X_test"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">Missing Values Awal</div>'
            f'<div class="kpi-value">{int(df_raw.isna().sum().sum())}</div>'
            f'<div class="kpi-sub">Diimputasi KNNImputer (k=5)</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">Duplikat Dihapus</div>'
            f'<div class="kpi-value">{dup_removed}</div>'
            f'<div class="kpi-sub">Sisa data bersih: {len(df_clean)} baris</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">Data Latih (Train)</div>'
            f'<div class="kpi-value">{len(X_train)}</div>'
            f'<div class="kpi-sub">{(1 - test_size) * 100:.0f}% Stratified Split</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">Data Uji (Test)</div>'
            f'<div class="kpi-value">{len(X_test)}</div>'
            f'<div class="kpi-sub">{test_size * 100:.0f}% Stratified Split</div></div>',
            unsafe_allow_html=True,
        )

    st.subheader("1. Audit Kelengkapan & Transformasi Kolom")
    audit_table = pd.DataFrame(
        {
            "Kolom": df_raw.columns,
            "Tipe Data": [str(df_raw[c].dtype) for c in df_raw.columns],
            "Missing Values": df_raw.isna().sum().values,
            "Perlakuan Preprocessing": [
                "Variabel Target (Label Biner 0/1)"
                if c == "target"
                else "KNNImputer(n_neighbors=5) → StandardScaler()"
                for c in df_raw.columns
            ],
        }
    )
    st.dataframe(audit_table, use_container_width=True, hide_index=True)

    st.subheader("2. Perbandingan Data Sebelum vs Sesudah Scaling (X_train)")
    t_raw, t_scaled = st.tabs(["Sebelum Preprocessing (Asli)", "Sesudah Imputasi & StandardScaler"])

    with t_raw:
        st.dataframe(X_train.head(8), use_container_width=True)

    with t_scaled:
        preview_pipe = Pipeline(
            [("imputer", KNNImputer(n_neighbors=5)), ("scaler", StandardScaler())]
        )
        X_scaled_df = pd.DataFrame(
            preview_pipe.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        st.dataframe(X_scaled_df.head(8).round(4), use_container_width=True)


# ==========================================================
# 4. MODELING
# ==========================================================
elif nav_menu == "Modeling":
    st.title("🤖 Modeling & Evaluasi Performa")
    st.markdown(
        """
        <div class="info-banner">
        Evaluasi komparatif antara <b>Random Forest</b>, <b>Logistic Regression</b>, dan <b>K-Nearest Neighbors</b> 
        menggunakan data uji terstratifikasi serta validasi silang 5-Fold.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("1. Tabel Komparasi Metrik Evaluasi")
    st.dataframe(artifacts["metrics_df"], use_container_width=True, hide_index=True)

    chosen_model = st.selectbox(
        "Pilih Model untuk Analisis Detail:",
        list(artifacts["pipelines"].keys()),
        index=0,
    )

    y_test = artifacts["y_test"]
    y_pred = artifacts["preds"][chosen_model]

    col_cm, col_roc = st.columns(2)
    with col_cm:
        st.subheader(f"2. Confusion Matrix ({chosen_model})")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="RdPu",
            xticklabels=["Prediksi Sehat (0)", "Prediksi Sakit (1)"],
            yticklabels=["Aktual Sehat (0)", "Aktual Sakit (1)"],
            ax=ax_cm,
        )
        ax_cm.set_xlabel("Prediksi Model")
        ax_cm.set_ylabel("Nilai Aktual")
        st.pyplot(fig_cm)

    with col_roc:
        st.subheader("3. Perbandingan Kurva ROC-AUC")
        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
        palette_roc = ["#D9778A", "#6C5B7B", "#F67280"]
        for (m_name, y_prob), col_hex in zip(artifacts["probs"].items(), palette_roc):
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = roc_auc_score(y_test, y_prob)
            ax_roc.plot(fpr, tpr, lw=2, color=col_hex, label=f"{m_name} (AUC={auc_score:.3f})")

        ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Random Baseline")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(loc="lower right", fontsize=8.5)
        st.pyplot(fig_roc)

    st.subheader(f"4. Classification Report ({chosen_model})")
    rep_df = pd.DataFrame(
        classification_report(
            y_test,
            y_pred,
            target_names=["Sehat (0)", "Penyakit Jantung (1)"],
            output_dict=True,
        )
    ).T.round(4)
    st.dataframe(rep_df, use_container_width=True)

    st.subheader("5. Feature Importance (Random Forest)")
    rf_clf = artifacts["pipelines"]["Random Forest"].named_steps["classifier"]
    feat_imp = pd.Series(
        rf_clf.feature_importances_, index=artifacts["X_train"].columns
    ).sort_values(ascending=True)

    fig_fi, ax_fi = plt.subplots(figsize=(9, 4.5))
    feat_imp.plot(kind="barh", color="#E88C9B", edgecolor="#D9778A", ax=ax_fi)
    ax_fi.set_xlabel("Skor Kepentingan Fitur")
    ax_fi.set_title("Kontribusi Fitur Klinis terhadap Prediksi Risiko")
    st.pyplot(fig_fi)


# ==========================================================
# 5. PREDICTION
# ==========================================================
elif nav_menu == "Prediction":
    st.title("🩺 Clinical Risk Prediction")
    st.markdown(
        """
        <div class="info-banner">
        Masukkan parameter klinis pasien untuk memprediksi risiko penyakit jantung secara <i>real-time</i> 
        menggunakan pipeline model terpilih.
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_pred_model = st.selectbox(
        "Pilih Algoritma Prediksi:",
        list(artifacts["pipelines"].keys()),
        index=0,
    )

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Usia (age)", min_value=18, max_value=100, value=54)
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
            chol = st.number_input("Kolesterol Serum mg/dl (chol)", 100, 600, 246)

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
                    1: "1 - Kelainan ST-T",
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

        submit_btn = st.form_submit_button("🔍 Prediksi Risiko Sekarang")

    if submit_btn:
        patient_df = pd.DataFrame(
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
        )[artifacts["X_train"].columns]

        pipe = artifacts["pipelines"][selected_pred_model]
        pred_label = int(pipe.predict(patient_df)[0])
        pred_prob = pipe.predict_proba(patient_df)[0]

        st.markdown("---")
        if pred_label == 1:
            st.error(
                f"**Hasil Prediksi ({selected_pred_model}): TERINDIKASI PENYAKIT JANTUNG (Kelas 1)**\n\n"
                f"Probabilitas Risiko: **{pred_prob[1] * 100:.2f}%** | Probabilitas Sehat: **{pred_prob[0] * 100:.2f}%**"
            )
        else:
            st.success(
                f"**Hasil Prediksi ({selected_pred_model}): TIDAK TERINDIKASI PENYAKIT JANTUNG (Kelas 0)**\n\n"
                f"Probabilitas Sehat: **{pred_prob[0] * 100:.2f}%** | Probabilitas Risiko: **{pred_prob[1] * 100:.2f}%**"
            )
        st.caption(
            "Disclaimer: Simulasi ini ditujukan untuk keperluan akademik dan tidak menggantikan pemeriksaan dokter."
        )


# ==========================================================
# 6. DATASET
# ==========================================================
elif nav_menu == "Dataset":
    st.title("📁 Dataset Heart Disease Cleveland")
    st.markdown(
        """
        <div class="info-banner">
        Eksplorasi data mentah, kamus deskripsi fitur klinis, statistik ringkasan, serta unduh file CSV.
        </div>
        """,
        unsafe_allow_html=True,
    )

    filter_target = st.radio(
        "Filter Kelas Diagnosis:",
        ["Semua Data", "Sehat (0)", "Penyakit Jantung (1)"],
        horizontal=True,
    )

    df_view = df_raw.copy()
    if filter_target == "Sehat (0)":
        df_view = df_view[df_view["target"] == 0]
    elif filter_target == "Penyakit Jantung (1)":
        df_view = df_view[df_view["target"] == 1]

    st.dataframe(df_view, use_container_width=True)

    csv_bytes = df_view.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Unduh Dataset (CSV)",
        data=csv_bytes,
        file_name="heart_disease_cleveland_filtered.csv",
        mime="text/csv",
    )

    col_dict, col_stat = st.columns([1, 1.1])
    with col_dict:
        st.subheader("Kamus Atribut")
        dict_df = pd.DataFrame(
            [
                {"Kolom": col, "Deskripsi": FEATURE_INFO.get(col, "-")}
                for col in df_raw.columns
            ]
        )
        st.dataframe(dict_df, use_container_width=True, hide_index=True)

    with col_stat:
        st.subheader("Statistik Deskriptif")
        st.dataframe(df_raw.describe().T.round(2), use_container_width=True)
