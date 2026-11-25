import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
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
    confusion_matrix
)

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Heart Disease Cleveland | Kelompok 2",
    page_icon="🎀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CUSTOM CSS: SOFT PINK AESTHETIC
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

section[data-testid="stSidebar"] {
    border-right: 1.5px solid #fbcfe8;
}

span[data-baseweb="tag"] {
    background-color: #ec4899 !important;
    border-radius: 8px !important;
}
span[data-baseweb="tag"] span {
    color: #ffffff !important;
    font-weight: 600 !important;
}

.hero-banner {
    background: linear-gradient(120deg, #f43f5e 0%, #ec4899 55%, #d946ef 100%);
    padding: 2rem 2.4rem;
    border-radius: 22px;
    color: #ffffff;
    box-shadow: 0 10px 25px rgba(236, 72, 153, 0.22);
    margin-bottom: 1.6rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.25);
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-bottom: 0.7rem;
    color: #ffffff;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0 0 0.4rem 0;
    color: #ffffff;
}
.hero-desc {
    font-size: 0.96rem;
    line-height: 1.6;
    color: #fff1f2;
    margin: 0;
    max-width: 820px;
}

.metric-card {
    background-color: #ffffff;
    border-radius: 18px;
    padding: 1.25rem 1.4rem;
    border: 1.5px solid #fbcfe8;
    box-shadow: 0 6px 18px rgba(244, 114, 182, 0.08);
}
.metric-label {
    font-size: 0.85rem;
    font-weight: 700;
    color: #9d174d;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #500724;
    margin-bottom: 0.45rem;
}
.metric-pill {
    display: inline-block;
    background-color: #fce7f3;
    color: #be185d;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 999px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #fce7f3;
    padding: 6px;
    border-radius: 14px;
    border: 1px solid #f9a8d4;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background-color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(219, 39, 119, 0.15);
}

.stButton > button, div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(90deg, #f43f5e 0%, #ec4899 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.6rem !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOAD DATASET & TRAIN PIPELINE (CACHED)
# ==========================================
@st.cache_data
def load_default_data():
    return pd.read_csv("heart.csv")

@st.cache_resource
def train_and_evaluate_models(df_train_source: pd.DataFrame):
    X = df_train_source.drop(columns=["target"])
    y = df_train_source["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_configs = {
        "Random Forest (Utama)": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5)
    }

    trained_pipelines = {}
    metrics_list = []
    conf_matrices = {}

    for name, clf in model_configs.items():
        pipe = Pipeline([
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler()),
            ("classifier", clf)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        trained_pipelines[name] = pipe
        conf_matrices[name] = confusion_matrix(y_test, y_pred)
        metrics_list.append({
            "Algoritma": name,
            "Accuracy (%)": round(accuracy_score(y_test, y_pred) * 100, 2),
            "Precision (%)": round(precision_score(y_test, y_pred) * 100, 2),
            "Recall (%)": round(recall_score(y_test, y_pred) * 100, 2),
            "F1-Score (%)": round(f1_score(y_test, y_pred) * 100, 2),
            "ROC-AUC (%)": round(roc_auc_score(y_test, y_prob) * 100, 2)
        })

    rf_model = trained_pipelines["Random Forest (Utama)"].named_steps["classifier"]
    feat_imp = pd.DataFrame({
        "Fitur": X.columns,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=True)

    return trained_pipelines, pd.DataFrame(metrics_list), conf_matrices, feat_imp, len(X_train), len(X_test)

cp_map = {
    1: "Typical Angina (1)",
    2: "Atypical Angina (2)",
    3: "Non-anginal Pain (3)",
    4: "Asymptomatic (4)"
}
target_map = {
    0: "Sehat (0)",
    1: "Sakit Jantung (1)"
}

# ==========================================
# 4. SIDEBAR FILTER & DATASET UPLOADER
# ==========================================
with st.sidebar:
    st.markdown("## 🎀 Pengaturan Dashboard")
    st.caption("Sesuaikan filter klinis di bawah ini untuk melihat analisis data secara real-time ✨")

    with st.expander("📂 Sumber Dataset (Opsional)"):
        uploaded_file = st.file_uploader("Upload CSV baru (format sama)", type=["csv"])

    if uploaded_file is not None:
        df_asli = pd.read_csv(uploaded_file)
        st.success("Menggunakan dataset unggahan!")
    else:
        df_asli = load_default_data()

    st.divider()

    df_display = df_asli.copy()
    df_display["cp_label"] = df_display["cp"].map(cp_map)
    df_display["target_label"] = df_display["target"].map(target_map)

    pilih_cp = st.multiselect(
        "🌸 Tipe Nyeri Dada (cp)",
        options=list(cp_map.values()),
        default=list(cp_map.values())
    )

    pilih_target = st.multiselect(
        "💗 Status Diagnosis (target)",
        options=list(target_map.values()),
        default=list(target_map.values())
    )

    min_chol, max_chol = float(df_asli["chol"].min()), float(df_asli["chol"].max())
    rentang_chol = st.slider(
        "🍓 Rentang Kolesterol (mg/dl)",
        min_value=min_chol,
        max_value=max_chol,
        value=(min_chol, max_chol)
    )

    min_age, max_age = int(df_asli["age"].min()), int(df_asli["age"].max())
    rentang_age = st.slider(
        "🌷 Rentang Usia Pasien (tahun)",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age)
    )

    st.divider()
    st.caption("Made with 💕 by Kelompok 2 • Teknik Informatika UNPAM")

trained_pipelines, metrics_df, conf_matrices, feat_imp_df, n_train, n_test = train_and_evaluate_models(df_asli)

df = df_display[
    (df_display["cp_label"].isin(pilih_cp)) &
    (df_display["target_label"].isin(pilih_target)) &
    (df_display["chol"].between(rentang_chol[0], rentang_chol[1])) &
    (df_display["age"].between(rentang_age[0], rentang_age[1]))
]

# ==========================================
# 5. HERO BANNER
# ==========================================
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-badge">✨ KELOMPOK 2 • PRAKTIKUM MACHINE LEARNING (UNPAM)</div>
    <div class="hero-title">Heart Disease Cleveland 🎀</div>
    <p class="hero-desc">
        Dashboard klinis dan klasifikasi Machine Learning berbasis metodologi <b>CRISP-DM</b> menggunakan 
        <b>KNNImputer</b>, <b>StandardScaler</b>, serta komparasi algoritma <b>Random Forest</b>, <b>Logistic Regression</b>, 
        dan <b>K-Nearest Neighbors</b> pada dataset Cleveland Clinic Foundation (<b>{len(df_asli)} pasien</b>, 14 atribut).
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 6. TABS UTAMA
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Ringkasan",
    "🔍 Eksplorasi",
    "🗺️ Peta Klinis",
    "🤖 Evaluasi Model",
    "📁 Dataset",
    "💌 Simulasi Prediksi"
])

# --- TAB 1: RINGKASAN ---
with tab1:
    st.subheader("🌸 Ringkasan Dataset Klinis")
    st.write("Gambaran umum kondisi kesehatan pasien berdasarkan filter yang sedang aktif.")

    if len(df) == 0:
        st.warning("Tidak ada data yang cocok dengan filter saat ini. Coba perluas pilihan filter di sebelah kiri ya! 🌷")
    else:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Sampel Pasien</div>
                <div class="metric-value">{len(df)}</div>
                <div class="metric-pill">✨ dari {len(df_asli)} total data</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            avg_chol = df["chol"].mean()
            skew_chol = df["chol"].skew()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Rata-rata Kolesterol</div>
                <div class="metric-value">{avg_chol:.1f} <span style="font-size:1rem;">mg/dl</span></div>
                <div class="metric-pill">📈 {skew_chol:+.2f} skewness</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            avg_thalach = df["thalach"].mean()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Detak Jantung Maks</div>
                <div class="metric-value">{avg_thalach:.1f} <span style="font-size:1rem;">bpm</span></div>
                <div class="metric-pill">💓 Rata-rata thalach</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            sakit_count = int((df["target"] == 1).sum())
            persen_sakit = (sakit_count / len(df)) * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Terindikasi Sakit</div>
                <div class="metric-value">{sakit_count} <span style="font-size:1rem;">pasien</span></div>
                <div class="metric-pill">🩺 {persen_sakit:.1f}% dari sampel</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📈 Statistik Deskriptif Fitur Numerik Utama")
        st.dataframe(
            df[["age", "trestbps", "chol", "thalach", "oldpeak"]].describe().T.round(2),
            use_container_width=True
        )

# --- TAB 2: EKSPLORASI ---
with tab2:
    st.subheader("🔍 Visualisasi & Eksplorasi Klinis")
    if len(df) == 0:
        st.warning("Tidak ada data yang cocok dengan filter saat ini. Coba perluas pilihan filter di sebelah kiri ya! 🌷")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Distribusi Status Diagnosis Pasien**")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            fig1.patch.set_facecolor("#ffffff")
            ax1.set_facecolor("#ffffff")
            counts = df["target_label"].value_counts()
            bars = ax1.bar(counts.index, counts.values, color=["#f472b6", "#e11d48"], width=0.45)
            for bar in bars:
                yval = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2, yval + 1.5, int(yval),
                         ha="center", va="bottom", fontweight="bold", color="#500724")
            ax1.set_ylabel("Jumlah Pasien", color="#500724")
            ax1.set_ylim(0, max(counts.values) * 1.15)
            ax1.tick_params(colors="#500724")
            for spine in ["top", "right"]:
                ax1.spines[spine].set_visible(False)
            st.pyplot(fig1)
            plt.close(fig1)

        with c2:
            st.markdown("**Sebaran Usia vs Detak Jantung Maksimum**")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            fig2.patch.set_facecolor("#ffffff")
            ax2.set_facecolor("#ffffff")
            for tgt_val, label_txt, col_hex in [(0, "Sehat (0)", "#f472b6"), (1, "Sakit Jantung (1)", "#be123c")]:
                subset = df[df["target"] == tgt_val]
                ax2.scatter(subset["age"], subset["thalach"], c=col_hex, label=label_txt,
                            alpha=0.85, edgecolors="white", s=65)
            ax2.set_xlabel("Usia (Tahun)", color="#500724")
            ax2.set_ylabel("Detak Jantung Maks (bpm)", color="#500724")
            ax2.legend(frameon=True, facecolor="#fff1f2")
            ax2.tick_params(colors="#500724")
            for spine in ["top", "right"]:
                ax2.spines[spine].set_visible(False)
            st.pyplot(fig2)
            plt.close(fig2)

        st.markdown("<br>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Distribusi Usia Berdasarkan Diagnosis**")
            fig3, ax3 = plt.subplots(figsize=(6, 3.8))
            ax3.hist(df[df["target"] == 0]["age"], bins=12, alpha=0.7, color="#f472b6", label="Sehat (0)")
            ax3.hist(df[df["target"] == 1]["age"], bins=12, alpha=0.7, color="#be123c", label="Sakit Jantung (1)")
            ax3.set_xlabel("Usia (Tahun)", color="#500724")
            ax3.set_ylabel("Frekuensi", color="#500724")
            ax3.legend(frameon=True, facecolor="#fff1f2")
            for spine in ["top", "right"]:
                ax3.spines[spine].set_visible(False)
            st.pyplot(fig3)
            plt.close(fig3)

        with c4:
            st.markdown("**Distribusi Depresi ST (`oldpeak`) Berdasarkan Diagnosis**")
            fig4, ax4 = plt.subplots(figsize=(6, 3.8))
            ax4.hist(df[df["target"] == 0]["oldpeak"], bins=12, alpha=0.7, color="#f472b6", label="Sehat (0)")
            ax4.hist(df[df["target"] == 1]["oldpeak"], bins=12, alpha=0.7, color="#be123c", label="Sakit Jantung (1)")
            ax4.set_xlabel("ST Depression (oldpeak)", color="#500724")
            ax4.set_ylabel("Frekuensi", color="#500724")
            ax4.legend(frameon=True, facecolor="#fff1f2")
            for spine in ["top", "right"]:
                ax4.spines[spine].set_visible(False)
            st.pyplot(fig4)
            plt.close(fig4)

# --- TAB 3: PETA KLINIS ---
with tab3:
    st.subheader("🗺️ Peta Hubungan Tipe Nyeri Dada & Korelasi Klinis")
    if len(df) == 0:
        st.warning("Tidak ada data yang cocok dengan filter saat ini. Coba perluas pilihan filter di sebelah kiri ya! 🌷")
    else:
        col_k1, col_k2 = st.columns([1.1, 1])
        with col_k1:
            st.markdown("**Perbandingan Diagnosis per Tipe Nyeri Dada (`cp`)**")
            crosstab_cp = pd.crosstab(df["cp_label"], df["target_label"])
            warna_peta = {"Sehat (0)": "#f472b6", "Sakit Jantung (1)": "#e11d48"}
            list_warna = [warna_peta[col] for col in crosstab_cp.columns]
            st.bar_chart(crosstab_cp, color=list_warna)

        with col_k2:
            st.markdown("**Korelasi Setiap Fitur terhadap Target Diagnosis**")
            corr_target = df_asli.corr(numeric_only=True)["target"].drop("target").sort_values()
            fig_corr, ax_corr = plt.subplots(figsize=(6, 4.3))
            colors_corr = ["#f472b6" if v < 0 else "#e11d48" for v in corr_target.values]
            ax_corr.barh(corr_target.index, corr_target.values, color=colors_corr)
            ax_corr.axvline(0, color="#831843", linewidth=0.8)
            ax_corr.set_xlabel("Koefisien Korelasi Pearson", color="#500724")
            for spine in ["top", "right"]:
                ax_corr.spines[spine].set_visible(False)
            st.pyplot(fig_corr)
            plt.close(fig_corr)

# --- TAB 4: EVALUASI MODEL ---
with tab4:
    st.subheader("🤖 Evaluasi & Perbandingan Performa Model (CRISP-DM)")
    st.write(
        f"Evaluasi dilakukan dengan pembagian data terstratifikasi (**{n_train} data latih** : **{n_test} data uji**) "
        "menggunakan `Pipeline` (`KNNImputer` $\\rightarrow$ `StandardScaler` $\\rightarrow$ Classifier) untuk mencegah kebocoran data (*data leakage*)."
    )

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("**Confusion Matrix (Data Uji)**")
        pilih_model_eval = st.selectbox(
            "Pilih Model untuk Melihat Confusion Matrix:",
            options=list(trained_pipelines.keys()),
            index=0
        )
        cm = conf_matrices[pilih_model_eval]
        fig_cm, ax_cm = plt.subplots(figsize=(5.2, 4))
        ax_cm.imshow(cm, cmap="RdPu")
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["Prediksi Sehat (0)", "Prediksi Sakit (1)"])
        ax_cm.set_yticklabels(["Aktual Sehat (0)", "Aktual Sakit (1)"])
        for i in range(2):
            for j in range(2):
                text_color = "white" if cm[i, j] > cm.max() / 2 else "#500724"
                ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center",
                           fontsize=14, fontweight="bold", color=text_color)
        ax_cm.set_title(pilih_model_eval, color="#500724", fontweight="bold")
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with col_m2:
        st.markdown("**Tingkat Kepentingan Fitur (Feature Importance - Random Forest)**")
        fig_fi, ax_fi = plt.subplots(figsize=(6, 4.5))
        ax_fi.barh(feat_imp_df["Fitur"], feat_imp_df["Importance"], color="#ec4899")
        ax_fi.set_xlabel("Skor Importance", color="#500724")
        ax_fi.tick_params(colors="#500724")
        for spine in ["top", "right"]:
            ax_fi.spines[spine].set_visible(False)
        st.pyplot(fig_fi)
        plt.close(fig_fi)

# --- TAB 5: DATASET ---
with tab5:
    st.subheader("📁 Tabel Data Pasien (Cleveland Clinic)")
    st.write(f"Menampilkan **{len(df)}** baris data sesuai filter aktif:")
    df_clean_view = df.drop(columns=["cp_label", "target_label"])
    st.dataframe(df_clean_view, use_container_width=True)

    csv_export = df_clean_view.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Unduh Data Terfilter (CSV)",
        data=csv_export,
        file_name="heart_cleveland_filtered.csv",
        mime="text/csv"
    )

# --- TAB 6: SIMULASI PREDIKSI ---
with tab6:
    st.subheader("💌 Simulasi Prediksi Risiko Penyakit Jantung")
    st.write("Pilih algoritma klasifikasi lalu masukkan parameter klinis pasien untuk melihat hasil prediksi serta tingkat keyakinan model.")

    model_pilihan = st.selectbox(
        "🧠 Pilih Algoritma Klasifikasi:",
        options=list(trained_pipelines.keys()),
        index=0
    )

    with st.form("form_prediksi"):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            age = st.number_input("Usia (age)", min_value=1, max_value=120, value=52)
            trestbps = st.number_input("Tekanan Darah Istirahat (trestbps)", min_value=50, max_value=250, value=125)
            restecg = st.selectbox(
                "Hasil EKG Istirahat (restecg)",
                options=[0, 1, 2],
                format_func=lambda x: {
                    0: "0 - Normal",
                    1: "1 - Kelainan Gelombang ST-T",
                    2: "2 - Hipertrofi Ventrikel Kiri"
                }[x]
            )
            oldpeak = st.number_input("Depresi ST (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            thal = st.selectbox(
                "Thalassemia (thal)",
                options=[3, 6, 7],
                format_func=lambda x: {
                    3: "3 - Normal",
                    6: "6 - Fixed Defect",
                    7: "7 - Reversible Defect"
                }[x]
            )

        with col_b:
            sex = st.selectbox(
                "Jenis Kelamin (sex)",
                options=[1, 0],
                format_func=lambda x: "1 - Pria" if x == 1 else "0 - Wanita"
            )
            chol = st.number_input("Kolesterol Serum (chol)", min_value=100, max_value=600, value=212)
            thalach = st.number_input("Detak Jantung Maksimum (thalach)", min_value=50, max_value=250, value=168)
            slope = st.selectbox(
                "Kemiringan Segmen ST (slope)",
                options=[1, 2, 3],
                format_func=lambda x: {
                    1: "1 - Upsloping",
                    2: "2 - Flat",
                    3: "3 - Downsloping"
                }[x]
            )

        with col_c:
            cp = st.selectbox(
                "Tipe Nyeri Dada (cp)",
                options=[1, 2, 3, 4],
                format_func=lambda x: cp_map[x]
            )
            fbs = st.selectbox(
                "Gula Darah Puasa > 120 mg/dl (fbs)",
                options=[0, 1],
                format_func=lambda x: "0 - Tidak" if x == 0 else "1 - Ya"
            )
            exang = st.selectbox(
                "Angina Akibat Olahraga (exang)",
                options=[0, 1],
                format_func=lambda x: "0 - Tidak" if x == 0 else "1 - Ya"
            )
            ca = st.selectbox(
                "Jumlah Pembuluh Darah Utama (ca)",
                options=[0, 1, 2, 3]
            )

        submit_btn = st.form_submit_button("✨ Prediksi Diagnosis Sekarang")

    if submit_btn:
        fitur_input = pd.DataFrame([{
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
            "thal": thal
        }])

        active_pipeline = trained_pipelines[model_pilihan]
        prediksi = active_pipeline.predict(fitur_input)[0]
        probabilitas = active_pipeline.predict_proba(fitur_input)[0]
        prob_sehat = probabilitas[0] * 100
        prob_sakit = probabilitas[1] * 100

        st.markdown("---")
        res_col1, res_col2 = st.columns([1.4, 1])

        with res_col1:
            if prediksi == 1:
                st.error(
                    f"🚨 **Hasil Prediksi ({model_pilihan}): Terindikasi Risiko Penyakit Jantung (Target = 1)**\n\n"
                    "Disarankan melakukan pemeriksaan klinis lanjutan dengan tenaga medis."
                )
            else:
                st.success(
                    f"🌸 **Hasil Prediksi ({model_pilihan}): Sehat / Tidak Terindikasi Penyakit Jantung (Target = 0)**\n\n"
                    "Parameter klinis pasien berada dalam kategori risiko rendah."
                )

        with res_col2:
            st.metric("Probabilitas Sehat (0)", f"{prob_sehat:.1f}%")
            st.metric("Probabilitas Risiko Sakit (1)", f"{prob_sakit:.1f}%")
            st.progress(max(0, min(100, int(round(prob_sakit)))))