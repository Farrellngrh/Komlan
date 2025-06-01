import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from factor_analyzer.factor_analyzer import calculate_kmo
import statsmodels.api as sm
import plotly.graph_objects as go
import chardet
import io

st.set_page_config(page_title="K-Means Clustering", layout="wide")

st.title("📊 K-Means Clustering GUI")

# Global vars
df = None
cluster_labels = []
kmeans_model = None
daerah_col = None

# --- Enhanced File Reading Functions ---
def detect_csv_separator(file_content):
    """Deteksi pemisah CSV yang paling cocok"""
    separators = [';', ',', '\t', '|']
    separator_scores = {}
    
    for sep in separators:
        try:
            # Test dengan beberapa baris pertama
            test_df = pd.read_csv(io.StringIO(file_content[:2000]), sep=sep, nrows=5)
            # Skor berdasarkan jumlah kolom yang masuk akal (2-50 kolom)
            if 2 <= len(test_df.columns) <= 50:
                separator_scores[sep] = len(test_df.columns)
        except:
            separator_scores[sep] = 0
    
    # Kembalikan separator dengan skor tertinggi
    best_sep = max(separator_scores, key=separator_scores.get)
    return best_sep if separator_scores[best_sep] > 1 else ';'

def read_csv_file(uploaded_file):
    """Baca file CSV dengan deteksi encoding dan separator otomatis"""
    try:
        # Baca file sebagai bytes
        file_bytes = uploaded_file.read()
        
        # Deteksi encoding
        encoding_result = chardet.detect(file_bytes)
        detected_encoding = encoding_result['encoding']
        
        # Fallback encodings jika deteksi gagal
        encodings_to_try = [detected_encoding, 'utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'utf-16']
        encodings_to_try = [enc for enc in encodings_to_try if enc is not None]
        
        df = None
        used_encoding = None
        used_separator = None
        
        for encoding in encodings_to_try:
            try:
                # Decode file content
                file_content = file_bytes.decode(encoding)
                
                # Deteksi separator
                separator = detect_csv_separator(file_content)
                
                # Reset file pointer
                uploaded_file.seek(0)
                
                # Baca CSV
                df = pd.read_csv(uploaded_file, sep=separator, encoding=encoding)
                
                used_encoding = encoding
                used_separator = separator
                break
                
            except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
                continue
        
        if df is not None:
            st.success(f"✅ CSV berhasil dibaca dengan encoding: {used_encoding}, separator: '{used_separator}'")
            return df
        else:
            st.error("❌ Gagal membaca file CSV dengan semua encoding yang dicoba")
            return None
            
    except Exception as e:
        st.error(f"❌ Error saat membaca CSV: {e}")
        return None

def read_excel_file(uploaded_file):
    """Baca file Excel dengan penanganan error yang lebih baik"""
    try:
        # Coba baca semua sheet dan ambil yang pertama
        excel_file = pd.ExcelFile(uploaded_file)
        
        if len(excel_file.sheet_names) > 1:
            st.info(f"📋 File Excel memiliki {len(excel_file.sheet_names)} sheet. Menggunakan sheet pertama: '{excel_file.sheet_names[0]}'")
        
        # Baca sheet pertama
        df = pd.read_excel(uploaded_file, sheet_name=0)
        st.success(f"✅ Excel berhasil dibaca dari sheet: '{excel_file.sheet_names[0]}'")
        return df
        
    except Exception as e:
        st.error(f"❌ Error saat membaca Excel: {e}")
        return None

def clean_and_convert_data(df):
    """Bersihkan dan konversi data ke format yang sesuai"""
    if df is None or df.empty:
        return df
    
    # Hapus baris yang sepenuhnya kosong
    df = df.dropna(how='all')
    
    # Bersihkan nama kolom
    df.columns = df.columns.astype(str).str.strip()
    
    # Konversi kolom numerik
    numeric_converted = 0
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # Bersihkan data: hapus spasi, karakter non-numerik kecuali titik dan minus
                cleaned_series = df[col].astype(str).str.strip()
                cleaned_series = cleaned_series.str.replace(r'[^\d\.\-\+eE]+', '', regex=True)
                cleaned_series = cleaned_series.replace('', np.nan)
                
                # Coba konversi ke numerik
                numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
                
                # Jika lebih dari 50% berhasil dikonversi, gunakan versi numerik
                non_null_ratio = numeric_series.notna().sum() / len(numeric_series)
                if non_null_ratio > 0.5:
                    df[col] = numeric_series
                    numeric_converted += 1
                    
            except Exception as e:
                # Biarkan kolom tetap sebagai object jika konversi gagal
                pass
    
    if numeric_converted > 0:
        st.info(f"🔢 {numeric_converted} kolom berhasil dikonversi ke numerik")
    
    return df

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    uploaded_file = st.file_uploader("Unggah file CSV atau Excel", type=["csv", "xls", "xlsx"])

    if uploaded_file is not None:
        filename = uploaded_file.name
        st.info(f"📁 Memproses file: {filename}")
        
        # Reset df
        df = None
        
        # Baca file berdasarkan ekstensi
        if filename.lower().endswith(".csv"):
            df = read_csv_file(uploaded_file)
        elif filename.lower().endswith((".xls", ".xlsx")):
            df = read_excel_file(uploaded_file)
        else:
            st.error("❌ Format file tidak didukung")

        # Proses data jika berhasil dibaca
        if df is not None:
            # Bersihkan dan konversi data
            df = clean_and_convert_data(df)
            
            if df is not None and not df.empty:
                st.success(f"✅ Data berhasil diproses: {df.shape[0]} baris, {df.shape[1]} kolom")
                st.markdown("---")
                
                # Set kolom daerah (kolom pertama)
                if df.columns.size > 0:
                    daerah_col = df.columns[0]
                    st.info(f"🗺️ Kolom identifier: '{daerah_col}'")

                # Tampilkan info kolom
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                non_numeric_cols = df.select_dtypes(exclude=np.number).columns.tolist()
                
                st.markdown("**📊 Info Kolom:**")
                st.markdown(f"- Numerik: {len(numeric_cols)} kolom")
                st.markdown(f"- Non-numerik: {len(non_numeric_cols)} kolom")
                
                # Tampilkan preview data
                st.markdown("**👀 Preview Data:**")
                st.dataframe(df.head(10))
                
                # Tampilkan statistik missing values
                missing_counts = df.isnull().sum()
                if missing_counts.sum() > 0:
                    st.markdown("**⚠️ Missing Values:**")
                    missing_df = pd.DataFrame({
                        'Kolom': missing_counts.index,
                        'Missing': missing_counts.values,
                        'Persentase': (missing_counts.values / len(df) * 100).round(2)
                    })
                    missing_df = missing_df[missing_df['Missing'] > 0]
                    st.dataframe(missing_df)
            else:
                st.error("❌ File kosong atau tidak dapat diproses.")
    else:
        st.info("📂 Silakan unggah file data Anda.")

# --- Fungsi KMO & VIF ---
def hitung_kmo_vif(df_num):
    """Hitung KMO dan VIF dengan error handling yang lebih baik"""
    try:
        # KMO calculation
        kmo_all, kmo_model = calculate_kmo(df_num)
        
        # VIF calculation
        vif_data = pd.DataFrame()
        vif_data["feature"] = df_num.columns
        vif_values = []
        
        for col in df_num.columns:
            try:
                X = df_num.drop(columns=[col])
                y = df_num[col]
                
                # Skip jika ada missing values
                mask = ~(X.isnull().any(axis=1) | y.isnull())
                if mask.sum() < 2:  # Perlu minimal 2 observasi
                    vif_values.append(np.nan)
                    continue
                
                X_clean = X[mask]
                y_clean = y[mask]
                
                model = sm.OLS(y_clean, sm.add_constant(X_clean)).fit()
                vif = 1 / (1 - model.rsquared) if model.rsquared < 0.9999 else np.inf
                vif_values.append(vif)
            except Exception:
                vif_values.append(np.nan)
        
        vif_data["VIF"] = vif_values
        return kmo_model, vif_data
        
    except Exception as e:
        st.error(f"❌ Error dalam perhitungan KMO/VIF: {e}")
        return None, None

# --- Area Utama ---
st.markdown("---")
st.subheader("📊 Proses Clustering")

if df is not None:
    col_kmo_vif, col_kmeans = st.columns(2)

    with col_kmo_vif:
        if st.button("1️⃣ Hitung KMO dan VIF"):
            df_num = df.select_dtypes(include=np.number).dropna()
            if df_num.empty:
                st.warning("⚠️ Tidak ada kolom numerik untuk dianalisis.")
            else:
                with st.spinner("🔄 Menghitung KMO dan VIF..."):
                    kmo_score, vif_df = hitung_kmo_vif(df_num)
                    if kmo_score is not None:
                        st.subheader(f"✔️ Hasil KMO dan VIF")
                        
                        # Interpretasi KMO
                        if kmo_score >= 0.8:
                            kmo_status = "Sangat Baik 🟢"
                        elif kmo_score >= 0.7:
                            kmo_status = "Baik 🟡"
                        elif kmo_score >= 0.6:
                            kmo_status = "Cukup 🟠"
                        else:
                            kmo_status = "Kurang Baik 🔴"
                            
                        st.markdown(f"**KMO Score: {kmo_score:.3f}** ({kmo_status})")
                        st.dataframe(vif_df.round(3))

    with col_kmeans:
        if st.button("2️⃣ Jalankan K-Means Clustering (k=3)"):
            df_num = df.select_dtypes(include=np.number).dropna()
            if df_num.empty:
                st.warning("⚠️ Tidak ada kolom numerik untuk clustering.")
            else:
                with st.spinner("🔄 Menjalankan clustering..."):
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(df_num)

                    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
                    labels = kmeans.fit_predict(X_scaled)
                    
                    # Urutkan cluster berdasarkan centroid
                    centroids_mean = kmeans.cluster_centers_.mean(axis=1)
                    cluster_order = centroids_mean.argsort()
                    label_mapping = {old: new for new, old in enumerate(cluster_order)}
                    sorted_labels = [label_mapping[label] for label in labels]

                    silhouette_avg = silhouette_score(X_scaled, sorted_labels)

                    # Update dataframe dengan hasil cluster
                    cluster_labels = sorted_labels
                    
                    # Hapus kolom Cluster yang sudah ada jika ada
                    if "Cluster" in df.columns:
                        df = df.drop(columns=["Cluster"])
                    
                    df["Cluster"] = cluster_labels

                    st.subheader("✔️ Hasil Clustering")
                    
                    # Interpretasi Silhouette Score
                    if silhouette_avg >= 0.7:
                        sil_status = "Sangat Baik 🟢"
                    elif silhouette_avg >= 0.5:
                        sil_status = "Baik 🟡"
                    elif silhouette_avg >= 0.25:
                        sil_status = "Cukup 🟠"
                    else:
                        sil_status = "Kurang Baik 🔴"
                        
                    st.success(f"✅ Clustering selesai!")
                    st.markdown(f"**Silhouette Score: {silhouette_avg:.3f}** ({sil_status})")
                    
                    # Preview hasil (hindari duplikasi kolom)
                    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                    if "Cluster" in numeric_cols:
                        numeric_cols.remove("Cluster")
                    
                    preview_cols = [daerah_col, "Cluster"] + numeric_cols
                    # Pastikan tidak ada duplikasi kolom
                    preview_cols = list(dict.fromkeys(preview_cols))  # Menghapus duplikasi sambil mempertahankan urutan
                    
                    st.dataframe(df[preview_cols].head(10))

                    # Bar Plot Cluster Count
                    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
                    fig = go.Figure(data=[
                        go.Bar(x=[f"Cluster {i}" for i in cluster_counts.index], 
                               y=cluster_counts.values,
                               marker_color=["#1F618D", "#196F3D", "#B9770E"],
                               text=cluster_counts.values,
                               textposition='auto')
                    ])
                    fig.update_layout(
                        title="📊 Distribusi Jumlah Data per Cluster",
                        xaxis_title="Cluster", 
                        yaxis_title="Jumlah Data",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # Profil Cluster
    if df is not None and "Cluster" in df.columns:
        if st.button("3️⃣ Tampilkan Profil Cluster"):
            st.markdown("---")
            st.subheader("📌 Profil Cluster")
            
            # Profil cluster untuk kolom numerik
            df_num_cluster = df.select_dtypes(include=np.number).copy()
            
            # Pastikan kolom Cluster tidak duplikat
            if "Cluster" in df_num_cluster.columns:
                df_num_cluster = df_num_cluster.drop(columns=["Cluster"])
            
            if "Cluster" in df.columns:
                df_num_cluster["Cluster"] = df["Cluster"]
                cluster_profile = df_num_cluster.groupby("Cluster").agg(['mean', 'std', 'count']).round(3)
                
                st.subheader("📈 Statistik Fitur per Cluster")
                st.dataframe(cluster_profile)

                # Daftar entitas per cluster
                st.subheader(f"📍 Daftar {daerah_col} per Cluster")
                for cluster_num in sorted(df["Cluster"].unique()):
                    cluster_data = df[df["Cluster"] == cluster_num]
                    st.markdown(f"**Cluster {cluster_num}** ({len(cluster_data)} entitas):")
                    
                    # Tampilkan dalam kolom untuk menghemat ruang
                    entities = cluster_data[daerah_col].tolist()
                    cols = st.columns(3)
                    for i, entity in enumerate(entities):
                        cols[i % 3].write(f"• {entity}")
                    st.markdown("---")
            else:
                st.warning("⚠️ Kolom 'Cluster' tidak ditemukan.")

else:
    st.info("📂 Unggah file data untuk memulai analisis.")