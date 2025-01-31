import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.tree import plot_tree
from database import get_connection  # Import koneksi database
import io
# Fungsi untuk mengonversi DataFrame ke format Excel dan mengunduhnya
def to_excel(df):
    # Membuat objek BytesIO untuk menyimpan file Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    output.seek(0)
    return output

def pengelolaan_model():
    st.title("Pengelolaan Model")
    st.write("Halaman ini digunakan untuk mengelola model yang digunakan dalam aplikasi.")
    
    # Membuat koneksi ke database
    conn = get_connection()
    if conn is None:
        st.error("Koneksi ke database gagal!")
        return

    try:
        cursor = conn.cursor()
        query = "SELECT tahun, provinsi, kabupatenkota, plastik, kertas_karton, logam, kaca FROM datasetbersih"
        cursor.execute(query)
        data = cursor.fetchall()
        df = pd.DataFrame(data, columns=['tahun', 'provinsi', 'kabupatenkota', 'plastik', 'kertas_karton', 'logam', 'kaca'])

        # Proses clustering
        data_clustering = df[['plastik', 'kertas_karton', 'logam', 'kaca']].dropna()
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data_clustering)
        kmeans = KMeans(n_clusters=3, random_state=42)
        data_clustering['cluster'] = kmeans.fit_predict(data_normalized)
        cluster_labels = {0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2'}
        data_clustering['hasil_cluster'] = data_clustering['cluster'].map(cluster_labels)
        df['hasil_cluster'] = data_clustering['hasil_cluster']

        silhouette_values = silhouette_samples(data_normalized, data_clustering['cluster'])
        data_clustering['silhouette_score'] = silhouette_values
        silhouette_per_cluster = data_clustering.groupby('cluster')['silhouette_score'].mean()
        db_index = davies_bouldin_score(data_normalized, data_clustering['cluster'])
        
        # Menambahkan kolom prediksi nilai ekonomis
        def nilai_ekonomis(cluster):
            if cluster == 'Cluster 0':
                return 'Rendah'
            elif cluster == 'Cluster 1':
                return 'Sedang'
            elif cluster == 'Cluster 2':
                return 'Tinggi'
            else:
                return 'Tidak Dikenal'

        df['prediksi_nilai_ekonomis'] = df['hasil_cluster'].apply(nilai_ekonomis)

        # Tabs untuk hasil
        tab1, tab2, tab3 = st.tabs(["Hasil Clustering", "Hasil Klasifikasi", "Evaluasi Model"])

        with tab1:
            st.subheader("Hasil Clustering")
            st.write(df[['provinsi', 'kabupatenkota', 'plastik', 'logam', 'kaca', 'hasil_cluster']])
            silhouette_avg = silhouette_score(data_normalized, data_clustering['cluster'])

            # Tombol download untuk Hasil Clustering
            st.download_button(
                label="Unduh Hasil Clustering (Excel)",
                data=to_excel(df[['provinsi', 'kabupatenkota', 'plastik', 'logam', 'kaca', 'hasil_cluster']]),
                file_name="hasil_clustering.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # Menambahkan desain kustom untuk tombol
            st.markdown("""
                <style>
                    .stDownloadButton>button {
                        background-color: #dbd28c;
                        color: white;
                        font-size: 16px;
                        font-weight: bold;
                        border: none;
                        padding: 10px 20px;
                        border-radius: 5px;
                        cursor: pointer;
                        transition: background-color 0.3s ease;
                    }

                    .stDownloadButton>button:hover {
                        background-color: #4F4D793;
                    }

                    .stDownloadButton>button:focus {
                        outline: none;
                    }
                </style>
            """, unsafe_allow_html=True)

            st.write(f"Rata-rata Silhouette Score untuk seluruh dataset: {silhouette_avg}")
            # Visualisasi dengan deskripsi cluster, centroid, dan label cluster
            st.subheader("Visualisasi Clustering dengan Deskripsi Cluster")
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(data_clustering['plastik'], data_clustering['logam'], 
                                  c=data_clustering['cluster'], cmap='viridis', label='Cluster')
            centroids = kmeans.cluster_centers_
            plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroid')

            # Menambahkan deskripsi warna sesuai cluster
            cluster_labels = ['Rendah', 'Sedang', 'Tinggi']
            colors = plt.cm.viridis(np.linspace(0, 1, 3))
            for i, color in enumerate(colors):
                plt.scatter([], [], c=[color], label=f'Cluster {i}: {cluster_labels[i]}')

            plt.xlabel('Plastik')
            plt.ylabel('Logam')
            plt.title('Clustering: Plastik vs Logam')
            plt.colorbar(scatter, label='Cluster')
            plt.legend()
            st.pyplot(plt)

        with tab2:
            # Model Decision Tree
            X = df[['plastik', 'logam', 'kaca']]
            y = df['hasil_cluster']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            clf = DecisionTreeClassifier(random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            st.subheader("Evaluasi Model Klasifikasi (Decision Tree)")
            st.write("Akurasi Model Decision Tree:", accuracy_score(y_test, y_pred))
            # st.write("Laporan Klasifikasi:")
            # st.text(classification_report(y_test, y_pred))
            # Menambahkan hasil prediksi ke DataFrame
            df['Hasil Cluster Prediksi'] = clf.predict(X)
            # Menampilkan tabel dengan provinsi dan kabupaten/kota
            st.write("Hasil klasifikasi dengan kolom Provinsi dan Kabupaten/Kota:")
            hasil_klasifikasi = df[['provinsi', 'kabupatenkota', 'plastik', 'logam', 'kaca', 'hasil_cluster', 'prediksi_nilai_ekonomis']]
            st.write(hasil_klasifikasi)

            # Tombol download untuk Hasil Klasifikasi
            st.download_button(
                label="Unduh Hasil Klasifikasi (Excel)",
                data=to_excel(hasil_klasifikasi),
                file_name="hasil_klasifikasi.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # Menambahkan desain kustom untuk tombol
            st.markdown("""
                <style>
                    .stDownloadButton>button {
                        background-color: #dbd28c;
                        color: white;
                        font-size: 16px;
                        font-weight: bold;
                        border: none;
                        padding: 10px 20px;
                        border-radius: 5px;
                        cursor: pointer;
                        transition: background-color 0.3s ease;
                    }

                    .stDownloadButton>button:hover {
                        background-color: #4F4D793;
                    }

                    .stDownloadButton>button:focus {
                        outline: none;
                    }
                </style>
            """, unsafe_allow_html=True)

            
            st.subheader("Visualisasi Klasifikasi")
            fig2, ax2 = plt.subplots(figsize=(12, 8))
            plot_tree(clf, filled=True, feature_names=['plastik', 'logam', 'kaca'], class_names=['Cluster 0', 'Cluster 1', 'Cluster 2'], rounded=True)
            st.pyplot(fig2)

            conf_matrix = confusion_matrix(y_test, y_pred)
            fig3, ax3 = plt.subplots(figsize=(6, 5))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Rendah', 'Sedang', 'Tinggi'], yticklabels=['Rendah', 'Sedang', 'Tinggi'])
            ax3.set_xlabel('Prediksi')
            ax3.set_ylabel('Aktual')
            st.pyplot(fig3)

        with tab3:
            st.markdown("### Evaluasi Model :bar_chart:")
            st.markdown("### **Evaluasi Clustering**")
            
            # Menambahkan visualisasi dengan penekanan pada nilai evaluasi
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**Rata-rata Silhouette Score:**")
                st.markdown(f"<h3 style='color: #E74C3C;'>{silhouette_avg:.2f}</h3>", unsafe_allow_html=True)
            
            # Memberikan keterangan tentang hasil clustering
            # st.markdown("""
            # **Silhouette Score** mengukur seberapa baik objek dalam sebuah cluster berada di dekat objek lainnya dalam cluster yang sama dibandingkan dengan objek di cluster yang berbeda. 
            # Semakin tinggi nilai Silhouette Score, semakin baik hasil clustering yang dihasilkan.
            # """)
            
            st.markdown("### **Evaluasi Klasifikasi (Decision Tree)**")
            # Menampilkan evaluasi klasifikasi dengan format yang lebih bersih dan rapi
            st.write(f"**Akurasi Model Decision Tree:**")
            st.markdown(f"<h2 style='color: #3498DB;'>{accuracy_score(y_test, y_pred):.2f}</h2>", unsafe_allow_html=True)
            # Menampilkan laporan klasifikasi
            st.write("**Laporan Klasifikasi:**")
            classification_report_str = classification_report(y_test, y_pred)
            st.text(classification_report_str)

            # Menambahkan visualisasi confusion matrix dengan heatmap yang lebih menarik
            # st.write("### **Confusion Matrix**")
            # conf_matrix = confusion_matrix(y_test, y_pred)
            # fig3, ax3 = plt.subplots(figsize=(6, 5))
            # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Cluster 0', 'Cluster 1', 'Cluster 2'], yticklabels=['Cluster 0', 'Cluster 1', 'Cluster 2'], cbar=False)
            # ax3.set_xlabel('Prediksi')
            # ax3.set_ylabel('Aktual')
            # ax3.set_title('Confusion Matrix - Model Decision Tree', fontsize=14)
            # st.pyplot(fig3)

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
    finally:
        conn.close()
