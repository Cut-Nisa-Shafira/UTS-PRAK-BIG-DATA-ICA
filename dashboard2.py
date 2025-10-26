import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
import pandas as pd
import time
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import io

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(
    page_title="Dashboard Klasifikasi & Deteksi Tanaman",
    page_icon="🌿",
    layout="wide"
)

# ==========================
# CSS KUSTOM
# ==========================
def load_css():
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(135deg, #1a0028, #33005a);
            color: #ffffff;
        }
        .stButton>button {
            background-color: #7b2cbf;
            color: white;
            border-radius: 12px;
            padding: 8px 16px;
        }
        .stButton>button:hover {
            background-color: #9d4edd;
        }
        .block-container {
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
        }
        footer {
            text-align: center;
            font-size: small;
            color: #aaa;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


load_css()

# ==========================
# MUAT MODEL DENGAN CACHE
# ==========================
@st.cache_resource
def load_models():
    yolo_model, keras_model = None, None

    try:
        yolo_model = YOLO("model/ica_Laporan4.pt")
        st.success("✅ Model YOLO berhasil dimuat.")
    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLO: {e}")

    try:
        keras_model = tf.keras.models.load_model("model/ica_laporan2.h5")
        st.success("✅ Model Keras berhasil dimuat.")
    except Exception as e:
        st.error(f"❌ Gagal memuat model Keras: {e}")

    return yolo_model, keras_model


yolo_model, keras_model = load_models()

# ==========================
# INISIALISASI SESSION STATE
# ==========================
if "page" not in st.session_state:
    st.session_state.page = "Deteksi Objek (YOLO)"

if "history" not in st.session_state or not isinstance(st.session_state.history, list):
    st.session_state.history = []

# ==========================
# NAVIGASI
# ==========================
st.sidebar.title("🌿 Navigasi Utama")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["Deteksi Objek (YOLO)", "Klasifikasi Gambar", "Histori", "Tentang"]
)
st.session_state.page = menu

# ==========================
# HALAMAN 1 - YOLO DETEKSI
# ==========================
if st.session_state.page == "Deteksi Objek (YOLO)":
    st.title("🧠 Deteksi Objek Menggunakan YOLO")

    uploaded_file = st.file_uploader("Unggah gambar...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Gambar yang diunggah", use_container_width=True)

        if st.button("🔍 Jalankan Deteksi"):
            if yolo_model is None:
                st.warning("⚠️ Model YOLO belum tersedia.")
            else:
                with st.spinner("Sedang mendeteksi objek..."):
                    results = yolo_model.predict(image, imgsz=640)
                    boxes = results[0].boxes
                    time.sleep(1)

                st.success("✅ Deteksi selesai!")

                # Tampilkan hasil deteksi
                result_img = results[0].plot()
                st.image(result_img, caption="📦 Hasil Deteksi", use_container_width=True)

                labels = [yolo_model.names[int(cls)] for cls in boxes.cls]
                st.write("**Objek terdeteksi:**", ", ".join(labels))

                # Simpan ke histori
                st.session_state.history.append({
                    "Mode": "YOLO",
                    "Hasil": ", ".join(labels),
                    "File": uploaded_file.name
                })

# ==========================
# HALAMAN 2 - KERAS KLASIFIKASI
# ==========================
elif st.session_state.page == "Klasifikasi Gambar":
    st.title("🌾 Klasifikasi Tanaman Menggunakan CNN")

    uploaded_file = st.file_uploader("Unggah gambar tanaman...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).resize((224, 224))
        st.image(image, caption="📷 Gambar yang diunggah", use_container_width=True)

        if st.button("🔍 Lakukan Klasifikasi"):
            if keras_model is None:
                st.warning("⚠️ Model Keras belum tersedia.")
            else:
                with st.spinner("Sedang melakukan klasifikasi..."):
                    time.sleep(1)
                    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
                    prediction = keras_model.predict(img_array)
                    pred_class = np.argmax(prediction, axis=1)[0]

                class_names = ["jute", "maize", "rice", "sugarcane", "wheat"]
                hasil = class_names[pred_class]

                emoji_map = {
                    "maize": "🌽",
                    "jute": "🌿",
                    "rice": "🌾",
                    "wheat": "🌾",
                    "sugarcane": "🍯"
                }

                st.subheader("📊 Hasil Klasifikasi:")
                st.success(f"{emoji_map[hasil]} Prediksi: **{hasil}**")

                st.session_state.history.append({
                    "Mode": "Keras",
                    "Hasil": hasil,
                    "File": uploaded_file.name
                })

# ==========================
# HALAMAN 3 - HISTORI
# ==========================
elif st.session_state.page == "Histori":
    st.title("📖 Histori Prediksi")
    st.write("Riwayat prediksi Anda, termasuk gambar sebelumnya dan hasil deteksi/klasifikasi.")

    if isinstance(st.session_state.history, list) and len(st.session_state.history) > 0:
        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history, use_container_width=True)

        # Pie Chart distribusi hasil
        fig = px.pie(df_history, names="Hasil", title="Distribusi Prediksi")
        st.plotly_chart(fig, use_container_width=True)

        # Tombol hapus
        if st.button("🗑️ Hapus Riwayat"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("ℹ️ Belum ada data riwayat untuk ditampilkan.")

# ==========================
# HALAMAN 4 - TENTANG
# ==========================
elif st.session_state.page == "Tentang":
    st.title("👩‍💻 Tentang Aplikasi")
    st.info("""
    Dashboard ini dikembangkan sebagai implementasi model **YOLOv8** untuk deteksi objek 
    dan **CNN Keras** untuk klasifikasi gambar tanaman. 
    Aplikasi ini dibuat untuk keperluan pembelajaran Big Data dan Computer Vision.
    """)
    st.markdown("---")
    st.markdown(
        "<footer>© 2025 Praktikan Big Data ICA — All Rights Reserved 🌱</footer>",
        unsafe_allow_html=True
    )
