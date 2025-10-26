import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# ==========================
# KONFIGURASI DASHBOARD
# ==========================
st.set_page_config(
    page_title="📷 Image Classification & Object Detection App",
    page_icon="📷",
    layout="wide"
)

# ==========================
# HEADER NAVIGATION
# ==========================
st.markdown("""
    <style>
    .header {
        background-color: #008080;  /* Teal Green */
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 24px;
        font-weight: bold;
    }
    .nav-button {
        background-color: #20B2AA;  /* Light Teal Green */
        border: none;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
    }
    .nav-button:hover {
        background-color: #5F9EA0;  /* Darker Teal on hover */
    }
    .content {
        padding: 20px;
        background-color: #F0F8FF;  /* Light Teal-tinted background */
        border-radius: 10px;
        margin-top: 20px;
    }
    .image-label {
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        color: #008080;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="header">🌟 Selamat Datang di Image Classification & Detection App 🌟</div>',
    unsafe_allow_html=True
)

# ==========================
# NAVIGATION BUTTONS
# ==========================
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🔍 Deteksi Objek (YOLO)", key="yolo"):
        st.session_state.page = "Deteksi Objek (YOLO)"
with col2:
    if st.button("🧠 Klasifikasi Gambar", key="classify"):
        st.session_state.page = "Klasifikasi Gambar"
with col3:
    if st.button("📖 Tentang", key="about"):
        st.session_state.page = "Tentang"
with col4:
    if st.button("📊 Histori", key="history"):
        st.session_state.page = "Histori"

if "page" not in st.session_state:
    st.session_state.page = "Deteksi Objek (YOLO)"
if "history" not in st.session_state:
    st.session_state.history = []

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    """Memuat model YOLO dan Keras dengan aman dan efisien."""
    yolo_model = None
    keras_model = None

    try:
        yolo_path = "model/ica_Laporan4.pt"
        if os.path.exists(yolo_path):
            if os.path.getsize(yolo_path) == 0:
                st.error("❌ File model YOLO kosong atau rusak.")
            else:
                yolo_model = YOLO(yolo_path)
                st.success("✅ Model YOLO berhasil dimuat.")
        else:
            st.warning("⚠️ File model YOLO tidak ditemukan di folder /model/")
    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLO: {e}")

    try:
        keras_path = "model/ica_laporan2.h5"
        if os.path.exists(keras_path):
            # coba load, jika gagal akan ter-catch
            keras_model = tf.keras.models.load_model(
                keras_path,
                compile=False,
                safe_mode=False
            )
            keras_model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )
            st.success("✅ Model Keras berhasil dimuat.")
        else:
            st.warning("⚠️ File model Keras tidak ditemukan di folder /model/")
    except Exception as e:
        # tangkap error load model dan berikan pesan yang ramah
        st.error(f"❌ Gagal memuat model Keras: {e}")

    return yolo_model, keras_model


yolo_model, keras_model = load_models()

# ==========================
# PAGE CONTENT
# ==========================
st.markdown('<div class="content">', unsafe_allow_html=True)

# --------------------------
# TENTANG APLIKASI
# --------------------------
if st.session_state.page == "Tentang":
    st.title("📖 Tentang Aplikasi")

    st.subheader("👨‍💻 Biodata Developer")
    col_left, col_right = st.columns([1, 1])
    with col_left:
        google_drive_id = "1f_6kkQdVlo013ZR4c5KERL17PtzXv6nh"
        google_drive_url = f"https://drive.google.com/thumbnail?id={google_drive_id}&sz=w200"
        try:
            st.image(google_drive_url, caption="Foto Developer", width=200)
        except Exception:
            photo_path = "assets/Tezza_2024_10_20_190012490.jpg"
            if os.path.exists(photo_path):
                st.image(photo_path, caption="Foto Developer", width=200)
            else:
                st.image("https://via.placeholder.com/200x250?text=Developer+Photo",
                         caption="Foto Developer (Placeholder)", width=200)
    with col_right:
        st.write("""
        **Nama:** Cut Nisa Shafira  
        **Jurusan:** S1 Statistika, Universitas Syiah Kuala  
        **Angkatan:** 2022  
        **Praktikum:** Pemrograman Big Data  
        **Asisten Lab:** Diaz Darsya Rizqullah | Musliadi  
        **Kontak:** cutnisa386@gmail.com | LinkedIn: Cut Nisa
        """)

    st.markdown("---")
    st.subheader("ℹ️ Informasi Tentang Aplikasi")
    st.write("""
        Aplikasi ini membantu mengklasifikasikan gambar tanaman dan mendeteksi objek menggunakan AI.

        **Fitur Utama:**
        - 🔍 Deteksi Objek (YOLO)
        - 🧠 Klasifikasi Gambar (Keras)
        - 📊 Riwayat Prediksi dan Analisis

        **Teknologi:** YOLO, TensorFlow/Keras, dan Streamlit.
    """)
    st.image("https://via.placeholder.com/800x400?text=AI+Powered+App",
             caption="Ilustrasi Aplikasi AI", use_container_width=True)

# --------------------------
# DETEKSI OBJEK (YOLO)
# --------------------------
elif st.session_state.page == "Deteksi Objek (YOLO)":
    st.title("🔍 Deteksi Objek (YOLO)")
    st.write("Unggah gambar untuk deteksi objek secara real-time menggunakan YOLO.")

    if yolo_model is None:
        st.error("❌ Model YOLO tidak tersedia.")
    else:
        uploaded_file = st.file_uploader("📤 Unggah gambar", type=["jpg", "jpeg", "png"], key="yolo_uploader")
        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert("RGB")
            img_np = np.array(img)

            with st.spinner("🔍 Mendeteksi objek..."):
                try:
                    results = yolo_model(img_np)
                    result_img = results[0].plot()

                    col_before, col_after = st.columns(2)
                    with col_before:
                        st.image(img, caption="📸 Gambar Asli", use_container_width=True)
                    with col_after:
                        st.image(result_img, caption="📦 Hasil Deteksi YOLO", use_container_width=True)

                    detections = []
                    for box in results[0].boxes:
                        cls_id = int(box.cls)
                        label = yolo_model.names.get(cls_id, "Unknown")
                        conf = float(box.conf)
                        detections.append((label, conf))
                        st.write(f"- **{label}** ({conf:.2f})")

                    if detections:
                        st.success(f"✅ Ditemukan {len(detections)} objek.")
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        # simpan hanya tipe dasar (hindari objek kompleks yang tidak bisa serialisasi)
                        st.session_state.history.append({
                            "timestamp": timestamp,
                            "type": "Deteksi Objek",
                            "result": f"{len(detections)} objek terdeteksi",
                            "details": ", ".join([f"{label} ({conf:.2f})" for label, conf in detections]),
                            # menyimpan gambar PIL untuk tampilan saja (session_state dapat menahan objek kecil)
                            "image": img
                        })
                    else:
                        st.info("ℹ️ Tidak ada objek terdeteksi.")
                except Exception as e:
                    st.error(f"❌ Kesalahan deteksi: {e}")

# --------------------------
# KLASIFIKASI GAMBAR
# --------------------------
elif st.session_state.page == "Klasifikasi Gambar":
    st.title("🧠 Klasifikasi Gambar")
    st.write("Unggah gambar tanaman untuk diklasifikasikan.")

    uploaded_file = st.file_uploader("📤 Unggah gambar", type=["jpg", "jpeg", "png"], key="classify_uploader")
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="📸 Gambar yang diunggah", use_container_width=True)

        if keras_model is None:
            st.warning("⚠️ Model Keras belum tersedia. Fitur klasifikasi tidak bisa digunakan.")
        else:
            with st.spinner("🧠 Mengklasifikasikan gambar..."):
                try:
                    input_shape = keras_model.input_shape[1:3]
                    img_resized = img.resize(input_shape)
                    x = image.img_to_array(img_resized)
                    x = np.expand_dims(x, axis=0) / 255.0

                    preds = keras_model.predict(x)
                    pred_class = np.argmax(preds, axis=1)[0]
                    class_names = ["maize", "jute", "rice", "wheat", "sugarcane"]

                    st.subheader("📊 Hasil Klasifikasi:")
                    st.write(f"🌾 **Prediksi:** {class_names[pred_class]}")
                    st.write(f"📈 **Probabilitas:** {np.max(preds) * 100:.2f}%")

                    max_prob = float(np.max(preds))
                    st.progress(max_prob)

                    emoji_map = {"maize": "🌽", "jute": "🌿", "rice": "🌾", "wheat": "🌾", "sugarcane": "🍯"}
                    st.write(f"{emoji_map[class_names[pred_class]]} Wow, ini terlihat seperti {class_names[pred_class]}!")

                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.history.append({
                        "timestamp": timestamp,
                        "type": "Klasifikasi Gambar",
                        "result": class_names[pred_class],
                        "details": f"Probabilitas: {max_prob * 100:.2f}%",
                        "image": img
                    })

                    st.subheader("📊 Probabilitas Semua Kelas:")
                    for i, prob in enumerate(preds[0]):
                        st.write(f"- {class_names[i]}: {prob * 100:.2f}%")

                    if max_prob < 0.5:
                        st.warning("⚠️ Probabilitas rendah. Model mungkin kurang yakin.")

                except Exception as e:
                    st.error(f"❌ Kesalahan klasifikasi: {e}")

# --------------------------
# HISTORI
# --------------------------
elif st.session_state.page == "Histori":
    st.title("📊 Histori Prediksi")
    st.write("Riwayat prediksi Anda, termasuk gambar sebelumnya dan visualisasi distribusi hasil.")

    # Validasi history sebelum membuat DataFrame
    history = st.session_state.get("history", [])
    if not isinstance(history, list) or len(history) == 0:
        st.info("ℹ️ Belum ada riwayat prediksi. Lakukan deteksi objek atau klasifikasi gambar terlebih dahulu!")
    else:
        # Pastikan setiap entri adalah dict dan berisi keys yang diharapkan
        safe_entries = []
        for entry in history:
            if isinstance(entry, dict):
                # Extract minimal fields and provide defaults bila perlu
                safe_entries.append({
                    "timestamp": entry.get("timestamp", ""),
                    "type": entry.get("type", ""),
                    "result": entry.get("result", ""),
                    "details": entry.get("details", "")
                })

        if len(safe_entries) == 0:
            st.info("ℹ️ Riwayat tidak berisi entri yang dapat ditampilkan.")
        else:
            st.subheader("📋 Tabel Riwayat Prediksi")
            df_history = pd.DataFrame(safe_entries)
            st.dataframe(df_history, use_container_width=True)

            # Tombol clear history
            if st.button("🗑️ Clear History", key="clear_history"):
                st.session_state.history = []
                st.success("✅ Histori telah dibersihkan!")
                st.experimental_rerun()

            # Tampilkan gambar dari histori terbaru jika ada
            st.subheader("🖼️ Gambar dari Prediksi Terbaru")
            # Cari entri terakhir yang juga punya 'image'
            latest_with_image = None
            for e in reversed(history):
                if isinstance(e, dict) and e.get("image") is not None:
                    latest_with_image = e
                    break
            if latest_with_image:
                try:
                    st.image(latest_with_image["image"],
                             caption=f"Gambar dari {latest_with_image.get('type','')} - {latest_with_image.get('timestamp','')}",
                             use_container_width=True)
                except Exception:
                    st.info("⚠️ Gagal menampilkan gambar dari histori (format tidak dikenali).")

            # Visualisasi grafik distribusi jenis prediksi (khusus klasifikasi)
            st.subheader("📈 Distribusi Jenis Prediksi")
            results = [entry.get("result") for entry in history if isinstance(entry, dict) and entry.get("type") == "Klasifikasi Gambar"]
            if results:
                result_counts = pd.Series(results).value_counts()
                fig, ax = plt.subplots()
                result_counts.plot(kind="bar", ax=ax, color="#008080")
                ax.set_title("Distribusi Prediksi Klasifikasi Gambar")
                ax.set_xlabel("Jenis Tanaman")
                ax.set_ylabel("Jumlah Prediksi")
                st.pyplot(fig)
            else:
                st.info("ℹ️ Belum ada prediksi klasifikasi gambar untuk divisualisasikan.")

st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# FOOTER
# ==========================
st.markdown("""
    <hr>
    <div style="text-align: center; color: #008080; font-weight: bold;">
        Dibuat dengan ❤️ oleh Cut Nisa Shafira. © 2025 AI App.
    </div>
""", unsafe_allow_html=True)
