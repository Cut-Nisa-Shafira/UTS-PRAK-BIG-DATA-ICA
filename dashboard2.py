import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# ==========================
# KONFIGURASI DASHBOARD
# ==========================
st.set_page_config(
    page_title="🧠 Image Classification & Object Detection App",
    page_icon="🧠",
    layout="wide"
)

# ==========================
# HEADER NAVIGATION
# ==========================
st.markdown("""
    <style>
    .header {
        background-color: #4CAF50;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 24px;
        font-weight: bold;
    }
    .nav-button {
        background-color: #f1f1f1;
        border: none;
        color: black;
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
        background-color: #ddd;
    }
    .content {
        padding: 20px;
        background-color: #f9f9f9;
        border-radius: 10px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">🌟 Selamat Datang di Image Classification & Detection App 🌟</div>', unsafe_allow_html=True)

# Navigation buttons in header
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🔍 Deteksi Objek (YOLO)", key="yolo"):
        st.session_state.page = "Deteksi Objek (YOLO)"
with col2:
    if st.button("🧠 Klasifikasi Gambar", key="classify"):
        st.session_state.page = "Klasifikasi Gambar"
with col3:
    if st.button("📖 Tentang", key="about"):
        st.session_state.page = "Tentang"

# Default page
if "page" not in st.session_state:
    st.session_state.page = "Deteksi Objek (YOLO)"

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    """Memuat model YOLO dan Keras dengan aman dan efisien."""
    yolo_model = None
    keras_model = None

    # --- Load YOLO ---
    try:
        yolo_path = "model/ica_Laporan4.pt"
        if os.path.exists(yolo_path):
            yolo_model = YOLO(yolo_path)
            st.success("✅ Model YOLO berhasil dimuat.")
        else:
            st.warning("⚠️ File model YOLO tidak ditemukan di folder /model/")
    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLO: {e}")

    # --- Load Keras ---
    try:
        keras_path = "model/ica_laporan2.h5"
        if os.path.exists(keras_path):
            keras_model = tf.keras.models.load_model(
                keras_path,
                compile=False,
                safe_mode=False  # ✅ untuk kompatibilitas TF >= 2.16
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
        st.error(f"❌ Gagal memuat model Keras: {e}")

    return yolo_model, keras_model

# Memanggil fungsi load_models()
yolo_model, keras_model = load_models()

# ==========================
# PAGE CONTENT
# ==========================
st.markdown('<div class="content">', unsafe_allow_html=True)

if st.session_state.page == "Tentang":
    st.title("📖 Tentang Aplikasi")
    st.write("""
        Selamat datang di **Image Classification & Detection App**! 🎉
        
        Aplikasi ini dirancang untuk membantu Anda mengklasifikasikan gambar tanaman dan mendeteksi objek menggunakan teknologi AI canggih.
        
        **Fitur Utama:**
        - 🔍 **Deteksi Objek (YOLO)**: Deteksi objek dalam gambar secara real-time menggunakan model YOLO, diikuti dengan klasifikasi gambar jika diinginkan.
        - 🧠 **Klasifikasi Gambar**: Unggah gambar tanaman dan dapatkan prediksi jenisnya (misalnya: jagung, jute, padi, gandum, tebu).
        
        **Teknologi yang Digunakan:**
        - YOLO (Ultralytics) untuk deteksi objek.
        - TensorFlow/Keras untuk klasifikasi.
        - Streamlit untuk antarmuka web yang interaktif.
        
        Dibuat dengan ❤️ oleh tim AI Enthusiast. Jika ada pertanyaan, hubungi kami!
    """)
    st.image("https://via.placeholder.com/800x400?text=AI+Powered+App", caption="Ilustrasi Aplikasi AI", use_column_width=True)

elif st.session_state.page == "Deteksi Objek (YOLO)":
    st.title("🔍 Deteksi Objek (YOLO)")
    st.write("Unggah gambar untuk deteksi objek secara real-time menggunakan YOLO, diikuti dengan klasifikasi gambar jika diinginkan!")
    
    uploaded_file = st.file_uploader("📤 Unggah gambar", type=["jpg", "jpeg", "png"], key="yolo_uploader")
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="📸 Gambar yang diunggah", use_column_width=True)
        img_np = np.array(img)

        # --- Deteksi Objek dengan YOLO ---
        if yolo_model:
            with st.spinner("🔍 Mendeteksi objek..."):
                try:
                    results = yolo_model(img_np)
                    result_img = results[0].plot()
                    st.image(result_img, caption="📦 Hasil Deteksi YOLO", use_column_width=True)

                    st.subheader("📋 Daftar Deteksi:")
                    detections = []
                    for box in results[0].boxes:
                        cls_id = int(box.cls)
                        label = yolo_model.names.get(cls_id, "Unknown")
                        conf = float(box.conf)
                        detections.append((label, conf))
                        st.write(f"- **{label}** ({conf:.2f})")
                    
                    if detections:
                        st.success(f"✅ Ditemukan {len(detections)} objek.")
                    else:
                        st.info("ℹ️ Tidak ada objek terdeteksi.")
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan saat deteksi YOLO: {e}")
        else:
            st.warning("⚠️ Model YOLO belum berhasil dimuat.")

        # --- Lanjutkan dengan Klasifikasi Gambar ---
        st.markdown("---")
        st.subheader("🧠 Klasifikasi Gambar (Opsional)")
        st.write("Jika ingin, lanjutkan dengan klasifikasi gambar menggunakan model Keras.")
        
        if st.button("🔄 Lakukan Klasifikasi", key="classify_after_yolo"):
            if keras_model:
                with st.spinner("🧠 Mengklasifikasikan gambar..."):
                    try:
                        # Sesuaikan ukuran input model
                        input_shape = keras_model.input_shape[1:3]
                        img_resized = img.resize(input_shape)

                        # Preprocessing
                        x = image.img_to_array(img_resized)
                        x = np.expand_dims(x, axis=0) / 255.0

                        preds = keras_model.predict(x)
                        pred_class = np.argmax(preds, axis=1)[0]
                        class_names = ["maize", "jute", "rice", "wheat", "sugarcane"]

                        st.subheader("📊 Hasil Klasifikasi:")
                        st.write(f"🌾 **Prediksi:** {class_names[pred_class]}")
                        st.write(f"📈 **Probabilitas:** {np.max(preds) * 100:.2f}%")
                        
                        # Tambahan atraktif: Progress bar untuk probabilitas
                        st.progress(np.max(preds))
                        
                        # Tambahan: Emoji berdasarkan prediksi
                        emoji_map = {"maize": "🌽", "jute": "🌿", "rice": "🌾", "wheat": "🌾", "sugarcane": "🍯"}
                        st.write(f"{emoji_map[class_names[pred_class]]} Wow, ini terlihat seperti {class_names[pred_class]}!")
                        
                    except Exception as e:
                        st.error(f"❌ Terjadi kesalahan saat klasifikasi: {e}")
            else:
                st.warning("⚠️ Model Keras belum berhasil dimuat.")

elif st.session_state.page == "Klasifikasi Gambar":
    st.title("🧠 Klasifikasi Gambar")
    st.write("Unggah gambar tanaman untuk diklasifikasikan. Model AI kami akan memprediksi jenis tanaman dengan akurasi tinggi! 🌾")
    
    uploaded_file = st.file_uploader("📤 Unggah gambar", type=["jpg", "jpeg", "png"], key="classify_uploader")

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="📸 Gambar yang diunggah", use_column_width=True)

        if keras_model:
            with st.spinner("🧠 Mengklasifikasikan gambar..."):
                try:
                    # Sesuaikan ukuran input model
                    input_shape = keras_model.input_shape[1:3]
                    img_resized = img.resize(input_shape)

                    # Preprocessing
                    x = image.img_to_array(img_resized)
                    x = np.expand_dims(x, axis=0) / 255.0

                    preds = keras_model.predict(x)
                    pred_class = np.argmax(preds, axis=1)[0]
                    class_names = ["maize", "jute", "rice", "wheat", "sugarcane"]

                    st.subheader("📊 Hasil Klasifikasi:")
                    st.write(f"🌾 **Prediksi:** {class_names[pred_class]}")
                    st.write(f"📈 **Probabilitas:** {np.max(preds) * 100:.2f}%")
                    
                    # Tambahan atraktif: Progress bar untuk probabilitas
                    st.progress(np.max(preds))
                    
                    # Tambahan: Emoji berdasarkan prediksi
                    emoji_map = {"maize": "🌽", "jute": "🌿", "rice": "🌾", "wheat": "🌾", "sugarcane": "🍯"}
                    st.write(f"{emoji_map[class_names[pred_class]]} Wow, ini terlihat seperti {class_names[pred_class]}!")
                    
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan saat klasifikasi: {e}")
        else:
            st.warning("⚠️ Model Keras belum berhasil dimuat.")

st.markdown('</div>', unsafe_allow_html=True)

# Footer atraktif
st.markdown("""
    <hr>
    <div style="text-align: center; color: #666;">
        Dibuat dengan ❤️ menggunakan Streamlit. © 2023 AI App.
    </div>
""", unsafe_allow_html=True)
