import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    # Muat model YOLO
    yolo_model = YOLO("model/ica_Laporan4.pt")

    # Muat model klasifikasi Keras
    try:
        classifier = tf.keras.models.load_model(
            "model/ica_laporan2.keras",
            safe_mode=False,      # Matikan mode aman untuk model lama
            compile=False         # Tidak perlu kompilasi ulang di tahap ini
        )
        st.success("✅ Model klasifikasi berhasil dimuat.")
    except Exception as e:
        st.error(f"⚠️ Gagal memuat model klasifikasi: {e}")
        classifier = None

    return yolo_model, classifier


yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification & Object Detection App")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        # Deteksi objek
        results = yolo_model(img)
        result_img = results[0].plot()  # hasil deteksi (gambar dengan box)
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

    elif menu == "Klasifikasi Gambar":
        if classifier is None:
            st.error("❌ Model klasifikasi belum berhasil dimuat. Tidak dapat melakukan prediksi.")
        else:
            # Preprocessing
            img_resized = img.resize((224, 224))  # sesuaikan ukuran dengan model kamu
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Prediksi
            prediction = classifier.predict(img_array)
            class_index = int(np.argmax(prediction))
            confidence = float(np.max(prediction))

            st.write("### 🔍 Hasil Prediksi:")
            st.write(f"**Kelas:** {class_index}")
            st.write(f"**Probabilitas:** {confidence:.4f}")
