import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import os

# ==========================
# CONFIGURASI DASHBOARD
# ==========================
st.set_page_config(page_title="Image Classification & Object Detection App", page_icon="🧠", layout="wide")

st.title("🧠 Image Classification & Object Detection App")

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    try:
        yolo_model = YOLO("model/ica_Laporan4.pt")
    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLO: {e}")
        yolo_model = None

    try:
        keras_model = tf.keras.models.load_model("model/ica_laporan2.keras", compile=False)
        keras_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    except Exception as e:
        st.error(f"❌ Gagal memuat model Keras: {e}")
        keras_model = None

    return yolo_model, keras_model


yolo_model, keras_model = load_models()

# ==========================
# PILIH MODE
# ==========================
mode = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

# ==========================
# UNGGAH GAMBAR
# ==========================
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diunggah", use_column_width=True)

    # Konversi ke numpy array
    img_np = np.array(img)

    # ==========================
    # MODE YOLO
    # ==========================
    if mode == "Deteksi Objek (YOLO)":
        if yolo_model is not None:
            with st.spinner("🔍 Mendeteksi objek..."):
                results = yolo_model(img_np)
                result_img = results[0].plot()  # Visualisasi hasil deteksi

                # Tampilkan hasil
                st.image(result_img, caption="Hasil Deteksi YOLO", use_column_width=True)

                # Tampilkan label dan confidence
                st.subheader("📦 Hasil Deteksi:")
                for box in results[0].boxes:
                    cls_id = int(box.cls)
                    label = yolo_model.names[cls_id]
                    conf = float(box.conf)
                    st.write(f"- **{label}** ({conf:.2f})")
        else:
            st.warning("Model YOLO belum dimuat.")

    # ==========================
    # MODE KERAS
    # ==========================
    elif mode == "Klasifikasi Gambar":
        if keras_model is not None:
            with st.spinner("🧠 Mengklasifikasikan gambar..."):
                try:
                    # Pastikan ukuran input sesuai model
                    input_shape = keras_model.input_shape[1:3]  # contoh: (224, 224)
                    img_resized = img.resize(input_shape)

                    # Preprocessing
                    x = image.img_to_array(img_resized)
                    x = np.expand_dims(x, axis=0)
                    x = x / 255.0

                    preds = keras_model.predict(x)
                    pred_class = np.argmax(preds, axis=1)[0]

                    # Daftar label (ganti sesuai label model kamu)
                    class_names = ["jute", "rice", "maize", "sugarcane", "wheat"]

                    st.subheader("📊 Hasil Klasifikasi:")
                    st.write(f"Prediksi: **{class_names[pred_class]}**")
                    st.write(f"Probabilitas: {np.max(preds)*100:.2f}%")

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat klasifikasi: {e}")
        else:
            st.warning("Model Keras belum dimuat.")
