import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import os
import pickle

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
    # --- Load YOLO model ---
    yolo_model = None
    try:
        if os.path.exists("model/ica_Laporan4.pt"):
            yolo_model = YOLO("model/ica_Laporan4.pt")
        else:
            st.warning("⚠️ File model YOLO tidak ditemukan di folder /model/")
    except (EOFError, pickle.UnpicklingError):
        st.error("❌ File YOLO .pt rusak atau tidak lengkap. Harap unggah ulang model.")
    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLO: {e}")

    # --- Load Keras model ---
    keras_model = None
    try:
        keras_model = tf.keras.models.load_model("model/ica_laporan2.keras", compile=False)
        keras_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    except Exception as e:
        st.error(f"❌ Gagal memuat model Keras: {e}")

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
    img_np = np.array(img)

    # ==========================
    # MODE YOLO
    # ==========================
    if mode == "Deteksi Objek (YOLO)":
        if yolo_model:
            with st.spinner("🔍 Mendeteksi objek..."):
                try:
                    results = yolo_model(img_np)
                    result_img = results[0].plot()
                    st.image(result_img, caption="Hasil Deteksi YOLO", use_column_width=True)

                    st.subheader("📦 Hasil Deteksi:")
                    for box in results[0].boxes:
                        cls_id = int(box.cls)
                        label = yolo_model.names[cls_id]
                        conf = float(box.conf)
                        st.write(f"- **{label}** ({conf:.2f})")
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan saat deteksi YOLO: {e}")
        else:
            st.warning("⚠️ Model YOLO belum berhasil dimuat.")

    # ==========================
    # MODE KERAS
    # ==========================
    elif mode == "Klasifikasi Gambar":
        if keras_model:
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
                    st.write(f"Prediksi: **{class_names[pred_class]}**")
                    st.write(f"Probabilitas: {np.max(preds)*100:.2f}%")
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan saat klasifikasi: {e}")
        else:
            st.warning("⚠️ Model Keras belum dimuat.")
