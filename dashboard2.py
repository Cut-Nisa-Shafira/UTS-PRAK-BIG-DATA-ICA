import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# ==========================
# KONFIGURASI DASHBOARD
# ==========================
st.set_page_config(
    page_title="🧠 Image Classification App",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Image Classification App")

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_keras_model():
    """Memuat model Keras (.h5) dengan aman."""
    keras_path = "model/model_compressed.h5"
    if not os.path.exists(keras_path):
        st.error("❌ File model.h5 tidak ditemukan di folder /model/")
        return None

    try:
        model = tf.keras.models.load_model(keras_path, compile=False)
        model.compile(optimizer="adam",
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        st.success("✅ Model Keras (.h5) berhasil dimuat.")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None


model = load_keras_model()

# ==========================
# UPLOAD GAMBAR
# ==========================
uploaded_file = st.file_uploader("📤 Unggah gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📸 Gambar yang diunggah", use_column_width=True)

    if model:
        with st.spinner("🧠 Mengklasifikasikan gambar..."):
            try:
                # Preprocessing
                input_shape = model.input_shape[1:3]
                img_resized = img.resize(input_shape)

                x = image.img_to_array(img_resized)
                x = np.expand_dims(x, axis=0) / 255.0

                preds = model.predict(x)
                pred_class = np.argmax(preds, axis=1)[0]

                # Ganti sesuai kelas kamu
                class_names = ["maize", "jute", "rice", "wheat", "sugarcane"]

                st.subheader("📊 Hasil Klasifikasi:")
                st.write(f"🌾 **Prediksi:** {class_names[pred_class]}")
                st.write(f"📈 **Probabilitas:** {np.max(preds) * 100:.2f}%")
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat klasifikasi: {e}")
