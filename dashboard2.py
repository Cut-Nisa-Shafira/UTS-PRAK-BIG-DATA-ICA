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
    page_title="📷 Image Classification & Object Detection App",
    page_icon="📷",
    layout="wide"
)

# ==========================
# HEADER NAVIGATION
# ==========================
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #E8F5E8, #F1F8E9);  /* Light green gradient background */
        font-family: 'Arial', sans-serif;
    }
    .header {
        background: linear-gradient(135deg, #4CAF50, #81C784);  /* Soft green gradient */
        padding: 15px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 26px;
        font-weight: bold;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
    }
    .header::before {
        content: '🌿🐾';
        position: absolute;
        top: 10px;
        left: 20px;
        font-size: 30px;
        opacity: 0.7;
    }
    .header::after {
        content: '🌸🦋';
        position: absolute;
        top: 10px;
        right: 20px;
        font-size: 30px;
        opacity: 0.7;
    }
    .nav-button {
        background-color: #66BB6A;  /* Light green */
        border: none;
        color: white;
        padding: 12px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .nav-button:hover {
        background-color: #43A047;  /* Darker green on hover */
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    .content {
        padding: 25px;
        background: rgba(255, 255, 255, 0.9);  /* Semi-transparent white */
        border-radius: 15px;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        position: relative;
    }
    .content::before {
        content: '🌳';
        position: absolute;
        top: 10px;
        left: 10px;
        font-size: 40px;
        opacity: 0.3;
        z-index: -1;
    }
    .content::after {
        content: '🐦';
        position: absolute;
        bottom: 10px;
        right: 10px;
        font-size: 40px;
        opacity: 0.3;
        z-index: -1;
    }
    .image-label {
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        color: #2E7D32;  /* Dark green */
        margin-bottom: 10px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .animal-plant-bg {
        background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text fill="%234CAF50" font-size="20" y="50%">🌿🐾</text></svg>');
        background-repeat: no-repeat;
        background-position: center;
        background-size: 200px 200px;
        opacity: 0.1;
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
    }
    .footer {
        text-align: center;
        color: #2E7D32;
        font-weight: bold;
        font-size: 16px;
        margin-top: 30px;
        padding: 15px;
        background: linear-gradient(135deg, #C8E6C9, #A5D6A7);
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        position: relative;
    }
    .footer::before {
        content: '🌍';
        position: absolute;
        left: 20px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 24px;
    }
    .footer::after {
        content: '🌱';
        position: absolute;
        right: 20px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 24px;
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
            # Tambahan: Periksa ukuran file untuk mendeteksi korupsi
            if os.path.getsize(yolo_path) == 0:
                st.error("❌ File model YOLO kosong atau rusak. Periksa file Anda.")
            else:
                yolo_model = YOLO(yolo_path)
                st.success("✅ Model YOLO berhasil dimuat.")
        else:
            st.warning("⚠️ File model YOLO tidak ditemukan di folder /model/")
    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLO: {e}")
        st.info("💡 Tips: Pastikan file .pt valid dan tidak rusak. Coba unduh ulang model atau periksa versi Ultralytics (pip install ultralytics --upgrade).")

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
st.markdown('<div class="content"><div class="animal-plant-bg"></div>', unsafe_allow_html=True)

if st.session_state.page == "Tentang":
    st.title("📖 Tentang Aplikasi")
    
    # --- Biodata Developer ---
    st.subheader("👨‍💻 Biodata Developer")
    col_left, col_right = st.columns([1, 1], gap="small")  # Kolom sama ukuran, gap kecil untuk jarak dekat
    
    with col_left:
        # Coba load dari Google Drive terlebih dahulu (link: https://drive.google.com/file/d/1f_6kkQdVlo013ZR4c5KERL17PtzXv6nh/view?usp=sharing)
        # Pastikan file dibagikan secara publik (set ke "Anyone with the link can view")
        google_drive_id = "1f_6kkQdVlo013ZR4c5KERL17PtzXv6nh"
        google_drive_url = f"https://drive.google.com/thumbnail?id={google_drive_id}&sz=w200"
        
        image_loaded = False
        try:
            st.image(google_drive_url, caption="Foto Developer", use_container_width=False, width=200)
            image_loaded = True
        except Exception as e:
            st.warning(f"⚠️ Gagal memuat foto dari Google Drive: {e}. Mencoba path lokal...")
        
        # Jika Google Drive gagal, coba path lokal
        if not image_loaded:
            photo_path = "assets/Tezza_2024_10_20_190012490.jpg"
            if os.path.exists(photo_path):
                st.image(photo_path, caption="Foto Developer", use_container_width=False, width=200)
            else:
                st.warning("⚠️ Foto developer tidak ditemukan di path lokal. Pastikan file gambar berada di folder 'assets' di direktori aplikasi.")
                # Fallback ke placeholder
                st.image("https://via.placeholder.com/200x250?text=Developer+Photo", caption="Foto Developer (Placeholder)", use_container_width=False, width=200)
    
    with col_right:
        st.write("""
        **Nama:** Cut Nisa Shafira  
        **Jurusan:** S1 Statistika, Universitas Syiah Kuala  
        **Angkatan:** 2022  
        **Praktikum:** Pemrograman Big Data  
        **Asisten Lab:** Diaz Darsya Rizqullah | Musliadi  
        **Kontak:** cutnisa386@gmail.com | LinkedIn: Cut Nisa  
        
        Developer mengembangkan dashboard aplikasi ini untuk memenuhi tugas praktikum mata kuliah pemrograman Big Data.
        """)
    
    st.markdown("---")
    
    # --- Informasi Tentang Aplikasi ---
    st.subheader("ℹ️ Informasi Tentang Aplikasi")
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
        
        Dibuat dengan ❤️ oleh Cut Nisa Shafira. Jika ada pertanyaan, hubungi kami!
    """)
    st.image("https://via.placeholder.com/800x400?text=AI+Powered+App", caption="Ilustrasi Aplikasi AI", use_container_width=True)

elif st.session_state.page == "Deteksi Objek (YOLO)":
    st.title("🔍 Deteksi Objek (YOLO)")
    st.write("Unggah gambar untuk deteksi objek secara real-time menggunakan YOLO, diikuti dengan klasifikasi gambar jika diinginkan!")
    
    if yolo_model is None:
        st.error("❌ Model YOLO tidak tersedia. Fitur deteksi objek tidak dapat digunakan. Silakan periksa model YOLO.")
    else:
        uploaded_file = st.file_uploader("📤 Unggah gambar", type=["jpg", "jpeg", "png"], key="yolo_uploader")
        
        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert("RGB")
            img_np = np.array(img)

            # --- Deteksi Objek dengan YOLO ---
            with st.spinner("🔍 Mendeteksi objek..."):
                try:
                    results = yolo_model(img_np)
                    result_img = results[0].plot()

                    # Tampilkan gambar sebelum dan sesudah berdampingan
                    col_before, col_after = st.columns(2)
                    with col_before:
                        st.markdown('<div class="image-label">📸 Gambar Asli</div>', unsafe_allow_html=True)
                        st.image(img, use_container_width=True)
                    with col_after:
                        st.markdown('<div class="image-label">📦 Hasil Deteksi YOLO</div>', unsafe_allow_html=True)
                        st.image(result_img, use_container_width=True)

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
                        
                        # --- Crop Gambar Berdasarkan Bounding Box Pertama ---
                        # Ambil bounding box pertama (atau yang paling confident)
                        if len(results[0].boxes) > 0:
                            box = results[0].boxes[0]  # Ambil box pertama
                            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinat bounding box
                            cropped_img = img.crop((x1, y1, x2, y2))  # Crop gambar
                            st.image(cropped_img, caption="✂️ Gambar yang Dicrop dari Deteksi", use_container_width=False, width=200)
                            
                            # Simpan cropped_img untuk klasifikasi
                            st.session_state.cropped_img = cropped_img
                        else:
                            st.session_state.cropped_img = None
                    else:
                        st.info("ℹ️ Tidak ada objek terdeteksi.")
                        st.session_state.cropped_img = None
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan saat deteksi YOLO: {e}")
                    st.session_state.cropped_img = None

            # --- Lanjutkan dengan Klasifikasi Gambar ---
            st.markdown("---")
            st.subheader("🧠 Klasifikasi Gambar (Opsional)")
            st.write("Jika ingin, lanjutkan dengan klasifikasi gambar menggunakan model Keras. Jika ada objek terdeteksi, klasifikasi akan menggunakan gambar yang dicrop untuk akurasi lebih baik.")
            
            if st.button("🔄 Lakukan Klasifikasi", key="classify_after_yolo"):
                # Gunakan cropped_img jika ada, jika tidak gunakan img asli
                img_to_classify = st.session_state.get('cropped_img', img)
                
                if keras_model:
                    with st.spinner("🧠 Mengklasifikasikan gambar..."):
                        try:
                            # Sesuaikan ukuran input model
                            input_shape = keras_model.input_shape[1:3]
                            img_resized = img_to_classify.resize(input_shape)

                            # Preprocessing
                            x = image.img_to_array(img_resized)
                            x = np.expand_dims(x, axis=0) / 255.0

                            preds = keras_model.predict(x)
                            pred_class = np.argmax(preds, axis=1)[0]
                            class_names = ["maize", "jute", "rice", "wheat", "sugarcane"]

                            st.subheader("📊 Hasil Klasifikasi:")
                            st.write(f"🌾 **Prediksi:** {class_names[pred_class]}")
                            st.write(f"📈 **Probabilitas:** {np.max(preds) * 100:.2f}%")
                            
                            # Perbaikan: Konversi ke float untuk progress bar
                            max_prob = float(np.max(preds))
                            st.progress(max_prob)
                            
                            # Tambahan: Emoji berdasarkan prediksi
                            emoji_map = {"maize": "🌽", "jute": "🌿", "rice": "🌾", "wheat": "🌾", "sugarcane": "🍯"}
                            st.write(f"{emoji_map[class_names[pred_class]]} Wow, ini terlihat seperti {class_names[pred_class]}!")
                            
                            # Tambahan: Tampilkan semua probabilitas kelas
                            st.subheader("📊 Probabilitas Semua Kelas:")
                            for i, prob in enumerate(preds[0]):
                                st.write(f"- {class_names[i]}: {prob * 100:.2f}%")
                            
                            # Tambahan: Peringatan jika probabilitas rendah
                            if max_prob < 0.5:
                                st.warning("⚠️ Probabilitas prediksi rendah. Model mungkin kurang yakin. Coba gambar yang lebih jelas, fokus pada tanaman utama, atau latih ulang model.")
                            
                        except Exception as e:
                            st.error(f"❌ Terjadi kesalahan saat klasifikasi: {e}")
                else:
                    st.warning("⚠️ Model Keras belum berhasil dimuat.")

elif st.session_state.page == "Klasifikasi Gambar":
    st.title("🧠 Klasifikasi Gambar")
   
