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
        transition: background-color 0.3s, transform 0.2s;
    }
    .nav-button:hover {
        background-color: #5F9EA0;  /* Darker Teal on hover */
        transform: translateY(-2px);
    }
    .content {
        padding: 20px;
        background-color: #F0F8FF;  /* Light Teal-tinted background for content */
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
    .landing-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #008080;
        margin-bottom: 20px;
    }
    .landing-subtitle {
        text-align: center;
        font-size: 20px;
        color: #5F9EA0;
        margin-bottom: 30px;
    }
    .feature-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s;
    }
    .feature-card:hover {
        transform: scale(1.05);
    }
    .fade-in {
        animation: fadeIn 2s;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">🌟 Selamat Datang di Image Classification & Detection App 🌟</div>', unsafe_allow_html=True)

# Navigation buttons in header (updated order: Home, Deteksi Objek, Klasifikasi Gambar, Tentang)
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🏠 Home", key="home"):
        st.session_state.page = "Home"
with col2:
    if st.button("🔍 Deteksi Objek (YOLO)", key="yolo"):
        st.session_state.page = "Deteksi Objek (YOLO)"
with col3:
    if st.button("🧠 Klasifikasi Gambar", key="classify"):
        st.session_state.page = "Klasifikasi Gambar"
with col4:
    if st.button("📖 Tentang", key="about"):
        st.session_state.page = "Tentang"

# Default page (updated to Home)
if "page" not in st.session_state:
    st.session_state.page = "Home"

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

if st.session_state.page == "Home":
    # Landing Page
    st.markdown('<div class="landing-title fade-in">🏠 Selamat Datang di Home</div>', unsafe_allow_html=True)
    st.markdown('<div class="landing-subtitle">Jelajahi fitur deteksi objek dan klasifikasi gambar dengan AI canggih!</div>', unsafe_allow_html=True)
    
    # Feature Overview
    st.markdown("### 🌟 Fitur Unggulan Kami")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("#### 🔍 Deteksi Objek (YOLO)")
        st.write("Deteksi objek dalam gambar secara real-time. Lihat perbandingan gambar asli dan hasil deteksi!")
        
        # Contoh gambar deteksi (placeholder atau dummy)
        # Asumsikan kita punya gambar contoh; jika tidak, gunakan placeholder
        try:
            # Gambar asli contoh (ganti dengan path gambar nyata jika ada)
            img_original = Image.open("https://via.placeholder.com/300x200?text=Gambar+Asli").convert("RGB")
            img_detected = Image.open("https://via.placeholder.com/300x200?text=Hasil+Deteksi+YOLO").convert("RGB")
            
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                if st.button("📸 Lihat Gambar Asli", key="popup_original"):
                    with st.popover("📸 Gambar Asli"):
                        st.image(img_original, caption="Gambar Asli Contoh", use_container_width=True)
                        st.write("Ini adalah gambar asli sebelum deteksi objek.")
            with col_img2:
                if st.button("📦 Lihat Hasil Deteksi", key="popup_detected"):
                    with st.popover("📦 Hasil Deteksi YOLO"):
                        st.image(img_detected, caption="Hasil Deteksi YOLO Contoh", use_container_width=True)
                        st.write("Objek terdeteksi dengan bounding box dan label.")
        except:
            st.info("Gambar contoh tidak tersedia. Unggah gambar di halaman Deteksi Objek untuk melihat hasil nyata!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("#### 🧠 Klasifikasi Gambar")
        st.write("Klasifikasikan jenis tanaman dari gambar. Dapatkan prediksi dengan probabilitas tinggi!")
        
        # Contoh gambar klasifikasi (placeholder atau dummy)
        try:
            # Gambar tanaman contoh (ganti dengan path gambar nyata jika ada)
            img_plant = Image.open("https://via.placeholder.com/300x200?text=Gambar+Tanaman").convert("RGB")
            
            if st.button("🌾 Lihat Prediksi Klasifikasi", key="popup_classify"):
                with st.popover("🌾 Prediksi Klasifikasi"):
                    st.image(img_plant, caption="Gambar Tanaman Contoh", use_container_width=True)
                    st.write("**Prediksi:** Jagung (Maize)")
                    st.write("**Probabilitas:** 95.67%")
                    st.progress(0.9567)
        except:
            st.info("Gambar contoh tidak tersedia. Unggah gambar di halaman Klasifikasi Gambar untuk melihat hasil nyata!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Call to Action
    st.markdown("---")
    st.markdown("### 🚀 Mulai Sekarang!")
    st.write("Pilih fitur di navigasi di atas untuk mencoba deteksi objek atau klasifikasi gambar. Aplikasi ini menggunakan AI untuk memberikan hasil akurat dan cepat!")
    st.image("https://via.placeholder.com/800x400?text=AI+Powered+App", caption="Ilustrasi Aplikasi AI", use_container_width=True)

elif st.session_state.page == "Tentang":
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
                            st.write(f"{emoji_map[class_names[pred_class]]} Wow, ini terlihat seperti {class_names[pred
