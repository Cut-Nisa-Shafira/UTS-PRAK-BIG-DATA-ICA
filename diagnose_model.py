import tensorflow as tf
import os
import traceback

MODEL_PATH = os.path.join("model", "ica_laporan2.keras")

print("=======================================")
print("🔍 MEMERIKSA STRUKTUR MODEL KERAS")
print("=======================================")
print(f"TensorFlow version: {tf.__version__}")
print(f"Model path: {os.path.abspath(MODEL_PATH)}\n")

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
    print("✅ Model berhasil dimuat!\n")

    print("=== INFORMASI INPUT MODEL ===")
    if isinstance(model.input, (list, tuple)):
        print(f"Jumlah input: {len(model.input)}")
        for i, inp in enumerate(model.input):
            print(f"Input {i+1}: name={inp.name}, shape={inp.shape}, dtype={inp.dtype}")
    else:
        print(f"Hanya satu input: {model.input.name}, shape={model.input.shape}, dtype={model.input.dtype}")

    print("\n=== INFORMASI OUTPUT MODEL ===")
    if isinstance(model.output, (list, tuple)):
        print(f"Jumlah output: {len(model.output)}")
        for i, outp in enumerate(model.output):
            print(f"Output {i+1}: name={outp.name}, shape={outp.shape}, dtype={outp.dtype}")
    else:
        print(f"Hanya satu output: {model.output.name}, shape={model.output.shape}, dtype={model.output.dtype}")

    print("\n=== RINGKASAN MODEL ===")
    model.summary()

except Exception as e:
    print("❌ Gagal memuat model!")
    print("Pesan error:")
    print(e)
    print("\nTraceback lengkap:")
    traceback.print_exc()



