# ============================================================
# 06.1 PEMERIKSAAN MODEL DAN KONFIGURASI
# ============================================================

from pathlib import Path
import hashlib
import json

import numpy as np
import mediapipe as mp
import tensorflow as tf
import cv2

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_sibi_1dcnn.keras"
LANDMARKER_PATH = BASE_DIR / "models" / "hand_landmarker.task"

MAPPING_PATH = BASE_DIR / "config" / "class_mapping.json"
NORMALIZATION_PATH = BASE_DIR / "config" / "normalization_config.json"

# ------------------------------------------------------------
# 1. Pemeriksaan file
# ------------------------------------------------------------

required_files = [
    MODEL_PATH,
    LANDMARKER_PATH,
    MAPPING_PATH,
    NORMALIZATION_PATH
]

print("=== PEMERIKSAAN FILE ===")

for path in required_files:
    exists = path.is_file()
    print(f"{path.name}: {exists}")
    assert exists, f"File tidak ditemukan: {path}"

# ------------------------------------------------------------
# 2. Membaca konfigurasi
# ------------------------------------------------------------

with open(MAPPING_PATH, encoding="utf-8") as f:
    class_mapping = json.load(f)

with open(NORMALIZATION_PATH, encoding="utf-8") as f:
    normalization_config = json.load(f)

classes = class_mapping["classes"]

assert len(classes) == 24
assert len(set(classes)) == 24

# ------------------------------------------------------------
# 3. Memeriksa SHA-256 model MediaPipe
# ------------------------------------------------------------

def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


actual_sha = sha256_file(LANDMARKER_PATH)

expected_sha = normalization_config[
    "mediapipe"
]["model_sha256"]

assert actual_sha == expected_sha, (
    "Model MediaPipe berbeda dari Notebook 03."
)

# ------------------------------------------------------------
# 4. Memuat model 1D-CNN
# ------------------------------------------------------------

model = tf.keras.models.load_model(
    str(MODEL_PATH),
    compile=False
)

assert model.input_shape[1:] == (21, 3)
assert model.output_shape[-1] == 24

# ------------------------------------------------------------
# 5. Memeriksa model Hand Landmarker
# ------------------------------------------------------------

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(LANDMARKER_PATH)
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

with mp.tasks.vision.HandLandmarker.create_from_options(
    options
):
    print("MediaPipe Hand Landmarker berhasil dimuat.")

# ------------------------------------------------------------
# 6. Ringkasan
# ------------------------------------------------------------

print("\n=== HASIL PEMERIKSAAN ===")

print("TensorFlow :", tf.__version__)
print("MediaPipe  :", mp.__version__)
print("OpenCV     :", cv2.__version__)

print("Input model :", model.input_shape)
print("Output model:", model.output_shape)

print("Jumlah kelas:", len(classes))
print("Model MediaPipe SHA-256: Sesuai")

print("\nSTATUS: MODEL DAN KONFIGURASI SIAP")
