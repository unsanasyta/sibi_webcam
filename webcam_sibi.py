
# ============================================================
# SIBI VISION — REAL-TIME SIGN LANGUAGE RECOGNITION
# MediaPipe Hand Landmarker + 1D-CNN
# ============================================================

from pathlib import Path
import json
import time

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf


# ============================================================
# 1. KONFIGURASI FILE DAN MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_sibi_1dcnn.keras"
LANDMARKER_PATH = BASE_DIR / "models" / "hand_landmarker.task"
CLASS_PATH = BASE_DIR / "config" / "class_mapping.json"
NORMALIZATION_PATH = BASE_DIR / "config" / "normalization_config.json"

with open(CLASS_PATH, encoding="utf-8") as f:
    class_mapping = json.load(f)

with open(NORMALIZATION_PATH, encoding="utf-8") as f:
    normalization_config = json.load(f)

CLASSES = class_mapping["classes"]

assert len(CLASSES) == 24

model = tf.keras.models.load_model(
    str(MODEL_PATH),
    compile=False
)

assert model.input_shape[1:] == (21, 3)
assert model.output_shape[-1] == 24


# ============================================================
# 2. KONFIGURASI WARNA DAN LAYOUT
# Warna menggunakan format BGR OpenCV
# ============================================================

BG = (24, 20, 17)
SURFACE = (39, 34, 30)
SURFACE_2 = (52, 45, 39)
BORDER = (72, 63, 55)

WHITE = (245, 244, 241)
MUTED = (169, 159, 149)

CYAN = (234, 215, 42)
GREEN = (148, 216, 75)
YELLOW = (66, 207, 255)
RED = (105, 105, 245)

CANVAS_W = 1280
CANVAS_H = 730

CAM_X = 26
CAM_Y = 112
CAM_W = 850
CAM_H = 546

PANEL_X = 898
PANEL_Y = 112
PANEL_W = 356
PANEL_H = 546

WINDOW_NAME = "SIBI Vision - Real-Time Recognition"


# ============================================================
# 3. HELPER TAMPILAN
# ============================================================

def draw_text(
    image,
    content,
    x,
    y,
    scale=0.6,
    color=WHITE,
    thickness=1
):
    cv2.putText(
        image,
        str(content),
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_rounded_rect(
    image,
    x1,
    y1,
    x2,
    y2,
    radius,
    color
):
    radius = int(min(
        radius,
        (x2 - x1) / 2,
        (y2 - y1) / 2
    ))

    cv2.rectangle(
        image,
        (x1 + radius, y1),
        (x2 - radius, y2),
        color,
        -1
    )

    cv2.rectangle(
        image,
        (x1, y1 + radius),
        (x2, y2 - radius),
        color,
        -1
    )

    for cx, cy in [
        (x1 + radius, y1 + radius),
        (x2 - radius, y1 + radius),
        (x1 + radius, y2 - radius),
        (x2 - radius, y2 - radius)
    ]:
        cv2.circle(
            image, (cx, cy), radius, color, -1,
            cv2.LINE_AA
        )


def draw_corner_box(image, x1, y1, x2, y2):
    """Bounding box berbentuk sudut, bukan kotak penuh."""

    color = CYAN
    length = 22
    thickness = 2

    corners = [
        ((x1, y1 + length), (x1, y1), (x1 + length, y1)),
        ((x2 - length, y1), (x2, y1), (x2, y1 + length)),
        ((x1, y2 - length), (x1, y2), (x1 + length, y2)),
        ((x2 - length, y2), (x2, y2), (x2, y2 - length))
    ]

    for a, b, c in corners:
        cv2.line(image, a, b, color, thickness, cv2.LINE_AA)
        cv2.line(image, b, c, color, thickness, cv2.LINE_AA)


# ============================================================
# 4. NORMALISASI — SAMA DENGAN NOTEBOOK 03
# ============================================================

def normalize_landmarks(landmarks, width, height):

    raw = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float64
    )

    if raw.shape != (21, 3):
        return None

    if not np.isfinite(raw).all():
        return None

    # Penyesuaian skala koordinat: xW, yH, zW
    adjusted = raw * np.array(
        [width, height, width],
        dtype=np.float64
    )

    # Wrist-relative
    relative = adjusted - adjusted[0:1, :]

    # Skala: jarak XY landmark 0–9
    scale = np.linalg.norm(
        adjusted[9, :2] - adjusted[0, :2]
    )

    if (
        not np.isfinite(scale)
        or scale <= np.finfo(np.float64).eps
    ):
        return None

    normalized = relative / scale

    if not np.isfinite(normalized).all():
        return None

    return normalized.astype(np.float32)


# ============================================================
# 5. VISUALISASI LANDMARK TANGAN
# ============================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

FINGERTIPS = {4, 8, 12, 16, 20}


def draw_hand_overlay(frame, landmarks):

    height, width = frame.shape[:2]

    points = []

    for lm in landmarks:

        # Frame visual dicerminkan, input model tidak
        x = int((1.0 - lm.x) * (width - 1))
        y = int(lm.y * (height - 1))

        x = int(np.clip(x, 0, width - 1))
        y = int(np.clip(y, 0, height - 1))

        points.append((x, y))

    # Garis skeleton
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            CYAN,
            2,
            cv2.LINE_AA
        )

    # Titik landmark
    for i, point in enumerate(points):

        color = YELLOW if i in FINGERTIPS else WHITE
        radius = 6 if i in FINGERTIPS else 4

        if i == 0:
            color = GREEN
            radius = 7

        cv2.circle(
            frame,
            point,
            radius,
            color,
            -1,
            cv2.LINE_AA
        )

        cv2.circle(
            frame,
            point,
            radius + 1,
            SURFACE,
            1,
            cv2.LINE_AA
        )

    # Bounding box berdasarkan titik terdeteksi
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    padding = 23

    x1 = max(0, min(xs) - padding)
    y1 = max(0, min(ys) - padding)
    x2 = min(width - 1, max(xs) + padding)
    y2 = min(height - 1, max(ys) + padding)

    draw_corner_box(frame, x1, y1, x2, y2)


# ============================================================
# 6. MEMBUAT DASHBOARD REAL-TIME
# ============================================================

def render_dashboard(
    camera_frame,
    predicted_class,
    score,
    hand_detected,
    normalization_valid,
    fps
):

    canvas = np.full(
        (CANVAS_H, CANVAS_W, 3),
        BG,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    draw_rounded_rect(
        canvas, 26, 24, 68, 66, 11, CYAN
    )

    draw_text(
        canvas, "S", 39, 55,
        0.95, BG, 2
    )

    draw_text(
        canvas, "SIBI VISION",
        82, 48,
        1.00, WHITE, 2
    )

    draw_text(
        canvas,
        "REAL-TIME SIGN LANGUAGE RECOGNITION",
        83, 73,
        0.42, MUTED, 1
    )

    draw_rounded_rect(
        canvas, 1040, 29, 1254, 68,
        12, SURFACE
    )

    cv2.circle(
        canvas, (1061, 49), 5, GREEN, -1
    )

    draw_text(
        canvas, "SYSTEM ACTIVE",
        1076, 54,
        0.55, WHITE, 1
    )

    # --------------------------------------------------------
    # PANEL KAMERA
    # --------------------------------------------------------

    draw_rounded_rect(
        canvas,
        CAM_X - 2, CAM_Y - 2,
        CAM_X + CAM_W + 2,
        CAM_Y + CAM_H + 2,
        16, BORDER
    )

    # Latar area kamera
    cv2.rectangle(
        canvas,
        (CAM_X, CAM_Y),
        (CAM_X + CAM_W, CAM_Y + CAM_H),
        SURFACE,
        -1
    )

    # Fit frame ke area kamera tanpa mengubah aspect ratio
    source_h, source_w = camera_frame.shape[:2]

    ratio = min(
        CAM_W / source_w,
        CAM_H / source_h
    )

    new_w = max(1, int(source_w * ratio))
    new_h = max(1, int(source_h * ratio))

    resized = cv2.resize(
        camera_frame,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    offset_x = CAM_X + (CAM_W - new_w) // 2
    offset_y = CAM_Y + (CAM_H - new_h) // 2

    canvas[
        offset_y:offset_y + new_h,
        offset_x:offset_x + new_w
    ] = resized

    # Label live camera
    draw_rounded_rect(
        canvas,
        CAM_X + 16, CAM_Y + 16,
        CAM_X + 176, CAM_Y + 54,
        10, SURFACE
    )

    cv2.circle(
        canvas,
        (CAM_X + 34, CAM_Y + 35),
        5, GREEN, -1
    )

    draw_text(
        canvas, "LIVE CAMERA",
        CAM_X + 48, CAM_Y + 41,
        0.52, WHITE, 1
    )

    # FPS
    draw_rounded_rect(
        canvas,
        CAM_X + CAM_W - 134,
        CAM_Y + 16,
        CAM_X + CAM_W - 16,
        CAM_Y + 54,
        10, SURFACE
    )

    draw_text(
        canvas,
        f"{fps:.1f} FPS",
        CAM_X + CAM_W - 116,
        CAM_Y + 41,
        0.53, WHITE, 1
    )

    # --------------------------------------------------------
    # PANEL INFORMASI
    # --------------------------------------------------------

    draw_rounded_rect(
        canvas,
        PANEL_X, PANEL_Y,
        PANEL_X + PANEL_W,
        PANEL_Y + PANEL_H,
        18, SURFACE
    )

    px = PANEL_X + 22

    draw_text(
        canvas,
        "HASIL KLASIFIKASI",
        px, PANEL_Y + 38,
        0.63, WHITE, 2
    )

    draw_text(
        canvas,
        "Prediksi alfabet SIBI statis",
        px, PANEL_Y + 66,
        0.45, MUTED, 1
    )

    # Kartu huruf utama
    draw_rounded_rect(
        canvas,
        px, PANEL_Y + 88,
        PANEL_X + PANEL_W - 22,
        PANEL_Y + 264,
        16, SURFACE_2
    )

    draw_text(
        canvas,
        "HURUF TERDETEKSI",
        px + 20, PANEL_Y + 117,
        0.47, MUTED, 1
    )

    letter = (
        predicted_class
        if predicted_class is not None
        else "-"
    )

    # Huruf prediksi besar
    font_scale = 4.4

    (text_w, _), _ = cv2.getTextSize(
        letter,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        7
    )

    letter_center_x = (
        PANEL_X + PANEL_W // 2 - text_w // 2
    )

    draw_text(
        canvas,
        letter,
        letter_center_x,
        PANEL_Y + 236,
        font_scale,
        CYAN if predicted_class else MUTED,
        7
    )

    # --------------------------------------------------------
    # SKOR SOFTMAX
    # --------------------------------------------------------

    draw_text(
        canvas,
        "SKOR SOFTMAX",
        px, PANEL_Y + 303,
        0.50, MUTED, 1
    )

    score_value = (
        float(np.clip(score, 0.0, 1.0))
        if score is not None
        else 0.0
    )

    score_label = (
        f"{score_value * 100:.1f}%"
        if score is not None
        else "--"
    )

    draw_text(
        canvas,
        score_label,
        PANEL_X + PANEL_W - 105,
        PANEL_Y + 304,
        0.68, WHITE, 2
    )

    bar_x = px
    bar_y = PANEL_Y + 322
    bar_w = PANEL_W - 44
    bar_h = 12

    draw_rounded_rect(
        canvas,
        bar_x, bar_y,
        bar_x + bar_w,
        bar_y + bar_h,
        6, SURFACE_2
    )

    filled_w = int(bar_w * score_value)

    if filled_w > 0:
        cv2.rectangle(
            canvas,
            (bar_x, bar_y),
            (bar_x + filled_w, bar_y + bar_h),
            CYAN,
            -1
        )

    draw_text(
        canvas,
        "Skor model, bukan jaminan kebenaran",
        px, PANEL_Y + 359,
        0.38, MUTED, 1
    )

    # --------------------------------------------------------
    # STATUS SISTEM
    # --------------------------------------------------------

    cv2.line(
        canvas,
        (px, PANEL_Y + 380),
        (PANEL_X + PANEL_W - 22, PANEL_Y + 380),
        BORDER, 1, cv2.LINE_AA
    )

    draw_text(
        canvas,
        "STATUS SISTEM",
        px, PANEL_Y + 411,
        0.52, WHITE, 1
    )

    detection_color = GREEN if hand_detected else YELLOW

    cv2.circle(
        canvas,
        (px + 6, PANEL_Y + 439),
        5, detection_color, -1
    )

    draw_text(
        canvas,
        "Tangan terdeteksi"
        if hand_detected else
        "Menunggu tangan...",
        px + 22, PANEL_Y + 445,
        0.48, WHITE, 1
    )

    cv2.circle(
        canvas,
        (px + 6, PANEL_Y + 478),
        5, GREEN if normalization_valid else MUTED,
        -1
    )

    draw_text(
        canvas,
        "Normalisasi valid"
        if normalization_valid else
        "Belum ada fitur valid",
        px + 22, PANEL_Y + 484,
        0.48, WHITE, 1
    )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    draw_text(
        canvas,
        "21 LANDMARKS  /  1D-CNN  /  24 STATIC LETTERS",
        27, 696,
        0.48, MUTED, 1
    )

    draw_text(
        canvas,
        "TEKAN Q UNTUK KELUAR",
        1008, 696,
        0.48, CYAN, 1
    )

    return canvas


# ============================================================
# 7. INISIALISASI MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(LANDMARKER_PATH)
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# 8. WEBCAM REAL-TIME
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Webcam tidak dapat dibuka.")

# Resolusi permintaan kamera; perangkat dapat memilih
# resolusi berbeda jika tidak didukung
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    CANVAS_W,
    CANVAS_H
)

print("=====================================")
print("      SIBI VISION - READY")
print("=====================================")
print("Model       : 1D-CNN")
print("Jumlah kelas: 24 alfabet")
print("Landmarker  : MediaPipe")
print("Webcam      : Aktif")
print("Tekan Q untuk keluar")
print("=====================================")

previous_time = time.perf_counter()
smoothed_fps = 0.0

try:

    with HandLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:

            success, frame = cap.read()

            if not success:
                print("Gagal membaca frame.")
                break

            height, width = frame.shape[:2]

            # MediaPipe menerima frame asli, bukan hasil flip
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            rgb_frame = np.ascontiguousarray(
                rgb_frame
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            result = landmarker.detect(mp_image)

            # Hanya tampilan webcam yang dicerminkan
            display_frame = cv2.flip(frame, 1)

            predicted_class = None
            score = None

            hand_detected = False
            normalization_valid = False

            if result.hand_landmarks:

                hand_detected = True

                landmarks = result.hand_landmarks[0]

                # Visualisasi landmark pada frame tampilan
                draw_hand_overlay(
                    display_frame,
                    landmarks
                )

                # Preprocessing identik dengan Notebook 03
                normalized = normalize_landmarks(
                    landmarks,
                    width,
                    height
                )

                if normalized is not None:

                    normalization_valid = True

                    input_data = normalized[
                        np.newaxis, :, :
                    ]

                    # Tidak memperbarui bobot model
                    probabilities = model(
                        input_data,
                        training=False
                    ).numpy()[0]

                    predicted_id = int(
                        np.argmax(probabilities)
                    )

                    score = float(
                        probabilities[predicted_id]
                    )

                    predicted_class = CLASSES[
                        predicted_id
                    ]

            # Hitung FPS tampilan
            current_time = time.perf_counter()

            elapsed = current_time - previous_time
            previous_time = current_time

            instant_fps = (
                1.0 / elapsed
                if elapsed > 0
                else 0.0
            )

            if smoothed_fps == 0:
                smoothed_fps = instant_fps
            else:
                smoothed_fps = (
                    0.85 * smoothed_fps
                    + 0.15 * instant_fps
                )

            # Dashboard
            dashboard = render_dashboard(
                camera_frame=display_frame,
                predicted_class=predicted_class,
                score=score,
                hand_detected=hand_detected,
                normalization_valid=normalization_valid,
                fps=smoothed_fps
            )

            cv2.imshow(
                WINDOW_NAME,
                dashboard
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()

print("SIBI Vision ditutup.")