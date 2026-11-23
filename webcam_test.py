# ============================================================
# 06.2 PENGUJIAN WEBCAM
# ============================================================

import cv2

# Membuka kamera utama
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Webcam tidak dapat dibuka. "
        "Periksa izin kamera dan perangkat."
    )

print("=== WEBCAM AKTIF ===")
print("Tekan Q untuk keluar.")

try:
    while True:
        success, frame = cap.read()

        if not success:
            print("Gagal membaca frame.")
            break

        # Flip hanya untuk tampilan seperti cermin
        display_frame = cv2.flip(frame, 1)

        cv2.putText(
            display_frame,
            "SIBI Webcam Test - Press Q to Quit",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "SIBI Webcam Test",
            display_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

print("Webcam ditutup.")
