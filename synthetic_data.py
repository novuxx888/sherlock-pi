import cv2
import numpy as np
import os
import sys

# --- PATHS ---
INPUT_DIR = os.path.expanduser("~/videos")
OUTPUT_BASE = os.path.expanduser("~/synthetic_data")
ESP32_RES = (352, 288)
FACE_SIZE = (112, 112)

if not os.path.exists(OUTPUT_BASE):
    os.makedirs(OUTPUT_BASE)

# --- FACE DETECTOR SETUP ---
USE_HAAR = True
face_detector = None

yunet_path = os.path.expanduser("~/face_detection_yunet_2023mar.onnx")
if os.path.isfile(yunet_path):
    try:
        face_detector = cv2.FaceDetectorYN.create(yunet_path, "", (0, 0), 0.7, 0.3, 5000)
        USE_HAAR = False
        print("Using YuNet face detector")
    except Exception as e:
        print(f"YuNet failed to load: {e}, falling back to Haar")

if USE_HAAR:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    print("Using Haar cascade (less accurate on angled faces)")


def detect_and_crop_face(frame):
    """Detect the largest face and return a cropped region."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if USE_HAAR:
        faces = face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(60, 60))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    else:
        face_detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, detections = face_detector.detect(frame)
        if detections is None or len(detections) == 0:
            return None
        det = detections[0]
        x, y, w, h = int(det[0]), int(det[1]), int(det[2]), int(det[3])

    # Add 20% padding around the face
    pad_w, pad_h = int(w * 0.2), int(h * 0.2)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(frame.shape[1], x + w + pad_w)
    y2 = min(frame.shape[0], y + h + pad_h)

    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None
    face_crop = cv2.resize(face_crop, FACE_SIZE, interpolation=cv2.INTER_AREA)
    return face_crop


def normalize_exposure(frame):
    """CLAHE on L channel to standardize exposure across cameras."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_esp32_effects(frame):
    """Simulate ESP32 OV2640 camera characteristics."""
    # 1. Downsample to ESP32 resolution then back up
    small = cv2.resize(frame, ESP32_RES, interpolation=cv2.INTER_AREA)
    frame = cv2.resize(small, FACE_SIZE, interpolation=cv2.INTER_LINEAR)

    # 2. Add sensor noise (float math to avoid uint8 overflow)
    noise_sigma = np.random.uniform(3, 8)
    noise = np.random.normal(0, noise_sigma, frame.shape)
    frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 3. Moderate barrel distortion
    h, w = frame.shape[:2]
    dist_coeff = np.array([-0.22, 0.05, 0, 0], dtype=np.float32)
    cam_matrix = np.array([
        [w, 0, w / 2],
        [0, h, h / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    frame = cv2.undistort(frame, cam_matrix, dist_coeff)

    # 4. Slight color shift (ESP32 white balance is worse)
    frame = frame.astype(np.float32)
    frame[:, :, 0] *= np.random.uniform(0.92, 1.0)   # B
    frame[:, :, 1] *= np.random.uniform(0.95, 1.0)   # G
    frame[:, :, 2] *= np.random.uniform(1.0, 1.08)    # R (tends warm)
    frame = np.clip(frame, 0, 255).astype(np.uint8)

    # 5. JPEG compression artifacts (ESP32 sends JPEG)
    quality = np.random.randint(60, 85)
    _, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    frame = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return frame


# --- MAIN ---
print("Regenerating synthetic data with optimized settings...\n")
video_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.mov', '.mp4'))]

if not video_files:
    print(f"No video files found in {INPUT_DIR}")
    sys.exit(1)

total_saved = 0
total_missed = 0

for video_file in video_files:
    person_name = os.path.splitext(video_file)[0]
    print(f"Processing: {person_name}")

    out_person_path = os.path.join(OUTPUT_BASE, person_name)
    os.makedirs(out_person_path, exist_ok=True)

    cap = cv2.VideoCapture(os.path.join(INPUT_DIR, video_file))
    if not cap.isOpened():
        print(f"  ERROR: Could not open {video_file}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {total_frames} frames @ {fps:.1f} FPS")

    saved_count = 0
    frame_idx = 0
    no_face_count = 0

    while cap.isOpened() and saved_count < 150:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % 3 == 0:
            # Step 1: Detect and crop face
            face = detect_and_crop_face(frame)
            if face is None:
                no_face_count += 1
                frame_idx += 1
                continue

            # Step 2: Normalize exposure
            face = normalize_exposure(face)

            # Step 3: Save clean normalized version
            cv2.imwrite(
                f"{out_person_path}/{person_name}_clean_{saved_count}.jpg", face
            )

            # Step 4: Save ESP32-simulated version
            esp_face = apply_esp32_effects(face)
            esp_face = normalize_exposure(esp_face)
            cv2.imwrite(
                f"{out_person_path}/{person_name}_esp32_{saved_count}.jpg", esp_face
            )

            saved_count += 1

        frame_idx += 1
        if saved_count % 20 == 0 and saved_count > 0:
            sys.stdout.write(f"\r  Generated {saved_count} pairs...")
            sys.stdout.flush()

    cap.release()
    total_saved += saved_count
    total_missed += no_face_count
    print(f"\r  Done: {saved_count} pairs saved, {no_face_count} frames had no face detected")

print(f"\n{'='*50}")
print(f"Total: {total_saved} pairs saved across {len(video_files)} videos")
print(f"Missed frames: {total_missed}")
print(f"Output: {OUTPUT_BASE}")

if total_missed > total_saved * 2:
    print(f"\nWARNING: More than 2/3 of frames had no face detected.")
    if USE_HAAR:
        print("Download YuNet for much better detection:")
        print("  wget -O ~/face_detection_yunet_2023mar.onnx \\")
        print('    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"')
    else:
        print("Videos may have faces too small or too far from camera.")
        print("Re-record with face filling at least 1/4 of the frame.")
