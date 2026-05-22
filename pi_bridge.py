"""
Sherlock Smart Home - Pi Bridge
On-demand face recognition (no continuous stream),
Supabase command polling for remote capture/upload,
UART sensor data from ESP-NOW hub.

Install: pip install supabase httpx insightface onnxruntime pyserial --break-system-packages
"""

import cv2
import numpy as np
import os
import sys
import time
import pickle
import struct
import urllib.request
import threading
from datetime import datetime, timezone
from supabase import create_client
import serial

# --- CONFIG ---
SUPABASE_URL = os.environ.get("SUPABASE_URL", "YOUR_SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "YOUR_SUPABASE_ANON_KEY")
ESP32_STREAM_URL = "http://192.168.137.123:81/stream"
BRAIN_FILE = os.path.expanduser("~/known_faces.pkl")
POLL_INTERVAL = 2  # seconds
COSINE_THRESHOLD = 0.35

# UART config for ESP-NOW hub
UART_PORT = "/dev/ttyAMA0"
UART_BAUD = 1200

# --- INIT SUPABASE ---
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- LOAD FACE RECOGNITION ---
import insightface
from insightface.app import FaceAnalysis

print("Loading face recognition model...")
face_app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(320, 320))

try:
    with open(BRAIN_FILE, "rb") as f:
        data = pickle.load(f)
        known_embeddings = data["embeddings"]
        known_names = data["names"]
    print(f"Brain loaded: {len(known_embeddings)} embeddings")
    print(f"Known people: {list(set(known_names))}")
except Exception as e:
    print(f"WARNING: No brain file found. Recognition disabled.")
    known_embeddings = None
    known_names = None


def normalize_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def recognize_face(embedding):
    if known_embeddings is None:
        return "Unknown", 0.0, {}
    embedding = embedding / np.linalg.norm(embedding)
    similarities = np.dot(known_embeddings, embedding)
    unique_names = list(set(known_names))
    person_scores = {}
    for name in unique_names:
        idxs = [i for i, n in enumerate(known_names) if n == name]
        person_sims = similarities[idxs]
        person_scores[name] = float(np.mean(np.sort(person_sims)[-5:]))
    best_name = max(person_scores, key=person_scores.get)
    best_sim = person_scores[best_name]
    return best_name, best_sim, person_scores


def grab_single_frame():
    """Connect to ESP32 stream, grab one frame, disconnect."""
    try:
        stream = urllib.request.urlopen(ESP32_STREAM_URL, timeout=10)
        byte_buffer = b''
        deadline = time.time() + 5  # 5 second timeout

        while time.time() < deadline:
            chunk = stream.read(1024)
            if not chunk:
                break
            byte_buffer += chunk
            a = byte_buffer.find(b'\xff\xd8')
            b = byte_buffer.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = byte_buffer[a:b + 2]
                img_np = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                stream.close()
                return frame

        stream.close()
    except Exception as e:
        print(f"[CAPTURE] Failed to grab frame: {e}")

    return None


def capture_and_upload():
    """Grab a single frame, run recognition, upload to Supabase."""
    print("[CAPTURE] Grabbing frame from ESP32...")
    frame = grab_single_frame()

    if frame is None:
        print("[CAPTURE] Camera offline — no frame available")
        try:
            supabase.table("captures").insert({
                "image_url": "",
                "detected_name": "Camera Offline",
                "confidence": 0.0
            }).execute()
        except Exception as e:
            print(f"[ERROR] {e}")
        return

    # Rotate if ESP32 is mounted upside down
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    frame = normalize_frame(frame)
    display = cv2.resize(frame, (640, 480))

    # Run face recognition
    faces = face_app.get(display)
    detected_name = "No face"
    confidence = 0.0

    for face in faces:
        name, sim, scores = recognize_face(face.embedding)

        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        if sim > COSINE_THRESHOLD:
            detected_name = name
            confidence = sim
            color = (0, 255, 0)
        else:
            detected_name = "Unknown"
            confidence = sim
            color = (0, 0, 255)

        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display, f"{detected_name} ({sim:.0%})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        score_str = " | ".join([f"{n}: {s:.3f}" for n, s in sorted(scores.items(), key=lambda x: -x[1])])
        print(f"[CAPTURE] {detected_name} || {score_str}")

    # Upload
    try:
        _, buffer = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_bytes = buffer.tobytes()

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"

        supabase.storage.from_("captures").upload(
            filename, img_bytes,
            file_options={"content-type": "image/jpeg"}
        )

        image_url = supabase.storage.from_("captures").get_public_url(filename)

        supabase.table("captures").insert({
            "image_url": image_url,
            "detected_name": detected_name,
            "confidence": round(confidence, 3)
        }).execute()

        print(f"[UPLOADED] {filename} | {detected_name} ({confidence:.0%})")

    except Exception as e:
        print(f"[UPLOAD ERROR] {e}")


def poll_commands_thread():
    """Background thread that polls Supabase for commands."""
    while True:
        try:
            result = supabase.table("commands") \
                .select("*") \
                .eq("status", "pending") \
                .order("created_at", desc=False) \
                .limit(1) \
                .execute()

            if result.data:
                cmd = result.data[0]
                cmd_id = cmd["id"]
                command = cmd["command"]
                print(f"\n[COMMAND] Received: {command}")

                supabase.table("commands").update({
                    "status": "processing",
                    "processed_at": datetime.now(timezone.utc).isoformat()
                }).eq("id", cmd_id).execute()

                if command == "capture":
                    capture_and_upload()

                supabase.table("commands").update({
                    "status": "done"
                }).eq("id", cmd_id).execute()

        except Exception as e:
            print(f"\n[POLL ERROR] {e}")

        time.sleep(POLL_INTERVAL)


def heartbeat_thread():
    """Background thread that updates device status in Supabase."""
    while True:
        try:
            # Check if ESP32 stream is reachable (quick probe)
            try:
                probe = urllib.request.urlopen(ESP32_STREAM_URL, timeout=3)
                probe.close()
                status = "online"
            except Exception:
                status = "camera_offline"

            supabase.table("device_status").upsert({
                "id": "pi",
                "status": status,
                "last_seen": datetime.now(timezone.utc).isoformat()
            }).execute()
        except Exception:
            pass
        time.sleep(10)


def uart_reader_thread():
    """Background thread that reads sensor data from ESP-NOW hub via UART."""
    try:
        ser = serial.Serial(UART_PORT, UART_BAUD, timeout=1)
        print(f"[UART] Connected to {UART_PORT} @ {UART_BAUD}")
    except Exception as e:
        print(f"[UART] Failed to open {UART_PORT}: {e}")
        print("[UART] Sensor data will not be available.")
        return

    while True:
        try:
            if ser.in_waiting == 0:
                time.sleep(0.1)
                continue

            msg_type = ser.read(1)
            if len(msg_type) == 0:
                continue

            msg_type = msg_type[0]

            if msg_type == 0x00:
                payload = ser.read(18)
                if len(payload) < 18:
                    print("[UART] Incomplete environment packet")
                    continue

                temperature = struct.unpack('<f', payload[0:4])[0]
                pressure = struct.unpack('<f', payload[4:8])[0]
                humidity = struct.unpack('<f', payload[8:12])[0]
                mac = ':'.join(f'{b:02X}' for b in payload[12:18])

                print(f"\n[SENSOR] {mac} | Temp: {temperature:.1f}°C | Humidity: {humidity:.1f}% | Pressure: {pressure:.1f} hPa")

                try:
                    supabase.table("sensor_data").insert({
                        "mac_address": mac,
                        "data_type": "environment",
                        "temperature": round(temperature, 2),
                        "humidity": round(humidity, 2),
                        "pressure": round(pressure, 2),
                        "leak_detected": False
                    }).execute()
                except Exception as e:
                    print(f"[UART] Supabase upload error: {e}")

            elif msg_type == 0x01:
                payload = ser.read(6)
                if len(payload) < 6:
                    print("[UART] Incomplete leak packet")
                    continue

                mac = ':'.join(f'{b:02X}' for b in payload[0:6])
                print(f"\n[LEAK ALERT] {mac} - LEAK DETECTED!")

                try:
                    supabase.table("sensor_data").insert({
                        "mac_address": mac,
                        "data_type": "leak",
                        "leak_detected": True
                    }).execute()
                except Exception as e:
                    print(f"[UART] Supabase upload error: {e}")

            else:
                ser.reset_input_buffer()

        except Exception as e:
            print(f"[UART] Read error: {e}")
            time.sleep(1)


def main():
    print("\n=== Sherlock Smart Home Bridge ===")
    print(f"Supabase polling every {POLL_INTERVAL}s")
    print(f"Camera: on-demand capture (no continuous stream)\n")

    # Start background threads
    threading.Thread(target=poll_commands_thread, daemon=True).start()
    threading.Thread(target=heartbeat_thread, daemon=True).start()
    threading.Thread(target=uart_reader_thread, daemon=True).start()

    print("[READY] Waiting for commands. Ctrl+C to quit.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
