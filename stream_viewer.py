import cv2
import numpy as np
import os
import sys
import time
import pickle
import threading
import urllib.request
import insightface
from insightface.app import FaceAnalysis

# --- CONFIG ---
STREAM_URL = "http://10.42.0.103:81/stream"
BRAIN_FILE = os.path.expanduser("~/known_faces.pkl")
COSINE_THRESHOLD = 0.35

os.environ["DISPLAY"] = ":0"


class StreamThread:
    def __init__(self, url):
        self.url = url
        self.frame = None
        self.stopped = False
        self.last_access = time.time()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            try:
                stream = urllib.request.urlopen(self.url, timeout=5)
                byte_buffer = b''
                while not self.stopped:
                    chunk = stream.read(1024)
                    if not chunk:
                        break
                    byte_buffer += chunk
                    a = byte_buffer.find(b'\xff\xd8')
                    b = byte_buffer.find(b'\xff\xd9')
                    if a != -1 and b != -1:
                        jpg = byte_buffer[a:b + 2]
                        byte_buffer = byte_buffer[b + 2:]
                        img_np = np.frombuffer(jpg, dtype=np.uint8)
                        self.frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                        self.last_access = time.time()
            except Exception:
                time.sleep(1)

    def stop(self):
        self.stopped = True


def normalize_frame(frame):
    """Same normalization used during training."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# --- LOAD BRAIN ---
try:
    with open(BRAIN_FILE, "rb") as f:
        data = pickle.load(f)
        known_embeddings = data["embeddings"]
        known_names = data["names"]
    print(f"Brain loaded: {len(known_embeddings)} embeddings")
    print(f"Known people: {list(set(known_names))}")
except Exception as e:
    print(f"CRITICAL: Could not load {BRAIN_FILE}. Run build_database.py first.")
    sys.exit(1)

# --- INIT INSIGHTFACE ---
print("Loading face recognition model...")
app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(320, 320))
print("Model ready.\n")


def recognize_face(embedding):
    """Compare embedding against known faces using cosine similarity."""
    embedding = embedding / np.linalg.norm(embedding)
    similarities = np.dot(known_embeddings, embedding)

    # Get best match per person (average of top 5 matches)
    unique_names = list(set(known_names))
    person_scores = {}
    for name in unique_names:
        idxs = [i for i, n in enumerate(known_names) if n == name]
        person_sims = similarities[idxs]
        person_scores[name] = float(np.mean(np.sort(person_sims)[-5:]))

    best_name = max(person_scores, key=person_scores.get)
    best_sim = person_scores[best_name]

    return best_name, best_sim, person_scores


def main():
    stream = StreamThread(STREAM_URL).start()
    print("[READY] Looking for faces. Press 'q' to quit.\n")

    frame_count = 0
    fps_time = time.time()
    fps = 0

    while True:
        frame = stream.frame
        if frame is None:
            time.sleep(0.01)
            continue

        # Rotate if your ESP32 is mounted upside down
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Normalize exposure (matches training preprocessing)
        frame_norm = normalize_frame(frame)

        # Resize for display
        display = cv2.resize(frame_norm, (640, 480))

        # Run face detection + embedding
        faces = app.get(display)

        label = "No face"
        color = (128, 128, 128)

        for face in faces:
            name, sim, scores = recognize_face(face.embedding)

            # Draw bounding box
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # Now threshold is directly on similarity (higher = better match)
            if sim > COSINE_THRESHOLD:
                label = f"{name} ({sim:.0%})"
                color = (0, 255, 0)
            else:
                label = f"Unknown ({sim:.0%})"
                color = (0, 0, 255)

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Print all person scores to terminal
            score_str = " | ".join([f"{n}: {s:.3f}" for n, s in sorted(scores.items(), key=lambda x: -x[1])])
            sys.stdout.write(f"\r{label} || {score_str}   ")
            sys.stdout.flush()

        # FPS counter
        frame_count += 1
        if time.time() - fps_time >= 1.0:
            fps = frame_count
            frame_count = 0
            fps_time = time.time()

        cv2.putText(display, f"FPS: {fps}", (540, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Face Recognition", display)

        if time.time() - stream.last_access > 5:
            cv2.destroyAllWindows()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stream.stop()
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
