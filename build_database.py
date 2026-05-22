import cv2
import numpy as np
import os
import pickle
import insightface
from insightface.app import FaceAnalysis

# --- CONFIG ---
DATA_DIR = os.path.expanduser("~/synthetic_data")
OUTPUT_FILE = os.path.expanduser("~/known_faces.pkl")

# Initialize InsightFace (uses ArcFace embeddings)
app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(320, 320))

embeddings = []
names = []

persons = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
print(f"Found {len(persons)} people: {persons}\n")

for person in persons:
    person_dir = os.path.join(DATA_DIR, person)
    images = [f for f in os.listdir(person_dir) if f.endswith(".jpg")]
    count = 0

    for img_file in images:
        img = cv2.imread(os.path.join(person_dir, img_file))
        if img is None:
            continue

        # Upscale for better detection
        img_up = cv2.resize(img, (320, 320))
        faces = app.get(img_up)

        if len(faces) > 0:
            embeddings.append(faces[0].embedding)
            names.append(person)
            count += 1

    print(f"  {person}: {count} embeddings extracted from {len(images)} images")

embeddings = np.array(embeddings)

# Normalize embeddings
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / norms

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump({"embeddings": embeddings, "names": names}, f)

print(f"\nSaved {len(embeddings)} embeddings to {OUTPUT_FILE}")
print(f"People: {list(set(names))}")
