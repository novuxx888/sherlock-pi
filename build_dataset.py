import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import sys

DATA_PATH = os.path.expanduser("~/synthetic_data")
OUTPUT_FILE = "known_faces_mp.pkl"

mp_face_mesh = mp.solutions.face_mesh
# Keep confidence at 0.3 to be safe
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, 
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3
)

known_fingerprints = []
known_names = []

print(f"Training on NEW synthetic data...")

for person_name in sorted(os.listdir(DATA_PATH)):
    person_dir = os.path.join(DATA_PATH, person_name)
    if not os.path.isdir(person_dir): continue
    
    print(f"\nLearning: {person_name}")
    success_count = 0
    total_count = 0
    
    for image_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, image_name)
        image = cv2.imread(img_path)
        if image is None: continue
        
        total_count += 1
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_img)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                fp = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark]).flatten()
                known_fingerprints.append(fp)
                known_names.append(person_name)
                success_count += 1
        
        if total_count % 20 == 0:
            sys.stdout.write(f"\r  Processed {total_count} images...")
            sys.stdout.flush()

    print(f"\r  {person_name}: Successfully mapped {success_count}/{total_count}")

if len(known_fingerprints) > 0:
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump({"fingerprints": np.array(known_fingerprints), "names": known_names}, f)
    print(f"\n[SUCCESS] Brain saved to {OUTPUT_FILE}")
else:
    print("\n[ERROR] No faces detected!")
