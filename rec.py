import face_recognition
import numpy as np
import cv2
import os
import json
from datetime import datetime
import queue
import threading
#from deepface import DeepFace

frame_queue = queue.Queue(maxsize=1)
exit_event = threading.Event()

DATA_DIR = "data"
FACES_DIR = os.path.join(DATA_DIR, "known_faces")
USER_FILE = os.path.join(DATA_DIR, "users.json")
THRESHOLD = 0.50

def load_known_faces():
    encodings = []
    ids = []

    for file in os.listdir(FACES_DIR):
        if file.endswith(".npy"):
            user_id = file.replace(".npy", "")
            user_encodings = np.load(os.path.join(FACES_DIR, file))

            for enc in user_encodings:
                encodings.append(enc)
                ids.append(user_id)

    with open(USER_FILE, "r") as f:
        id_name_map = json.load(f)

    return encodings, ids, id_name_map

def mark_attendance(name):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("attendance.csv", "a") as f:
        f.write(f"{name},{now}\n")
    print(f"[ATTENDANCE] {name} marked present at {now}")

def recognize():
    known_encodings, ids, id_name_map = load_known_faces()
    print("[INFO] Starting camera...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open video device.")
        return
    
    marked_users = set()

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding, box in zip(encodings, boxes):
            distances = face_recognition.face_distance(known_encodings, encoding)
            min_index = np.argmin(distances)
            min_distance = distances[min_index]

            #try:
             #   result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
              #  mood = result[0]['dominant_emotion']
               # cv2.putText(frame, f'Mood: {mood}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            #except Exception as e:
             #   print("Face not detected:", e)
    
            #cv2.imshow("Mood Detection", frame)

            if min_distance < THRESHOLD:
                user_id = ids[min_index]
                name = id_name_map.get(user_id, "Unknown")

                if name not in marked_users:
                    mark_attendance(name)
                    marked_users.add(name)

                top, right, bottom, left = box
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({min_distance:.2f})", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                top, right, bottom, left = box
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass
                
        if frame is None or frame.size == 0:
            print("[ERROR] Empty frame received.")
            continue

        cv2.imshow("AI Attendance - Press 'q' to Quit", frame)
        #cv2.imshow("AI Attendance - Press 'q' to Quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize()
