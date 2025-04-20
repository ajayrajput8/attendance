import face_recognition
import numpy as np
import os
import json

DATA_DIR = "data"
FACES_DIR = os.path.join(DATA_DIR, "known_faces")
USER_FILE = os.path.join(DATA_DIR, "users.json")

os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as f:
        json.dump({}, f)

def register_from_images(user_id, user_name, folder_path):
    encodings = []

    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png",".webp")):
            image_path = os.path.join(folder_path, file)
            image = face_recognition.load_image_file(image_path)
            boxes = face_recognition.face_locations(image)

            if len(boxes) == 1:
                encoding = face_recognition.face_encodings(image, boxes)[0]
                encodings.append(encoding)
                print(f"[INFO] Processed {file}")
            else:
                print(f"[WARNING] Skipped {file}: Face not clear or multiple faces")

    if encodings:
        encodings_array = np.array(encodings)
        np.save(os.path.join(FACES_DIR, f"{user_id}.npy"), encodings_array)

        with open(USER_FILE, "r+") as f:
            users = json.load(f)
            users[user_id] = user_name
            f.seek(0)
            json.dump(users, f, indent=4)

        print(f"[SUCCESS] Registered '{user_name}' with {len(encodings)} images.")
    else:
        print("[ERROR] No valid encodings found. Check the images.")

# Example usage
if __name__ == "__main__":
    user_id = input("Enter User ID: ")
    name = input("Enter Name: ")
    folder_path = f"/Users/ajaysinghrajput/Desktop/Ai Based Minor/attendance/dataset/{user_id}"
    register_from_images(user_id, name, folder_path)
