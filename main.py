from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import shutil
import os
import cv2
import queue
import uvicorn
from img import register_from_images
from rec import recognize,frame_queue, exit_event
import threading

app = FastAPI()

def run_gui():
    while not exit_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
            cv2.imshow("AI Attendance - Press 'q' to Quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_event.set()
                break
        except queue.Empty:
            continue
    cv2.destroyAllWindows()

@app.get("/")
def home():
    return {"message": "AI Attendance System Backend Running"}

@app.post("/register")
async def register_user(
    user_id: str = Form(...),
    name: str = Form(...),
    files: list[UploadFile] = File(...)
):
    upload_folder = f"dataset/{user_id}"
    os.makedirs(upload_folder, exist_ok=True)

    for file in files:
        file_path = os.path.join(upload_folder, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

    register_from_images(user_id, name, upload_folder)
    return {"message": f"User '{name}' registered successfully."}

@app.get("/recognize/")
def start_recognition():
    thread = threading.Thread(target=recognize)
    thread.start()
    return {"message": "Face recognition started (check camera window)"}

if __name__ == "__main__":
    # Start GUI in main thread
    import sys
    if "render" not in sys.argv:
        # Local mode with GUI
        gui_thread = threading.Thread(target=run_gui)
        gui_thread.start()

        uvicorn.run(app, host="0.0.0.0", port=8000)

        exit_event.set()
        gui_thread.join()
    else:
        # Headless server mode (e.g. on Render)
        uvicorn.run(app, host="0.0.0.0", port=8000)