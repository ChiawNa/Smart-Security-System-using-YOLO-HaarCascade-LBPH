from fastapi import FastAPI, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import cv2
import os
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import sqlite3
import time
import requests
from io import BytesIO
from threading import Thread
from playsound import playsound
import threading
import tempfile
import asyncio
import base64
import json

# === FASTAPI SETUP ===
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# === TELEGRAM SETUP ===
TELEGRAM_BOT_TOKEN = '8154457175:AAFjwrR_d4BO_UCoV94CHtuavsIX4HbDDXE'
TELEGRAM_CHAT_ID = '-1002409276148'
last_alert_time = 0
alert_cooldown = 30

# Camera control variables
rtsp_active = False
webcam_active = False

def play_sound_async(filename):
    try:
        sound_path = os.path.abspath(filename)
        threading.Thread(target=playsound, args=(sound_path,), daemon=True).start()
    except Exception as e:
        print("Sound playback error:", e)

def send_telegram_alert(bot_token, chat_id, message, image_path=None):
    try:
        # Send text message
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {"chat_id": chat_id, "text": message}
        response = requests.post(url, data=data)
        if not response.ok:
            print("Failed to send message:", response.text)

        # Send image if provided
        if image_path and os.path.exists(image_path):
            url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
            with open(image_path, "rb") as photo:
                files = {"photo": photo}
                data = {"chat_id": chat_id}
                response = requests.post(url, files=files, data=data)
                if not response.ok:
                    print("Failed to send photo:", response.text)
        elif image_path:
            print("Image file not found:", image_path)
    except Exception as e:
        print("Telegram Error:", e)

def safe_send_telegram_alert(bot_token, chat_id, message, image_path=None):
    global last_alert_time
    if time.time() - last_alert_time > alert_cooldown:
        send_telegram_alert(bot_token, chat_id, message, image_path)
        last_alert_time = time.time()

# === FACE RECOGNITION SETUP ===
def get_face_detector():
    cascade_paths = [
        "haarcascade_frontalface_default.xml",
    ]
    
    detectors = [cv2.CascadeClassifier(path) for path in cascade_paths if os.path.exists(path)]
    if not detectors:
        raise Exception("No face detection cascades found!")
    
    def detect_faces(gray_img):
        faces = []
        for detector in detectors:
            detected = detector.detectMultiScale(
                gray_img,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            faces.extend(detected)
        return faces
    
    return detect_faces

face_detector = get_face_detector()

training_file = "recognizer/trainingdata.yml"
data_file = "recognizer/training_data.npz"

# Initialize LBPH recognizer with optimized parameters
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=2,
    neighbors=16,
    grid_x=8,
    grid_y=8,
    threshold=80.0
)

if os.path.exists(training_file):
    recognizer.read(training_file)

# === OBJECT DETECTION SETUP ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt").to(device)

# === DATABASE SETUP ===
def insert_or_update(Id, Name, Age):
    conn = sqlite3.connect("sqlite.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS students (
                        Id TEXT PRIMARY KEY,
                        Name TEXT,
                        age TEXT)''')
    cursor.execute("SELECT * FROM students WHERE Id=?", (Id,))
    if cursor.fetchone():
        cursor.execute("UPDATE students SET Name=?, age=? WHERE Id=?", (Name, Age, Id))
    else:
        cursor.execute("INSERT INTO students (Id, Name, age) VALUES (?, ?, ?)", (Id, Name, Age))
    conn.commit()
    conn.close()

def get_profile(id):
    conn = sqlite3.connect("sqlite.db")
    cursor = conn.execute("SELECT * FROM students WHERE id=?", (id,))
    profile = cursor.fetchone()
    conn.close()
    return profile

# === PREPROCESSING FUNCTIONS ===
def preprocess_face(face_img):
    face_img = cv2.equalizeHist(face_img)
    face_img = cv2.GaussianBlur(face_img, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face_img = clahe.apply(face_img)
    
    return face_img

# === RTSP STREAM SETUP ===
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|buffer_size;1024"

class RTSPCamera:
    def __init__(self, url):
        self.url = url
        self.cap = None
        self.frame = None
        self.running = False
        self.thread = None

    def start(self):
        if not self.running:
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            self.running = True
            self.thread = Thread(target=self.update, daemon=True)
            self.thread.start()
            global rtsp_active
            rtsp_active = True

    def update(self):
        while self.running:
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            self.cap.grab()
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def read(self):
        return self.frame if self.running else None

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        global rtsp_active
        rtsp_active = False

rtsp_stream = RTSPCamera("rtsp://Tp-200:Kangcn_2001@192.168.253.205:554/stream1")

# === WEBSOCKET MANAGER ===
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_bytes(self, data: bytes, websocket: WebSocket):
        try:
            await websocket.send_bytes(data)
        except Exception as e:
            print(f"Error sending bytes: {e}")
            self.disconnect(websocket)

    async def send_text(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            print(f"Error sending text: {e}")
            self.disconnect(websocket)

manager = ConnectionManager()

# === STREAMING FUNCTIONS ===
async def detection_frames(websocket: WebSocket):
    confidence_threshold = 0.6
    target_classes = ['person', 'cat', 'dog']
    
    while rtsp_active:
        frame = rtsp_stream.read()
        if frame is None:
            await asyncio.sleep(0.1)
            continue

        class_counts = {cls: 0 for cls in target_classes}

        results = model(frame, stream=True)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                if conf >= confidence_threshold and label.lower() in target_classes:
                    class_counts[label.lower()] += 1

                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Bigger font for person, smaller for others
                    font_scale = 2.0 if label.lower() == 'person' else 0.7
                    thickness = 3 if label.lower() == 'person' else 2

                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        y_offset = 80
        for cls, count in class_counts.items():
            # Bigger font for person count
            font_scale = 2 
            thickness = 3 

            cv2.putText(frame, f"{cls.capitalize()}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
            y_offset += int(30 * font_scale)  # Adjust vertical spacing based on font size

        _, buffer = cv2.imencode('.jpg', frame)
        await manager.send_bytes(buffer.tobytes(), websocket)
        await asyncio.sleep(0.05)  # Control frame rate

async def recognition_frames(websocket: WebSocket):
    last_sound_time = 0
    sound_cooldown = 10
    
    def get_dynamic_threshold(face_size):
        base_threshold = 60  # For large faces
        min_threshold = 80   # For small faces
        min_face_size = 100  # Pixels
        max_face_size = 300  # Pixels
        
        if face_size >= max_face_size:
            return base_threshold
        elif face_size <= min_face_size:
            return min_threshold
        else:
            return base_threshold + (min_threshold - base_threshold) * \
                   (1 - (face_size - min_face_size)/(max_face_size - min_face_size))
    
    cap = cv2.VideoCapture(0)
    global webcam_active
    webcam_active = True
    
    while webcam_active:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)

        for (x, y, w, h) in faces:
            face_size = max(w, h)
            face_roi = gray[y:y+h, x:x+w]
            face_roi = preprocess_face(face_roi)
            
            id_, conf_ = recognizer.predict(face_roi)
            dynamic_threshold = get_dynamic_threshold(face_size)
            
            if conf_ < dynamic_threshold:
                profile = get_profile(id_)
                if profile:
                    cv2.putText(frame, f"{profile[1]}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 127), 2)
                else:
                    cv2.putText(frame, "Unknown", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Draw bounding box (already done above)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                current_time = time.time()
                if current_time - last_sound_time > sound_cooldown:
                    alert_frame = frame.copy()
                    
                    padding = 20
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    y2 = min(frame.shape[0], y + h + padding)
                    cropped_face = frame[y1:y2, x1:x2]
                    
                    # Save current frame to a temp file
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                        temp_image_path = tmp_file.name
                        cv2.imwrite(temp_image_path, frame)

                    # Pass the image path to your alert function
                    safe_send_telegram_alert(
                        TELEGRAM_BOT_TOKEN, 
                        TELEGRAM_CHAT_ID, 
                        "Alert: Unknown person detected!",
                        image_path=temp_image_path
                    )
                    play_sound_async("alert-sound.wav")
                    last_sound_time = current_time
                    # Clean up the temp file after sending
                    try:
                        os.unlink(temp_image_path)
                    except:
                        pass
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        await manager.send_bytes(buffer.tobytes(), websocket)
        await asyncio.sleep(0.05)  # Control frame rate
    
    cap.release()
    webcam_active = False

async def capture_feed(websocket: WebSocket, id: str, name: str, age: str):
    cap = cv2.VideoCapture(0)
    sampleNum = 0
    min_face_size = 100
    
    while sampleNum < 50:  # Ensure we capture exactly 50 samples
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)

        for (x, y, w, h) in faces:
            if w >= min_face_size and h >= min_face_size:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                sampleNum += 1
                face_roi = gray[y:y+h, x:x+w]
                face_roi = preprocess_face(face_roi)
                cv2.imwrite(f"dataset/user.{id}.{sampleNum}.jpg", face_roi)
                
                # Only show sample number on frame (removed the modal text updates)
                cv2.putText(frame, f"Samples: {sampleNum}/50", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if sampleNum >= 50:
                    break

        _, buffer = cv2.imencode('.jpg', frame)
        await manager.send_bytes(buffer.tobytes(), websocket)
        await manager.send_text(json.dumps({"samples": sampleNum}), websocket)
            
        # Small delay to prevent overwhelming the system
        await asyncio.sleep(0.1)
            
    cap.release()
    play_sound_async("capture-complete.wav")
    insert_or_update(id, name, age)  # Insert data after capture is complete
    await manager.send_text(json.dumps({"status": "complete"}), websocket)

# === ROUTES ===
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index_ws.html", {"request": request})

@app.post("/capture")
def capture(id: str = Form(...), name: str = Form(...), age: str = Form(...)):
    # Data is now inserted after capture is complete in capture_feed
    return RedirectResponse("/", status_code=303)

@app.get("/train")
def train_model():
    # Create augmented dataset
    def augment_image(img):
        augmented = []
        # Original
        augmented.append(img)
        # Flip horizontally
        augmented.append(cv2.flip(img, 1))
        # Rotate slightly
        rows, cols = img.shape
        for angle in [-5, 5]:
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            rotated = cv2.warpAffine(img, M, (cols, rows))
            augmented.append(rotated)
        return augmented

    faces, ids = [], []
    image_paths = [os.path.join("dataset", f) for f in os.listdir("dataset")]
    
    for path in image_paths:
        img = Image.open(path).convert("L")
        face_np = np.array(img, np.uint8)
        face_np = preprocess_face(face_np)
        
        # Augment each image
        for augmented_face in augment_image(face_np):
            faces.append(augmented_face)
            ids.append(int(os.path.split(path)[-1].split(".")[1]))

    recognizer.train(faces, np.array(ids))
    recognizer.save(training_file)
    np.savez(data_file, faces=np.array(faces, dtype=object), ids=np.array(ids))

    return {"status": "Training complete with augmentation!"}

@app.get("/start_cameras")
def start_cameras():
    rtsp_stream.start()
    return {"status": "Cameras activated"}

@app.get("/stop_cameras")
def stop_cameras():
    rtsp_stream.stop()
    global webcam_active
    webcam_active = False
    return {"status": "Cameras deactivated"}

# === WEBSOCKET ENDPOINTS ===
@app.websocket("/ws/detection_feed")
async def websocket_detection_feed(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await detection_frames(websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"Detection feed error: {e}")
        manager.disconnect(websocket)

@app.websocket("/ws/recognition_feed")
async def websocket_recognition_feed(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await recognition_frames(websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"Recognition feed error: {e}")
        manager.disconnect(websocket)

@app.websocket("/ws/capture_feed")
async def websocket_capture_feed(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Receive initial parameters
        data = await websocket.receive_text()
        params = json.loads(data)
        await capture_feed(websocket, params["id"], params["name"], params["age"])
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"Capture feed error: {e}")
        manager.disconnect(websocket)