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
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn.functional as F

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

# === FACE RECOGNITION SETUP ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize MTCNN for face detection
mtcnn = MTCNN(
    image_size=160,
    margin=0,
    min_face_size=60,
    thresholds=[0.6, 0.7, 0.7],
    device=device
)

# Initialize FaceNet for face recognition
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load or create embeddings database
embeddings_file = "embeddings.pkl"
if os.path.exists(embeddings_file):
    with open(embeddings_file, 'rb') as f:
        embeddings_db = pickle.load(f)
else:
    embeddings_db = {}

def get_embedding(face_img):
    """Convert face image to FaceNet embedding with proper tensor dimensions"""
    try:
        # Convert BGR to RGB and resize
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img_pil = Image.fromarray(face_img_rgb)
        
        # Get face tensor and ensure proper dimensions
        face_tensor = mtcnn(face_img_pil)
        
        if face_tensor is not None:
            # Add batch dimension if needed and move to device
            if face_tensor.dim() == 3:
                face_tensor = face_tensor.unsqueeze(0)
            face_tensor = face_tensor.to(device)
            
            # Get embedding and normalize
            embedding = resnet(face_tensor)
            return F.normalize(embedding, p=2, dim=1)[0].detach().cpu()
    except Exception as e:
        print(f"Embedding error: {e}")
    return None

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

# === OBJECT DETECTION SETUP ===
yolo_device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt").to(yolo_device)

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

                    font_scale = 2.0 if label.lower() == 'person' else 0.7
                    thickness = 3 if label.lower() == 'person' else 2

                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        y_offset = 80
        for cls, count in class_counts.items():
            font_scale = 2 
            thickness = 3 

            cv2.putText(frame, f"{cls.capitalize()}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
            y_offset += int(30 * font_scale)

        _, buffer = cv2.imencode('.jpg', frame)
        await manager.send_bytes(buffer.tobytes(), websocket)
        await asyncio.sleep(0.05)

async def recognition_frames(websocket: WebSocket):
    last_sound_time = 0
    sound_cooldown = 10
    recognition_threshold = 0.7  # Similarity threshold for face recognition
    
    cap = cv2.VideoCapture(0)
    global webcam_active
    webcam_active = True
    
    while webcam_active:
        success, frame = cap.read()
        if not success:
            break

        # Detect faces with MTCNN
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(frame_rgb)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face_img = frame[y1:y2, x1:x2]
                
                if face_img.size == 0:
                    continue
                
                # Get embedding for the detected face
                embedding = get_embedding(face_img)
                
                if embedding is not None:
                    min_dist = float('inf')
                    identity = "Unknown"
                    
                    for name, known_emb in embeddings_db.items():
                        dist = (embedding - known_emb.to(device)).norm().item()
                        if dist < min_dist and dist < recognition_threshold:
                            min_dist = dist
                            identity = name
                    
                    # Draw results
                    color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{identity}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Alert for unknown faces
                    if identity == "Unknown" and (time.time() - last_sound_time) > sound_cooldown:
                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                            cv2.imwrite(tmp_file.name, frame)
                            safe_send_telegram_alert(
                                TELEGRAM_BOT_TOKEN,
                                TELEGRAM_CHAT_ID,
                                "Alert: Unknown person detected!",
                                image_path=tmp_file.name
                            )
                        play_sound_async("alert-sound.wav")
                        last_sound_time = time.time()

        _, buffer = cv2.imencode('.jpg', frame)
        await manager.send_bytes(buffer.tobytes(), websocket)
        await asyncio.sleep(0.05)
    
    cap.release()
    webcam_active = False

async def capture_feed(websocket: WebSocket, id: str, name: str, age: str):
    cap = cv2.VideoCapture(0)
    sampleNum = 0
    min_samples = 30  
    
    # Create user directory if it doesn't exist
    user_dir = os.path.join('dataset', name)
    os.makedirs(user_dir, exist_ok=True)
    
    while sampleNum < min_samples:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect faces with MTCNN
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(frame_rgb)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face_img = frame[y1:y2, x1:x2]
                
                if face_img.size == 0:
                    continue
                
                # Save face sample
                sampleNum += 1
                save_path = os.path.join(user_dir, f"{id}_{sampleNum}.jpg")
                cv2.imwrite(save_path, face_img)
                
                # Draw bounding box and sample count
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Samples: {sampleNum}/{min_samples}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if sampleNum >= min_samples:
                    break

        _, buffer = cv2.imencode('.jpg', frame)
        await manager.send_bytes(buffer.tobytes(), websocket)
        await manager.send_text(json.dumps({"samples": sampleNum}), websocket)
        await asyncio.sleep(0.1)
            
    cap.release()
    play_sound_async("capture-complete.wav")
    insert_or_update(id, name, age)
    
    # Train the model after capturing new data
    await train_model()
    await manager.send_text(json.dumps({"status": "complete"}), websocket)

async def train_model():
    """Train the FaceNet model by generating embeddings for all faces in the dataset"""
    global embeddings_db
    
    new_embeddings_db = {}
    dataset_path = 'dataset'
    
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue
        
        embeddings = []
        
        for img_file in os.listdir(person_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(person_path, img_file)
                img = cv2.imread(img_path)
                
                if img is not None:
                    embedding = get_embedding(img)
                    if embedding is not None:
                        embeddings.append(embedding)
        
        if embeddings:
            avg_embedding = torch.stack(embeddings).mean(dim=0)
            new_embeddings_db[person_name] = avg_embedding
    
    # Update the embeddings database
    embeddings_db = new_embeddings_db
    
    # Save embeddings to file
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings_db, f)
    
    return {"status": "Training complete! Generated face embeddings."}

# === ROUTES ===
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index5.html", {"request": request})

@app.post("/capture")
def capture(id: str = Form(...), name: str = Form(...), age: str = Form(...)):
    return RedirectResponse("/", status_code=303)

@app.get("/train")
async def train_route():
    return await train_model()

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