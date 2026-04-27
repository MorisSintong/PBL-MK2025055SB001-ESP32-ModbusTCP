from flask import Flask, render_template, Response, jsonify
import cv2
import os
import sys
import requests
import numpy as np
from PIL import Image
import time
import csv
import threading
from datetime import datetime

app = Flask(__name__)

# --- CONFIGURATION ---
ESP32_IP = "http://10.251.64.200" 
# FIX 1: Reduce JPEG Quality to 65 (Default is 95). 
# This makes the video stream much lighter and faster over WiFi.
JPEG_QUALITY = [int(cv2.IMWRITE_JPEG_QUALITY), 65] 

# --- CASCADE LOADING ---
if hasattr(cv2, 'data'):
    FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
else:
    FACE_CASCADE_PATH = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascades', 'haarcascade_frontalface_default.xml')
    if not os.path.exists(FACE_CASCADE_PATH):
        FACE_CASCADE_PATH = os.path.join(sys.prefix, 'Library', 'etc', 'haarcascades', 'haarcascade_frontalface_default.xml')

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

if os.path.exists('trainer/trainer.yml'):
    recognizer.read('trainer/trainer.yml')
    model_trained = True
    print("Model Loaded.")
else:
    model_trained = False
    print("WARNING: Trainer not found.")

# --- VIDEO STREAM CLASS (Optimized) ---
class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

vs = VideoStream(0).start()

blank_image = np.zeros((360, 640, 3), np.uint8)
cv2.putText(blank_image, "KAMERA OFF / EMERGENCY", (150, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
ret, buffer = cv2.imencode('.jpg', blank_image, JPEG_QUALITY)
BLACK_FRAME_BYTES = buffer.tobytes()

# --- GLOBALS ---
registration_mode = False
registration_id = 0
registration_count = 0
registration_limit = 30
training_in_progress = False

system_status = {
    "authorized": False,
    "name": "UNKNOWN",
    "emergency": False,
    "camera_on": True,
    "last_operator": "-",
    "plc_running": False,
    "plc_stopped": False,
    "plc_emergency": False,
    "plc_connected": False
}

# --- FIX 2: BACKGROUND STATUS POLLER ---
# This thread talks to ESP32 silently. 
# The web interface reads the 'system_status' variable INSTANTLY.
def poll_esp32_status():
    while True:
        try:
            # We poll every 0.5 seconds
            r = requests.get(f"{ESP32_IP}/status", timeout=0.4)
            if r.status_code == 200:
                esp_data = r.json()
                system_status["plc_running"] = esp_data.get("running", False)
                system_status["plc_stopped"] = esp_data.get("stopped", False)
                system_status["plc_emergency"] = esp_data.get("emergency", False)
                system_status["plc_connected"] = True
            else:
                system_status["plc_connected"] = False
        except:
            system_status["plc_connected"] = False
            
        time.sleep(0.5)

# Start the poller in the background
polling_thread = threading.Thread(target=poll_esp32_status)
polling_thread.daemon = True
polling_thread.start()

# --- HELPER FUNCTIONS ---
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        try:
            PIL_img = Image.open(imagePath).convert('L') 
            img_numpy = np.array(PIL_img,'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faceSamples.append(img_numpy)
            ids.append(id)
        except Exception as e:
            pass
    return faceSamples,ids

def get_registered_users():
    users = set()
    if os.path.exists('dataset'):
        for f in os.listdir('dataset'):
            if f.startswith("User.") and f.endswith(".jpg"):
                parts = f.split('.')
                if len(parts) >= 3:
                    try:
                        users.add(int(parts[1]))
                    except:
                        pass
    return sorted(list(users))

def train_face_model():
    global recognizer, training_in_progress, model_trained
    
    if training_in_progress: return

    training_in_progress = True
    print("Training started...")
    
    try:
        path = 'dataset'
        if not os.path.exists(path):
            os.makedirs(path)
            
        faces, ids = getImagesAndLabels(path)
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        if len(ids) > 0:
            recognizer.train(faces, np.array(ids))
            if not os.path.exists('trainer'):
                os.makedirs('trainer')
            recognizer.write('trainer/trainer.yml')
            model_trained = True
            print("Training finished.")
        else:
            if os.path.exists('trainer/trainer.yml'):
                try: os.remove('trainer/trainer.yml')
                except: pass
            model_trained = False
            print("No data. Model cleared.")
            
    except Exception as e:
        print(f"TRAIN ERROR: {e}")
        
    finally:
        training_in_progress = False

# --- NETWORK COMMANDS ---
_csv_lock = threading.Lock()
_csv_path = 'timings.csv'

def _log_timing_csv(source, cmd, duration_ms, success):
    header = ['timestamp', 'source', 'command', 'duration_ms', 'success']
    row = [datetime.utcnow().isoformat(timespec='milliseconds') + 'Z', source, cmd, '' if duration_ms is None else f"{duration_ms:.2f}", str(bool(success))]
    with _csv_lock:
        write_header = not os.path.exists(_csv_path)
        with open(_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header: writer.writerow(header)
            writer.writerow(row)

def _send_esp32_task(cmd):
    start_time = time.time()
    success = False
    duration_ms = None
    try:
        r = requests.get(f"{ESP32_IP}/action?type={cmd}", timeout=1.0)
        duration_ms = (time.time() - start_time) * 1000
        success = (r.status_code == 200)
        print(f"[BG] Sent ESP32: {cmd}")
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        print(f"[BG] Error: {e}")
    finally:
        try: _log_timing_csv('WebApp', cmd, duration_ms, success)
        except: pass

def send_esp32(cmd):
    t = threading.Thread(target=_send_esp32_task, args=(cmd,))
    t.daemon = True
    t.start()

# --- USER DETECTION COIL CONTROL ---
_last_detection_state = None

def _send_user_detection_task(detected):
    """Send user detection status to ESP32 to write Modbus coil"""
    try:
        r = requests.get(f"{ESP32_IP}/user_detection?detected={'1' if detected else '0'}", timeout=0.5)
        if r.status_code == 200:
            print(f"[BG] User detection sent: {'DETECTED' if detected else 'NOT DETECTED'}")
    except Exception as e:
        print(f"[BG] User detection error: {e}")

def send_user_detection(detected):
    """Send user detection status only if state changed"""
    global _last_detection_state
    if _last_detection_state != detected:
        _last_detection_state = detected
        t = threading.Thread(target=_send_user_detection_task, args=(detected,))
        t.daemon = True
        t.start()

# --- GENERATE FRAMES (Optimized) ---
def generate_frames():
    global registration_mode, registration_count
    
    frame_count = 0
    SKIP_FRAMES = 3 
    
    last_faces = []
    detected = False
    authorized_name = "UNKNOWN"
    
    while True:
        # Limit loop speed to ~30 FPS
        time.sleep(0.03) 
        
        raw_frame = vs.read()
        if raw_frame is None:
            continue

        # Prevent Shadowing
        frame = raw_frame.copy() 

        if not system_status["camera_on"]:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + BLACK_FRAME_BYTES + b'\r\n')
            continue
        
        # Resize for faster detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        if frame_count % SKIP_FRAMES == 0:
            faces_rects = face_cascade.detectMultiScale(gray_small, 1.2, 7, minSize=(30, 30))
            
            # Filter: Largest face only
            if len(faces_rects) > 0:
                faces_rects = sorted(list(faces_rects), key=lambda x: x[2] * x[3], reverse=True)[:1]

            last_faces = [(int(x*2), int(y*2), int(w*2), int(h*2)) for (x,y,w,h) in faces_rects]
            
            detected = False
            authorized_name = "UNKNOWN"
            
            for (x, y, w, h) in last_faces:
                face_roi = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                
                # Registration
                if registration_mode and registration_count < registration_limit:
                    if not os.path.exists('dataset'): os.makedirs('dataset')
                    cv2.imwrite(f"dataset/User.{registration_id}.{registration_count+1}.jpg", face_roi)
                    registration_count += 1
                
                # Recognition
                if not training_in_progress and model_trained:
                    try:
                        id, confidence = recognizer.predict(face_roi)
                        if confidence < 60: 
                            detected = True
                            authorized_name = f"User {id}"
                            system_status["last_operator"] = authorized_name
                        else:
                            authorized_name = "UNKNOWN"
                    except: pass
        
        frame_count += 1
        
        for (x, y, w, h) in last_faces:
            if registration_mode:
                cv2.putText(frame, f"REG: {registration_count}/{registration_limit}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            color = (0, 255, 0) if detected else (0, 0, 255)
            display_name = authorized_name if detected else "UNKNOWN"
            if training_in_progress: display_name = "TRAINING..."
            if not model_trained and not training_in_progress: display_name = "NO MODEL"

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, display_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        system_status["authorized"] = detected
        system_status["name"] = authorized_name

        # Send user detection status to ESP32 (writes Modbus coil)
        send_user_detection(detected)

        if registration_mode and registration_count >= registration_limit:
             registration_mode = False

        # Encode with LOWER QUALITY for SPEED
        ret, buffer = cv2.imencode('.jpg', frame, JPEG_QUALITY)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_status')
def check_status():
    # FIX: INSTANT RESPONSE
    # No requests.get here! We just return the variable that the background thread updates.
    return jsonify(system_status)

@app.route('/command/<cmd>')
def command(cmd):
    global system_status
    if cmd == "emergency":
        system_status["emergency"] = True
        system_status["camera_on"] = False
        send_esp32("emergency")
    elif cmd == "stop":
        send_esp32("stop")
    elif cmd == "reset":
        system_status["emergency"] = False
        system_status["camera_on"] = True
        send_esp32("reset")
    elif cmd == "start":
        if system_status["authorized"] and not system_status["emergency"]:
            send_esp32("start")
    return "OK"

@app.route('/start_register/<int:user_id>')
def start_register(user_id):
    global registration_mode, registration_id, registration_count
    registration_id = user_id
    registration_count = 0
    registration_mode = True
    return jsonify({"status": "started", "id": user_id})

@app.route('/train_model')
def train_model_route():
    # Keep Synchronous so UI buttons behave correctly
    if not training_in_progress:
        train_face_model()
        return jsonify({"status": "trained"})
    else:
        return jsonify({"status": "busy"})

@app.route('/get_registration_status')
def get_registration_status():
    return jsonify({
        "mode": registration_mode,
        "count": registration_count,
        "limit": registration_limit,
        "training": training_in_progress
    })

@app.route('/get_users')
def get_users_route():
    return jsonify(get_registered_users())

@app.route('/delete_user/<int:user_id>')
def delete_user_route(user_id):
    if os.path.exists('dataset'):
        for f in os.listdir('dataset'):
            if f.startswith(f"User.{user_id}."):
                try: os.remove(os.path.join('dataset', f))
                except: pass
    return jsonify({"status": "deleted", "id": user_id})

if __name__ == "__main__":
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        vs.stop()