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
import socket  # <--- NEW: For UDP Communication
from datetime import datetime

app = Flask(__name__)

# --- CONFIGURATION ---
# ESP32 IP (Use the IP Address only, no "http://")
ESP32_IP = "10.72.228.7" 
ESP32_UDP_PORT = 8888
ESP32_HTTP_PORT = 80 # Assuming your web server is on port 80

# UDP Socket Setup
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(0.05) # 50ms timeout for ACK (Very fast)

# Command Mapping (Python String -> ESP32 Char)
CMD_MAP = {
    "start": "1",
    "stop": "2",
    "emergency": "3",
    "reset": "4"
}

# Fix for cv2.data attribute error
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
    print("ERROR: Trainer not found!")

# --- GLOBALS ---
registration_mode = False
registration_id = 0
registration_count = 0
registration_limit = 30
training_in_progress = False
model_trained = False

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
            print(f"Skipping file {imagePath}: {e}")
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
    training_in_progress = True
    print("Training started...")
    
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
            try:
                os.remove('trainer/trainer.yml')
            except:
                pass
        model_trained = False
        print("No data to train. Model cleared.")
        
    training_in_progress = False

# Global Status
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

cap = cv2.VideoCapture(0)

# Optimization: Pre-render black frame
blank_image = np.zeros((360, 640, 3), np.uint8)
cv2.putText(blank_image, "KAMERA OFF / EMERGENCY", (150, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
ret, buffer = cv2.imencode('.jpg', blank_image)
BLACK_FRAME_BYTES = buffer.tobytes()

# --- UDP HELPER FUNCTION ---
def send_udp_command(cmd_str):
    """Sends command via UDP with retry logic for reliability"""
    if cmd_str not in CMD_MAP:
        print(f"Unknown command: {cmd_str}")
        return False

    cmd_char = CMD_MAP[cmd_str]
    cmd_bytes = cmd_char.encode()
    
    # Try sending up to 3 times
    for attempt in range(3):
        try:
            sock.sendto(cmd_bytes, (ESP32_IP, ESP32_UDP_PORT))
            
            # Wait for ACK
            data, addr = sock.recvfrom(1024)
            response = data.decode()
            
            # Check for ACK
            if f"ACK:{cmd_char}" in response:
                return True # Success
        except socket.timeout:
            continue # Retry
        except Exception as e:
            print(f"UDP Error: {e}")
            return False
            
    return False # Failed after retries

def send_esp32(cmd):
    start_time = time.time()
    success = False
    duration_ms = 0
    internal_latency = None

    try:
        # 1. SEND COMMAND VIA UDP (Fast Path)
        # This will return typically in < 5ms
        success = send_udp_command(cmd)
        
        # Calculate duration strictly for the command transmission
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        print(f"Sent UDP: {cmd} (took {duration_ms:.2f} ms) | Success: {success}")

        # 2. FETCH STATUS VIA HTTP (Slow Path - Logging only)
        # We do this AFTER the command so the machine has already reacted
        if success:
            try:
                # Use HTTP to get detailed status logic
                s = requests.get(f"http://{ESP32_IP}:{ESP32_HTTP_PORT}/status", timeout=0.5)
                if s.status_code == 200:
                    jd = s.json()
                    internal_latency = jd.get('last_internal_latency_us', None)
            except Exception:
                pass # Don't fail the whole operation if just logging fails

    except Exception as e:
        print("ESP32 Error:", e)
    
    finally:
        # Log to CSV
        try:
            _log_timing_csv('WebApp', cmd, duration_ms, success, internal_latency)
        except Exception:
            pass

# --- CSV Logging ---
_csv_lock = threading.Lock()
_csv_path = 'timings.csv'

def _log_timing_csv(source, cmd, duration_ms, success, internal_latency_us=None):
    header = ['timestamp', 'source', 'command', 'duration_ms', 'success', 'internal_latency_us']
    row = [datetime.utcnow().isoformat(timespec='milliseconds') + 'Z', source, cmd, '' if duration_ms is None else f"{duration_ms:.2f}", str(bool(success)), '' if internal_latency_us is None else str(internal_latency_us)]
    with _csv_lock:
        write_header = False
        if not os.path.exists(_csv_path):
            write_header = True
        with open(_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)

def stop_camera():
    system_status["camera_on"] = False

def start_camera():
    system_status["camera_on"] = True

def generate_frames():
    global registration_mode, registration_count
    while True:
        success, frame = cap.read()
        if not success:
            continue

        if not system_status["camera_on"]:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + BLACK_FRAME_BYTES + b'\r\n')
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(30, 30))

        detected = False
        authorized_name = "UNKNOWN"
        
        for (x, y, w, h) in faces:
            # Registration Logic
            if registration_mode and registration_count < registration_limit:
                if not os.path.exists('dataset'):
                    os.makedirs('dataset')
                cv2.imwrite(f"dataset/User.{registration_id}.{registration_count+1}.jpg", gray[y:y+h,x:x+w])
                registration_count += 1
                cv2.putText(frame, f"CAPTURING: {registration_count}/{registration_limit}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if not training_in_progress and model_trained:
                try:
                    id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                    
                    if confidence < 60:
                        name = f"User {id}"
                        color = (0, 255, 0)
                        detected = True
                        authorized_name = name
                        system_status["last_operator"] = name
                    else:
                        name = "UNKNOWN"
                        color = (0, 0, 255)
                except Exception as e:
                    name = "ERROR"
                    color = (0, 0, 255)
            else:
                name = "TRAINING..." if training_in_progress else "NO MODEL"
                color = (255, 255, 0)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if detected:
            system_status["authorized"] = True
            system_status["name"] = authorized_name
        else:
            system_status["authorized"] = False
            system_status["name"] = "UNKNOWN"

        if registration_mode and registration_count >= registration_limit:
             registration_mode = False

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_status')
def check_status():
    try:
        # HTTP is fine for periodic status polling
        r = requests.get(f"http://{ESP32_IP}:{ESP32_HTTP_PORT}/status", timeout=0.5)
        if r.status_code == 200:
            esp_data = r.json()
            system_status["plc_running"] = esp_data["running"]
            system_status["plc_stopped"] = esp_data["stopped"]
            system_status["plc_emergency"] = esp_data["emergency"]
            system_status["plc_connected"] = True
        else:
            system_status["plc_connected"] = False
    except:
        system_status["plc_connected"] = False

    return jsonify(system_status)

@app.route('/command/<cmd>')
def command(cmd):
    global system_status
    
    if cmd == "emergency":
        system_status["emergency"] = True
        stop_camera() 
        send_esp32("emergency")
        
    elif cmd == "stop":
        send_esp32("stop")

    elif cmd == "reset":
        system_status["emergency"] = False
        start_camera()
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
                try:
                    os.remove(os.path.join('dataset', f))
                except:
                    pass
    return jsonify({"status": "deleted", "id": user_id})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)