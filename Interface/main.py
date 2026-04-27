import cv2
import requests
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import time
import threading
import os
import sys
import csv
from datetime import datetime

# CSV logging
_csv_lock = threading.Lock()
_csv_path = 'timings.csv'

def _log_timing_csv(source, cmd, duration_ms, success):
    header = ['timestamp', 'source', 'command', 'duration_ms', 'success', 'internal_latency_us']
    row = [datetime.utcnow().isoformat(timespec='milliseconds') + 'Z', source, cmd, '' if duration_ms is None else f"{duration_ms:.2f}", str(bool(success)), '']
    with _csv_lock:
        write_header = False
        if not os.path.exists(_csv_path):
            write_header = True
        with open(_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)

# --- KONFIGURASI ---
ESP32_IP = "http://10.194.229.48"  

# Fix for cv2.data attribute error
if hasattr(cv2, 'data'):
    FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
else:
    # Fallback 1: Inside cv2 package
    FACE_CASCADE_PATH = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascades', 'haarcascade_frontalface_default.xml')
    
    # Fallback 2: Conda Windows path
    if not os.path.exists(FACE_CASCADE_PATH):
        FACE_CASCADE_PATH = os.path.join(sys.prefix, 'Library', 'etc', 'haarcascades', 'haarcascade_frontalface_default.xml')

class FaceControlApp:
    def __init__(self, window):
        self.window = window
        self.window.title("SISTEM KONTROL - POLIBATAM (FINAL)")
        self.window.geometry("900x750")
        self.window.configure(bg="#1a1a1d")

        # Load AI
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        if os.path.exists('trainer/trainer.yml'):
            self.recognizer.read('trainer/trainer.yml')
            print("Model Trainer dimuat.")
        else:
            messagebox.showerror("Error", "File trainer/trainer.yml tidak ditemukan!")
            return

        self.cap = None
        self.is_camera_on = False
        self.is_authorized = False
        self.is_emergency_active = False 
        
        # --- GUI LAYOUT ---
        tk.Label(window, text="MK202505SB001 - PROCESSING SYSTEM", font=("Arial", 20, "bold"), bg="#1a1a1d", fg="white").pack(pady=10)

        # Frame Video
        self.video_frame = tk.Label(window, bg="black", width=85, height=20, relief="sunken", bd=2)
        self.video_frame.pack(pady=5)

        # Status Bar
        self.status_label = tk.Label(window, text="SISTEM STANDBY - SILAKAN BUKA KAMERA", font=("Courier", 14, "bold"), bg="#333", fg="white", width=60, height=2)
        self.status_label.pack(pady=10)

        # Tombol BUKA KAMERA
        self.btn_open_cam = tk.Button(window, text="📷 BUKA KAMERA / SCAN WAJAH", bg="#007bff", fg="white", font=("Arial", 12, "bold"), 
                                      width=40, height=2, command=self.start_camera)
        self.btn_open_cam.pack(pady=5)

        # Frame Tombol Kontrol
        self.btn_frame = tk.Frame(window, bg="#1a1a1d")
        self.btn_frame.pack(pady=15)

        # Tombol-Tombol Mesin
        self.btn_start = self.create_button("START", "#28a745", self.cmd_start)
        self.btn_stop = self.create_button("STOP", "#dc3545", self.cmd_stop)
        self.btn_emg = self.create_button("EMERGENCY", "#ffc107", self.cmd_emg)
        self.btn_reset = self.create_button("RESET", "#6c757d", self.cmd_reset)

        # Pastikan Reset disabled di awal
        self.btn_reset.config(state="disabled") 

    def create_button(self, text, color, command):
        btn = tk.Button(self.btn_frame, text=text, bg=color, fg="white", font=("Arial", 11, "bold"), 
                        width=15, height=2, command=command, state="disabled")
        btn.pack(side="left", padx=10)
        return btn

    def start_camera(self):
        # Jika sedang Emergency, tolak akses buka kamera
        if self.is_emergency_active:
            messagebox.showwarning("PERINGATAN", "Sistem dalam Mode EMERGENCY!\nTekan tombol RESET terlebih dahulu.")
            return

        if not self.is_camera_on:
            self.cap = cv2.VideoCapture(0)
            self.is_camera_on = True
            self.btn_open_cam.config(text="TUTUP KAMERA", bg="#555")
            self.status_label.config(text="MEMINDAI WAJAH...", bg="#333", fg="yellow")
            self.update_video()
        else:
            self.stop_camera_logic()

    def stop_camera_logic(self):
        self.is_camera_on = False
        if self.cap:
            self.cap.release()
        self.video_frame.config(image='', bg="black", width=85, height=20) 
        self.btn_open_cam.config(text="📷 BUKA KAMERA / SCAN WAJAH", bg="#007bff")
        
        # Kunci sistem kembali
        self.lock_system()
        self.status_label.config(text="KAMERA MATI - SISTEM STOP", bg="#333", fg="white")

    def update_video(self):
        if self.is_camera_on and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    id, confidence = self.recognizer.predict(gray[y:y+h,x:x+w])
                    
                    if confidence < 60: 
                        detected_id = id
                        name = "OWNER (ACCESS GRANTED)"
                        color = (0, 255, 0)
                        self.unlock_system()
                    else:
                        name = "UNKNOWN"
                        color = (0, 0, 255)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, str(name), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                if len(faces) == 0:
                    self.lock_system()

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = img.resize((640, 360))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_frame.imgtk = imgtk
                self.video_frame.configure(image=imgtk, width=640, height=360)

            self.window.after(10, self.update_video)

    def unlock_system(self):
        # Jangan buka kunci jika sedang EMERGENCY
        if not self.is_authorized and not self.is_emergency_active:
            self.is_authorized = True
            self.status_label.config(text="AKSES DITERIMA: OPERATOR DIKENALI", bg="#0f3d0f", fg="#00ff00")
            
            self.btn_start.config(state="normal", cursor="hand2")
            self.btn_stop.config(state="normal", cursor="hand2")
            self.btn_emg.config(state="normal", cursor="hand2")
            self.btn_reset.config(state="disabled")

    def lock_system(self):
        if self.is_authorized:
            self.is_authorized = False
            self.status_label.config(text="AKSES DITOLAK: WAJAH TIDAK ADA", bg="#3d1010", fg="#ff4444")
            
            self.btn_start.config(state="disabled", cursor="arrow")
            self.btn_stop.config(state="disabled", cursor="arrow")
            self.btn_emg.config(state="disabled", cursor="arrow")

    # --- LOGIC TOMBOL UTAMA ---

    def send_request(self, cmd):
        def thread_task():
            start_time = time.time()
            success = False
            duration_ms = None
            try:
                print(f"--> Mengirim Perintah: {cmd}")
                r = requests.get(f"{ESP32_IP}/action?type={cmd}", timeout=1)
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                success = (r.status_code == 200)
                print(f"--> Perintah '{cmd}' selesai dalam {duration_ms:.2f} ms")
            except Exception as e: 
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                print("Gagal konek ke ESP32 (Cek WiFi):", e)
            finally:
                try:
                    # Try to read ESP32 internal latency
                    internal_latency = None
                    try:
                        s = requests.get(f"{ESP32_IP}/status", timeout=0.5)
                        if s.status_code == 200:
                            jd = s.json()
                            internal_latency = jd.get('last_internal_latency_us', None)
                    except Exception:
                        internal_latency = None

                    # write row with internal latency
                    with _csv_lock:
                        # append header if needed
                        write_header = not os.path.exists(_csv_path)
                        with open(_csv_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            if write_header:
                                writer.writerow(['timestamp', 'source', 'command', 'duration_ms', 'success', 'internal_latency_us'])
                            writer.writerow([datetime.utcnow().isoformat(timespec='milliseconds') + 'Z', 'GUI', cmd, '' if duration_ms is None else f"{duration_ms:.2f}", str(bool(success)), '' if internal_latency is None else str(internal_latency)])
                except Exception:
                    pass
        threading.Thread(target=thread_task).start()

    def cmd_start(self): 
        self.send_request("start")

    def cmd_stop(self): 
        # 1. Kirim perintah STOP ke ESP32
        self.send_request("stop")
        
        # 2. Matikan Kamera (Sesuai Permintaan)
        self.stop_camera_logic()
        
        # 3. Update Status
        self.status_label.config(text="MESIN BERHENTI - KAMERA OFF", bg="#3d1010", fg="white")

    def cmd_emg(self): 
        self.send_request("emergency")
        self.is_emergency_active = True
        self.is_authorized = False
        
        # Matikan Kamera
        self.stop_camera_logic()
        
        self.status_label.config(text="!!! EMERGENCY STOP ACTIVATED !!!", bg="red", fg="white")
        
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="disabled")
        self.btn_emg.config(state="disabled") 
        self.btn_open_cam.config(state="disabled", bg="#333") 
        
        # HANYA RESET YANG HIDUP
        self.btn_reset.config(state="normal", bg="#6c757d", cursor="hand2")

    def cmd_reset(self): 
        self.send_request("reset")
        self.is_emergency_active = False
        
        self.status_label.config(text="SYSTEM RESET - SILAKAN SCAN ULANG", bg="#333", fg="white")
        
        self.btn_open_cam.config(state="normal", bg="#007bff")
        self.btn_reset.config(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceControlApp(root)
    root.mainloop()