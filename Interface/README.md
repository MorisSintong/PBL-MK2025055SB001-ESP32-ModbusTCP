# Interface

PC-side HMI and face-recognition application. The interface is written in
Python (Flask + OpenCV) and provides:

- A browser-based HMI with the live webcam feed, start/stop/emergency/reset
  buttons, and PLC status indicators.
- Face **detection** (Haar cascade) and **recognition** (LBPH) to authorize
  operators before the machine can be started.
- HTTP communication with the ESP32 web server to forward commands to the PLC.
- CSV timing logs for performance measurement.

## Main scripts

| File | Purpose |
| --- | --- |
| `app.py` | Primary Flask web HMI: video feed, face recognition, training, registration, PLC command forwarding. **Run this for the production interface.** |
| `AppImproved.py` | Performance-optimized variant of `app.py` (threaded `VideoStream`, lower JPEG quality, etc.). |
| `main.py` | Standalone Tkinter desktop GUI version of the HMI (no browser needed). |
| `pythonUDP.py` | Experimental variant that sends commands to the ESP32 over UDP instead of HTTP. |
| `01_Ambil_Wajah.py` | CLI tool to capture face images into `dataset/` for a given user ID. |
| `02_training.py` | CLI tool to train the LBPH model from `dataset/` into `trainer/trainer.yml`. |
| `test_modbus.py` | Standalone Modbus TCP client test to verify the ESP32 coils (connects to `192.168.1.50:503`). |

## Subdirectories

- `templates/` – Jinja2 HTML templates for the Flask HMI.
- `GUI/` – Stand-alone Arduino sketch (`GUI_REAL.ino`) for an early prototype
  that put the web HMI directly on a second ESP32.
- `dataset/` – Captured face images (`User.<id>.<n>.jpg`).
- `trainer/` – Trained LBPH model (`trainer.yml`).
- `Timings_log/` – Recorded timing CSVs from earlier test runs.

## Running

```bash
# 1. Capture faces (optional, or use the web HMI registration button)
python 01_Ambil_Wajah.py
# 2. Train the model
python 02_training.py
# 3. Launch the web HMI
python app.py          # open http://<PC-IP>:5000
```

Make sure the ESP32 IP inside each script matches the ESP32's actual Wi-Fi IP
(`ESP32_IP = "http://<IP>"`).