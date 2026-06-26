# GUI

Early-prototype GUI option that hosts a minimal web interface directly on a
separate ESP32 (instead of on the PC). Kept for reference/history.

## Contents

- `GUI_REAL/` – Arduino sketch (`GUI_REAL.ino`) that creates a Wi-Fi AP named
  `MEKAPBL` with password `mekagui1` and exposes `/action?type=...`
  endpoints to print `start` / `stop` / `emergency` / `reset` to the serial
  monitor (placeholder for driving relays/LEDs). See its `README.md`.

This prototype was later replaced by the PC-side Flask HMI in the parent
`Interface/` directory (`app.py`), which also performs face recognition.