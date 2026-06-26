# src

Main application source for the ESP32 firmware.

## Files

- `main.cpp` – The complete firmware entry point. It:
  - Initializes SPI for the W5500 Ethernet module and starts a Modbus TCP
    server on port 503.
  - Spawns two pinned FreeRTOS tasks:
    - `modbusTask` (Core 1, priority 2) – poll Modbus, process command flags,
      handle momentary coil timers, monitor W5500 socket/Ethernet link.
    - `wifiTask` (Core 0, priority 1) – connect to Wi-Fi with a static IP,
      start mDNS (`esp32-server.local`), serve the HTTP endpoints used by the
      PC interface, and auto-reconnect on Wi-Fi loss.
  - Exposes HTTP endpoints: `/`, `/status`, `/action?type=...`,
    `/user_detection?detected=1|0`.
  - Uses `std::atomic` flags + a Serial mutex for safe cross-core access.
- `src.7z` – Archive backup of previous source revisions.

To rebuild, run `pio run -t upload` from the parent `ESP32-Program/` directory.