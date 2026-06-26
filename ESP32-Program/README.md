# ESP32-Program

PlatformIO firmware project for the **ESP32-S3 DevKitC-1**. The firmware turns
the ESP32 into a protocol bridge between the PC face-recognition interface and
the Mitsubishi FX5U PLC:

- **Wi-Fi (Core 0)** – runs a `WebServer` that receives HTTP commands from the
  Python HMI (`/action`, `/status`, `/user_detection`).
- **Ethernet + Modbus TCP (Core 1)** – drives a W5500 module over SPI and runs
  a Modbus TCP server on **port 503** that the FX5U PLC reads/writes.

## Structure

- `platformio.ini` – PlatformIO configuration (board `esp32-s3-devkitc-1`,
  Arduino framework, PSRAM enabled, libraries: `Ethernet_Generic`,
  `modbus-esp8266`).
- `src/` – Application source code (`main.cpp`).
- `include/` – Project header files (currently empty/PlatformIO default).
- `lib/` – Project-specific (private) libraries (currently empty).
- `test/` – PlatformIO unit tests (currently empty).

## Build & flash

```bash
pio run -t upload
pio device monitor   # 921600 baud
```

## Modbus coil map (defined in `src/main.cpp`)

| Coil | Address | Direction | Description |
| --- | --- | --- | --- |
| BTN 1 (Start)    | 112 | PC -> PLC | Momentary start command |
| BTN 2 (Stop)     | 113 | PC -> PLC | Momentary stop command |
| BTN 3 (Emergency)| 114 | PC -> PLC | Momentary emergency command |
| BTN 4 (Reset)    | 115 | PC -> PLC | Momentary reset command |
| User Detected    | 116 | PC -> PLC | Face-recognition authorized flag |
| STS_RUNNING      | 208 | PLC -> PC | Machine running indicator |
| STS_STOPPED      | 209 | PLC -> PC | Machine stopped indicator |
| STS_EMG          | 210 | PLC -> PC | Emergency active indicator |

Default Ethernet IP: `192.168.1.50`. Wi-Fi IP is configured statically inside
`main.cpp` (default `10.251.64.200`) – change these to match your network.