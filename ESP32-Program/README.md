# ESP32-Program - Modbus TCP & REST Gateway

Firmware for the **ESP32-S3 DevKitC-1** acting as a high-performance protocol bridge between the PC Face-Recognition HMI and the **Mitsubishi FX5U PLC**:

- **Wi-Fi + REST Server (Core 0)**: Non-blocking HTTP endpoints (`/`, `/status`, `/action`, `/user_detection`) with modem sleep disabled for minimum response latency.
- **Ethernet + Modbus TCP Server (Core 1)**: Hardware SPI driving a W5500 Ethernet transceiver at 8 MHz, serving Modbus TCP on **Port 503** with atomic cross-core synchronization and momentary pulse timer handling.

---

## Architecture & Directory Layout

Refactored from legacy PlatformIO into a modern, modular CMake + Ninja embedded architecture with dual-targeting host simulation capabilities:

```text
ESP32-Program/
├── App/
│   ├── Inc/
│   │   ├── app_config.hpp       # Static IPs, SPI pinouts, Modbus addresses, timings
│   │   ├── bridge_state.hpp     # Lock-free atomic inter-core synchronization & metrics
│   │   ├── eth_driver.hpp       # W5500 SPI driver, link status, socket diagnostics
│   │   ├── logger.hpp           # Mutex-protected thread-safe serial logging
│   │   ├── modbus_bridge.hpp    # Modbus TCP server, coil dispatching & PLC feedback
│   │   └── web_server.hpp       # HTTP server request routing & JSON serialization
│   └── Src/
│       ├── bridge_state.cpp     # State transition, latency tracking & momentary pulse logic
│       ├── eth_driver.cpp       # W5500 initialization & socket maintenance
│       ├── logger.cpp           # FreeRTOS mutex / Host std::mutex logger
│       ├── modbus_bridge.cpp    # Modbus TCP engine & coil mapping
│       ├── web_server.cpp       # REST API endpoints & responses
│       └── main.cpp             # Clean FreeRTOS dual-core task orchestrator
├── tests/
│   ├── CMakeLists.txt           # Host CTest harness using Clang
│   ├── test_bridge_state.cpp    # Unit tests: atomic commands, pulse timeouts, latency
│   ├── test_modbus_coils.cpp    # Unit tests: register bounds, PLC coil addresses
│   └── test_rest_api.cpp        # Unit tests: command parsing, JSON serialization
├── CMakeLists.txt               # Main CMake build definition (target + dual-target tests)
├── CMakePresets.json            # Config presets (host-test, release)
├── justfile                     # Task runner recipes (build, test, flash, monitor)
├── .clangd                      # Zed / Clangd language server AST configuration
└── README.md                    # Firmware documentation
```

---

## Toolchain & Requirements

In alignment with the developer system toolchain profile (`AGENTS.md`):
- **Build System**: CMake 4.2+ & Ninja 1.13+
- **Host Compiler**: LLVM-MinGW (Clang 23+)
- **Primary Editor**: Zed with clangd semantic intelligence via `compile_commands.json`
- **Task Runner**: `just` (v1.58+)
- **Debug & Flash Probes**: `probe-rs` (0.32+) over SWD/JTAG, or `uvx esptool` (5.5+) over USB/UART

---

## Modbus Coil Register Map

| Coil Address | Symbol | Direction | Function | Behavior |
| :---: | :---: | :---: | :--- | :--- |
| **112** | `COIL_BTN_START` | PC -> PLC | Momentary Start Command | Active HIGH for 500 ms, then resets |
| **113** | `COIL_BTN_STOP` | PC -> PLC | Momentary Stop Command | Active HIGH for 500 ms, then resets |
| **114** | `COIL_BTN_EMERGENCY` | PC -> PLC | Momentary Emergency Stop | Active HIGH for 500 ms, then resets |
| **115** | `COIL_BTN_RESET` | PC -> PLC | Momentary Fault Reset | Active HIGH for 500 ms, then resets |
| **116** | `COIL_USER_DETECTED` | PC -> PLC | Operator Authorized Flag | Continuous latch written by face recognition |
| **208** | `COIL_STS_RUNNING` | PLC -> PC | Machine Running Feedback | Read from PLC, reported in `/status` |
| **209** | `COIL_STS_STOPPED` | PLC -> PC | Machine Stopped Feedback | Read from PLC, reported in `/status` |
| **210** | `COIL_STS_EMG` | PLC -> PC | Machine Emergency Feedback | Read from PLC, reported in `/status` |

---

## REST API Endpoints (Port 80)

| Endpoint | Method | Parameters | Description |
| :--- | :---: | :--- | :--- |
| `/` | `GET` | None | Health check; responds with `"ESP32 Opti-Server Online"` |
| `/status` | `GET` | None | Returns JSON: `{"running":bool,"stopped":bool,"emergency":bool,"last_internal_latency_us":int}` |
| `/action` | `GET` | `?type=start\|stop\|emergency\|reset` | Sets momentary coil flag and records start timestamp |
| `/user_detection` | `GET` | `?detected=1\|0` | Updates continuous coil 116 (`COIL_USER_DETECTED`) |

---

## Build & Test Workflow

All operations are automated using `just`:

```powershell
# 1. Run full host simulation test suite (CTest)
just test

# 2. Build host simulation binaries
just build

# 3. Update compile_commands.json for Zed language server
just update-compile-commands

# 4. Clean build artifacts
just clean

# 5. Flash ESP32-S3 over SWD/JTAG using probe-rs
just flash-probe-rs

# 6. Flash ESP32-S3 over UART using uvx esptool
just flash-esptool port="COM3" bin="firmware.bin"

# 7. Monitor serial output @ 921600 baud
just monitor port="COM3"
```
