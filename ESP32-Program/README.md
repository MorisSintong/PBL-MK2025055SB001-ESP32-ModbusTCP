# ESP32-Program - Modbus TCP & REST Gateway

Firmware for the **ESP32-S3 DevKitC-1** acting as a high-performance protocol bridge between the PC Face-Recognition HMI and the **Mitsubishi FX5U PLC**:

- **Wi-Fi + REST Server (Core 0)**: Non-blocking HTTP endpoints (`/`, `/status`, `/action`, `/user_detection`) with modem sleep disabled for minimum response latency.
- **Ethernet + Modbus TCP Server (Core 1)**: Hardware SPI driving a W5500 Ethernet transceiver at 8 MHz, serving Modbus TCP on **Port 503** with atomic cross-core synchronization and momentary pulse timer handling.
- **Native Embedded Toolchain**: Built with **CMake + Ninja** using the standalone **Xtensa GCC 16.1.0** cross-compiler (`C:\Tools\xtensa-esp-elf`) and a custom ROM-backed linker script (`ld/esp32s3_out.ld`), with **zero dependencies on PlatformIO or Arduino IDE**.

---

## Architecture & Directory Layout

Built on a modular CMake + Ninja embedded C++17 architecture with dual-targeting host simulation and CTest verification capabilities:

```text
ESP32-Program/
├── App/
│   ├── Inc/
│   │   ├── app_config.hpp       # Static IPs, SPI pinouts, Modbus addresses, timings
│   │   ├── bridge_state.hpp     # Lock-free atomic inter-core synchronization & metrics
│   │   ├── esp32s3_hal.hpp      # Hardware abstraction: ROM functions, cycle timers, SPI
│   │   ├── eth_driver.hpp       # W5500 SPI driver, link status, socket diagnostics
│   │   ├── logger.hpp           # Mutex-protected thread-safe serial logging (UART0 @ 921600)
│   │   ├── modbus_bridge.hpp    # Modbus TCP server, coil dispatching & PLC feedback
│   │   └── web_server.hpp       # HTTP server request routing & JSON serialization
│   └── Src/
│       ├── bridge_state.cpp     # State transition, latency tracking & momentary pulse logic
│       ├── eth_driver.cpp       # W5500 initialization & socket maintenance
│       ├── logger.cpp           # Direct UART0 / FreeRTOS mutex / Host std::mutex logger
│       ├── modbus_bridge.cpp    # Modbus TCP engine & coil mapping
│       ├── web_server.cpp       # REST API endpoints & responses
│       └── main.cpp             # Clean dual-core orchestrator & on-chip verification entry
├── cmake/
│   └── esp32s3-toolchain.cmake  # Xtensa GCC 16.1.0 cross-compilation toolchain file
├── ld/
│   └── esp32s3_out.ld           # ESP32-S3 SRAM1 memory layout & ROM symbol table
├── tests/
│   ├── CMakeLists.txt           # Host CTest harness using Clang
│   ├── test_bridge_state.cpp    # Unit tests: atomic commands, pulse timeouts, 10k-iter concurrency
│   ├── test_modbus_coils.cpp    # Unit tests: register bounds, PLC coil addresses, SPI supervision
│   └── test_rest_api.cpp        # Unit tests: command parsing, JSON serialization, Wi-Fi supervisor
├── CMakeLists.txt               # Main CMake build definition (cross-compile target + dual-target host tests)
├── CMakePresets.json            # Config presets (host-test, release)
├── justfile                     # Task runner recipes (build, test, build-release, flash, monitor)
├── .clangd                      # Zed / Clangd language server AST configuration
└── README.md                    # Firmware documentation
```

---

## Toolchain & Requirements

In alignment with the developer system toolchain profile:
- **Build Systems**: CMake 4.2+ & Ninja 1.13+
- **Host Compiler**: LLVM-MinGW (Clang 23+)
- **Cross-Compiler**: Xtensa GCC 16.1.0 (`C:\Tools\xtensa-esp-elf\bin\xtensa-esp32s3-elf-g++.exe`)
- **Primary Editor**: Zed with clangd semantic intelligence via `compile_commands.json`
- **Task Runner**: `just`
- **Debug & Flash Tools**: `uvx esptool` (v5.6+) over UART @ 921600 baud, or `probe-rs` over SWD/JTAG

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

## Build, Test & Flash Workflow

All operations are automated using `just`:

```powershell
# 1. Run full host simulation test suite (CTest - 0.04s)
just test

# 2. Build host simulation binaries
just build

# 3. Cross-compile native ESP32-S3 ELF & firmware.bin with Xtensa GCC + Ninja
just build-release

# 4. Update compile_commands.json for Zed language server
just update-compile-commands

# 5. Clean build artifacts (idempotent, exit code 0)
just clean

# 6. Run silicon flash and security diagnostics on COM10
just diag-esptool

# 7. Flash ESP32-S3 over UART using uvx esptool @ 921600 baud (0.9s write time)
just flash-esptool

# 8. Monitor serial output @ 921600 baud via uvx pyserial
just monitor
```
