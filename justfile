# Root Justfile for PBL Face Detection & Assembly Station SCADA

set shell := ["powershell", "-NoProfile", "-Command"]

default: test

# Run ESP32 firmware host unit test suite
test:
    just -f ESP32-Program/justfile -d ESP32-Program test

# Build ESP32 firmware host simulation targets
build:
    just -f ESP32-Program/justfile -d ESP32-Program build

# Clean all build artifacts
clean:
    just -f ESP32-Program/justfile -d ESP32-Program clean

# Run full end-to-end multi-node system emulation & QoS benchmark
emulate-qos samples="50":
    python Emulation/system_emulator.py {{samples}}

# Run Renode headless hardware emulation
emulate-renode:
    python Emulation/renode_runner.py

# Flash ESP32-S3 over SWD/JTAG using probe-rs
flash-probe-rs:
    just -f ESP32-Program/justfile -d ESP32-Program flash-probe-rs

# Flash ESP32-S3 over UART using uvx esptool
flash-esptool port="COM3" bin="firmware.bin":
    just -f ESP32-Program/justfile -d ESP32-Program flash-esptool port={{port}} bin={{bin}}

# Serial monitor @ 921600 baud
monitor port="COM3" baud="921600":
    just -f ESP32-Program/justfile -d ESP32-Program monitor port={{port}} baud={{baud}}
