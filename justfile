# Root Justfile for PBL Face Detection & Assembly Station SCADA

set shell := ["powershell", "-NoProfile", "-Command"]

default: test

# Run ESP32 firmware host unit test suite
test:
    just -f ESP32-Program/justfile -d ESP32-Program test

# Build ESP32 firmware host simulation targets
build:
    just -f ESP32-Program/justfile -d ESP32-Program build

# Build ESP32 firmware release target
build-release:
    just -f ESP32-Program/justfile -d ESP32-Program build-release

# Clean all build artifacts
clean:
    just -f ESP32-Program/justfile -d ESP32-Program clean
    if (Test-Path __cmake_systeminformation) { Remove-Item -Recurse -Force __cmake_systeminformation }; if (Test-Path compile_commands.json) { Remove-Item -Force compile_commands.json }

# Copy compile_commands.json to root and ESP32-Program for Zed AST indexing
update-compile-commands:
    just -f ESP32-Program/justfile -d ESP32-Program update-compile-commands
    Copy-Item -Force ESP32-Program/build-host/compile_commands.json ./compile_commands.json

# Run full end-to-end multi-node system emulation & QoS benchmark
emulate-qos samples="50":
    python Emulation/system_emulator.py {{samples}}

# Run Renode headless hardware emulation
emulate-renode:
    python Emulation/renode_runner.py

# Flash ESP32-S3 over SWD/JTAG using probe-rs
flash-probe-rs elf="build-release/ESP32_Modbus_App.elf":
    just -f ESP32-Program/justfile -d ESP32-Program flash-probe-rs "{{elf}}"

# Flash ESP32-S3 over UART using uvx esptool
flash-esptool port="COM10" bin="build-release/firmware.bin":
    just -f ESP32-Program/justfile -d ESP32-Program flash-esptool "{{port}}" "{{bin}}"

# Run hardware flash & security diagnostics on ESP32-S3 via uvx esptool
diag-esptool port="COM10":
    just -f ESP32-Program/justfile -d ESP32-Program diag-esptool "{{port}}"

# Serial monitor @ 921600 baud via uvx pyserial
monitor port="COM10" baud="921600":
    just -f ESP32-Program/justfile -d ESP32-Program monitor "{{port}}" "{{baud}}"