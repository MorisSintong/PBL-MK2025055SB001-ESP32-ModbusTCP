# Comprehensive System Test & QoS Performance Results

**Project:** PBL-MK2025055SB001 — Face Detection & Assembly Station SCADA  
**Platform:** ESP32-S3 DevKitC-1 + W5500 SPI Ethernet + Mitsubishi FX5U PLC + PC Face Recognition HMI  
**Branch:** `refactor/embedded-toolchain-migration`  
**Date of Compilation:** October 2026  

---

## Executive Summary

This document compiles all test methodologies, verification suites, and performance measurements gathered for the SCADA protocol bridge. It presents both:
1. **Real-World Empirical Results**: Historical physical measurements captured on the physical Assembly Station hardware prior to decommissioning (Wi-Fi 802.11 + physical W5500 SPI + physical Mitsubishi FX5U PLC).
2. **Off-Target Emulation & Simulation Results**: Unit-test verification and end-to-end distributed multi-node simulation benchmarks executed on the refactored architecture.

---

## 1. Test Architecture & Methodology

```mermaid
flowchart TD
    subgraph Layer1 [1. Unit Testing - CTest]
        UT1[BridgeState Atomic Logic, Concurrency & Wrap-Around]
        UT2[Modbus Coils, Ethernet Driver Supervision & Hardware Sanity]
        UT3[REST API Endpoints, Wi-Fi Supervisor & Injection Defense]
    end

    subgraph Layer2 [2. Host Emulation - Multi-Node Network]
        EMU_HMI[Virtual Face Recognition HMI]
        EMU_ESP[Virtual ESP32 Bridge<br>HTTP :8080 | Modbus :5503]
        EMU_PLC[Virtual FX5U PLC Client<br>10ms Scan Cycle]
        EMU_HMI <-->|HTTP REST| EMU_ESP
        EMU_PLC <==>|Modbus TCP| EMU_ESP
    end

    subgraph Layer3 [3. Physical Field Testing - Hardware Baseline]
        HW_PC[Physical PC Interface]
        HW_ESP[Physical ESP32-S3 + W5500]
        HW_PLC[Physical Mitsubishi FX5U PLC]
        HW_PC <-->|802.11 Wi-Fi| HW_ESP
        HW_ESP <==>|100BASE-TX Ethernet| HW_PLC
    end
```

---

## 2. Off-Target Firmware Unit Tests (CTest Suite)

The firmware was compiled using **LLVM-MinGW Clang 23.1.1** with `-std=c++17` and `-Wall -Wextra -Wpedantic`, and executed headlessly via CTest. The test suite provides comprehensive coverage of nominal states, negative inputs, 32-bit timer wrap-around, and multi-threaded inter-core concurrency:

| Test Suite | File | Targets & Edge Cases Verified | Execution Time | Status |
| :--- | :--- | :--- | :---: | :---: |
| **BridgeStateTest** | `tests/test_bridge_state.cpp` | • Atomic command transitions across all 4 channels (Start, Stop, Emergency, Reset)<br>• Per-channel start timestamps preventing cross-channel latency clobbering<br>• 32-bit `micros()` wrap-around & clock jitter/underflow protection<br>• Momentary pulse timers (`500 ms`) across all 4 channels & 32-bit `millis()` wrap-around<br>• Packed atomic `user_detect_mask_` (`bit 0 = detected`, `bit 1 = dirty`) & coherent `PlcStatus` snapshot<br>• Multi-threaded `std::thread` Core 0 producer / Core 1 consumer stress test (10,000 iterations)<br>• Host `ModbusBridge` coil & pulse integration | ~0.04 s | **PASSED** |
| **ModbusCoilsTest** | `tests/test_modbus_coils.cpp` | • FX5U command coils (112–115) and operator auth (116)<br>• Status feedback coils (208–210)<br>• Strict address uniqueness and disjoint range check between command & status blocks<br>• Register table boundaries and expansion headroom (>= 50 spare coils)<br>• Hardware SPI pin distinctness, 8 MHz clock rate, & dual-core FreeRTOS task priorities<br>• `EthernetDriver` link check (`2000 ms`) & socket check (`3000 ms`) supervision with 32-bit `millis()` wrap-around | ~0.04 s | **PASSED** |
| **RestApiTest** | `tests/test_rest_api.cpp` | • Positive command parsing (`start`, `stop`, `emergency`, `reset`)<br>• Case-sensitivity, whitespace, prefix/suffix, and injection defense (`<script>`, SQLi, 4KB overflow -> `Unknown`)<br>• Full `WebServerManager` HTTP endpoint dispatching (`/`, `/status`, `/action`, `/user_detection`) & `200`/`400` status verification<br>• Wi-Fi auto-reconnect supervisor timer (`10,000 ms`) & 32-bit `millis()` wrap-around | ~0.04 s | **PASSED** |

**CTest Summary:** 3/3 test executables passed (16 high-impact test functions, 100% pass rate, 0 failures, execution time: ~0.12 s).

---

## 3. Real-World Field Test Results (Physical Hardware Baseline)

The real-world tests recorded round-trip time (RTT in ms), command success rate, and ESP32 internal inter-core latency ($\mu\text{s}$) from the PC Face Recognition interface over Wi-Fi to the ESP32, which asserted coils to the physical Mitsubishi FX5U PLC.

### Dataset Overview
- **`timings.csv`**: Initial baseline during exploratory development with unoptimized FreeRTOS polling (captured prior to `internal_latency_us` instrumentation).
- **`timings1.csv`**: Physical Run 1 with stable network conditions, active operator face authentication, and ESP32 internal latency instrumentation.
- **`timings2.csv`**: Physical Run 2 under continuous operational cycling.
- **`timings_Jauh.csv`**: Physical Run 3 conducted at an extended physical distance (simulating shop-floor range / Wi-Fi relay conditions).

### 3.1 Successful-Request RTT & ESP32 Internal Bridge Latency (`success == True`)

Filtering out 1000 ms HTTP client timeout events isolates the true wireless + ESP32 + Modbus bridge performance on delivered commands:

| Dataset | Delivered / Total | Success Rate | RTT Min | RTT Mean | RTT Median (P50) | RTT 95th % | RTT Max | Chronological Jitter | Std Dev | ESP32 Internal Latency (Mean) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`timings.csv`** (Early Baseline) | 321 / 647 | 49.61% | 15.14 ms | 89.60 ms | 54.22 ms | 271.23 ms | 1021.37 ms | 61.00 ms | 119.81 ms | N/A (untracked) |
| **`timings1.csv`** (Physical Run 1) | 95 / 100 | 95.00% | 21.11 ms | 100.11 ms | 83.61 ms | 230.66 ms | 437.32 ms | 64.81 ms | 65.91 ms | **915.7 $\mu\text{s}$** |
| **`timings2.csv`** (Physical Run 2) | 96 / 100 | 96.00% | 24.66 ms | 158.03 ms | 122.04 ms | 431.07 ms | 531.84 ms | 108.85 ms | 118.93 ms | **675.1 $\mu\text{s}$** |
| **`timings_Jauh.csv`** (Extended Range) | 100 / 100 | 100.00% | 20.75 ms | 140.47 ms | 118.66 ms | 308.42 ms | 405.08 ms | 83.63 ms | 80.14 ms | **817.7 $\mu\text{s}$** |

### 3.2 Overall RTT Including Timeout Rows (All Recorded Attempts)

Including failed HTTP requests (which hit the 1000 ms client timeout in `Interface/app.py`) reflects end-to-end user-perceived latency during RF contention:

| Dataset | Total Samples | Success Rate | RTT Min | RTT Mean | RTT Median (P50) | RTT 95th % | RTT Max | Chronological Jitter | Std Dev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`timings.csv`** (Early Baseline) | 647 | 49.61% | 15.14 ms | 525.11 ms | 223.19 ms | 1050.89 ms | 1392.44 ms | 54.97 ms | 469.49 ms |
| **`timings1.csv`** (Physical Run 1) | 100 | 95.00% | 21.11 ms | 145.62 ms | 87.18 ms | 1003.06 ms | 1015.73 ms | 78.32 ms | 208.54 ms |
| **`timings2.csv`** (Physical Run 2) | 100 | 96.00% | 24.66 ms | 192.10 ms | 128.39 ms | 531.84 ms | 1014.33 ms | 138.92 ms | 203.55 ms |
| **`timings_Jauh.csv`** (Extended Range) | 100 | 100.00% | 20.75 ms | 140.47 ms | 118.66 ms | 308.42 ms | 405.08 ms | 83.63 ms | 80.14 ms |

### Key Physical Observations
1. **Sub-Millisecond Internal ESP32 Bridge Overhead**: Across `timings1.csv`, `timings2.csv`, and `timings_Jauh.csv`, the physical ESP32-S3 inter-core handoff and Modbus coil update averaged **675.1 $\mu\text{s}$ to 915.7 $\mu\text{s}$** (< 1 ms), proving that >99% of end-to-end RTT originates in the 802.11 Wi-Fi transport layer rather than firmware execution.
2. **Network Jitter & Timeout Impact**: On delivered packets, median RTT was **83.61 ms to 122.04 ms** with chronological jitter of **64.81 ms to 108.85 ms**. When including the 4–5% Wi-Fi timeout rows (`~1000 ms`) in `timings1.csv` and `timings2.csv`, overall P95–Max tail latency rises to `1003–1015 ms`.
3. **Reliability at Distance**: In `timings_Jauh.csv`, the connection achieved a **100.00% success rate** with a mean RTT of **140.47 ms**, chronological jitter of **83.63 ms**, and mean internal bridge latency of **817.7 $\mu\text{s}$**, confirming that static IP allocation and disabling modem sleep (`WiFi.setSleep(false)`) stabilized the physical link.

---

## 4. Multi-Node Emulation Results (Host Software Loopback)

The emulation suite ([`Emulation/system_emulator.py`](file:///C:/Users/Moris/Documents/KULIAH/SEM5/PBL/Program/25.11.24/Sistem_Deteksi_Wajah/PBL-MK2025055SB001-ESP32-ModbusTCP/Emulation/system_emulator.py)) was executed with 50 Phase 3 stress iterations (plus 6 Phase 2 functional verification commands = 56 total transactions) over host loopback sockets to evaluate the lock-free state engine and safety-dominant PLC ladder logic (`Emergency -> Reset -> Stop -> Start`).

### Dataset: `timings_emulated.csv` / `qos_report.csv`

| Metric | Measured Emulated Value | Physical Baseline (`timings1.csv` Delivered / Overall) | Operational Significance |
| :--- | :---: | :---: | :--- |
| **Total Test Transactions** | **56 cycles** | 95 delivered / 100 total | 6 functional state checks + 50 stress iterations |
| **Packet Delivery / Success Rate** | **100.00%** | 95.00% | Zero dropped HTTP requests or socket dropouts |
| **HTTP RTT Duration (Min)** | **0.93 ms** | 21.11 ms | Instantaneous loopback processing floor |
| **HTTP RTT Duration (Mean)** | **17.00 ms** | 100.11 ms / 145.62 ms | Reflects non-blocking REST + status telemetry fetch |
| **HTTP RTT Duration (Median / P50)** | **12.10 ms** | 83.61 ms / 87.18 ms | Typical command round-trip |
| **HTTP RTT Duration (P95)** | **26.04 ms** | 230.66 ms / 1003.06 ms | Absence of RF retry tail spikes under local scheduling |
| **HTTP RTT Duration (P99)** | **26.22 ms** | 437.32 ms / 1015.73 ms | Tight upper bound under high-frequency burst |
| **HTTP RTT Duration (Max)** | **26.22 ms** | 437.32 ms / 1015.73 ms | Peak latency across all 56 transactions |
| **RTT Std Dev** | **7.55 ms** | 65.91 ms / 208.54 ms | Low dispersion across burst cycles |
| **RTT Jitter (Mean Chronological Delta)** | **7.96 ms** | 64.81 ms / 78.32 ms | Consecutive-packet delta on localhost TCP |
| **ESP32 Internal Bridge Latency** | **3.9 $\mu\text{s}$** | 915.7 $\mu\text{s}$ | High-speed atomic queue transfer across cores |
| **PLC Cyclic Scan Execution** | **10.0 ms** | 10.0 ms | Matches FX5U typical ladder program cycle |

---

## 5. Comparative Analysis: Physical vs. Emulation

| Attribute | Physical Field Deployment | Emulation Framework |
| :--- | :--- | :--- |
| **Physical Channel** | 2.4 GHz 802.11 Wi-Fi + CAT5e W5500 Ethernet | Windows Localhost Loopback (`127.0.0.1`) |
| **Modem Sleep / Contention** | Subject to RF beacon intervals & packet collisions | Zero physical RF interference |
| **SPI Bus Hardware** | Physical 4-wire SPI @ 8 MHz (W5500 silicon registers) | Direct in-memory / socket buffer transfers |
| **Mean Command RTT** | **100.11 ms — 158.03 ms** (delivered) / **140.47 ms — 192.10 ms** (overall) | **17.00 ms** |
| **Mean Internal Bridge Latency** | **675.1 $\mu\text{s}$ — 915.7 $\mu\text{s}$** | **3.9 $\mu\text{s}$** |
| **Primary Use Case** | True end-to-end operational validation on shop floor | CI/CD automated regression, logic, & protocol validation |

---

## 6. Verification Commands

To reproduce these results using the project's [`justfile`](file:///C:/Users/Moris/Documents/KULIAH/SEM5/PBL/Program/25.11.24/Sistem_Deteksi_Wajah/PBL-MK2025055SB001-ESP32-ModbusTCP/justfile):

```powershell
# Run the host CTest unit tests:
just test

# Update clangd compilation database:
just update-compile-commands

# Run the full distributed multi-node QoS emulation benchmark (50 stress samples):
just emulate-qos samples="50"

# Test headless Renode controller initialization:
just emulate-renode
```
