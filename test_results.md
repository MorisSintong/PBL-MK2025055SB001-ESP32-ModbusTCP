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
        UT1[BridgeState Atomic Logic]
        UT2[Modbus Coils & Pinouts]
        UT3[REST API Query Parsing]
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

The firmware was compiled using **LLVM-MinGW Clang 23.1.1** with `-std=c++17` and executed headlessly via CTest.

| Test Suite | File | Targets Verified | Execution Time | Status |
| :--- | :--- | :--- | :---: | :---: |
| **BridgeStateTest** | `tests/test_bridge_state.cpp` | Atomic command transitions, $\mu\text{s}$ latency calculation, momentary 500 ms pulse timeout logic, face recognition latching | 0.01 s | **PASSED** |
| **ModbusCoilsTest** | `tests/test_modbus_coils.cpp` | FX5U command coils (112–115), user auth (116), status coils (208–210), table boundary checks (300 registers), SPI pin assignments | 0.01 s | **PASSED** |
| **RestApiTest** | `tests/test_rest_api.cpp` | Command string parsing (`start`, `stop`, `emergency`, `reset`), invalid command rejection, boolean parameter conversion | 0.01 s | **PASSED** |

**CTest Summary:** 3/3 tests passed (100% pass rate, 0 failures, total time: 0.04 s).

---

## 3. Real-World Field Test Results (Physical Hardware Baseline)

The real-world tests recorded round-trip time (RTT in ms) and command success rate from the PC Face Recognition interface over Wi-Fi to the ESP32, which asserted coils to the physical Mitsubishi FX5U PLC.

### Dataset Overview
- **`timings.csv`**: Initial baseline during exploratory development with unoptimized FreeRTOS polling.
- **`timings1.csv`**: Run 1 with stable network conditions and active operator face authentication.
- **`timings2.csv`**: Run 2 under continuous cycling.
- **`timings_Jauh.csv`**: Run 3 conducted at an extended physical distance (simulating shop-floor range / Wi-Fi relay conditions).

### Statistical Breakdown

| Dataset | Sample Count | Success Rate | RTT Min | RTT Mean | RTT Median (P50) | RTT 95th % | RTT Max | RTT Jitter | Std Dev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`timings.csv`** (Early Baseline) | 647 | 49.61% | 15.14 ms | 525.11 ms | 223.19 ms | 1050.89 ms | 1392.44 ms | 2.13 ms | 469.49 ms |
| **`timings1.csv`** (Physical Run 1) | 100 | 95.00% | 21.11 ms | 145.62 ms | 87.18 ms | 1003.06 ms | 1015.73 ms | 10.05 ms | 208.54 ms |
| **`timings2.csv`** (Physical Run 2) | 100 | 96.00% | 24.66 ms | 192.10 ms | 128.39 ms | 531.84 ms | 1014.33 ms | 10.00 ms | 203.55 ms |
| **`timings_Jauh.csv`** (Extended Range) | 100 | 100.00% | 20.75 ms | 140.47 ms | 118.66 ms | 308.42 ms | 405.08 ms | 3.88 ms | 80.14 ms |

### Key Physical Observations
1. **Network Jitter & Tail Latency**: Across physical hardware runs, typical median latency hovered between **87 ms and 128 ms**. However, occasional Wi-Fi retries produced 95th-percentile tail spikes of ~500 ms to 1000 ms.
2. **Reliability at Distance**: In `timings_Jauh.csv`, the connection achieved a **100% success rate** with a mean RTT of **140.47 ms** and reduced jitter (3.88 ms), confirming that static IP allocation and disabling modem sleep (`WiFi.setSleep(false)`) stabilized the physical link.

---

## 4. Multi-Node Emulation Results (Host Software Loopback)

The emulation suite ([`Emulation/system_emulator.py`](file:///C:/Users/Moris/Documents/KULIAH/SEM5/PBL/Program/25.11.24/Sistem_Deteksi_Wajah/PBL-MK2025055SB001-ESP32-ModbusTCP/Emulation/system_emulator.py)) was executed over host loopback sockets to evaluate the refactored lock-free state engine under load.

### Dataset: `timings_emulated.csv` / `qos_report.csv`

| Metric | Measured Emulated Value | Physical Baseline (`timings1.csv`) | Operational Significance |
| :--- | :---: | :---: | :--- |
| **Total Test Transactions** | 46 cycles | 100 cycles | Covers face auth, start, stop, emergency, reset |
| **Packet Delivery / Success Rate** | **100.00%** | 95.00% | Zero dropped HTTP requests or socket dropouts |
| **HTTP RTT Duration (Min)** | **0.83 ms** | 21.11 ms | Instantaneous loopback processing floor |
| **HTTP RTT Duration (Mean)** | **16.61 ms** | 145.62 ms | Reflects non-blocking REST + Modbus task cycle |
| **HTTP RTT Duration (P50)** | **14.00 ms** | 87.18 ms | Typical command round-trip |
| **HTTP RTT Duration (P95)** | **26.28 ms** | 1003.06 ms | Absence of tail spikes under local scheduling |
| **HTTP RTT Duration (Max)** | **26.36 ms** | 1015.73 ms | Peak latency under high-frequency burst |
| **RTT Jitter (Mean Delta)** | **0.57 ms** | 10.05 ms | Deterministic loopback scheduling |
| **ESP32 Internal Bridge Latency** | **4.3 $\mu\text{s}$** | N/A (untracked) | High-speed atomic queue transfer across cores |
| **PLC Cyclic Scan Execution** | **10.0 ms** | 10.0 ms | Matches FX5U typical ladder program cycle |

---

## 5. Comparative Analysis: Physical vs. Emulation

| Attribute | Physical Field Deployment | Emulation Framework |
| :--- | :--- | :--- |
| **Physical Channel** | 2.4 GHz 802.11 Wi-Fi + CAT5e W5500 Ethernet | Windows Localhost Loopback (`127.0.0.1`) |
| **Modem Sleep / Contention** | Subject to RF beacon intervals & packet collisions | Zero physical interference |
| **SPI Bus Hardware** | Physical 4-wire SPI @ 8 MHz (W5500 silicon registers) | Direct in-memory / socket buffer transfers |
| **Mean Command RTT** | **140 ms — 192 ms** | **16.6 ms** |
| **Primary Use Case** | True end-to-end operational validation on shop floor | CI/CD automated regression, logic, & protocol validation |

---

## 6. Verification Commands

To reproduce these results using the project's [`justfile`](file:///C:/Users/Moris/Documents/KULIAH/SEM5/PBL/Program/25.11.24/Sistem_Deteksi_Wajah/PBL-MK2025055SB001-ESP32-ModbusTCP/justfile):

```powershell
# Run the host CTest unit tests:
just test

# Run the full distributed multi-node QoS emulation benchmark:
just emulate-qos 50

# Test headless Renode controller initialization:
just emulate-renode
```
