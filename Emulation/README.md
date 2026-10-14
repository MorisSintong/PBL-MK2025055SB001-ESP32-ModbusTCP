# System Emulation & QoS Benchmark Infrastructure

Automated end-to-end hardware emulation and Quality of Service (QoS) evaluation infrastructure for the **Face Detection & Mitsubishi FX5U Assembly Station SCADA**.

Since the physical station has been decommissioned, this infrastructure provides multi-node distributed emulation to validate communication integrity, network latency, PLC scan-reaction performance, and jitter under high-rate load.

---

## 1. Multi-Node Emulation Topology

```mermaid
graph TD
    subgraph Host [Emulated Distributed System]
        HMI[Virtual PC Face Recognition HMI<br>Workload Generator]
        ESP32[Virtual ESP32 Protocol Bridge<br>Core 0 HTTP :8080 | Core 1 Modbus :5503]
        PLC[Virtual Mitsubishi FX5U PLC Client<br>10ms Cyclic Scan Loop]
        RenodeNode[Renode Virtual Controller<br>Cortex-M Node :15503]
    end

    HMI -->|HTTP REST: /action, /status, /user_detection| ESP32
    PLC ===|Modbus TCP Client: Coils 112-116, 208-210| ESP32
    RenodeNode -.->|TCP Socket Bridge| HMI
```

1. **Virtual ESP32 Bridge (`VirtualESP32Bridge`)**:
   - Emulates the dual-core FreeRTOS architecture.
   - Core 0 runs an HTTP REST server on port `8080`.
   - Core 1 runs a Modbus TCP server on port `5503`.
   - Simulates microsecond-level atomic inter-core synchronization and manages momentary 500 ms coil pulse timeouts.
2. **Virtual Mitsubishi FX5U PLC (`VirtualMitsubishiPLC`)**:
   - Implements a 10 ms cyclic ladder scan loop over Modbus TCP.
   - Reads command coils 112 (`START`), 113 (`STOP`), 114 (`EMG`), 115 (`RESET`), and 116 (`USER_DETECTED`).
   - Writes machine status coils 208 (`STS_RUNNING`), 209 (`STS_STOPPED`), and 210 (`STS_EMG`).
3. **Virtual HMI & Workload Generator (`VirtualHMIClient`)**:
   - Emulates operator face recognition authentication and issues operational cycles (`start`, `stop`, `emergency`, `reset`).
   - Measures packet delivery ratio, round-trip time (RTT), internal ESP32 processing latency, and jitter.
4. **Renode Hardware Emulation (`RenodeController` + `plc_node.resc`)**:
   - Spawns headless Renode (`C:\Tools\renode\renode.exe`) with socket bridging on port `15503`.

---

## 2. Running the Emulation & Benchmark

Run with `just`:

```powershell
# Run 50-cycle end-to-end QoS evaluation
just emulate-qos samples="50"

# Run headless Renode controller test
just emulate-renode
```

---

## 3. Emulated QoS Benchmark Results

| Metric | Result | Benchmark Target | Status |
| :--- | :---: | :---: | :---: |
| **Packet Delivery / Success Rate** | **100.00%** | > 99.0% | PASS |
| **HTTP RTT Duration (Min)** | **0.88 ms** | < 50 ms | PASS |
| **HTTP RTT Duration (Mean)** | **19.25 ms** | < 100 ms | PASS |
| **HTTP RTT Duration (P95)** | **26.36 ms** | < 100 ms | PASS |
| **HTTP RTT Duration (Max)** | **26.79 ms** | < 200 ms | PASS |
| **RTT Jitter (Mean Delta)** | **0.47 ms** | < 10 ms | PASS |
| **ESP32 Internal Bridge Latency** | **4.3 μs** | < 50 μs | PASS |
| **PLC Cyclic Scan Cycle** | **10.0 ms** | 10.0 ms | PASS |

> [!NOTE]
> Detailed per-transaction timing telemetry is saved to `Interface/Timings_log/timings_emulated.csv` and `Emulation/logs/qos_report.csv` following the historical format in `timings.csv`.
