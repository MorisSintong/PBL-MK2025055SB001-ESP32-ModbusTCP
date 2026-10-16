#!/usr/bin/env python3
"""End-to-End Multi-Node Hardware Emulation & QoS Benchmark for PBL Modbus SCADA System.

Emulates the complete distributed system:
  1. Virtual ESP32-S3 Protocol Bridge (Dual-core FreeRTOS architecture: HTTP Port 8080 + Modbus TCP Port 5503)
  2. Virtual Mitsubishi FX5U PLC (Modbus TCP Client with 10ms cyclic ladder scan)
  3. Virtual PC Face Recognition HMI (Workload generator & QoS metric collector)

Produces real CSV telemetry matching Interface/Timings_log/timings.csv and computes statistical QoS metrics.
"""

from __future__ import annotations

import csv
import datetime
import http.server
import json
import math
import socket
import socketserver
import struct
import sys
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configuration
HTTP_PORT = 8080
MODBUS_PORT = 5503
MOMENTARY_PULSE_DURATION_S = 0.500  # 500 ms momentary pulse
PLC_SCAN_CYCLE_S = 0.010  # 10 ms PLC ladder execution cycle

# Coil Definitions
COIL_BTN_START = 112
COIL_BTN_STOP = 113
COIL_BTN_EMG = 114
COIL_BTN_RESET = 115
COIL_USER_DETECTED = 116

COIL_STS_RUNNING = 208
COIL_STS_STOPPED = 209
COIL_STS_EMG = 210

BASE_DIR = Path(__file__).resolve().parent.parent
TIMINGS_OUTPUT_CSV = BASE_DIR / "Interface" / "Timings_log" / "timings_emulated.csv"
EMULATION_LOG_CSV = BASE_DIR / "Emulation" / "logs" / "qos_report.csv"


@dataclass
class TimingRecord:
    timestamp: str
    source: str
    command: str
    duration_ms: float
    success: bool
    internal_latency_us: int
    plc_reaction_ms: float = 0.0


# =======================================================================
# NODE 1: VIRTUAL ESP32 PROTOCOL BRIDGE
# =======================================================================
class VirtualESP32Bridge:
    def __init__(self, http_port: int = HTTP_PORT, modbus_port: int = MODBUS_PORT):
        self.http_port = http_port
        self.modbus_port = modbus_port

        # Modbus registers (300 coils)
        self.coils: Dict[int, bool] = {i: False for i in range(300)}
        self.coils[COIL_STS_STOPPED] = True  # Default initial state: Stopped
        self.coil_lock = threading.Lock()

        # Inter-core atomic queues
        self.pending_command: Optional[str] = None
        self.action_start_time_us: int = 0
        self.last_internal_latency_us: int = 0
        self.cmd_lock = threading.Lock()

        # Momentary timers
        self.pulse_timers: Dict[int, float] = {}

        self.running = False
        self.httpd: Optional[socketserver.TCPServer] = None
        self.modbus_sock: Optional[socket.socket] = None
        self.threads: List[threading.Thread] = []

    def start(self):
        self.running = True

        # Launch HTTP Server Thread (Core 0 simulation)
        http_thread = threading.Thread(
            target=self._run_http_server, name="ESP32-Core0-HTTP", daemon=True
        )
        http_thread.start()
        self.threads.append(http_thread)

        # Launch Modbus Server Thread (Core 1 simulation)
        modbus_thread = threading.Thread(
            target=self._run_modbus_server, name="ESP32-Core1-Modbus", daemon=True
        )
        modbus_thread.start()
        self.threads.append(modbus_thread)

        # Launch Pulse Supervisor Thread
        pulse_thread = threading.Thread(
            target=self._run_pulse_supervisor, name="ESP32-Core1-Timers", daemon=True
        )
        pulse_thread.start()
        self.threads.append(pulse_thread)

        time.sleep(0.1)  # Allow sockets to bind

    def stop(self):
        self.running = False
        if self.httpd:
            try:
                self.httpd.shutdown()
            except Exception:
                pass
        if self.modbus_sock:
            try:
                self.modbus_sock.close()
            except Exception:
                pass

    def _run_http_server(self):
        bridge = self

        class BridgeHTTPHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress default stdout log

            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                params = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)

                if parsed.path == "/":
                    self.send_response(200)
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"ESP32 Opti-Server Online")

                elif parsed.path == "/status":
                    with bridge.coil_lock:
                        run = bridge.coils.get(COIL_STS_RUNNING, False)
                        stp = bridge.coils.get(COIL_STS_STOPPED, False)
                        emg = bridge.coils.get(COIL_STS_EMG, False)

                    payload = {
                        "running": run,
                        "stopped": stp,
                        "emergency": emg,
                        "last_internal_latency_us": bridge.last_internal_latency_us,
                    }
                    self.send_response(200)
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(payload).encode("utf-8"))

                elif parsed.path == "/action":
                    cmd = params.get("type", [None])[0]
                    if cmd is None or cmd == "":
                        self.send_response(400)
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.send_header("Content-Type", "text/plain")
                        self.end_headers()
                        self.wfile.write(b"Missing type")
                    elif cmd in ("start", "stop", "emergency", "reset"):
                        now_us = int(time.perf_counter() * 1_000_000)
                        with bridge.cmd_lock:
                            bridge.action_start_time_us = now_us
                            bridge.pending_command = cmd

                        # Process into Modbus coil immediately (simulating Core 1 pick-up)
                        bridge._process_pending_action()

                        self.send_response(200)
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.send_header("Content-Type", "text/plain")
                        self.end_headers()
                        self.wfile.write(b"OK")
                    else:
                        self.send_response(400)
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.send_header("Content-Type", "text/plain")
                        self.end_headers()
                        self.wfile.write(b"Unknown command")

                elif parsed.path == "/user_detection":
                    if "detected" not in params:
                        self.send_response(400)
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.send_header("Content-Type", "text/plain")
                        self.end_headers()
                        self.wfile.write(b"Missing detected parameter")
                    else:
                        detected = params.get("detected", ["0"])[0]
                        is_detected = detected in ("1", "true")
                        with bridge.coil_lock:
                            bridge.coils[COIL_USER_DETECTED] = is_detected

                        self.send_response(200)
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.send_header("Content-Type", "text/plain")
                        self.end_headers()
                        self.wfile.write(b"OK")
                else:
                    self.send_response(404)
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()

        socketserver.TCPServer.allow_reuse_address = True
        self.httpd = socketserver.TCPServer(
            ("127.0.0.1", self.http_port), BridgeHTTPHandler
        )
        self.httpd.serve_forever()

    def _process_pending_action(self):
        with self.cmd_lock:
            cmd = self.pending_command
            start_us = self.action_start_time_us
            self.pending_command = None

        if not cmd:
            return

        now_us = int(time.perf_counter() * 1_000_000)
        self.last_internal_latency_us = max(1, now_us - start_us)
        now_s = time.time()

        coil_map = {
            "start": COIL_BTN_START,
            "stop": COIL_BTN_STOP,
            "emergency": COIL_BTN_EMG,
            "reset": COIL_BTN_RESET,
        }

        target_coil = coil_map.get(cmd)
        if target_coil:
            with self.coil_lock:
                # Clear conflicting momentary command coils before asserting new pulse
                for c_addr in (
                    COIL_BTN_START,
                    COIL_BTN_STOP,
                    COIL_BTN_EMG,
                    COIL_BTN_RESET,
                ):
                    if c_addr != target_coil:
                        self.coils[c_addr] = False
                        self.pulse_timers.pop(c_addr, None)
                self.coils[target_coil] = True
                self.pulse_timers[target_coil] = now_s

    def _run_pulse_supervisor(self):
        while self.running:
            now_s = time.time()
            expired = []
            with self.coil_lock:
                for coil_addr, start_s in list(self.pulse_timers.items()):
                    if now_s - start_s >= MOMENTARY_PULSE_DURATION_S:
                        self.coils[coil_addr] = False
                        expired.append(coil_addr)
                for coil_addr in expired:
                    del self.pulse_timers[coil_addr]
            time.sleep(0.010)

    def _run_modbus_server(self):
        self.modbus_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.modbus_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.modbus_sock.bind(("127.0.0.1", self.modbus_port))
        self.modbus_sock.listen(5)
        self.modbus_sock.settimeout(1.0)

        while self.running:
            try:
                conn, _ = self.modbus_sock.accept()
                client_thread = threading.Thread(
                    target=self._handle_modbus_client, args=(conn,), daemon=True
                )
                client_thread.start()
            except socket.timeout:
                continue
            except Exception:
                break

    def _handle_modbus_client(self, conn: socket.socket):
        conn.settimeout(2.0)
        while self.running:
            try:
                header = conn.recv(6)
                if not header or len(header) < 6:
                    break
                tid, pid, length = struct.unpack(">HHH", header)
                pdu = conn.recv(length)
                if not pdu or len(pdu) < length:
                    break

                unit_id = pdu[0]
                func_code = pdu[1]

                # FC 01: Read Coils
                if func_code == 0x01:
                    start_addr, count = struct.unpack(">HH", pdu[2:6])
                    byte_count = (count + 7) // 8
                    coil_bytes = bytearray(byte_count)

                    with self.coil_lock:
                        for i in range(count):
                            addr = start_addr + i
                            if self.coils.get(addr, False):
                                coil_bytes[i // 8] |= 1 << (i % 8)

                    resp_pdu = bytes([unit_id, func_code, byte_count]) + coil_bytes
                    resp_hdr = struct.unpack(">HH", header[0:4])
                    resp = (
                        struct.pack(
                            ">HHHB", resp_hdr[0], resp_hdr[1], len(resp_pdu), unit_id
                        )
                        + resp_pdu[1:]
                    )
                    conn.sendall(resp)

                # FC 05: Write Single Coil
                elif func_code == 0x05:
                    coil_addr, val = struct.unpack(">HH", pdu[2:6])
                    coil_val = val == 0xFF00
                    with self.coil_lock:
                        self.coils[coil_addr] = coil_val

                    # Echo request as standard Modbus response
                    resp = header + pdu
                    conn.sendall(resp)

                # FC 03: Read Holding Registers
                elif func_code == 0x03:
                    start_addr, count = struct.unpack(">HH", pdu[2:6])
                    byte_count = count * 2
                    resp_pdu = bytes([unit_id, func_code, byte_count]) + bytes(
                        byte_count
                    )
                    resp = struct.pack(">HHH", tid, pid, len(resp_pdu)) + resp_pdu
                    conn.sendall(resp)

                else:
                    # Modbus Exception: Illegal Function
                    err_pdu = bytes([unit_id, func_code | 0x80, 0x01])
                    resp = struct.pack(">HHH", tid, pid, len(err_pdu)) + err_pdu
                    conn.sendall(resp)

            except (socket.timeout, ConnectionResetError, BrokenPipeError):
                break
            except Exception:
                break
        conn.close()


# =======================================================================
# NODE 2: VIRTUAL MITSUBISHI FX5U PLC CLIENT
# =======================================================================
class VirtualMitsubishiPLC:
    """Emulates Mitsubishi FX5U PLC polling ESP32 Modbus coils and updating actuators."""

    def __init__(
        self, modbus_port: int = MODBUS_PORT, scan_time_s: float = PLC_SCAN_CYCLE_S
    ):
        self.modbus_port = modbus_port
        self.scan_time_s = scan_time_s
        self.running = False
        self.thread: Optional[threading.Thread] = None

        self.emergency_latched: bool = False
        self.machine_running: bool = False
        self.last_command_reaction_time_ms: float = 0.0
        self.scan_count = 0

    def start(self):
        self.running = True
        self.thread = threading.Thread(
            target=self._run_plc_cycle, name="FX5U-PLC-Scan", daemon=True
        )
        self.thread.start()

    def stop(self):
        self.running = False

    def _read_coils(
        self, sock: socket.socket, start_addr: int, count: int
    ) -> List[bool]:
        tid = 1
        req = struct.pack(">HHHBBHH", tid, 0, 6, 1, 0x01, start_addr, count)
        sock.sendall(req)
        hdr = sock.recv(6)
        if len(hdr) < 6:
            return [False] * count
        _, _, length = struct.unpack(">HHH", hdr)
        pdu = sock.recv(length)
        if len(pdu) < 3:
            return [False] * count
        byte_count = pdu[2]
        data = pdu[3 : 3 + byte_count]

        result = []
        for i in range(count):
            byte_idx = i // 8
            bit_idx = i % 8
            val = (
                bool((data[byte_idx] >> bit_idx) & 1) if byte_idx < len(data) else False
            )
            result.append(val)
        return result

    def _write_coil(self, sock: socket.socket, coil_addr: int, value: bool):
        tid = 2
        raw_val = 0xFF00 if value else 0x0000
        req = struct.pack(">HHHBBHH", tid, 0, 6, 1, 0x05, coil_addr, raw_val)
        sock.sendall(req)
        sock.recv(12)  # Read echo response

    def _run_plc_cycle(self):
        while self.running:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(("127.0.0.1", self.modbus_port))
                sock.settimeout(1.0)
                break
            except Exception:
                time.sleep(0.05)
                continue

        self.machine_running = False
        self.emergency_latched = False

        while self.running:
            t_scan_start = time.perf_counter()
            self.scan_count += 1

            try:
                # Read command coils 112 to 116 (5 coils)
                coils = self._read_coils(sock, COIL_BTN_START, 5)
                btn_start = coils[0]
                btn_stop = coils[1]
                btn_emg = coils[2]
                btn_reset = coils[3]
                user_ok = coils[4]

                t_react_start = time.perf_counter()

                # Safety-Dominant PLC Ladder Logic Evaluation Order:
                # 1. Emergency Stop dominates all other inputs and latches fault
                if btn_emg:
                    self.emergency_latched = True
                    self.machine_running = False
                    self._write_coil(sock, COIL_STS_RUNNING, False)
                    self._write_coil(sock, COIL_STS_STOPPED, True)
                    self._write_coil(sock, COIL_STS_EMG, True)
                    self.last_command_reaction_time_ms = (
                        time.perf_counter() - t_react_start
                    ) * 1000

                # 2. Fault Reset unlatches emergency while keeping machine stopped
                elif btn_reset and self.emergency_latched:
                    self.emergency_latched = False
                    self.machine_running = False
                    self._write_coil(sock, COIL_STS_RUNNING, False)
                    self._write_coil(sock, COIL_STS_STOPPED, True)
                    self._write_coil(sock, COIL_STS_EMG, False)
                    self.last_command_reaction_time_ms = (
                        time.perf_counter() - t_react_start
                    ) * 1000

                # 3. Normal operation when not in emergency: STOP always dominates START
                elif not self.emergency_latched:
                    if btn_stop:
                        self.machine_running = False
                        self._write_coil(sock, COIL_STS_RUNNING, False)
                        self._write_coil(sock, COIL_STS_STOPPED, True)
                        self.last_command_reaction_time_ms = (
                            time.perf_counter() - t_react_start
                        ) * 1000
                    elif btn_start:
                        if user_ok:
                            self.machine_running = True
                            self._write_coil(sock, COIL_STS_RUNNING, True)
                            self._write_coil(sock, COIL_STS_STOPPED, False)
                            self.last_command_reaction_time_ms = (
                                time.perf_counter() - t_react_start
                            ) * 1000

            except Exception:
                time.sleep(0.05)

            elapsed = time.perf_counter() - t_scan_start
            sleep_time = max(0.001, self.scan_time_s - elapsed)
            time.sleep(sleep_time)

        try:
            sock.close()
        except Exception:
            pass


# =======================================================================
# NODE 3: VIRTUAL PC FACE RECOGNITION HMI & WORKLOAD GENERATOR
# =======================================================================
class VirtualHMIClient:
    def __init__(self, http_port: int = HTTP_PORT):
        self.base_url = f"http://127.0.0.1:{http_port}"
        self.records: List[TimingRecord] = []

    def send_action(self, cmd: str) -> TimingRecord:
        t0 = time.perf_counter()
        success = False
        duration_ms = 0.0
        internal_latency_us = 0

        url = f"{self.base_url}/action?type={cmd}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    success = True
            duration_ms = (time.perf_counter() - t0) * 1000
        except Exception:
            duration_ms = (time.perf_counter() - t0) * 1000

        # Query /status to obtain internal latency from ESP32
        try:
            req_status = urllib.request.Request(f"{self.base_url}/status")
            with urllib.request.urlopen(req_status, timeout=0.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                internal_latency_us = data.get("last_internal_latency_us", 0)
        except Exception:
            pass

        rec = TimingRecord(
            timestamp=datetime.datetime.now(datetime.timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            source="WebApp",
            command=cmd,
            duration_ms=round(duration_ms, 2),
            success=success,
            internal_latency_us=internal_latency_us,
        )
        self.records.append(rec)
        return rec

    def set_user_detection(self, detected: bool) -> bool:
        url = f"{self.base_url}/user_detection?detected={'1' if detected else '0'}"
        try:
            with urllib.request.urlopen(url, timeout=0.5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def get_status(self) -> Dict:
        url = f"{self.base_url}/status"
        try:
            with urllib.request.urlopen(url, timeout=0.5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return {}


# =======================================================================
# QOS METRICS CALCULATOR & EMULATION RUNNER
# =======================================================================
def compute_qos_statistics(records: List[TimingRecord]) -> Dict[str, float]:
    durations = [r.duration_ms for r in records if r.success]
    latencies = [r.internal_latency_us for r in records if r.internal_latency_us > 0]

    if not durations:
        return {}

    n = len(durations)
    mean_val = sum(durations) / n
    variance = sum((x - mean_val) ** 2 for x in durations) / n
    stddev = math.sqrt(variance)

    # Jitter: mean difference between consecutive chronological round trip times
    jitters = [abs(durations[i] - durations[i - 1]) for i in range(1, n)]
    mean_jitter = (sum(jitters) / len(jitters)) if jitters else 0.0

    durations.sort()
    p50 = durations[int(n * 0.50)]
    p95 = durations[min(n - 1, int(n * 0.95))]
    p99 = durations[min(n - 1, int(n * 0.99))]

    mean_internal_us = (sum(latencies) / len(latencies)) if latencies else 0.0

    return {
        "count": n,
        "success_rate_pct": (len(durations) / len(records)) * 100.0,
        "min_ms": durations[0],
        "mean_ms": mean_val,
        "median_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "max_ms": durations[-1],
        "stddev_ms": stddev,
        "jitter_ms": mean_jitter,
        "mean_internal_us": mean_internal_us,
    }


def run_emulation_benchmark(sample_count: int = 40):
    print("=" * 70)
    print("  LAUNCHING FULL HARDWARE EMULATION & QOS BENCHMARK")
    print("=" * 70)
    print(
        f"[Init] Starting Virtual ESP32 Bridge on HTTP:{HTTP_PORT} & Modbus:{MODBUS_PORT}..."
    )
    bridge = VirtualESP32Bridge(HTTP_PORT, MODBUS_PORT)
    bridge.start()

    print(
        f"[Init] Starting Virtual Mitsubishi FX5U PLC Client (Scan rate: {PLC_SCAN_CYCLE_S * 1000:.0f} ms)..."
    )
    plc = VirtualMitsubishiPLC(MODBUS_PORT, PLC_SCAN_CYCLE_S)
    plc.start()

    hmi = VirtualHMIClient(HTTP_PORT)
    time.sleep(0.2)

    print("\n--- PHASE 1: OPERATOR FACE RECOGNITION & INITIALIZATION ---")
    hmi.set_user_detection(True)
    print("[HMI] Face recognized: Authorized Operator Active (Coil 116 = True)")
    st = hmi.get_status()
    print(
        f"[PLC State] Running: {st.get('running')}, Stopped: {st.get('stopped')}, EMG: {st.get('emergency')}"
    )

    print("\n--- PHASE 2: FUNCTIONAL COMMAND VERIFICATION ---")
    cmds = ["start", "stop", "start", "emergency", "reset", "start"]
    for c in cmds:
        rec = hmi.send_action(c)
        time.sleep(0.05)  # PLC scan execution window
        st = hmi.get_status()
        print(
            f"  Command: {c:<10} | RTT: {rec.duration_ms:5.2f} ms | ESP32 Latency: {rec.internal_latency_us:4d} us | PLC Running: {st.get('running')} | EMG: {st.get('emergency')}"
        )
        time.sleep(0.05)

    print(
        f"\n--- PHASE 3: QOS PERFORMANCE BENCHMARK & LOAD STRESS ({sample_count} Iterations) ---"
    )
    test_cycle = ["start", "stop", "start", "emergency", "reset"]
    for i in range(sample_count):
        cmd = test_cycle[i % len(test_cycle)]
        rec = hmi.send_action(cmd)
        if (i + 1) % 10 == 0 or (i + 1) == sample_count:
            print(
                f"  Progress: [{i + 1:2d}/{sample_count}] | Last RTT: {rec.duration_ms:.2f} ms | ESP32 Latency: {rec.internal_latency_us} us"
            )
        time.sleep(0.02)  # High-rate burst

    print("\n[Teardown] Shutting down virtual nodes...")
    plc.stop()
    bridge.stop()

    # Save CSV
    TIMINGS_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    EMULATION_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(TIMINGS_OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "source",
                "command",
                "duration_ms",
                "success",
                "internal_latency_us",
            ]
        )
        for r in hmi.records:
            writer.writerow(
                [
                    r.timestamp,
                    r.source,
                    r.command,
                    f"{r.duration_ms:.2f}",
                    str(r.success),
                    r.internal_latency_us,
                ]
            )

    with open(EMULATION_LOG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "source",
                "command",
                "duration_ms",
                "success",
                "internal_latency_us",
            ]
        )
        for r in hmi.records:
            writer.writerow(
                [
                    r.timestamp,
                    r.source,
                    r.command,
                    f"{r.duration_ms:.2f}",
                    str(r.success),
                    r.internal_latency_us,
                ]
            )

    print(f"\n[Artifacts] Telemetry logged to:")
    print(f"  - {TIMINGS_OUTPUT_CSV}")
    print(f"  - {EMULATION_LOG_CSV}")

    # Compute Statistics
    stats = compute_qos_statistics(hmi.records)

    print("\n" + "=" * 70)
    print("  QOS BENCHMARK STATISTICAL SUMMARY")
    print("=" * 70)
    print(f"  Total Transactions Evaluated : {len(hmi.records)}")
    print(
        f"  Packet Delivery / Success Rate: {stats.get('success_rate_pct', 0.0):.2f}%"
    )
    print(f"  RTT Duration Min             : {stats.get('min_ms', 0.0):.2f} ms")
    print(f"  RTT Duration Mean            : {stats.get('mean_ms', 0.0):.2f} ms")
    print(f"  RTT Duration Median (P50)    : {stats.get('median_ms', 0.0):.2f} ms")
    print(f"  RTT Duration 95th Percentile : {stats.get('p95_ms', 0.0):.2f} ms")
    print(f"  RTT Duration 99th Percentile : {stats.get('p99_ms', 0.0):.2f} ms")
    print(f"  RTT Duration Max             : {stats.get('max_ms', 0.0):.2f} ms")
    print(f"  RTT Jitter (Mean Delta)      : {stats.get('jitter_ms', 0.0):.2f} ms")
    print(
        f"  ESP32 Internal Bridge Latency: {stats.get('mean_internal_us', 0.0):.1f} us"
    )
    print("=" * 70)


def _parse_cli_args(argv: List[str]) -> int:
    if any(a in ("-h", "--help") for a in argv[1:]):
        print(
            "Usage: python Emulation/system_emulator.py [sample_count | samples=<N>]\n"
            "\n"
            "Runs the multi-node SCADA hardware emulation (Virtual ESP32 Bridge, Virtual\n"
            "Mitsubishi FX5U PLC, and Virtual Face-Recognition HMI) and writes QoS\n"
            "telemetry CSVs.\n"
            "\n"
            "Arguments:\n"
            "  sample_count   Number of Phase 3 stress-test iterations (default: 40)\n"
            "  -h, --help     Show this help message and exit"
        )
        sys.exit(0)

    if len(argv) <= 1:
        return 40

    arg = argv[1]
    if "=" in arg:
        arg = arg.split("=", 1)[1]
    try:
        return max(1, int(arg))
    except ValueError:
        print(
            f"Invalid sample count: '{argv[1]}'. Expected positive integer.",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    count = _parse_cli_args(sys.argv)
    run_emulation_benchmark(count)
