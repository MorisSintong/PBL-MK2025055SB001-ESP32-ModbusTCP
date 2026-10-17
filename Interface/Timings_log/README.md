# Timings_log

Archived timing measurements from interface/ESP32 round-trip tests and host multi-node emulation benchmarks.
Each CSV records `timestamp, source, command, duration_ms, success,
internal_latency_us` rows produced by the `_log_timing_csv` helper in the
interface scripts and the multi-node emulator.

## Files

- `timings.csv` – Early baseline run log (captured prior to ESP32 internal bridge latency instrumentation).
- `timings1.csv`, `timings2.csv` – Physical hardware baseline runs with internal bridge latency tracking (`915.7 us` and `675.1 us`).
- `timings_Jauh.csv` – Physical tests performed at extended distance between PC and ESP32 (`817.7 us` internal latency, 100% packet delivery).
- `timings_emulated.csv` – High-frequency multi-node host emulation benchmark (56 transactions, 100% success rate, `17.00 ms` mean RTT, `3.9 us` bridge latency).

These datasets are analyzed in detail in [`test_results.md`](../../test_results.md).
