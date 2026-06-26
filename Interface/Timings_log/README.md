# Timings_log

Archived timing measurements from earlier interface/ESP32 round-trip tests.
Each CSV records `timestamp, source, command, duration_ms, success,
internal_latency_us` rows produced by the `_log_timing_csv` helper in the
interface scripts.

## Files

- `timings.csv` – General run log (default output path used by the scripts).
- `timings1.csv`, `timings2.csv` – Additional runs captured during testing.
- `timings_Jauh.csv` – Tests performed with a larger physical distance between
  the PC/ESP32 and the PLC (latency under range/relay conditions).

These files are kept for reference and for plotting performance in the project
report. New runs will recreate `timings.csv` in the `Interface/` working
directory unless the `_csv_path` variable is changed inside the scripts.