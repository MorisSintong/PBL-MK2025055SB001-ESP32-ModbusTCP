# GUI_REAL

Arduino sketch for the early-prototype stand-alone ESP32 web HMI.

## `GUI_REAL.ino`

- Creates a Wi-Fi **Access Point** (`MEKAPBL` / `mekagui1`).
- Starts a `WebServer` on port 80 with the `/action?type=...` route.
- Accepts `start`, `stop`, `emergency`, `reset` commands and currently
  serial-prints the corresponding status (placeholder for driving
  relays/LEDs/actuators).
- Responds with `OK - <type>` on success or `Error` if the `type` argument is
  missing.

This was an initial feasibility prototype. The final system instead runs the
HMI on the PC (`Interface/app.py`) and uses this ESP32 only for face-gated
command forwarding; the production firmware lives in `ESP32-Program/`.