# templates

Jinja2 HTML templates rendered by the Flask HMI in the parent `Interface/`
directory.

## Files

- `index.html` – Single-page HMI shown to the operator. It contains:
  - An MJPEG `<img>` element pointing at `/video_feed` for the live webcam
    stream with face-detection overlays.
  - A colored status bar (Access Granted / Denied / System Stop).
  - Start / Stop / Emergency / Reset control buttons that issue fetch
    requests to the Flask routes (`/command/<cmd>`).
  - A poller to `/check_status` that updates PLC running/stopped/emergency
    indicators and the connection state.
  - Operator registration & model-training UI.

This file is served by the `/` route in `app.py` / `AppImproved.py`.