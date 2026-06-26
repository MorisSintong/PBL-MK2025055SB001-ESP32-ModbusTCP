# test

Reserved for PlatformIO unit tests for the ESP32 firmware. Tests placed here
(e.g. `test/test_main.cpp`) are run with `pio test` on the target hardware or
under a native simulator depending on the environment configuration.

Currently empty – no automated tests have been written yet. Manual validation
is performed through the serial monitor and the `test_modbus.py` script in the
`Interface/` directory.