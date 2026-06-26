# lib

Reserved for project-specific (private) libraries used by the ESP32 firmware.
Each library should live in its own subdirectory, e.g. `lib/MyLib/`, and
PlatformIO will automatically compile and link it.

Currently empty – all functionality for this project lives in
`src/main.cpp` and external libraries declared in `platformio.ini`
(`Ethernet_Generic`, `modbus-esp8266`).

See the PlatformIO Library Dependency Finder docs for more:
https://docs.platformio.org/page/librarymanager/ldf.html