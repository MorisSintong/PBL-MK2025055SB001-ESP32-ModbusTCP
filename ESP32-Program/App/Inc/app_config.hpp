#pragma once

#include <cstddef>
#include <cstdint>

namespace pbl {
namespace config {

// =======================================================================
// HARDWARE SPI & W5500 PIN CONFIGURATION (ESP32-S3)
// =======================================================================
constexpr uint8_t ETH_CS_PIN = 10;
constexpr uint8_t ETH_MOSI_PIN = 11;
constexpr uint8_t ETH_SCLK_PIN = 12;
constexpr uint8_t ETH_MISO_PIN = 13;

// SPI Frequency for W5500: 8 MHz for maximum reliability in industrial
// multi-drop SPI
constexpr uint32_t ETH_SPI_FREQ_HZ = 8000000;

// =======================================================================
// ETHERNET NETWORK CONFIGURATION
// =======================================================================
constexpr uint8_t ETH_MAC[6] = {0x02, 0xAB, 0xCD, 0x01, 0x02, 0x03};

// Static IP: 192.168.1.50
constexpr uint8_t ETH_IP[4] = {192, 168, 1, 50};
constexpr uint8_t ETH_GATEWAY[4] = {192, 168, 1, 1};
constexpr uint8_t ETH_SUBNET[4] = {255, 255, 255, 0};
constexpr uint8_t ETH_DNS[4] = {192, 168, 1, 1};

constexpr uint16_t ETH_TIMEOUT_MS = 200; // Retransmission timeout (ms)
constexpr uint8_t ETH_RETRIES = 4;       // Retransmission count

// =======================================================================
// MODBUS TCP CONFIGURATION (MITSUBISHI FX5U PLC INTERFACE)
// =======================================================================
constexpr uint16_t MODBUS_TCP_PORT = 503;

// Command Coils (PC Interface -> ESP32 -> PLC)
constexpr uint16_t COIL_BTN_START = 112;     // Momentary Start
constexpr uint16_t COIL_BTN_STOP = 113;      // Momentary Stop
constexpr uint16_t COIL_BTN_EMERGENCY = 114; // Momentary Emergency Stop
constexpr uint16_t COIL_BTN_RESET = 115;     // Momentary Fault Reset
constexpr uint16_t COIL_USER_DETECTED =
    116; // Continuous Latch (Face Recognition)

// Status Coils (PLC -> ESP32 -> PC Interface)
constexpr uint16_t COIL_STS_RUNNING = 208; // Machine is running
constexpr uint16_t COIL_STS_STOPPED = 209; // Machine is stopped
constexpr uint16_t COIL_STS_EMG = 210;     // Emergency active

// Register Table Allocation Limits
constexpr uint16_t MODBUS_COIL_COUNT = 300;
constexpr uint16_t MODBUS_HREG_COUNT = 300;

// Pulse timing for momentary command coils (milliseconds)
constexpr uint32_t MOMENTARY_PULSE_DURATION_MS = 500;

// =======================================================================
// WI-FI & REST API CONFIGURATION
// =======================================================================
constexpr const char *WIFI_SSID = "TestEsp32";
constexpr const char *WIFI_PASSWORD = "12345678";

// Wi-Fi Static IP: 10.251.64.200
constexpr uint8_t WIFI_IP[4] = {10, 251, 64, 200};
constexpr uint8_t WIFI_GATEWAY[4] = {10, 251, 64, 8};
constexpr uint8_t WIFI_SUBNET[4] = {255, 255, 255, 0};
constexpr uint8_t WIFI_DNS[4] = {10, 251, 64, 8};

constexpr const char *MDNS_HOSTNAME = "esp32-server";
constexpr uint16_t WEB_SERVER_PORT = 80;

// =======================================================================
// DIAGNOSTIC & SUPERVISION INTERVALS (MILLISECONDS)
// =======================================================================
constexpr uint32_t ETH_LINK_CHECK_INTERVAL_MS = 2000;
constexpr uint32_t ETH_SOCKET_CHECK_INTERVAL_MS = 3000;
constexpr uint32_t WIFI_RECONNECT_INTERVAL_MS = 10000;

// =======================================================================
// FREERTOS TASK ALLOCATION
// =======================================================================
constexpr uint32_t TASK_STACK_SIZE_BYTES = 8192;

constexpr uint32_t MODBUS_TASK_CORE = 1;     // App Core (Fast Modbus execution)
constexpr uint32_t MODBUS_TASK_PRIORITY = 2; // High priority

constexpr uint32_t WIFI_TASK_CORE = 0;     // Pro Core (Network stack & HTTP)
constexpr uint32_t WIFI_TASK_PRIORITY = 1; // Normal priority

constexpr uint32_t SERIAL_BAUD_RATE = 921600;

} // namespace config
} // namespace pbl
