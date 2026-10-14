#include "app_config.hpp"
#include "bridge_state.hpp"
#include "eth_driver.hpp"
#include "logger.hpp"
#include "modbus_bridge.hpp"
#include "web_server.hpp"

#if defined(ARDUINO) || defined(ESP_PLATFORM)
#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

static pbl::BridgeState g_bridge_state;
static pbl::EthernetDriver g_eth_driver;
static pbl::ModbusBridge g_modbus_bridge;
static pbl::WebServerManager g_web_server(g_bridge_state);

static TaskHandle_t s_modbus_task_handle = nullptr;
static TaskHandle_t s_wifi_task_handle = nullptr;

// =======================================================================
// FREERTOS TASK 1: MODBUS & ETHERNET (CORE 1 - HIGH PRIORITY)
// =======================================================================
void modbusTask(void *pvParameters) {
  pbl::Logger::log("[Core 1] Modbus & Ethernet task running");

  // Safe tick calculation: guaranteed >= 1 tick to prevent IDLE1 starvation and
  // TWDT reset
  const TickType_t delay_ticks = (pdMS_TO_TICKS(1) > 0) ? pdMS_TO_TICKS(1) : 1;

  for (;;) {
    uint32_t now_ms = millis();
    uint32_t now_us = static_cast<uint32_t>(micros());

    // 1. Check Ethernet Link and Socket diagnostics
    g_eth_driver.check_link_status(now_ms);
    g_eth_driver.check_sockets_and_maintain(
        now_ms, g_modbus_bridge.get_connection_count());

    // 2. Process Modbus bridge cycle (command consumption, coils, timers, PLC
    // feedback)
    g_modbus_bridge.update(g_bridge_state, now_ms, now_us);

    // Yield for OS scheduling & watchdog reset
    vTaskDelay(delay_ticks);
  }
}

// =======================================================================
// FREERTOS TASK 2: WI-FI & WEB SERVER (CORE 0 - PRO CORE)
// =======================================================================
void wifiTask(void *pvParameters) {
  pbl::Logger::log("[Core 0] Launching Wi-Fi & REST API...");
  g_web_server.init();

  const TickType_t delay_ticks = (pdMS_TO_TICKS(2) > 0) ? pdMS_TO_TICKS(2) : 1;

  for (;;) {
    uint32_t now_ms = millis();
    g_web_server.update(now_ms);

    // Yield to background Wi-Fi stack & watchdog reset
    vTaskDelay(delay_ticks);
  }
}

// =======================================================================
// SYSTEM SETUP
// =======================================================================
void setup() {
  Serial.begin(pbl::config::SERIAL_BAUD_RATE);
  while (!Serial && millis() < 2000) {
    delay(10);
  }

  pbl::Logger::init();
  pbl::Logger::log("==================================================");
  pbl::Logger::log("   PBL-MK2025055SB001 ESP32-S3 MODBUS TCP BRIDGE   ");
  pbl::Logger::log("==================================================");
  pbl::Logger::log("System Booting...");

  // Initialize W5500 Ethernet on main thread before tasks start
  g_eth_driver.init();

  // Initialize Modbus TCP Server on port 503
  g_modbus_bridge.init();

  // Launch FreeRTOS Core 1 Task: Modbus Server
  BaseType_t ret1 = xTaskCreatePinnedToCore(
      modbusTask, "ModbusTask", pbl::config::TASK_STACK_SIZE_BYTES, nullptr,
      pbl::config::MODBUS_TASK_PRIORITY, &s_modbus_task_handle,
      pbl::config::MODBUS_TASK_CORE);
  if (ret1 != pdPASS) {
    pbl::Logger::log("[FATAL] Failed to create ModbusTask on Core 1!");
  }

  // Launch FreeRTOS Core 0 Task: Wi-Fi REST Server
  BaseType_t ret2 = xTaskCreatePinnedToCore(
      wifiTask, "WifiTask", pbl::config::TASK_STACK_SIZE_BYTES, nullptr,
      pbl::config::WIFI_TASK_PRIORITY, &s_wifi_task_handle,
      pbl::config::WIFI_TASK_CORE);
  if (ret2 != pdPASS) {
    pbl::Logger::log("[FATAL] Failed to create WifiTask on Core 0!");
  }
}

void loop() {
  // Free setup task memory since FreeRTOS pinned tasks manage execution
  vTaskDelete(nullptr);
}

#else // Host / Test Stub Entry

int main() {
  pbl::Logger::init();
  pbl::Logger::log("Host target binary stub - run tests via CTest.");
  return 0;
}

#endif
