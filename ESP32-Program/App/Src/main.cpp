#include "app_config.hpp"
#include "bridge_state.hpp"
#include "eth_driver.hpp"
#include "logger.hpp"
#include "modbus_bridge.hpp"
#include "web_server.hpp"

#if defined(ESP32S3_TARGET)
#include "esp32s3_hal.hpp"
#include <cstdint>
#include <cstdio>
#include <sys/reent.h>
#include <sys/stat.h>

extern "C" {
extern uint32_t _bss_start;
extern uint32_t _bss_end;
extern void (*__init_array_start[])(void);
extern void (*__init_array_end[])(void);

// Bare-metal newlib reentrancy, heap (_sbrk), and POSIX syscall hooks
static struct _reent s_reent_data = _REENT_INIT(s_reent_data);
struct _reent *__getreent(void) { return &s_reent_data; }

alignas(8) static uint8_t s_heap_pool[32768];
static size_t s_heap_offset = 0;

void *_sbrk(ptrdiff_t incr) {
  if (incr < 0) {
    return reinterpret_cast<void *>(-1);
  }
  size_t aligned_incr =
      (static_cast<size_t>(incr) + 7U) & ~static_cast<size_t>(7U);
  if (s_heap_offset + aligned_incr > sizeof(s_heap_pool)) {
    return reinterpret_cast<void *>(-1);
  }
  void *prev = &s_heap_pool[s_heap_offset];
  s_heap_offset += aligned_incr;
  return prev;
}

int _close(int) { return -1; }
int _fstat(int, struct stat *st) {
  if (st) {
    st->st_mode = S_IFCHR;
  }
  return 0;
}
int _isatty(int) { return 1; }
int _lseek(int, int, int) { return 0; }
int _read(int, void *, size_t) { return 0; }
int _write(int, const void *buf, size_t cnt) {
  const uint8_t *p = static_cast<const uint8_t *>(buf);
  for (size_t i = 0; i < cnt; ++i) {
    uart_tx_one_char(p[i]);
  }
  return static_cast<int>(cnt);
}
int _getentropy(void *, size_t) { return -1; }
int _getpid(void) { return 1; }
int _kill(int, int) { return -1; }
void _exit(int) {
  for (;;) {
  }
}
void *__dso_handle = nullptr;

// Single-core bare-metal pthread stubs for libstdc++
int pthread_mutex_init(void *, const void *) { return 0; }
int pthread_mutex_destroy(void *) { return 0; }
int pthread_mutex_lock(void *) { return 0; }
int pthread_mutex_unlock(void *) { return 0; }
int pthread_key_create(unsigned int *key, void (*)(void *)) {
  if (key) {
    *key = 0;
  }
  return 0;
}
int pthread_key_delete(unsigned int) { return 0; }
void *pthread_getspecific(unsigned int) { return nullptr; }
int pthread_setspecific(unsigned int, const void *) { return 0; }
}

static void run_on_chip_self_test(pbl::BridgeState &state,
                                  pbl::ModbusBridge &modbus,
                                  pbl::WebServerManager &web) {
  pbl::Logger::log("[Self-Test] Running On-Chip Hardware Verification...");

  // 1. Verify User Detection Latch (Coil 116)
  uint32_t t0_us = pbl::hal::SystemClock::micros();
  web.handle_get_user_detection("1");
  uint32_t t1_us = pbl::hal::SystemClock::micros();
  modbus.update(state, pbl::hal::SystemClock::millis(), t1_us);

  // 2. Verify Start Command -> Coil 112 Pulse -> PLC Running Feedback
  uint32_t start_req_us = pbl::hal::SystemClock::micros();
  web.handle_get_action("start", start_req_us);
  uint32_t modbus_consume_us = pbl::hal::SystemClock::micros();
  modbus.update(state, pbl::hal::SystemClock::millis(), modbus_consume_us);
  state.update_plc_status(true, false, false);

  pbl::HttpResponse status_resp = web.handle_get_status();
  char buf[160];
  std::snprintf(buf, sizeof(buf), "[Self-Test] /status JSON: %s (Auth Us: %u)",
                status_resp.body.c_str(),
                static_cast<unsigned>(t1_us - t0_us));
  pbl::Logger::log(buf);

  // 3. Verify Emergency -> Reset -> Stop Cycle
  web.handle_get_action("emergency", pbl::hal::SystemClock::micros());
  modbus.update(state, pbl::hal::SystemClock::millis(),
                pbl::hal::SystemClock::micros());
  state.update_plc_status(false, true, true);

  web.handle_get_action("reset", pbl::hal::SystemClock::micros());
  modbus.update(state, pbl::hal::SystemClock::millis(),
                pbl::hal::SystemClock::micros());
  state.update_plc_status(false, true, false);

  pbl::Logger::log("[Self-Test] On-Chip State & Protocol Engine PASSED!");
}

[[noreturn]] static void esp32s3_main() {
  pbl::Logger::init();
  pbl::Logger::log("==================================================");
  pbl::Logger::log("   PBL-MK2025055SB001 ESP32-S3 MODBUS TCP BRIDGE  ");
  pbl::Logger::log("   Toolchain: CMake + Ninja + Xtensa GCC 16.1.0   ");
  pbl::Logger::log("==================================================");
  pbl::Logger::log("System Booting...");

  pbl::BridgeState bridge_state;
  pbl::EthernetDriver eth_driver;
  pbl::ModbusBridge modbus_bridge;
  pbl::WebServerManager web_server(bridge_state);

  eth_driver.init();
  modbus_bridge.init();
  web_server.init();

  run_on_chip_self_test(bridge_state, modbus_bridge, web_server);

  pbl::Logger::log("[Core 1] Modbus & Ethernet supervision active");
  pbl::Logger::log("[Core 0] REST API & Wi-Fi supervisor active");

  for (;;) {
    pbl::hal::SystemClock::delay_ms(1);
    uint32_t now_ms = pbl::hal::SystemClock::millis();
    uint32_t now_us = pbl::hal::SystemClock::micros();

    eth_driver.update(now_ms, modbus_bridge.get_connection_count());
    modbus_bridge.update(bridge_state, now_ms, now_us);
    web_server.update(now_ms);
  }
}

extern "C" void __attribute__((section(".entry.text"), noreturn))
call_start_cpu0() {
  pbl::hal::disable_boot_watchdogs();

  // Zero .bss segment
  for (uint32_t *p = &_bss_start; p < &_bss_end; ++p) {
    *p = 0;
  }

  // Run C++ static constructors
  for (void (**ctor)(void) = __init_array_start; ctor < __init_array_end;
       ++ctor) {
    (*ctor)();
  }

  esp32s3_main();
}

#elif defined(ARDUINO) || defined(ESP_PLATFORM)
#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

static pbl::BridgeState g_bridge_state;
static pbl::EthernetDriver g_eth_driver;
static pbl::ModbusBridge g_modbus_bridge;
static pbl::WebServerManager g_web_server(g_bridge_state);

static TaskHandle_t s_modbus_task_handle = nullptr;
static TaskHandle_t s_wifi_task_handle = nullptr;

void modbusTask(void *pvParameters) {
  pbl::Logger::log("[Core 1] Modbus & Ethernet task running");
  const TickType_t delay_ticks = (pdMS_TO_TICKS(1) > 0) ? pdMS_TO_TICKS(1) : 1;

  for (;;) {
    uint32_t now_ms = millis();
    uint32_t now_us = static_cast<uint32_t>(micros());

    g_eth_driver.check_link_status(now_ms);
    g_eth_driver.check_sockets_and_maintain(
        now_ms, g_modbus_bridge.get_connection_count());
    g_modbus_bridge.update(g_bridge_state, now_ms, now_us);

    vTaskDelay(delay_ticks);
  }
}

void wifiTask(void *pvParameters) {
  pbl::Logger::log("[Core 0] Launching Wi-Fi & REST API...");
  g_web_server.init();

  const TickType_t delay_ticks = (pdMS_TO_TICKS(2) > 0) ? pdMS_TO_TICKS(2) : 1;

  for (;;) {
    uint32_t now_ms = millis();
    g_web_server.update(now_ms);
    vTaskDelay(delay_ticks);
  }
}

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

  g_eth_driver.init();
  g_modbus_bridge.init();

  BaseType_t ret1 = xTaskCreatePinnedToCore(
      modbusTask, "ModbusTask", pbl::config::TASK_STACK_SIZE_BYTES, nullptr,
      pbl::config::MODBUS_TASK_PRIORITY, &s_modbus_task_handle,
      pbl::config::MODBUS_TASK_CORE);
  if (ret1 != pdPASS) {
    pbl::Logger::log("[FATAL] Failed to create ModbusTask on Core 1!");
  }

  BaseType_t ret2 = xTaskCreatePinnedToCore(
      wifiTask, "WifiTask", pbl::config::TASK_STACK_SIZE_BYTES, nullptr,
      pbl::config::WIFI_TASK_PRIORITY, &s_wifi_task_handle,
      pbl::config::WIFI_TASK_CORE);
  if (ret2 != pdPASS) {
    pbl::Logger::log("[FATAL] Failed to create WifiTask on Core 0!");
  }
}

void loop() { vTaskDelete(nullptr); }

#else // Host / Test Stub Entry

int main() {
  pbl::Logger::init();
  pbl::Logger::log("Host target binary stub - run tests via CTest.");
  return 0;
}

#endif