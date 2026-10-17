#include "logger.hpp"

#if defined(ESP32S3_TARGET)
#include "app_config.hpp"
#include "esp32s3_hal.hpp"

namespace pbl {

void Logger::init() {
  uart_tx_wait_idle(0);
  const uint32_t clk_div = (40000000UL << 4) / config::SERIAL_BAUD_RATE;
  uart_div_modify(0, clk_div);
  ets_delay_us(500);
}

void Logger::log(const std::string &msg) { log(msg.c_str()); }

void Logger::log(const char *msg) {
  if (!msg) {
    return;
  }
  while (*msg) {
    uart_tx_one_char(static_cast<uint8_t>(*msg++));
  }
  uart_tx_one_char(static_cast<uint8_t>('\r'));
  uart_tx_one_char(static_cast<uint8_t>('\n'));
}

} // namespace pbl

#elif defined(ARDUINO) || defined(ESP_PLATFORM)
#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>

static SemaphoreHandle_t s_serial_mutex = nullptr;

namespace pbl {

void Logger::init() {
  if (!s_serial_mutex) {
    s_serial_mutex = xSemaphoreCreateMutex();
  }
}

void Logger::log(const std::string &msg) { log(msg.c_str()); }

void Logger::log(const char *msg) {
  if (s_serial_mutex && xSemaphoreTake(s_serial_mutex, portMAX_DELAY)) {
    Serial.println(msg);
    xSemaphoreGive(s_serial_mutex);
  } else {
    Serial.println(msg);
  }
}

} // namespace pbl

#else // Host / Simulation mode

#include <iostream>
#include <mutex>

static std::mutex s_host_mutex;

namespace pbl {

void Logger::init() {
  // No-op for host
}

void Logger::log(const std::string &msg) {
  std::lock_guard<std::mutex> lock(s_host_mutex);
  std::cout << msg << std::endl;
}

void Logger::log(const char *msg) {
  std::lock_guard<std::mutex> lock(s_host_mutex);
  std::cout << msg << std::endl;
}

} // namespace pbl

#endif
