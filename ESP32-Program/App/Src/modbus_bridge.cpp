#include "logger.hpp"
#include "modbus_bridge.hpp"
#include <cstdio>

namespace pbl {

uint32_t ModbusBridge::connection_count_ = 0;

#if defined(ARDUINO) || defined(ESP_PLATFORM)

bool ModbusBridge::on_client_connect(IPAddress ip) {
  connection_count_++;
  char log_buf[128];
  std::snprintf(log_buf, sizeof(log_buf),
                "[Modbus] >>> Client CONNECTED from: %s",
                ip.toString().c_str());
  Logger::log(log_buf);
  return true; // Accept connection
}

void ModbusBridge::init() {
  mb_.onConnect(ModbusBridge::on_client_connect);
  mb_.server(config::MODBUS_TCP_PORT);
  mb_.addCoil(0, false, config::MODBUS_COIL_COUNT);
  mb_.addHreg(0, 0, config::MODBUS_HREG_COUNT);
  char buf[96];
  std::snprintf(buf, sizeof(buf), "Modbus TCP Server initialized on port %u",
                static_cast<unsigned>(config::MODBUS_TCP_PORT));
  Logger::log(buf);
}

void ModbusBridge::update(BridgeState &state, uint32_t now_ms,
                          uint32_t now_us) {
  char log_buf[96];
  // 1. Process command requests (Atomic inter-core)
  if (state.consume_start_request(now_us)) {
    mb_.Coil(config::COIL_BTN_START, true);
    state.start_pulse_btn1(now_ms);
    std::snprintf(log_buf, sizeof(log_buf),
                  "BTN 1 (Start) Triggered. Latency: %u us",
                  static_cast<unsigned>(state.get_last_latency_us()));
    Logger::log(log_buf);
  }

  if (state.consume_stop_request(now_us)) {
    mb_.Coil(config::COIL_BTN_STOP, true);
    state.start_pulse_btn2(now_ms);
    std::snprintf(log_buf, sizeof(log_buf),
                  "BTN 2 (Stop) Triggered. Latency: %u us",
                  static_cast<unsigned>(state.get_last_latency_us()));
    Logger::log(log_buf);
  }

  if (state.consume_emergency_request(now_us)) {
    mb_.Coil(config::COIL_BTN_EMERGENCY, true);
    state.start_pulse_btn3(now_ms);
    std::snprintf(log_buf, sizeof(log_buf),
                  "BTN 3 (Emergency) Triggered. Latency: %u us",
                  static_cast<unsigned>(state.get_last_latency_us()));
    Logger::log(log_buf);
  }

  if (state.consume_reset_request(now_us)) {
    mb_.Coil(config::COIL_BTN_RESET, true);
    state.start_pulse_btn4(now_ms);
    std::snprintf(log_buf, sizeof(log_buf), "Reset Triggered. Latency: %u us",
                  static_cast<unsigned>(state.get_last_latency_us()));
    Logger::log(log_buf);
  }

  // 2. Process face recognition user detection flag
  bool detected_val = false;
  if (state.consume_user_detection_changed(detected_val)) {
    mb_.Coil(config::COIL_USER_DETECTED, detected_val);
    Logger::log(detected_val ? "User Detection Coil: HIGH"
                             : "User Detection Coil: LOW");
  }

  // 3. Process Modbus TCP engine tasks
  mb_.task();

  // 4. Momentary coil pulse timers
  if (state.check_and_clear_pulse_btn1(now_ms)) {
    mb_.Coil(config::COIL_BTN_START, false);
  }
  if (state.check_and_clear_pulse_btn2(now_ms)) {
    mb_.Coil(config::COIL_BTN_STOP, false);
  }
  if (state.check_and_clear_pulse_btn3(now_ms)) {
    mb_.Coil(config::COIL_BTN_EMERGENCY, false);
  }
  if (state.check_and_clear_pulse_btn4(now_ms)) {
    mb_.Coil(config::COIL_BTN_RESET, false);
  }

  // 5. Read status feedback coils from PLC
  bool run = mb_.Coil(config::COIL_STS_RUNNING);
  bool stp = mb_.Coil(config::COIL_STS_STOPPED);
  bool emg = mb_.Coil(config::COIL_STS_EMG);
  state.update_plc_status(run, stp, emg);
}

#else // Native ESP32S3_TARGET or Host Simulation mode

void ModbusBridge::init() {
#if defined(ESP32S3_TARGET)
  char buf[96];
  std::snprintf(buf, sizeof(buf), "Modbus TCP Server initialized on port %u",
                static_cast<unsigned>(config::MODBUS_TCP_PORT));
  Logger::log(buf);
#else
  Logger::log("ModbusBridge (Host Mock) initialized");
#endif
}

void ModbusBridge::update(BridgeState &state, uint32_t now_ms,
                          uint32_t now_us) {
#if defined(ESP32S3_TARGET)
  char log_buf[96];
#endif
  if (state.consume_start_request(now_us)) {
    state.start_pulse_btn1(now_ms);
#if defined(ESP32S3_TARGET)
    std::snprintf(log_buf, sizeof(log_buf),
                  "[Core 1] BTN 1 (Start) Coil %u -> HIGH | Latency: %u us",
                  static_cast<unsigned>(config::COIL_BTN_START),
                  static_cast<unsigned>(state.get_last_latency_us()));
    Logger::log(log_buf);
#endif
  }
  if (state.consume_stop_request(now_us)) {
    state.start_pulse_btn2(now_ms);
#if defined(ESP32S3_TARGET)
    std::snprintf(log_buf, sizeof(log_buf),
                  "[Core 1] BTN 2 (Stop) Coil %u -> HIGH | Latency: %u us",
                  static_cast<unsigned>(config::COIL_BTN_STOP),
                  static_cast<unsigned>(state.get_last_latency_us()));
    Logger::log(log_buf);
#endif
  }
  if (state.consume_emergency_request(now_us)) {
    state.start_pulse_btn3(now_ms);
#if defined(ESP32S3_TARGET)
    std::snprintf(log_buf, sizeof(log_buf),
                  "[Core 1] BTN 3 (Emergency) Coil %u -> HIGH | Latency: %u us",
                  static_cast<unsigned>(config::COIL_BTN_EMERGENCY),
                  static_cast<unsigned>(state.get_last_latency_us()));
    Logger::log(log_buf);
#endif
  }
  if (state.consume_reset_request(now_us)) {
    state.start_pulse_btn4(now_ms);
#if defined(ESP32S3_TARGET)
    std::snprintf(log_buf, sizeof(log_buf),
                  "[Core 1] BTN 4 (Reset) Coil %u -> HIGH | Latency: %u us",
                  static_cast<unsigned>(config::COIL_BTN_RESET),
                  static_cast<unsigned>(state.get_last_latency_us()));
    Logger::log(log_buf);
#endif
  }

  bool detected = false;
  if (state.consume_user_detection_changed(detected)) {
#if defined(ESP32S3_TARGET)
    Logger::log(detected ? "[Core 1] User Detection Coil 116 -> HIGH"
                         : "[Core 1] User Detection Coil 116 -> LOW");
#endif
  }

  state.check_and_clear_pulse_btn1(now_ms);
  state.check_and_clear_pulse_btn2(now_ms);
  state.check_and_clear_pulse_btn3(now_ms);
  state.check_and_clear_pulse_btn4(now_ms);
}

#endif

} // namespace pbl
