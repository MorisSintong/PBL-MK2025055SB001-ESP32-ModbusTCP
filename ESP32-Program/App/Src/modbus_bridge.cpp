#include "logger.hpp"
#include "modbus_bridge.hpp"

namespace pbl {

uint32_t ModbusBridge::connection_count_ = 0;

#if defined(ARDUINO) || defined(ESP_PLATFORM)

bool ModbusBridge::on_client_connect(IPAddress ip) {
  connection_count_++;
  Logger::log("[Modbus] >>> Client CONNECTED from: " + ip.toString());
  return true; // Accept connection
}

void ModbusBridge::init() {
  mb_.onConnect(ModbusBridge::on_client_connect);
  mb_.server(config::MODBUS_TCP_PORT);
  mb_.addCoil(0, false, config::MODBUS_COIL_COUNT);
  mb_.addHreg(0, 0, config::MODBUS_HREG_COUNT);
  Logger::log("Modbus TCP Server initialized on port " +
              std::to_string(config::MODBUS_TCP_PORT));
}

void ModbusBridge::update(BridgeState &state, uint32_t now_ms,
                          uint64_t now_us) {
  // 1. Process command requests (Atomic inter-core)
  if (state.consume_start_request(now_us)) {
    mb_.Coil(config::COIL_BTN_START, true);
    state.start_pulse_btn1(now_ms);
    Logger::log("BTN 1 (Start) Triggered. Latency: " +
                std::to_string(state.get_last_latency_us()) + " us");
  }

  if (state.consume_stop_request(now_us)) {
    mb_.Coil(config::COIL_BTN_STOP, true);
    state.start_pulse_btn2(now_ms);
    Logger::log("BTN 2 (Stop) Triggered. Latency: " +
                std::to_string(state.get_last_latency_us()) + " us");
  }

  if (state.consume_emergency_request(now_us)) {
    mb_.Coil(config::COIL_BTN_EMERGENCY, true);
    state.start_pulse_btn3(now_ms);
    Logger::log("BTN 3 (Emergency) Triggered. Latency: " +
                std::to_string(state.get_last_latency_us()) + " us");
  }

  if (state.consume_reset_request(now_us)) {
    mb_.Coil(config::COIL_BTN_RESET, true);
    state.start_pulse_btn4(now_ms);
    Logger::log("Reset Triggered. Latency: " +
                std::to_string(state.get_last_latency_us()) + " us");
  }

  // 2. Process face recognition user detection flag
  bool detected_val = false;
  if (state.consume_user_detection_changed(detected_val)) {
    mb_.Coil(config::COIL_USER_DETECTED, detected_val);
    Logger::log(std::string("User Detection Coil: ") +
                (detected_val ? "HIGH" : "LOW"));
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

#else // Host / Simulation mode

void ModbusBridge::init() {
  Logger::log("ModbusBridge (Host Mock) initialized");
}

void ModbusBridge::update(BridgeState &state, uint32_t now_ms,
                          uint64_t now_us) {
  if (state.consume_start_request(now_us)) {
    state.start_pulse_btn1(now_ms);
  }
  if (state.consume_stop_request(now_us)) {
    state.start_pulse_btn2(now_ms);
  }
  if (state.consume_emergency_request(now_us)) {
    state.start_pulse_btn3(now_ms);
  }
  if (state.consume_reset_request(now_us)) {
    state.start_pulse_btn4(now_ms);
  }

  bool detected = false;
  state.consume_user_detection_changed(detected);

  state.check_and_clear_pulse_btn1(now_ms);
  state.check_and_clear_pulse_btn2(now_ms);
  state.check_and_clear_pulse_btn3(now_ms);
  state.check_and_clear_pulse_btn4(now_ms);
}

#endif

} // namespace pbl
