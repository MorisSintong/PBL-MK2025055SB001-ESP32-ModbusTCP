#include "bridge_state.hpp"

namespace pbl {

CommandType BridgeState::parse_command_str(const std::string &cmd_str) {
  if (cmd_str == "start") {
    return CommandType::Start;
  } else if (cmd_str == "stop") {
    return CommandType::Stop;
  } else if (cmd_str == "emergency") {
    return CommandType::Emergency;
  } else if (cmd_str == "reset") {
    return CommandType::Reset;
  }
  return CommandType::Unknown;
}

bool BridgeState::trigger_command(CommandType cmd, uint32_t timestamp_us) {
  action_start_time_us_.store(timestamp_us, std::memory_order_relaxed);

  switch (cmd) {
  case CommandType::Start:
    req_btn1_.store(true, std::memory_order_release);
    return true;
  case CommandType::Stop:
    req_btn2_.store(true, std::memory_order_release);
    return true;
  case CommandType::Emergency:
    req_btn3_.store(true, std::memory_order_release);
    return true;
  case CommandType::Reset:
    req_btn4_.store(true, std::memory_order_release);
    return true;
  default:
    return false;
  }
}

void BridgeState::set_user_detection(bool detected) {
  bool current = user_detected_.load(std::memory_order_relaxed);
  if (current != detected) {
    user_detected_.store(detected, std::memory_order_relaxed);
    user_detected_changed_.store(true, std::memory_order_release);
  }
}

bool BridgeState::get_user_detection() const {
  return user_detected_.load(std::memory_order_relaxed);
}

bool BridgeState::consume_user_detection_changed(bool &current_val) {
  if (user_detected_changed_.exchange(false, std::memory_order_acq_rel)) {
    current_val = user_detected_.load(std::memory_order_acquire);
    return true;
  }
  return false;
}

void BridgeState::record_latency(uint32_t current_time_us) {
  uint32_t start = action_start_time_us_.load(std::memory_order_relaxed);
  if (start > 0 && current_time_us >= start) {
    uint32_t latency = current_time_us - start;
    last_internal_latency_us_.store(latency, std::memory_order_relaxed);
  }
}

bool BridgeState::consume_start_request(uint32_t current_time_us) {
  if (req_btn1_.exchange(false, std::memory_order_acq_rel)) {
    record_latency(current_time_us);
    return true;
  }
  return false;
}

bool BridgeState::consume_stop_request(uint32_t current_time_us) {
  if (req_btn2_.exchange(false, std::memory_order_acq_rel)) {
    record_latency(current_time_us);
    return true;
  }
  return false;
}

bool BridgeState::consume_emergency_request(uint32_t current_time_us) {
  if (req_btn3_.exchange(false, std::memory_order_acq_rel)) {
    record_latency(current_time_us);
    return true;
  }
  return false;
}

bool BridgeState::consume_reset_request(uint32_t current_time_us) {
  if (req_btn4_.exchange(false, std::memory_order_acq_rel)) {
    record_latency(current_time_us);
    return true;
  }
  return false;
}

void BridgeState::start_pulse_btn1(uint32_t now_ms) {
  timer_btn1_.active = true;
  timer_btn1_.start_ms = now_ms;
}

void BridgeState::start_pulse_btn2(uint32_t now_ms) {
  timer_btn2_.active = true;
  timer_btn2_.start_ms = now_ms;
}

void BridgeState::start_pulse_btn3(uint32_t now_ms) {
  timer_btn3_.active = true;
  timer_btn3_.start_ms = now_ms;
}

void BridgeState::start_pulse_btn4(uint32_t now_ms) {
  timer_btn4_.active = true;
  timer_btn4_.start_ms = now_ms;
}

bool BridgeState::check_and_clear_pulse_btn1(uint32_t now_ms,
                                             uint32_t duration_ms) {
  if (timer_btn1_.active && (now_ms - timer_btn1_.start_ms) >= duration_ms) {
    timer_btn1_.active = false;
    return true;
  }
  return false;
}

bool BridgeState::check_and_clear_pulse_btn2(uint32_t now_ms,
                                             uint32_t duration_ms) {
  if (timer_btn2_.active && (now_ms - timer_btn2_.start_ms) >= duration_ms) {
    timer_btn2_.active = false;
    return true;
  }
  return false;
}

bool BridgeState::check_and_clear_pulse_btn3(uint32_t now_ms,
                                             uint32_t duration_ms) {
  if (timer_btn3_.active && (now_ms - timer_btn3_.start_ms) >= duration_ms) {
    timer_btn3_.active = false;
    return true;
  }
  return false;
}

bool BridgeState::check_and_clear_pulse_btn4(uint32_t now_ms,
                                             uint32_t duration_ms) {
  if (timer_btn4_.active && (now_ms - timer_btn4_.start_ms) >= duration_ms) {
    timer_btn4_.active = false;
    return true;
  }
  return false;
}

void BridgeState::update_plc_status(bool running, bool stopped,
                                    bool emergency) {
  uint8_t mask = 0;
  if (running)
    mask |= STATUS_FLAG_RUNNING;
  if (stopped)
    mask |= STATUS_FLAG_STOPPED;
  if (emergency)
    mask |= STATUS_FLAG_EMERGENCY;
  plc_status_mask_.store(mask, std::memory_order_release);
}

PlcStatus BridgeState::get_plc_status() const {
  uint8_t mask = plc_status_mask_.load(std::memory_order_acquire);
  PlcStatus status;
  status.running = (mask & STATUS_FLAG_RUNNING) != 0;
  status.stopped = (mask & STATUS_FLAG_STOPPED) != 0;
  status.emergency = (mask & STATUS_FLAG_EMERGENCY) != 0;
  status.last_latency_us =
      last_internal_latency_us_.load(std::memory_order_relaxed);
  return status;
}

uint32_t BridgeState::get_last_latency_us() const {
  return last_internal_latency_us_.load(std::memory_order_relaxed);
}

std::string BridgeState::generate_status_json() const {
  PlcStatus st = get_plc_status();
  std::string json;
  json.reserve(128);
  json += "{\"running\":";
  json += (st.running ? "true" : "false");
  json += ",\"stopped\":";
  json += (st.stopped ? "true" : "false");
  json += ",\"emergency\":";
  json += (st.emergency ? "true" : "false");
  json += ",\"last_internal_latency_us\":";
  json += std::to_string(st.last_latency_us);
  json += "}";
  return json;
}

} // namespace pbl
