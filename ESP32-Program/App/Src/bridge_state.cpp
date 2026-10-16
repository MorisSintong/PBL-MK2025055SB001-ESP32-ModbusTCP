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
  switch (cmd) {
  case CommandType::Start:
    action_start_time_us_[0].store(timestamp_us, std::memory_order_relaxed);
    req_btn1_.store(true, std::memory_order_release);
    return true;
  case CommandType::Stop:
    action_start_time_us_[1].store(timestamp_us, std::memory_order_relaxed);
    req_btn2_.store(true, std::memory_order_release);
    return true;
  case CommandType::Emergency:
    action_start_time_us_[2].store(timestamp_us, std::memory_order_relaxed);
    req_btn3_.store(true, std::memory_order_release);
    return true;
  case CommandType::Reset:
    action_start_time_us_[3].store(timestamp_us, std::memory_order_relaxed);
    req_btn4_.store(true, std::memory_order_release);
    return true;
  default:
    return false;
  }
}

void BridgeState::set_user_detection(bool detected) {
  const uint8_t target_val = detected ? USER_DETECT_VAL_BIT : 0x00;
  uint8_t cur = user_detect_mask_.load(std::memory_order_relaxed);
  while ((cur & USER_DETECT_VAL_BIT) != target_val) {
    const uint8_t next = target_val | USER_DETECT_DIRTY_BIT;
    if (user_detect_mask_.compare_exchange_weak(
            cur, next, std::memory_order_acq_rel, std::memory_order_relaxed)) {
      break;
    }
  }
}

bool BridgeState::get_user_detection() const {
  return (user_detect_mask_.load(std::memory_order_acquire) &
          USER_DETECT_VAL_BIT) != 0;
}

bool BridgeState::consume_user_detection_changed(bool &current_val) {
  uint8_t prev = user_detect_mask_.fetch_and(USER_DETECT_VAL_BIT,
                                             std::memory_order_acq_rel);
  if ((prev & USER_DETECT_DIRTY_BIT) != 0) {
    current_val = (prev & USER_DETECT_VAL_BIT) != 0;
    return true;
  }
  return false;
}

void BridgeState::record_latency(size_t channel_idx, uint32_t current_time_us) {
  uint32_t start =
      action_start_time_us_[channel_idx].load(std::memory_order_relaxed);
  if (start != 0) {
    uint32_t elapsed = current_time_us - start;
    // RFC 1982 serial number arithmetic: handles 32-bit micros() wrap-around
    // seamlessly while rejecting backward clock steps (where elapsed > 2^31-1)
    if (elapsed <= 0x7FFFFFFFu) {
      last_internal_latency_us_.store(elapsed, std::memory_order_relaxed);
    }
  }
}

bool BridgeState::consume_start_request(uint32_t current_time_us) {
  if (req_btn1_.exchange(false, std::memory_order_acq_rel)) {
    record_latency(0, current_time_us);
    return true;
  }
  return false;
}

bool BridgeState::consume_stop_request(uint32_t current_time_us) {
  if (req_btn2_.exchange(false, std::memory_order_acq_rel)) {
    record_latency(1, current_time_us);
    return true;
  }
  return false;
}

bool BridgeState::consume_emergency_request(uint32_t current_time_us) {
  if (req_btn3_.exchange(false, std::memory_order_acq_rel)) {
    record_latency(2, current_time_us);
    return true;
  }
  return false;
}

bool BridgeState::consume_reset_request(uint32_t current_time_us) {
  if (req_btn4_.exchange(false, std::memory_order_acq_rel)) {
    record_latency(3, current_time_us);
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
