#pragma once

#include "app_config.hpp"
#include <atomic>
#include <cstdint>
#include <string>

namespace pbl {

enum class CommandType { Start, Stop, Emergency, Reset, Unknown };

struct PlcStatus {
  bool running{false};
  bool stopped{false};
  bool emergency{false};
  uint32_t last_latency_us{0};
};

class BridgeState {
public:
  BridgeState() = default;

  // Trigger command from Web Server (Core 0)
  bool trigger_command(CommandType cmd, uint64_t timestamp_us);
  static CommandType parse_command_str(const std::string &cmd_str);

  // User detection from Face Recognition
  void set_user_detection(bool detected);
  bool get_user_detection() const;
  bool consume_user_detection_changed(bool &current_val);

  // Consume pending commands in Modbus Task (Core 1)
  bool consume_start_request(uint64_t current_time_us);
  bool consume_stop_request(uint64_t current_time_us);
  bool consume_emergency_request(uint64_t current_time_us);
  bool consume_reset_request(uint64_t current_time_us);

  // Momentary coil pulse timer management (Core 1)
  void start_pulse_btn1(uint32_t now_ms);
  void start_pulse_btn2(uint32_t now_ms);
  void start_pulse_btn3(uint32_t now_ms);
  void start_pulse_btn4(uint32_t now_ms);

  bool check_and_clear_pulse_btn1(
      uint32_t now_ms,
      uint32_t duration_ms = config::MOMENTARY_PULSE_DURATION_MS);
  bool check_and_clear_pulse_btn2(
      uint32_t now_ms,
      uint32_t duration_ms = config::MOMENTARY_PULSE_DURATION_MS);
  bool check_and_clear_pulse_btn3(
      uint32_t now_ms,
      uint32_t duration_ms = config::MOMENTARY_PULSE_DURATION_MS);
  bool check_and_clear_pulse_btn4(
      uint32_t now_ms,
      uint32_t duration_ms = config::MOMENTARY_PULSE_DURATION_MS);

  // PLC feedback status indicators (updated by Core 1, read by Core 0)
  void update_plc_status(bool running, bool stopped, bool emergency);
  PlcStatus get_plc_status() const;

  // Latency metrics
  uint32_t get_last_latency_us() const;

  // JSON serialization for HTTP /status endpoint
  std::string generate_status_json() const;

private:
  // Atomic command triggers
  std::atomic<bool> req_btn1_{false};
  std::atomic<bool> req_btn2_{false};
  std::atomic<bool> req_btn3_{false};
  std::atomic<bool> req_btn4_{false};

  // User detection state
  std::atomic<bool> user_detected_{false};
  std::atomic<bool> user_detected_changed_{false};

  // Latency timestamps
  std::atomic<uint64_t> action_start_time_us_{0};
  std::atomic<uint32_t> last_internal_latency_us_{0};

  // PLC status cache
  std::atomic<bool> plc_running_{false};
  std::atomic<bool> plc_stopped_{false};
  std::atomic<bool> plc_emergency_{false};

  // Momentary pulse timers (Core 1 only, no atomic required)
  uint32_t timer_btn1_ms_{0};
  uint32_t timer_btn2_ms_{0};
  uint32_t timer_btn3_ms_{0};
  uint32_t timer_btn4_ms_{0};

  void record_latency(uint64_t current_time_us);
};

} // namespace pbl
