#pragma once

#include "app_config.hpp"
#include <atomic>
#include <cstddef>
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

struct PulseTimer {
  bool active{false};
  uint32_t start_ms{0};
};

class BridgeState {
public:
  static constexpr uint8_t STATUS_FLAG_RUNNING = 0x01;
  static constexpr uint8_t STATUS_FLAG_STOPPED = 0x02;
  static constexpr uint8_t STATUS_FLAG_EMERGENCY = 0x04;

  static constexpr uint8_t USER_DETECT_VAL_BIT = 0x01;
  static constexpr uint8_t USER_DETECT_DIRTY_BIT = 0x02;

  static constexpr size_t CHANNEL_COUNT = 4;

  BridgeState() = default;

  // Trigger command from Web Server (Core 0)
  bool trigger_command(CommandType cmd, uint32_t timestamp_us);
  static CommandType parse_command_str(const std::string &cmd_str);

  // User detection from Face Recognition (packed atomic value + dirty bit)
  void set_user_detection(bool detected);
  bool get_user_detection() const;
  bool consume_user_detection_changed(bool &current_val);

  // Consume pending commands in Modbus Task (Core 1)
  bool consume_start_request(uint32_t current_time_us);
  bool consume_stop_request(uint32_t current_time_us);
  bool consume_emergency_request(uint32_t current_time_us);
  bool consume_reset_request(uint32_t current_time_us);

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

  // Packed user detection state: bit 0 = detected value, bit 1 = changed/dirty
  std::atomic<uint8_t> user_detect_mask_{0};

  // Per-channel start timestamps & last recorded latency (32-bit native atomic)
  std::atomic<uint32_t> action_start_time_us_[CHANNEL_COUNT]{
      {0}, {0}, {0}, {0}};
  std::atomic<uint32_t> last_internal_latency_us_{0};

  // Atomic bitmask for PLC status (eliminates state tearing across cores)
  std::atomic<uint8_t> plc_status_mask_{0};

  // Momentary pulse timers (Core 1 only, with explicit active flags)
  PulseTimer timer_btn1_;
  PulseTimer timer_btn2_;
  PulseTimer timer_btn3_;
  PulseTimer timer_btn4_;

  void record_latency(size_t channel_idx, uint32_t current_time_us);
};

} // namespace pbl
