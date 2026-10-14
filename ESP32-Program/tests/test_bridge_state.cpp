#include "bridge_state.hpp"
#include "modbus_bridge.hpp"
#include <cassert>
#include <iostream>
#include <set>
#include <vector>

void test_initial_state() {
  pbl::BridgeState bridge;
  pbl::PlcStatus status = bridge.get_plc_status();

  assert(!status.running);
  assert(!status.stopped);
  assert(!status.emergency);
  assert(status.last_latency_us == 0);
  assert(!bridge.get_user_detection());

  std::cout << "  [PASS] test_initial_state" << std::endl;
}

void test_command_triggers_and_latency_all_channels() {
  pbl::BridgeState bridge;

  // 1. Test Channel 1: Start
  uint64_t t_trigger = 1000;
  assert(bridge.trigger_command(pbl::CommandType::Start, t_trigger));
  // Cross-channel isolation: only Start should be pending
  assert(!bridge.consume_stop_request(1100));
  assert(!bridge.consume_emergency_request(1100));
  assert(!bridge.consume_reset_request(1100));

  uint64_t t_consume = 1250;
  assert(bridge.consume_start_request(t_consume));
  assert(bridge.get_last_latency_us() == 250);
  // Atomic exchange: second consume must return false
  assert(!bridge.consume_start_request(t_consume));

  // 2. Test Channel 2: Stop
  assert(bridge.trigger_command(pbl::CommandType::Stop, 2000));
  assert(!bridge.consume_start_request(2100));
  assert(!bridge.consume_emergency_request(2100));
  assert(!bridge.consume_reset_request(2100));
  assert(bridge.consume_stop_request(2350));
  assert(bridge.get_last_latency_us() == 350);
  assert(!bridge.consume_stop_request(2350));

  // 3. Test Channel 3: Emergency
  assert(bridge.trigger_command(pbl::CommandType::Emergency, 5000));
  assert(!bridge.consume_start_request(5100));
  assert(!bridge.consume_stop_request(5100));
  assert(!bridge.consume_reset_request(5100));
  assert(bridge.consume_emergency_request(5420));
  assert(bridge.get_last_latency_us() == 420);
  assert(!bridge.consume_emergency_request(5420));

  // 4. Test Channel 4: Reset
  assert(bridge.trigger_command(pbl::CommandType::Reset, 8000));
  assert(!bridge.consume_start_request(8100));
  assert(!bridge.consume_stop_request(8100));
  assert(!bridge.consume_emergency_request(8100));
  assert(bridge.consume_reset_request(8150));
  assert(bridge.get_last_latency_us() == 150);
  assert(!bridge.consume_reset_request(8150));

  std::cout << "  [PASS] test_command_triggers_and_latency_all_channels"
            << std::endl;
}

void test_unknown_and_negative_commands() {
  pbl::BridgeState bridge;

  // Unknown command should return false and not arm any channel
  assert(!bridge.trigger_command(pbl::CommandType::Unknown, 1000));
  assert(!bridge.consume_start_request(1100));
  assert(!bridge.consume_stop_request(1100));
  assert(!bridge.consume_emergency_request(1100));
  assert(!bridge.consume_reset_request(1100));
  assert(bridge.get_last_latency_us() == 0);

  // Consuming when nothing was triggered
  assert(!bridge.consume_start_request(2000));
  assert(!bridge.consume_stop_request(2000));
  assert(!bridge.consume_emergency_request(2000));
  assert(!bridge.consume_reset_request(2000));

  std::cout << "  [PASS] test_unknown_and_negative_commands" << std::endl;
}

void test_rapid_command_toggling_and_overwrite() {
  pbl::BridgeState bridge;

  // Trigger Start at t = 1000 us
  assert(bridge.trigger_command(pbl::CommandType::Start, 1000));
  // Overwrite with second Start at t = 2000 us before consumption
  assert(bridge.trigger_command(pbl::CommandType::Start, 2000));

  // Consumption at t = 2400 us should calculate latency from the latest trigger
  assert(bridge.consume_start_request(2400));
  assert(bridge.get_last_latency_us() == 400);

  // Only one consume succeeds
  assert(!bridge.consume_start_request(2400));

  // Rapid inter-channel triggers: Start then immediate Stop
  assert(bridge.trigger_command(pbl::CommandType::Start, 5000));
  assert(bridge.trigger_command(pbl::CommandType::Stop, 5100));

  // Both should be independently pending
  assert(bridge.consume_start_request(5200));
  assert(bridge.consume_stop_request(5300));
  assert(bridge.get_last_latency_us() == 200); // 5300 - 5100

  std::cout << "  [PASS] test_rapid_command_toggling_and_overwrite"
            << std::endl;
}

void test_latency_edge_cases_and_underflow_protection() {
  pbl::BridgeState bridge;

  // Edge case 1: Zero timestamp on trigger (t = 0)
  // Implementation requires start > 0 to record latency
  assert(bridge.trigger_command(pbl::CommandType::Start, 0));
  assert(bridge.consume_start_request(100));
  assert(bridge.get_last_latency_us() == 0);

  // Edge case 2: Backward clock anomaly (current_time_us < start)
  // Should NOT cause unsigned integer underflow wrap-around
  assert(bridge.trigger_command(pbl::CommandType::Stop, 5000));
  assert(bridge.consume_stop_request(4000)); // Clock stepped backwards
  // Latency should remain protected (0 from previous or unchanged)
  assert(bridge.get_last_latency_us() == 0);

  // Edge case 3: Exact same timestamp (zero latency)
  assert(bridge.trigger_command(pbl::CommandType::Emergency, 10000));
  assert(bridge.consume_emergency_request(10000));
  assert(bridge.get_last_latency_us() == 0);

  // Edge case 4: Large latency (e.g. 15 seconds = 15,000,000 us)
  assert(bridge.trigger_command(pbl::CommandType::Reset, 1000000));
  assert(bridge.consume_reset_request(16000000));
  assert(bridge.get_last_latency_us() == 15000000);

  std::cout << "  [PASS] test_latency_edge_cases_and_underflow_protection"
            << std::endl;
}

void test_momentary_pulse_timers_all_channels() {
  pbl::BridgeState bridge;

  // Verify all 4 pulse timers: BTN1 (Start), BTN2 (Stop), BTN3 (Emergency),
  // BTN4 (Reset)

  // Inactive check: when not started, check_and_clear must return false
  assert(!bridge.check_and_clear_pulse_btn1(100, 500));
  assert(!bridge.check_and_clear_pulse_btn2(100, 500));
  assert(!bridge.check_and_clear_pulse_btn3(100, 500));
  assert(!bridge.check_and_clear_pulse_btn4(100, 500));

  // Start all 4 at t = 1000 ms
  bridge.start_pulse_btn1(1000);
  bridge.start_pulse_btn2(1000);
  bridge.start_pulse_btn3(1000);
  bridge.start_pulse_btn4(1000);

  // At t = 1499 ms (elapsed 499 ms < 500 ms): all still active
  assert(!bridge.check_and_clear_pulse_btn1(1499, 500));
  assert(!bridge.check_and_clear_pulse_btn2(1499, 500));
  assert(!bridge.check_and_clear_pulse_btn3(1499, 500));
  assert(!bridge.check_and_clear_pulse_btn4(1499, 500));

  // At exact boundary t = 1500 ms (elapsed 500 ms >= 500 ms): all expire &
  // clear
  assert(bridge.check_and_clear_pulse_btn1(1500, 500));
  assert(bridge.check_and_clear_pulse_btn2(1500, 500));
  assert(bridge.check_and_clear_pulse_btn3(1500, 500));
  assert(bridge.check_and_clear_pulse_btn4(1500, 500));

  // Subsequent check: must return false (timers cleared to 0)
  assert(!bridge.check_and_clear_pulse_btn1(1501, 500));
  assert(!bridge.check_and_clear_pulse_btn2(1501, 500));
  assert(!bridge.check_and_clear_pulse_btn3(1501, 500));
  assert(!bridge.check_and_clear_pulse_btn4(1501, 500));

  std::cout << "  [PASS] test_momentary_pulse_timers_all_channels" << std::endl;
}

void test_pulse_timer_edge_cases() {
  pbl::BridgeState bridge;

  // Edge case 1: Zero duration (duration_ms = 0)
  // A pulse started at t = 200 ms with duration 0 expires immediately at t =
  // 200 ms
  bridge.start_pulse_btn1(200);
  assert(bridge.check_and_clear_pulse_btn1(200, 0));
  assert(!bridge.check_and_clear_pulse_btn1(200, 0));

  // Edge case 2: Pulse re-trigger / extension
  // Start at t = 100 ms with 500 ms nominal expiry (t = 600)
  bridge.start_pulse_btn2(100);
  // Re-start at t = 400 ms (new nominal expiry t = 900)
  bridge.start_pulse_btn2(400);

  // At t = 600 ms: should NOT expire because timer was reset to 400 ms
  assert(!bridge.check_and_clear_pulse_btn2(600, 500));
  // At t = 900 ms: now expires
  assert(bridge.check_and_clear_pulse_btn2(900, 500));
  assert(!bridge.check_and_clear_pulse_btn2(901, 500));

  // Edge case 3: Negative/zero timer state check
  // Timer never initialized
  assert(!bridge.check_and_clear_pulse_btn3(0, 0));
  assert(!bridge.check_and_clear_pulse_btn3(500, 0));

  std::cout << "  [PASS] test_pulse_timer_edge_cases" << std::endl;
}

void test_user_detection_latch_and_rapid_toggling() {
  pbl::BridgeState bridge;
  bool current = false;

  // Initial state: no change
  assert(!bridge.consume_user_detection_changed(current));

  // Set user detected = true
  bridge.set_user_detection(true);
  assert(bridge.get_user_detection() == true);
  assert(bridge.consume_user_detection_changed(current));
  assert(current == true);

  // Consumed: changed flag reset
  assert(!bridge.consume_user_detection_changed(current));

  // Setting same value again should NOT trigger change flag
  bridge.set_user_detection(true);
  assert(!bridge.consume_user_detection_changed(current));

  // Rapid toggle: true -> false -> true before consumption
  // In this case, user_detected_ is true, changed flag was set
  bridge.set_user_detection(false);
  bridge.set_user_detection(true);
  assert(bridge.consume_user_detection_changed(current));
  assert(current == true);
  assert(!bridge.consume_user_detection_changed(current));

  // Toggle to false
  bridge.set_user_detection(false);
  assert(bridge.consume_user_detection_changed(current));
  assert(current == false);

  // Redundant false
  bridge.set_user_detection(false);
  assert(!bridge.consume_user_detection_changed(current));

  std::cout << "  [PASS] test_user_detection_latch_and_rapid_toggling"
            << std::endl;
}

void test_plc_status_and_json_serialization_edge_cases() {
  pbl::BridgeState bridge;

  // 1. Initial / Idle state
  std::string json0 = bridge.generate_status_json();
  assert(json0.find("\"running\":false") != std::string::npos);
  assert(json0.find("\"stopped\":false") != std::string::npos);
  assert(json0.find("\"emergency\":false") != std::string::npos);
  assert(json0.find("\"last_internal_latency_us\":0") != std::string::npos);

  // 2. All true state (Fault / Alarm condition)
  bridge.update_plc_status(true, true, true);
  bridge.trigger_command(pbl::CommandType::Start, 1000);
  bridge.consume_start_request(1543);

  std::string json1 = bridge.generate_status_json();
  assert(json1.find("\"running\":true") != std::string::npos);
  assert(json1.find("\"stopped\":true") != std::string::npos);
  assert(json1.find("\"emergency\":true") != std::string::npos);
  assert(json1.find("\"last_internal_latency_us\":543") != std::string::npos);

  // 3. High latency serialization
  bridge.trigger_command(pbl::CommandType::Reset, 1000000);
  bridge.consume_reset_request(19876543);
  std::string json2 = bridge.generate_status_json();
  assert(json2.find("\"last_internal_latency_us\":18876543") !=
         std::string::npos);

  std::cout << "  [PASS] test_plc_status_and_json_serialization_edge_cases"
            << std::endl;
}

void test_modbus_bridge_host_integration() {
  pbl::BridgeState state;
  pbl::ModbusBridge bridge;

  bridge.init();

  // Trigger Start
  state.trigger_command(pbl::CommandType::Start, 1000);
  // Bridge update consumes Start and initiates pulse timer
  bridge.update(state, 100, 1200);
  assert(state.get_last_latency_us() == 200);

  // At t = 300 ms (elapsed 200 ms < 500 ms): pulse still running
  bridge.update(state, 300, 1300);

  // At t = 650 ms (elapsed 550 ms >= 500 ms): pulse cleared
  bridge.update(state, 650, 1650);

  std::cout << "  [PASS] test_modbus_bridge_host_integration" << std::endl;
}

int main() {
  std::cout << "--- Running BridgeState Unit Tests ---" << std::endl;
  test_initial_state();
  test_command_triggers_and_latency_all_channels();
  test_unknown_and_negative_commands();
  test_rapid_command_toggling_and_overwrite();
  test_latency_edge_cases_and_underflow_protection();
  test_momentary_pulse_timers_all_channels();
  test_pulse_timer_edge_cases();
  test_user_detection_latch_and_rapid_toggling();
  test_plc_status_and_json_serialization_edge_cases();
  test_modbus_bridge_host_integration();
  std::cout << "All BridgeState tests PASSED!" << std::endl;
  return 0;
}
