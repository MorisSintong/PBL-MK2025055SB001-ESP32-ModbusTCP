#include "bridge_state.hpp"
#include <cassert>
#include <iostream>

void test_initial_state() {
  pbl::BridgeState bridge;
  pbl::PlcStatus status = bridge.get_plc_status();

  assert(!status.running);
  assert(!status.stopped);
  assert(!status.emergency);
  assert(status.last_latency_us == 0);
  assert(!bridge.get_user_detection());

  std::cout << "[PASS] test_initial_state" << std::endl;
}

void test_command_triggers_and_latency() {
  pbl::BridgeState bridge;

  // Trigger Start at t = 1000 us
  uint64_t t_trigger = 1000;
  assert(bridge.trigger_command(pbl::CommandType::Start, t_trigger));

  // Consume Start at t = 1250 us -> Latency should be 250 us
  uint64_t t_consume = 1250;
  assert(bridge.consume_start_request(t_consume));
  assert(bridge.get_last_latency_us() == 250);

  // Should only consume once (atomic exchange)
  assert(!bridge.consume_start_request(t_consume));

  // Trigger Emergency at t = 5000 us
  assert(bridge.trigger_command(pbl::CommandType::Emergency, 5000));
  assert(bridge.consume_emergency_request(5420));
  assert(bridge.get_last_latency_us() == 420);

  std::cout << "[PASS] test_command_triggers_and_latency" << std::endl;
}

void test_momentary_pulse_timers() {
  pbl::BridgeState bridge;

  // Start pulse at t = 100 ms
  bridge.start_pulse_btn1(100);

  // At t = 300 ms (elapsed 200ms < 500ms): should still be active
  assert(!bridge.check_and_clear_pulse_btn1(300, 500));

  // At t = 600 ms (elapsed 500ms >= 500ms): pulse expires and clears
  assert(bridge.check_and_clear_pulse_btn1(600, 500));

  // Subsequent check should return false (timer reset to 0)
  assert(!bridge.check_and_clear_pulse_btn1(650, 500));

  std::cout << "[PASS] test_momentary_pulse_timers" << std::endl;
}

void test_user_detection_latch() {
  pbl::BridgeState bridge;
  bool current = false;

  // Initial state: no change
  assert(!bridge.consume_user_detection_changed(current));

  // Set user detected = true
  bridge.set_user_detection(true);
  assert(bridge.get_user_detection() == true);
  assert(bridge.consume_user_detection_changed(current));
  assert(current == true);

  // After consuming, changed flag should be reset
  assert(!bridge.consume_user_detection_changed(current));

  // Setting same value again should NOT trigger change flag
  bridge.set_user_detection(true);
  assert(!bridge.consume_user_detection_changed(current));

  // Setting to false should trigger change flag
  bridge.set_user_detection(false);
  assert(bridge.consume_user_detection_changed(current));
  assert(current == false);

  std::cout << "[PASS] test_user_detection_latch" << std::endl;
}

void test_status_json_serialization() {
  pbl::BridgeState bridge;

  bridge.update_plc_status(true, false, false);
  bridge.trigger_command(pbl::CommandType::Start, 100);
  bridge.consume_start_request(242);

  std::string json = bridge.generate_status_json();

  assert(json.find("\"running\":true") != std::string::npos);
  assert(json.find("\"stopped\":false") != std::string::npos);
  assert(json.find("\"emergency\":false") != std::string::npos);
  assert(json.find("\"last_internal_latency_us\":142") != std::string::npos);

  std::cout << "[PASS] test_status_json_serialization" << std::endl;
}

int main() {
  std::cout << "--- Running BridgeState Unit Tests ---" << std::endl;
  test_initial_state();
  test_command_triggers_and_latency();
  test_momentary_pulse_timers();
  test_user_detection_latch();
  test_status_json_serialization();
  std::cout << "All BridgeState tests PASSED!" << std::endl;
  return 0;
}
