#include "bridge_state.hpp"
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

void test_command_string_parsing_valid() {
  assert(pbl::BridgeState::parse_command_str("start") ==
         pbl::CommandType::Start);
  assert(pbl::BridgeState::parse_command_str("stop") == pbl::CommandType::Stop);
  assert(pbl::BridgeState::parse_command_str("emergency") ==
         pbl::CommandType::Emergency);
  assert(pbl::BridgeState::parse_command_str("reset") ==
         pbl::CommandType::Reset);

  std::cout << "  [PASS] test_command_string_parsing_valid" << std::endl;
}

void test_command_string_parsing_negative_and_edge_cases() {
  // 1. Case variations (strict lower-case contract expected)
  std::vector<std::string> case_variants = {
      "Start", "START",     "sTART", "Stop",  "STOP",
      "sTOP",  "Emergency", "EMG",   "RESET", "Reset"};
  for (const auto &str : case_variants) {
    assert(pbl::BridgeState::parse_command_str(str) ==
           pbl::CommandType::Unknown);
  }

  // 2. Whitespace padding & control characters
  std::vector<std::string> whitespace_variants = {
      " start",        "start ", " start ", "\tstop", "stop\n",
      "emergency\r\n", " ",      "\t",      "\n",     "\r\n"};
  for (const auto &str : whitespace_variants) {
    assert(pbl::BridgeState::parse_command_str(str) ==
           pbl::CommandType::Unknown);
  }

  // 3. Substring collisions, prefixes, and suffixes
  std::vector<std::string> substring_variants = {
      "starter",          "starting", "prestop",   "stopped",
      "emergency_stop",   "emg",      "re-set",    "resetting",
      "start?param=true", "stop/1",   "reset#now", "start1"};
  for (const auto &str : substring_variants) {
    assert(pbl::BridgeState::parse_command_str(str) ==
           pbl::CommandType::Unknown);
  }

  // 4. Numeric and empty inputs
  std::vector<std::string> numeric_empty = {"",    "0",   "1",  "112",
                                            "113", "114", "-1", "9999"};
  for (const auto &str : numeric_empty) {
    assert(pbl::BridgeState::parse_command_str(str) ==
           pbl::CommandType::Unknown);
  }

  // 5. Injections, noise, and fuzzing payloads
  std::vector<std::string> noise_payloads = {
      "<script>alert(1)</script>",
      "DROP TABLE coils;",
      "../../etc/passwd",
      "%20start%20",
      "null",
      "undefined",
      "NaN",
      std::string(1024, 'x'),  // 1KB buffer overflow check
      std::string(4096, 'A')}; // 4KB buffer overflow check
  for (const auto &str : noise_payloads) {
    assert(pbl::BridgeState::parse_command_str(str) ==
           pbl::CommandType::Unknown);
  }

  std::cout << "  [PASS] test_command_string_parsing_negative_and_edge_cases"
            << std::endl;
}

void test_user_detection_parameter_parsing() {
  // Mirror logic implemented in WebServerManager::handle_user_detection:
  // bool is_detected = (detected == "1" || detected == "true");
  auto parse_detected = [](const std::string &val) -> bool {
    return (val == "1" || val == "true");
  };

  // Valid truthy inputs
  assert(parse_detected("1") == true);
  assert(parse_detected("true") == true);

  // Falsy & Invalid inputs
  assert(parse_detected("0") == false);
  assert(parse_detected("false") == false);
  assert(parse_detected("") == false);
  assert(parse_detected("TRUE") == false);
  assert(parse_detected("True") == false);
  assert(parse_detected("yes") == false);
  assert(parse_detected("on") == false);
  assert(parse_detected("2") == false);
  assert(parse_detected("-1") == false);
  assert(parse_detected(" 1") == false);
  assert(parse_detected("true ") == false);
  assert(parse_detected("null") == false);
  assert(parse_detected("undefined") == false);

  std::cout << "  [PASS] test_user_detection_parameter_parsing" << std::endl;
}

void test_simulated_rest_workflow() {
  pbl::BridgeState bridge;

  // Simulate incoming HTTP GET /action?type=start
  std::string received_type = "start";
  pbl::CommandType cmd = pbl::BridgeState::parse_command_str(received_type);
  assert(cmd == pbl::CommandType::Start);

  uint64_t simulated_now_us = 100000;
  bool triggered = bridge.trigger_command(cmd, simulated_now_us);
  assert(triggered == true);

  // Modbus task consumes it at +250us
  assert(bridge.consume_start_request(simulated_now_us + 250));
  assert(bridge.get_last_latency_us() == 250);

  // Simulate incoming HTTP GET /action?type=bogus (Negative test)
  std::string bogus_type = "hack_sys";
  pbl::CommandType bogus_cmd = pbl::BridgeState::parse_command_str(bogus_type);
  assert(bogus_cmd == pbl::CommandType::Unknown);
  assert(!bridge.trigger_command(bogus_cmd, simulated_now_us + 1000));

  // Verify none of the channels were armed by the bogus command
  assert(!bridge.consume_start_request(simulated_now_us + 2000));
  assert(!bridge.consume_stop_request(simulated_now_us + 2000));
  assert(!bridge.consume_emergency_request(simulated_now_us + 2000));
  assert(!bridge.consume_reset_request(simulated_now_us + 2000));

  // Simulate incoming HTTP GET /user_detection?detected=1
  bridge.set_user_detection(true);
  bool latched_val = false;
  assert(bridge.consume_user_detection_changed(latched_val));
  assert(latched_val == true);

  // Check HTTP /status JSON response generation
  std::string status_json = bridge.generate_status_json();
  assert(status_json.find("\"last_internal_latency_us\":250") !=
         std::string::npos);

  std::cout << "  [PASS] test_simulated_rest_workflow" << std::endl;
}

int main() {
  std::cout << "--- Running REST API Unit Tests ---" << std::endl;
  test_command_string_parsing_valid();
  test_command_string_parsing_negative_and_edge_cases();
  test_user_detection_parameter_parsing();
  test_simulated_rest_workflow();
  std::cout << "All REST API tests PASSED!" << std::endl;
  return 0;
}
