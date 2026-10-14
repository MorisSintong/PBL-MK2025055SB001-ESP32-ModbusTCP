#include "bridge_state.hpp"
#include <cassert>
#include <iostream>

void test_command_string_parsing() {
  assert(pbl::BridgeState::parse_command_str("start") ==
         pbl::CommandType::Start);
  assert(pbl::BridgeState::parse_command_str("stop") == pbl::CommandType::Stop);
  assert(pbl::BridgeState::parse_command_str("emergency") ==
         pbl::CommandType::Emergency);
  assert(pbl::BridgeState::parse_command_str("reset") ==
         pbl::CommandType::Reset);

  // Invalid commands
  assert(pbl::BridgeState::parse_command_str("START") ==
         pbl::CommandType::Unknown);
  assert(pbl::BridgeState::parse_command_str("invalid") ==
         pbl::CommandType::Unknown);
  assert(pbl::BridgeState::parse_command_str("") == pbl::CommandType::Unknown);

  std::cout << "[PASS] test_command_string_parsing" << std::endl;
}

void test_user_detection_parsing() {
  auto parse_detected = [](const std::string &val) -> bool {
    return (val == "1" || val == "true");
  };

  assert(parse_detected("1") == true);
  assert(parse_detected("true") == true);
  assert(parse_detected("0") == false);
  assert(parse_detected("false") == false);
  assert(parse_detected("random") == false);

  std::cout << "[PASS] test_user_detection_parsing" << std::endl;
}

int main() {
  std::cout << "--- Running REST API Unit Tests ---" << std::endl;
  test_command_string_parsing();
  test_user_detection_parsing();
  std::cout << "All REST API tests PASSED!" << std::endl;
  return 0;
}
