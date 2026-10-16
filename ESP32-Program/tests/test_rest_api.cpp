#include "bridge_state.hpp"
#include "web_server.hpp"
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define TEST_ASSERT(expr)                                                      \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__                    \
                << " Assertion failed: " #expr << std::endl;                   \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

void test_command_string_parsing() {
  TEST_ASSERT(pbl::BridgeState::parse_command_str("start") ==
              pbl::CommandType::Start);
  TEST_ASSERT(pbl::BridgeState::parse_command_str("stop") ==
              pbl::CommandType::Stop);
  TEST_ASSERT(pbl::BridgeState::parse_command_str("emergency") ==
              pbl::CommandType::Emergency);
  TEST_ASSERT(pbl::BridgeState::parse_command_str("reset") ==
              pbl::CommandType::Reset);

  std::vector<std::string> invalid_inputs = {"",
                                             "Start",
                                             "STOP",
                                             "Emergency",
                                             "RESET",
                                             " start",
                                             "start ",
                                             "\tstop",
                                             "emergency\r\n",
                                             "starter",
                                             "stopped",
                                             "emg",
                                             "0",
                                             "112",
                                             "null",
                                             "<script>",
                                             std::string(1024, 'x')};
  for (const auto &str : invalid_inputs) {
    TEST_ASSERT(pbl::BridgeState::parse_command_str(str) ==
                pbl::CommandType::Unknown);
  }

  std::cout << "  [PASS] test_command_string_parsing" << std::endl;
}

void test_web_server_http_endpoints() {
  pbl::BridgeState bridge;
  pbl::WebServerManager server(bridge);
  server.init();

  // 1. GET /
  pbl::HttpResponse root_resp = server.handle_get_root();
  TEST_ASSERT(root_resp.status_code == 200);
  TEST_ASSERT(std::string(root_resp.content_type) == "text/plain");
  TEST_ASSERT(root_resp.body == "ESP32 Opti-Server Online");

  // 2. GET /action validation & execution
  pbl::HttpResponse missing_type = server.handle_get_action("", 1000);
  TEST_ASSERT(missing_type.status_code == 400);
  TEST_ASSERT(missing_type.body == "Missing type");

  pbl::HttpResponse bad_type = server.handle_get_action("invalid_cmd", 1000);
  TEST_ASSERT(bad_type.status_code == 400);
  TEST_ASSERT(bad_type.body == "Unknown command");

  pbl::HttpResponse ok_start = server.handle_get_action("start", 5000);
  TEST_ASSERT(ok_start.status_code == 200);
  TEST_ASSERT(ok_start.body == "OK");
  TEST_ASSERT(bridge.consume_start_request(5320));
  TEST_ASSERT(bridge.get_last_latency_us() == 320);

  // 3. GET /user_detection validation & latching
  pbl::HttpResponse missing_det = server.handle_get_user_detection("");
  TEST_ASSERT(missing_det.status_code == 400);
  TEST_ASSERT(missing_det.body == "Missing detected parameter");

  pbl::HttpResponse det_true = server.handle_get_user_detection("1");
  TEST_ASSERT(det_true.status_code == 200);
  bool latched = false;
  TEST_ASSERT(bridge.consume_user_detection_changed(latched));
  TEST_ASSERT(latched == true);

  pbl::HttpResponse det_false = server.handle_get_user_detection("0");
  TEST_ASSERT(det_false.status_code == 200);
  TEST_ASSERT(bridge.consume_user_detection_changed(latched));
  TEST_ASSERT(latched == false);

  pbl::HttpResponse det_str_true = server.handle_get_user_detection("true");
  TEST_ASSERT(det_str_true.status_code == 200);
  TEST_ASSERT(bridge.consume_user_detection_changed(latched));
  TEST_ASSERT(latched == true);

  // Non-"1"/"true" values evaluate to false
  pbl::HttpResponse det_other = server.handle_get_user_detection("TRUE");
  TEST_ASSERT(det_other.status_code == 200);
  TEST_ASSERT(bridge.consume_user_detection_changed(latched));
  TEST_ASSERT(latched == false);

  // 4. GET /status JSON endpoint
  bridge.update_plc_status(true, false, false);
  pbl::HttpResponse status_resp = server.handle_get_status();
  TEST_ASSERT(status_resp.status_code == 200);
  TEST_ASSERT(std::string(status_resp.content_type) == "application/json");
  TEST_ASSERT(status_resp.body ==
              "{\"running\":true,\"stopped\":false,\"emergency\":false,"
              "\"last_internal_latency_us\":320}");

  std::cout << "  [PASS] test_web_server_http_endpoints" << std::endl;
}

void test_wifi_supervisor_timer_and_wraparound() {
  pbl::BridgeState bridge;
  pbl::WebServerManager server(bridge);

  TEST_ASSERT(server.get_last_wifi_check_ms() == 0);
  TEST_ASSERT(!server.check_wifi_supervisor(9999));
  TEST_ASSERT(server.check_wifi_supervisor(10000));
  TEST_ASSERT(server.get_last_wifi_check_ms() == 10000);

  // Wrap-around across 0xFFFFFFFF (interval = 10000 ms = 0x2710)
  server.update(0xFFFFFF00u);
  TEST_ASSERT(server.get_last_wifi_check_ms() == 0xFFFFFF00u);

  // 0xFFFFFF00 + 9999 = 0x0000260F -> should not trigger yet
  TEST_ASSERT(!server.check_wifi_supervisor(0x0000260Fu));

  // 0xFFFFFF00 + 10000 = 0x00002610 -> triggers across 32-bit boundary
  TEST_ASSERT(server.check_wifi_supervisor(0x00002610u));
  TEST_ASSERT(server.get_last_wifi_check_ms() == 0x00002610u);

  std::cout << "  [PASS] test_wifi_supervisor_timer_and_wraparound"
            << std::endl;
}

int main() {
  std::cout << "--- Running REST API & WebServer Unit Tests ---" << std::endl;
  test_command_string_parsing();
  test_web_server_http_endpoints();
  test_wifi_supervisor_timer_and_wraparound();
  std::cout << "All REST API & WebServer tests PASSED!" << std::endl;
  return 0;
}
