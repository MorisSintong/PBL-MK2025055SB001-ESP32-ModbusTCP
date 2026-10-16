#include "app_config.hpp"
#include "eth_driver.hpp"
#include <cstdlib>
#include <iostream>
#include <set>

#define TEST_ASSERT(expr)                                                      \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__                    \
                << " Assertion failed: " #expr << std::endl;                   \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

void test_modbus_coil_mappings() {
  using namespace pbl::config;

  // Verify FX5U PLC Command Coils
  TEST_ASSERT(COIL_BTN_START == 112);
  TEST_ASSERT(COIL_BTN_STOP == 113);
  TEST_ASSERT(COIL_BTN_EMERGENCY == 114);
  TEST_ASSERT(COIL_BTN_RESET == 115);
  TEST_ASSERT(COIL_USER_DETECTED == 116);

  // Verify FX5U PLC Feedback Coils
  TEST_ASSERT(COIL_STS_RUNNING == 208);
  TEST_ASSERT(COIL_STS_STOPPED == 209);
  TEST_ASSERT(COIL_STS_EMG == 210);

  // Verify port 503 and pulse duration 500ms
  TEST_ASSERT(MODBUS_TCP_PORT == 503);
  TEST_ASSERT(MOMENTARY_PULSE_DURATION_MS == 500);

  std::cout << "  [PASS] test_modbus_coil_mappings" << std::endl;
}

void test_coil_address_uniqueness_and_disjoint_ranges() {
  using namespace pbl::config;

  std::set<uint16_t> command_coils = {COIL_BTN_START, COIL_BTN_STOP,
                                      COIL_BTN_EMERGENCY, COIL_BTN_RESET,
                                      COIL_USER_DETECTED};
  TEST_ASSERT(command_coils.size() == 5);

  std::set<uint16_t> feedback_coils = {COIL_STS_RUNNING, COIL_STS_STOPPED,
                                       COIL_STS_EMG};
  TEST_ASSERT(feedback_coils.size() == 3);

  for (uint16_t cmd_coil : command_coils) {
    TEST_ASSERT(feedback_coils.find(cmd_coil) == feedback_coils.end());
  }

  // Contiguous blocks for single-transaction Modbus read/write optimization
  TEST_ASSERT(COIL_BTN_STOP == COIL_BTN_START + 1);
  TEST_ASSERT(COIL_BTN_EMERGENCY == COIL_BTN_STOP + 1);
  TEST_ASSERT(COIL_BTN_RESET == COIL_BTN_EMERGENCY + 1);
  TEST_ASSERT(COIL_USER_DETECTED == COIL_BTN_RESET + 1);

  TEST_ASSERT(COIL_STS_STOPPED == COIL_STS_RUNNING + 1);
  TEST_ASSERT(COIL_STS_EMG == COIL_STS_STOPPED + 1);

  std::cout << "  [PASS] test_coil_address_uniqueness_and_disjoint_ranges"
            << std::endl;
}

void test_table_bounds_and_hardware_configuration() {
  using namespace pbl::config;

  TEST_ASSERT(COIL_USER_DETECTED < MODBUS_COIL_COUNT);
  TEST_ASSERT(COIL_STS_EMG < MODBUS_COIL_COUNT);
  TEST_ASSERT(MODBUS_COIL_COUNT >= 260);
  TEST_ASSERT(MODBUS_HREG_COUNT >= 100);

  std::set<uint8_t> spi_pins = {ETH_CS_PIN, ETH_MOSI_PIN, ETH_SCLK_PIN,
                                ETH_MISO_PIN};
  TEST_ASSERT(spi_pins.size() == 4);
  TEST_ASSERT(ETH_SPI_FREQ_HZ == 8000000);
  TEST_ASSERT(ETH_TIMEOUT_MS > 0 && ETH_TIMEOUT_MS <= 1000);
  TEST_ASSERT(ETH_RETRIES >= 2 && ETH_RETRIES <= 10);

  TEST_ASSERT(ETH_IP[0] == 192 && ETH_IP[1] == 168 && ETH_IP[2] == 1 &&
              ETH_IP[3] == 50);
  TEST_ASSERT(MODBUS_TASK_CORE == 1 && WIFI_TASK_CORE == 0);
  TEST_ASSERT(MODBUS_TASK_PRIORITY > WIFI_TASK_PRIORITY);
  TEST_ASSERT(SERIAL_BAUD_RATE == 921600);

  std::cout << "  [PASS] test_table_bounds_and_hardware_configuration"
            << std::endl;
}

void test_eth_driver_supervision_and_millis_wraparound() {
  pbl::EthernetDriver eth;
  eth.init();

  TEST_ASSERT(eth.get_last_link_check_ms() == 0);
  TEST_ASSERT(eth.get_last_socket_check_ms() == 0);

  // Before 2000ms link interval and 3000ms socket interval
  TEST_ASSERT(!eth.check_link_status(1999));
  TEST_ASSERT(!eth.check_sockets_and_maintain(2999, 1));

  // At 2000ms: link check triggers, socket check still waiting
  TEST_ASSERT(eth.check_link_status(2000));
  TEST_ASSERT(eth.get_last_link_check_ms() == 2000);
  TEST_ASSERT(!eth.check_link_status(3999));

  // At 3000ms: socket check triggers
  TEST_ASSERT(eth.check_sockets_and_maintain(3000, 2));
  TEST_ASSERT(eth.get_last_socket_check_ms() == 3000);
  TEST_ASSERT(!eth.check_sockets_and_maintain(5999, 2));

  // Test 32-bit millis() wrap-around via update(now_ms)
  eth.update(0xFFFFFF00u, 1);
  TEST_ASSERT(eth.get_last_link_check_ms() == 0xFFFFFF00u);
  TEST_ASSERT(eth.get_last_socket_check_ms() == 0xFFFFFF00u);

  // 0xFFFFFF00 + 2000 (0x7D0) = 0x000006D0 -> link check fires, socket does not
  eth.update(0x000006D0u, 1);
  TEST_ASSERT(eth.get_last_link_check_ms() == 0x000006D0u);
  TEST_ASSERT(eth.get_last_socket_check_ms() == 0xFFFFFF00u);

  // 0xFFFFFF00 + 3000 (0xBB8) = 0x00000AB8 -> socket check fires
  eth.update(0x00000AB8u, 1);
  TEST_ASSERT(eth.get_last_socket_check_ms() == 0x00000AB8u);

  std::cout << "  [PASS] test_eth_driver_supervision_and_millis_wraparound"
            << std::endl;
}

int main() {
  std::cout << "--- Running Modbus & Ethernet Configuration Unit Tests ---"
            << std::endl;
  test_modbus_coil_mappings();
  test_coil_address_uniqueness_and_disjoint_ranges();
  test_table_bounds_and_hardware_configuration();
  test_eth_driver_supervision_and_millis_wraparound();
  std::cout << "All Modbus & Ethernet configuration tests PASSED!" << std::endl;
  return 0;
}
