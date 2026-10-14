#include "app_config.hpp"
#include <cassert>
#include <iostream>
#include <set>

void test_modbus_coil_mappings() {
  using namespace pbl::config;

  // Verify FX5U PLC Command Coils
  assert(COIL_BTN_START == 112);
  assert(COIL_BTN_STOP == 113);
  assert(COIL_BTN_EMERGENCY == 114);
  assert(COIL_BTN_RESET == 115);
  assert(COIL_USER_DETECTED == 116);

  // Verify FX5U PLC Feedback Coils
  assert(COIL_STS_RUNNING == 208);
  assert(COIL_STS_STOPPED == 209);
  assert(COIL_STS_EMG == 210);

  // Verify port 503
  assert(MODBUS_TCP_PORT == 503);

  // Verify pulse timing is 500ms
  assert(MOMENTARY_PULSE_DURATION_MS == 500);

  std::cout << "  [PASS] test_modbus_coil_mappings" << std::endl;
}

void test_coil_address_uniqueness_and_disjoint_ranges() {
  using namespace pbl::config;

  std::set<uint16_t> command_coils = {COIL_BTN_START, COIL_BTN_STOP,
                                      COIL_BTN_EMERGENCY, COIL_BTN_RESET,
                                      COIL_USER_DETECTED};

  // Ensure all 5 command coils have distinct addresses
  assert(command_coils.size() == 5);

  std::set<uint16_t> feedback_coils = {COIL_STS_RUNNING, COIL_STS_STOPPED,
                                       COIL_STS_EMG};

  // Ensure all 3 feedback coils have distinct addresses
  assert(feedback_coils.size() == 3);

  // Ensure zero collision / complete disjointness between commands and feedback
  for (uint16_t cmd_coil : command_coils) {
    assert(feedback_coils.find(cmd_coil) == feedback_coils.end());
  }

  // Ensure contiguous block for command coils (112..116)
  assert(COIL_BTN_STOP == COIL_BTN_START + 1);
  assert(COIL_BTN_EMERGENCY == COIL_BTN_STOP + 1);
  assert(COIL_BTN_RESET == COIL_BTN_EMERGENCY + 1);
  assert(COIL_USER_DETECTED == COIL_BTN_RESET + 1);

  // Ensure contiguous block for status feedback coils (208..210)
  assert(COIL_STS_STOPPED == COIL_STS_RUNNING + 1);
  assert(COIL_STS_EMG == COIL_STS_STOPPED + 1);

  std::cout << "  [PASS] test_coil_address_uniqueness_and_disjoint_ranges"
            << std::endl;
}

void test_table_bounds_and_capacity_margins() {
  using namespace pbl::config;

  // Verify all coils fit well within preallocated table
  assert(COIL_BTN_START < MODBUS_COIL_COUNT);
  assert(COIL_BTN_STOP < MODBUS_COIL_COUNT);
  assert(COIL_BTN_EMERGENCY < MODBUS_COIL_COUNT);
  assert(COIL_BTN_RESET < MODBUS_COIL_COUNT);
  assert(COIL_USER_DETECTED < MODBUS_COIL_COUNT);
  assert(COIL_STS_RUNNING < MODBUS_COIL_COUNT);
  assert(COIL_STS_STOPPED < MODBUS_COIL_COUNT);
  assert(COIL_STS_EMG < MODBUS_COIL_COUNT);

  // Capacity headroom check: at least 50 spare coils reserved for expansion
  assert(MODBUS_COIL_COUNT >= 210 + 50);

  // Holding register allocation
  assert(MODBUS_HREG_COUNT >= 100);

  std::cout << "  [PASS] test_table_bounds_and_capacity_margins" << std::endl;
}

void test_hardware_pin_and_network_configuration() {
  using namespace pbl::config;

  // ESP32-S3 W5500 SPI Pins distinctness
  std::set<uint8_t> spi_pins = {ETH_CS_PIN, ETH_MOSI_PIN, ETH_SCLK_PIN,
                                ETH_MISO_PIN};
  assert(spi_pins.size() == 4);

  // SPI Frequency: 8 MHz
  assert(ETH_SPI_FREQ_HZ == 8000000);

  // Network retry and timeout bounds
  assert(ETH_TIMEOUT_MS > 0 && ETH_TIMEOUT_MS <= 1000);
  assert(ETH_RETRIES >= 2 && ETH_RETRIES <= 10);

  // Static IP configuration validation
  assert(ETH_IP[0] == 192 && ETH_IP[1] == 168 && ETH_IP[2] == 1 &&
         ETH_IP[3] == 50);
  assert(ETH_SUBNET[0] == 255 && ETH_SUBNET[1] == 255 && ETH_SUBNET[2] == 255 &&
         ETH_SUBNET[3] == 0);

  // FreeRTOS Task Architecture sanity
  assert(TASK_STACK_SIZE_BYTES >= 4096);
  // Dual-core affinity isolation: Modbus on Core 1, Web on Core 0
  assert(MODBUS_TASK_CORE != WIFI_TASK_CORE);
  assert(MODBUS_TASK_CORE == 1);
  assert(WIFI_TASK_CORE == 0);
  // Modbus task given higher priority for determinism
  assert(MODBUS_TASK_PRIORITY > WIFI_TASK_PRIORITY);

  // Serial Baud Rate
  assert(SERIAL_BAUD_RATE == 921600);

  std::cout << "  [PASS] test_hardware_pin_and_network_configuration"
            << std::endl;
}

int main() {
  std::cout << "--- Running Modbus Configuration Unit Tests ---" << std::endl;
  test_modbus_coil_mappings();
  test_coil_address_uniqueness_and_disjoint_ranges();
  test_table_bounds_and_capacity_margins();
  test_hardware_pin_and_network_configuration();
  std::cout << "All Modbus configuration tests PASSED!" << std::endl;
  return 0;
}
