#include "app_config.hpp"
#include <cassert>
#include <iostream>

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

  // Verify address bounds within preallocated table
  assert(COIL_BTN_START < MODBUS_COIL_COUNT);
  assert(COIL_BTN_STOP < MODBUS_COIL_COUNT);
  assert(COIL_BTN_EMERGENCY < MODBUS_COIL_COUNT);
  assert(COIL_BTN_RESET < MODBUS_COIL_COUNT);
  assert(COIL_USER_DETECTED < MODBUS_COIL_COUNT);
  assert(COIL_STS_RUNNING < MODBUS_COIL_COUNT);
  assert(COIL_STS_STOPPED < MODBUS_COIL_COUNT);
  assert(COIL_STS_EMG < MODBUS_COIL_COUNT);

  // Verify port 503
  assert(MODBUS_TCP_PORT == 503);

  // Verify pulse timing is 500ms
  assert(MOMENTARY_PULSE_DURATION_MS == 500);

  std::cout << "[PASS] test_modbus_coil_mappings" << std::endl;
}

void test_hardware_pin_assignments() {
  using namespace pbl::config;

  // ESP32-S3 W5500 SPI Pins
  assert(ETH_CS_PIN == 10);
  assert(ETH_MOSI_PIN == 11);
  assert(ETH_SCLK_PIN == 12);
  assert(ETH_MISO_PIN == 13);

  // SPI Frequency: 8 MHz
  assert(ETH_SPI_FREQ_HZ == 8000000);

  std::cout << "[PASS] test_hardware_pin_assignments" << std::endl;
}

int main() {
  std::cout << "--- Running Modbus Configuration Unit Tests ---" << std::endl;
  test_modbus_coil_mappings();
  test_hardware_pin_assignments();
  std::cout << "All Modbus configuration tests PASSED!" << std::endl;
  return 0;
}
