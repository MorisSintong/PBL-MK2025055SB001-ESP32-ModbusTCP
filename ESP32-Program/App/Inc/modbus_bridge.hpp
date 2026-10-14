#pragma once

#include "app_config.hpp"
#include "bridge_state.hpp"

#if defined(ARDUINO) || defined(ESP_PLATFORM)
#include <Ethernet_Generic.h>
#include <ModbusEthernet.h>
#endif

namespace pbl {

class ModbusBridge {
public:
  ModbusBridge() = default;

  void init();
  void update(BridgeState &state, uint32_t now_ms, uint64_t now_us);

  uint32_t get_connection_count() const { return connection_count_; }

#if defined(ARDUINO) || defined(ESP_PLATFORM)
  static bool on_client_connect(IPAddress ip);
#endif

private:
#if defined(ARDUINO) || defined(ESP_PLATFORM)
  ModbusEthernet mb_;
#endif
  static uint32_t connection_count_;
};

} // namespace pbl
