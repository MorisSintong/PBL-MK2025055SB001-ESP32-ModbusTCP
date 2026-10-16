#pragma once

#include "app_config.hpp"
#include <cstdint>

#if defined(ARDUINO) || defined(ESP_PLATFORM)
#include <Ethernet_Generic.h>
#include <SPI.h>
#endif

namespace pbl {

class EthernetDriver {
public:
  EthernetDriver() = default;

  void init();
  void update(uint32_t now_ms, uint32_t active_connections = 0);
  bool check_link_status(uint32_t now_ms);
  bool check_sockets_and_maintain(uint32_t now_ms,
                                  uint32_t active_connections = 0);

  uint32_t get_last_link_check_ms() const { return last_link_check_ms_; }
  uint32_t get_last_socket_check_ms() const { return last_socket_check_ms_; }

private:
  uint32_t last_link_check_ms_{0};
  uint32_t last_socket_check_ms_{0};
};

} // namespace pbl
