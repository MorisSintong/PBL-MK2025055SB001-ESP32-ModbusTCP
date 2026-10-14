#pragma once

#include "app_config.hpp"
#include "bridge_state.hpp"

#if defined(ARDUINO) || defined(ESP_PLATFORM)
#include <ESPmDNS.h>
#include <WebServer.h>
#include <WiFi.h>
#endif

namespace pbl {

class WebServerManager {
public:
  explicit WebServerManager(BridgeState &state);

  void init();
  void update(uint32_t now_ms);

#if defined(ARDUINO) || defined(ESP_PLATFORM)
  void handle_root();
  void handle_status();
  void handle_action();
  void handle_user_detection();
#endif

private:
  BridgeState &state_;
  uint32_t last_wifi_check_ms_{0};

#if defined(ARDUINO) || defined(ESP_PLATFORM)
  WebServer server_{config::WEB_SERVER_PORT};
#endif
};

} // namespace pbl
