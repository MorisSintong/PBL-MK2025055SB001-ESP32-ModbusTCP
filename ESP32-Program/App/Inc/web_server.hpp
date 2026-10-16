#pragma once

#include "app_config.hpp"
#include "bridge_state.hpp"
#include <cstdint>
#include <string>

#if defined(ARDUINO) || defined(ESP_PLATFORM)
#include <ESPmDNS.h>
#include <WebServer.h>
#include <WiFi.h>
#endif

namespace pbl {

struct HttpResponse {
  int status_code{200};
  const char *content_type{"text/plain"};
  std::string body;
};

class WebServerManager {
public:
  explicit WebServerManager(BridgeState &state);

  void init();
  void update(uint32_t now_ms);
  bool check_wifi_supervisor(uint32_t now_ms);

  // Platform-independent request handlers (callable on both host and target)
  HttpResponse handle_get_root() const;
  HttpResponse handle_get_status() const;
  HttpResponse handle_get_action(const std::string &type, uint32_t now_us);
  HttpResponse handle_get_user_detection(const std::string &detected);

  uint32_t get_last_wifi_check_ms() const { return last_wifi_check_ms_; }

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
