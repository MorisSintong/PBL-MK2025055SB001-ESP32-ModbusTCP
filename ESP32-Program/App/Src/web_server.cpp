#include "logger.hpp"
#include "web_server.hpp"
#include <cstdio>

namespace pbl {

WebServerManager::WebServerManager(BridgeState &state) : state_(state) {}

HttpResponse WebServerManager::handle_get_root() const {
  return {200, "text/plain", "ESP32 Opti-Server Online"};
}

HttpResponse WebServerManager::handle_get_status() const {
  return {200, "application/json", state_.generate_status_json()};
}

HttpResponse WebServerManager::handle_get_action(const std::string &type,
                                                 uint32_t now_us) {
  if (type.empty()) {
    Logger::log("[Web] Error: Missing type argument");
    return {400, "text/plain", "Missing type"};
  }

  Logger::log("[Web] Received command: " + type);
  CommandType cmd = BridgeState::parse_command_str(type);
  if (cmd == CommandType::Unknown) {
    return {400, "text/plain", "Unknown command"};
  }

  state_.trigger_command(cmd, now_us);
  return {200, "text/plain", "OK"};
}

HttpResponse
WebServerManager::handle_get_user_detection(const std::string &detected) {
  if (detected.empty()) {
    return {400, "text/plain", "Missing detected parameter"};
  }

  bool is_detected = (detected == "1" || detected == "true");
  state_.set_user_detection(is_detected);
  Logger::log(std::string("[Web] User detection: ") +
              (is_detected ? "DETECTED" : "NOT DETECTED"));
  return {200, "text/plain", "OK"};
}

bool WebServerManager::check_wifi_supervisor(uint32_t now_ms) {
  if ((now_ms - last_wifi_check_ms_) >= config::WIFI_RECONNECT_INTERVAL_MS) {
    last_wifi_check_ms_ = now_ms;
#if defined(ARDUINO) || defined(ESP_PLATFORM)
    if (WiFi.status() != WL_CONNECTED) {
      Logger::log("[Core 0] Wi-Fi lost. Reconnecting...");
      WiFi.disconnect();
      WiFi.reconnect();
    }
#endif
    return true;
  }
  return false;
}

#if defined(ARDUINO) || defined(ESP_PLATFORM)

void WebServerManager::init() {
  Logger::log("[Core 0] Wi-Fi Task Started");

  // Disable Wi-Fi modem sleep for ultra-low latency HTTP response
  WiFi.setSleep(false);

  IPAddress ip(config::WIFI_IP[0], config::WIFI_IP[1], config::WIFI_IP[2],
               config::WIFI_IP[3]);
  IPAddress gw(config::WIFI_GATEWAY[0], config::WIFI_GATEWAY[1],
               config::WIFI_GATEWAY[2], config::WIFI_GATEWAY[3]);
  IPAddress subnet(config::WIFI_SUBNET[0], config::WIFI_SUBNET[1],
                   config::WIFI_SUBNET[2], config::WIFI_SUBNET[3]);
  IPAddress dns(config::WIFI_DNS[0], config::WIFI_DNS[1], config::WIFI_DNS[2],
                config::WIFI_DNS[3]);

  if (!WiFi.config(ip, gw, subnet, dns)) {
    Logger::log("[Core 0] Error: Failed to configure Static IP!");
  }

  WiFi.begin(config::WIFI_SSID, config::WIFI_PASSWORD);

  // Bounded connection attempt: prevent infinite blocking if router is offline
  // at boot
  constexpr uint32_t MAX_BOOT_ATTEMPTS = 20; // 20 * 500ms = 10 seconds timeout
  uint32_t attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < MAX_BOOT_ATTEMPTS) {
    delay(500);
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    char ip_buf[128];
    std::snprintf(ip_buf, sizeof(ip_buf), "[Core 0] Wi-Fi Connected. IP: %s",
                  WiFi.localIP().toString().c_str());
    Logger::log(ip_buf);
  } else {
    Logger::log("[Core 0] Wi-Fi initial connection timed out. Will retry in "
                "background supervisor.");
  }

  if (MDNS.begin(config::MDNS_HOSTNAME)) {
    Logger::log(std::string("[Core 0] mDNS responder started: http://") +
                config::MDNS_HOSTNAME + ".local");
  }

  server_.on("/", [this]() { handle_root(); });
  server_.on("/status", [this]() { handle_status(); });
  server_.on("/action", [this]() { handle_action(); });
  server_.on("/user_detection", [this]() { handle_user_detection(); });

  server_.begin();
  Logger::log("[Core 0] HTTP Web Server listening on port " +
              std::to_string(config::WEB_SERVER_PORT));
}

void WebServerManager::handle_root() {
  HttpResponse resp = handle_get_root();
  server_.sendHeader("Access-Control-Allow-Origin", "*");
  server_.send(resp.status_code, resp.content_type, resp.body.c_str());
}

void WebServerManager::handle_status() {
  HttpResponse resp = handle_get_status();
  server_.sendHeader("Access-Control-Allow-Origin", "*");
  server_.send(resp.status_code, resp.content_type, resp.body.c_str());
}

void WebServerManager::handle_action() {
  std::string type_str =
      server_.hasArg("type") ? std::string(server_.arg("type").c_str()) : "";
  uint32_t now_us = static_cast<uint32_t>(micros());
  HttpResponse resp = handle_get_action(type_str, now_us);
  server_.sendHeader("Access-Control-Allow-Origin", "*");
  server_.send(resp.status_code, resp.content_type, resp.body.c_str());
}

void WebServerManager::handle_user_detection() {
  std::string detected_str = server_.hasArg("detected")
                                 ? std::string(server_.arg("detected").c_str())
                                 : "";
  HttpResponse resp = handle_get_user_detection(detected_str);
  server_.sendHeader("Access-Control-Allow-Origin", "*");
  server_.send(resp.status_code, resp.content_type, resp.body.c_str());
}

void WebServerManager::update(uint32_t now_ms) {
  server_.handleClient();
  check_wifi_supervisor(now_ms);
}

#else // Native ESP32S3_TARGET or Host Simulation mode

void WebServerManager::init() {
#if defined(ESP32S3_TARGET)
  char buf[128];
  std::snprintf(
      buf, sizeof(buf),
      "[Core 0] REST API initialized (Target IP: %u.%u.%u.%u:%u, SSID: %s)",
      config::WIFI_IP[0], config::WIFI_IP[1], config::WIFI_IP[2],
      config::WIFI_IP[3], static_cast<unsigned>(config::WEB_SERVER_PORT),
      config::WIFI_SSID);
  Logger::log(buf);
#else
  Logger::log("WebServerManager (Host Mock) initialized");
#endif
}

void WebServerManager::update(uint32_t now_ms) {
  check_wifi_supervisor(now_ms);
}

#endif

} // namespace pbl
