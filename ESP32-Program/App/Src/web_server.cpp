#include "logger.hpp"
#include "web_server.hpp"

namespace pbl {

WebServerManager::WebServerManager(BridgeState &state) : state_(state) {}

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
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
  }
  Serial.println();

  Logger::log("[Core 0] Wi-Fi IP: " + WiFi.localIP().toString());

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
  server_.send(200, "text/plain", "ESP32 Opti-Server Online");
}

void WebServerManager::handle_status() {
  std::string json = state_.generate_status_json();
  server_.send(200, "application/json", json.c_str());
}

void WebServerManager::handle_action() {
  if (!server_.hasArg("type")) {
    server_.send(400, "text/plain", "Missing type");
    Logger::log("[Web] Error: Missing type argument");
    return;
  }

  String type = server_.arg("type");
  std::string type_str = type.c_str();
  Logger::log("[Web] Received command: " + type_str);

  CommandType cmd = BridgeState::parse_command_str(type_str);
  if (cmd == CommandType::Unknown) {
    server_.send(400, "text/plain", "Unknown command");
    return;
  }

  uint64_t now_us = micros();
  state_.trigger_command(cmd, now_us);
  server_.send(200, "text/plain", "OK");
}

void WebServerManager::handle_user_detection() {
  if (!server_.hasArg("detected")) {
    server_.send(400, "text/plain", "Missing detected parameter");
    return;
  }

  String detected = server_.arg("detected");
  bool is_detected = (detected == "1" || detected == "true");

  state_.set_user_detection(is_detected);
  Logger::log(std::string("[Web] User detection: ") +
              (is_detected ? "DETECTED" : "NOT DETECTED"));

  server_.send(200, "text/plain", "OK");
}

void WebServerManager::update(uint32_t now_ms) {
  server_.handleClient();

  // Auto-reconnect supervision logic
  if (now_ms - last_wifi_check_ms_ > config::WIFI_RECONNECT_INTERVAL_MS) {
    last_wifi_check_ms_ = now_ms;
    if (WiFi.status() != WL_CONNECTED) {
      Logger::log("[Core 0] Wi-Fi lost. Reconnecting...");
      WiFi.disconnect();
      WiFi.reconnect();
    }
  }
}

#else // Host / Simulation mode

void WebServerManager::init() {
  Logger::log("WebServerManager (Host Mock) initialized");
}

void WebServerManager::update(uint32_t /*now_ms*/) {}

#endif

} // namespace pbl
