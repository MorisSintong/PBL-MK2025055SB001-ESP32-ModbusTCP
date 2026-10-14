#include "eth_driver.hpp"
#include "logger.hpp"

namespace pbl {

#if defined(ARDUINO) || defined(ESP_PLATFORM)

void EthernetDriver::init() {
  Logger::log("Initializing W5500 SPI Ethernet...");

  SPI.begin(config::ETH_SCLK_PIN, config::ETH_MISO_PIN, config::ETH_MOSI_PIN);
  SPI.setFrequency(config::ETH_SPI_FREQ_HZ);

  Ethernet.init(config::ETH_CS_PIN);
  Ethernet.setRetransmissionTimeout(config::ETH_TIMEOUT_MS);
  Ethernet.setRetransmissionCount(config::ETH_RETRIES);

  IPAddress ip(config::ETH_IP[0], config::ETH_IP[1], config::ETH_IP[2],
               config::ETH_IP[3]);
  IPAddress dns(config::ETH_DNS[0], config::ETH_DNS[1], config::ETH_DNS[2],
                config::ETH_DNS[3]);
  IPAddress gw(config::ETH_GATEWAY[0], config::ETH_GATEWAY[1],
               config::ETH_GATEWAY[2], config::ETH_GATEWAY[3]);
  IPAddress subnet(config::ETH_SUBNET[0], config::ETH_SUBNET[1],
                   config::ETH_SUBNET[2], config::ETH_SUBNET[3]);

  Ethernet.begin(const_cast<uint8_t *>(config::ETH_MAC), ip, dns, gw, subnet);

  Logger::log("Ethernet IP: " + Ethernet.localIP().toString());
  Logger::log("Ethernet MAC: 02:AB:CD:01:02:03");
}

void EthernetDriver::check_link_status(uint32_t now_ms) {
  if (now_ms - last_link_check_ms_ > config::ETH_LINK_CHECK_INTERVAL_MS) {
    last_link_check_ms_ = now_ms;
    EthernetLinkStatus status = Ethernet.linkStatus();
    if (status == LinkOFF) {
      Logger::log("[Core 1] !!! Ethernet Link OFF !!!");
    } else if (status == Unknown) {
      Logger::log("[Core 1] Ethernet Link: Unknown");
    }
  }
}

void EthernetDriver::check_sockets_and_maintain(uint32_t now_ms,
                                                uint32_t active_connections) {
  if (now_ms - last_socket_check_ms_ > config::ETH_SOCKET_CHECK_INTERVAL_MS) {
    last_socket_check_ms_ = now_ms;
    Ethernet.maintain(); // Periodic DHCP / socket cleanup

    std::string log_msg =
        "[Core 1] Connections: " + std::to_string(active_connections) +
        " | Link: " + (Ethernet.linkStatus() == LinkON ? "ON" : "OFF");
    Logger::log(log_msg);

    // Debug socket state register dump
    Serial.print("[W5500 Sockets] ");
    for (int i = 0; i < 8; i++) {
      uint8_t status = W5100.readSnSR(i);
      Serial.print(i);
      Serial.print(":");
      Serial.print(status, HEX);
      Serial.print(" ");
    }
    Serial.println();
  }
}

#else // Host / Simulation mode

void EthernetDriver::init() {
  Logger::log("EthernetDriver (Host Mock) initialized");
}

void EthernetDriver::check_link_status(uint32_t /*now_ms*/) {}

void EthernetDriver::check_sockets_and_maintain(
    uint32_t /*now_ms*/, uint32_t /*active_connections*/) {}

#endif

} // namespace pbl
