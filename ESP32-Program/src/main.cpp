#include <Arduino.h>
#include <WiFi.h>
#include <ESPmDNS.h>
#include <WebServer.h>
#include <SPI.h>
#include <Ethernet_Generic.h>
#include <ModbusEthernet.h>
#include <atomic> // REQUIRED for thread safety



// =======================================================================
// USER CONFIGURATION
// =======================================================================

const char* ssid = "TestEsp32";
const char* password = "12345678";

// W5500 Pins
constexpr uint8_t ETH_CS_PIN = 10;
constexpr uint8_t ETH_MOSI_PIN = 11;
constexpr uint8_t ETH_SCLK_PIN = 12;
constexpr uint8_t ETH_MISO_PIN = 13;


// Ethernet Network Settings
// IMPORTANT: Use a unique MAC address to avoid ARP conflicts
byte macAddress[] = {0x02, 0xAB, 0xCD, 0x01, 0x02, 0x03};
IPAddress staticIp(192, 168, 1, 50);
IPAddress staticGateway(192, 168, 1, 1);
IPAddress staticSubnet(255, 255, 255, 0);
IPAddress staticDns(192, 168, 1, 1);

// Modbus Coils (Buttons)
const uint16_t COIL_BTN_1 = 112;
const uint16_t COIL_BTN_2 = 113;
const uint16_t COIL_BTN_3 = 114;
const uint16_t COIL_BTN_4 = 115;

// Modbus Coil (User Detection Status - Written by Face Recognition System)
const uint16_t COIL_USER_DETECTED = 116;

// Modbus Coils (Status Indicators - Read from PLC)
const uint16_t STS_RUNNING = 208;
const uint16_t STS_STOPPED = 209;
const uint16_t STS_EMG     = 210;

// Shared Variables (Atomic for thread safety)
std::atomic<bool> reqBtn1(false);
std::atomic<bool> reqBtn2(false);
std::atomic<bool> reqBtn3(false);
std::atomic<bool> reqReset(false);
std::atomic<unsigned long> actionStartTime(0);
std::atomic<unsigned long> lastInternalLatencyUs(0);

// User Detection Status (from Face Recognition System)
std::atomic<bool> userDetected(false);
std::atomic<bool> userDetectedChanged(false);

volatile unsigned long timerBtn1 = 0;
volatile unsigned long timerBtn2 = 0;
volatile unsigned long timerBtn3 = 0;
volatile unsigned long timerBtn4 = 0;
const unsigned long MOMENTARY_DELAY = 500; 

// Objects
WebServer server(80);
ModbusEthernet mb;
SemaphoreHandle_t serialMutex; // Mutex for Serial safety

// Modbus Activity Counters (for real tracking)
volatile uint32_t modbusConnections = 0;
volatile uint32_t modbusDisconnections = 0;

// Task Handles
TaskHandle_t ModbusTaskHandle;
TaskHandle_t WifiTaskHandle;

// Forward declaration for callbacks
void safePrintln(const String &msg);

// Modbus Callbacks
bool onModbusConnect(IPAddress ip) {
  modbusConnections++;
  Serial.print("[Modbus] >>> Client CONNECTED from: ");
  Serial.println(ip);
  return true;  // Accept connection
}

// Helper for safe printing
void safePrintln(const String &msg) {
  if (xSemaphoreTake(serialMutex, portMAX_DELAY)) {
    Serial.println(msg);
    xSemaphoreGive(serialMutex);
  }
}


// =======================================================================
// WEB SERVER FUNCTIONS (Run on Core 0) 
// =======================================================================

void handleRoot() {
  server.send(200, "text/plain", "ESP32 Opti-Server Online");
}

void handleStatus() {
  // Read Modbus Coils (Thread-safe enough for bool read)
  bool running = mb.Coil(STS_RUNNING);
  bool stopped = mb.Coil(STS_STOPPED);
  bool emg     = mb.Coil(STS_EMG);
  unsigned long lastLatency = lastInternalLatencyUs;

  String json = "{";
  json += "\"running\":" + String(running ? "true" : "false") + ",";
  json += "\"stopped\":" + String(stopped ? "true" : "false") + ",";
  json += "\"emergency\":" + String(emg ? "true" : "false") + ",";
  json += "\"last_internal_latency_us\":" + String(lastLatency);
  json += "}";

  server.send(200, "application/json", json);
}

void handleAction() {
  if (!server.hasArg("type")) {
    server.send(400, "text/plain", "Missing type");
    safePrintln("[Web] Error: Missing type argument");
    return;
  }
  String type = server.arg("type");
  safePrintln("[Web] Received command: " + type);
  
  // THREAD-SAFE FIX:
  // Instead of calling mb.Coil() directly (which is dangerous across cores),
  // we just set a flag. The Modbus Task (Core 1) will pick it up safely.
  
  actionStartTime = micros();

  if (type == "start") {
    reqBtn1 = true;
  } else if (type == "stop") {
    reqBtn2 = true;
  } else if (type == "emergency") {
    reqBtn3 = true;
  } else if (type == "reset") {
    reqReset = true;
  } else {
    server.send(400, "text/plain", "Unknown command");
    return;
  }
  server.send(200, "text/plain", "OK");
}

// Handler for User Detection Status from Face Recognition System
void handleUserDetection() {
  if (!server.hasArg("detected")) {
    server.send(400, "text/plain", "Missing detected parameter");
    return;
  }
  String detected = server.arg("detected");
  bool newState = (detected == "1" || detected == "true");
  
  // Only update if state changed
  if (userDetected != newState) {
    userDetected = newState;
    userDetectedChanged = true;
    safePrintln("[Web] User detection: " + String(newState ? "DETECTED" : "NOT DETECTED"));
  }
  
  server.send(200, "text/plain", "OK");
}

// =======================================================================
// FREERTOS TASKS
// =======================================================================

// Task 1: Modbus & Ethernet (High Priority, Core 1)
void modbusTask(void * parameter) {
  // Modbus is already initialized in setup() for faster response
  // This task just runs mb.task() in a loop

  safePrintln("[Core 1] Modbus task running");

  static unsigned long lastLinkCheck = 0;
  static unsigned long lastSocketCheck = 0;

  for (;;) {
    // 0. Check Ethernet Link and Socket Status
    if (millis() - lastLinkCheck > 2000) {
        lastLinkCheck = millis();
        EthernetLinkStatus linkStatus = Ethernet.linkStatus();
        if (linkStatus == LinkOFF) {
             safePrintln("[Core 1] !!! Ethernet Link OFF !!!");
        } else if (linkStatus == Unknown) {
             safePrintln("[Core 1] Ethernet Link: Unknown");
        }
    }
    
    // 0b. Monitor Modbus Connections and W5500 Socket Status
    if (millis() - lastSocketCheck > 3000) {
        lastSocketCheck = millis();
        Ethernet.maintain(); // Force socket cleanup
        
        // Report actual Modbus TCP connections
        safePrintln("[Core 1] Connections: " + String(modbusConnections) + " | Link: " + 
                   (Ethernet.linkStatus() == LinkON ? "ON" : "OFF"));
        
        // Print W5500 socket status for debugging
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
    // 1. PROCESS FLAGS (Thread-Safe Write)
    if (reqBtn1) {
      mb.Coil(COIL_BTN_1, true);
      timerBtn1 = millis();
      reqBtn1 = false;
      unsigned long latency = micros() - actionStartTime;
      lastInternalLatencyUs.store(latency);
      safePrintln("BTN 1 Triggered. Latency: " + String(latency) + " us");
    }
    if (reqBtn2) {
      mb.Coil(COIL_BTN_2, true);
      timerBtn2 = millis();
      reqBtn2 = false;
      unsigned long latency = micros() - actionStartTime;
      lastInternalLatencyUs.store(latency);
      safePrintln("BTN 2 Triggered. Latency: " + String(latency) + " us");
    }
    if (reqBtn3) {
      mb.Coil(COIL_BTN_3, true);
      timerBtn3 = millis();
      reqBtn3 = false;
      unsigned long latency = micros() - actionStartTime;
      lastInternalLatencyUs.store(latency);
      safePrintln("BTN 3 Triggered. Latency: " + String(latency) + " us");
    }
    if (reqReset) {
      mb.Coil(COIL_BTN_4, true);
      timerBtn4 = millis();
      reqReset = false;
      unsigned long latency = micros() - actionStartTime;
      lastInternalLatencyUs.store(latency);
      safePrintln("Reset Triggered. Latency: " + String(latency) + " us");
    }

    // 1b. PROCESS USER DETECTION (Continuous coil, not momentary)
    if (userDetectedChanged) {
      bool detected = userDetected.load();
      mb.Coil(COIL_USER_DETECTED, detected);
      userDetectedChanged = false;
      safePrintln("User Detection Coil: " + String(detected ? "HIGH" : "LOW"));
    }

    // 2. Run Modbus Task
    mb.task();

    // 3. Handle Timers
    unsigned long now = millis();
    if (timerBtn1 > 0 && now - timerBtn1 > MOMENTARY_DELAY) {
      mb.Coil(COIL_BTN_1, false);
      timerBtn1 = 0;
    }
    if (timerBtn2 > 0 && now - timerBtn2 > MOMENTARY_DELAY) {
      mb.Coil(COIL_BTN_2, false);
      timerBtn2 = 0;
    }
    if (timerBtn3 > 0 && now - timerBtn3 > MOMENTARY_DELAY) {
      mb.Coil(COIL_BTN_3, false);
      timerBtn3 = 0;
    }
    if (timerBtn4 > 0 && now - timerBtn4 > MOMENTARY_DELAY) {
      mb.Coil(COIL_BTN_4, false);
      timerBtn4 = 0;
    }

    // Small yield
    vTaskDelay(1 / portTICK_PERIOD_MS); 
  }
}

// Task 2: Wi-Fi & Web Server (Lower Priority, Core 0)
void wifiTask(void * parameter) {
  safePrintln("[Core 0] Wi-Fi Task Started");
  
  // OPTIMIZATION: Disable Power Save for low latency HTTP response
  WiFi.setSleep(false);
  
  // --- STATIC IP CONFIGURATION ---
  // Must be called BEFORE WiFi.begin()
  IPAddress wifiIp(10, 251, 64, 200);       
  IPAddress wifiGateway(10, 251, 64, 8);
  IPAddress wifiSubnet(255, 255, 255, 0);
  IPAddress wifiDns(10, 251, 64, 8);
  
  if (!WiFi.config(wifiIp, wifiGateway, wifiSubnet, wifiDns)) {
    safePrintln("[Core 0] Error: Failed to configure Static IP!");
  }
  // ------------------------------------

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    vTaskDelay(500 / portTICK_PERIOD_MS);
  }
  Serial.println("");

  safePrintln("[Core 0] Wi-Fi IP: " + WiFi.localIP().toString());

  if (MDNS.begin("esp32-server")) {
    safePrintln("[Core 0] mDNS responder started: http://esp32-server.local");
  }

  server.on("/", handleRoot);
  server.on("/status", handleStatus); // New Endpoint
  server.on("/action", handleAction);
  server.on("/user_detection", handleUserDetection); // User Detection from Face Recognition
  server.begin();

  static unsigned long lastWifiCheck = 0;

  for (;;) {
    server.handleClient();
    
    // Auto Reconnect Logic (Non-blocking)
    if (millis() - lastWifiCheck > 10000) {
      lastWifiCheck = millis();
      if (WiFi.status() != WL_CONNECTED) {
        safePrintln("[Core 0] Wi-Fi lost. Reconnecting...");
        WiFi.disconnect();
        WiFi.reconnect();
      }
    }

    // Yield to let the OS handle background Wi-Fi stack
    vTaskDelay(2 / portTICK_PERIOD_MS);
  }
}

// =======================================================================
// MAIN SETUP
// =======================================================================

void setup() {
  // OPTIMIZATION: Force 240MHz
   setCpuFrequencyMhz(240); // DISABLED: Causes W5500 Instability
  
  Serial.begin(921600);
  while(!Serial) delay(10);

  // Create Mutex for Serial
  serialMutex = xSemaphoreCreateMutex();

  safePrintln("System Booting...");

  // Initialize Ethernet HERE (Main Thread) to avoid SPI conflicts
  SPI.begin(ETH_SCLK_PIN, ETH_MISO_PIN, ETH_MOSI_PIN);
  SPI.setFrequency(8000000); // Reduced to 8MHz for maximum stability in multi-device networks
  Ethernet.init(ETH_CS_PIN);
  
  // Socket timeout configuration for W5500 (helps with multi-device networks)
  Ethernet.setRetransmissionTimeout(200);  // 200ms timeout (default is 200)
  Ethernet.setRetransmissionCount(4);      // 4 retries (default is 8)
  
  Ethernet.begin(macAddress, staticIp, staticDns, staticGateway, staticSubnet);
  safePrintln("Ethernet IP: " + Ethernet.localIP().toString());
  safePrintln("Ethernet MAC: 02:AB:CD:01:02:03");

  // CRITICAL: Initialize Modbus server HERE in setup() so it's ready IMMEDIATELY
  // This ensures the server is listening BEFORE any PLC connection attempts
  mb.onConnect(onModbusConnect);
  mb.server(503);
  mb.addCoil(0, false, 300); 
  mb.addHreg(0, 0, 300);
  safePrintln("Modbus TCP Server initialized on port 503");

  // Create Task for Modbus (Core 1)
  xTaskCreatePinnedToCore(
    modbusTask,       // Function
    "ModbusTask",     // Name
    8192,             // Stack size (Bytes)
    NULL,             // Param
    2,                // Priority (Higher = 2)
    &ModbusTaskHandle,// Handle
    1                 // Core ID (1 = App Core)
  );

  // Create Task for Wi-Fi (Core 0)
  xTaskCreatePinnedToCore(
    wifiTask,         // Function
    "WifiTask",       // Name
    8192,             // Stack size
    NULL,             // Param
    1,                // Priority (Lower = 1)
    &WifiTaskHandle,  // Handle
    0                 // Core ID (0 = Pro Core)
  );
}

void loop() {
  // Main loop is empty because tasks handle everything.
  // We can delete the setup task to free memory.
  vTaskDelete(NULL);
}