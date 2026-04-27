#include <WiFi.h>
#include <WebServer.h>

// --- KONFIGURASI WIFI ---
// Ganti nama dan password ini jika ingin diubah
const char* AP_SSID = "MEKAPBL";
const char* AP_PASS = "mekagui1";

WebServer server(80);

// Fungsi ini yang akan dijalankan ketika Laptop mengirim perintah
void handleAction() {
  if (server.hasArg("type")) {
    String type = server.arg("type");
    
    // --- DEBUGGING SERIAL MONITOR ---
    // Ini agar kamu bisa lihat di layar hitam bawah Arduino IDE
    Serial.println("-------------------------");
    Serial.print("PERINTAH MASUK: ");
    Serial.println(type);

    // --- LOGIKA PROGRAM ESP32 ---
    if(type == "start") {
       Serial.println("-> STATUS: MESIN MENYALA (RUNNING)");
       // Masukkan kodingan menyalakan Relay/LED disini nanti
    }
    else if(type == "stop") {
       Serial.println("-> STATUS: MESIN BERHENTI (STOP)");
    }
    else if(type == "emergency") {
       Serial.println("-> STATUS: !!! EMERGENCY !!! (SEMUA MATI)");
    }
    else if(type == "reset") {
       Serial.println("-> STATUS: SYSTEM RESET (READY)");
    }

    server.send(200, "text/plain", "OK - " + type);
  } else {
    server.send(400, "text/plain", "Error");
  }
}

// --- WAJIB ADA: SETUP ---
// Dijalankan sekali saat ESP32 baru nyala
void setup() {
  Serial.begin(115200);
  
  // Menyalakan WiFi Hotspot
  WiFi.softAP(AP_SSID, AP_PASS);
  
  Serial.println("\n--- ESP32 SIAP MENERIMA PERINTAH ---");
  Serial.print("IP Address: ");
  Serial.println(WiFi.softAPIP()); // Biasanya 192.168.4.1

  // Mengatur jalur komunikasi
  server.on("/action", handleAction);
  server.begin();
}

// --- WAJIB ADA: LOOP ---
// Dijalankan terus menerus selamanya
void loop() {
  server.handleClient(); // Cek apakah ada perintah masuk dari Laptop
}