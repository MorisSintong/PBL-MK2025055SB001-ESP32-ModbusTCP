"""
Test Modbus TCP connection to ESP32
Run this while ESP32 and all devices are connected
"""
from pymodbus.client import ModbusTcpClient
import time

ESP32_IP = "192.168.1.50"
ESP32_PORT = 503  # We changed to 503

print("="*50)
print("Modbus TCP Test - Connecting to ESP32")
print(f"Target: {ESP32_IP}:{ESP32_PORT}")
print("="*50)

# Test 1: Connect to ESP32
print("\n[Test 1] Connecting...")
client = ModbusTcpClient(ESP32_IP, port=ESP32_PORT, timeout=5)
result = client.connect()
print(f"Connection result: {result}")

if result:
    print("[OK] Connected to ESP32!")
    
    # Test 2: Read coil 112 (BTN1)
    print("\n[Test 2] Reading coil 112...")
    try:
        response = client.read_coils(112, 1)
        if not response.isError():
            print(f"[OK] Coil 112 = {response.bits[0]}")
        else:
            print(f"[ERROR] {response}")
    except Exception as e:
        print(f"[ERROR] {e}")
    
    # Test 3: Read coils 208-210 (Status)
    print("\n[Test 3] Reading coils 208-210 (status)...")
    try:
        response = client.read_coils(208, 3)
        if not response.isError():
            print(f"[OK] Running={response.bits[0]}, Stopped={response.bits[1]}, Emergency={response.bits[2]}")
        else:
            print(f"[ERROR] {response}")
    except Exception as e:
        print(f"[ERROR] {e}")
    
    # Test 4: Write coil 112 (simulate button press)
    print("\n[Test 4] Writing coil 112 = True...")
    try:
        response = client.write_coil(112, True)
        if not response.isError():
            print(f"[OK] Write successful")
        else:
            print(f"[ERROR] {response}")
    except Exception as e:
        print(f"[ERROR] {e}")
    
    time.sleep(0.5)
    
    # Reset coil
    client.write_coil(112, False)
    print("[OK] Reset coil 112 = False")
    
    client.close()
    print("\n[DONE] All tests completed!")
else:
    print("[FAILED] Could not connect to ESP32")
    print("Possible causes:")
    print("  - ESP32 not running")
    print("  - Wrong IP or port")
    print("  - Firewall blocking")
    print("  - Network issue")

print("\n" + "="*50)
