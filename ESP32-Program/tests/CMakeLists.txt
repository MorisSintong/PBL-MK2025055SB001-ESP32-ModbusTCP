cmake_minimum_required(VERSION 3.22)
project(ESP32ModbusBridgeTests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../App/Inc
)

set(APP_COMMON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../App/Src/bridge_state.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../App/Src/logger.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../App/Src/modbus_bridge.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../App/Src/eth_driver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../App/Src/web_server.cpp
)

enable_testing()

# Test 1: BridgeState Logic
add_executable(test_bridge_state
    test_bridge_state.cpp
    ${APP_COMMON_SOURCES}
)
add_test(NAME BridgeStateTest COMMAND test_bridge_state)

# Test 2: Modbus Coils & Register Configuration
add_executable(test_modbus_coils
    test_modbus_coils.cpp
    ${APP_COMMON_SOURCES}
)
add_test(NAME ModbusCoilsTest COMMAND test_modbus_coils)

# Test 3: REST API Parsing
add_executable(test_rest_api
    test_rest_api.cpp
    ${APP_COMMON_SOURCES}
)
add_test(NAME RestApiTest COMMAND test_rest_api)
