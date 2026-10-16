cmake_minimum_required(VERSION 3.22)
project(ESP32ModbusBridgeTests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(NOT TARGET app_common)
    find_package(Threads REQUIRED)

    add_library(app_common STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../App/Src/bridge_state.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../App/Src/logger.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../App/Src/modbus_bridge.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../App/Src/eth_driver.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../App/Src/web_server.cpp
    )

    target_include_directories(app_common PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../App/Inc
    )

    target_link_libraries(app_common PUBLIC Threads::Threads)
endif()

enable_testing()

# Test 1: BridgeState Logic & Multi-Threaded Concurrency
add_executable(test_bridge_state test_bridge_state.cpp)
target_link_libraries(test_bridge_state PRIVATE app_common)
add_test(NAME BridgeStateTest COMMAND test_bridge_state)

# Test 2: Modbus Coils, Register Configuration & Ethernet Supervision
add_executable(test_modbus_coils test_modbus_coils.cpp)
target_link_libraries(test_modbus_coils PRIVATE app_common)
add_test(NAME ModbusCoilsTest COMMAND test_modbus_coils)

# Test 3: REST API Handlers & Wi-Fi Supervision
add_executable(test_rest_api test_rest_api.cpp)
target_link_libraries(test_rest_api PRIVATE app_common)
add_test(NAME RestApiTest COMMAND test_rest_api)
