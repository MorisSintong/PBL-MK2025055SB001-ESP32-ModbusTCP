cmake_minimum_required(VERSION 3.22)
project(ESP32-ModbusTCP-Bridge C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile_commands.json for Zed IDE and clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_TESTS "Build host off-target simulation test suite" ON)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/App/Inc
)

set(FIRMWARE_SOURCES
    App/Src/bridge_state.cpp
    App/Src/logger.cpp
    App/Src/modbus_bridge.cpp
    App/Src/eth_driver.cpp
    App/Src/web_server.cpp
    App/Src/main.cpp
)

# Host off-target tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Stub executable for host verification and AST generation for clangd
add_executable(ESP32_Modbus_App
    ${FIRMWARE_SOURCES}
)
