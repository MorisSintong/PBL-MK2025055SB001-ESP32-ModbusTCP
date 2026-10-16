cmake_minimum_required(VERSION 3.22)
project(ESP32-ModbusTCP-Bridge C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile_commands.json for Zed IDE and clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

option(BUILD_TESTS "Build host off-target simulation test suite" ON)

find_package(Threads REQUIRED)

add_library(app_common STATIC
    App/Src/bridge_state.cpp
    App/Src/logger.cpp
    App/Src/modbus_bridge.cpp
    App/Src/eth_driver.cpp
    App/Src/web_server.cpp
)

target_include_directories(app_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/App/Inc
)

target_link_libraries(app_common PUBLIC Threads::Threads)

# Host off-target tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Host verification binary and AST target for clangd
add_executable(ESP32_Modbus_App
    App/Src/main.cpp
)
target_link_libraries(ESP32_Modbus_App PRIVATE app_common)
