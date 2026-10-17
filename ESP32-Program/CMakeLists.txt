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

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "xtensa")
    # Native ESP32-S3 Cross-Compilation Target
    add_executable(ESP32_Modbus_App
        App/Src/main.cpp
    )
    set_target_properties(ESP32_Modbus_App PROPERTIES SUFFIX ".elf")
    target_link_libraries(ESP32_Modbus_App PRIVATE app_common)
    target_link_options(ESP32_Modbus_App PRIVATE
        -T${CMAKE_CURRENT_SOURCE_DIR}/ld/esp32s3_out.ld
        -Wl,-Map=${CMAKE_BINARY_DIR}/ESP32_Modbus_App.map
    )

    add_custom_command(TARGET ESP32_Modbus_App POST_BUILD
        COMMAND "${CMAKE_SIZE}" "$<TARGET_FILE:ESP32_Modbus_App>"
        COMMAND uvx esptool --chip esp32s3 elf2image --flash-mode dio --flash-freq 80m --flash-size 16MB -o "${CMAKE_BINARY_DIR}/firmware.bin" "$<TARGET_FILE:ESP32_Modbus_App>"
        COMMENT "Generating ESP32-S3 flashable image build-release/firmware.bin via esptool elf2image"
    )
else()
    # Host Simulation & Unit Testing Target
    find_package(Threads REQUIRED)
    target_link_libraries(app_common PUBLIC Threads::Threads)

    if(BUILD_TESTS)
        enable_testing()
        add_subdirectory(tests)
    endif()

    add_executable(ESP32_Modbus_App
        App/Src/main.cpp
    )
    target_link_libraries(ESP32_Modbus_App PRIVATE app_common)
endif()
