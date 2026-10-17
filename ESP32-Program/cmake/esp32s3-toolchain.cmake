set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR xtensa)

set(TOOLCHAIN_PREFIX "C:/Tools/xtensa-esp-elf/bin/xtensa-esp32s3-elf-")

set(CMAKE_C_COMPILER   "${TOOLCHAIN_PREFIX}gcc.exe")
set(CMAKE_CXX_COMPILER "${TOOLCHAIN_PREFIX}g++.exe")
set(CMAKE_ASM_COMPILER "${TOOLCHAIN_PREFIX}gcc.exe")
set(CMAKE_AR           "${TOOLCHAIN_PREFIX}ar.exe")
set(CMAKE_RANLIB       "${TOOLCHAIN_PREFIX}ranlib.exe")
set(CMAKE_OBJCOPY      "${TOOLCHAIN_PREFIX}objcopy.exe")
set(CMAKE_SIZE         "${TOOLCHAIN_PREFIX}size.exe")

set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(ARCH_FLAGS "-mlongcalls -ffunction-sections -fdata-sections -DESP32S3_TARGET=1")

set(CMAKE_C_FLAGS_INIT   "${ARCH_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${ARCH_FLAGS} -fno-exceptions -fno-rtti -fno-threadsafe-statics")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-nostartfiles -Wl,--gc-sections")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
