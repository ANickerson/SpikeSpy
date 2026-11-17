cmake_minimum_required(VERSION 3.10)
project(SpikeSpyStreamer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Locate ZeroMQ.
#
# Order: (1) CMake package config - vcpkg installs a find_package(zeromq)
# wrapper defining the `libzmq` imported target, and upstream libzmq installs
# ZeroMQConfig.cmake with the same target; (2) pkg-config, used as a hint
# provider only; (3) plain find_path/find_library over standard prefixes plus
# ZMQ_ROOT. The link always uses an imported target or a fully resolved
# library path: linking the bare name `zmq` fails on MSVC (it looks for
# zmq.lib while vcpkg and the official Windows builds ship libzmq.lib) and on
# macOS the linker would need a matching -L directory for -lzmq.

find_package(zeromq CONFIG QUIET)
if(NOT TARGET libzmq)
    find_package(ZeroMQ CONFIG QUIET)
endif()

if(TARGET libzmq)
    set(ZMQ_LINK libzmq)
elseif(TARGET libzmq-static)
    set(ZMQ_LINK libzmq-static)
else()
    set(ZMQ_PC_INCLUDE_DIRS "")
    set(ZMQ_PC_LIBRARY_DIRS "")
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(ZMQ_PC QUIET libzmq)
    endif()

    find_path(ZMQ_INCLUDE_DIR zmq.h
        HINTS ${ZMQ_PC_INCLUDE_DIRS} ${ZMQ_ROOT} ENV ZMQ_ROOT
        PATHS /usr/local /usr /opt/homebrew /opt/local
        PATH_SUFFIXES include)

    find_library(ZMQ_LIBRARY
        NAMES zmq libzmq
        HINTS ${ZMQ_PC_LIBRARY_DIRS} ${ZMQ_ROOT} ENV ZMQ_ROOT
        PATHS /usr/local /usr /opt/homebrew /opt/local
        PATH_SUFFIXES lib lib64)

    if(ZMQ_INCLUDE_DIR AND ZMQ_LIBRARY)
        set(ZMQ_LINK "${ZMQ_LIBRARY}")
    else()
        message(FATAL_ERROR
            "ZeroMQ was not found (looked for zmq.h and the zmq/libzmq import "
            "library). Install it (vcpkg install zeromq:x64-windows, "
            "brew install zeromq, or apt install libzmq3-dev) and rerun "
            "cmake, or point -DZMQ_ROOT=<prefix> at an existing installation.")
    endif()
endif()

# Source files
set(SOURCES
    Source/OpenEphysLib.cpp
    Source/SpikeSpyStreamer.cpp
)

set(HEADERS
    Source/SpikeSpyStreamer.h
)

# Open Ephys Plugin shared library target
add_library(SpikeSpyStreamer SHARED ${SOURCES} ${HEADERS})

target_include_directories(SpikeSpyStreamer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Source
    ${ZMQ_PC_INCLUDE_DIRS}
    ${ZMQ_INCLUDE_DIR}
)

target_link_libraries(SpikeSpyStreamer PRIVATE ${ZMQ_LINK})

set_target_properties(SpikeSpyStreamer PROPERTIES
    PREFIX ""
    OUTPUT_NAME "SpikeSpyStreamer"
    POSITION_INDEPENDENT_CODE ON
)

# Installation destination for Open Ephys GUI plugins directory
install(TARGETS SpikeSpyStreamer DESTINATION bin/plugins)
