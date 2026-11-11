cmake_minimum_required(VERSION 3.10)
project(SpikeSpyStreamer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ZeroMQ library
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(ZMQ libzmq)
endif()

if(NOT ZMQ_FOUND)
    find_path(ZMQ_INCLUDE_DIRS zmq.h PATHS /usr/local/include /usr/include)
    find_library(ZMQ_LIBRARIES NAMES zmq libzmq PATHS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu)
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
    ${ZMQ_INCLUDE_DIRS}
)

if(ZMQ_LIBRARIES)
    target_link_libraries(SpikeSpyStreamer PRIVATE ${ZMQ_LIBRARIES})
else()
    target_link_libraries(SpikeSpyStreamer PRIVATE zmq)
endif()

set_target_properties(SpikeSpyStreamer PROPERTIES
    PREFIX ""
    OUTPUT_NAME "SpikeSpyStreamer"
    POSITION_INDEPENDENT_CODE ON
)

# Installation destination for Open Ephys GUI plugins directory
install(TARGETS SpikeSpyStreamer DESTINATION bin/plugins)
