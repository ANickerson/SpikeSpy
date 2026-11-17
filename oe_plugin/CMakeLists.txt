cmake_minimum_required(VERSION 3.15)
project(SpikeSpyStreamer VERSION 1.0.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# plugin-GUI sources (headers only)
#
#   git clone https://github.com/open-ephys/plugin-GUI.git --branch v1.1.0 \
#       ../plugin-GUI
#
# Override with -DGUI_BASE_DIR=<path> or the GUI_BASE_DIR environment variable.
# ---------------------------------------------------------------------------
if(NOT DEFINED GUI_BASE_DIR)
    if(DEFINED ENV{GUI_BASE_DIR})
        set(GUI_BASE_DIR $ENV{GUI_BASE_DIR})
    else()
        set(GUI_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../plugin-GUI)
    endif()
endif()

if(NOT EXISTS ${GUI_BASE_DIR}/Plugins/Headers/PluginInfo.h)
    message(FATAL_ERROR
        "Open Ephys plugin-GUI sources not found at \"${GUI_BASE_DIR}\".\n"
        "Clone them (headers are required to build the plugin):\n"
        "  git clone https://github.com/open-ephys/plugin-GUI.git --branch v1.1.0 ${GUI_BASE_DIR}\n"
        "or pass -DGUI_BASE_DIR=<path> / set the GUI_BASE_DIR environment variable.")
endif()

# Generate JuceLibraryCode/JuceHeader.h exactly like the GUI's own
# CMakeLists.txt does (JuceHeader.h.in uses @GUI_VERSION@/@GUI_VERSION_HEX@).
if(NOT EXISTS ${GUI_BASE_DIR}/JuceLibraryCode/JuceHeader.h)
    set(GUI_VERSION 1.1.0)
    string(REGEX MATCHALL "[0-9]+" GUI_VERSION_LIST ${GUI_VERSION})
    list(GET GUI_VERSION_LIST 0 _gui_ver_major)
    list(GET GUI_VERSION_LIST 1 _gui_ver_minor)
    list(GET GUI_VERSION_LIST 2 _gui_ver_patch)
    math(EXPR GUI_VERSION_HEX "(${_gui_ver_major} << 16) + (${_gui_ver_minor} << 8) + ${_gui_ver_patch}"
         OUTPUT_FORMAT HEXADECIMAL)
    configure_file(${GUI_BASE_DIR}/JuceLibraryCode/JuceHeader.h.in
                   ${GUI_BASE_DIR}/JuceLibraryCode/JuceHeader.h)
endif()

set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS
    OEPLUGIN
    "$<$<PLATFORM_ID:Windows>:JUCE_API=__declspec(dllimport)>"
    $<$<PLATFORM_ID:Windows>:_CRT_SECURE_NO_WARNINGS>
    $<$<PLATFORM_ID:Linux>:JUCE_DISABLE_NATIVE_FILECHOOSERS=1>
    $<$<CONFIG:Debug>:DEBUG=1>
    $<$<CONFIG:Debug>:_DEBUG=1>
    $<$<CONFIG:Release>:NDEBUG=1>)

add_library(SpikeSpyStreamer SHARED
    Source/OpenEphysLib.cpp
    Source/SpikeSpyStreamer.cpp
    Source/SpikeSpyStreamer.h)

target_compile_features(SpikeSpyStreamer PRIVATE cxx_std_17)

target_include_directories(SpikeSpyStreamer PRIVATE
    ${GUI_BASE_DIR}/JuceLibraryCode
    ${GUI_BASE_DIR}/JuceLibraryCode/modules
    ${GUI_BASE_DIR}/Plugins/Headers)

# ---------------------------------------------------------------------------
# Platform settings
# ---------------------------------------------------------------------------
if(MSVC)
    # The official Open Ephys executable ships its exported symbols via an
    # import library (open-ephys.lib). Plugin DLLs must link against it so
    # GenericProcessor/JUCE symbols resolve against open-ephys.exe at load time.
    # Download from:
    #   https://openephys.jfrog.io/artifactory/GUI-binaries/Libraries/open-ephys-lib-v1.1.0.zip
    find_file(OPEN_EPHYS_IMPORT_LIB open-ephys.lib
        HINTS ${CMAKE_CURRENT_SOURCE_DIR} ${GUI_BASE_DIR}/Build/Release ENV OPEN_EPHYS_LIB_DIR
        NO_DEFAULT_PATH)

    if(NOT OPEN_EPHYS_IMPORT_LIB)
        message(FATAL_ERROR
            "open-ephys.lib was not found.\n"
            "Download https://openephys.jfrog.io/artifactory/GUI-binaries/Libraries/open-ephys-lib-v1.1.0.zip\n"
            "and extract open-ephys.lib into ${CMAKE_CURRENT_SOURCE_DIR} "
            "(or ${GUI_BASE_DIR}/Build/Release, or set OPEN_EPHYS_LIB_DIR).")
    endif()

    target_link_libraries(SpikeSpyStreamer PRIVATE ${OPEN_EPHYS_IMPORT_LIB})
    target_compile_options(SpikeSpyStreamer PRIVATE /sdl- /nologo /MP /W0 /bigobj)

elseif(LINUX)
    target_link_libraries(SpikeSpyStreamer PRIVATE
        GL X11 Xext Xinerama asound dl freetype pthread rt)
    set_property(TARGET SpikeSpyStreamer APPEND_STRING PROPERTY LINK_FLAGS
        "-fvisibility=hidden -fPIC -rdynamic -Wl,-rpath,'\$ORIGIN' -Wl,-rpath,'\$ORIGIN/../shared'")
    target_compile_options(SpikeSpyStreamer PRIVATE -fPIC -rdynamic -fvisibility=hidden)

elseif(APPLE)
    set_target_properties(SpikeSpyStreamer PROPERTIES BUNDLE TRUE)
    set_property(TARGET SpikeSpyStreamer APPEND_STRING PROPERTY LINK_FLAGS
        "-fPIC -rdynamic -undefined dynamic_lookup -rpath @loader_path")
    target_link_libraries(SpikeSpyStreamer PRIVATE dl)

endif()

# ---------------------------------------------------------------------------
# ZeroMQ
#
# Preferred: vcpkg (Windows), libzmq3-dev (Linux), Homebrew zeromq (macOS).
# With the vcpkg toolchain, the runtime DLL is also copied next to the plugin
# (VCPKG_APPLOCAL_DEPS) so the artifact only needs the plugin DLL itself.
# ---------------------------------------------------------------------------
find_package(zeromq CONFIG QUIET)
find_package(unofficial-zeromq CONFIG QUIET)

if(TARGET libzmq)
    set(ZMQ_LINK libzmq)
elseif(TARGET unofficial-zeromq::libzmq)
    set(ZMQ_LINK unofficial-zeromq::libzmq)
elseif(TARGET libzmq-static)
    set(ZMQ_LINK libzmq-static)
elseif(TARGET unofficial-zeromq::libzmq-static)
    set(ZMQ_LINK unofficial-zeromq::libzmq-static)
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
        set(ZMQ_LINK ${ZMQ_LIBRARY})
    else()
        message(FATAL_ERROR
            "ZeroMQ was not found. Install it (Windows: vcpkg install zeromq:x64-windows, "
            "Linux: libzmq3-dev, macOS: brew install zeromq) or set ZMQ_ROOT to an "
            "existing ZeroMQ build.")
    endif()
endif()

target_link_libraries(SpikeSpyStreamer PRIVATE ${ZMQ_LINK})
if(ZMQ_PC_INCLUDE_DIRS)
    target_include_directories(SpikeSpyStreamer PRIVATE ${ZMQ_PC_INCLUDE_DIRS})
endif()
if(ZMQ_INCLUDE_DIR)
    target_include_directories(SpikeSpyStreamer PRIVATE ${ZMQ_INCLUDE_DIR})
endif()

install(TARGETS SpikeSpyStreamer
    RUNTIME DESTINATION plugins
    LIBRARY DESTINATION plugins
    BUNDLE DESTINATION plugins)
