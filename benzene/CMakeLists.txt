# Host runner (aarch64 Linux) + DSP skel (Hexagon).
# Configure inside the Snapdragon toolchain image (see build.sh).
cmake_minimum_required(VERSION 3.22.2)
project(benzene C CXX)

set(DSP_VERSION v75 CACHE STRING "Hexagon arch of the skel")
include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)
include(ExternalProject)

add_executable(benzene
    host/anysplat_commands.cpp
    host/command_line.cpp
    host/cosmos3_edge_commands.cpp
    host/fastrpc.cpp
    host/feature_matching.cpp
    host/feature_matching_commands.cpp
    host/files.cpp
    host/loma_commands.cpp
    host/main.cpp
    host/model_file.cpp
    host/timing.cpp
    host/vggt_commands.cpp
    host/vggt_network.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/benzene_stub.c)
build_idl(benzene.idl benzene)
set_target_properties(benzene PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_include_directories(benzene PRIVATE
    ${HEXAGON_SDK_ROOT}/incs ${HEXAGON_SDK_ROOT}/incs/stddef
    ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(benzene PRIVATE dl pthread)
install(TARGETS benzene)

ExternalProject_Add(benzene-dsp
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dsp BUILD_ALWAYS ON
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_CURRENT_SOURCE_DIR}/dsp/cmake-toolchain.cmake
        -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
        -DHEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT}
        -DHEXAGON_TOOLS_ROOT=${HEXAGON_TOOLS_ROOT}
        -DDSP_VERSION=${DSP_VERSION}
        -DPREBUILT_LIB_DIR=toolv21_${DSP_VERSION})
