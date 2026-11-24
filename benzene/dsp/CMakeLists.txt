cmake_minimum_required(VERSION 3.22.2)
project(benzene-dsp C CXX ASM)

include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)

include_directories(
    ${HEXAGON_SDK_ROOT}/incs
    ${HEXAGON_SDK_ROOT}/incs/stddef
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_BINARY_DIR})

# (no source file directly in this directory: the SDK's cmake helpers treat top-level C++ files as a request to
# package libc++ with the skel)
set(RUNTIME_SOURCES
    runtime/int8_scale_table.cpp
    runtime/session.cpp
    runtime/thread_pool.cpp)
set(KERNEL_SOURCES
    kernels/gemm_w8a8/gemm_w8a8.cpp
    kernels/flash_attention/flash_attention.cpp
    kernels/gemm_fp16/gemm_fp16.cpp
    kernels/convolution_int8/convolution_int8.cpp
    kernels/depthwise_int8/depthwise_int8.cpp
    kernels/gemv/gemv.cpp)
set(MODEL_SOURCES
    models/aliked/aliked.cpp
    models/cosmos3_edge/cosmos3_edge.cpp
    models/lightglue/lightglue.cpp
    models/loma/loma.cpp
    models/shared/cell_heatmap.cpp
    models/shared/feature_slots.cpp
    models/shared/vision_transformer.cpp
    models/superpoint/superpoint.cpp
    models/vggt/vggt.cpp
    models/xfeat/xfeat.cpp)
set(RPC_SOURCES
    rpc/cosmos3_edge_calls.cpp
    rpc/extractor_calls.cpp
    rpc/lightglue_calls.cpp
    rpc/loma_calls.cpp
    rpc/session_calls.cpp
    rpc/vggt_calls.cpp)

set(SKEL benzene_skel)
add_library(${SKEL} SHARED ${RUNTIME_SOURCES} ${KERNEL_SOURCES} ${MODEL_SOURCES} ${RPC_SOURCES}
            ${CMAKE_CURRENT_BINARY_DIR}/benzene_skel.c)
target_compile_definitions(${SKEL} PRIVATE NDEBUG=1)
# the kernels use the IEEE sf / hf HVX instructions (v75 executes them; without the flag the backend cannot select them)
target_compile_options(${SKEL} PRIVATE -mhvx-ieee-fp)
# C++ without the parts that need the C++ runtime library, which the DSP process does not load: no exceptions, RTTI,
# guarded statics or exit-time destructors (static objects that own memory release it explicitly at session close);
# no operator new either - heap memory comes from malloc / memalign (dsp/runtime/heap.h)
target_compile_options(${SKEL} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++23 -fno-exceptions -fno-rtti -fno-threadsafe-statics -fno-c++-static-destructors>)
build_idl(${CMAKE_CURRENT_SOURCE_DIR}/../benzene.idl ${SKEL})
install(TARGETS ${SKEL})
