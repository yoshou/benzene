# The DSP skel's toolchain: hexagon-clang of the Hexagon SDK for a QuRT shared object loaded through FastRPC.
# HEXAGON_SDK_ROOT / HEXAGON_TOOLS_ROOT: from the command line (CMakeLists.txt passes them), else the environment.
# DSP_VERSION picks the arch through the SDK's hexagon_arch.cmake (HEXAGON_ARCH, XQF_ARGS).

# CMake reads a toolchain file more than once per configure: the include directories go in once
include_guard(GLOBAL)

foreach(root HEXAGON_SDK_ROOT HEXAGON_TOOLS_ROOT)
    if(NOT ${root})
        set(${root} "$ENV{${root}}")
    endif()
    if(NOT ${root})
        message(FATAL_ERROR "${root} is not set")
    endif()
    file(TO_CMAKE_PATH "${${root}}" ${root})
endforeach()
# try_compile projects load this file again and need the roots too
set(CMAKE_TRY_COMPILE_PLATFORM_VARIABLES HEXAGON_SDK_ROOT HEXAGON_TOOLS_ROOT DSP_VERSION)

# QURT: the SDK's hexagon_fun.cmake builds for the DSP on this system name
set(CMAKE_SYSTEM_NAME QURT)
set(CMAKE_SYSTEM_PROCESSOR Hexagon)
set(CMAKE_SYSTEM_VERSION 1)
include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_arch.cmake)

set(tools ${HEXAGON_TOOLS_ROOT}/Tools)
set(HEXAGON_LIB_DIR ${tools}/target/hexagon/lib)  # hexagon_fun.cmake reads it
set(CMAKE_C_COMPILER ${tools}/bin/hexagon-clang)
set(CMAKE_CXX_COMPILER ${tools}/bin/hexagon-clang++)
set(CMAKE_ASM_COMPILER ${tools}/bin/hexagon-clang++)
set(CMAKE_AR ${tools}/bin/hexagon-ar)
set(CMAKE_PREFIX_PATH ${tools}/target/hexagon)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# The QuRT headers of the arch, then the SDK's own and FastRPC's
set(qurt ${HEXAGON_SDK_ROOT}/rtos/qurt/compute${HEXAGON_ARCH})
include_directories(${qurt}/include ${qurt}/include/qurt ${qurt}/include/posix)
include_directories(SYSTEM ${HEXAGON_SDK_ROOT}/incs ${HEXAGON_SDK_ROOT}/incs/stddef
                    ${HEXAGON_SDK_ROOT}/ipc/fastrpc/incs)

# HVX and HMX of the arch; small data off (-G0) and position independent code, as a FastRPC skel needs
set(arch_flags "-mcpu=${HEXAGON_ARCH} -m${HEXAGON_ARCH} -mhvx=${HEXAGON_ARCH} -mhmx")
set(compile_flags "${arch_flags} -fvectorize -flto -Wall -Werror")
set(compile_flags "${compile_flags} -fno-zero-initialized-in-bss -G0 -fdata-sections -fpic ${XQF_ARGS}")
foreach(lang C CXX ASM)
    set(CMAKE_${lang}_FLAGS_RELEASE "${compile_flags} -O2")
    set(CMAKE_${lang}_FLAGS_RELWITHDEBINFO "${compile_flags} -O2 -g")
    set(CMAKE_${lang}_FLAGS_DEBUG "${compile_flags} -O0 -D_DEBUG -g")
    set(CMAKE_SHARED_LIBRARY_SONAME_${lang}_FLAG "-Wl,-soname,")
endforeach()

# The skel links through the C driver against the G0 PIC libraries. The heap calls are wrapped so that the SDK's
# libraries route them to the DSP process's heap.
set(link_rule
    ${tools}/bin/hexagon-clang ${arch_flags} -G0 -fpic -Wl,-Bsymbolic
    -Wl,-L${HEXAGON_LIB_DIR}/${HEXAGON_ARCH}/G0/pic -Wl,-L${HEXAGON_LIB_DIR}/
    -Wl,--no-threads -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=free -Wl,--wrap=realloc -Wl,--wrap=memalign
    -shared "-o <TARGET> <SONAME_FLAG><TARGET_SONAME>" "<LINK_FLAGS>"
    -Wl,--start-group "<OBJECTS>" "<LINK_LIBRARIES>" -Wl,--end-group -lc)
string(REPLACE ";" " " link_rule "${link_rule}")
set(CMAKE_C_CREATE_SHARED_LIBRARY "${link_rule}")
set(CMAKE_CXX_CREATE_SHARED_LIBRARY "${link_rule}")
