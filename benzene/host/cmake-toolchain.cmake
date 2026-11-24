# The host runner's toolchain: clang of the Snapdragon toolchain image, cross-compiling for the board's aarch64 Linux
# (build.sh).

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm64)

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_C_COMPILER_TARGET aarch64-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET aarch64-linux-gnu)

# Fast floating point, keeping infinities and NaNs
set(host_flags "-march=armv8.7-a -fvectorize -ffp-model=fast -fno-finite-math-only")
set(host_flags "${host_flags} -Wall -Werror -Wno-unused-command-line-argument")
set(CMAKE_C_FLAGS_INIT "${host_flags}")
set(CMAKE_CXX_FLAGS_INIT "${host_flags}")
