#pragma once

#include <cstdint>

namespace benzene::host {

double now_microseconds();

// Prints "norm1 <ms> qkv <ms> ... fc2 <ms>": the protocol::block_stat stages at the start of `statistics`.
void print_block_stages(const int32_t * statistics);

}  // namespace benzene::host
