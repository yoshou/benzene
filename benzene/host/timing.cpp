#include "host/timing.h"

#include <cstdio>
#include <ctime>

#include "benzene/protocol.h"

namespace benzene::host {

double now_microseconds() {
    timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return static_cast<double>(time.tv_sec) * 1e6 + static_cast<double>(time.tv_nsec) / 1e3;
}

void print_block_stages(const int32_t * statistics) {
    static const char * const kStages[protocol::block_stat::kCount] = {"norm1", "qkv", "attention", "output",
                                                                       "norm2", "fc1", "fc2"};
    for (int stage = 0; stage < protocol::block_stat::kCount; stage++) {
        printf("%s%s %.1f", stage != 0 ? " " : "", kStages[stage], statistics[stage] / 1e3);
    }
}

}  // namespace benzene::host
