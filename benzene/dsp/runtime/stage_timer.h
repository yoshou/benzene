#pragma once

#include <cstdint>

#include <HAP_perf.h>

namespace benzene::runtime {

class StageTimer {
  public:
    explicit StageTimer(int32_t * statistics)
        : statistics_(statistics), start_(HAP_perf_get_time_us()), last_(start_) {}

    // Adds the microseconds since the previous lap (or the construction) to statistics[stage].
    void lap(int stage) {
        const uint64_t now = HAP_perf_get_time_us();
        statistics_[stage] += static_cast<int32_t>(now - last_);
        last_ = now;
    }

    // Microseconds since the construction.
    int elapsed() const { return static_cast<int>(HAP_perf_get_time_us() - start_); }

  private:
    int32_t * statistics_;
    uint64_t start_;
    uint64_t last_;
};

}  // namespace benzene::runtime
