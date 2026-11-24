#pragma once

#include <atomic>

namespace benzene::runtime {

class SpinBarrier {
  public:
    SpinBarrier() = default;
    explicit SpinBarrier(int participants) : participants_(participants) {}

    void reset(int participants) {
        arrived_.store(0);
        generation_.store(0);
        participants_ = participants;
    }

    void wait() {
        const int generation = generation_.load();
        if (arrived_.fetch_add(1) == participants_ - 1) {
            arrived_.store(0);
            generation_.fetch_add(1);
            return;
        }
        while (generation_.load() == generation) {}
    }

  private:
    std::atomic<int> arrived_{0};
    std::atomic<int> generation_{0};
    int participants_ = 0;
};

// Takes `in_use` (0 -> 1) for the scope if it is 0, without waiting; taken() is false when another call holds it.
class ExclusiveUse {
  public:
    explicit ExclusiveUse(std::atomic<int> & in_use) : in_use_(in_use) {
        int idle = 0;
        taken_ = in_use_.compare_exchange_strong(idle, 1);
    }
    ~ExclusiveUse() {
        if (taken_) in_use_.store(0);
    }
    ExclusiveUse(const ExclusiveUse &) = delete;
    ExclusiveUse & operator=(const ExclusiveUse &) = delete;

    bool taken() const { return taken_; }

  private:
    std::atomic<int> & in_use_;
    bool taken_;
};

}  // namespace benzene::runtime
