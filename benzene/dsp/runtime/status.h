#pragma once

#include <expected>
#include <utility>

#include <AEEStdErr.h>

namespace benzene::runtime {

// AEE error codes (AEEStdErr.h). An Error made by status_of_code keeps the SDK call's code, named here or not.
enum class Error : unsigned {
    kFailed = AEE_EFAILED,         // a system call failed
    kNoMemory = AEE_ENOMEMORY,     // the heap, VTCM or a fixed-size table is full
    kBadParameter = AEE_EBADPARM,  // arguments the call does not support
};

using Status = std::expected<void, Error>;
template <class T>
using Result = std::expected<T, Error>;

inline int aee_code(Error error) {
    return static_cast<int>(std::to_underlying(error));
}
inline int aee_code(const Status & status) {
    return status.has_value() ? AEE_SUCCESS : aee_code(status.error());
}

inline Status status_of_code(int code) {
    if (code == AEE_SUCCESS) return {};
    return std::unexpected(static_cast<Error>(code));
}

}  // namespace benzene::runtime
