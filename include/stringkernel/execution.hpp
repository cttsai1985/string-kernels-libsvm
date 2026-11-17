#ifndef STRINGKERNEL_EXECUTION_HPP
#define STRINGKERNEL_EXECUTION_HPP

#include <string_view>

namespace stringkernel {

enum class CpuBackend { automatic, scalar };

struct ExecutionConfig {
    int threads = 1;
    CpuBackend backend = CpuBackend::automatic;
    void validate() const;
};

// Preferred implementation available on this CPU; a call may request scalar.
std::string_view cpu_backend() noexcept;

} // namespace stringkernel
#endif
