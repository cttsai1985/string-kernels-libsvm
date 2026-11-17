#include "stringkernel/execution.hpp"
#include "cpu_kernels.hpp"

#include <stdexcept>

namespace stringkernel {

void ExecutionConfig::validate() const {
    if (threads <= 0) {
        throw std::invalid_argument("thread count must be a positive integer");
    }
    if (backend != CpuBackend::automatic && backend != CpuBackend::scalar) {
        throw std::invalid_argument("CPU backend must be automatic or scalar");
    }
}

std::string_view cpu_backend() noexcept { return detail::cpu_backend(); }

} // namespace stringkernel
