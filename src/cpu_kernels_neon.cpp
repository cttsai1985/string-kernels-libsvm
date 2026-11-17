#include "cpu_kernels.hpp"

#if defined(_MSC_VER)
#include <arm64_neon.h>
#else
#include <arm_neon.h>
#endif

namespace stringkernel::detail::cpu {
namespace {
std::uint32_t mask_16(const char* a, const char* b) noexcept {
    const auto left = vld1q_u8(reinterpret_cast<const std::uint8_t*>(a));
    const auto right = vld1q_u8(reinterpret_cast<const std::uint8_t*>(b));
    const std::uint8_t bits[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    const auto selected = vandq_u8(vceqq_u8(left, right), vld1q_u8(bits));
    const auto low = vaddv_u8(vget_low_u8(selected));
    const auto high = vaddv_u8(vget_high_u8(selected));
    return static_cast<std::uint32_t>(low) | (static_cast<std::uint32_t>(high) << 8);
}

struct NeonOperations {
    std::uint32_t equal_32(const char* a, const char* b) const noexcept {
        return mask_16(a, b) | (mask_16(a + 16, b + 16) << 16);
    }
    unsigned count(std::uint32_t mask) const noexcept { return portable_popcount(mask); }
};
} // namespace

double weighted_degree_neon(std::string_view a, std::string_view b, int degree) noexcept {
    return weighted_degree_masked(a, b, degree, NeonOperations{});
}

std::size_t positional_neon(std::string_view a, std::string_view b) noexcept {
    return positional_masked(a, b, NeonOperations{});
}
} // namespace stringkernel::detail::cpu
