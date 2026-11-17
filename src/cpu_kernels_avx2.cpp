#include "cpu_kernels.hpp"

#include <immintrin.h>

namespace stringkernel::detail::cpu {
namespace {
struct Avx2Operations {
    std::uint32_t equal_32(const char* a, const char* b) const noexcept {
        const auto left = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
        const auto right = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
        return static_cast<std::uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(left, right)));
    }
    unsigned count(std::uint32_t mask) const noexcept {
        return static_cast<unsigned>(_mm_popcnt_u32(mask));
    }
};
} // namespace

double weighted_degree_avx2(std::string_view a, std::string_view b, int degree) noexcept {
    return weighted_degree_masked(a, b, degree, Avx2Operations{});
}

std::size_t positional_avx2(std::string_view a, std::string_view b) noexcept {
    return positional_masked(a, b, Avx2Operations{});
}
} // namespace stringkernel::detail::cpu
