#ifndef STRINGKERNEL_CPU_KERNELS_HPP
#define STRINGKERNEL_CPU_KERNELS_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace stringkernel::detail {

// Configuration is validated at the public boundary. Views remain caller-owned.
double weighted_degree_cpu(std::string_view a, std::string_view b, int degree,
                           bool use_simd = true);
std::size_t positional_matches(std::string_view a, std::string_view b, bool use_simd = true);
std::string_view cpu_backend() noexcept;

namespace cpu {
// Scalar and ISA-specific backends use these same counting operations. A mask
// contains one equality bit per byte; absent tail bytes always have zero bits.
inline unsigned portable_popcount(std::uint32_t value) noexcept {
    value -= (value >> 1) & 0x55555555U;
    value = (value & 0x33333333U) + ((value >> 2) & 0x33333333U);
    value = (value + (value >> 4)) & 0x0f0f0f0fU;
    return (value * 0x01010101U) >> 24;
}

template <typename Operations>
std::uint32_t equality_mask(const char* a, const char* b, std::size_t length,
                            const Operations& operations) noexcept {
    if (length == 32) {
        return operations.equal_32(a, b);
    }
    std::uint32_t mask = 0;
    for (std::size_t i = 0; i < length; ++i) {
        mask |= static_cast<std::uint32_t>(a[i] == b[i]) << i;
    }
    return mask;
}

template <typename Operations>
std::size_t positional_masked(std::string_view a, std::string_view b,
                              const Operations& operations) noexcept {
    const auto length = std::min(a.size(), b.size());
    std::size_t result = 0;
    std::size_t position = 0;
    while (position < length) {
        const auto bytes = std::min(std::size_t{32}, length - position);
        result += operations.count(
            equality_mask(a.data() + position, b.data() + position, bytes, operations));
        position += bytes;
    }
    return result;
}

template <typename Operations>
double weighted_degree_masked(std::string_view a, std::string_view b, int degree,
                              const Operations& operations) noexcept {
    const auto length = std::min(a.size(), b.size());
    if (length == 0) {
        return 0;
    }
    if (degree == 1) {
        return static_cast<double>(positional_masked(a, b, operations));
    }
    // This evaluator is used only for degrees <=32, regardless of input length.
    std::array<std::size_t, 32> counts{};
    const auto maximum = std::min(length, static_cast<std::size_t>(degree));
    auto low = equality_mask(a.data(), b.data(), std::min(length, std::size_t{32}), operations);
    std::size_t position = 0;
    while (position < length) {
        const auto remaining = length - position;
        const auto high = remaining > 32
                              ? equality_mask(a.data() + position + 32, b.data() + position + 32,
                                              std::min(remaining - 32, std::size_t{32}), operations)
                              : 0;
        const auto mask =
            static_cast<std::uint64_t>(low) | (static_cast<std::uint64_t>(high) << 32);
        auto matches = mask;
        for (std::size_t order = 0; order < maximum; ++order) {
            const auto starts = static_cast<std::uint32_t>(matches);
            if (starts == 0) {
                break;
            }
            counts[order] += operations.count(starts);
            matches &= mask >> (order + 1);
        }
        low = high;
        position += std::min(remaining, std::size_t{32});
    }
    const auto d = static_cast<double>(degree);
    const auto scale = 2.0 / (d * (d + 1.0));
    double result = 0;
    for (std::size_t order = 0; order < maximum; ++order) {
        result += static_cast<double>(counts[order]) * (d - static_cast<double>(order)) * scale;
    }
    return result;
}

#if defined(STRINGKERNEL_HAS_AVX2)
double weighted_degree_avx2(std::string_view a, std::string_view b, int degree) noexcept;
std::size_t positional_avx2(std::string_view a, std::string_view b) noexcept;
#endif
#if defined(STRINGKERNEL_HAS_NEON)
double weighted_degree_neon(std::string_view a, std::string_view b, int degree) noexcept;
std::size_t positional_neon(std::string_view a, std::string_view b) noexcept;
#endif
} // namespace cpu
} // namespace stringkernel::detail
#endif
