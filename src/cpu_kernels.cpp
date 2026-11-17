#include "cpu_kernels.hpp"

#if defined(STRINGKERNEL_HAS_AVX2) && defined(_MSC_VER)
#include <intrin.h>
#endif

namespace stringkernel::detail {
namespace {
struct ScalarOperations {
    std::uint32_t equal_32(const char* a, const char* b) const noexcept {
        std::uint32_t result = 0;
        for (std::size_t i = 0; i < 32; ++i) {
            result |= static_cast<std::uint32_t>(a[i] == b[i]) << i;
        }
        return result;
    }
    unsigned count(std::uint32_t mask) const noexcept { return cpu::portable_popcount(mask); }
};

double weighted_degree_scalar(std::string_view a, std::string_view b, int degree) noexcept {
    return cpu::weighted_degree_masked(a, b, degree, ScalarOperations{});
}

std::size_t positional_scalar(std::string_view a, std::string_view b) noexcept {
    return cpu::positional_masked(a, b, ScalarOperations{});
}

#if defined(STRINGKERNEL_HAS_AVX2)
bool avx2_supported() noexcept {
#if defined(_MSC_VER)
    int registers[4];
    __cpuid(registers, 0);
    if (registers[0] < 7) {
        return false;
    }
    __cpuidex(registers, 1, 0);
    constexpr auto required = (1U << 23) | (1U << 27) | (1U << 28);
    if ((static_cast<unsigned>(registers[2]) & required) != required || (_xgetbv(0) & 6) != 6) {
        return false;
    }
    __cpuidex(registers, 7, 0);
    // MSVC /arch:AVX2 may use BMI2 for the mask algorithm's variable shifts.
    constexpr auto vector_and_shifts = (1U << 5) | (1U << 8);
    return (static_cast<unsigned>(registers[1]) & vector_and_shifts) == vector_and_shifts;
#else
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("popcnt");
#endif
}
#endif

struct Backend {
    double (*weighted_degree)(std::string_view, std::string_view, int) noexcept;
    std::size_t (*positional)(std::string_view, std::string_view) noexcept;
    std::string_view name;
};

const Backend& selected_backend() noexcept {
    // CPU capabilities are immutable. Initialization is thread-safe and never
    // depends on environment variables or mutable numerical configuration.
    static const Backend backend = []() noexcept -> Backend {
#if defined(STRINGKERNEL_HAS_AVX2)
        if (avx2_supported()) {
            return {cpu::weighted_degree_avx2, cpu::positional_avx2, "avx2"};
        }
#elif defined(STRINGKERNEL_HAS_NEON)
        return {cpu::weighted_degree_neon, cpu::positional_neon, "neon"};
#endif
        return {weighted_degree_scalar, positional_scalar, "scalar"};
    }();
    return backend;
}

// Preserve the v1.0.1 linear scan when degree exceeds the fixed mask window.
double weighted_degree_runs(std::string_view a, std::string_view b, int degree) noexcept {
    const auto length = std::min(a.size(), b.size());
    const auto maximum = static_cast<std::size_t>(degree);
    const auto d = static_cast<double>(degree);
    const auto scale = 2.0 / (d * (d + 1.0));
    double score = 0;
    std::size_t run = 0;
    for (std::size_t position = length; position > 0; --position) {
        run = a[position - 1] == b[position - 1] ? run + 1 : 0;
        const auto count = static_cast<double>(std::min(run, maximum));
        score += count * (2.0 * d - count + 1.0) * (scale / 2.0);
    }
    return score;
}
} // namespace

double weighted_degree_cpu(std::string_view a, std::string_view b, int degree, bool use_simd) {
    if (degree > 32) {
        return weighted_degree_runs(a, b, degree);
    }
    return use_simd ? selected_backend().weighted_degree(a, b, degree)
                    : weighted_degree_scalar(a, b, degree);
}

std::size_t positional_matches(std::string_view a, std::string_view b, bool use_simd) {
    return use_simd ? selected_backend().positional(a, b) : positional_scalar(a, b);
}

std::string_view cpu_backend() noexcept { return selected_backend().name; }
} // namespace stringkernel::detail
