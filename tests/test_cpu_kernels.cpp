#include "cpu_kernels.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <string_view>

namespace {
// Enumerate the mathematical definition rather than masks or matching runs.
double byte_oracle(std::string_view a, std::string_view b, int degree) {
    const auto shared = std::min(a.size(), b.size());
    const auto maximum = std::min(shared, static_cast<std::size_t>(degree));
    long double total = 0;
    for (std::size_t order = 1; order <= maximum; ++order) {
        for (std::size_t start = 0; start <= shared - order; ++start) {
            if (a.substr(start, order) == b.substr(start, order)) {
                total += static_cast<long double>(degree) - order + 1;
            }
        }
    }
    const auto d = static_cast<long double>(degree);
    return static_cast<double>(2 * total / (d * (d + 1)));
}

std::string byte_sequence(std::size_t length) {
    std::string result(length, '\0');
    for (std::size_t i = 0; i < length; ++i) {
        constexpr std::array<unsigned char, 8> bytes{0, 128, 255, 'A', 'B', 'C', 17, 71};
        result[i] = static_cast<char>(bytes[i % bytes.size()]);
    }
    return result;
}

void require_cpu_oracle(std::string_view a, std::string_view b, int degree) {
    const auto expected = byte_oracle(a, b, degree);
    for (const bool simd : {false, true}) {
        CAPTURE(a.size(), b.size(), degree, simd);
        REQUIRE(stringkernel::detail::weighted_degree_cpu(a, b, degree, simd) ==
                Catch::Approx(expected).epsilon(1e-12).margin(1e-14));
    }
}
} // namespace

TEST_CASE("CPU kernels preserve arbitrary-byte matches across vector boundaries") {
    for (const std::size_t length : {0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 95, 96, 97}) {
        const auto original = byte_sequence(length);
        auto interrupted = original;
        for (const std::size_t position : {0, 15, 16, 31, 32, 63, 64, 95}) {
            if (position < interrupted.size()) {
                interrupted[position] ^= '\x01';
            }
        }
        auto different = original;
        for (auto& value : different) {
            value ^= '\x01';
        }
        for (const int degree : {1, 2, 5, 16, 31, 32, 33, std::numeric_limits<int>::max()}) {
            require_cpu_oracle(original, original, degree);
            require_cpu_oracle(original, interrupted, degree);
            require_cpu_oracle(original, different, degree);
            require_cpu_oracle(original, std::string_view(interrupted).substr(0, length / 2),
                               degree);
        }
        std::size_t expected = 0;
        for (std::size_t i = 0; i < length; ++i) {
            expected += original[i] == interrupted[i];
        }
        for (const bool simd : {false, true}) {
            REQUIRE(stringkernel::detail::positional_matches(original, original, simd) == length);
            REQUIRE(stringkernel::detail::positional_matches(original, different, simd) == 0);
            REQUIRE(stringkernel::detail::positional_matches(original, interrupted, simd) ==
                    expected);
        }
    }
}

TEST_CASE("CPU loads stay within unaligned sequence views") {
    auto left = byte_sequence(132);
    auto right = left;
    right[32] = '\0';
    right[63] = static_cast<char>(128);
    right[64] = static_cast<char>(255);
    for (const std::size_t offset : {1, 2, 3, 7, 15, 31}) {
        for (const std::size_t length : {1, 31, 32, 33, 63, 64, 65, 97}) {
            for (const int degree : {1, 5, 32, 33}) {
                require_cpu_oracle(std::string_view(left).substr(offset, length),
                                   std::string_view(right).substr(offset, length), degree);
            }
        }
    }
}

TEST_CASE("CPU weighted degree preserves matching runs across chunk boundaries") {
    std::string left(129, 'A');
    auto right = left;
    right[0] = 'B';
    right[48] = 'B';
    right[80] = 'B';
    right[128] = 'B';
    for (const int degree : {1, 2, 5, 16, 31, 32, 33, 64, std::numeric_limits<int>::max()}) {
        require_cpu_oracle(left, right, degree);
    }
    REQUIRE(stringkernel::detail::weighted_degree_cpu("AC", "AD", 2) == Catch::Approx(2.0 / 3.0));
}

TEST_CASE("CPU backend reports a portable or supported instruction path") {
    const auto backend = stringkernel::detail::cpu_backend();
    REQUIRE((backend == "scalar" || backend == "avx2" || backend == "neon"));
}

TEST_CASE("CPU tail comparisons respect exact allocation boundaries") {
    for (const std::size_t length : {0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65}) {
        auto left = std::make_unique<char[]>(length);
        auto right = std::make_unique<char[]>(length);
        for (std::size_t i = 0; i < length; ++i) {
            left[i] = static_cast<char>(i % 256);
            right[i] = left[i];
        }
        const std::string_view a(left.get(), length);
        const std::string_view b(right.get(), length);
        for (const int degree : {1, 5, 32, 33}) {
            require_cpu_oracle(a, b, degree);
        }
        for (const bool simd : {false, true}) {
            REQUIRE(stringkernel::detail::positional_matches(a, b, simd) == length);
        }
    }
}
