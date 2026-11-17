#include "kernel_evaluator.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace stringkernel::detail {
SpectrumCounts spectrum_counts(std::string_view sequence, int degree) {
    SpectrumCounts result;
    const auto order = static_cast<std::size_t>(degree);
    if (order > sequence.size()) {
        return result;
    }
    result.reserve(sequence.size() - order + 1);
    for (std::size_t start = 0; start <= sequence.size() - order; ++start) {
        ++result[sequence.substr(start, order)];
    }
    return result;
}

long double count_dot(const SpectrumCounts& left, const SpectrumCounts& right) {
    long double result = 0;
    for (const auto& [token, count] : left) {
        const auto found = right.find(token);
        if (found != right.end()) {
            result += static_cast<long double>(count) * found->second;
        }
    }
    return result;
}

KernelEvaluator::Cohort
KernelEvaluator::prepare(const std::vector<std::string_view>& sequences) const {
    Cohort result;
    result.sequences = sequences;
    result.self.reserve(sequences.size());
    if (config_.type == KernelType::spectrum) {
        result.counts.reserve(sequences.size());
    }
    for (const auto sequence : sequences) {
        if (config_.type == KernelType::spectrum) {
            result.counts.push_back(spectrum_counts(sequence, config_.degree));
            result.self.push_back(
                static_cast<double>(count_dot(result.counts.back(), result.counts.back())));
        } else {
            result.self.push_back(self_kernel(sequence, config_));
        }
    }
    return result;
}

KernelEvaluator::KernelEvaluator(const KernelConfig& config,
                                 const std::vector<std::string_view>& reference,
                                 const std::vector<std::string_view>& target)
    : config_(config), symmetric_(reference == target), reference_(prepare(reference)),
      target_(symmetric_ ? Cohort{} : prepare(target)) {}

double KernelEvaluator::value(std::size_t row, std::size_t column) const {
    const auto& target = symmetric_ ? reference_ : target_;
    if (symmetric_ && row == column) {
        return reference_.self[row] == 0 ? 0 : 1;
    }
    const auto raw =
        config_.type == KernelType::spectrum
            ? static_cast<double>(count_dot(target.counts[row], reference_.counts[column]))
            : raw_kernel(target.sequences[row], reference_.sequences[column], config_);
    return normalize_kernel(raw, target.self[row], reference_.self[column]);
}

} // namespace stringkernel::detail
