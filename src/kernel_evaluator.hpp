#ifndef STRINGKERNEL_KERNEL_EVALUATOR_HPP
#define STRINGKERNEL_KERNEL_EVALUATOR_HPP

#include "stringkernel/kernels.hpp"

#include <string_view>
#include <unordered_map>
#include <vector>

namespace stringkernel::detail {

using SpectrumCounts = std::unordered_map<std::string_view, std::size_t>;
SpectrumCounts spectrum_counts(std::string_view sequence, int degree);
long double count_dot(const SpectrumCounts& left, const SpectrumCounts& right);

// Internal, call-scoped preparation. Callers own the strings and keep them
// unchanged until evaluation ends. Configuration is validated at the boundary.
class KernelEvaluator {
  public:
    KernelEvaluator(const KernelConfig& config, const std::vector<std::string_view>& reference,
                    const std::vector<std::string_view>& target);
    bool symmetric() const { return symmetric_; }
    double value(std::size_t row, std::size_t column) const;

  private:
    struct Cohort {
        std::vector<std::string_view> sequences;
        std::vector<double> self;
        std::vector<SpectrumCounts> counts;
    };
    Cohort prepare(const std::vector<std::string_view>& sequences) const;
    KernelConfig config_;
    bool symmetric_;
    Cohort reference_;
    Cohort target_;
};

} // namespace stringkernel::detail
#endif
