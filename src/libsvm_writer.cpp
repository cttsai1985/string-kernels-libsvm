#include "stringkernel/libsvm_writer.hpp"
#include "kernel_evaluator.hpp"
#include "parallel_rows.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace stringkernel {
namespace {
bool is_space(unsigned char character) {
    return character == ' ' || character == '\t' || character == '\n' || character == '\r' ||
           character == '\v' || character == '\f';
}

void validate_samples(const Dataset& data) {
    if (data.empty()) {
        throw std::invalid_argument("kernel datasets must not be empty");
    }
    if (data.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("dataset exceeds LIBSVM's integer index limit");
    }
    for (const auto& sample : data) {
        if (sample.sequence.empty() ||
            std::any_of(sample.sequence.begin(), sample.sequence.end(), is_space)) {
            throw std::invalid_argument("sequences must be nonempty whitespace-free tokens");
        }
        if (!std::isfinite(sample.label)) {
            throw std::invalid_argument("labels must be finite");
        }
    }
}

void validate_lengths(const Dataset& reference, const Dataset& target) {
    const auto length = reference.front().sequence.size();
    for (const auto* data : {&reference, &target}) {
        if (std::any_of(data->begin(), data->end(), [length](const Sample& sample) {
                return sample.sequence.size() != length;
            })) {
            throw std::invalid_argument("DSK requires equal sequence lengths in all datasets");
        }
    }
}

void require_output(const std::ostream& output) {
    if (!output) {
        throw std::runtime_error("failed to write precomputed kernel output");
    }
}
} // namespace

void write_precomputed(std::ostream& output, const Dataset& reference, const Dataset& target,
                       const KernelConfig& config, const ExecutionConfig& execution) {
    config.validate();
    execution.validate();
    validate_samples(reference);
    validate_samples(target);
    if (config.type == KernelType::distance) {
        validate_lengths(reference, target);
    }
    require_output(output);
    std::vector<std::string_view> reference_sequences;
    std::vector<std::string_view> target_sequences;
    reference_sequences.reserve(reference.size());
    target_sequences.reserve(target.size());
    for (const auto& sample : reference) {
        reference_sequences.push_back(sample.sequence);
    }
    for (const auto& sample : target) {
        target_sequences.push_back(sample.sequence);
    }
    const detail::KernelEvaluator evaluator(config, reference_sequences, target_sequences,
                                            execution.backend);
    const auto format_row = [&](std::size_t i) {
        std::ostringstream row;
        row.imbue(std::locale::classic());
        row << std::setprecision(std::numeric_limits<double>::max_digits10);
        row << target[i].label << " 0:" << i + 1;
        for (std::size_t j = 0; j < reference.size(); ++j) {
            row << ' ' << j + 1 << ':' << evaluator.value(i, j);
        }
        row << '\n';
        return row.str();
    };
    const auto workers = std::min(target.size(), static_cast<std::size_t>(execution.threads));
    // Bound buffered text independently of matrix size; amortize worker creation.
    const auto block_rows =
        workers == 1 ? 1 : std::min(target.size(), detail::maximum_buffered_rows);
    for (std::size_t first = 0; first < target.size(); first += block_rows) {
        const auto count = std::min(block_rows, target.size() - first);
        std::vector<std::string> rows(count);
        detail::parallel_rows(count, execution.threads, [&](std::size_t offset) {
            rows[offset] = format_row(first + offset);
        });
        for (const auto& row : rows) {
            output << row;
            require_output(output);
        }
    }
    output.flush();
    require_output(output);
}
} // namespace stringkernel
