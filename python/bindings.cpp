#include "stringkernel/kernels.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

stringkernel::KernelConfig configuration(int type, int degree, double decay) {
    const stringkernel::KernelConfig config{static_cast<stringkernel::KernelType>(type), degree,
                                            decay};
    config.validate();
    return config;
}

py::array_t<double> pairwise(const std::vector<std::string>& targets,
                             const std::vector<std::string>& references, int type, int degree,
                             double decay) {
    const auto config = configuration(type, degree, decay);
    const auto limit = static_cast<std::size_t>(std::numeric_limits<py::ssize_t>::max());
    if (references.empty() || targets.empty()) {
        throw std::invalid_argument("sequence collections must not be empty");
    }
    if (references.size() > limit || targets.size() > limit / references.size()) {
        throw std::length_error("kernel matrix is too large");
    }
    py::array_t<double> result(
        {static_cast<py::ssize_t>(targets.size()), static_cast<py::ssize_t>(references.size())});
    auto* output = result.mutable_data();
    py::gil_scoped_release release;
    std::vector<double> target_self;
    std::vector<double> reference_self;
    target_self.reserve(targets.size());
    reference_self.reserve(references.size());
    for (const auto& sequence : targets) {
        target_self.push_back(stringkernel::self_kernel(sequence, config));
    }
    for (const auto& sequence : references) {
        reference_self.push_back(stringkernel::self_kernel(sequence, config));
    }
    for (std::size_t row = 0; row < targets.size(); ++row) {
        for (std::size_t column = 0; column < references.size(); ++column) {
            output[row * references.size() + column] = stringkernel::normalize_kernel(
                stringkernel::raw_kernel(targets[row], references[column], config),
                target_self[row], reference_self[column]);
        }
    }
    return result;
}

} // namespace

PYBIND11_MODULE(_native, module) {
    module.doc() = "Native normalized sequence kernels; use stringkernels.pairwise.";
    module.attr("__version__") = STRINGKERNEL_VERSION;
    module.def("validate_config", [](int type, int degree, double decay) {
        static_cast<void>(configuration(type, degree, decay));
    });
    module.def("pairwise", &pairwise);
}
