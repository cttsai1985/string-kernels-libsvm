#ifndef STRINGKERNEL_LIBSVM_WRITER_H
#define STRINGKERNEL_LIBSVM_WRITER_H

#include "stringkernel/dataset.hpp"
#include "stringkernel/kernels.hpp"

#include <iosfwd>

namespace stringkernel {

void write_precomputed(std::ostream& output, const Dataset& reference, const Dataset& target,
                       const KernelConfig& config = {});

} // namespace stringkernel

#endif
