#ifndef STRINGKERNEL_PARALLEL_ROWS_HPP
#define STRINGKERNEL_PARALLEL_ROWS_HPP

#include <cstddef>
#include <functional>

namespace stringkernel::detail {

// Hard bound on CLI text buffering, independent of the number of matrix rows.
inline constexpr std::size_t maximum_buffered_rows = 64;

// The boundary validates threads > 0. Tasks may run concurrently; the caller
// keeps their captured data alive and gives each row disjoint output storage.
// All active tasks finish before this function returns or propagates an error.
void parallel_rows(std::size_t count, int threads, const std::function<void(std::size_t)>& task);

} // namespace stringkernel::detail
#endif
