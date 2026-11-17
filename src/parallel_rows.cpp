#include "parallel_rows.hpp"

#include <algorithm>
#include <atomic>
#include <exception>
#include <thread>
#include <vector>

namespace stringkernel::detail {
namespace {

class JoinThreads {
  public:
    JoinThreads(std::vector<std::thread>& threads, std::atomic<bool>& cancelled)
        : threads_(threads), cancelled_(cancelled) {}
    ~JoinThreads() {
        cancelled_.store(true, std::memory_order_relaxed);
        join();
    }
    void join() {
        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

  private:
    std::vector<std::thread>& threads_;
    std::atomic<bool>& cancelled_;
};

} // namespace

void parallel_rows(std::size_t count, int threads, const std::function<void(std::size_t)>& task) {
    const auto worker_count = std::min(count, static_cast<std::size_t>(threads));
    if (worker_count <= 1) {
        for (std::size_t row = 0; row < count; ++row) {
            task(row);
        }
        return;
    }

    std::atomic<std::size_t> next{0};
    std::atomic<bool> cancelled{false};
    std::vector<std::exception_ptr> errors(worker_count);
    const auto work = [&](std::size_t worker) {
        while (!cancelled.load(std::memory_order_relaxed)) {
            const auto row = next.fetch_add(1, std::memory_order_relaxed);
            if (row >= count) {
                break;
            }
            try {
                task(row);
            } catch (...) {
                errors[worker] = std::current_exception();
                cancelled.store(true, std::memory_order_relaxed);
                break;
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(worker_count - 1);
    JoinThreads joined(workers, cancelled);
    for (std::size_t worker = 1; worker < worker_count; ++worker) {
        workers.emplace_back(work, worker);
    }
    work(0);
    joined.join();
    for (const auto& error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
}

} // namespace stringkernel::detail
