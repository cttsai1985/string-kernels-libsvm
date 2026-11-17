#include "parallel_rows.hpp"
#include "stringkernel/execution.hpp"

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

namespace {
using stringkernel::CpuBackend;
using stringkernel::ExecutionConfig;
using stringkernel::detail::parallel_rows;
} // namespace

TEST_CASE("execution defaults use one thread and automatic CPU dispatch") {
    const ExecutionConfig config;
    CHECK(config.threads == 1);
    CHECK(stringkernel::detail::maximum_buffered_rows == 64);
    CHECK(config.backend == CpuBackend::automatic);
    CHECK_NOTHROW(config.validate());
    CHECK_NOTHROW((ExecutionConfig{4, CpuBackend::scalar}.validate()));
    CHECK_NOTHROW(
        (ExecutionConfig{std::numeric_limits<int>::max(), CpuBackend::automatic}.validate()));
    const auto backend = stringkernel::cpu_backend();
    CHECK((backend == "scalar" || backend == "avx2" || backend == "neon"));
}

TEST_CASE("execution rejects invalid thread counts and CPU selections") {
    for (const auto count : {0, -1, std::numeric_limits<int>::min()}) {
        CHECK_THROWS_AS((ExecutionConfig{count, CpuBackend::automatic}.validate()),
                        std::invalid_argument);
    }
    CHECK_THROWS_AS((ExecutionConfig{1, static_cast<CpuBackend>(-1)}.validate()),
                    std::invalid_argument);
}

TEST_CASE("parallel rows write every requested result exactly once") {
    for (const auto threads : {1, 2, 4, 100}) {
        std::vector<std::atomic<unsigned>> visits(37);
        std::vector<std::atomic<std::size_t>> results(37);
        for (std::size_t row = 0; row < visits.size(); ++row) {
            visits[row].store(0);
            results[row].store(0);
        }
        parallel_rows(visits.size(), threads, [&](std::size_t row) {
            visits[row].fetch_add(1);
            results[row].store((row + 1) * (row + 1));
        });
        for (std::size_t row = 0; row < visits.size(); ++row) {
            CHECK(visits[row].load() == 1);
            CHECK(results[row].load() == (row + 1) * (row + 1));
        }
    }
}

TEST_CASE("single-thread rows stay on the caller and empty ranges do no work") {
    const auto caller = std::this_thread::get_id();
    bool used_caller = true;
    parallel_rows(5, 1, [&](std::size_t) {
        used_caller = used_caller && std::this_thread::get_id() == caller;
    });
    CHECK(used_caller);
    bool invoked = false;
    parallel_rows(0, 4, [&](std::size_t) { invoked = true; });
    CHECK_FALSE(invoked);
    CHECK_THROWS_AS(
        parallel_rows(1, 1, [](std::size_t) { throw std::domain_error("serial task"); }),
        std::domain_error);
}

TEST_CASE("parallel task exceptions propagate after active tasks have finished") {
    std::mutex mutex;
    std::condition_variable ready;
    bool first_started = false;
    bool second_throwing = false;
    std::atomic<bool> first_finished{false};
    bool joined_before_rethrow = false;
    try {
        parallel_rows(2, 2, [&](std::size_t row) {
            if (row == 0) {
                std::unique_lock<std::mutex> lock(mutex);
                first_started = true;
                ready.notify_all();
                // The timeout only bounds a broken serial implementation; task
                // ordering is determined by the predicates, without sleeps.
                if (!ready.wait_for(lock, std::chrono::seconds(5),
                                    [&] { return second_throwing; })) {
                    throw std::runtime_error("second task never ran concurrently");
                }
                lock.unlock();
                first_finished.store(true);
            } else {
                std::unique_lock<std::mutex> lock(mutex);
                if (!ready.wait_for(lock, std::chrono::seconds(5), [&] { return first_started; })) {
                    throw std::runtime_error("first task never ran concurrently");
                }
                second_throwing = true;
                lock.unlock();
                ready.notify_all();
                throw std::domain_error("parallel task");
            }
        });
    } catch (const std::domain_error& error) {
        CHECK(std::string_view(error.what()) == "parallel task");
        joined_before_rethrow = first_finished.load();
    }
    CHECK(joined_before_rethrow);
}
