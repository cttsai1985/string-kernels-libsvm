# Validation evidence

Tracking issue: [#1](https://github.com/cttsai1985/string-kernels-libsvm/issues/1).
This record distinguishes observed results from planned checks.

## RED: process regressions before implementation

Built the original `origin/master` sources in a scratch directory with:

```sh
g++ -std=c++17 -w StringKernelConverter.cpp StringKernels.cpp GetLongOpt.cpp -o StringKernelConverter
python3 tests/test_cli.py --converter /path/to/original/StringKernelConverter
```

Observed failures include successful exit on empty/missing input, acceptance of
invalid degrees/types/options, ignored malformed/nonfinite records, regression
label precision loss, asymmetric unequal-length DSK, and successful exit when
redirected output fails. These tests were written and run before replacing the
converter or implementing the writer. Original valid kernel/CLI characterization
remains a separate compatibility check using parsed values.

Writer API RED command:

```sh
g++ -std=c++17 -I. -fsyntax-only tests/test_writer.cpp
```

Observed compiler error: `LibsvmWriter.h: No such file or directory`. Tests existed
before the new writer API and implementation. Semantic process regressions above
provide failure evidence beyond missing declarations.

Further unit, integration, sanitizer, coverage, and hosted CI results will be
recorded after they actually run.

## Numerical characterization and GREEN results

MEASURED on the original source through a temporary API adapter: the existing
weighted-degree behavior passed the independent feature-map/characterization
suite with 5,474 assertions across three cases. Legacy normalization and DSK
regressions failed 22 assertions. The corrected kernel suite subsequently passed
5,544 assertions across 12 cases under ASan/UBSan and warnings as errors. A
separate unequal-subnormal normalization regression failed both assertions
before its targeted fix and passed afterward.

MEASURED parser suite: 109 assertions across 16 cases passed. Besides initial
missing-API REDs, focused semantic REDs demonstrated silent underflow of a
nonzero label and erroneous acceptance of the malformed integer `+-0`, followed
by GREEN fixes.

MEASURED CLI baseline: 14 test methods produced 29 failing assertions/subtests
against the original executable. Those process contracts subsequently passed
against the modern executable.

An actual old/new conversion comparison used both supplied data files with
default WDSK degree. Parsed values, labels, serials, and matrix entries agreed
within the legacy decimal-rounding bound. MEASURED training output: eight rows,
80 compared values, maximum absolute difference `4.900000014629313e-7`.
MEASURED target output: six rows, 60 compared values, maximum absolute difference
`4.78260869574143e-7`. The comparison asserted a maximum difference no greater
than `5.0000001e-7`; it did not require identical decimal text.

Dependency tests also rejected an isolated copy in which both `svm.cpp` and its
manifest checksum were changed, demonstrating that source/manifest drift cannot
bypass the independently pinned expected hashes.

## Full local verification

MEASURED development, AddressSanitizer/UndefinedBehaviorSanitizer, and GCC coverage
configurations each passed all 39 CTest checks. The suite includes the real
classification and supplied regression workflows, independent matrix/prediction
oracles, model persistence/reload, installation, dependency pins, and CLI errors.
Project sources build with warnings treated as errors.

MEASURED project-only coverage: 260/262 lines (99.2%), 28/28 functions (100%), and
382/514 branches (74.3%). CI enforces a line-coverage floor of 98%; vendored solver
and test code are excluded. Reports are generated under `coverage/` and uploaded
by hosted CI. The unreachable-in-small-fixtures record-count guard remains outside
the exercised input range.

MEASURED documentation checks: the literal README Unix release workflow generated
eight training rows, six target rows, and six finite predictions. A separate
installed-library consumer compiled the documented C++ example through the
exported CMake target and produced the analytical degree-two kernel value.
These are correctness checks, not biological model-quality claims.

Production configuration/build also succeeded with testing disabled, Python
discovery explicitly disabled, and FetchContent disconnected. This configuration
creates no test dependency directory.

In this workspace, Catch2 discovery initially targeted an unwritable inherited
`XDG_RUNTIME_DIR`. The diagnostic was traced to Catch2's temp-file implementation;
local checks use `env -u XDG_RUNTIME_DIR TMPDIR=/tmp ctest --preset dev` (and the
corresponding sanitizer/coverage preset). No product workaround was added.

## Hosted CI and release gates

The first hosted Windows configure exposed a real toolchain mismatch: CMake
3.31.6 predates Visual Studio 2026 support and selected unavailable `nmake` on the
current runner. CI now pins CMake 4.4.4, which supports that generator. Vendor
newline conversion is disabled so independent byte hashes remain valid under
Windows Git settings. That historical revision also executed a Windows batch
workflow; the first Python package release removes it in favor of portable
installed-wheel tests.

Current-head hosted checks and an independent reviewer verdict are required
before merge. Consult the linked PR for their exact SHA, logs, and final result.

## Cross-platform numeric parsing regression

[Hosted run 39899594757](https://github.com/cttsai1985/string-kernels-libsvm/actions/runs/39899594757)
established additional RED evidence after the build corrections. macOS rejected
the valid subnormal in `dataset labels retain the finite double range and signed
zero`; macOS and Windows both accepted the hexadecimal token in `malformed
dataset records identify the source and physical line`. Linux passed those
same existing assertions.

The correction uses decimal `std::from_chars` with complete-token, range, and
finite-value checks, preserving a single optional leading plus. It removes
standard-stream parsing differences without relaxing the expected values.
Additional process checks cover subnormal/maximal finite labels and malformed
hexadecimal/double-sign inputs. MEASURED corrected local development suite:
39/39 CTest checks passed. The hosted matrix remains the cross-platform gate.


## Dependency bootstrap and layout cleanup (#3)

Tracking issue: [#3](https://github.com/cttsai1985/string-kernels-libsvm/issues/3).
The modernization evidence above records historical paths and remains unchanged.
Current examples are `examples/regression/train_sequences.tsv` and
`examples/regression/test_sequences.tsv`; historical source is
`legacy/paired_sequence_analyzer.cpp`. The active internal option parser is
`CliOptions.h/.cpp`. Independent tests pin the original bytes of both examples
and the historical analyzer. No kernel or solver mathematics changed.

RED: `python3 tests/test_fetch_libsvm.py -v` failed before implementation because
`scripts/fetch_libsvm.py` did not exist. An additional external-response test
raised `http.client.IncompleteRead` through the initial implementation; it passed
after interrupted HTTP responses became actionable bootstrap errors.

MEASURED bootstrap tests: all 19 standard-library tests passed, covering real
byte digests, temporary files, exact upstream pins, complete offline reuse,
partial cache recovery, modified cache rejection, bad hashes, interrupted/network
failures, manifest validation, and CLI diagnostics. HTTP is mocked only at its
external boundary; unit tests make no network requests.

MEASURED CMake preflight tests accept the verified cache and reject missing or
modified sources with the setup command. Production configure/build succeeded
with `BUILD_TESTING=OFF`, Python discovery disabled, and FetchContent disconnected.
Production configuration invokes no dependency fetcher. The formatting check
passed using clang-format 19.1.7.

MEASURED full local verification: development, ASan/UBSan, and GCC coverage each
passed all 40 CTest checks, including the 19-case bootstrap driver, independent
matrix and prediction oracles, actual trainer/predictor workflows, installation,
and dependency/layout pins. Project-only coverage was 260/262 lines (99.2%),
28/28 functions (100%), and 382/508 branches (75.2%); the 98% line gate passed.
The local Catch2 source override and temporary-directory environment remain as
documented above.

MEASURED fresh-source verification used `git archive` of implementation commit
`8ba1ab61efebb14c7982ed6859d451a13eb34c34`, rather than the warm worktree cache.
`git ls-files third_party/libsvm` was empty and the archive contained no dependency
sources. Production configure failed with the explicit bootstrap command. The
actual official download then passed all independent pins, a clean development
build succeeded, and all 40 CTest checks passed. The literal updated README Unix
workflow also built successfully and produced eight training rows, six target
rows, and six finite predictions. Existing integration checks verify numerical
values and solver predictions independently, beyond those artifact counts.

CI checks a clean checkout before bootstrap on Linux, macOS, and Windows; the
source-tracking check and expected configure failure prevent a committed solver
copy or implicit setup from silently returning. Consult PR #4 for exact-head
hosted results and independent review before the dev/master promotion.

## Spectrum and subsequence kernels (issue #6)

Test-first RED: before adding the public enum values and helper APIs, the selected
Catch2 build failed on missing `KernelType::spectrum`, `KernelType::subsequence`,
the decay field, and the new kernel functions. The tests specified the feature
maps independently by counting contiguous k-mers and explicitly enumerating
ordered subsequence embeddings. Existing numeric IDs `0/1`, WDSK default type,
degree `5`, and the SSK decay default `0.5` are pinned.

MEASURED local development gate after implementation: all 49 CTest checks passed,
including the process-level CLI suite, the real LIBSVM workflows, and the
installation test. MEASURED AddressSanitizer/UndefinedBehaviorSanitizer gate:
49/49 passed. MEASURED GCC coverage gate: 49/49 passed; project-only coverage
was 391/398 lines (98.2%), 41/41 functions (100%), and 514/685 branches (75.0%).
The 98% line floor remained unchanged.

MEASURED independent explicit-embedding oracle: 92,256 normalized matrix
cells matched explicit feature enumeration across spectrum and subsequence
`k=1..6`, with subsequence decay `0.5`, `1.0`, and `0.1`; maximum absolute
numeric difference was `3.3306690738754696e-16`. This is formula verification,
not a timing or model-quality benchmark. MEASURED real LIBSVM
classification checks passed exact matrix oracles and recovered the two held-out
labels for spectrum, SSK at the default decay, and SSK with decay `0.25`. The
fixture validates execution and expected behavior only; it does not establish
generalization accuracy.

The new README commands use the same fixture and have been exercised by that
integration test. MEASURED production (`BUILD_TESTING=OFF`) build succeeded and
the literal README spectrum and SSK commands each emitted two correct
predictions (`1`, `-1`). Spectrum is position independent and counts k-mer occurrences;
SSK accepts ordered gapped subsequences and is the more expensive choice. The
docs explicitly state the SSK gap-only feature rescaling, empty-feature behavior,
parameter coupling, and derived `O(k*n*m)` pair-work bound. All speed/accuracy
language remains analytical or descriptive; no performance claim was measured.
The tiny-decay regression checks both `lambda=1e-300` and the smallest positive
representable `double`; a gap-free `AA`, `k=2` diagonal stays `1` in both cases.

## Native Python package and naming migration (2026-11-16)

The owner approved implementation/publication, selected BSD-3-Clause, requested
removal of the planning file and batch workflow, and requested consistent source
filenames. The command is now `string-kernels`; headers are under
`include/stringkernel/` and implementations under `src/`.

RED: independently specified Python feature-map and pipeline tests failed to
import the not-yet-created package. The renamed-command help test then failed
against the original compiled executable. These failures were observed before
implementing the corresponding behavior.

The Python build creates an sdist and compiles its wheel without the LIBSVM
bootstrap. Installed-artifact and cross-platform results are recorded in the
release and issue closeout, including the exact tested commit. The verifier is
`python scripts/verify_package.py --dist build/distributions`.

MEASURED locally: 50/50 CTest cases pass after relocation; 38/38 installed-wheel
Python tests pass. Independent review found an iterator-consumption bug in the
inherited sklearn `fit_transform`; its new regression failed against the first
installed wheel, then passed after fitting once and transforming saved references.
The package's wheel/sdist build does not include the LIBSVM cache. A fresh outside-
checkout install passed the initial 37-test suite; hosted release-wheel verification
also includes the iterator regression. These fixtures prove implementation behavior,
not real-world predictive accuracy.

The hosted macOS wheel jobs exposed an accidental build of CLI dataset/parser
code in the Python-only configuration. A source-boundary regression poisoned
those sources with `#error` and failed before the fix. Python-only CMake now
builds only `src/kernels.cpp` and the binding; the full C++ library/CLI remain
available in their normal configuration. This avoids imposing the CLI parser's
macOS library requirements on mathematical wheels.


## Kernel preparation and performance refactor (2026-11-17)

Issue [#22](https://github.com/cttsai1985/string-kernels-libsvm/issues/22) records
the owner-authorized design. RED: running the performance gate with the baseline
on both sides failed its threefold high-similarity requirement (MEASURED ratio
1.082). Independent raw weighted-degree feature-map tests cover interrupted runs,
unequal lengths and `INT_MAX` degree; matrix tests cover square prediction inputs
that are not symmetric, mixed zero/nonzero diagonals, and SSK self overflow.

MEASURED locally after refactoring: 51/51 CTest cases passed, including actual
LIBSVM regression/classification predictions and installed CLI checks; 45/45
installed-candidate Python tests passed; 2/2 benchmark validation tests passed.
GCC project coverage was 418/425 lines (98.4%), 43/43 functions (100%), and
535/725 branches (73.8%); the 98% line floor is unchanged.

The final same-toolchain benchmark ran without concurrent project builds or tests.
All complete matrices passed parity across five fresh-process trials per backend.
The report retains weak gains, overlapping ranges, a slightly slower CLI case,
and the multithreaded comparator's degree-one win. The initial published-wheel
comparison was diagnostic; the committed timings use a locally rebuilt baseline
and candidate with GCC 13.3.0 and matching Release settings. See
[BENCHMARKS.md](BENCHMARKS.md) and its linked raw data/profile for method and limits.
Independent review found no concrete correctness or benchmark blockers. Hosted
current-commit CI, artifact verification and the merge are linked in the issue
closeout and release. Python-only builds now include `src/kernel_evaluator.cpp`
alongside the kernel functions and binding, and still exclude CLI/parser code.


## SIMD and native CPU workers (2026-11-17)

Issue [#24](https://github.com/cttsai1985/string-kernels-libsvm/issues/24) records
the hardware scope requested by the owner. RED: 35 focused Python execution/
solver cases failed against v1.0.1 because n_jobs/backend were absent; the CLI
execution-control test failed on unknown --threads. Independently specified
native CPU/thread tests first failed on missing interfaces before implementation.

MEASURED local GREEN: 62/62 CTest cases; 62/62 ASan/UBSan; a compiled portable-only
wheel reports scalar and passes all 105 installed Python tests. After adding long
vector-sized feature-map fixtures, the final Python/benchmark suite passes 116
cases (105 installed-package cases plus 11 benchmark checks). These fixtures run
on every wheel platform and cover automatic and portable execution, block tails,
large requested degree, thread counts above row count, rectangular reference
ordering, cloning/persistence and real SVC/SVR predictions.

The first threaded coverage run exposed non-atomic GCC instrumentation counters,
not a numerical failure. Coverage compilation now uses -fprofile-update=atomic;
no parser errors are ignored and the 98% floor is unchanged. The filter now includes private implementation headers, so the SIMD templates
remain inside the gate. MEASURED final local coverage: 585/593 lines (98.7%),
74/74 functions and 631/835 branches. The 64-row cap has an independent pin test.
Final hosted current-commit results remain
in the issue/release closeout.

Independent review caught an installed-C++ consumer failure: exported
Threads::Threads was unresolved by the documented include-only CMake example.
Its new real consumer test failed to configure before adding find_package(Threads
REQUIRED), then compiled and executed the installed threaded library successfully.
RAII owns cancellation/joining, including partial worker launch failure; no worker
can retain call-scoped inputs after return. GCC TSan could not start on this WSL
host (unexpected memory mapping, with PIE and non-PIE), so no TSan pass is claimed.

MEASURED independent short CPU checks: 4,800 byte-oracle comparisons passed in
AVX2, 4,800 in a separately compiled portable backend, and 4,800 under ASan/UBSan.
AVX2 flags are source-local; runtime CPU/OS-state checks protect older machines.
The actual benchmark extension contains vector compare/movemask/popcount
instructions, and its raw hash matches the measured artifact. Hosted ARM/Windows
checks validate their compiled paths before merge.

Five-trial hardware runs compare v1.0.1, portable, SIMD-only, and two-/four-thread
modes on twelve regimes, including complete CLI costs. All matrices matched.
Additional CLI batching and mirror-layout experiments retain losses and noisy
ranges. The latter had mixed results, so production keeps single-pass mirrored
ownership. See [OPTIMIZATION.md](OPTIMIZATION.md) and its raw data/disassembly.
