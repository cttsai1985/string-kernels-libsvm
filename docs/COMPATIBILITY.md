# Compatibility reference

## Repository identity

The repository is named `string-kernels-libsvm`; its documentation title is
**String Kernels for LIBSVM**. This names the supported string kernels and their
LIBSVM precomputed-matrix integration.

The first packaged release adopts the installed command `string-kernels` and
namespaced public headers under `include/stringkernel/`. The former mixed-case
executable and flat header names are intentionally replaced; update scripts and
C++ includes to the names below. Kernel mathematics, option meanings, and
reference-column ordering remain unchanged. The historical CMake project ID and
`stringkernel::stringkernel` target remain stable.

## Supported command line

```text
string-kernels -train FILE [-test FILE] [-k K] [-t TYPE | --kernel NAME] [--decay LAMBDA] [--threads N] [--backend auto|scalar]
string-kernels --help
string-kernels --version
```

| Option | Contract |
| --- | --- |
| `-train FILE` | Required training reference dataset. |
| `-test FILE` | Optional target dataset. Without it, targets are the training records. |
| `-k K` | Positive integer in the platform's `int` range. Default `5`; means WDSK maximum order, spectrum k-mer length, or SSK subsequence length. |
| `-t TYPE` | Stable IDs: `0` positional, `1` weighted degree (default), `2` spectrum, `3` subsequence. |
| `--kernel NAME` | `positional`, `weighted-degree`, `spectrum`, or `subsequence`; cannot be combined with `-t`. |
| `--decay LAMBDA` | SSK only; finite in `(0, 1]`, default `0.5`. |
| `--threads N` | Positive native worker count in the platform int range; default `1`. |
| `--backend auto\|scalar` | Runtime CPU dispatch (default) or portable execution. |
| `--help` | Standalone usage command. |
| `--version` | Standalone project and bundled LIBSVM version command. |

Unknown options, duplicate options, mixed kernel selectors, missing values,
malformed parameters, invalid degrees/decay, and unsupported kernel types are errors. Successful conversion writes
only matrix data to stdout. Diagnostics go to stderr, with a nonzero exit status.
Paths containing spaces are supported when quoted by the calling shell.

The version string is `string-kernels 1.1.0 (LIBSVM 3.37)`. Version `1.0.0`
is the modernization's first explicit project version. It does not imply
historical release continuity.

## Input and output changes

| Behavior | Modernized contract |
| --- | --- |
| Sequence records | Exactly one sequence token and one finite double label per nonblank line. |
| Whitespace | Spaces and tabs delimit fields. Blank lines are ignored. LF and CRLF are accepted. |
| Sequences | Case-sensitive, nonwhitespace byte strings with no imposed amino-acid alphabet. |
| Invalid records | Rejected with a source and one-based line number instead of silently skipped. |
| Empty or unreadable files | Rejected with a diagnostic and failing exit status. |
| Long records | Read without a fixed-size line buffer. |
| Numeric formatting | Classic-locale decimal points and round-trip double precision. |
| DSK unequal lengths | Rejected across all reference and target records before matrix output. |
| WDSK unequal lengths | Shared-position comparisons with full-length self-kernel normalization are preserved. |

The old six-decimal output is not a byte compatibility target. The modernized
output preserves finite double labels and kernel values at higher precision.
The mathematical WDSK weights, default WDSK type and degree, numeric IDs `0`
and `1`, and training reference column order remain compatibility targets. New
types are appended without renumbering these IDs.

The legacy DSK calculation divided by the reference length and could produce an
asymmetric training matrix for unequal lengths. Equal-length validation removes
that ambiguous behavior. DSK still means positional similarity.

## C++ source compatibility

The old pointer-owning API, global dataset state, mutable self-kernel caches, and
unscoped names are replaced. This is an intentional source API change. There is
no adapter for old constructors or internal fields.

Supported headers are `stringkernel/kernels.hpp`, `stringkernel/dataset.hpp`, and `stringkernel/libsvm_writer.hpp`, and `stringkernel/execution.hpp`.
Their declarations live in `namespace stringkernel`. `Dataset` is a vector of
value-owned `Sample` records. Each calculation receives its kernel configuration
explicitly. See the [library example](DEVELOPMENT.md#use-the-c-library).

Library contract violations throw `std::invalid_argument`. Malformed or unreadable
input and output failures throw `std::runtime_error`. The converter catches
standard exceptions and reports them through stderr and its exit status.

## Historical analyzer

`legacy/paired_sequence_analyzer.cpp` remains byte-for-byte historical source and is outside all
supported build targets. It depends on the missing `OrthogonalArray.h`, a paired
sequence data model absent from the supplied header, incompatible constructors,
and private fields from a customized LIBSVM integration.

Restoration requires a separate design for those dependencies, data contracts,
fold ownership, and supported public solver APIs. Adding the current `svm.h`
alone does not restore the analyzer. The dormant RBF-over-similarity path is also
outside the supported converter and library API.

Python bindings and the scikit-learn transformer are described in [the Python
guide](PYTHON.md). FASTA parsing, automatic alignment,
and automatic model tuning remain outside the package.


## Hardware execution in 1.1.0

Mathematical parameters remain in `KernelConfig`; `ExecutionConfig` supplies
thread count and automatic/scalar CPU backend separately. `write_precomputed`
accepts it as an optional final argument. Existing source calls remain valid;
recompile C++ clients against the updated static library. Installed CMake clients
must resolve `Threads` before including the exported targets, as shown in the
[development guide](DEVELOPMENT.md#use-the-c-library).

Python adds positive `n_jobs` and `backend='auto'|'scalar'` to `pairwise` and
`StringKernel`; cloning and persistence retain these constructor parameters.
Defaults select one native thread and a runtime-safe CPU implementation. Order,
shape, normalization, zero-feature handling and overflow errors are preserved.
Floating-point summation order may change within the existing numerical
precision contract; byte-for-byte decimal output is not guaranteed across
versions or compilers. CLI threading retains ordered streaming through row blocks capped at 64 rows. See [the optimization guide](OPTIMIZATION.md).
