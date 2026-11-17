# Python and scikit-learn

## Install a compiled package

Download a wheel matching your Python version and platform from
[the v1.0.1 release](https://github.com/cttsai1985/string-kernels-libsvm/releases/tag/v1.0.1).
Alternatively, let pip select the matching release wheel and install the optional
scikit-learn dependency in one command:

```sh
python -m pip install --only-binary=string-kernels-libsvm --find-links https://github.com/cttsai1985/string-kernels-libsvm/releases/expanded_assets/v1.0.1 "string-kernels-libsvm[sklearn]==1.0.1"
```

This is a GitHub Release installation, not a claim of PyPI publication. NumPy and
scikit-learn dependencies are obtained from the configured Python package index.
The command fails instead of compiling this package when no matching wheel is
available. Use a virtual environment for your application.

The release targets CPython 3.10–3.13 on Linux x86_64, Windows AMD64, and macOS
Intel and Apple Silicon. Check the actual wheel tags in the release for platform
minimums. Other Python implementations, free-threaded builds, and other
architectures are not included in this release.

## Train and predict from sequences

```python
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from stringkernels.sklearn import StringKernel

model = make_pipeline(StringKernel(kernel="spectrum", degree=2), SVC(kernel="precomputed"))
model.fit(["ACG", "ACT", "GGA", "GGT"], [1, 1, -1, -1])
print(model.predict(["ACC", "GGC"]))  # [1, -1]
```

Use `SVR(kernel="precomputed")` for continuous targets. Scikit-learn owns its
solver; this extension supplies normalized C++ sequence similarities. It does
not link to scikit-learn's private LIBSVM ABI. `LinearSVC` uses LIBLINEAR and
cannot consume a precomputed kernel directly.

The example is a correctness fixture, not a measured generalization result.
Choose your kernel and evaluation split to suit the scientific problem. Cite
the chosen kernel method, LIBSVM, and scikit-learn as described in the README.

## Matrix API

`stringkernels.pairwise(X, Y=None, *, kernel="weighted-degree", degree=5, decay=0.5)`
returns a C-contiguous `float64` NumPy array. Rows follow `X`; columns follow `Y`.
With `Y=None`, `X` supplies both axes. There are no label or serial columns.

| Parameter | Meaning |
| --- | --- |
| `X`, `Y` | Nonempty one-dimensional collections of nonempty Python strings without ASCII whitespace. Iterables are copied before native computation. |
| `kernel` | `positional`, `weighted-degree`, `spectrum`, or `subsequence`. |
| `degree` | Positive integer representable as a C++ `int`; default `5`. Exact pattern length for spectrum/SSK, maximum substring order for weighted degree. |
| `decay` | Default `0.5`; configurable only for subsequence, finite in `(0,1]`. |

Python strings are encoded as UTF-8 and compared as **bytes**, preserving the C++
contract. Case and Unicode normalization remain unchanged. Positional matching
requires equal byte lengths across both axes. A degree longer than a sequence
produces empty spectrum/SSK features and normalized zeros. See
[the mathematical reference](KERNELS.md) for exact definitions and costs.

A dense matrix uses `8 * len(X) * len(Y)` bytes for its numeric values, plus input
and native working storage. The extension releases the Python GIL during kernel
computation. This does not make the computation parallel. For large prediction
sets, process chunks of target rows against the same complete reference order.
SSK also needs working memory proportional to the product of sequence lengths.

## Cross-validation and persistence

`StringKernel.fit(X, y=None)` copies the reference strings and validates the
configuration without allocating a training matrix. `transform(Y)` computes
new-to-reference similarities. A refit replaces the references; a failed refit
leaves the estimator unfitted. The transformer supports `get_params`, `set_params`,
`clone`, and pickle, and can be placed before precomputed SVC/SVR in a Pipeline.
After changing parameters on a fitted pipeline, refit the pipeline before predicting.

Keep the transformer **inside** the Pipeline passed to `GridSearchCV`:

```python
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(model, {
    "stringkernel__degree": [1, 2],
    "svc__C": [1.0, 3.0],
}, cv=2)
```

Each fold then fits its own reference strings. Do not fit the transformer on the
entire dataset before cross-validation. The tests inspect fold reference counts
and verify held-out classification and regression against the public solver API.

## Build and verify as a contributor

A source installation requires a C++17 compiler and the pinned build dependencies
in `pyproject.toml`; pip's build isolation supplies the Python build tooling.
The Python build does not fetch or compile LIBSVM and needs no shell-specific
bootstrap script. It builds only the native kernel extension, with C++ tests off.

```sh
python -m pip install build==1.6.1
python -m build --outdir build/distributions
python scripts/verify_package.py --dist build/distributions
```

`build` creates an sdist and builds a wheel from it. The verification script
inspects licenses and archive contents, proves the wheel still builds with
deliberately unbuildable CLI sources, creates a fresh environment outside the
source tree, installs the wheel, and runs the numerical, scikit-learn, and
installed-metadata tests. CI repeats installed-wheel tests for every supported
Python/platform combination. Raw C++ development still uses the
[development workflow](DEVELOPMENT.md).

## License

Project code is BSD-3-Clause. Wheels carry LICENSE and the full pybind11 binding
notice. They do not bundle LIBSVM, Catch2, NumPy, or scikit-learn. Optional runtime
dependencies retain the terms in their own distributions. See
[licensing](LICENSING.md) and [third-party notices](../THIRD_PARTY_NOTICES.md).
