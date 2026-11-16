# README, distribution, and repository refresh

## Goal and decisions
Explain the science before the tooling; provide a compiler-free Python installation
and use the existing kernels with scikit-learn without modifying either solver.
The request supersedes the previous master-branch delivery convention: main will
be the default branch. Naming review is tracked separately in issue #16.

## Stage A: documentation and repository housekeeping (low risk)
Scope: README.md, docs/QUICKSTART.md, docs/KERNELS.md, REFERENCES.bib,
docs/LICENSING.md, THIRD_PARTY_NOTICES.md, implementation_plan.md,
.github/workflows/ci.yml, AGENTS.md, CONTRIBUTING.md.

Move existing build/run recipes intact into QUICKSTART. Explain normalized feature
maps, supported methods, limits, and references in the README; retain the flowchart.
Document actual license status and upstream binary redistribution requirements.
Do not invent a project license or advertise an unavailable package.
Merge through a PR after current-commit CI, then rename master to main and update
CI and contributor instructions. Show obsolete branches before deletion, verify
their tips are ancestors of the default branch, and remove only verified copies.

## Stage B: compiled Python package (high risk; approval required)
Standard practice: pybind11 + scikit-build-core for the existing C++17 math,
NumPy matrices, and scikit-learn's public precomputed-kernel interface.
We differ only in computing sequence similarities here, leaving sklearn's solver
and private ABI untouched. LIBLINEAR does not accept precomputed kernels.

Exact proposed scope (in addition to the Stage A files if documentation changes):
- CMakeLists.txt; cmake/Python.cmake (new)
- pyproject.toml (new)
- python/bindings.cpp (new)
- python/stringkernels/__init__.py (new)
- python/stringkernels/sklearn.py (new)
- tests/test_python_kernels.py (new)
- tests/test_sklearn.py (new)
- tests/test_package.py (new)
- .github/workflows/wheels.yml (new)
- docs/PYTHON.md (new)
- docs/DESIGN.md; docs/COMPATIBILITY.md; docs/DEVELOPMENT.md; docs/VALIDATION.md
- LICENSE (only after the rights holder explicitly selects/authorizes a license)

Distribution name `string-kernels-libsvm` and import `stringkernels` are proposals,
subject to collision checks and issue #16. Expose pairwise(X, Y=None, kernel=...,
degree=5, decay=0.5) returning float64 arrays in target/reference order. Encode
Python str as UTF-8 bytes explicitly; preserve case and reject empty/whitespace
records at the input boundary. Pin all existing defaults and byte semantics.

Provide a cloneable sklearn transformer with fit(X) retaining reference order
and transform(Y) producing Y-by-X matrices; demonstrate a Pipeline ending in
SVC(kernel='precomputed') and GridSearchCV without leaking validation references.
Use the existing C++ functions, not a Python rewrite. Pure matrix computation
must not fetch or compile LIBSVM; make solver CLI builds optional while preserving
existing default CMake behavior and offline production builds.

Use TDD: fail independent hand-computed matrix tests first, then add bindings;
exercise all four kernels, invalid inputs, zero feature maps, rectangular outputs,
normalization and byte semantics. Compare installed-wheel results to C++/CLI
oracles and verify held-out SVC labels, SVR predictions, cloning, pipeline tuning,
and fold reference ownership. Build wheel and sdist, install outside the source
tree into a clean environment, inspect bundled notices, and run the tests there.

Build and test CPython wheels on Linux, macOS, and Windows with cibuildwheel,
initially CPython 3.10–3.13 on x86_64 and macOS arm64; publish supported tags only
after those jobs pass. Pin build/test dependencies independently. Upload verified
artifacts to GitHub Releases; do not claim PyPI availability until a real upload
and clean installation succeed. No new account or paid service is needed.

## License decision
The history lists the owner's identities and no project LICENSE. That does not
establish permission to choose a license for every contribution. Recommend
BSD-3-Clause for project code to align with the scientific Python ecosystem,
subject to explicit rights-holder authorization. Preserve separate LIBSVM,
Catch2, pybind11, and build dependency terms. No publication of newly packaged
binaries as licensed open-source releases until project terms are settled.

## Risks and acceptance
Dense matrices require O(n*m) storage; SSK additionally has O(k*L1*L2) pair work.
Document resource costs and avoid silently allocating huge intermediary matrices.
Do not promise cross-solver bitwise predictions or ABI compatibility. Existing
CLI names and math remain intact pending the separate naming migration design.
Done means changes merged, all current-commit checks passing, tested wheel assets
visible at their destination, issues closed with evidence, and worktree removed.
