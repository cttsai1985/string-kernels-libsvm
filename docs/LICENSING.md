# Licensing and distribution

## Project code

Project-owned code is licensed under [BSD-3-Clause](../LICENSE), selected by the
owner on 2026-11-16. The license permits source and binary redistribution subject
to retention of its notice, conditions, and disclaimer, and prohibits using the
copyright holders' or contributors' names for endorsement without permission.
The complete LICENSE text is authoritative. Academic citations attribute the
published methods and are separate from software licensing.

## Dependencies

| Component | License | How it is distributed here |
| --- | --- | --- |
| LIBSVM 3.37 | BSD-3-Clause | Unmodified source is bootstrapped for the optional C++ CLI build; CMake installs its complete COPYRIGHT with the tools. It is not bundled in Python wheels. |
| Catch2 v3.16.1 | BSL-1.0 | Test-only; not linked into installed project libraries or wheels. Retain its license if distributing its source. |
| pybind11 3.1.0 | BSD-3-Clause | Header-only binding code is compiled into the Python extension. Its full notice ships in each wheel. |
| NumPy; optional scikit-learn | Their own distribution terms | Resolved as separate runtime dependencies, not bundled into this project's wheel. |
| scikit-build-core, CMake, Ninja, cibuildwheel | Their own distribution terms | Build/release tools installed separately; not bundled into the Python package. |

The pinned LIBSVM
[COPYRIGHT](https://github.com/cjlin1/libsvm/blob/6b907139084abf2da4d6d3cb10dc3b7eaffa2fbb/COPYRIGHT)
is obtained by the source bootstrap and installed beside THIRD_PARTY_NOTICES.md.
The project license does not replace any upstream terms. See
[third-party notices](../THIRD_PARTY_NOTICES.md) for provenance and the binding notice.

The release-verification script checks the project license, binding notice, and
native extension inside the actual wheel. The C++ installation tests verify the
upstream LIBSVM notice. Recheck these contents when changing dependencies or
adding another bundled component.
