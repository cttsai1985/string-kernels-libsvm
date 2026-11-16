# License status and distribution requirements

## Project code

The repository has no declared license for project-owned code. Do not label the
whole repository BSD, MIT, or public domain on the strength of a dependency's
license. A public GitHub repository is not itself a general reuse license;
[GitHub's license guidance](https://choosealicense.com/no-permission/) explains
that distinction.

Selecting project terms requires authorization from the relevant rights holder(s).
A permissive license suitable for scientific Python distribution is proposed in
the implementation plan, but it is not adopted by this document. Academic
citation gives attribution; it does not supply software redistribution rights.

## Dependency terms

| Component | License | Distribution obligation |
| --- | --- | --- |
| LIBSVM 3.37 | BSD-3-Clause | Preserve the complete upstream copyright, conditions, and disclaimer in source distributions and in documentation/materials accompanying binaries; do not imply endorsement. |
| Catch2 v3.16.1 | BSL-1.0 | Retain its notice/license when distributing its source. It is used only for tests and is not linked into the installed library or tools. |

The authoritative LIBSVM terms are its pinned
[COPYRIGHT](https://github.com/cjlin1/libsvm/blob/6b907139084abf2da4d6d3cb10dc3b7eaffa2fbb/COPYRIGHT).
The existing bootstrap obtains that full file; CMake installs it beside
THIRD_PARTY_NOTICES.md. The BSD label summarizes those terms and does not replace
the actual notice. See [dependency provenance](../THIRD_PARTY_NOTICES.md).

## Future packages

A binary package needs explicit project licensing and all applicable bundled
third-party notices. A Python extension computing kernels need not bundle LIBSVM:
scikit-learn supplies its own solver. A package that ships the upstream CLI tools
must also carry LIBSVM's complete notice. Future binding dependencies need their
own provenance and notice review when introduced.

Do not advertise PyPI availability, a supported wheel platform, or a release
license until the corresponding artifact and terms have been verified.
