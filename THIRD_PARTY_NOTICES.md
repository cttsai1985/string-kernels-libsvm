# Third-party notices

## LIBSVM

The build includes unmodified source from
[LIBSVM 3.37, tag `v337`](https://github.com/cjlin1/libsvm/tree/v337), commit
`6b907139084abf2da4d6d3cb10dc3b7eaffa2fbb`.

Copyright (c) 2000-2023 Chih-Chung Chang and Chih-Jen Lin. All rights reserved.

The full redistribution conditions and disclaimer are in
[the pinned upstream COPYRIGHT](https://github.com/cjlin1/libsvm/blob/6b907139084abf2da4d6d3cb10dc3b7eaffa2fbb/COPYRIGHT).
The dependency bootstrap fetches that complete file into ignored
`third_party/libsvm/COPYRIGHT`; installation copies it beside this notice.
The exact files, source URLs, and SHA-256 digests are recorded in
[`cmake/libsvm.json`](cmake/libsvm.json).
In an installation, the complete `COPYRIGHT` file is beside this notice.

LIBSVM uses BSD-3-Clause terms. That dependency license does not license
project-owned code. See [distribution requirements](https://github.com/cttsai1985/string-kernels-libsvm/blob/main/docs/LICENSING.md).

## Catch2

Tests use [Catch2 v3.16.1](https://github.com/catchorg/Catch2/tree/v3.16.1), licensed
under [BSL-1.0](https://github.com/catchorg/Catch2/blob/v3.16.1/LICENSE.txt).
Catch2 is downloaded only when tests are enabled and no local source override is
provided. It is not linked into the converter or installed project library.

## Project license

Project-owned code is licensed under BSD-3-Clause; see LICENSE. The owner selected
these terms on 2026-11-16. Dependency licenses remain separate.

## pybind11

The Python extension is built with pybind11 3.1.0, a header-only binding library
under BSD-3-Clause. Its full notice follows. The wheel contains the kernel
extension, not the LIBSVM CLI tools or Catch2. NumPy and optional scikit-learn are
installed separately with their own distribution notices.

Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>, All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Please also refer to the file .github/CONTRIBUTING.md, which clarifies licensing of
external contributions to this project including patches, pull requests, etc.
