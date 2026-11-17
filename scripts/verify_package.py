#!/usr/bin/env python3
"""Inspect release archives and exercise an installed wheel outside the checkout."""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile


def inspect_archives(directory: Path) -> Path:
    wheels = sorted(directory.glob('*.whl'))
    if len(wheels) != 1:
        raise ValueError(f'expected exactly one local wheel, found {len(wheels)}')
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        if not any('/licenses/LICENSE' in name for name in names):
            raise ValueError('wheel lacks the project license')
        notices = archive.read('stringkernels/THIRD_PARTY_NOTICES.md').decode()
        if 'Wenzel Jakob' not in notices or 'THIS SOFTWARE IS PROVIDED' not in notices:
            raise ValueError('wheel lacks the complete binding notice')
        if not any(name.startswith('stringkernels/_native.') for name in names):
            raise ValueError('wheel lacks its compiled kernel extension')
        if any(name.endswith(('.a', '.lib', '.cpp', '.bat')) for name in names):
            raise ValueError('wheel contains developer-only artifacts')
    sdists = sorted(directory.glob('*.tar.gz'))
    if len(sdists) != 1:
        raise ValueError('expected exactly one source distribution')
    with tarfile.open(sdists[0]) as archive:
        names = archive.getnames()
        relative = {name.split('/', 1)[1] for name in names if '/' in name}
        required = {'LICENSE', 'pyproject.toml', 'CMakeLists.txt', 'src/kernels.cpp',
                    'src/kernel_evaluator.cpp', 'src/kernel_evaluator.hpp',
                    'src/cpu_kernels.cpp', 'src/cpu_kernels.hpp', 'src/cpu_kernels_avx2.cpp',
                    'src/cpu_kernels_neon.cpp', 'src/execution.cpp', 'src/parallel_rows.cpp',
                    'src/parallel_rows.hpp', 'include/stringkernel/execution.hpp', 'cmake/Cpu.cmake',
                    'include/stringkernel/kernels.hpp', 'python/bindings.cpp'}
        if not required <= relative:
            raise ValueError(f'sdist lacks {required - relative}')
        if {'sample.bat', 'implementation_plan.md'} & relative:
            raise ValueError('sdist contains removed workflow files')
        if any('/third_party/libsvm/' in name or '/build/' in name for name in names):
            raise ValueError('sdist includes cached dependencies or build artifacts')
    return wheels[0]


def verify_kernel_only_build(directory: Path, destination: Path) -> None:
    """Prove CLI parsing/filesystem code is outside the wheel's build boundary."""
    with tarfile.open(next(directory.glob('*.tar.gz'))) as archive:
        # Only materialize regular files/directories from our inspected sdist.
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError('source archive contains an unsafe path')
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as stream:
                    target.write_bytes(stream.read())
            else:
                raise ValueError('source archive contains a link or special file')
    source = next(path for path in destination.iterdir() if path.is_dir())
    for name in ('dataset.cpp', 'libsvm_writer.cpp', 'cli.cpp', 'cli_options.cpp'):
        (source / 'src' / name).write_text('#error CLI code must not enter a Python build\n')
    (source / 'cmake/Libsvm.cmake').write_text('message(FATAL_ERROR "LIBSVM must not enter a Python build")\n')
    subprocess.run([sys.executable, '-m', 'build', str(source), '--wheel',
                    '--outdir', str(destination / 'kernel-only-wheel')], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dist', type=Path, required=True)
    parser.add_argument('--tests', type=Path, default=Path(__file__).resolve().parents[1] / 'tests')
    args = parser.parse_args()
    wheel = inspect_archives(args.dist.resolve())
    tests = args.tests.resolve()
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    with tempfile.TemporaryDirectory(prefix='string-kernels-installed-') as temporary:
        root = Path(temporary)
        verify_kernel_only_build(args.dist.resolve(), root / 'source-boundary')
        venv.EnvBuilder(with_pip=True).create(root / 'environment')
        executable = root / 'environment' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        subprocess.run([str(executable), '-m', 'pip', 'install', '--no-cache-dir',
                        f'{wheel}[test]'], check=True, cwd=root, env=environment)
        subprocess.run([str(executable), '-m', 'pytest', '-q',
                        *[str(tests / name) for name in
                          ('test_python_kernels.py', 'test_sklearn.py', 'test_package.py')]],
                       check=True, cwd=root, env=environment)
        subprocess.run([str(executable), '-c',
                        'import stringkernels; from stringkernels import _native; '
                        'print("Verified installed native artifact:", _native.__file__)'],
                       check=True, cwd=root, env=environment)
    print('Verified wheel and sdist:', wheel.name, flush=True)


if __name__ == '__main__':
    main()
