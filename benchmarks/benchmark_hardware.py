#!/usr/bin/env python3
"""Separate portable, SIMD, and multicore kernel costs with complete-output parity."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time

# Keep the previous published benchmark unchanged; share its deterministic inputs,
# affinity selection, clock accounting, and complete-matrix correctness checks.
_spec = importlib.util.spec_from_file_location(
    '_kernel_benchmark_common', Path(__file__).with_name('benchmark_kernels.py'))
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)
dataset = _common.dataset
validate_matrices = _common.validate_matrices

CASES = {
    **_common.CASES,
    'large_protein_training': dict(kernel='weighted-degree', rows=512, columns=None,
                                   length=256, degree=5, alphabet='ACDEFGHIKLMNPQRSTVWY'),
    'large_protein_prediction': dict(kernel='weighted-degree', rows=384, columns=768,
                                     length=256, degree=5, alphabet='ACDEFGHIKLMNPQRSTVWY'),
    'large_high_similarity': dict(kernel='weighted-degree', rows=256, columns=None,
                                  length=256, degree=64, alphabet='ACGT', similar=True),
    'large_degree_one': dict(kernel='weighted-degree', rows=512, columns=None,
                             length=256, degree=1, alphabet='ACGT'),
    'positional_training': dict(kernel='positional', rows=512, columns=None,
                                length=256, degree=1, alphabet='ACGT'),
    'subsequence_training': dict(kernel='subsequence', rows=16, columns=None,
                                 length=24, degree=3, alphabet='ACGT'),
}


def variants(baseline_cli, candidate_cli, available_cpus):
    modes = [('baseline', 'baseline', 1), ('scalar1', 'scalar', 1), ('auto1', 'auto', 1)]
    modes += [(f'auto{n}', 'auto', n) for n in (2, 4) if available_cpus >= n]
    result = [dict(label=label, backend=backend, threads=n, converter=None)
              for label, backend, n in modes]
    if baseline_cli and candidate_cli:
        result += [dict(label=f'{label}_cli', backend=backend, threads=n,
                        converter=baseline_cli if backend == 'baseline' else candidate_cli)
                   for label, backend, n in modes]
    return result


def cpu_metadata(requested, reported):
    # Readback describes the preferred CPU backend. Individual kernels/degrees
    # can retain scalar code, so this is not a per-call SIMD execution counter.
    return dict(requested_backend=requested, reported_cpu_backend=reported,
                simd_available=None if reported is None else reported in ('avx2', 'neon'),
                simd_dispatch_allowed=(None if reported is None and requested == 'auto' else
                              requested == 'auto' and reported in ('avx2', 'neon')),
                dispatch_scope='preferred backend; per-kernel/degree fallbacks may apply')


def check_speedups(results, requirements):
    for requirement in requirements:
        selector, floor = requirement.split('=')
        name, modes = selector.split(':')
        numerator, denominator = modes.split('/')
        if not math.isfinite(float(floor)) or float(floor) <= 0:
            raise ValueError('speedup requirements must be finite and positive')
        case = next(c for c in results['cases'] if c['name'] == name)
        ratio = (case['results'][numerator]['median_seconds'] /
                 case['results'][denominator]['median_seconds'])
        if ratio < float(floor):
            raise RuntimeError(f'{selector}: measured speedup {ratio:.3f} below required {floor}')


def parse_cli_matrix(path, shape):
    import numpy as np
    rows = []
    for serial, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split()
        assert len(fields) == shape[1] + 2
        assert fields[1] == f'0:{serial}'
        entries = []
        for column, token in enumerate(fields[2:], 1):
            index, value = token.split(':')
            assert int(index) == column
            entries.append(float(value))
        rows.append(entries)
    return np.asarray(rows, dtype=np.float64)


def worker(args):
    if args.package_root:
        sys.path.insert(0, str(args.package_root.resolve()))
    import numpy as np
    case = CASES[args.case]
    x, y = dataset(case, args.seed)
    config = dict(kernel=case['kernel'], degree=case['degree'])
    data_hash = hashlib.sha256(('\0'.join([*x, '|', *y])).encode()).hexdigest()
    is_cli = args.converter is not None
    if is_cli:
        directory = args.matrix.parent
        train, target, output = (directory / f'{args.case}-{name}'
                                 for name in ('train.txt', 'test.txt', 'matrix.txt'))
        train.write_text(''.join(f'{s} 0\n' for s in y))
        target.write_text(''.join(f'{s} 0\n' for s in x))
        command = [str(args.converter), '-train', str(train), '--kernel', case['kernel'],
                   '-k', str(case['degree'])]
        if y is not x:
            command += ['-test', str(target)]
        if args.backend != 'baseline':
            command += ['--threads', str(args.threads), '--backend', args.backend]
        def compute():
            with output.open('wb') as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.PIPE,
                               check=True, timeout=300)
        version = subprocess.check_output([str(args.converter), '--version'], text=True).strip()
        if args.backend == 'baseline' and not version.startswith('string-kernels 1.0.1 '):
            raise ValueError(f'expected immutable v1.0.1 CLI baseline, found {version}')
        origin, reported = args.converter.resolve(), None
    else:
        import stringkernels
        from stringkernels import _native
        if args.backend != 'baseline':
            config.update(n_jobs=args.threads, backend=args.backend)
        compute = lambda: stringkernels.pairwise(x, None if y is x else y, **config)
        version = stringkernels.__version__
        if args.backend == 'baseline' and version != '1.0.1':
            raise ValueError(f'expected immutable v1.0.1 baseline, found {version}')
        origin = Path(_native.__file__).resolve()
        reported = _native.cpu_backend() if args.backend != 'baseline' else None
    started = time.perf_counter()
    matrix = compute()
    warmup = time.perf_counter() - started
    loops = 1
    while True:
        cpu_start, wall_start = _common.cpu_seconds(), time.perf_counter()
        for _ in range(loops):
            matrix = compute()
        elapsed = time.perf_counter() - wall_start
        cpu = _common.cpu_seconds() - cpu_start
        if elapsed >= args.seconds or loops >= 100000:
            break
        loops = min(100000, max(loops + 1, math.ceil(loops * args.seconds / elapsed)))
    if is_cli:
        matrix = parse_cli_matrix(output, (len(x), len(y)))
    assert matrix.shape == (len(x), len(y))
    assert matrix.dtype == np.float64 and np.isfinite(matrix).all()
    if x is y:
        np.testing.assert_allclose(np.diag(matrix), 1, rtol=1e-12, atol=1e-12)
    np.save(args.matrix, matrix)
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
    except ImportError:
        pools = []
    child_cpu_available = not is_cli or importlib.util.find_spec('resource') is not None
    print(json.dumps(dict(seconds=elapsed / loops,
                          cpu_seconds=cpu / loops if child_cpu_available else None,
                          cpu_to_wall=cpu / elapsed if child_cpu_available else None,
                          loops=loops, block_wall_seconds=elapsed, warmup_seconds=warmup,
                          version=version, origin=str(origin), threads=args.threads,
                          binary_sha256=hashlib.sha256(origin.read_bytes()).hexdigest(),
                          data_sha256=data_hash, numpy=np.__version__, pools=pools,
                          **cpu_metadata(args.backend, reported))))


def run(args):
    import numpy as np
    cpus = _common.affinity(args.affinity_width)
    available = len(cpus) if cpus is not None else (os.cpu_count() or 1)
    modes = variants(args.baseline_cli, args.candidate_cli, available)
    results = dict(seed=args.seed, trials=args.trials, minimum_trial_seconds=args.seconds,
                   environment=dict(python=sys.version, platform=platform.platform(),
                                    processor=_common.cpu_model(), affinity=cpus,
                                    cpu_count=os.cpu_count(),
                                    scheduling_basis='distinct physical cores' if cpus is not None
                                    else 'logical CPU count; physical topology unverified',
                                    load_average=os.getloadavg() if hasattr(os, 'getloadavg') else None),
                   provenance=json.loads(args.provenance.read_text()) if args.provenance else {},
                   timing_boundary='complete Python matrix call; CLI includes startup/input/text output',
                   cases=[])
    rng = random.Random(args.seed)
    with tempfile.TemporaryDirectory(prefix='string-kernel-hardware-') as directory:
        root = Path(directory)
        for name in args.cases:
            runs = {mode['label']: [] for mode in modes}
            expected = None
            for trial in range(args.trials):
                order = modes.copy()
                rng.shuffle(order)
                matrices = {}
                for mode in order:
                    label = mode['label']
                    matrix = root / f'{name}-{label}.npy'
                    command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                               '--case', name, '--backend', mode['backend'], '--threads', str(mode['threads']),
                               '--seed', str(args.seed), '--seconds', str(args.seconds), '--matrix', str(matrix)]
                    if mode['converter']:
                        command += ['--converter', str(mode['converter'].resolve())]
                    else:
                        package = args.baseline_root if mode['backend'] == 'baseline' else args.candidate_root
                        command += ['--package-root', str(package.resolve())]
                    environment = os.environ.copy()
                    environment.pop('PYTHONPATH', None)
                    environment.update(OMP_NUM_THREADS='1', OMP_DYNAMIC='FALSE',
                                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                                       VECLIB_MAXIMUM_THREADS='1', BLIS_NUM_THREADS='1')
                    process = subprocess.run(command, env=environment, capture_output=True,
                                             text=True, timeout=300)
                    if process.returncode:
                        raise RuntimeError(f'{label}/{name} worker failed: {process.stderr}')
                    runs[label].append(json.loads(process.stdout))
                    matrices[label] = np.load(matrix)
                if expected is None:
                    expected = matrices['baseline']
                assert len({values[-1]['data_sha256'] for values in runs.values()}) == 1
                validate_matrices(matrices, expected)
            summaries = {}
            for label, values in runs.items():
                timings = [v['seconds'] for v in values]
                summaries[label] = dict(median_seconds=statistics.median(timings),
                                        minimum_seconds=min(timings), maximum_seconds=max(timings),
                                        runs=values, full_matrix_parity=True)
            kernel = CASES[name]['kernel']
            notes = []
            if kernel == 'weighted-degree' and CASES[name]['degree'] > 32:
                notes.append('Degree >32 retains scalar pair evaluation; compare thread scaling here.')
            elif kernel not in ('weighted-degree', 'positional'):
                notes.append('This kernel has no explicit SIMD pair evaluator; native threading still applies.')
            elif CASES[name]['length'] < 32:
                notes.append('Short sequences use scalar tails; preferred SIMD availability does not imply vector work.')
            results['cases'].append(dict(name=name, parameters=CASES[name], results=summaries, notes=notes))
            print(name, {label: round(summary['median_seconds'] * 1000, 4)
                         for label, summary in summaries.items()}, 'ms', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    print('MEASURED results:', args.output, flush=True)
    check_speedups(results, args.require_speedup)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path)
    parser.add_argument('--candidate-root', type=Path)
    parser.add_argument('--baseline-cli', type=Path)
    parser.add_argument('--candidate-cli', type=Path)
    parser.add_argument('--provenance', type=Path)
    parser.add_argument('--cases', nargs='+', choices=list(CASES), default=list(CASES))
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--seconds', type=float, default=.2)
    parser.add_argument('--affinity-width', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261117)
    parser.add_argument('--require-speedup', nargs='*', default=[], metavar='CASE:NUMERATOR/DENOMINATOR=FACTOR')
    parser.add_argument('--output', type=Path, default=Path('build/hardware-benchmark.json'))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--package-root', type=Path)
    parser.add_argument('--converter', type=Path)
    parser.add_argument('--backend', choices=['baseline', 'scalar', 'auto'], default='auto')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--case', choices=list(CASES))
    parser.add_argument('--matrix', type=Path)
    args = parser.parse_args()
    if args.trials < 1 or not math.isfinite(args.seconds) or args.seconds <= 0 or args.affinity_width < 1 or args.threads < 1:
        parser.error('trials, finite seconds, affinity width and threads must be positive')
    if not args.worker and (args.baseline_root is None or args.candidate_root is None):
        parser.error('--baseline-root and --candidate-root are required')
    if bool(args.baseline_cli) != bool(args.candidate_cli):
        parser.error('both CLI paths must be supplied together')
    worker(args) if args.worker else run(args)


if __name__ == '__main__':
    main()
