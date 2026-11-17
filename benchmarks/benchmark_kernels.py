#!/usr/bin/env python3
"""Compare installed native kernels on deterministic inputs, with full-output parity."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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

CASES = {
    'tiny_rectangular': dict(kernel='weighted-degree', rows=16, columns=12, length=12, degree=3, alphabet='ACDEFGHIKLMNPQRSTVWY'),
    'protein_training': dict(kernel='weighted-degree', rows=192, columns=None, length=64, degree=5, alphabet='ACDEFGHIKLMNPQRSTVWY'),
    'high_similarity': dict(kernel='weighted-degree', rows=48, columns=None, length=192, degree=64, alphabet='ACGT', similar=True),
    'protein_prediction': dict(kernel='weighted-degree', rows=128, columns=256, length=64, degree=5, alphabet='ACDEFGHIKLMNPQRSTVWY'),
    'short_degree_one': dict(kernel='weighted-degree', rows=512, columns=None, length=32, degree=1, alphabet='ACGT'),
    'spectrum_training': dict(kernel='spectrum', rows=128, columns=None, length=128, degree=3, alphabet='ACGT'),
}


def dataset(case, seed):
    import numpy as np
    rng = random.Random(seed)
    def sequences(count):
        result = []
        for _ in range(count):
            if case.get('similar'):
                chars = ['A'] * case['length']
                chars[rng.randrange(len(chars))] = rng.choice('CGT')
                result.append(''.join(chars))
            else:
                result.append(''.join(rng.choice(case['alphabet']) for _ in range(case['length'])))
        return np.asarray(result)
    x = sequences(case['rows'])
    y = x if case['columns'] is None else sequences(case['columns'])
    return x, y


def cpu_seconds():
    value = time.process_time()
    try:
        import resource
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        value += children.ru_utime + children.ru_stime
    except ImportError:
        pass
    return value


def worker(args):
    # Each worker is a fresh process; never attempt to swap an imported extension.
    if args.package_root:
        sys.path.insert(0, str(Path(args.package_root).resolve()))
    import numpy as np
    case = CASES[args.case]
    x, y = dataset(case, args.seed)
    data_hash = hashlib.sha256(('\0'.join([*x, '|', *y])).encode()).hexdigest()
    config = dict(kernel=case['kernel'], degree=case['degree'])
    if args.backend == 'strkernels':
        from strkernels import WeightedDegreeStringKernel
        if case['kernel'] != 'weighted-degree':
            raise ValueError('Only weighted-degree has matched semantics in this comparator')
        kernel = WeightedDegreeStringKernel(normalizer='sqrt_diagonal', degree=case['degree'])
        compute = lambda: kernel(x, y)
        version = importlib.metadata.version('strkernels')
        origin = str(next(Path(sys.modules['strkernels'].__file__).parent.glob('libcore*')).resolve())
    elif args.backend == 'cli':
        directory = Path(args.matrix).parent
        train, target, output = (directory / f'{args.case}-{name}' for name in ('train.txt', 'test.txt', 'matrix.txt'))
        train.write_text(''.join(f'{s} 0\n' for s in y))
        target.write_text(''.join(f'{s} 0\n' for s in x))
        command = [args.converter, '-train', str(train), '--kernel', case['kernel'], '-k', str(case['degree'])]
        if y is not x:
            command += ['-test', str(target)]
        def compute():
            with output.open('wb') as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.PIPE, check=True)
        version = subprocess.check_output([args.converter, '--version'], text=True).strip()
        origin = str(Path(args.converter).resolve())
    else:
        import stringkernels
        compute = lambda: stringkernels.pairwise(x, None if y is x else y, **config)
        version = stringkernels.__version__
        from stringkernels import _native
        origin = str(Path(_native.__file__).resolve())
    started = time.perf_counter()
    matrix = compute()
    warmup = time.perf_counter() - started
    loops = 1
    while True:
        cpu_start, wall_start = cpu_seconds(), time.perf_counter()
        for _ in range(loops):
            matrix = compute()
        elapsed = time.perf_counter() - wall_start
        cpu = cpu_seconds() - cpu_start
        if elapsed >= args.seconds or loops >= 100000:
            break
        loops = min(100000, max(loops + 1, math.ceil(loops * args.seconds / max(elapsed, 1e-9))))
    if args.backend == 'cli':
        matrix = np.asarray([[float(cell.split(':')[1]) for cell in line.split()[2:]]
                             for line in output.read_text().splitlines()])
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
    print(json.dumps(dict(seconds=elapsed / loops, cpu_seconds=cpu / loops,
                          cpu_to_wall=cpu / elapsed, loops=loops, block_wall_seconds=elapsed, warmup_seconds=warmup, version=version,
                          origin=origin, binary_sha256=hashlib.sha256(Path(origin).read_bytes()).hexdigest(), data_sha256=data_hash, pools=pools,
                          numpy=np.__version__)))


def affinity(width):
    if not hasattr(os, 'sched_getaffinity'):
        return None
    allowed = sorted(os.sched_getaffinity(0))
    chosen, physical = [], set()
    for cpu in allowed:
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        try:
            key = ((topology / 'physical_package_id').read_text().strip(),
                   (topology / 'core_id').read_text().strip())
        except OSError:
            key = cpu
        if key not in physical:
            chosen.append(cpu)
            physical.add(key)
        if len(chosen) == width:
            break
    os.sched_setaffinity(0, chosen)
    return chosen


def cpu_model():
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def validate_matrices(matrices, expected):
    import numpy as np
    for matrix in matrices.values():
        assert matrix.shape == expected.shape, (matrix.shape, expected.shape)
        assert matrix.dtype == np.float64 and np.isfinite(matrix).all()
        np.testing.assert_allclose(matrix, expected, rtol=1e-11, atol=1e-12)


def check_speedups(results, requirements):
    for requirement in requirements:
        name, floor = requirement.split('=')
        case = next(c for c in results['cases'] if c['name'] == name)
        ratio = case['results']['baseline']['median_seconds'] / case['results']['candidate']['median_seconds']
        if ratio < float(floor):
            raise RuntimeError(f'{name}: measured speedup {ratio:.3f} below required {floor}')


def run(args):
    import numpy as np
    cpus = affinity(args.affinity_width)
    variants = [('baseline', 'native', args.baseline_root, None, 1),
                ('candidate', 'native', args.candidate_root, None, 1)]
    if args.include_strkernels:
        variants.append(('strkernels_1', 'strkernels', None, None, 1))
        if cpus is None or len(cpus) >= 4:
            variants.append(('strkernels_4', 'strkernels', None, None, 4))
    if args.baseline_cli and args.candidate_cli:
        variants.extend([('baseline_cli', 'cli', None, args.baseline_cli, 1),
                         ('candidate_cli', 'cli', None, args.candidate_cli, 1)])
    results = dict(seed=args.seed, trials=args.trials, minimum_trial_seconds=args.seconds,
                   environment=dict(python=sys.version, platform=platform.platform(),
                                    processor=cpu_model(), affinity=cpus,
                                    cpu_count=os.cpu_count(), load_average=os.getloadavg() if hasattr(os, 'getloadavg') else None),
                   provenance=json.loads(args.provenance.read_text()) if args.provenance else {},
                   cases=[])
    rng = random.Random(args.seed)
    with tempfile.TemporaryDirectory(prefix='string-kernel-bench-') as directory:
        root = Path(directory)
        for case_name in args.cases:
            case_variants = [v for v in variants if CASES[case_name]['kernel'] == 'weighted-degree' or v[1] != 'strkernels']
            case_results = {v[0]: [] for v in case_variants}
            expected = None
            for trial in range(args.trials):
                order = case_variants.copy()
                rng.shuffle(order)
                matrices = {}
                for label, backend, package_root, converter, threads in order:
                    matrix_path = root / f'{case_name}-{label}.npy'
                    command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                               '--backend', backend, '--case', case_name, '--seed', str(args.seed),
                               '--seconds', str(args.seconds), '--matrix', str(matrix_path)]
                    if package_root:
                        command += ['--package-root', str(Path(package_root).resolve())]
                    if converter:
                        command += ['--converter', str(Path(converter).resolve())]
                    env = os.environ.copy()
                    env.pop('PYTHONPATH', None)
                    env.update(OMP_NUM_THREADS=str(threads), OMP_DYNAMIC='FALSE',
                               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                               VECLIB_MAXIMUM_THREADS='1', BLIS_NUM_THREADS='1')
                    process = subprocess.run(command, env=env, capture_output=True, text=True, timeout=300)
                    if process.returncode:
                        raise RuntimeError(f'{label}/{case_name} worker failed: {process.stderr}')
                    result = json.loads(process.stdout)
                    result['threads'] = threads
                    case_results[label].append(result)
                    matrices[label] = np.load(matrix_path)
                if expected is None:
                    expected = matrices['baseline']
                assert len({runs[-1]['data_sha256'] for runs in case_results.values()}) == 1
                validate_matrices(matrices, expected)
            summaries = {}
            for label, runs in case_results.items():
                times = [r['seconds'] for r in runs]
                summaries[label] = dict(median_seconds=statistics.median(times),
                                        minimum_seconds=min(times), maximum_seconds=max(times),
                                        runs=runs, full_matrix_parity=True)
            results['cases'].append(dict(name=case_name, parameters=CASES[case_name], results=summaries,
                                         exclusions=['strkernels spectrum sums all orders; this project uses exact-k features'] if CASES[case_name]['kernel'] == 'spectrum' else []))
            print(case_name, {label: round(s['median_seconds'] * 1000, 4) for label, s in summaries.items()}, 'ms', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    print('MEASURED results:', args.output, flush=True)
    check_speedups(results, args.require_speedup)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provenance', type=Path, help='JSON build/source descriptions retained with the results')
    parser.add_argument('--baseline-root', type=Path)
    parser.add_argument('--candidate-root', type=Path)
    parser.add_argument('--baseline-cli', type=Path)
    parser.add_argument('--candidate-cli', type=Path)
    parser.add_argument('--include-strkernels', action='store_true')
    parser.add_argument('--cases', nargs='+', choices=list(CASES), default=list(CASES))
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--seconds', type=float, default=.1)
    parser.add_argument('--affinity-width', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261117)
    parser.add_argument('--require-speedup', nargs='*', default=[], metavar='CASE=FACTOR')
    parser.add_argument('--output', type=Path, default=Path('build/benchmark.json'))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--backend', choices=['native', 'strkernels', 'cli'], default='native')
    parser.add_argument('--package-root', type=Path)
    parser.add_argument('--converter')
    parser.add_argument('--case', choices=list(CASES))
    parser.add_argument('--matrix', type=Path)
    args = parser.parse_args()
    if args.trials < 1 or args.seconds <= 0 or args.affinity_width < 1:
        parser.error('trials, seconds and affinity width must be positive')
    worker(args) if args.worker else run(args)


if __name__ == '__main__':
    main()
