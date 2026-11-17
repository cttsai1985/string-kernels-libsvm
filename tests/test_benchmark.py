"""The benchmark must reject incomplete or numerically wrong work."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'benchmark_kernels', Path(__file__).resolve().parents[1] / 'benchmarks/benchmark_kernels.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_parity_checks_off_diagonal_and_dimensions():
    expected = np.array([[1., .25], [.25, 1.]])
    benchmark.validate_matrices({'baseline': expected, 'candidate': expected.copy()}, expected)
    wrong = expected.copy()
    wrong[0, 1] = .5
    with pytest.raises(AssertionError):
        benchmark.validate_matrices({'candidate': wrong}, expected)
    with pytest.raises(AssertionError):
        benchmark.validate_matrices({'candidate': expected[:, :1]}, expected)
    with pytest.raises(AssertionError):
        benchmark.validate_matrices({'candidate': np.full((2, 2), np.nan)}, expected)


def test_performance_requirement_can_fail():
    results = {'cases': [{'name': 'high_similarity', 'results': {
        'baseline': {'median_seconds': 1.}, 'candidate': {'median_seconds': 1.}}}]}
    with pytest.raises(RuntimeError, match='below required'):
        benchmark.check_speedups(results, ['high_similarity=3'])
    results['cases'][0]['results']['candidate']['median_seconds'] = .25
    benchmark.check_speedups(results, ['high_similarity=3'])
