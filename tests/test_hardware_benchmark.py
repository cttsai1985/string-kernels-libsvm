"""Hardware timings are accepted only with equivalent work and honest controls."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'benchmark_hardware', Path(__file__).resolve().parents[1] / 'benchmarks/benchmark_hardware.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_hardware_parity_rejects_missing_or_wrong_matrix_cells():
    expected = np.array([[1., .25], [.25, 1.]])
    benchmark.validate_matrices({'auto4': expected.copy()}, expected)
    for wrong in (np.eye(2), expected[:, :1], np.full((2, 2), np.nan)):
        with pytest.raises(AssertionError):
            benchmark.validate_matrices({'auto4': wrong}, expected)


def test_requirement_compares_named_hardware_modes_and_can_fail():
    result = {'cases': [{'name': 'large_protein_training', 'results': {
        'scalar1': {'median_seconds': 4.}, 'auto1': {'median_seconds': 2.},
        'auto4': {'median_seconds': 2.}}}]}
    benchmark.check_speedups(result, ['large_protein_training:scalar1/auto1=1.5'])
    with pytest.raises(RuntimeError, match='below required'):
        benchmark.check_speedups(result, ['large_protein_training:auto1/auto4=1.5'])


def test_only_available_cpu_counts_are_scheduled():
    labels = [v['label'] for v in benchmark.variants(None, None, 1)]
    assert labels == ['baseline', 'scalar1', 'auto1']
    labels = [v['label'] for v in benchmark.variants(None, None, 2)]
    assert labels == ['baseline', 'scalar1', 'auto1', 'auto2']
    labels = [v['label'] for v in benchmark.variants(Path('old'), Path('new'), 4)]
    assert labels == ['baseline', 'scalar1', 'auto1', 'auto2', 'auto4',
                      'baseline_cli', 'scalar1_cli', 'auto1_cli', 'auto2_cli', 'auto4_cli']


def test_portable_cpu_readback_is_not_reported_as_simd():
    metadata = benchmark.cpu_metadata('auto', 'scalar')
    assert metadata['reported_cpu_backend'] == 'scalar'
    assert not metadata['simd_available']
    metadata = benchmark.cpu_metadata('scalar', 'avx2')
    assert metadata['reported_cpu_backend'] == 'avx2'
    assert metadata['requested_backend'] == 'scalar'
    assert not metadata['simd_dispatch_allowed']


def test_cli_matrix_parser_checks_serials_and_column_order(tmp_path):
    matrix = tmp_path / 'matrix.txt'
    matrix.write_text('0 0:1 1:1 2:0.25\n0 0:2 1:0.25 2:1\n')
    np.testing.assert_array_equal(benchmark.parse_cli_matrix(matrix, (2, 2)),
                                  [[1., .25], [.25, 1.]])
    for wrong in ('0 0:2 1:1 2:0.25\n', '0 0:1 2:1 1:0.25\n', '0 0:1 1:1\n'):
        matrix.write_text(wrong)
        with pytest.raises(AssertionError):
            benchmark.parse_cli_matrix(matrix, (1, 2))


@pytest.mark.parametrize('floor', ['nan', 'inf', '0', '-1'])
def test_invalid_requirement_cannot_silently_pass(floor):
    with pytest.raises(ValueError, match='finite and positive'):
        benchmark.check_speedups({'cases': []}, [f'case:scalar1/auto1={floor}'])
