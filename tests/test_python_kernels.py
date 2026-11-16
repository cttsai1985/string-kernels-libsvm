"""Independent feature-map specifications for the installed native extension."""
from collections import Counter
from itertools import combinations
import math

import numpy as np
import pytest
from stringkernels import pairwise


def features(text, kernel, degree, decay):
    text = text.encode('utf-8')
    if kernel == 'positional':
        return Counter({(i, value): 1.0 for i, value in enumerate(text)})
    if kernel == 'weighted-degree':
        return Counter({(i, size, text[i:i + size]):
                        math.sqrt(2 * (degree - size + 1) / (degree * (degree + 1)))
                        for size in range(1, min(degree, len(text)) + 1)
                        for i in range(len(text) - size + 1)})
    result = Counter()
    if kernel == 'spectrum':
        result.update(text[i:i + degree] for i in range(len(text) - degree + 1))
    else:
        for positions in combinations(range(len(text)), degree):
            token = bytes(text[i] for i in positions)
            result[token] += decay ** (positions[-1] - positions[0] + 1 - degree)
    return result


@pytest.mark.parametrize('kernel', ['positional', 'weighted-degree', 'spectrum', 'subsequence'])
def test_native_matrix_matches_independent_features(kernel):
    texts = ['AAA', 'ACA', 'CAA', 'GGT']
    maps = [features(x, kernel, 2, 0.5) for x in texts]
    expected = [[sum(v * b.get(k, 0) for k, v in a.items()) /
                 math.sqrt(sum(v*v for v in a.values()) * sum(v*v for v in b.values()))
                 for b in maps] for a in maps]
    actual = pairwise(texts, kernel=kernel, degree=2)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
    assert actual.dtype == np.float64
    assert actual.flags.c_contiguous


def test_hand_computed_weighted_degree_and_default_pins():
    np.testing.assert_allclose(pairwise(['AC', 'AD'], degree=2), [[1, .4], [.4, 1]])
    np.testing.assert_array_equal(pairwise(['AC', 'AD']),
                                  pairwise(['AC', 'AD'], kernel='weighted-degree', degree=5, decay=.5))
    assert pairwise(['AC', 'AD'])[0, 1] == pytest.approx(5/14)


def test_rectangular_matrix_preserves_reference_order_and_targets():
    expected = pairwise(['ACG', 'GAC', 'TTT'], kernel='spectrum', degree=2)
    actual = pairwise(['TTT', 'ACG'], ['GAC', 'TTT', 'ACG'], kernel='spectrum', degree=2)
    np.testing.assert_allclose(actual, expected[np.ix_([2, 0], [1, 2, 0])])
    assert actual.shape == (2, 3)


def test_gapped_features_and_zero_self_convention():
    assert pairwise(['ACG'], ['AG'], kernel='subsequence', degree=2)[0, 0] == pytest.approx(1/3)
    for kernel in ('spectrum', 'subsequence'):
        np.testing.assert_array_equal(pairwise(['A', 'C'], kernel=kernel, degree=2), np.zeros((2, 2)))
        assert pairwise(['AC'], kernel=kernel, degree=2, decay=.5)[0, 0] == 1


def test_byte_semantics_case_and_no_unicode_normalization():
    assert pairwise(['é'], ['ê'], kernel='positional')[0, 0] == .5
    assert pairwise(['A'], ['a'], kernel='spectrum', degree=1)[0, 0] == 0
    assert pairwise(['é'], ['e\u0301'], kernel='spectrum', degree=1)[0, 0] == 0
    np.testing.assert_array_equal(pairwise(np.array(['AC', 'AD'])), pairwise(['AC', 'AD']))


@pytest.mark.parametrize('bad', [[], [''], ['A C'], ['A\tC'], ['A\n'], [1], [None], [b'AC'], 'AC', [['AC']]])
def test_invalid_sequence_collections(bad):
    with pytest.raises((TypeError, ValueError)):
        pairwise(bad)


@pytest.mark.parametrize('kwargs', [dict(kernel='rbf'), dict(degree=0), dict(degree=-1),
    dict(degree=True), dict(degree=1.5), dict(degree=2**31), dict(decay=.9),
    dict(kernel='subsequence', decay=0), dict(kernel='subsequence', decay=float('nan')),
    dict(kernel='subsequence', decay=float('inf'))])
def test_invalid_kernel_configuration(kwargs):
    with pytest.raises((TypeError, ValueError, OverflowError)):
        pairwise(['AC'], **kwargs)


def test_positional_rejects_unequal_lengths_across_both_axes():
    with pytest.raises(ValueError, match='equal length'):
        pairwise(['AC'], ['A'], kernel='positional')
