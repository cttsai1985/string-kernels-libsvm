"""Normalized C++ string kernels as NumPy matrices."""
from numbers import Integral, Real

from . import _native

__version__ = _native.__version__
__all__ = ['pairwise', '__version__']

_KERNELS = {'positional': 0, 'weighted-degree': 1, 'spectrum': 2, 'subsequence': 3}


def _sequences(values):
    if isinstance(values, (str, bytes)):
        raise TypeError('expected a one-dimensional collection of strings')
    sequences = tuple(values)
    if not sequences:
        raise ValueError('sequence collections must not be empty')
    for value in sequences:
        if not isinstance(value, str):
            raise TypeError('each sequence must be a string')
        if not value or any(c in value for c in ' \t\r\n\v\f'):
            raise ValueError('sequences must be nonempty tokens without ASCII whitespace')
    return sequences


def _parameters(kernel, degree, decay):
    if not isinstance(kernel, str) or kernel not in _KERNELS:
        raise ValueError('kernel must be positional, weighted-degree, spectrum, or subsequence')
    if isinstance(degree, bool) or not isinstance(degree, Integral):
        raise TypeError('degree must be an integer')
    if isinstance(decay, bool) or not isinstance(decay, Real):
        raise TypeError('decay must be a real number')
    return _KERNELS[kernel], int(degree), float(decay)


def _equal_lengths(targets, references, kernel):
    if kernel == 'positional':
        size = len(references[0].encode('utf-8'))
        if any(len(x.encode('utf-8')) != size for x in (*targets, *references)):
            raise ValueError('positional kernel requires equal length byte strings')


def pairwise(X, Y=None, *, kernel='weighted-degree', degree=5, decay=0.5):
    """Return a float64 matrix with rows from X and reference columns from Y.

    With Y=None, compute the square training Gram matrix. Strings are compared
    as case-sensitive UTF-8 bytes without normalization or automatic alignment.
    A dense result uses 8 * len(X) * len(Y) bytes, excluding working storage.
    """
    targets = _sequences(X)
    references = targets if Y is None else _sequences(Y)
    parameters = _parameters(kernel, degree, decay)
    _equal_lengths(targets, references, kernel)
    return _native.pairwise(targets, references, *parameters)
