"""Fold-local string kernels for scikit-learn pipelines."""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_consistent_length, check_is_fitted

from . import _execution_parameters, _equal_lengths, _native, _parameters, _sequences, pairwise


class StringKernel(TransformerMixin, BaseEstimator):
    """Transform strings to similarities against the fitted reference strings.

    Use before SVC(kernel='precomputed') or SVR(kernel='precomputed') in a
    Pipeline so cross-validation fits references only on each training fold.
    """

    def __init__(self, *, kernel='weighted-degree', degree=5, decay=0.5, n_jobs=1, backend='auto'):
        self.kernel = kernel
        self.degree = degree
        self.decay = decay
        self.n_jobs = n_jobs
        self.backend = backend

    def fit(self, X, y=None):
        """Copy and validate the training reference order without computing a Gram matrix."""
        self.__dict__.pop('reference_', None)
        reference = _sequences(X)
        if y is not None:
            check_consistent_length(reference, y)
        _native.validate_config(*_parameters(self.kernel, self.degree, self.decay))
        _native.validate_execution(*_execution_parameters(self.n_jobs, self.backend))
        _equal_lengths(reference, reference, self.kernel)
        self.reference_ = reference
        return self

    def fit_transform(self, X, y=None):
        """Materialize a training iterable once and return its Gram matrix."""
        self.fit(X, y)
        return self.transform(self.reference_)

    def transform(self, X):
        """Return similarities with one column per fitted training reference."""
        check_is_fitted(self, 'reference_')
        return pairwise(X, self.reference_, kernel=self.kernel, degree=self.degree, decay=self.decay,
                        n_jobs=self.n_jobs, backend=self.backend)
