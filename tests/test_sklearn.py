"""Real solver predictions and fold ownership through public sklearn APIs."""
import pickle
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, SVR
from stringkernels import pairwise
from stringkernels.sklearn import StringKernel


TRAIN = ['ACG', 'ACT', 'GGA', 'GGT']
LABELS = [1, 1, -1, -1]


def test_pipeline_predicts_held_out_labels():
    pipeline = make_pipeline(StringKernel(kernel='spectrum', degree=2), SVC(kernel='precomputed'))
    pipeline.fit(TRAIN, LABELS)
    np.testing.assert_array_equal(pipeline.predict(['ACC', 'GGC']), [1, -1])
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(pipeline)).predict(['ACC', 'GGC']), [1, -1])


def test_transformer_clone_copy_and_reference_order():
    original = TRAIN.copy()
    transformer = StringKernel(kernel='spectrum', degree=2).fit(original)
    original.reverse()
    np.testing.assert_array_equal(transformer.transform(['ACC']),
                                  pairwise(['ACC'], TRAIN, kernel='spectrum', degree=2))
    assert clone(transformer).get_params() == transformer.get_params()
    with pytest.raises(NotFittedError):
        clone(transformer).transform(['ACC'])
    with pytest.raises(NotFittedError):
        StringKernel().transform(['ACC'])


def test_refit_failure_removes_stale_reference():
    transformer = StringKernel().fit(TRAIN)
    with pytest.raises(ValueError):
        transformer.fit([''])
    with pytest.raises(NotFittedError):
        transformer.transform(['ACC'])


def test_svr_matches_direct_precomputed_solver():
    target = ['ACC', 'GGC']
    labels = [0.2, 0.4, 1.5, 1.7]
    config = dict(kernel='spectrum', degree=2)
    pipeline = make_pipeline(StringKernel(**config), SVR(kernel='precomputed', C=3))
    pipeline.fit(TRAIN, labels)
    direct = SVR(kernel='precomputed', C=3).fit(pairwise(TRAIN, **config), labels)
    np.testing.assert_allclose(pipeline.predict(target), direct.predict(pairwise(target, TRAIN, **config)))


def test_cross_validation_and_grid_search_keep_fold_references():
    texts = ['ACA', 'ACC', 'ACG', 'ACT', 'GGA', 'GGC', 'GGG', 'GGT']
    labels = [1]*4 + [-1]*4
    pipeline = make_pipeline(StringKernel(kernel='spectrum', degree=2), SVC(kernel='precomputed'))
    result = cross_validate(pipeline, texts, labels, cv=2, return_estimator=True)
    for estimator in result['estimator']:
        assert len(estimator[0].reference_) == 4
        assert estimator[0].transform(['ACC']).shape == (1, 4)
    search = GridSearchCV(pipeline, {'stringkernel__degree': [1, 2], 'svc__C': [1, 3]}, cv=2)
    search.fit(texts, labels)
    np.testing.assert_array_equal(search.predict(['ACC', 'GGC']), [1, -1])


def test_transformer_defaults_and_length_validation():
    assert StringKernel().get_params() == dict(kernel='weighted-degree', degree=5, decay=.5)
    with pytest.raises(ValueError):
        StringKernel(kernel='positional').fit(['AC', 'A'])
    with pytest.raises(ValueError):
        StringKernel().fit(TRAIN, [1])


def test_fit_transform_materializes_a_generator_once():
    transformer = StringKernel(kernel='spectrum', degree=2)
    actual = transformer.fit_transform(iter(TRAIN), LABELS)
    np.testing.assert_array_equal(actual, pairwise(TRAIN, kernel='spectrum', degree=2))
    assert transformer.reference_ == tuple(TRAIN)
