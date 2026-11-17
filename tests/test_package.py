"""Installed distribution metadata and binary provenance."""
import importlib.metadata
from pathlib import Path
import stringkernels


def test_installed_native_extension_and_version():
    from stringkernels import _native
    assert Path(_native.__file__).suffix in ('.so', '.pyd')
    assert stringkernels.__version__ == '1.1.0'
    assert importlib.metadata.version('string-kernels-libsvm') == stringkernels.__version__


def test_distribution_carries_project_and_binding_terms():
    distribution = importlib.metadata.distribution('string-kernels-libsvm')
    metadata = distribution.metadata
    assert metadata['License-Expression'] == 'BSD-3-Clause'
    files = distribution.files
    licenses = [p for p in files if '/licenses/' in str(p)]
    assert any(p.name == 'LICENSE' for p in licenses)
    notices = next(p for p in files if p.name == 'THIRD_PARTY_NOTICES.md')
    content = distribution.locate_file(notices).read_text(encoding='utf-8')
    assert 'pybind11' in content
    assert 'Wenzel Jakob' in content
    assert 'THIS SOFTWARE IS PROVIDED' in content
