"""``myutils.app_info.read_version``: the version of an installed Cell-ACDC
(``site-packages``) or, in a git checkout, the one setuptools_scm
computes from the repository that contains the ``cellacdc`` package.

``read_version`` lives in ``cellacdc/myutils/app_info.py`` (it was in
``myutils.py``): it must use the package folder (``cellacdc_path``), not
the folder of its own module.
"""
import os

import pytest
import setuptools_scm

import cellacdc
from cellacdc.myutils import app_info

pytestmark = pytest.mark.infra


def test_checkout_version_is_read_from_the_package_root(monkeypatch):
    calls = []

    def get_version(root, relative_to):
        calls.append((root, relative_to))
        return '1.2.3.dev4'

    monkeypatch.setattr(setuptools_scm, 'get_version', get_version)
    monkeypatch.setattr(
        app_info, 'cellacdc_path', os.path.join('repo', 'cellacdc')
    )
    assert app_info.read_version(return_success=True) == ('1.2.3.dev4', True)
    assert calls == [('..', os.path.join('repo', 'cellacdc', '__init__.py'))]


def test_package_path_is_the_cellacdc_folder():
    assert os.path.samefile(
        app_info.cellacdc_path, os.path.dirname(cellacdc.__file__)
    )


def test_installed_version_is_the_package_version(monkeypatch):
    site_packages = os.path.join('env', 'Lib', 'site-packages', 'cellacdc')
    monkeypatch.setattr(app_info, 'cellacdc_path', site_packages)
    monkeypatch.setattr(cellacdc, '__version__', '9.8.7', raising=False)
    assert app_info.read_version(return_success=True) == ('9.8.7', True)


def test_failing_checkout_version_falls_back(monkeypatch):
    def get_version(root, relative_to):
        raise LookupError('no git')

    logged = []
    monkeypatch.setattr(setuptools_scm, 'get_version', get_version)
    version, success = app_info.read_version(
        logger=logged.append, return_success=True
    )
    assert success is False
    assert version
    assert any('could not determine the current version' in str(line)
               for line in logged)
