"""acdc_bioio_bioformats.install: required_reader_packages, missing_packages
and install_reader_dependencies (case-insensitive extensions,
auto_install/logger_func plumbing, chained cause). Nothing is installed:
myutils.check_install_package is replaced by a recorder.
"""
from __future__ import annotations

import pytest

from cellacdc import myutils
from cellacdc.acdc_bioio_bioformats import install

PackageSpec = install.PackageSpec


def _pypi_names(specs):
    return [spec.pypi_name for spec in specs]


# -------------------------------------------------- required_reader_packages
def test_required_packages_case_insensitive_and_ordered():
    specs = install.required_reader_packages(
        ['b.CZI', 'a.TIF', 'c.czi', 'd.tiff', 'e.ND2'], lazy_load=False
    )
    assert _pypi_names(specs) == [
        'bioio', 'bioio-czi', 'bioio-tifffile', 'bioio-nd2'
    ]
    assert specs[0] == PackageSpec('bioio', 'bioio', 'pip', specs[0].reason)
    tifffile_spec = specs[2]
    assert tifffile_spec.import_name == 'bioio_tifffile'
    assert tifffile_spec.installer == 'pip'
    assert '.tif' in tifffile_spec.reason and '.tiff' in tifffile_spec.reason


def test_unmapped_extensions_need_bioformats():
    specs = install.required_reader_packages(
        ['x.vsi', 'y.VSI', 'z.lsm'], lazy_load=False
    )
    assert _pypi_names(specs) == ['bioio', 'scyjava', 'bioio-bioformats']
    scyjava, bioformats = specs[1:]
    assert (scyjava.import_name, scyjava.installer) == ('scyjava', 'conda')
    assert (bioformats.import_name, bioformats.installer) == (
        'bioio_bioformats', 'pip'
    )
    assert '.vsi' in bioformats.reason and '.lsm' in bioformats.reason


def test_compound_and_special_extensions():
    specs = install.required_reader_packages(
        ['a.OME.TIFF', 'b.ome.tif', 'c.ome.zarr', 'd.ims'], lazy_load=False
    )
    assert _pypi_names(specs) == [
        'bioio', 'bioio-ome-tiff', 'bioio-tifffile', 'bioio-ome-zarr',
        'imaris-ims-file-reader'
    ]
    assert specs[-1].import_name == 'imaris_ims_file_reader'


@pytest.mark.parametrize('lazy_load', [True, False])
def test_zarr_only_when_lazy(lazy_load):
    specs = install.required_reader_packages(['a.tif'], lazy_load=lazy_load)
    assert ('zarr' in _pypi_names(specs)) is lazy_load
    if lazy_load:
        assert specs[-1] == PackageSpec('zarr', 'zarr', 'pip', specs[-1].reason)


def test_no_paths_needs_bioio_only():
    assert _pypi_names(
        install.required_reader_packages([], lazy_load=False)
    ) == ['bioio']


# ------------------------------------------------------------ missing_packages
def test_missing_packages():
    present = PackageSpec('json', 'json', 'pip', 'stdlib')
    absent = PackageSpec(
        'acdc-no-such-pkg', 'acdc_no_such_pkg_x', 'pip', 'never installed'
    )
    absent_child = PackageSpec(
        'acdc-no-such-pkg', 'acdc_no_such_pkg_x.sub', 'pip', 'dotted'
    )
    assert install.missing_packages([present, absent, absent_child]) == [
        absent, absent_child
    ]


# --------------------------------------------------- install_reader_dependencies
@pytest.fixture
def recorded_installs(monkeypatch):
    calls = []

    def check_install_package(pkg_name, **kwargs):
        calls.append((pkg_name, kwargs))

    monkeypatch.setattr(myutils, 'check_install_package', check_install_package)
    return calls


def test_plugin_lookup_is_case_insensitive(recorded_installs):
    install.install_reader_dependencies('C:/data/A.CZI', RuntimeError('x'))
    assert [name for name, _ in recorded_installs] == ['bioio-czi']
    install.install_reader_dependencies('C:/data/B.OME.TIFF', RuntimeError())
    assert recorded_installs[-1][0] == 'bioio-ome-tiff'


def test_auto_install_and_logger_are_plumbed(recorded_installs):
    def log(*args):
        pass

    install.install_reader_dependencies(
        'x.vsi', RuntimeError('x'), auto_install=True, logger_func=log
    )
    assert [name for name, _ in recorded_installs] == [
        'scyjava', 'bioio-bioformats'
    ]
    scyjava_kwargs = recorded_installs[0][1]
    assert scyjava_kwargs['installer'] == 'conda'
    assert scyjava_kwargs['exact_version'] == '1.10.2'
    for _, kwargs in recorded_installs:
        assert kwargs['auto_install'] is True
        assert kwargs['logger_func'] is log
        assert kwargs['is_cli'] is True and kwargs['parent'] is None


def test_defaults_keep_the_gui_behaviour(recorded_installs):
    parent = object()
    install.install_reader_dependencies('x.nd2', RuntimeError(), qparent=parent)
    (name, kwargs), = recorded_installs
    assert name == 'bioio-nd2'
    assert kwargs['auto_install'] is False
    assert kwargs['logger_func'] is print
    assert kwargs['is_cli'] is False and kwargs['parent'] is parent


def test_failure_raises_the_given_exception_chained(monkeypatch):
    cause = ModuleNotFoundError('install it with `pip install bioio-czi`')

    def check_install_package(pkg_name, **kwargs):
        raise cause

    monkeypatch.setattr(myutils, 'check_install_package', check_install_package)
    exception = RuntimeError('Failed installing reader dependencies')
    with pytest.raises(RuntimeError) as info:
        install.install_reader_dependencies('a.czi', exception)
    assert info.value is exception
    assert info.value.__cause__ is cause
