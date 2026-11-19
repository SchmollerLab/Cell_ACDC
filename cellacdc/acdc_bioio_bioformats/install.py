from __future__ import annotations

import os

import re

import importlib.util
from dataclasses import dataclass
from typing import Literal, Sequence

from cellacdc import myutils

from . import EXTENSION_PACKAGE_MAPPER

pkg_regex = r'[a-zA-Z0-9_\-]+'

@dataclass(frozen=True)
class PackageSpec:
    """A package needed to read raw files (see `required_reader_packages`)."""
    pypi_name: str
    import_name: str
    installer: Literal['pip', 'conda']
    reason: str

def _reader_extension(image_filepath: os.PathLike) -> str:
    """Lower-cased extension used to look up `EXTENSION_PACKAGE_MAPPER`
    ('.ome.tiff' is kept as a compound extension)."""
    filepath = os.fspath(image_filepath).lower()
    if filepath.endswith('.ome.tiff'):
        return '.ome.tiff'
    _, ext = os.path.splitext(filepath)
    return ext

def _check_install_bioio_bioformats(
        qparent=None, auto_install=False, logger_func=print
    ):
    myutils.check_install_package(
        'scyjava',
        installer='conda', 
        is_cli=qparent is None,
        exact_version='1.10.2',
        parent=qparent,
        auto_install=auto_install,
        logger_func=logger_func,
    )
    
    myutils.check_install_package(
        'bioio-bioformats',
        installer='pip', 
        is_cli=qparent is None,
        min_version='1.0.0',
        max_version='2.0.0',
        include_higher_version=False,
        include_lower_version=True,
        parent=qparent,
        auto_install=auto_install,
        logger_func=logger_func,
    )
    
    return True

def _check_install_extra_format_dependency(
        image_filepath: os.PathLike,
        qparent=None,
        auto_install=False,
        logger_func=print
    ):
    
    ext = _reader_extension(image_filepath)
    package_name = EXTENSION_PACKAGE_MAPPER.get(ext)
    
    if package_name is None:
        _check_install_bioio_bioformats(
            qparent=qparent, auto_install=auto_install,
            logger_func=logger_func
        )
        return
    
    myutils.check_install_package(
        package_name,
        installer='pip', 
        is_cli=qparent is None,
        parent=qparent,
        auto_install=auto_install,
        logger_func=logger_func,
    )

def install_reader_dependencies(
        image_filepath: os.PathLike, 
        exception: Exception,
        qparent=None,
        *,
        auto_install=False,
        logger_func=print
    ):
    """Check (and install, asking unless `auto_install`) the reader packages
    needed for `image_filepath`. Any failure raises `exception`, chained to
    the original error."""
    try:
        _check_install_extra_format_dependency(
            image_filepath, qparent=qparent, auto_install=auto_install,
            logger_func=logger_func
        )
        
    except Exception as err:
        raise exception from err

def required_reader_packages(
        paths: Sequence[str], *, lazy_load: bool
    ) -> list[PackageSpec]:
    """Packages needed to read `paths` (pure; nothing is imported).

    bioio, then, in order of first appearance, the plugin of every distinct
    lower-cased extension (`EXTENSION_PACKAGE_MAPPER`), scyjava (conda) and
    bioio-bioformats for extensions without a plugin, and zarr when
    `lazy_load` is True. Each package is listed once.
    """
    extensions_per_package = {}
    for path in paths:
        ext = _reader_extension(path)
        package_name = EXTENSION_PACKAGE_MAPPER.get(ext)
        packages = (
            ('scyjava', 'bioio-bioformats') if package_name is None
            else (package_name,)
        )
        for package in packages:
            extensions = extensions_per_package.setdefault(package, [])
            if ext not in extensions:
                extensions.append(ext)

    specs = [PackageSpec('bioio', 'bioio', 'pip', 'BioIO image reader')]
    for package, extensions in extensions_per_package.items():
        files = ', '.join(extensions)
        if package == 'scyjava':
            installer = 'conda'
            reason = f'Java bridge of the Bio-Formats reader ({files} files)'
        elif package == 'bioio-bioformats':
            installer = 'pip'
            reason = f'Bio-Formats reader ({files} files)'
        else:
            installer = 'pip'
            reason = f'BioIO reader plugin ({files} files)'
        specs.append(PackageSpec(
            package, package.replace('-', '_'), installer, reason
        ))

    if lazy_load:
        specs.append(PackageSpec('zarr', 'zarr', 'pip', 'lazy loading'))

    return specs

def _is_importable(import_name: str) -> bool:
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError):
        return False

def missing_packages(specs: Sequence[PackageSpec]) -> list[PackageSpec]:
    """The `specs` whose package cannot be imported (importlib find_spec)."""
    return [spec for spec in specs if not _is_importable(spec.import_name)]