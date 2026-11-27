"""Every folder of ``cellacdc`` with Python modules is a regular package.

``pyproject.toml`` builds the wheel with ``namespaces = false``
(``[tool.setuptools.packages.find]``): a folder without ``__init__.py``
is left out of the wheel, so its modules would be missing from an
installed Cell-ACDC (while they import fine from a checkout). The
package layout added many folders (CODEMAP.md); each needs an
``__init__.py``.
"""
import re

import pytest

from tests.source_scan import PACKAGE_DIR, REPO_ROOT

pytestmark = pytest.mark.infra

# Folders without __init__.py that already existed before the package
# layout and are not imported as packages (scripts, docs, demos, files
# loaded by path)
NOT_PACKAGES = {
    'cellacdc/_deprecated',
    'cellacdc/bioformats/tests',
    'cellacdc/docs/source/citations',
    'cellacdc/metrics',
    'cellacdc/resources',
    'cellacdc/scripts',
    'cellacdc/volume_renderer/_demos',
}


def test_folders_with_modules_have_an_init_file():
    folders = {
        path.parent for path in PACKAGE_DIR.rglob('*.py')
        if '__pycache__' not in path.parts
    }
    missing = sorted(
        folder.relative_to(REPO_ROOT).as_posix() for folder in folders
        if not (folder / '__init__.py').is_file()
    )
    unexpected = [folder for folder in missing if folder not in NOT_PACKAGES]
    assert not unexpected, (
        f'Add an __init__.py (with a docstring) to {unexpected}: '
        'pyproject.toml packages only regular packages (namespaces = false)'
    )
    now_packages = sorted(
        folder for folder in NOT_PACKAGES
        if (REPO_ROOT / folder / '__init__.py').is_file()
    )
    assert not now_packages, f'Remove from NOT_PACKAGES: {now_packages}'


def test_wheel_packages_only_regular_packages():
    text = (REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8')
    section = text.split('[tool.setuptools.packages.find]', 1)[1]
    section = section.split('\n[', 1)[0]
    assert re.search(r'^namespaces\s*=\s*false\s*$', section, re.MULTILINE)
