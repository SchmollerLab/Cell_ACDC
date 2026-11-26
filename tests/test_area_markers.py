"""Every test file names the area of the code it tests with exactly one area
marker in its module-level `pytestmark`, so that `pytest -m <area>` selects
every test of an area and the areas split the suite without overlaps (see
"Running the tests" in the contributing docs)."""
import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.infra

AREA_MARKERS = {'viewer', 'gui', 'models', 'core', 'infra'}
TESTS_FOLDER = Path(__file__).resolve().parent
AREA_PATTERN = re.compile(r'\bmark\.(%s)\b' % '|'.join(sorted(AREA_MARKERS)))
PYTESTMARK_PATTERN = re.compile(r'^pytestmark\s*=', re.MULTILINE)


def area_markers(path):
    """Area markers of the module `pytestmark` and of the whole file.

    Parsing every test file takes about half a second, so only the
    `pytestmark` statement is parsed, and the whole file only when area
    names appear elsewhere too (markers of single tests, or text in strings).
    """
    text = path.read_text(encoding='utf-8')
    in_pytestmark = []
    for match in PYTESTMARK_PATTERN.finditer(text):
        statement = _statement_at(text, match.start())
        in_pytestmark.extend(_area_names(statement))
    if len(AREA_PATTERN.findall(text)) == len(in_pytestmark):
        return in_pytestmark, list(in_pytestmark)
    return in_pytestmark, _area_names(ast.parse(text))


def _statement_at(text, start):
    lines = text[start:].split('\n')
    for end in range(1, len(lines) + 1):
        try:
            return ast.parse('\n'.join(lines[:end]))
        except SyntaxError:
            continue
    raise SyntaxError(f'Incomplete statement: {lines[0]}')


def _area_names(node):
    return [
        sub.attr for sub in ast.walk(node)
        if isinstance(sub, ast.Attribute)
        and isinstance(sub.value, ast.Attribute)
        and sub.value.attr == 'mark'
        and sub.attr in AREA_MARKERS
    ]


def test_area_markers_are_registered(pytestconfig):
    registered = {
        line.split(':', 1)[0].strip()
        for line in pytestconfig.getini('markers')
    }
    assert AREA_MARKERS <= registered


def test_every_test_file_has_one_area_marker():
    wrong = {}
    for path in sorted(TESTS_FOLDER.rglob('test_*.py')):
        in_pytestmark, in_file = area_markers(path)
        if len(in_pytestmark) != 1 or len(in_file) != 1:
            wrong[path.relative_to(TESTS_FOLDER).as_posix()] = in_file
    assert not wrong, (
        'Each test file needs exactly one area marker, set in its module '
        'pytestmark (e.g. "pytestmark = pytest.mark.viewer"), and none on '
        f'single tests. Areas: {sorted(AREA_MARKERS)}. Files: {wrong}'
    )


@pytest.mark.parametrize('source, expected', [
    (
        'import pytest\n'
        'pytestmark = [\n'
        '    pytest.mark.gui,\n'
        '    pytest.mark.skipif(True, reason="")\n'
        ']\n'
        'def test_x():\n'
        '    """Unlike pytest.mark.viewer, ..."""\n',
        (['gui'], ['gui']),
    ),
    (
        'import pytest\n'
        'pytestmark = pytest.mark.gui\n'
        '@pytest.mark.viewer\n'
        '@pytest.mark.slow\n'
        'def test_x():\n'
        '    pass\n',
        (['gui'], ['gui', 'viewer']),
    ),
    (
        'import pytest\n'
        '@pytest.mark.parametrize("x", [pytest.param(1, marks=pytest.mark.core)])\n'
        'def test_x(x):\n'
        '    pass\n',
        ([], ['core']),
    ),
])
def test_area_markers_are_read_from_pytestmark_and_code(
        tmp_path, source, expected
    ):
    path = tmp_path / 'test_example.py'
    path.write_text(source, encoding='utf-8')

    in_pytestmark, in_file = area_markers(path)

    assert (in_pytestmark, sorted(in_file)) == expected
