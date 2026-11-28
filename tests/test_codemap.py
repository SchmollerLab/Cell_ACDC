"""CODEMAP.md (repository root) indexes every package and module of cellacdc.

Checks that every package and module is listed in its package tree, that
every listed path exists, and that every path and `path::Name` named
anywhere in CODEMAP.md exists. After adding, moving or deleting a module run
`python tools/codemap.py` and describe the new entries.

The listing rules are those of tools/codemap.py (not imported: tests do not
import tools/); keep PLUGIN_PARENTS and CONTENTS_NOT_LISTED in sync.
"""
import os
import re

import pytest

pytestmark = pytest.mark.infra

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODEMAP = os.path.join(ROOT, 'CODEMAP.md')
START = '<!-- codemap:tree:start -->'
END = '<!-- codemap:tree:end -->'

PLUGIN_PARENTS = {
    'cellacdc/models', 'cellacdc/trackers', 'cellacdc/promptable_models',
    'cellacdc/measurement_processing',
}
CONTENTS_NOT_LISTED = {'cellacdc/bioformats', 'cellacdc/docs'}
SHIM_RE = re.compile(r'^__cellacdc_compat__\s*=\s*True\b', re.MULTILINE)
CODE_SPAN_RE = re.compile(r'`([^`\n]+)`')
PATH_RE = re.compile(r'[A-Za-z0-9_.\-/]+')
ROOT_PREFIXES = ('cellacdc/', 'tests/', 'tools/', '.github/')


def _read(path):
    with open(path, encoding='utf-8', errors='replace') as file:
        return file.read()


def _codemap_text():
    return _read(CODEMAP).replace('\r\n', '\n')


def _required_entries():
    """Packages (with a trailing '/') and modules that must be listed."""
    required = set()

    def walk(rel_dir):
        required.add(rel_dir + '/')
        if rel_dir in CONTENTS_NOT_LISTED:
            return
        abs_dir = os.path.join(ROOT, *rel_dir.split('/'))
        for name in os.listdir(abs_dir):
            path = os.path.join(abs_dir, name)
            rel = f'{rel_dir}/{name}'
            if os.path.isdir(path):
                if name == '__pycache__':
                    continue
                if rel_dir in PLUGIN_PARENTS and not name.startswith('_'):
                    continue
                if os.path.isfile(os.path.join(path, '__init__.py')):
                    walk(rel)
                continue
            if not name.endswith('.py') or name == '__init__.py':
                continue
            if name.startswith('qrc_resources'):
                continue
            if SHIM_RE.search(_read(path)):
                continue
            required.add(rel)

    walk('cellacdc')
    return required


def _tree_paths(text):
    """Paths of the package tree, rebuilt from the indentation."""
    start = text.index(START)
    end = text.index(END)
    paths = []
    stack = []
    for line in text[start + len(START):end].split('\n'):
        if not line.strip() or line.startswith('```'):
            continue
        indent = len(line) - len(line.lstrip(' '))
        assert indent % 2 == 0, f'odd indentation in the tree: {line!r}'
        depth = indent // 2
        name = line.split()[0]
        assert depth <= len(stack), f'tree line without parent: {line!r}'
        del stack[depth:]
        path = '/'.join(stack + [name.rstrip('/')])
        if name.endswith('/'):
            stack.append(name.rstrip('/'))
            path += '/'
        paths.append(path)
    return paths


def _resolve(token):
    """Absolute path of a path named in CODEMAP.md: relative to cellacdc/
    unless it starts with one of ROOT_PREFIXES (then the repository)."""
    candidates = []
    if not token.startswith(ROOT_PREFIXES):
        candidates.append(os.path.join(ROOT, 'cellacdc', token))
    candidates.append(os.path.join(ROOT, token))
    for candidate in candidates:
        if os.path.exists(candidate.rstrip('/')):
            return candidate.rstrip('/')
    return None


def _named_paths(text):
    """(token, path part, names) of the code spans that name a path: text
    with a slash or ending in .py/.md/.rst, optionally followed by
    `::Name` or `::Class.method`."""
    text = text[:text.index(START)] + text[text.index(END):]
    named = []
    for match in CODE_SPAN_RE.finditer(text):
        token = match.group(1).strip()
        path, _, names = token.partition('::')
        if not PATH_RE.fullmatch(path):
            continue
        if '/' not in path and not path.endswith(('.py', '.md', '.rst')):
            continue
        if path.startswith(('.', 'python')) or path in ('/', '//'):
            continue
        named.append((token, path, names))
    return named


def _defines(source, names):
    """Whether `source` defines `Name` (any depth) or, for `Class.attr`,
    defines `attr` inside the class body of `Class`."""
    def find(name, lines):
        pattern = re.compile(
            rf'^(\s*)(?:(?:async\s+)?def|class)\s+{re.escape(name)}\b'
            rf'|^(\s*){re.escape(name)}\s*(?::[^=]*)?=')
        for k, line in enumerate(lines):
            match = pattern.match(line)
            if match:
                indent = match.group(1) if match.group(1) is not None \
                    else match.group(2)
                return k, len(indent)
        return None

    lines = source.split('\n')
    parts = names.split('.')
    found = find(parts[0], lines)
    if found is None:
        return False
    for part in parts[1:]:
        start, indent = found
        body = []
        for line in lines[start + 1:]:
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and \
                    len(line) - len(line.lstrip()) <= indent:
                break
            body.append(line)
        found = find(part, body)
        if found is None:
            return False
        found = (start + 1 + found[0], found[1])
    return True


def test_codemap_lists_every_package_and_module():
    listed = set(_tree_paths(_codemap_text()))
    missing = sorted(_required_entries() - listed)
    assert not missing, (
        'Not listed in the package tree of CODEMAP.md (run python '
        'tools/codemap.py and describe them):\n' + '\n'.join(missing))


def test_codemap_tree_paths_exist():
    paths = _tree_paths(_codemap_text())
    duplicates = sorted({p for p in paths if paths.count(p) > 1})
    assert not duplicates, f'listed twice in CODEMAP.md: {duplicates}'
    gone = [p for p in paths
            if not os.path.exists(os.path.join(ROOT, *p.rstrip('/').split('/')))]
    assert not gone, (
        'Listed in the package tree of CODEMAP.md but not found (run python '
        'tools/codemap.py):\n' + '\n'.join(gone))


def test_codemap_named_paths_and_names_exist():
    problems = []
    sources = {}
    for token, path, names in _named_paths(_codemap_text()):
        resolved = _resolve(path)
        if resolved is None:
            problems.append(f'path not found: {token}')
            continue
        if not names:
            continue
        if resolved not in sources:
            sources[resolved] = _read(resolved).replace('\r\n', '\n')
        if not _defines(sources[resolved], names):
            problems.append(f'name not defined: {token}')
    assert not problems, 'CODEMAP.md names:\n' + '\n'.join(problems)
