"""Keep the package tree of CODEMAP.md in sync with the cellacdc package.

CODEMAP.md (repository root) is the index of the cellacdc package. Its
section "Package tree" lists one line per package, module and data folder,
between the lines `<!-- codemap:tree:start -->` and
`<!-- codemap:tree:end -->`, indented by two spaces per folder level:

    cellacdc/                              the package root: ...
      load.py                       4.9k   loadData (posData), ...
      gui/                                 the main GUI ...
        helpers.py                   250   ...

The number is the module's line count (rounded: 47, 250, 1.4k).

Usage, from the repository root (standard library only, Python >= 3.10):

    python tools/codemap.py           rewrite the tree: add new packages and
                                      modules, drop deleted ones, refresh the
                                      line counts; descriptions already in
                                      CODEMAP.md are kept
    python tools/codemap.py --check   exit 1 when a package or module is not
                                      listed, or a listed one is gone
                                      (tests/test_codemap.py checks the same
                                      with its own scan)
    python tools/codemap.py --print   print the tree instead of writing it

A new entry gets the description of `--themes` (a JSON file
{"cellacdc/<path>": "text"}) or else the first sentence of its module
docstring (a package: of its __init__.py), else "TODO: describe".
`--refresh` re-reads every description that way (hand edits are lost).

What is listed (tests/test_codemap.py uses the same rules):

- every package (a folder with __init__.py) reachable from cellacdc/
  through packages, and every module in it; an __init__.py gets its own
  line only when it has at least INIT_MIN_LINES lines;
- folders that are not packages (data, scripts, docs) get a folder line,
  without their contents;
- not listed: __pycache__, qrc_resources*.py (generated Qt resources),
  compat shims at old import paths (modules that assign
  `__cellacdc_compat__ = True`, see CODEMAP.md "Old name -> new location"),
  the contents of the folders in CONTENTS_NOT_LISTED (vendored code,
  docs) and of the plugin folders
  inside PLUGIN_PARENTS (one folder per plugin, e.g. models/<Name>/;
  folders starting with "_" there are shared code and are listed).
"""
import argparse
import ast
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = 'cellacdc'
CODEMAP = 'CODEMAP.md'
START = '<!-- codemap:tree:start -->'
END = '<!-- codemap:tree:end -->'
FENCE = '```'

# Keep these rules in sync with tests/test_codemap.py
PLUGIN_PARENTS = {
    'cellacdc/models', 'cellacdc/trackers', 'cellacdc/promptable_models',
    'cellacdc/measurement_processing',
}
# folders listed with one line, without their contents
CONTENTS_NOT_LISTED = {'cellacdc/bioformats', 'cellacdc/docs'}
SKIPPED_DIRS = {'__pycache__'}
INIT_MIN_LINES = 40
DESCRIPTION_MAX = 110
TODO = 'TODO: describe'

SHIM_RE = re.compile(r'^__cellacdc_compat__\s*=\s*True\b', re.MULTILINE)
SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z`(])')


class Entry:
    """One line of the tree: a folder (path ends with '/') or a module."""

    def __init__(self, path, depth, lines=None, docstring='',
                 required=True):
        self.path = path
        self.depth = depth
        self.lines = lines
        self.docstring = docstring
        # packages and modules must be listed, data folders may be
        self.required = required

    @property
    def name(self):
        if self.path.endswith('/'):
            return self.path.rstrip('/').rsplit('/', 1)[-1] + '/'
        return self.path.rsplit('/', 1)[-1]


def _read(path):
    with open(path, encoding='utf-8', errors='replace') as file:
        return file.read()


def is_shim(text):
    return SHIM_RE.search(text) is not None


def _docstring(text):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ''
    return ast.get_docstring(tree) or ''


def _count_lines(text):
    return text.count('\n') + (0 if text.endswith('\n') or not text else 1)


def scan(root=ROOT):
    """Entries of the tree in display order (see the module docstring)."""
    entries = []

    def walk(rel_dir, depth):
        abs_dir = os.path.join(root, *rel_dir.split('/'))
        names = sorted(os.listdir(abs_dir), key=lambda n: (n.lower(), n))
        init = os.path.join(abs_dir, '__init__.py')
        init_text = _read(init)
        entries.append(Entry(rel_dir + '/', depth,
                             docstring=_docstring(init_text)))
        if rel_dir in CONTENTS_NOT_LISTED:
            return
        init_lines = _count_lines(init_text)
        if init_lines >= INIT_MIN_LINES:
            entries.append(Entry(f'{rel_dir}/__init__.py', depth + 1,
                                 init_lines, _docstring(init_text),
                                 required=False))
        subdirs = []
        for name in names:
            path = os.path.join(abs_dir, name)
            if os.path.isdir(path):
                subdirs.append(name)
                continue
            if not name.endswith('.py') or name == '__init__.py':
                continue
            if name.startswith('qrc_resources'):
                continue
            text = _read(path)
            if is_shim(text):
                continue
            entries.append(Entry(f'{rel_dir}/{name}', depth + 1,
                                 _count_lines(text), _docstring(text)))
        for name in subdirs:
            if name in SKIPPED_DIRS:
                continue
            rel = f'{rel_dir}/{name}'
            if rel_dir in PLUGIN_PARENTS and not name.startswith('_'):
                continue
            if os.path.isfile(os.path.join(root, *rel.split('/'),
                                           '__init__.py')):
                walk(rel, depth + 1)
            else:
                entries.append(Entry(rel + '/', depth + 1, required=False))

    walk(PACKAGE, 0)
    return entries


def first_sentence(docstring):
    text = ' '.join(docstring.strip().split('\n\n')[0].split())
    if not text:
        return ''
    text = SENTENCE_END_RE.split(text)[0]
    if len(text) > DESCRIPTION_MAX:
        text = text[:DESCRIPTION_MAX].rsplit(' ', 1)[0].rstrip(',;:') + '...'
    return text


def format_lines(n):
    if n is None:
        return ''
    if n < 100:
        return str(n)
    if n < 1000:
        return str(int(round(n, -1)))
    return f'{n / 1000:.1f}k'


# ------------------------------------------------------------ CODEMAP.md
def split_codemap(text):
    """(head, tree lines, tail) of CODEMAP.md; the tree lines are the
    lines between the code fences inside the markers."""
    try:
        start = text.index(START)
        end = text.index(END)
    except ValueError:
        raise SystemExit(f'{CODEMAP}: markers {START} / {END} not found')
    head = text[:start + len(START)]
    tail = text[end:]
    body = text[start + len(START):end].strip('\n').split('\n')
    lines = [line for line in body if not line.startswith(FENCE)]
    return head, lines, tail


def parse_tree(lines):
    """{path: description} of the tree lines (paths rebuilt from the
    indentation: two spaces per folder level)."""
    listed = {}
    stack = []
    for line in lines:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(' '))
        depth = indent // 2
        parts = line.strip().split(None, 1)
        name = parts[0]
        rest = parts[1] if len(parts) > 1 else ''
        # drop the line count column
        rest = re.sub(r'^(\d+(\.\d+)?k?)\s+', '', rest)
        del stack[depth:]
        path = '/'.join(stack + [name.rstrip('/')])
        if name.endswith('/'):
            stack.append(name.rstrip('/'))
            path += '/'
        listed[path] = rest.strip()
    return listed


def render(entries, descriptions):
    width = max(2 * e.depth + len(e.name) for e in entries) + 2
    out = []
    for e in entries:
        left = ' ' * (2 * e.depth) + e.name
        desc = descriptions.get(e.path, TODO)
        line = f'{left:<{width}}{format_lines(e.lines):>5}   {desc}'
        out.append(line.rstrip())
    return out


def describe(entries, listed, themes, refresh=False):
    descriptions = {}
    for e in entries:
        current = listed.get(e.path, '')
        if current and current != TODO and not refresh:
            descriptions[e.path] = current
            continue
        descriptions[e.path] = (themes.get(e.path)
                                or first_sentence(e.docstring)
                                or current or TODO)
    return descriptions


def check(entries, listed, root=ROOT):
    """Messages for packages/modules not listed and listed paths that are
    gone (folders that are not packages are optional)."""
    problems = []
    expected = {e.path for e in entries}
    for e in entries:
        if e.required and e.path not in listed:
            problems.append(f'not listed: {e.path}')
    for path in listed:
        if path not in expected and not os.path.exists(
                os.path.join(root, *path.rstrip('/').split('/'))):
            problems.append(f'listed but not found: {path}')
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('--root', default=ROOT, help='repository root')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--print', action='store_true', dest='print_tree')
    parser.add_argument('--themes', default=None,
                        help='JSON {path: description} for new entries')
    parser.add_argument('--refresh', action='store_true',
                        help='re-read every description')
    args = parser.parse_args(argv)

    codemap_path = os.path.join(args.root, CODEMAP)
    text = _read(codemap_path)
    head, lines, tail = split_codemap(text)
    listed = parse_tree(lines)
    entries = scan(args.root)

    if args.check:
        problems = check(entries, listed, args.root)
        for problem in problems:
            print(problem)
        todo = [p for p, d in listed.items() if d == TODO]
        for path in todo:
            print(f'no description: {path}')
        if problems:
            print(f'{CODEMAP} is out of date: run python tools/codemap.py '
                  'and describe the new entries')
            return 1
        print(f'{CODEMAP}: {len(entries)} entries, up to date')
        return 0

    themes = {}
    if args.themes:
        with open(args.themes, encoding='utf-8') as file:
            themes = json.load(file)
    descriptions = describe(entries, listed, themes, refresh=args.refresh)
    tree = render(entries, descriptions)
    if args.print_tree:
        print('\n'.join(tree))
        return 0
    eol = '\r\n' if '\r\n' in text else '\n'
    text = text.replace('\r\n', '\n')
    head, _, tail = split_codemap(text)
    new = head + '\n' + FENCE + '\n' + '\n'.join(tree) + '\n' + FENCE + \
        '\n' + tail
    if new != text:
        with open(codemap_path, 'w', encoding='utf-8', newline='') as file:
            file.write(new.replace('\n', eol))
        print(f'{CODEMAP}: tree updated ({len(entries)} entries)')
    else:
        print(f'{CODEMAP}: tree up to date ({len(entries)} entries)')
    missing = [p for p, d in descriptions.items() if d == TODO]
    for path in missing:
        print(f'  describe: {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
