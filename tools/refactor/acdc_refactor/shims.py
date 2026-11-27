"""Compat shims, compat blocks and the generated cellacdc/_compat_moves.py.

Kinds of old paths (see cellacdc/_compat.py, written from
templates/_compat.py in commit C2):

- 'alias' shim: the old module file becomes
  `_compat.alias_module(__name__)`: importing it returns the module at the
  new path (same object, sys.modules alias) with a MovedWarning;
- 'split' shim: `__getattr__, __dir__ = _compat.moved_module(__name__)`:
  every old name resolves lazily (never cached) through its home module;
- package: an old module replaced by a same-name package (gui.py ->
  gui/): the package __init__ calls `_compat.package_compat(__name__)`;
- compat block: a module that keeps existing but lost names gets the
  moved_module __getattr__ appended. Blocks in modules that are not old
  paths are marked INTERMEDIATE_MARKER and removed in C12 once every
  reference uses the new homes (remove_intermediate_blocks).

Shims and package inits carry `__cellacdc_compat__ = True` (shims) so
tests/related_tests.py and the no-old-paths test can skip them.
"""
import ast
import os
import pprint

from . import RefactorError, TOOL_DIR, commit_le
from .plan import dotted
from .source import SourceText, strip_eol, is_blank
from .textutil import format_docstring

INTERMEDIATE_MARKER = (
    '# cellacdc-refactor: intermediate compat block (removed in C12)')
COMPAT_BLOCK_MARKER = '# Names that moved out of this module'
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'templates', '_compat.py')


def slash(path):
    """'cellacdc/gui/mixins/navigation.py' -> 'gui/mixins/navigation.py'
    (docstrings use slash paths below cellacdc/)."""
    return path[len('cellacdc/'):] if path.startswith('cellacdc/') else path


def alias_shim_text(old_path, new_path):
    doc = (f'Deprecated: moved to {slash(new_path)}.\n\n'
           'Importing this module gives the module at the new path (the '
           'same object) and emits a DeprecationWarning. See CODEMAP.md.')
    return (format_docstring(doc) + '\n'
            '__cellacdc_compat__ = True\n\n'
            'from cellacdc import _compat\n\n'
            '_compat.alias_module(__name__)\n')


def split_shim_text(old_path, homes):
    homes = sorted({slash(h) for h in homes})
    if homes and len(homes) <= 12:
        where = 'Its names now live in: ' + ', '.join(homes) + '.'
    else:
        where = f'Its names now live in {len(homes)} modules.'
    doc = (f'Deprecated: the contents of {slash(old_path)} moved to other '
           f'modules.\n\n{where}\n\nOld names are forwarded lazily, with a '
           'DeprecationWarning (cellacdc/_compat.py). See CODEMAP.md.')
    return (format_docstring(doc) + '\n'
            '__cellacdc_compat__ = True\n\n'
            'from cellacdc import _compat\n\n'
            '__getattr__, __dir__ = _compat.moved_module(__name__)\n')


def compat_block_text(intermediate):
    lines = ['', '']
    if intermediate:
        lines.append(INTERMEDIATE_MARKER)
    lines += [
        COMPAT_BLOCK_MARKER + ' stay reachable here with a',
        '# DeprecationWarning (cellacdc/_compat.py, see CODEMAP.md).',
        'from cellacdc import _compat as _acdc_compat  # noqa: E402',
        '',
        '__getattr__, __dir__ = _acdc_compat.moved_module(__name__)',
    ]
    return '\n'.join(lines) + '\n'


def package_init_text(plan, path, spec):
    """Text of a new package __init__.py."""
    parts = []
    doc = spec.get('doc')
    if doc:
        parts.append(format_docstring(doc))
    code = spec.get('init_code')
    if isinstance(code, list):
        code = '\n'.join(code)
    if code:
        parts.append(code.rstrip('\n'))
    replaces_module = plan.base.exists(path + '.py')
    forward = spec.get('forward')
    if replaces_module or forward:
        call = '_acdc_compat.package_compat(__name__)'
        if forward and not replaces_module:
            call = (f'_acdc_compat.forward_module(__name__, '
                    f'{dotted(forward)!r})')
        elif forward:
            call = (f'_acdc_compat.package_compat(__name__, '
                    f'forward={dotted(forward)!r})')
        parts.append('from cellacdc import _compat as _acdc_compat\n\n'
                     f'__getattr__, __dir__ = {call}')
    return '\n\n'.join(parts) + '\n' if parts else ''


def forward_block_text(forward_dotted):
    return ('\n\n# Old names of ' + forward_dotted + ' that code reached '
            'through this module\n'
            '# path before it existed (cellacdc/_compat.py).\n'
            'from cellacdc import _compat as _acdc_compat  # noqa: E402\n\n'
            '__getattr__, __dir__ = _acdc_compat.forward_module(\n'
            f'    __name__, {forward_dotted!r})\n')


def _only_imports_left(src):
    tree = ast.parse(src.text)
    left = []
    for i, node in enumerate(tree.body):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Pass)):
            continue
        if i == 0 and isinstance(node, ast.Expr) and isinstance(
                node.value, ast.Constant) and isinstance(node.value.value,
                                                         str):
            continue
        from .items import is_import_block
        if is_import_block(node):
            continue
        if isinstance(node, ast.Assign) and all(
                isinstance(t, (ast.Name, ast.Tuple)) and all(
                    n.id.startswith('__') for n in ast.walk(t)
                    if isinstance(n, ast.Name))
                for t in node.targets):
            continue    # __getattr__, __dir__ = ... / __cellacdc_compat__
        left.append(f'line {node.lineno}: '
                    f'{strip_eol(src.line(node.lineno)).strip()}')
    return left


def has_compat_block(text):
    return 'moved_module(__name__)' in text or \
        'package_compat(__name__' in text or '__cellacdc_compat__' in text


def write_compat_moves(repo, plan, commit):
    """Write cellacdc/_compat_moves.py for the state after `commit`."""
    path = os.path.join(repo, 'cellacdc', '_compat_moves.py')
    text = compat_moves_text(plan, commit)
    eol = '\n'
    init = os.path.join(repo, 'cellacdc', '__init__.py')
    if os.path.exists(init):
        eol = SourceText.read(init).eol
    SourceText(text.replace('\n', eol), eol=eol).write(path)
    return path


def compat_tables(plan, commit):
    """(MODULE_ALIASES, NAME_MOVES, PLUGIN_API, PACKAGE_SUBMODULES,
    FORWARDERS) for the state after `commit` (empty before C3)."""
    table = plan.name_table(commit)
    aliases = {}
    for move in plan.moves_upto(commit):
        if move.is_python_file and plan.shim_kind(move.old,
                                                  commit) == 'alias':
            aliases[dotted(move.old)] = dotted(
                plan.module_path(move.old, commit))
    name_moves = {}
    for module, entries in table['runtime'].items():
        name_moves[module] = {n: tuple(v) for n, v in entries.items()}
    for module, entries in table['intermediate'].items():
        name_moves.setdefault(module, {}).update(
            {n: tuple(v) for n, v in entries.items()})
    plugin_api = {m: frozenset(names) for m, names
                  in table['plugin_api'].items()}
    submodules = {}
    forwarders = {}
    for path, spec in plan.packages.items():
        if not commit_le(spec['commit'], commit):
            continue
        if plan.base.exists(path + '.py'):
            submodules[dotted(path)] = tuple(sorted(
                plan.package_submodules(path, commit)))
            for name in submodules[dotted(path)]:
                name_moves.get(dotted(path), {}).pop(name, None)
        if spec.get('forward'):
            forwarders[dotted(path)] = dotted(spec['forward'])
    for path, spec in plan.modules.items():
        if spec.get('forward'):
            forwarders[dotted(path)] = dotted(spec['forward'])
    for path, spec in plan.forwarders.items():
        if commit_le(spec['commit'], commit):
            forwarders[dotted(path)] = dotted(spec['table'])
    return aliases, name_moves, plugin_api, submodules, forwarders


def patch_fanout(plan, commit, old_modules):
    """Old module -> the modules holding its code after `commit` (where it
    moved whole, and every destination of its items and of the methods of
    its split classes). A patch of an old module's global must reach the
    copies of that global in all of them (move.py copies import-bound
    names into every destination that uses them)."""
    out = {}
    old_files = ({m.old_file for m in plan.items.values()}
                 | {s.old_file for s in plan.classes.values()}
                 | {m.old for m in plan.module_moves if m.is_python_file})
    for old_file in sorted(old_files):
        old = dotted(old_file)
        if old not in old_modules:
            continue
        mods = set()
        here = plan.module_path(old_file, commit)
        if here.endswith('.py'):
            mods.add(dotted(here))
        for move in plan.items.values():
            if move.old_file == old_file and move.dest and \
                    commit_le(move.commit, commit):
                mods.add(dotted(plan.module_path(move.dest, commit)))
        for split in plan.classes.values():
            if split.old_file != old_file:
                continue
            for (file, _name), spec in split.mixins.items():
                if commit_le(spec.get('commit'), commit):
                    mods.add(dotted(plan.module_path(file, commit)))
        mods.discard(old)
        if len(mods) > 1:
            out[old] = tuple(sorted(mods))
    return out


def compat_moves_text(plan, commit):
    aliases, name_moves, plugin_api, submodules, forwarders = \
        compat_tables(plan, commit)
    fanout = patch_fanout(plan, commit, set(name_moves))
    out = [
        '"""Generated by tools/refactor (do not edit): where the names of '
        'moved',
        'Cell-ACDC modules live now (state after commit ' + commit + ').',
        '',
        'Read by cellacdc/_compat.py. Look up a name with',
        "python -c \"from cellacdc._compat_moves import NAME_MOVES as M; "
        "print(M['cellacdc.widgets']['QDialogListbox'])\"",
        '"""',
        '',
        '# Old module -> module at its new path (1:1 moves, sys.modules '
        'aliases)',
        'MODULE_ALIASES = ' + _fmt_dict(aliases),
        '',
        '# Module -> {name: (home module, attribute)}; attribute None: the '
        'name is',
        '# the home module itself',
        'NAME_MOVES = ' + _fmt_nested(name_moves),
        '',
        '# Names forwarded without a warning (documented plugin API)',
        'PLUGIN_API = ' + _fmt_plugin(plugin_api),
        '',
        '# Packages that replace an old module: their submodules (never '
        'forwarded)',
        'PACKAGE_SUBMODULES = ' + _fmt_dict(submodules),
        '',
        '# Modules that forward the old names of another module',
        'FORWARDERS = ' + _fmt_dict(forwarders),
        '',
        '# Old module -> modules holding its code now: a patch of an old',
        '# global also replaces the same object bound there (_compat.py)',
        'PATCH_FANOUT = ' + _fmt_dict(fanout),
    ]
    return '\n'.join(out) + '\n'


def _fmt_dict(d):
    if not d:
        return '{}'
    lines = ['{']
    for key in sorted(d):
        lines.append(f'    {key!r}: {d[key]!r},')
    lines.append('}')
    return '\n'.join(lines)


def _fmt_nested(d):
    if not d:
        return '{}'
    lines = ['{']
    for module in sorted(d):
        lines.append(f'    {module!r}: {{')
        for name in sorted(d[module]):
            lines.append(f'        {name!r}: {d[module][name]!r},')
        lines.append('    },')
    lines.append('}')
    return '\n'.join(lines)


def _fmt_plugin(d):
    if not d:
        return '{}'
    lines = ['{']
    for module in sorted(d):
        names = pprint.pformat(sorted(d[module]), width=70, compact=True)
        names = names.replace('\n', '\n        ')
        lines.append(f'    {module!r}: frozenset(')
        lines.append(f'        {names}),')
    lines.append('}')
    return '\n'.join(lines)


def apply_commit_shims(mover):
    """Shims, package compat, compat blocks and _compat_moves.py for the
    commit of `mover` (after its module and item moves)."""
    plan, commit, ws, log = mover.plan, mover.commit, mover.ws, mover.log
    repo = mover.repo
    eol = mover._default_eol()
    table = plan.name_table(commit)
    handled = set()
    # explicit shim specs and module moves of this commit
    specs = {}
    for old, spec in plan.shims.items():
        if spec['commit'] == commit:
            specs[old] = spec['kind']
    for move in plan.moves_for_commit(commit):
        if move.is_python_file and move.old not in specs:
            specs[move.old] = move.shim
    # modules emptied by this commit's item moves
    for move in plan.items_for_commit(commit):
        old = move.old_file
        if old in specs or plan.module_move_of(old):
            continue
        if not ws.exists(old):
            continue
        if plan.shim_kind(old, commit) == 'real' and \
                old in plan.fully_split_modules():
            specs[old] = 'split'
    for old, kind in sorted(specs.items()):
        handled.add(old)
        if plan.replaced_by_package(old, commit):
            if ws.exists(old):
                _check_empty(ws, old)
                ws.delete(old)
                log.add('other', 'module-removed', old,
                        'replaced by its package')
            continue
        if kind == 'none':
            continue
        if kind == 'package':
            continue
        if kind == 'alias':
            new = plan.module_path(old, commit)
            if new == old:
                raise RefactorError(f'Alias shim for {old}, which did not '
                                    'move')
            if ws.exists(old):
                raise RefactorError(f'Alias shim for {old}: the file still '
                                    'exists')
            text = alias_shim_text(old, new)
        else:
            if ws.exists(old):
                _check_empty(ws, old)
            text = split_shim_text(old, defined_homes(plan, old, commit))
        ws.set(old, SourceText(text.replace('\n', eol), eol=eol))
        log.add('other', 'shim', old, kind)
    # compat blocks in modules that lost names
    entries_by_module = dict(table['runtime'])
    for module, entries in table['intermediate'].items():
        entries_by_module.setdefault(module, {}).update(entries)
    old_dotted = {dotted(p) for p in plan.old_module_files()}
    for module in sorted(entries_by_module):
        path = None
        for candidate in (module.replace('.', '/') + '.py',
                          module.replace('.', '/') + '/__init__.py'):
            if ws.exists(candidate):
                path = candidate
                break
        if path is None or path in handled:
            continue
        src = ws.read(path)
        if has_compat_block(src.text):
            continue
        intermediate = module not in old_dotted
        _append_block(ws, path, src, compat_block_text(intermediate))
        log.add('other', 'compat-block', path,
                'intermediate' if intermediate else 'old path')
    # forwarders: modules that shadow an old attribute path
    forwards = {p: s.get('forward') for p, s in plan.modules.items()
                if s.get('forward')}
    for path, spec in plan.forwarders.items():
        if commit_le(spec['commit'], commit):
            forwards[path] = spec['table']
    for path, forward in sorted(forwards.items()):
        if not ws.exists(path):
            continue
        src = ws.read(path)
        if 'forward_module(' in src.text:
            continue
        _append_block(ws, path, src, forward_block_text(dotted(forward)))
        log.add('other', 'forwarder', path, dotted(forward))
    ws.flush()
    if os.path.exists(os.path.join(repo, 'cellacdc', '_compat.py')):
        write_compat_moves(repo, plan, commit)
        log.add('other', 'compat-moves', 'cellacdc/_compat_moves.py',
                commit)


def defined_homes(plan, old, commit):
    """Paths of the modules that hold the definitions of the base module
    `old` at `commit`."""
    if not plan.base.exists(old):
        return set()
    bmod = plan.base_module(old)
    homes = set()
    for item in bmod.defined.values():
        loc = plan.item_location(old, item.key, commit)
        if loc is not None and loc != old:
            homes.add(loc)
    return homes


def _check_empty(ws, path):
    left = _only_imports_left(ws.read(path))
    if left:
        raise RefactorError(
            f'{path} becomes a shim but still defines code:\n  '
            + '\n  '.join(left[:20]))


def _append_block(ws, path, src, block):
    lines = list(src.lines)
    tree = ast.parse(src.text)
    insert_at = len(lines)
    if tree.body:
        last = tree.body[-1]
        if isinstance(last, ast.If) and isinstance(last.test, ast.Compare) \
                and isinstance(last.test.left, ast.Name) and \
                last.test.left.id == '__name__':
            insert_at = last.lineno - 1
            while insert_at > 0 and is_blank(lines[insert_at - 1]):
                insert_at -= 1
    block_lines = [line + src.eol for line in block.rstrip('\n').split(
        '\n')]
    if insert_at < len(lines):
        block_lines += [src.eol]
    elif lines and not lines[-1].endswith(('\n', '\r')):
        lines[-1] += src.eol
    new = lines[:insert_at] + block_lines + lines[insert_at:]
    ws.set(path, src.with_text(''.join(new)))


def remove_intermediate_blocks(repo, paths=None, log=None):
    """Remove the intermediate compat blocks (C12, after rewrite.py)."""
    removed = []
    root = os.path.join(repo, 'cellacdc')
    for dirpath, _, files in os.walk(root):
        for name in files:
            if not name.endswith('.py'):
                continue
            full = os.path.join(dirpath, name)
            src = SourceText.read(full)
            if INTERMEDIATE_MARKER not in src.text:
                continue
            lines = src.lines
            out = []
            skip = False
            for line in lines:
                if strip_eol(line) == INTERMEDIATE_MARKER:
                    skip = True
                    # drop the blank lines before the block
                    while out and is_blank(out[-1]):
                        out.pop()
                    continue
                if skip:
                    if strip_eol(line).startswith('__getattr__, __dir__'):
                        skip = False
                    continue
                out.append(line)
            src.with_text(''.join(out)).write(full)
            rel = os.path.relpath(full, repo).replace('\\', '/')
            removed.append(rel)
            if log is not None:
                log.add('other', 'compat-block-removed', rel)
    return removed


def write_compat_module(repo):
    """Write cellacdc/_compat.py from the template (commit C2)."""
    with open(TEMPLATE, 'rb') as file:
        template = SourceText.from_bytes(file.read())
    init = os.path.join(repo, 'cellacdc', '__init__.py')
    eol = SourceText.read(init).eol if os.path.exists(init) else '\n'
    text = template.text.replace('\r\n', '\n').replace('\n', eol)
    path = os.path.join(repo, 'cellacdc', '_compat.py')
    SourceText(text, eol=eol).write(path)
    return path


def tool_out_dir(commit):
    return os.path.join(TOOL_DIR, 'out', commit)
