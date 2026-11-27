"""Unit tests of the codemod (acdc_refactor: source, items, scopes, imports,
plan, move, shims, plan_check, the _compat template and run.py).

Run without the repository conftest (these tests build synthetic packages
in temporary git repositories, see refactor_testlib.py):

    python -m pytest tools/refactor/tests/test_codemod.py -q
        -p no:cacheprovider --rootdir tools/refactor --noconftest
"""
import ast
import json
import os
import subprocess
import sys
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import refactor_testlib as T  # noqa: E402
from acdc_refactor import imports, items, scopes, shims, source  # noqa: E402
from acdc_refactor import move, plan_check  # noqa: E402
from acdc_refactor.plan import Plan  # noqa: E402

TOOL_DIR = T.TOOL_DIR


def parse_items(text):
    src = source.SourceText(textwrap.dedent(text).lstrip('\n'))
    tree = ast.parse(src.text)
    return src, items.module_items(src, tree)


# ----------------------------------------------------------------- source
@pytest.mark.parametrize('data', [
    b'a = 1\r\nb = 2\r\n',
    b'a = 1\nb = 2',
    b'\xef\xbb\xbfx = "\xc3\xa9"\r\n',
    b'# -*- coding: latin-1 -*-\nx = "\xe9"\n',
])
def test_source_roundtrip_is_byte_exact(data):
    src = source.SourceText.from_bytes(data)
    assert src.to_bytes() == data
    assert ''.join(src.lines) == src.text


def test_split_lines_only_on_python_line_ends():
    text = 'a = "\x0c"\r\nb = " "\nc'
    assert source.split_lines(text) == ['a = "\x0c"\r\n', 'b = " "\n',
                                        'c']


def test_file_edit_splices_and_line_ops():
    src = source.SourceText('a = X\r\nb = 2\r\nc = 3\r\n')
    edit = source.FileEdit(src)
    edit.splice(1, 4, 4, 'm.')
    edit.delete_lines(2, 2)
    edit.insert_before(4, ['d = 4\r\n'])
    assert edit.render() == 'a = m.X\r\nc = 3\r\nd = 4\r\n'
    assert edit.render_range(1, 1) == ['a = m.X\r\n']


def test_char_col_converts_utf8_byte_offsets():
    line = 'x = "é"; y'
    byte_col = len('x = "é"; '.encode('utf-8'))
    assert source.char_col(line, byte_col) == len('x = "é"; ')


# ------------------------------------------------------------------ items
def test_item_keys():
    _, its = parse_items('''
        """Doc."""
        import os
        try:
            import tk
        except Exception:
            pass
        X = 1
        A, B = 1, 2
        @deco
        def f():
            pass
        class C:
            pass
        f()
        f()
        X += 1
        def f():
            pass
        if __name__ == '__main__':
            f()
    ''')
    keys = [i.key for i in its]
    assert keys == [
        '__doc__', 'import:import os', 'stmt:try:', 'X', 'A', 'f', 'C',
        'stmt:f()', 'stmt:f() #2', 'stmt:X += 1', 'f #2',
        "stmt:if __name__ == '__main__':",
    ]
    kinds = {i.key: i.kind for i in its}
    assert kinds['stmt:try:'] == 'importblock'
    assert [i.names for i in its if i.key == 'A'] == [['A', 'B']]
    assert items.find_module_item(its, '__doc__').kind == 'docstring'


def test_item_spans_take_comments():
    src, its = parse_items('''
        import os

        # Section comment
        # (two lines)
        @deco
        def f():
            return 1
            # commented-out tail

            # more tail


        # Comment of g, after blank lines

        def g():
            pass
        # comment at module level after g
        X = 1
    ''')
    f = items.find_module_item(its, 'f')
    assert src.segment(f.first, f.last).startswith('# Section comment')
    assert src.segment(f.first, f.last).rstrip().endswith('# more tail')
    assert f.gap_before == 1
    g = items.find_module_item(its, 'g')
    assert src.line(g.first).startswith('# Comment of g')
    x = items.find_module_item(its, 'X')
    assert src.line(x.first).startswith('# comment at module level')


def test_import_block_detection():
    tree = ast.parse(textwrap.dedent('''
        if GUI:
            from a import b
        try:
            import c
        except ImportError:
            pass
        try:
            import d
            D = True
        except ImportError:
            D = False
    '''))
    flags = [items.is_import_block(n) for n in tree.body]
    assert flags == [True, True, False]


def test_class_items_methods_and_statements():
    src = source.SourceText(textwrap.dedent('''
        class C(
                Base,
            ):
            """Doc."""
            sig = 1

            # about m
            def m(self):
                pass

            @property
            def p(self):
                return 1

            @p.setter
            def p(self, v):
                pass
    ''').lstrip('\n'))
    tree = ast.parse(src.text)
    members = items.class_items(src, tree.body[0])
    assert [(m.kind, m.key) for m in members] == [
        ('classstmt', 'stmt:"""Doc."""'), ('classstmt', 'stmt:sig = 1'),
        ('method', 'm'), ('method', 'p'), ('method', 'p')]
    m = members[2]
    assert src.line(m.first).strip() == '# about m'
    assert items.class_header_end(src, tree.body[0]) == 3


# ----------------------------------------------------------------- scopes
def test_global_refs_are_scope_aware():
    text = textwrap.dedent('''
        import os
        G = 1
        @deco(G)
        def f(a, b=DEFAULT, *, c: ANN = 2) -> RET:
            local = a
            x = [local + y for y in os.listdir(W)]
            def inner():
                return a + V
            return x, local, Y
        class K(Base, metaclass=Meta):
            attr = CLASSVAL
            other = attr
            def m(self):
                return attr, self.__secret, super().m()
    ''')
    ma = scopes.ModuleAnalysis(text)
    names = {r.name for r in ma.refs}
    assert {'deco', 'G', 'DEFAULT', 'ANN', 'RET', 'os', 'W', 'V', 'Y',
            'Base', 'Meta', 'CLASSVAL', 'attr', 'super'} <= names
    assert not names & {'a', 'b', 'c', 'local', 'x', 'y', 'inner', 'self'}
    eager = {r.name for r in ma.refs if r.eager}
    assert {'deco', 'DEFAULT', 'ANN', 'RET', 'Base', 'Meta',
            'CLASSVAL'} <= eager
    assert 'W' not in eager
    kinds = {(h.kind, h.name) for h in ma.hazards}
    assert ('super0', 'super') in kinds and ('mangled', '__secret') in kinds


def test_future_annotations_are_not_eager():
    ma = scopes.ModuleAnalysis(
        'from __future__ import annotations\n'
        'def f(a: ANN) -> RET:\n    pass\n')
    refs = {r.name: r for r in ma.refs}
    # still needed by pyflakes (F821), never evaluated at import time
    assert {'ANN', 'RET'} <= set(refs)
    assert not refs['ANN'].eager and not refs['RET'].eager


def test_string_annotations_are_located():
    text = ("from typing import Literal\n"
            "def f(a: 'load.loadData', c: Literal['row']) -> 'Tuple[x.y]':\n"
            "    v: 'Inner' = 1\n")
    ma = scopes.ModuleAnalysis(text)
    refs = {r.name: r for r in ma.refs if r.string}
    assert set(refs) == {'load', 'Tuple', 'x', 'Inner'}
    line = text.splitlines()[1]
    assert line[refs['load'].col:refs['load'].end_col] == 'load'
    assert line[refs['x'].col:refs['x'].end_col] == 'x'
    assert not any(r.eager for r in refs.values())


def test_is_shadowed():
    text = textwrap.dedent('''
        def f(mod):
            return X
        def g():
            mod = 1
            def h():
                return X
            return h
        def k():
            return X
    ''')
    ma = scopes.ModuleAnalysis(text)
    refs = {r.lineno: r for r in ma.refs if r.name == 'X'}
    assert scopes.is_shadowed(refs[3], 'mod')
    assert scopes.is_shadowed(refs[7], 'mod')
    assert not scopes.is_shadowed(refs[10], 'mod')


# ---------------------------------------------------------------- imports
def _remap(moves):
    return lambda dotted: moves.get(dotted, dotted)


def test_relative_and_absolute_expressions():
    imports.set_part_packages(['gui', 'shared'])
    ctx = imports.ModuleCtx('cellacdc.gui.mixins.navigation')
    assert imports.resolve_relative(
        imports.ModuleCtx('cellacdc.gui_nav'), 1, None) == 'cellacdc'
    assert imports.from_expr(ctx, 'cellacdc') == 'cellacdc'
    assert imports.from_expr(ctx, 'cellacdc.gui.main_window') == \
        '..main_window'
    assert imports.from_expr(ctx, 'cellacdc.shared.widgets') == \
        'cellacdc.shared.widgets'
    root = imports.ModuleCtx('cellacdc.load')
    assert imports.from_expr(root, 'cellacdc') == '.'


def test_plan_importfrom_keeps_meaning_after_a_move():
    imports.set_part_packages(['gui'])
    old = imports.ModuleCtx('cellacdc.gui_nav')
    new = imports.ModuleCtx('cellacdc.gui.mixins.navigation')
    is_module = {'cellacdc.widgets', 'cellacdc.gui_overlay'}.__contains__
    remap = _remap({'cellacdc.gui_overlay': 'cellacdc.gui.mixins.overlay'})
    node = ast.parse('from . import widgets').body[0]
    assert imports.plan_importfrom(node, old, new, remap, is_module) == \
        ('prefix', 'cellacdc')
    node = ast.parse('from . import gui_overlay, printl').body[0]
    plan = imports.plan_importfrom(node, old, new, remap, is_module)
    assert plan == ('groups', [('.', [('overlay', 'gui_overlay')]),
                               ('cellacdc', [('printl', None)])])
    node = ast.parse('from .gui_overlay import X').body[0]
    assert imports.plan_importfrom(node, old, new, remap, is_module) == \
        ('prefix', '.overlay')
    node = ast.parse('from cellacdc.load import X').body[0]
    assert imports.plan_importfrom(node, old, new, remap, is_module) is None


def test_render_import_block_keeps_needed_imports_and_guards():
    src = source.SourceText(textwrap.dedent('''
        if GUI_INSTALLED:
            from qtpy.QtWidgets import QA, QB
            from . import widgets
        else:
            import nothing
    ''').lstrip('\n'))
    block = ast.parse(src.text).body[0]
    keep = {id(block.body[0]): {'QB'}}
    ctx = imports.ModuleCtx('cellacdc.myutils')
    new = imports.ModuleCtx('cellacdc.myutils.installs')
    lines = imports.render_import_block(
        src, block, keep, ctx, new, lambda d: d, lambda d: False)
    assert lines == ['if GUI_INSTALLED:',
                     '    from qtpy.QtWidgets import QB',
                     'else:', '    pass']


def test_format_from_wraps_long_imports():
    names = [(f'Name{i:02d}', None) for i in range(12)]
    lines = imports.format_from('qtpy.QtWidgets', names)
    assert lines[0] == 'from qtpy.QtWidgets import (' and lines[-1] == ')'
    assert all(len(line) <= 79 for line in lines)
    ast.parse('\n'.join(lines))


# ------------------------------------------------------------ pipeline
def _apply(repo, sha, commits):
    plan = Plan(repo, mapping_dir=os.path.join(repo, 'mapping'), base=sha)
    assert not [i for i in plan.issues if i.level == 'error'], plan.issues
    shims.write_compat_module(repo)
    shims.write_compat_moves(repo, plan, 'C2')
    logs = {}
    for commit in commits:
        logs[commit] = move.apply_commit(
            repo, plan, commit, out_dir=os.path.join(repo, 'out', commit))
    return plan, logs


RUNTIME_CHECK = '''
import warnings
from cellacdc.gui.main_window.window import Win
w = Win()
assert w.other() == 16 and w.show() == 'win-show', (w.other(), w.show())
assert w.nav()[0] == 2
import cellacdc.gui_nav as g
from cellacdc.gui.mixins import navigation
assert g is navigation
from cellacdc import gui, widgets
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    assert gui.Win is Win and gui.CONST_A == 3
    assert widgets.okPushButton().text().startswith('ok:b')
msgs = [str(c.message) for c in caught]
assert all(m.startswith('Cell-ACDC import path moved: ') for m in msgs), msgs
assert {c.filename for c in caught} == {'<string>'}, caught
import unittest.mock as mock
from cellacdc.shared.widgets import style, basic
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    with mock.patch('cellacdc.widgets.STYLE', 'blue'):
        assert style.STYLE == 'blue' and basic.helper('x').endswith('blue')
    assert style.STYLE == 'red'
    assert not hasattr(widgets, 'nope') and not hasattr(widgets, '__path__')
    assert 'tk' not in widgets.__all__
    assert 'PushButton' in dir(widgets)
import fakepg
assert fakepg.CONFIG == {'imageAxisOrder': 'row-major'}
print('RUNTIME OK')
'''


@pytest.mark.parametrize('crlf', [False, True], ids=['lf', 'crlf'])
def test_pipeline_on_synthetic_package(tmp_path, crlf):
    repo = str(tmp_path / 'repo')
    sha = T.make_repo(repo, crlf=crlf)
    plan, logs = _apply(repo, sha, ['C3', 'C5', 'C10'])
    eol = '\r\n' if crlf else '\n'
    # git mv kept the history of the moved files
    status = T.git(repo, 'status', '--porcelain')
    assert 'R  cellacdc/gui_nav.py -> cellacdc/gui/mixins/navigation.py' \
        in status or 'RM cellacdc/gui_nav.py -> ' \
        'cellacdc/gui/mixins/navigation.py' in status
    # moved text is byte-exact apart from the qualification
    basic = T.read(repo, 'cellacdc/shared/widgets/basic.py')
    expected = ('# A helper used by the buttons' + eol
                + '# (two comment lines)' + eol
                + 'def helper(x):' + eol
                + '    return os.path.join(x, widgets_style.STYLE)' + eol)
    assert expected in basic
    assert 'from . import style as widgets_style' in basic
    assert basic.count('\r\n') == (basic.count('\n') if crlf else 0)
    buttons = T.read(repo, 'cellacdc/shared/widgets/push_buttons.py')
    assert "        # return 'old'" in buttons       # trailing comment kept
    assert 'widgets_basic.helper' in buttons
    # nested relative import retargeted, guarded import deleted
    assert 'from cellacdc import config_mod as cm' in basic
    style = T.read(repo, 'cellacdc/shared/widgets/style.py')
    assert 'tkinter_missing_xyz' not in style + basic
    # shims
    assert '__cellacdc_compat__ = True' in T.read(repo, 'cellacdc/widgets.py')
    assert 'alias_module' in T.read(repo, 'cellacdc/gui_nav.py')
    assert 'package_compat' in T.read(repo, 'cellacdc/gui/__init__.py')
    # mixin split
    layout = T.read(repo, 'cellacdc/gui/main_window/layout.py')
    assert 'class LayoutGuiElements:' in layout
    assert '    # Layout helpers' + eol + '    def layout_stuff' in layout
    window = T.read(repo, 'cellacdc/gui/main_window/window.py')
    assert 'def show(self)' in window and 'def layout_stuff' not in window
    assert 'self.x = gui_constants.CONST_A' in window
    assert 'from .layout import LayoutGuiElements' in window
    # every edit is logged
    cats = {e['category'] for e in logs['C10'].entries}
    assert {'move', 'qualify', 'import'} <= cats
    proc = T.run_python(repo, RUNTIME_CHECK)
    assert proc.returncode == 0, proc.stderr
    assert 'RUNTIME OK' in proc.stdout

FANOUT_CHECK = '''
import warnings
from unittest import mock
warnings.simplefilter('ignore', DeprecationWarning)
import cellacdc.gui as old_gui
from cellacdc.gui.main_window import window, layout
from cellacdc._compat_moves import PATCH_FANOUT
assert 'cellacdc.gui.main_window.layout' in PATCH_FANOUT['cellacdc.gui']
# VALUE (imported by the old gui.py) is now bound in window.py and, for
# layout_stuff, in layout.py: a patch of the old path reaches both
win = window.Win()
assert win.layout_stuff() == 3 + 7
with mock.patch.object(old_gui, 'VALUE', 100):
    assert layout.VALUE == window.VALUE == 100
    assert win.layout_stuff() == 3 + 100
assert layout.VALUE == window.VALUE == 7, (layout.VALUE, window.VALUE)
old_gui.VALUE = 50
assert win.layout_stuff() == 53
old_gui.VALUE = 7
assert win.layout_stuff() == 10
print('FANOUT OK')
'''


def test_patches_of_old_globals_reach_every_copy(tmp_path):
    repo = str(tmp_path / 'repo')
    sha = T.make_repo(repo)
    plan, logs = _apply(repo, sha, ['C3', 'C5', 'C10'])
    assert 'from cellacdc.config_mod import VALUE' in T.read(
        repo, 'cellacdc/gui/main_window/layout.py')
    proc = T.run_python(repo, FANOUT_CHECK)
    assert 'FANOUT OK' in proc.stdout, proc.stdout + proc.stderr


MERGE_FILES = {
    'cellacdc/__init__.py': """
        NAME = 'pkg'
        """,
    'cellacdc/old_a.py': """
        \"\"\"Specification of A.

        Second paragraph.
        \"\"\"
        STYLE = 'same'


        def fa():
            return STYLE
        """,
    'cellacdc/old_b.py': """
        STYLE = 'same'


        def fb():
            return STYLE
        """,
    'cellacdc/dest.py': """
        \"\"\"Where the code lives.
        \"\"\"
        import os
        """,
}
MERGE_SHARDS = {
    'items.json': {
        'shard': 'items', 'default_commit': 'C5',
        'items': {
            'cellacdc/old_a.py::__doc__': 'cellacdc/dest.py',
            'cellacdc/old_a.py::STYLE': 'cellacdc/dest.py',
            'cellacdc/old_b.py::STYLE': 'cellacdc/dest.py',
        },
    },
}


def test_identical_bindings_and_docstrings_are_merged(tmp_path):
    """Two old modules with byte-identical copies of a constant mapped to
    one home give one binding; a moved module docstring is merged into
    the destination's own docstring (moved text first)."""
    repo = str(tmp_path / 'repo')
    sha = T.make_repo(repo, files=MERGE_FILES, shards=MERGE_SHARDS)
    plan, logs = _apply(repo, sha, ['C5'])
    dest = T.read(repo, 'cellacdc/dest.py')
    assert dest.count("STYLE = 'same'") == 1, dest
    assert ast.get_docstring(ast.parse(dest), clean=False) == (
        'Specification of A.\n\nSecond paragraph.\n\nWhere the code '
        'lives.\n'), dest
    kinds = [e['kind'] for e in logs['C5'].entries]
    assert kinds.count('merged-duplicate') == 1
    assert 'docstring-merged' in kinds
    for old in ('old_a', 'old_b'):
        text = T.read(repo, f'cellacdc/{old}.py')
        assert 'dest.STYLE' in text and "STYLE = 'same'" not in text
    proc = T.run_python(repo, """
        from cellacdc import old_a, old_b, dest
        assert old_a.fa() == old_b.fb() == dest.STYLE == 'same'
        print('MERGE OK')
        """)
    assert 'MERGE OK' in proc.stdout, proc.stderr

def test_name_table_and_compat_moves(tmp_path):
    repo = str(tmp_path / 'repo')
    sha = T.make_repo(repo)
    plan = Plan(repo, mapping_dir=os.path.join(repo, 'mapping'), base=sha)
    table = plan.name_table('C5')
    assert table['modules'] == {
        'cellacdc.gui': 'cellacdc.gui.main_window.window',
        'cellacdc.gui_nav': 'cellacdc.gui.mixins.navigation'}
    names = table['names']['cellacdc.widgets']
    assert names['PushButton'] == 'cellacdc.shared.widgets.push_buttons'
    assert names['printl'] == 'cellacdc'           # defining module
    assert table['module_values']['cellacdc.widgets']['config_mod'] == \
        'cellacdc.config_mod'
    assert 'tk' not in names                        # deleted import block
    assert table['names']['cellacdc.gui']['Win'] == \
        'cellacdc.gui.main_window.window'
    assert table['aliases']['cellacdc.shared.widgets.basic'] == \
        'widgets_basic'
    final = plan.name_table()
    assert final['names']['cellacdc.gui']['CONST_A'] == \
        'cellacdc.gui.main_window.constants'
    assert final['intermediate']['cellacdc.gui.main_window.window'][
        'CONST_A'][0] == 'cellacdc.gui.main_window.constants'
    text = shims.compat_moves_text(plan, 'C5')
    namespace = {}
    exec(compile(text, '_compat_moves.py', 'exec'), namespace)
    assert namespace['MODULE_ALIASES'] == {
        'cellacdc.gui_nav': 'cellacdc.gui.mixins.navigation'}
    assert namespace['NAME_MOVES']['cellacdc.widgets']['json'] == (
        'cellacdc.shared.widgets.basic', 'json')     # through its binder
    assert 'mixins' in namespace['PACKAGE_SUBMODULES']['cellacdc.gui']


def test_plan_check_passes_on_the_synthetic_mapping(tmp_path):
    repo = str(tmp_path / 'repo')
    sha = T.make_repo(repo)
    plan = Plan(repo, mapping_dir=os.path.join(repo, 'mapping'), base=sha)
    issues = plan_check.Checker(plan).run()
    errors = [str(i) for i in issues if i.level == 'error']
    assert errors == []


def test_plan_check_finds_hazards(tmp_path):
    files = dict(T.FILES)
    files['cellacdc/gui.py'] = T.FILES['cellacdc/gui.py'] + '''
            def uses_super(self):
                return super().keep()

            def mangles(self):
                return self.__hidden
    '''
    shards = json.loads(json.dumps(T.SHARDS))
    split = shards['gui.json']['classes']['cellacdc/gui.py::Win']
    for meth in ('show', 'uses_super', 'mangles'):
        split['methods'][meth] = ('cellacdc/gui/main_window/layout.py::'
                                  'LayoutGuiElements')
    del shards['widgets.json']['items']['cellacdc/widgets.py::make_buttons']
    shards['widgets.json']['modules'][
        'cellacdc/shared/widgets/style.py']['alias'] = 'x'
    repo = str(tmp_path / 'repo')
    sha = T.make_repo(repo, files=files, shards=shards)
    plan = Plan(repo, mapping_dir=os.path.join(repo, 'mapping'), base=sha)
    issues = plan_check.Checker(plan).run()
    text = '\n'.join(str(i) for i in issues if i.level == 'error')
    assert 'show is a QMainWindow attribute' in text
    assert 'zero-argument super()' in text
    assert 'mangled name __hidden' in text
    assert 'coverage' in text and 'make_buttons' in text
    assert "'x' is a local name there" in text   # helper(x) uses STYLE


def test_plan_check_allow_downgrades_errors(tmp_path):
    shards = json.loads(json.dumps(T.SHARDS))
    shards['gui.json']['classes']['cellacdc/gui.py::Win']['methods'][
        'show'] = 'cellacdc/gui/main_window/layout.py::LayoutGuiElements'
    shards['gui.json']['allow'] = [{'check': 'mixins', 'match': 'show',
                                    'reason': 'test'}]
    repo = str(tmp_path / 'repo')
    sha = T.make_repo(repo, shards=shards)
    plan = Plan(repo, mapping_dir=os.path.join(repo, 'mapping'), base=sha)
    issues = plan_check.Checker(plan).run()
    assert not [i for i in issues if i.level == 'error']
    assert any('allowed: test' in i.message for i in issues)


def test_eager_cycle_is_reported(tmp_path):
    shards = json.loads(json.dumps(T.SHARDS))
    # module_helper stays in window.py: layout.py (a base of Win, imported
    # by window.py at import time) then imports window.py
    del shards['gui.json']['items']['cellacdc/gui.py::module_helper']
    repo = str(tmp_path / 'repo')
    sha = T.make_repo(repo, shards=shards)
    plan = Plan(repo, mapping_dir=os.path.join(repo, 'mapping'), base=sha)
    checker = plan_check.Checker(plan)
    checker.check_cycles()
    text = '\n'.join(str(i) for i in checker.issues)
    assert 'cellacdc.gui.main_window.window uses ' \
        'cellacdc.gui.main_window.layout at import time' in text


def test_run_py_replay(tmp_path):
    repo = str(tmp_path / 'repo')
    sha = T.make_repo(repo)
    worktree = str(tmp_path / 'wt')
    fixups = tmp_path / 'fixups'
    fixups.mkdir()
    env = dict(os.environ, ACDC_REFACTOR_OUT=str(tmp_path / 'out'),
               ACDC_REFACTOR_FIXUPS=str(fixups))
    proc = subprocess.run(
        [sys.executable, os.path.join(TOOL_DIR, 'run.py'), '--repo', repo,
         '--mapping', os.path.join(repo, 'mapping'), 'replay', '--base', sha,
         '--until', 'C3', '--branch', 'replayed', '--worktree', worktree],
        capture_output=True, text=True, env=env, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    log = T.git(worktree, 'log', '--format=%s', f'{sha}..HEAD').splitlines()
    assert len(log) == 3 and log[-1].startswith('tools:'), log
    assert os.path.exists(os.path.join(worktree, 'tools', 'refactor',
                                       'run.py'))
    assert T.git(worktree, 'status', '--porcelain') == ''
    with open(tmp_path / 'out' / 'replay_shas.json') as file:
        shas = json.load(file)['shas']
    assert sorted(shas) == ['C0', 'C2', 'C3']


def test_run_py_cli(tmp_path):
    repo = str(tmp_path / 'repo')
    sha = T.make_repo(repo)
    env = dict(os.environ, ACDC_REFACTOR_BASE=sha,
               ACDC_REFACTOR_OUT=str(tmp_path / 'out'))
    run_py = os.path.join(TOOL_DIR, 'run.py')
    for commit in ('C2', 'C3'):
        proc = subprocess.run(
            [sys.executable, run_py, '--repo', repo, '--mapping',
             os.path.join(repo, 'mapping'), 'apply', commit],
            capture_output=True, text=True, env=env, timeout=120)
        assert proc.returncode == 0, proc.stderr
    assert os.path.exists(os.path.join(repo, 'cellacdc', '_compat.py'))
    assert os.path.exists(os.path.join(repo, 'cellacdc', 'gui', 'main_window',
                                       'window.py'))
    proc = subprocess.run(
        [sys.executable, os.path.join(TOOL_DIR, 'mapping.py'), '--repo',
         repo, '--mapping', os.path.join(repo, 'mapping'), '--base', sha,
         'validate'], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr


# --------------------------------------------------------- compat template
COMPAT_MOVES = '''
MODULE_ALIASES = {'pkg.oldmod': 'pkg.sub.newmod'}
NAME_MOVES = {
    'pkg.split': {'Thing': ('pkg.home', 'Thing'), 'VALUE': ('pkg.home',
                  'VALUE'), 'mod': ('pkg.home', None), 'api': ('pkg.home',
                  'api')},
    'pkg.sub': {'Old': ('pkg.home', 'Thing')},
}
PLUGIN_API = {'pkg.split': frozenset({'api'})}
PACKAGE_SUBMODULES = {'pkg.sub': ('newmod',)}
FORWARDERS = {'pkg.fwd': 'pkg.split'}
'''

COMPAT_FILES = {
    'pkg/__init__.py': '',
    'pkg/_compat_moves.py': COMPAT_MOVES,
    'pkg/home.py': 'class Thing:\n    pass\nVALUE = 1\n\ndef api():\n'
                   '    return 42\n',
    'pkg/sub/__init__.py': 'from pkg import _compat\n'
                           '__getattr__, __dir__ = '
                           '_compat.package_compat(__name__)\n',
    'pkg/sub/newmod.py': 'X = 5\n',
    'pkg/oldmod.py': '__cellacdc_compat__ = True\nfrom pkg import _compat\n'
                     '_compat.alias_module(__name__)\n',
    'pkg/split.py': '__cellacdc_compat__ = True\nfrom pkg import _compat\n'
                    '__getattr__, __dir__ = _compat.moved_module(__name__)\n',
    'pkg/fwd.py': 'OWN = 1\nfrom pkg import _compat\n'
                  '__getattr__, __dir__ = _compat.moved_module(__name__)\n',
}

COMPAT_CHECK = '''
import warnings, sys
warnings.simplefilter('error')
import pkg.home as home
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    import pkg.oldmod as old
    import pkg.sub.newmod as new
    assert old is new and sys.modules['pkg.oldmod'] is new
    from pkg import split
    assert split.Thing is home.Thing and split.mod is home
    assert split.api() == 42
    from pkg.split import VALUE
    assert VALUE == 1
msgs = [str(x.message) for x in w]
assert msgs[0] == ('Cell-ACDC import path moved: pkg.oldmod -> '
                   'pkg.sub.newmod (see CODEMAP.md)'), msgs
assert not any('split.api' in m for m in msgs), msgs      # plugin API
assert all(x.filename == '<string>' for x in w), [x.filename for x in w]
for name in ('nope', '__path__', '__wrapped__'):
    try:
        getattr(split, name)
    except AttributeError:
        pass
    else:
        raise AssertionError(name)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    assert sorted(split.__all__) == ['Thing', 'VALUE', 'api', 'mod']
    assert 'Thing' in dir(split)
    split.VALUE = 7                       # forwarded to the home module
    assert home.VALUE == 7 and 'VALUE' not in vars(split)
    split.VALUE = 8
    del split.VALUE                       # undoes the last pending patch
    assert home.VALUE == 7
    del split.VALUE
    assert home.VALUE == 1
    del split.VALUE                       # no pending patch: deletes
    assert not hasattr(home, 'VALUE')
    split.VALUE = 1
    import unittest.mock as mock
    with mock.patch('pkg.split.VALUE', 9):
        assert home.VALUE == 9
    assert home.VALUE == 1
    split.local_only = 3                  # unknown names stay local
    assert vars(split)['local_only'] == 3
    import pkg.sub as sub
    assert sub.newmod.X == 5              # submodule imported on demand
    assert sub.Old is home.Thing
    import pkg.fwd as fwd
    assert fwd.OWN == 1 and fwd.Thing is home.Thing
try:
    split.Thing
except DeprecationWarning as err:
    assert 'pkg.split.Thing -> pkg.home.Thing' in str(err)
else:
    raise AssertionError('no warning')
print('COMPAT OK')
'''


def test_compat_template(tmp_path):
    root = str(tmp_path)
    for path, text in COMPAT_FILES.items():
        full = os.path.join(root, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, 'w', encoding='utf-8') as file:
            file.write(text)
    template = os.path.join(TOOL_DIR, 'acdc_refactor', 'templates',
                            '_compat.py')
    with open(template, encoding='utf-8') as file:
        text = file.read()
    with open(os.path.join(root, 'pkg', '_compat.py'), 'w',
              encoding='utf-8') as file:
        file.write(text)
    proc = T.run_python(root, COMPAT_CHECK)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert 'COMPAT OK' in proc.stdout


# ------------------------------------------------------------- python 3.10
def test_tool_files_parse_on_python_310():
    for dirpath, _, files in os.walk(TOOL_DIR):
        if 'out' in dirpath.split(os.sep) or 'rewrite_fixtures' in dirpath:
            continue
        for name in files:
            if name.endswith('.py'):
                path = os.path.join(dirpath, name)
                with open(path, encoding='utf-8') as file:
                    ast.parse(file.read(), path, feature_version=(3, 10))
