"""Apply the moves of one commit id to the working tree.

Steps (all in one process, files written at the end of each step):

1. module moves: `git mv` of files and folders, new package __init__ files,
   then the relative imports of the moved files are recomputed (nested
   imports in functions included);
2. item moves and class splits: items are cut byte-exact (with their
   comments) from their current source module and appended to their
   destination module, in source order. The destination's imports are
   computed from the free global names of the moved code: import-bound
   names get a copy of the source's import statement (with its if/try
   guard), names defined in another module are qualified (`Name` ->
   `alias.Name`) and the module is imported under its canonical alias.
   References in the remaining source code to names that left are
   qualified the same way. Mixin classes are created as `class Name:`
   with a docstring; the host class gets its new bases;
3. shims and compat blocks (shims.py) and cellacdc/_compat_moves.py.

Every edit is logged to tools/refactor/out/<commit>/changes.jsonl with a
category: 'move' (text moved), 'import', 'qualify', 'depth' (allowed
edits) or 'other' (anything else; reviewed by hand and checked by the
verify scripts).
"""
import ast
import builtins
import json
import os
import re

from . import RefactorError, TOOL_DIR
from .gitutil import git_mv
from .imports import (
    path_to_ctx, import_bindings, rewrite_import_stmt,
    render_import_block, from_expr, set_part_packages, _module_expr_span,
    plan_importfrom, resolve_relative
)
from .items import (
    module_items, class_items, candidate_items, repeat_index,
    class_header_end
)
from .plan import dotted
from .scopes import ModuleAnalysis, is_shadowed
from .source import (
    SourceText, FileEdit, normalize_eol, strip_eol, char_col, is_blank,
    text_to_lines, ensure_eol
)
from . import shims as shims_mod
from .textutil import format_docstring

BUILTINS = set(dir(builtins)) | {'__file__', '__name__', '__doc__',
                                 '__spec__', '__loader__', '__package__',
                                 '__builtins__', '__path__',
                                 '__annotations__'}

#: Top-level statements that are copied after their import into every new
#: destination module that needs the import (idempotent configuration).
COMPANION_PATTERNS = [
    (re.compile(r'^pg\.setConfigOption\('), 'pg'),
]


class ChangeLog:
    def __init__(self, commit):
        self.commit = commit
        self.entries = []

    def add(self, category, kind, file, detail='', **extra):
        entry = {'commit': self.commit, 'category': category, 'kind': kind,
                 'file': file, 'detail': detail}
        entry.update(extra)
        self.entries.append(entry)

    def write(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'changes.jsonl')
        with open(path, 'w', encoding='utf-8') as file:
            for entry in self.entries:
                file.write(json.dumps(entry, sort_keys=True) + '\n')
        summary = {}
        for entry in self.entries:
            key = f"{entry['category']}:{entry['kind']}"
            summary[key] = summary.get(key, 0) + 1
        with open(os.path.join(out_dir, 'summary.json'), 'w',
                  encoding='utf-8') as file:
            json.dump(summary, file, indent=2, sort_keys=True)
        return path


class Workspace:
    """Working-tree files read once and written at the end."""

    def __init__(self, repo):
        self.repo = repo
        self.texts = {}
        self.dirty = set()
        self.deleted = set()

    def abspath(self, path):
        return os.path.join(self.repo, path)

    def exists(self, path):
        if path in self.deleted:
            return False
        return path in self.texts or os.path.exists(self.abspath(path))

    def read(self, path):
        if path not in self.texts:
            self.texts[path] = SourceText.read(self.abspath(path))
        return self.texts[path]

    def set(self, path, src):
        self.texts[path] = src
        self.dirty.add(path)
        self.deleted.discard(path)

    def delete(self, path):
        self.deleted.add(path)
        self.dirty.discard(path)
        self.texts.pop(path, None)

    def flush(self):
        for path in sorted(self.dirty):
            full = self.abspath(path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            self.texts[path].write(full)
        for path in sorted(self.deleted):
            full = self.abspath(path)
            if os.path.exists(full):
                os.remove(full)
        self.dirty.clear()
        self.deleted.clear()


# ---------------------------------------------------------------- units
class Unit:
    """A module being edited in this commit (source and/or destination)."""

    def __init__(self, ctx_mgr, path):
        self.mgr = ctx_mgr
        self.path = path
        self.ctx = path_to_ctx(path)
        self.exists = ctx_mgr.ws.exists(path)
        if self.exists:
            self.src = ctx_mgr.ws.read(path)
            try:
                self.tree = ast.parse(self.src.text, path)
            except SyntaxError as err:
                raise RefactorError(f'{path}: {err}')
            self.items = module_items(self.src, self.tree)
            self.edit = FileEdit(self.src, path)
            self._analysis = None
            block_ids = {id(i.node) for i in self.items
                         if i.kind == 'importblock'}
            self.bindings = import_bindings(
                self.tree, self.ctx, ctx_mgr.is_module, block_ids)
        else:
            self.src = None
            self.tree = None
            self.items = []
            self.edit = None
            self._analysis = None
            self.bindings = {}
        self.defined = {}
        for item in self.items:
            if item.kind in ('import', 'importblock', 'docstring'):
                continue
            for name in item.names:
                self.defined[name] = item
        # destination side
        self.incoming = []          # IncomingItem
        self.mixins = {}            # mixin name -> {'doc', 'methods': []}
        self.mixin_order = []
        self.import_lines = []      # generated import lines (no EOL)
        self.import_keys = set()
        self.alias_for_module = {}  # dotted module -> local name
        self.new_bindings = {}      # name -> description (added imports)
        self.leaving = []           # (item, dest path, kind)
        self.leaving_methods = []   # (class name, item, dest, mixin)
        self.long_line_check = set()

    @property
    def analysis(self):
        if self._analysis is None:
            self._analysis = ModuleAnalysis(self.src.text, self.path,
                                            tree=self.tree)
        return self._analysis

    def is_new(self):
        return not self.exists


class IncomingItem:
    def __init__(self, src_unit, item, lines, kind, mixin=None,
                 adjacent_prev=None):
        self.src_unit = src_unit
        self.item = item
        self.lines = lines          # rendered lines (with EOL)
        self.kind = kind            # 'item', 'import', 'method'
        self.mixin = mixin
        self.adjacent = False


class Mover:
    def __init__(self, repo, plan, commit, log=None):
        self.repo = repo
        self.plan = plan
        self.commit = commit
        self.ws = Workspace(repo)
        self.log = log or ChangeLog(commit)
        self.units = {}
        set_part_packages(self._part_packages())

    # ------------------------------------------------------- helpers
    def _part_packages(self):
        names = set()
        root = os.path.join(self.repo, 'cellacdc')
        if os.path.isdir(root):
            for name in os.listdir(root):
                if os.path.isfile(os.path.join(root, name, '__init__.py')):
                    names.add(name)
        for path in self.plan.packages:
            parts = path.split('/')
            if len(parts) >= 2 and parts[0] == 'cellacdc':
                names.add(parts[1])
        for move in self.plan.module_moves:
            parts = move.new.split('/')
            if len(parts) >= 3 and parts[0] == 'cellacdc':
                names.add(parts[1])
        return names

    def is_module(self, dotted_name):
        if self.plan.base_is_module(dotted_name):
            return True
        base = dotted_name.replace('.', '/')
        return (self.ws.exists(base + '.py')
                or self.ws.exists(base + '/__init__.py')
                or (base + '/__init__.py') in self.plan_new_inits())

    def plan_new_inits(self):
        return {p + '/__init__.py' for p in self.plan.packages}

    def remap(self, dotted_name):
        return self.plan.remap_module(dotted_name, self.commit)

    def unit(self, path):
        if path not in self.units:
            self.units[path] = Unit(self, path)
        return self.units[path]

    # ---------------------------------------------------------- step 1
    def apply_module_moves(self):
        moves = self.plan.moves_for_commit(self.commit)
        moved_files = []
        for move in moves:
            src = os.path.join(self.repo, move.old)
            if not os.path.exists(src):
                raise RefactorError(f'Cannot move {move.old}: missing')
            if os.path.exists(os.path.join(self.repo, move.new)):
                raise RefactorError(f'Cannot move {move.old}: {move.new} '
                                    'exists')
            git_mv(self.repo, move.old, move.new)
            self.log.add('move', 'git-mv', move.new, f'from {move.old}')
            if move.is_python_file:
                moved_files.append((move.old, move.new))
            else:
                for dirpath, _, files in os.walk(
                        os.path.join(self.repo, move.new)):
                    for name in files:
                        if name.endswith('.py'):
                            new = os.path.relpath(
                                os.path.join(dirpath, name), self.repo
                            ).replace('\\', '/')
                            old = move.old + new[len(move.new):]
                            moved_files.append((old, new))
        self.create_packages()
        self.create_modules()
        for old, new in moved_files:
            self.fix_moved_file_imports(old, new)
        self.ws.flush()
        return moved_files

    def create_packages(self):
        for path, spec in sorted(self.plan.packages_for_commit(
                self.commit).items()):
            init = path + '/__init__.py'
            if self.ws.exists(init):
                self.log.add('other', 'package-exists', init,
                             'kept the existing __init__.py')
                continue
            eol = self._default_eol()
            text = shims_mod.package_init_text(self.plan, path, spec)
            self.ws.set(init, SourceText(_with_eol(text, eol), eol=eol))
            self.log.add('other', 'package-created', init)

    def create_modules(self):
        """Modules entries with `code` and `commit`: new modules that
        receive no item (e.g. a __main__.py), created with their docstring
        and code."""
        for path, spec in sorted(self.plan.modules.items()):
            if spec.get('commit') != self.commit or not spec.get('code'):
                continue
            if self.ws.exists(path):
                self.log.add('other', 'module-exists', path,
                             'kept the existing module')
                continue
            code = spec['code']
            if isinstance(code, list):
                code = '\n'.join(code)
            parts = []
            if spec.get('doc'):
                parts.append(format_docstring(spec['doc']))
            parts.append(code.strip('\n'))
            eol = self._default_eol()
            self.ensure_package_inits(path)
            text = '\n'.join(parts) + '\n'
            self.ws.set(path, SourceText(_with_eol(text, eol), eol=eol))
            self.log.add('other', 'module-created', path, 'from code')

    def _default_eol(self):
        """EOL for new files: that of cellacdc/__init__.py (CRLF in a
        Windows working tree with autocrlf)."""
        try:
            return SourceText.read(os.path.join(
                self.repo, 'cellacdc', '__init__.py')).eol
        except OSError:
            return '\n'

    def ensure_package_inits(self, path):
        """Create missing __init__.py files for the folders of a new
        module (packages without a spec get a generated docstring)."""
        parts = path.split('/')[:-1]
        for i in range(2, len(parts) + 1):
            folder = '/'.join(parts[:i])
            init = folder + '/__init__.py'
            if self.ws.exists(init):
                continue
            spec = self.plan.packages.get(folder)
            if spec is None:
                spec = {'doc': f'Package {folder} (generated).'}
                self.log.add('other', 'package-without-spec', init,
                             'no packages entry in the mapping')
            eol = self._default_eol()
            text = shims_mod.package_init_text(self.plan, folder, spec)
            self.ws.set(init, SourceText(_with_eol(text, eol), eol=eol))
            self.log.add('other', 'package-created', init)

    def fix_moved_file_imports(self, old, new):
        src = self.ws.read(new)
        tree = ast.parse(src.text, new)
        edit = FileEdit(src, new)
        old_ctx = path_to_ctx(old)
        new_ctx = path_to_ctx(new)
        changed = self._retarget_imports(edit, tree, old_ctx, new_ctx,
                                         new, nodes=None)
        if changed:
            self.ws.set(new, edit.result())

    def _retarget_imports(self, edit, tree_or_node, old_ctx, new_ctx,
                          file_label, nodes=None):
        """Rewrite every from-import in `nodes` (default: all in the tree)
        for the move old_ctx -> new_ctx. Returns the number of changes."""
        if nodes is None:
            nodes = [n for n in ast.walk(tree_or_node)
                     if isinstance(n, ast.ImportFrom)]
        count = 0
        src = edit.src
        for node in nodes:
            if not isinstance(node, ast.ImportFrom):
                continue
            plan = plan_importfrom(node, old_ctx, new_ctx, self.remap,
                                   self.is_module)
            if plan is None:
                continue
            count += 1
            first_line = strip_eol(src.line(node.lineno))
            starts_line = not first_line[:char_col(
                first_line, node.col_offset)].strip()
            if plan[0] == 'prefix':
                text = ''.join(src.lines[node.lineno - 1:node.end_lineno])
                # offsets inside the statement text from its line start
                start, end = _module_expr_span(text[char_col(
                    first_line, node.col_offset):])
                base = char_col(first_line, node.col_offset)
                start += base
                end += base
                lineno, col_s = _offset_to_linecol(src, node.lineno, start)
                lineno2, col_e = _offset_to_linecol(src, node.lineno, end)
                if lineno != lineno2:
                    raise RefactorError(
                        f'{file_label}:{node.lineno}: module expression '
                        'spans lines')
                old_span = strip_eol(src.line(lineno))[col_s:col_e]
                lead = old_span[:len(old_span) - len(old_span.lstrip())]
                trail = old_span[len(old_span.rstrip()):]
                edit.splice(lineno, col_s, col_e,
                            (lead or ' ') + plan[1] + (trail or ' '))
                self.log.add('depth', 'import-retarget', file_label,
                             f'line {node.lineno}: '
                             f'{_orig_expr(node)} -> {plan[1]}')
            else:
                if not starts_line or (node.end_lineno < len(src.lines) and
                                       _shares_last_line(src, node)):
                    raise RefactorError(
                        f'{file_label}:{node.lineno}: cannot split an '
                        'import that shares its line with other code')
                lines, _ = rewrite_import_stmt(
                    src, node, old_ctx, new_ctx, self.remap, self.is_module)
                eol = edit.eol
                edit.replace_lines(node.lineno, node.end_lineno,
                                   [line + eol for line in lines])
                self.log.add('depth', 'import-split', file_label,
                             f'line {node.lineno}: ' + ' | '.join(lines))
        return count

    # ---------------------------------------------------------- step 2
    def apply_item_moves(self):
        moves = self.plan.items_for_commit(self.commit)
        splits = self.plan.class_splits_for_commit(self.commit)
        if not moves and not splits:
            return
        # 1. locate the leaving items in their current modules
        for move in sorted(moves, key=lambda m: (m.old_file, m.key)):
            src_path = self.plan.item_location(
                move.old_file, move.key, self.commit, inclusive=False)
            if src_path is None:
                raise RefactorError(f'{move.ref}: deleted before')
            # module moves of this commit already happened
            src_path = self.plan.module_path(src_path, self.commit)
            dest = (self.plan.module_path(move.dest, self.commit)
                    if move.dest else None)
            if dest == src_path:
                self.log.add('other', 'noop-move', src_path, move.ref)
                continue
            unit = self.unit(src_path)
            if not unit.exists:
                raise RefactorError(f'{move.ref}: {src_path} does not '
                                    'exist')
            item = self._find_item(unit, move.old_file, move.key)
            unit.leaving.append((item, dest, move))
        for split, methods in splits:
            host_path = self.plan.item_location(
                split.old_file, split.cls, self.commit, inclusive=False)
            host_path = self.plan.module_path(host_path, self.commit)
            unit = self.unit(host_path)
            class_node = _find_class(unit.tree, split.cls)
            if class_node is None:
                raise RefactorError(f'No class {split.cls} in {host_path}')
            members = class_items(unit.src, class_node)
            by_name = {}
            for member in members:
                if member.kind == 'method':
                    by_name.setdefault(member.key, []).append(member)
            for meth in methods:
                found = by_name.get(meth)
                if not found:
                    raise RefactorError(
                        f'{split.ref}: method {meth} not found in '
                        f'{host_path}')
                mixin_file, mixin_name = split.methods[meth]
                dest = self.plan.module_path(mixin_file, self.commit)
                for member in found:
                    unit.leaving_methods.append(
                        (split, class_node, member, dest, mixin_name))
        # 2. compute edits (items keep their source order)
        for path in sorted(list(self.units)):
            unit = self.units[path]
            unit.leaving.sort(key=lambda entry: entry[0].first)
            unit.leaving_methods.sort(key=lambda entry: entry[2].first)
            if unit.leaving or unit.leaving_methods:
                self._process_source(unit)
        # 3. host bases
        for split, methods in splits:
            self._set_bases(split)
        # 4. destinations
        for path in sorted(list(self.units)):
            unit = self.units[path]
            if unit.incoming or unit.mixins or unit.import_lines:
                self._assemble_destination(unit)
        # 5. write
        for path in sorted(self.units):
            unit = self.units[path]
            for lineno in sorted(unit.long_line_check):
                text = strip_eol(unit.edit._render_line(lineno))
                if len(text) > 127:
                    self.log.add('qualify', 'long-line', path,
                                 f'line {lineno}: {len(text)} characters')
            if unit.exists and unit.edit is not None:
                result = unit.edit.result()
                if result.text != unit.src.text:
                    self.ws.set(path, result)
        self.ws.flush()

    def _find_item(self, unit, old_file, key):
        cands = candidate_items(unit.items, key)
        if not cands:
            raise RefactorError(f'{old_file}::{key} not found in '
                                f'{unit.path}')
        if len(cands) == 1:
            return cands[0]
        base_item = self.plan.base_module(old_file).item(key)
        if base_item is not None:
            bsrc = self.plan.base_module(old_file).src
            btext = normalize_eol(bsrc.segment(base_item.code_first,
                                               base_item.node.end_lineno))
            exact = [c for c in cands if normalize_eol(unit.src.segment(
                c.code_first, c.node.end_lineno)) == btext]
            if len(exact) == 1:
                return exact[0]
        index = repeat_index(key)
        if index <= len(cands):
            return cands[index - 1]
        raise RefactorError(f'{old_file}::{key}: ambiguous in {unit.path}')

    # --------------------------------------------------- source side
    def _home_of_name(self, unit, name, leaving_index):
        """(kind, value): ('def', dest path) for names defined in the unit
        (dest = unit.path when the defining item stays), ('import',
        [bindings]) or (None, None)."""
        item = unit.defined.get(name)
        if item is not None:
            dest = leaving_index.get(id(item))
            if dest is None:
                return 'def', unit.path
            return 'def', dest
        if name in unit.bindings:
            return 'import', unit.bindings[name]
        return None, None

    def _process_source(self, unit):
        analysis = unit.analysis
        leaving_index = {}
        deleted_ids = set()
        for item, dest, move in unit.leaving:
            if dest is None:
                deleted_ids.add(id(item))
                leaving_index[id(item)] = '<deleted>'
            else:
                leaving_index[id(item)] = dest
        # spans of leaving top-level items and methods
        spans = []
        for item, dest, move in unit.leaving:
            spans.append((item.first, item.last, item, dest, None))
        for split, class_node, member, dest, mixin in unit.leaving_methods:
            spans.append((member.first, member.last, member, dest, mixin))
        spans.sort(key=lambda s: s[0])
        for a, b in zip(spans, spans[1:]):
            if b[0] <= a[1]:
                raise RefactorError(f'{unit.path}: overlapping items '
                                    f'{a[2]} and {b[2]}')

        def span_of(lineno):
            for first, last, item, dest, mixin in spans:
                if first <= lineno <= last:
                    return item, dest, mixin
            return None, None, None

        # imports used by leaving items, per destination
        needs = {}      # dest -> {name: [bindings]}
        for ref in analysis.refs:
            item, dest, mixin = span_of(ref.lineno)
            if item is None:
                # remaining code: qualify references to names that left
                if ref.ctx != 'load' and ref.scope_kind == 'module':
                    continue
                kind, value = self._home_of_name(unit, ref.name,
                                                 leaving_index)
                if kind == 'def' and value != unit.path:
                    if value == '<deleted>':
                        raise RefactorError(
                            f'{unit.path}:{ref.lineno}: {ref.name} is '
                            'deleted but still used')
                    if ref.ctx != 'load':
                        self.log.add('other', 'global-write-split',
                                     unit.path, f'line {ref.lineno}: '
                                     f'{ref.name} now lives in {value}')
                        continue
                    alias = self._module_alias(unit, dotted(value))
                    self._qualify(unit, ref, alias)
                continue
            if dest is None:
                continue        # deleted item
            if ref.ctx != 'load':
                if ref.scope_kind in ('module', 'class') and \
                        ref.name in (item.names or []):
                    continue
                if ref.scope_kind == 'module':
                    continue
                self.log.add('other', 'global-write', unit.path,
                             f'line {ref.lineno}: {ref.name} written from '
                             f'code moving to {dest}')
                continue
            kind, value = self._home_of_name(unit, ref.name, leaving_index)
            if kind == 'def':
                if value == '<deleted>':
                    raise RefactorError(
                        f'{unit.path}:{ref.lineno}: {ref.name} is deleted '
                        f'but used by code moving to {dest}')
                if value == dest:
                    continue
                dunit = self.unit(dest)
                alias = self._module_alias(dunit, dotted(value))
                self._qualify(unit, ref, alias)
            elif kind == 'import':
                needs.setdefault(dest, {})[ref.name] = value
            elif ref.name not in BUILTINS:
                self.log.add('other', 'unresolved-global', unit.path,
                             f'line {ref.lineno}: {ref.name} (moving to '
                             f'{dest})')
        # nested imports inside leaving code
        for first, last, item, dest, mixin in spans:
            if dest is None:
                continue
            dctx = path_to_ctx(dest)
            nodes = [n for n in ast.walk(item.node)
                     if isinstance(n, ast.ImportFrom)]
            self._retarget_imports(unit.edit, None, unit.ctx, dctx,
                                   unit.path, nodes=nodes)
        # copied imports for each destination
        for dest, names in needs.items():
            self._copy_imports(unit, self.unit(dest), names, leaving_index)
        # extract the texts and delete them from the source
        prev_item_by_dest = {}
        for item, dest, move in unit.leaving:
            if dest is None:
                self.log.add('other', 'deleted', unit.path,
                             f'{item.key} (lines {item.first}-{item.last})')
                continue
            dunit = self.unit(dest)
            self.ensure_package_inits(dest)
            lines = unit.edit.render_range(item.first, item.last)
            kind = 'import' if item.kind == 'import' else 'item'
            inc = IncomingItem(unit, item, lines, kind)
            inc.adjacent = prev_item_by_dest.get(dest) is not None and (
                prev_item_by_dest[dest].index + 1 == item.index)
            prev_item_by_dest[dest] = item
            dunit.incoming.append(inc)
            self.log.add('move', 'item', dest,
                         f'{item.key} from {unit.path}')
        prev_member = {}
        for split, class_node, member, dest, mixin in unit.leaving_methods:
            dunit = self.unit(dest)
            self.ensure_package_inits(dest)
            lines = unit.edit.render_range(member.first, member.last)
            if mixin not in dunit.mixins:
                spec = split.mixins.get((self._mixin_file(split, mixin),
                                         mixin), {})
                dunit.mixins[mixin] = {'doc': spec.get('doc', ''),
                                       'methods': []}
                dunit.mixin_order.append(mixin)
            inc = IncomingItem(unit, member, lines, 'method', mixin)
            key = (dest, mixin)
            inc.adjacent = prev_member.get(key) is not None and (
                prev_member[key].index + 1 == member.index)
            prev_member[key] = member
            dunit.mixins[mixin]['methods'].append(inc)
            self.log.add('move', 'method', dest,
                         f'{split.cls}.{member.key} -> {mixin}')
        self._delete_spans(unit, [s for s in spans])

    def _mixin_file(self, split, mixin):
        for (file, name) in split.mixins:
            if name == mixin:
                return file
        return None

    def _delete_spans(self, unit, spans):
        src = unit.src
        for first, last, item, _dest, _mixin in spans:
            # drop the item and the blank lines before it (down to the
            # previous statement or the container start): the gap after it
            # stays as the separator before the next statement
            start = self._container_start(unit, item)
            k = first - 1
            while k >= start and is_blank(src.line(k)) and \
                    not unit.edit.is_deleted(k):
                k -= 1
            unit.edit.delete_lines(k + 1, last)
        # classes left without statements get `pass`
        done = set()
        for _split, class_node, _m, _d, _x in unit.leaving_methods:
            if id(class_node) in done:
                continue
            done.add(id(class_node))
            if all(unit.edit.is_deleted(line) for stmt in class_node.body
                   for line in range(stmt.lineno, stmt.end_lineno + 1)):
                indent = ' ' * (class_node.col_offset + 4)
                unit.edit.insert_before(
                    class_node.body[0].lineno,
                    [indent + 'pass' + unit.edit.eol])
                self.log.add('other', 'pass-inserted', unit.path,
                             class_node.name)

    def _container_start(self, unit, item):
        """First line that may be deleted before `item`: the line after
        its class header (methods) or the module's first line."""
        for node in unit.tree.body:
            if isinstance(node, ast.ClassDef) and \
                    node.lineno < item.node.lineno <= node.end_lineno:
                return class_header_end(unit.src, node) + 1
        return 1

    def _qualify(self, unit, ref, alias):
        if ref.col is None:
            self.log.add('other', 'string-annotation-unqualified', unit.path,
                         f'line {ref.lineno}: {ref.name} (now in module '
                         f'alias {alias}) inside a string annotation that '
                         'cannot be located: fix by hand')
            return
        if is_shadowed(ref, alias):
            raise RefactorError(
                f'{unit.path}:{ref.lineno}: cannot qualify {ref.name} as '
                f'{alias}.{ref.name}: {alias!r} is a local name there '
                '(choose another alias in the mapping)')
        line = strip_eol(unit.src.line(ref.lineno))
        col = char_col(line, ref.col)
        unit.edit.splice(ref.lineno, col, col, alias + '.')
        self.log.add('qualify', 'qualify', unit.path,
                     f'line {ref.lineno}: {ref.name} -> {alias}.{ref.name}')
        unit.long_line_check.add(ref.lineno)

    # --------------------------------------------------- module aliases
    def _module_alias(self, unit, module_dotted):
        """Local name under which `unit` reaches module `module_dotted`;
        adds the import if needed."""
        if module_dotted in unit.alias_for_module:
            return unit.alias_for_module[module_dotted]
        for name, bindings in unit.bindings.items():
            b = bindings[-1]
            if not b.is_module_value or b.guard is not None:
                continue
            target = (f'{b.target}.{b.attr}' if b.attr else
                      (b.target if b.alias.asname else None))
            if target and self.remap(target) == module_dotted and \
                    name not in unit.defined:
                unit.alias_for_module[module_dotted] = name
                return name
        alias = self.plan.alias_of(module_dotted)
        if alias in unit.defined or (alias in unit.bindings and not any(
                b.is_module_value and self.remap(
                    f'{b.target}.{b.attr}' if b.attr else b.target)
                == module_dotted for b in unit.bindings[alias])):
            raise RefactorError(
                f'{unit.path}: alias {alias!r} of {module_dotted} collides '
                'with an existing name')
        if alias in unit.new_bindings and \
                unit.new_bindings[alias] != module_dotted:
            raise RefactorError(
                f'{unit.path}: alias {alias!r} is used for both '
                f'{unit.new_bindings[alias]} and {module_dotted}')
        parent, _, last = module_dotted.rpartition('.')
        expr = from_expr(unit.ctx, parent)
        names = [(last, None if alias == last else alias)]
        from .imports import format_from
        lines = format_from(expr, names)
        self._add_import_lines(unit, lines, ('alias', module_dotted),
                               rank=2)
        unit.new_bindings[alias] = module_dotted
        unit.alias_for_module[module_dotted] = alias
        self.log.add('import', 'alias-import', unit.path, lines[0])
        return alias

    def _add_import_lines(self, unit, lines, key, rank=1, after=None):
        """Queue import lines for `unit`: rank 1 copied imports (source
        order), 2 module aliases, 3 base classes; `after` attaches lines
        right after an earlier entry (companions)."""
        if key in unit.import_keys:
            return
        unit.import_keys.add(key)
        if after is not None:
            for entry in unit.import_lines:
                if entry[2] == after:
                    entry[3].extend(lines)
                    return
        unit.import_lines.append([rank, len(unit.import_lines), key,
                                  list(lines)])

    def sorted_import_lines(self, unit):
        out = []
        for rank, seq, key, lines in sorted(unit.import_lines,
                                            key=lambda e: (e[0], e[1])):
            out.extend(lines)
        return out

    # ---------------------------------------------------- copied imports
    def _copy_imports(self, sunit, dunit, names, leaving_index):
        """Copy the import statements binding `names` from sunit into
        dunit (with their guards; retargeted for dunit's location)."""
        per_top = {}
        names = dict(names)
        # the conditions of copied guards (`if GUI_INSTALLED:`) need their
        # names too (and so on, for nested guards)
        pending = list(names.items())
        while pending:
            name, bindings = pending.pop()
            for b in bindings:
                if b.guard is None:
                    continue
                for dep in _guard_names(b.guard):
                    if dep in names:
                        continue
                    if dep in sunit.bindings:
                        names[dep] = sunit.bindings[dep]
                        pending.append((dep, names[dep]))
                    elif dep in sunit.defined:
                        raise RefactorError(
                            f'{sunit.path}: the import guard at line '
                            f'{b.guard.lineno} uses {dep}, defined in the '
                            f'module: copy it to {dunit.path} by hand')
        for name, bindings in names.items():
            if self._dest_binds(dunit, name, bindings):
                continue
            last = bindings[-1]
            if last.attr is None and not last.alias.asname:
                # `import a.b` binds `a`: copy every `import a...`
                group = [b for b in bindings
                         if b.attr is None and not b.alias.asname]
            else:
                group = [last]
            for b in group:
                if b.guard is not None and leaving_index.get(
                        id(self._item_of_node(sunit, b.guard))) == \
                        dunit.path:
                    continue    # the whole block moves to dunit
                if b.guard is not None and leaving_index.get(
                        id(self._item_of_node(sunit, b.guard))) == \
                        '<deleted>':
                    raise RefactorError(
                        f'{sunit.path}: {name} is bound by a deleted import '
                        f'block but needed in {dunit.path}')
                per_top.setdefault(id(b.top), (b.top, {}))[1].setdefault(
                    id(b.node), set()).add(b.name)
        for _, (top, keep) in sorted(per_top.items(),
                                     key=lambda kv: kv[1][0].lineno):
            if isinstance(top, (ast.Import, ast.ImportFrom)):
                lines, _ = rewrite_import_stmt(
                    sunit.src, top, sunit.ctx, dunit.ctx, self.remap,
                    self.is_module, keep=keep[id(top)])
            else:
                lines = render_import_block(
                    sunit.src, top, keep, sunit.ctx, dunit.ctx, self.remap,
                    self.is_module)
            if not lines:
                continue
            if self._is_self_import(top, sunit, dunit):
                self.log.add('other', 'self-import', dunit.path,
                             ' | '.join(lines))
            key = ('copy', '\n'.join(lines))
            self._add_import_lines(dunit, lines, key)
            for comp_text in self._companions(sunit, top):
                if not dunit.is_new() or self._receives_text(
                        dunit, comp_text):
                    continue
                self._add_import_lines(dunit, [comp_text],
                                       ('companion', comp_text), after=key)
                self.log.add('other', 'companion-copy', dunit.path,
                             comp_text)
            for names_ in keep.values():
                for n in names_:
                    dunit.new_bindings.setdefault(n, ('import', sunit.path))

    def _receives_text(self, dunit, text):
        for sunit in self.units.values():
            for item, dest, _move in sunit.leaving:
                if dest == dunit.path and strip_eol(sunit.src.line(
                        item.node.lineno)).strip() == text:
                    return True
        return False

    def _companions(self, sunit, top):
        if not isinstance(top, ast.Import):
            return []
        bound = {a.asname or a.name.split('.')[0] for a in top.names}
        out = []
        for item in sunit.items:
            if item.kind != 'stmt':
                continue
            text = strip_eol(sunit.src.line(item.node.lineno)).strip()
            for pattern, name in COMPANION_PATTERNS:
                if name in bound and pattern.match(text) and \
                        item.node.lineno == item.node.end_lineno:
                    out.append(text)
        return out

    def _item_of_node(self, unit, node):
        for item in unit.items:
            if item.node is node:
                return item
        return None

    def _is_self_import(self, top, sunit, dunit):
        for node in ast.walk(top):
            if isinstance(node, ast.ImportFrom):
                target = resolve_relative(sunit.ctx, node.level, node.module)
                if self.remap(target) == dunit.ctx.dotted:
                    return True
                for alias in node.names:
                    if self.remap(f'{target}.{alias.name}') == \
                            dunit.ctx.dotted:
                        return True
        return False

    def _dest_binds(self, dunit, name, bindings):
        """True if dunit already binds `name` the same way."""
        b = bindings[-1]
        if b.attr is None and not b.alias.asname:
            # `import a.b` binds the package `a`: compatible with every other
            # `import a...` of the destination; the statements are copied
            # (deduplicated by text) so that each submodule is imported
            for e in dunit.bindings.get(name, []):
                if e.attr is not None or e.alias.asname:
                    raise RefactorError(
                        f'{dunit.path}: {name} is already imported '
                        f'differently ({e} vs {b})')
            if name in dunit.defined:
                raise RefactorError(
                    f'{dunit.path}: {name} is defined there but code moving '
                    f'in uses the module {name}')
            existing_targets = {e.target for e in dunit.bindings.get(name,
                                                                     [])}
            return all(x.target in existing_targets for x in bindings
                       if x.attr is None and not x.alias.asname)
        if name in dunit.new_bindings:
            return True
        existing = dunit.bindings.get(name)
        if existing:
            e = existing[-1]
            b = bindings[-1]
            if (self.remap(e.target or '') == self.remap(b.target or '')
                    and e.attr == b.attr):
                return True
            raise RefactorError(
                f'{dunit.path}: {name} is already imported differently '
                f'({e} vs {b})')
        if name in dunit.defined:
            b = bindings[-1]
            if b.attr and self.remap(b.target) == dunit.ctx.dotted:
                return True     # from-import of dunit's own definition
            raise RefactorError(
                f'{dunit.path}: {name} is defined there but code moving in '
                f'uses the imported {name} ({b})')
        return False

    # ------------------------------------------------------ host bases
    def _set_bases(self, split):
        host_path = self.plan.module_path(split.host or self.plan.item_location(
            split.old_file, split.cls, self.commit), self.commit)
        unit = self.unit(host_path)
        node = _find_class(unit.tree, split.cls)
        header_end = class_header_end(unit.src, node)
        existing = {}
        for base in node.bases:
            text = ast.get_source_segment(unit.src.text, base)
            resolved = self._resolve_base(unit, base)
            if resolved is not None:
                existing[resolved] = text
        new_bases = []
        for ref in split.bases:
            if '::' not in ref:
                new_bases.append(ref)
                continue
            file, _, cls = ref.partition('::')
            file = self.plan.module_path(file, self.commit)
            if (file, cls) in existing:
                new_bases.append(existing[(file, cls)])
                continue
            if file == host_path:
                new_bases.append(cls)
                continue
            module_dotted = dotted(file)
            parent_expr = from_expr(unit.ctx, module_dotted)
            from .imports import format_from
            lines = format_from(parent_expr, [(cls, None)])
            self._add_import_lines(unit, lines, ('base', module_dotted, cls),
                                   rank=3)
            unit.new_bindings[cls] = module_dotted
            new_bases.append(cls)
        indent = strip_eol(unit.src.line(node.lineno))[:node.col_offset]
        keywords = [ast.get_source_segment(unit.src.text, k)
                    for k in node.keywords]
        eol = unit.edit.eol
        lines = [f'{indent}class {node.name}(' + eol]
        for base in new_bases + keywords:
            lines.append(f'{indent}        {base},' + eol)
        lines.append(f'{indent}    ):' + _header_tail(unit.src, header_end)
                     + eol)
        unit.edit.replace_lines(node.lineno, header_end, lines)
        self.log.add('other', 'bases-set', host_path,
                     f'{split.cls}: {len(new_bases)} bases')

    def _resolve_base(self, unit, base):
        """(path, class) of a base class expression, or None."""
        if isinstance(base, ast.Name):
            if base.id in unit.defined:
                return (unit.path, base.id)
            bindings = unit.bindings.get(base.id)
            if bindings and bindings[-1].attr:
                b = bindings[-1]
                return self._class_location(b.target, b.attr)
            return None
        if isinstance(base, ast.Attribute) and isinstance(base.value,
                                                          ast.Name):
            bindings = unit.bindings.get(base.value.id)
            if not bindings or not bindings[-1].is_module_value:
                return None
            b = bindings[-1]
            module = f'{b.target}.{b.attr}' if b.attr else b.target
            return self._class_location(module, base.attr)
        return None

    def _class_location(self, module_dotted, cls):
        base_path = None
        for candidate in (module_dotted.replace('.', '/') + '.py',
                          module_dotted.replace('.', '/') + '/__init__.py'):
            if self.plan.base.exists(candidate):
                base_path = candidate
                break
        if base_path is not None:
            bmod = self.plan.base_module(base_path)
            item = bmod.defined.get(cls)
            if item is not None:
                loc = self.plan.item_location(base_path, item.key,
                                              self.commit)
                return (loc, cls)
        # a new module
        return (module_dotted.replace('.', '/') + '.py', cls)

    # ------------------------------------------------- destination side
    def _assemble_destination(self, unit):
        eol = unit.edit.eol if unit.edit is not None else \
            self._eol_for_new(unit)
        import_lines = []
        for line in self.sorted_import_lines(unit):
            import_lines.append(line + eol)
        imports_from_items = [inc for inc in unit.incoming
                              if inc.kind == 'import']
        for inc in imports_from_items:
            import_lines.extend(ensure_eol(list(inc.lines), eol))
        items = self._merge_identical_bindings(
            unit, [inc for inc in unit.incoming if inc.kind == 'item'])
        doc_incs = [inc for inc in items if inc.item.kind == 'docstring']
        items = [inc for inc in items if inc.item.kind != 'docstring']
        if len(doc_incs) > 1:
            raise RefactorError(f'{unit.path}: receives {len(doc_incs)} '
                                'module docstrings')
        incoming_doc = (ensure_eol(list(doc_incs[0].lines), eol)
                        if doc_incs else None)
        body = []
        prev = None
        for inc in items:
            lines = ensure_eol(list(inc.lines), eol)
            if prev is not None:
                if getattr(inc, 'adjacent', False):
                    gap = inc.item.gap_before or 0
                elif inc.item.kind in ('def', 'class') or \
                        prev.item.kind in ('def', 'class'):
                    gap = 2
                else:
                    gap = 1
                body.extend([eol] * gap)
            body.extend(lines)
            prev = inc
        mixin_blocks = []
        existing_classes = {}
        if unit.exists:
            for node in unit.tree.body:
                if isinstance(node, ast.ClassDef):
                    existing_classes[node.name] = node
        for name in unit.mixin_order:
            spec = unit.mixins[name]
            methods = []
            prev = None
            for inc in spec['methods']:
                lines = ensure_eol(list(inc.lines), eol)
                if prev is not None:
                    gap = (inc.item.gap_before or 1) if getattr(
                        inc, 'adjacent', False) else 1
                    methods.extend([eol] * gap)
                methods.extend(lines)
                prev = inc
            if name in existing_classes:
                node = existing_classes[name]
                unit.edit.insert_before(node.end_lineno + 1,
                                        [eol] + methods)
                self.log.add('move', 'mixin-extended', unit.path, name)
                continue
            block = [f'class {name}:' + eol]
            doc = spec.get('doc') or ''
            if doc:
                block.extend(text_to_lines(format_docstring(doc, '    '),
                                           eol))
                block.append(eol)
            block.extend(methods)
            mixin_blocks.append(block)
            self.log.add('other', 'mixin-created', unit.path, name)
        if unit.is_new():
            spec = self.plan.modules.get(unit.path, {})
            text = []
            doc = spec.get('doc')
            if doc and incoming_doc:
                text.extend(_merge_docstring_lines(
                    incoming_doc, text_to_lines(format_docstring(doc), eol),
                    eol, unit.path))
                self.log.add('other', 'docstring-merged', unit.path,
                             doc_incs[0].item.key)
            elif incoming_doc:
                text.extend(incoming_doc)
            elif doc:
                text.extend(text_to_lines(format_docstring(doc), eol))
            else:
                self.log.add('other', 'module-without-doc', unit.path)
            future = self._future_lines(unit, eol)
            text.extend(future)
            if import_lines:
                text.extend(import_lines)
            chunks = []
            if body:
                chunks.append(body)
            chunks.extend(mixin_blocks)
            for chunk in chunks:
                if text:
                    text.extend([eol, eol])
                text.extend(chunk)
            ensure_eol(text, eol)
            self.ws.set(unit.path, SourceText(''.join(text), eol=eol))
            self.log.add('other', 'module-created', unit.path)
            return
        # existing destination
        if incoming_doc:
            # the moved docstring comes first, the destination's own
            # docstring (e.g. a package overview) follows in the same one
            own = [i for i in unit.items if i.kind == 'docstring']
            if own:
                first, last = own[0].first, own[0].last
                old = [unit.src.line(k) for k in range(first, last + 1)]
                unit.edit.replace_lines(first, last, _merge_docstring_lines(
                    incoming_doc, ensure_eol(old, eol), eol, unit.path))
            else:
                unit.edit.insert_before(1, incoming_doc)
            self.log.add('other', 'docstring-merged', unit.path,
                         doc_incs[0].item.key)
        self._drop_shadowed_imports(unit)
        if import_lines:
            line = self._import_insertion_line(unit)
            unit.edit.insert_before(line, import_lines)
        chunks = []
        if body:
            chunks.append(body)
        chunks.extend(mixin_blocks)
        if chunks:
            line = self._append_line(unit)
            out = []
            for chunk in chunks:
                out.extend([eol, eol])
                out.extend(chunk)
            unit.edit.insert_before(line, out)

    def _merge_identical_bindings(self, unit, items):
        """Drop incoming module-level assignments that the destination
        already has (or receives earlier in this commit) as the same
        statement text: two old modules held byte-identical copies of a
        constant (apps.py and widgets.py: TREEWIDGET_STYLESHEET, ...) and
        both are mapped to the same home, which keeps one binding."""
        def norm(lines):
            return '\n'.join(strip_eol(line).rstrip() for line in lines
                             if strip_eol(line).strip()
                             and not strip_eol(line).lstrip().startswith('#'))

        seen = {}
        if unit.exists and unit.src is not None:
            for item in unit.items:
                if item.kind != 'assign':
                    continue
                text = norm(unit.src.line(k)
                            for k in range(item.code_first or item.first,
                                           item.last + 1))
                seen[(tuple(item.names or ()), text)] = unit.path
        kept = []
        for inc in items:
            if inc.item.kind == 'assign':
                code_lines = list(inc.lines)
                key = (tuple(inc.item.names or ()), norm(code_lines))
                if key in seen:
                    self.log.add('other', 'merged-duplicate', unit.path,
                                 f'{inc.item.key} from {inc.src_unit.path}: '
                                 f'identical binding already in {seen[key]}',
                                 items=[f'{inc.src_unit.path}::'
                                        f'{inc.item.key}'])
                    continue
                seen[key] = inc.src_unit.path
            kept.append(inc)
        return kept

    def _drop_shadowed_imports(self, unit):
        """An existing destination that from-imports a cellacdc name it now
        defines itself (the item moved in) loses that import: the old path
        would forward back to the module being imported."""
        arriving = set()
        for inc in unit.incoming:
            if inc.kind == 'item':
                arriving.update(inc.item.names or [])
        per_node = {}
        for name in sorted(arriving):
            for b in unit.bindings.get(name, []):
                if b.attr is None or not (b.target or '').startswith(
                        'cellacdc'):
                    continue
                per_node.setdefault(id(b.node), (b.node, set()))[1].add(
                    b.alias.name)
        eol = unit.edit.eol
        for node, names in per_node.values():
            keep = [a for a in node.names if a.name not in names]
            line = strip_eol(unit.src.line(node.lineno))
            indent = line[:char_col(line, node.col_offset)]
            if indent.strip():
                raise RefactorError(
                    f'{unit.path}:{node.lineno}: cannot drop the import of '
                    f'{sorted(names)} (statement shares its line)')
            if keep:
                from .imports import format_from, original_expr
                new = format_from(original_expr(node),
                                  [(a.name, a.asname) for a in keep], indent)
            else:
                new = [indent + 'pass']
                parent_is_module = any(n is node for n in unit.tree.body)
                if parent_is_module:
                    new = []
            unit.edit.replace_lines(node.lineno, node.end_lineno,
                                    [text + eol for text in new])
            self.log.add('import', 'import-dropped', unit.path,
                         f'line {node.lineno}: {sorted(names)} now defined '
                         'in the module')

    def _eol_for_new(self, unit):
        for inc in unit.incoming:
            return inc.src_unit.src.eol
        for spec in unit.mixins.values():
            for inc in spec['methods']:
                return inc.src_unit.src.eol
        return self._default_eol()

    def _future_lines(self, unit, eol):
        sources = {inc.src_unit for inc in unit.incoming}
        for spec in unit.mixins.values():
            sources.update(inc.src_unit for inc in spec['methods'])
        for sunit in sorted(sources, key=lambda u: u.path):
            if sunit.tree is not None and any(
                    isinstance(n, ast.ImportFrom) and n.module == '__future__'
                    for n in sunit.tree.body):
                return ['from __future__ import annotations' + eol]
        return []

    def _import_insertion_line(self, unit):
        """Line before which new imports go: after the last top-level
        import that precedes the first def/class (or after the
        docstring)."""
        last = None
        for item in unit.items:
            if item.kind in ('def', 'class'):
                break
            if item.kind in ('import', 'importblock') or (
                    item.kind == 'docstring'):
                last = item
        if last is None:
            return 1
        return last.node.end_lineno + 1

    def _append_line(self, unit):
        """Line before which appended items go: before a trailing
        `if __name__ == '__main__':` block, else the end."""
        if unit.items:
            last = unit.items[-1]
            node = last.node
            if isinstance(node, ast.If) and _is_main_check(node.test):
                k = last.first
                while k > 1 and is_blank(unit.src.line(k - 1)):
                    k -= 1
                return k
        return len(unit.src.lines) + 1

    # ---------------------------------------------------------- driver
    def run(self):
        self.apply_module_moves()
        self.units = {}
        self.apply_item_moves()
        shims_mod.apply_commit_shims(self)
        return self.log


def _merge_docstring_lines(first, second, eol, path):
    """One docstring from two raw docstrings (lists of lines with EOL):
    the text of `first`, a blank line, then the text of `second`. Both must
    use the same triple quotes, `first` must end with the closing quotes
    on a line of their own and `second` must open them on its first line
    (no string prefix)."""
    quote = first[0].lstrip()[:3]
    if quote not in ('"""', "'''") or not second or \
            not second[0].startswith(quote) or \
            strip_eol(first[-1]).strip() != quote:
        raise RefactorError(f'{path}: cannot merge the moved module '
                            'docstring into the existing one: merge them '
                            'in a fixup')
    head = second[0][len(quote):]
    return list(first[:-1]) + [eol, head] + list(second[1:])


def _guard_names(guard):
    """Names read by the non-import parts of an import block (conditions,
    exception types, context managers)."""
    out = []
    exprs = []
    for node in ast.walk(guard):
        if isinstance(node, ast.If):
            exprs.append(node.test)
        elif isinstance(node, ast.ExceptHandler) and node.type is not None:
            exprs.append(node.type)
        elif isinstance(node, ast.With):
            exprs.extend(item.context_expr for item in node.items)
    for expr in exprs:
        for sub in ast.walk(expr):
            if isinstance(sub, ast.Name) and sub.id not in out and \
                    sub.id not in BUILTINS:
                out.append(sub.id)
    return out


def _with_eol(text, eol):
    """Generated text (with \\n line ends) with the file's EOL."""
    return normalize_eol(text).replace('\n', eol)


def _is_main_check(test):
    return (isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == '__name__')


def _find_class(tree, name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _header_tail(src, header_end):
    """Comment after the ':' of a class header line (kept)."""
    line = strip_eol(src.line(header_end))
    idx = line.rfind(':')
    tail = line[idx + 1:]
    return tail.rstrip() if tail.strip().startswith('#') else ''


def _offset_to_linecol(src, first_line, offset):
    lineno = first_line
    while True:
        line = src.line(lineno)
        if offset <= len(strip_eol(line)):
            return lineno, offset
        offset -= len(line)
        lineno += 1


def _shares_last_line(src, node):
    line = strip_eol(src.line(node.end_lineno))
    tail = line[char_col(line, node.end_col_offset):].strip()
    return bool(tail) and not tail.startswith('#')


def _orig_expr(node):
    return '.' * node.level + (node.module or '')


def apply_commit(repo, plan, commit, out_dir=None):
    mover = Mover(repo, plan, commit)
    log = mover.run()
    out_dir = out_dir or os.path.join(TOOL_DIR, 'out', commit)
    log.write(out_dir)
    return log
