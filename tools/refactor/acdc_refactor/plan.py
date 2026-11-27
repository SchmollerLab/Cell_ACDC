"""The mapping shards and everything derived from them.

Plan loads tools/refactor/mapping/*.json, validates the schema, merges the
shards and answers, for any commit id:

- where a module (old path at the base commit) lives (module moves);
- where an item ('<old file>::<key>') lives;
- the name table (old dotted module -> {name: new home}).

Paths are POSIX paths relative to the repository root. Keys always use the
file path at the base commit; the base commit is read with `git show`, so
the plan works at any checked out commit.
"""
import ast
import json
import warnings
import os

from . import (
    COMMITS, MOVE_COMMITS, RefactorError, commit_le, commit_lt, commit_index,
    base_commit
)
from .gitutil import BaseTree
from .imports import import_bindings, path_to_ctx
from .items import module_items, find_module_item, class_items
from .scopes import ModuleAnalysis
from .source import SourceText

SHIM_KINDS = ('alias', 'split', 'none', 'package')
SHARD_KEYS = {
    'shard', 'default_commit', 'module_moves', 'shims', 'packages',
    'modules', 'items', 'classes', 'notes', 'plugin_api', 'allow',
    'package_compat', 'attribute_forwarders'
}
DELETE = 'DELETE'

PART_ABBREVIATIONS = {
    'measurements_viewer': 'viewer', 'segmentation_module': 'segm',
    'shared': '', 'myutils': '',
}


def dotted(path):
    return path_to_ctx(path).dotted


def default_alias(dotted_name):
    parts = dotted_name.split('.')[1:]
    if not parts:
        return 'cellacdc'
    if len(parts) == 1:
        return parts[0]
    head = PART_ABBREVIATIONS.get(parts[0], parts[0])
    rest = [head] + parts[1:] if head else parts[1:]
    return '_'.join(rest)


class Issue:
    def __init__(self, level, check, message, where=None):
        self.level = level      # 'error' or 'warning' or 'info'
        self.check = check
        self.message = message
        self.where = where

    def __str__(self):
        loc = f' [{self.where}]' if self.where else ''
        return f'{self.level.upper()} {self.check}: {self.message}{loc}'


class ModuleMove:
    def __init__(self, old, new, shim, commit, shard):
        self.old = old.rstrip('/')
        self.new = new.rstrip('/')
        self.shim = shim
        self.commit = commit
        self.shard = shard

    @property
    def is_python_file(self):
        return self.old.endswith('.py')

    def __repr__(self):
        return f'ModuleMove({self.old} -> {self.new}, {self.shim}, ' \
               f'{self.commit})'


class ItemMove:
    def __init__(self, old_file, key, dest, commit, shard):
        self.old_file = old_file
        self.key = key
        self.dest = dest            # path or None (DELETE)
        self.commit = commit
        self.shard = shard

    @property
    def ref(self):
        return f'{self.old_file}::{self.key}'

    def __repr__(self):
        return f'ItemMove({self.ref} -> {self.dest}, {self.commit})'


class ClassSplit:
    def __init__(self, old_file, cls, spec, shard, default_commit):
        self.old_file = old_file
        self.cls = cls
        self.shard = shard
        self.host = spec.get('host')
        self.mixins = {}            # (file, name) -> {'doc', 'commit'}
        for ref, mspec in (spec.get('mixins') or {}).items():
            file, _, name = ref.partition('::')
            mspec = dict(mspec or {})
            mspec.setdefault('commit', default_commit)
            self.mixins[(file, name)] = mspec
        self.methods = {}           # method name -> (file, mixin name)
        for meth, ref in (spec.get('methods') or {}).items():
            file, _, name = ref.partition('::')
            self.methods[meth] = (file, name)
        self.bases = list(spec.get('bases') or [])

    @property
    def ref(self):
        return f'{self.old_file}::{self.cls}'

    def mixin_commit(self, mixin):
        return self.mixins.get(mixin, {}).get('commit')

    def method_commit(self, method):
        return self.mixin_commit(self.methods[method])


class BaseModule:
    """A module of the base commit: text, items, import bindings, analysis.
    """

    def __init__(self, plan, path):
        self.plan = plan
        self.path = path
        self.ctx = path_to_ctx(path)
        self.src = SourceText.from_bytes(plan.base.read_bytes(path))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SyntaxWarning)
            self.tree = ast.parse(self.src.text, path)
        self.items = module_items(self.src, self.tree)
        block_ids = {id(i.node) for i in self.items
                     if i.kind == 'importblock'}
        self.bindings = import_bindings(self.tree, self.ctx,
                                        plan.base_is_module, block_ids)
        self.defined = {}
        for item in self.items:
            if item.kind in ('import', 'importblock', 'docstring'):
                continue
            for name in item.names:
                self.defined[name] = item
        self._analysis = None
        self._class_items = {}

    @property
    def analysis(self):
        if self._analysis is None:
            self._analysis = ModuleAnalysis(self.src.text, self.path,
                                            tree=self.tree)
        return self._analysis

    def item(self, key):
        return find_module_item(self.items, key)

    def free_names(self, item):
        """Global names that `item` reads (cached)."""
        if not hasattr(self, '_free'):
            import bisect
            self._free = {}
            spans = sorted((i.first, i.last, i.key) for i in self.items)
            starts = [s[0] for s in spans]
            for ref in self.analysis.refs:
                pos = bisect.bisect_right(starts, ref.lineno) - 1
                if pos < 0:
                    continue
                first, last, key = spans[pos]
                if first <= ref.lineno <= last:
                    self._free.setdefault(key, set()).add(ref.name)
        return self._free.get(item.key, set())

    def class_node(self, name):
        for node in self.tree.body:
            if isinstance(node, ast.ClassDef) and node.name == name:
                return node
        return None

    def class_items(self, name):
        if name not in self._class_items:
            node = self.class_node(name)
            if node is None:
                raise RefactorError(f'No class {name} in {self.path}')
            self._class_items[name] = class_items(self.src, node)
        return self._class_items[name]


class Plan:
    def __init__(self, repo, mapping_dir=None, base=None):
        self.repo = repo
        self.mapping_dir = mapping_dir or os.path.join(
            repo, 'tools', 'refactor', 'mapping')
        self.base = BaseTree(repo, base or base_commit())
        self.issues = []
        self.shards = []
        self.module_moves = []
        self.shims = {}             # old path -> {'kind', 'commit'}
        self.packages = {}          # path -> spec
        self.modules = {}           # path -> spec
        self.items = {}             # (old_file, key) -> ItemMove
        self.classes = {}           # (old_file, cls) -> ClassSplit
        self.plugin_api = {}        # old file -> [names]
        self.forwarders = {}        # module path -> {'table', 'commit'}
        self.package_compat_decl = {}   # package __init__ -> old module
        self.allow = []
        self._base_modules = {}
        self._load()

    # --------------------------------------------------------- loading
    def error(self, check, message, where=None):
        self.issues.append(Issue('error', check, message, where))

    def warning(self, check, message, where=None):
        self.issues.append(Issue('warning', check, message, where))

    def info(self, check, message, where=None):
        self.issues.append(Issue('info', check, message, where))

    def _load(self):
        if not os.path.isdir(self.mapping_dir):
            self.warning('schema', f'No mapping folder {self.mapping_dir}')
            return
        for name in sorted(os.listdir(self.mapping_dir)):
            if not name.endswith('.json'):
                continue
            path = os.path.join(self.mapping_dir, name)
            try:
                with open(path, encoding='utf-8') as file:
                    data = json.load(file)
            except (OSError, ValueError) as err:
                self.error('schema', f'Cannot read {name}: {err}')
                continue
            self._load_shard(name, data)

    def _commit_ok(self, commit, where):
        if commit not in COMMITS:
            self.error('schema', f'Unknown commit {commit!r}', where)
            return False
        return True

    def _load_shard(self, filename, data):
        if not isinstance(data, dict):
            self.error('schema', 'A shard must be a JSON object', filename)
            return
        shard = data.get('shard') or filename[:-5]
        self.shards.append(shard)
        for key in data:
            if key not in SHARD_KEYS:
                self.warning('schema', f'Unknown key {key!r}', filename)
        default = data.get('default_commit')
        if default is not None:
            self._commit_ok(default, f'{filename}: default_commit')
        for entry in data.get('module_moves') or []:
            where = f'{filename}: module_moves'
            if not isinstance(entry, dict) or not entry.get('old') \
                    or not entry.get('new'):
                self.error('schema', f'Bad module move {entry!r}', where)
                continue
            shim = entry.get('shim', 'alias')
            commit = entry.get('commit', default)
            if shim not in SHIM_KINDS:
                self.error('schema', f'Bad shim kind {shim!r}', where)
            if not commit or not self._commit_ok(commit, where):
                self.error('schema', f'No commit for {entry!r}', where)
                continue
            self.module_moves.append(
                ModuleMove(entry['old'], entry['new'], shim, commit, shard))
        for old, spec in (data.get('shims') or {}).items():
            where = f'{filename}: shims'
            spec = dict(spec or {})
            spec.setdefault('commit', default)
            if spec.get('kind') not in SHIM_KINDS:
                self.error('schema', f'Bad shim kind for {old}', where)
                continue
            if not spec.get('commit') or not self._commit_ok(spec['commit'],
                                                             where):
                continue
            if old in self.shims and self.shims[old] != spec:
                self.error('duplicate', f'Two shim specs for {old}', where)
            self.shims[old] = spec
        for path, spec in (data.get('packages') or {}).items():
            spec = dict(spec or {})
            spec.setdefault('commit', default)
            where = f'{filename}: packages'
            if not spec.get('commit') or not self._commit_ok(spec['commit'],
                                                             where):
                self.error('schema', f'No commit for package {path}', where)
                continue
            self._merge_spec(self.packages, path.rstrip('/'), spec, where)
        for path, spec in (data.get('modules') or {}).items():
            self._merge_spec(self.modules, path, dict(spec or {}),
                             f'{filename}: modules')
        for ref, value in (data.get('items') or {}).items():
            where = f'{filename}: items'
            old_file, sep, key = ref.partition('::')
            if not sep or not old_file.endswith('.py') or not key:
                self.error('schema', f'Bad item key {ref!r}', where)
                continue
            if isinstance(value, dict):
                dest = value.get('to')
                commit = value.get('commit', default)
            else:
                dest, commit = value, default
            if not isinstance(dest, str) or not dest:
                self.error('schema', f'Bad destination for {ref}', where)
                continue
            if not commit or not self._commit_ok(commit, where):
                self.error('schema', f'No commit for {ref}', where)
                continue
            if dest != DELETE and not dest.endswith('.py'):
                self.error('schema', f'Destination of {ref} is not a .py '
                           f'file: {dest}', where)
                continue
            move = ItemMove(old_file, key, None if dest == DELETE else dest,
                            commit, shard)
            if (old_file, key) in self.items:
                other = self.items[(old_file, key)]
                same = other.dest == move.dest and other.commit == commit
                (self.warning if same else self.error)(
                    'duplicate', f'{ref} is mapped twice (shards '
                    f'{other.shard} and {shard})'
                    + (' identically' if same else
                       f': {other.dest} {other.commit} vs {move.dest} '
                       f'{commit}'), where)
                continue
            self.items[(old_file, key)] = move
        for ref, spec in (data.get('classes') or {}).items():
            where = f'{filename}: classes'
            old_file, sep, cls = ref.partition('::')
            if not sep or not isinstance(spec, dict):
                self.error('schema', f'Bad class key {ref!r}', where)
                continue
            split = ClassSplit(old_file, cls, spec, shard, default)
            for mixin, mspec in split.mixins.items():
                if not mspec.get('commit') or not self._commit_ok(
                        mspec['commit'], where):
                    self.error('schema', f'No commit for mixin {mixin}',
                               where)
            for meth, mixin in split.methods.items():
                if mixin not in split.mixins:
                    self.error('schema', f'{ref}.{meth}: unknown mixin '
                               f'{mixin[0]}::{mixin[1]}', where)
            if (old_file, cls) in self.classes:
                self.error('duplicate', f'Class {ref} is split in two '
                           'shards', where)
                continue
            self.classes[(old_file, cls)] = split
        api = data.get('plugin_api')
        if isinstance(api, dict):
            for file, names in api.items():
                self.plugin_api.setdefault(file, []).extend(names)
        elif isinstance(api, list):
            sources = {}
            for (old_file, _), move in self.items.items():
                if move.shard == shard:
                    sources[old_file] = sources.get(old_file, 0) + 1
            if sources:
                main = max(sorted(sources), key=lambda f: sources[f])
                self.plugin_api.setdefault(main, []).extend(api)
        for entry in data.get('allow') or []:
            self.allow.append(entry)
        for path, spec in (data.get('attribute_forwarders') or {}).items():
            where = f'{filename}: attribute_forwarders'
            spec = dict(spec or {})
            spec.setdefault('commit', default)
            if not spec.get('table') or not spec.get('commit'):
                self.error('schema', f'Bad forwarder {path}', where)
                continue
            self._commit_ok(spec['commit'], where)
            self.forwarders[path] = spec
        for path, old in (data.get('package_compat') or {}).items():
            self.package_compat_decl[path] = old
            pkg = path[:-len('/__init__.py')] if path.endswith(
                '/__init__.py') else path
            if old != pkg + '.py':
                self.error('schema', f'package_compat {path}: {old} is not '
                           'the module of the same name', filename)

    def _merge_spec(self, target, path, spec, where):
        if path not in target:
            target[path] = spec
            return
        old = target[path]
        for key, value in spec.items():
            if key in old and old[key] != value and value not in (None, ''):
                if old[key] in (None, ''):
                    old[key] = value
                elif key == 'doc':
                    self.warning('duplicate', f'Two docstrings for {path}; '
                                 'keeping the first', where)
                else:
                    self.error('duplicate', f'Conflicting {key!r} for '
                               f'{path}: {old[key]!r} vs {value!r}', where)
            elif key not in old:
                old[key] = value

    # ------------------------------------------------------ base access
    def base_is_module(self, dotted_name):
        base = dotted_name.replace('.', '/')
        return (self.base.exists(base + '.py')
                or self.base.exists(base + '/__init__.py'))

    def base_module(self, path):
        if path not in self._base_modules:
            self._base_modules[path] = BaseModule(self, path)
        return self._base_modules[path]

    # -------------------------------------------------------- locations
    def moves_upto(self, commit, inclusive=True):
        moves = []
        for move in self.module_moves:
            ok = (commit_le(move.commit, commit) if inclusive
                  else commit_lt(move.commit, commit))
            if ok:
                moves.append(move)
        return sorted(moves, key=lambda m: commit_index(m.commit))

    def module_path(self, path, commit, inclusive=True):
        """Where the module at base path `path` is after (inclusive) or
        before `commit`'s module moves."""
        for move in self.moves_upto(commit, inclusive):
            if path == move.old:
                path = move.new
            elif path.startswith(move.old + '/'):
                path = move.new + path[len(move.old):]
        return path

    def item_move(self, old_file, key):
        return self.items.get((old_file, key))

    def item_location(self, old_file, key, commit, inclusive=True):
        """Path where the item is after (or before) `commit`; None if it
        was deleted."""
        move = self.items.get((old_file, key))
        if move is not None:
            done = (commit_le(move.commit, commit) if inclusive
                    else commit_lt(move.commit, commit))
            if done:
                if move.dest is None:
                    return None
                return self.module_path(move.dest, commit, inclusive)
        return self.module_path(old_file, commit, inclusive)

    def moves_for_commit(self, commit):
        return [m for m in self.module_moves if m.commit == commit]

    def items_for_commit(self, commit):
        return [m for m in self.items.values() if m.commit == commit]

    def packages_for_commit(self, commit):
        return {p: s for p, s in self.packages.items()
                if s.get('commit') == commit}

    def class_splits_for_commit(self, commit):
        out = []
        for split in self.classes.values():
            methods = [m for m in split.methods
                       if split.method_commit(m) == commit]
            if methods:
                out.append((split, methods))
        return out

    def old_module_files(self):
        """Every base .py file that the mapping moves, splits or shims."""
        files = set()
        for move in self.module_moves:
            if move.is_python_file:
                files.add(move.old)
            else:
                prefix = move.old + '/'
                files.update(f for f in self.base.files()
                             if f.startswith(prefix) and f.endswith('.py'))
        files.update(old for old, _ in self.items)
        files.update(old for old, _ in self.classes)
        files.update(p for p in self.shims if p.endswith('.py'))
        return sorted(f for f in files if self.base.exists(f))

    def module_move_of(self, path):
        for move in self.module_moves:
            if move.old == path:
                return move
        return None

    def replaced_by_package(self, old_path, commit):
        """The package path that replaces the old module `old_path`
        ('cellacdc/gui.py' -> 'cellacdc/gui') by `commit`, or None."""
        pkg = old_path[:-3]
        spec = self.packages.get(pkg)
        if spec and commit_le(spec['commit'], commit):
            return pkg
        return None

    def shim_kind(self, old_path, commit):
        """What the old dotted name of `old_path` resolves to at `commit`:
        'real' (the module itself), 'alias' (the moved module), 'split'
        (a lazy shim), 'package' (a package with package_compat) or
        'none' (nothing)."""
        if self.replaced_by_package(old_path, commit):
            return 'package'
        spec = self.shims.get(old_path)
        move = self.module_move_of(old_path)
        if move is not None and commit_le(move.commit, commit):
            kind = spec['kind'] if spec and commit_le(
                spec['commit'], commit) else move.shim
            return kind
        if spec and commit_le(spec['commit'], commit):
            return spec['kind']
        return 'real'

    def remap_module(self, dotted_name, commit, inclusive=True):
        """Dotted name that a reference to module `dotted_name` (a base
        module) should use at `commit`: the new path for modules moved
        with an alias shim or without shim, otherwise unchanged (the old
        dotted name keeps working through its shim)."""
        for path in (dotted_name.replace('.', '/') + '.py',
                     dotted_name.replace('.', '/')):
            move = self.module_move_of(path)
            if move is None:
                continue
            done = (commit_le(move.commit, commit) if inclusive
                    else commit_lt(move.commit, commit))
            if not done:
                return dotted_name
            kind = self.shim_kind(move.old, commit) if move.is_python_file \
                else move.shim
            if kind in ('alias', 'none'):
                return dotted(self.module_path(path, commit, inclusive))
            return dotted_name
        # modules inside a moved folder
        for move in self.moves_upto(commit, inclusive):
            prefix = move.old.replace('/', '.') + '.'
            if not move.is_python_file and dotted_name.startswith(prefix):
                return move.new.replace('/', '.') + dotted_name[
                    len(prefix) - 1:]
        return dotted_name

    # ------------------------------------------------------- aliases
    def alias_of(self, dotted_name):
        for path in (dotted_name.replace('.', '/') + '.py',
                     dotted_name.replace('.', '/') + '/__init__.py',
                     dotted_name.replace('.', '/')):
            spec = self.modules.get(path)
            if spec and spec.get('alias'):
                return spec['alias']
            spec = self.packages.get(path)
            if spec and spec.get('alias'):
                return spec['alias']
        return default_alias(dotted_name)

    def has_alias(self, dotted_name):
        for path in (dotted_name.replace('.', '/') + '.py',
                     dotted_name.replace('.', '/') + '/__init__.py',
                     dotted_name.replace('.', '/')):
            for table in (self.modules, self.packages):
                if table.get(path, {}).get('alias'):
                    return True
        return False

    # -------------------------------------------------------- name table
    def _base_key_of_name(self, bmod, name):
        item = bmod.defined.get(name)
        return item.key if item is not None else None

    def resolve_name(self, old_dotted, name, commit, _seen=None):
        """(home dotted, attr or None for module values) of `name` of the
        base module `old_dotted` at `commit`. Follows re-exports of
        cellacdc names to the module that defines them."""
        _seen = _seen or set()
        if (old_dotted, name) in _seen:
            return (old_dotted, name)
        _seen.add((old_dotted, name))
        path = self._base_path(old_dotted)
        if path is None:
            return (self.remap_module(old_dotted, commit), name)
        bmod = self.base_module(path)
        if name in bmod.defined:
            key = bmod.defined[name].key
            loc = self.item_location(path, key, commit)
            if loc is None:
                return None
            return (dotted(loc), name)
        bindings = bmod.bindings.get(name)
        if not bindings:
            return (self.remap_module(old_dotted, commit), name)
        b = bindings[-1]
        if b.is_module_value:
            if b.attr is None:
                if b.alias.asname is None and '.' in b.target:
                    target = b.target.split('.')[0]
                else:
                    target = b.target
            else:
                target = f'{b.target}.{b.attr}'
            if target.split('.')[0] == 'cellacdc':
                target = self.remap_module(target, commit)
            return (target, None)
        if b.target.split('.')[0] == 'cellacdc':
            return self.resolve_name(b.target, b.attr, commit, _seen)
        return None     # third party: decided by the caller

    def _base_path(self, dotted_name):
        base = dotted_name.replace('.', '/')
        for path in (base + '.py', base + '/__init__.py'):
            if self.base.exists(path):
                return path
        return None

    def _binder(self, bmod, name, commit):
        """New module that binds the third-party name `name` of the split
        base module `bmod` (the destination receiving most of the items
        that use it)."""
        counts = {}
        for item in bmod.items:
            if item.kind in ('import', 'docstring'):
                continue
            loc = self.item_location(bmod.path, item.key, commit)
            if loc is None or loc == bmod.path:
                continue
            if name in bmod.free_names(item):
                counts[loc] = counts.get(loc, 0) + 1
        if not counts:
            return None
        return max(sorted(counts), key=lambda p: counts[p])

    def name_entries(self, old_path, commit):
        """{name: (home dotted, attr|None)} for the base module
        `old_path` at `commit` (the rewrite view: names defined in
        cellacdc go to the module that defines them)."""
        return {n: v[0] for n, v in self._name_entries(old_path,
                                                       commit).items()}

    def runtime_entries(self, old_path, commit):
        """Like name_entries, but every import-bound name resolves through
        the module that binds it now (patches of the binding module stay
        visible through the old path; compat tables)."""
        return {n: v[1] for n, v in self._name_entries(old_path,
                                                       commit).items()}

    def _name_entries(self, old_path, commit):
        key = (old_path, commit)
        cache = self.__dict__.setdefault('_entries_cache', {})
        if key not in cache:
            cache[key] = self._compute_name_entries(old_path, commit)
        return cache[key]

    def _compute_name_entries(self, old_path, commit):
        """{name: ((home, attr), (runtime home, attr))}: every top-level
        name that the module the old dotted name now resolves to no longer
        defines."""
        kind = self.shim_kind(old_path, commit)
        bmod = self.base_module(old_path)
        if kind == 'real':
            resolved = old_path
        elif kind == 'alias':
            resolved = self.module_path(old_path, commit)
        else:
            resolved = None
        entries = {}
        moved_module = self.module_move_of(old_path)
        moved = moved_module is not None and commit_le(
            moved_module.commit, commit)
        submodules = set()
        if kind == 'package':
            submodules = self.package_submodules(old_path[:-3], commit)
        for name, item in bmod.defined.items():
            if name.startswith('__') and name.endswith('__'):
                continue
            loc = self.item_location(old_path, item.key, commit)
            if loc is None:
                continue
            if resolved is not None and loc == resolved:
                continue
            if name in submodules:
                continue
            entries[name] = ((dotted(loc), name),) * 2
        if resolved is not None:
            return entries
        for name, bindings in bmod.bindings.items():
            if name in bmod.defined or name in submodules:
                continue
            if name.startswith('__') and name.endswith('__'):
                continue
            b = bindings[-1]
            if b.guard is not None:
                guard_item = next((i for i in bmod.items
                                   if i.node is b.guard), None)
                move = guard_item and self.items.get((old_path,
                                                      guard_item.key))
                if move is not None and move.dest is None and commit_le(
                        move.commit, commit):
                    continue    # bound by a deleted import block
            if b.attr is None:
                source = (b.target if b.alias.asname
                          else b.target.split('.')[0], None)
            else:
                source = (b.target, b.attr)
            cellacdc_target = b.target.split('.')[0] == 'cellacdc'
            defining = None
            if cellacdc_target or b.is_module_value:
                defining = self.resolve_name(dotted(old_path), name, commit)
            if moved:
                binder = dotted(self.module_path(old_path, commit))
            else:
                binder = self._binder(bmod, name, commit)
                binder = dotted(binder) if binder else None
            runtime = (binder, name) if binder else (defining or source)
            if cellacdc_target and defining is not None:
                home = defining
            elif binder:
                home = (binder, name)
            else:
                home = defining or source
            entries[name] = (home, runtime)
        return entries

    def reexport_entries(self, old_path, commit):
        """{name: (home dotted, attr|None)}: names that the base module
        `old_path` bound by importing them from cellacdc and that the
        name entries leave out, for the rewrite view only (never in the
        compat tables):

        - modules moved whole (alias/none shim): every cellacdc import of
          the old module (`gui_custom_annotations.apps` is the old
          cellacdc.apps; the C12 rewrite replaces those imports in the
          moved module by the homes of the names it uses);
        - old modules replaced by a same-name package: the old globals
          that clash with a new submodule (`measurements_viewer.widgets`
          was the old cellacdc.widgets, `cellacdc.measurements_viewer.
          widgets` is a new module: only attribute access through an
          import binding means the old global).

        Module objects map to (module, None), other names to the module
        that defines them (resolve_name)."""
        kind = self.shim_kind(old_path, commit)
        if kind in ('alias', 'none'):
            clashing = None
        elif kind == 'package':
            clashing = self.package_submodules(old_path[:-3], commit)
        else:
            return {}
        bmod = self.base_module(old_path)
        known = self._name_entries(old_path, commit)
        out = {}
        for name, bindings in sorted(bmod.bindings.items()):
            if name in bmod.defined or name in known:
                continue
            if name.startswith('__') and name.endswith('__'):
                continue
            if clashing is not None and name not in clashing:
                continue
            b = bindings[-1]
            if b.target.split('.')[0] != 'cellacdc':
                continue
            resolved = self.resolve_name(dotted(old_path), name, commit)
            if resolved is None:
                continue
            out[name] = resolved
        return out

    def code_destinations(self, old_file, commit):
        """Sorted dotted names of the modules other than the old dotted
        name that hold code of the base module `old_file` after `commit`:
        where it moved whole and every destination of its items and of
        its split classes' mixins (the modules whose globals a patch of
        the old module must reach; shims.patch_fanout, the compat layer's
        PATCH_FANOUT, is the same set without single destinations)."""
        mods = set()
        here = self.module_path(old_file, commit)
        if here.endswith('.py'):
            mods.add(dotted(here))
        for move in self.items.values():
            if move.old_file == old_file and move.dest and                     commit_le(move.commit, commit):
                mods.add(dotted(self.module_path(move.dest, commit)))
        for split in self.classes.values():
            if split.old_file != old_file:
                continue
            for (file, _name), spec in split.mixins.items():
                if commit_le(spec.get('commit') or COMMITS[-1], commit):
                    mods.add(dotted(self.module_path(file, commit)))
        mods.discard(dotted(old_file))
        return sorted(mods)

    def package_submodules(self, pkg_path, commit):
        """Names of the submodules/subpackages of a package at `commit`
        (from the plan: package specs, module moves and item
        destinations)."""
        names = set()
        prefix = pkg_path + '/'
        paths = {p for p, s in self.packages.items()
                 if commit_le(s['commit'], commit)}
        paths |= {p for p, s in self.modules.items()
                  if s.get('commit') and s.get('code')
                  and commit_le(s['commit'], commit)}
        paths |= {self.module_path(m.old, commit) for m in
                  self.moves_upto(commit)}
        paths |= {m.dest for m in self.items.values()
                  if m.dest and commit_le(m.commit, commit)}
        for split in self.classes.values():
            for (file, _), spec in split.mixins.items():
                if commit_le(spec.get('commit') or 'C16', commit):
                    paths.add(file)
        for path in paths:
            if path and path.startswith(prefix):
                rest = path[len(prefix):]
                head = rest.split('/')[0]
                if head.endswith('.py'):
                    head = head[:-3]
                if head and head != '__init__':
                    names.add(head)
        # existing folders/files at base under the package
        for f in self.base.files():
            if f.startswith(prefix):
                head = f[len(prefix):].split('/')[0]
                if head.endswith('.py'):
                    head = head[:-3]
                if head and head != '__init__':
                    names.add(head)
        return names

    def item_locations_history(self, old_file, key, commit):
        """Distinct locations of an item from the base up to `commit`."""
        locs = [old_file]
        for c in MOVE_COMMITS:
            if not commit_le(c, commit):
                break
            loc = self.item_location(old_file, key, c)
            if loc is not None and loc != locs[-1]:
                locs.append(loc)
            if loc is None:
                break
        return locs

    def intermediate_entries(self, commit):
        """{module dotted: {name: (home, name)}} for modules that exist at
        `commit` at a path other than a base path and lost names."""
        out = {}
        old_files = self.old_module_files()
        base_paths = set(old_files)
        for old_file in old_files:
            bmod = self.base_module(old_file)
            for name, item in bmod.defined.items():
                hist = self.item_locations_history(old_file, item.key,
                                                   commit)
                final = self.item_location(old_file, item.key, commit)
                if final is None:
                    continue
                for loc in hist[1:]:
                    if loc != final and loc not in base_paths:
                        out.setdefault(dotted(loc), {})[name] = (
                            dotted(final), name)
        return out

    def name_table(self, commit=None):
        commit = commit or COMMITS[-1]
        table = {
            'commit': commit, 'base': self.base.commit, 'modules': {},
            'names': {}, 'entries': {}, 'module_values': {},
            'reexports': {}, 'fanout': {}, 'defined': {},
            'intermediate': {}, 'aliases': {}, 'packages': [],
            'plugin_api': {}, 'classes': {}, 'shims': {},
        }
        for move in self.moves_upto(commit):
            if move.is_python_file:
                table['modules'][dotted(move.old)] = dotted(move.new)
            else:
                for f in self.base.files():
                    if f.startswith(move.old + '/') and f.endswith('.py'):
                        table['modules'][dotted(f)] = dotted(
                            move.new + f[len(move.old):])
        homes = set()
        table['runtime'] = {}
        for old_file in self.old_module_files():
            od = dotted(old_file)
            fanout = self.code_destinations(old_file, commit)
            if fanout:
                table['fanout'][od] = fanout
            table['defined'][od] = sorted(
                n for n in self.base_module(old_file).defined
                if not (n.startswith('__') and n.endswith('__')))
            reexports = self.reexport_entries(old_file, commit)
            if reexports:
                table['reexports'][dotted(old_file)] = {
                    n: list(v) for n, v in reexports.items()}
                homes.update(h for h, _ in reexports.values())
            entries = self.name_entries(old_file, commit)
            if not entries:
                continue
            od = dotted(old_file)
            table['names'][od] = {n: h for n, (h, _) in entries.items()}
            table['entries'][od] = {n: [h, a] for n, (h, a)
                                    in entries.items()}
            table['runtime'][od] = {
                n: list(v) for n, v in self.runtime_entries(
                    old_file, commit).items()}
            mv = {n: h for n, (h, a) in entries.items() if a is None}
            if mv:
                table['module_values'][od] = mv
            homes.update(h for h, _ in entries.values())
            table['shims'][od] = self.shim_kind(old_file, commit)
        inter = self.intermediate_entries(commit)
        table['intermediate'] = {m: {n: [h, a] for n, (h, a) in e.items()}
                                 for m, e in inter.items()}
        for path, spec in self.packages.items():
            if commit_le(spec['commit'], commit):
                table['packages'].append(dotted(path))
        table['packages'].sort()
        aliased = set(homes)
        aliased.update(dotted(p) for p in self.modules)
        aliased.update(dotted(p) for p in self.packages)
        aliased.update(table['modules'].values())
        aliased.update(dotted(m.dest) for m in self.items.values()
                       if m.dest)
        for name in sorted(aliased):
            if name.split('.')[0] != 'cellacdc':
                continue
            table['aliases'][name] = self.alias_of(name)
        for file, names in self.plugin_api.items():
            table['plugin_api'][dotted(file)] = sorted(set(names))
        for (old_file, cls), split in self.classes.items():
            table['classes'][f'{dotted(old_file)}.{cls}'] = {
                'host': dotted(split.host) if split.host else None,
                'methods': {
                    m: f'{dotted(f)}.{mx}' for m, (f, mx)
                    in split.methods.items()
                    if commit_le(split.method_commit(m) or 'C16', commit)
                },
            }
        return table

    # --------------------------------------------------------- coverage
    def fully_split_modules(self):
        """Base modules that disappear as code: split shim (without module
        move) or replaced by a same-name package."""
        out = []
        for path in self.old_module_files():
            if self.module_move_of(path):
                continue
            spec = self.shims.get(path)
            if (spec and spec['kind'] in ('split', 'package', 'none')) or \
                    self.packages.get(path[:-3]):
                out.append(path)
        return out
