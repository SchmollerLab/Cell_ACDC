"""Static checks of the mapping (no tree is modified).

Errors (exit code 1 unless allowed by a shard's "allow" entry):

- schema / duplicate:  shard format, items or modules mapped twice;
- keys:       item or class keys that do not exist at the base commit;
- coverage:   fully split modules (split shim or replaced by a package)
              with unmapped items;
- dests:      destinations at old paths that become shims, folders
              without a package entry, gitignored new paths;
- paths:      a module file and a package with the same dotted name;
- aliases:    two modules with one alias, an alias that collides with a
              name used in a file that would import it;
- cycles:     eager import cycles in the final layout (an import-time use
              of a module that is in the same import cycle);
- mixins:     methods moved into mixins that are Qt names (dir(QMainWindow)
              of PyQt5 and PyQt6), use zero-argument super(), __class__,
              name mangling or `global`, use class-scope names in their
              decorators/defaults, are referenced by class-body statements
              (properties), would no longer override a base class method,
              or a bases list that drops a base or misses a mixin;
- globals:    a `global X` writer that ends up apart from X or its readers;
- names:      name-table homes that do not exist in the final layout.

Warnings: module sizes outside 150-3000 lines, derived aliases, mixins
shared by two hosts, plugin API names that do not exist.
"""
import ast
import json
import os
import subprocess
import sys

from . import COMMITS, MOVE_COMMITS, RefactorError, TOOL_DIR, commit_le
from .gitutil import is_ignored
from .plan import Plan, Issue, dotted
from .scopes import is_shadowed

FINAL = COMMITS[-1]
QT_NAMES_FILE = os.path.join(TOOL_DIR, 'qt_names.json')
EXTRA_QT_NAMES = ['exec_', 'exec', 'print_']
SIZE_MIN, SIZE_MAX = 150, 3000
SKIP_GRAPH_DIRS = ('cellacdc/models/', 'cellacdc/trackers/',
                   'cellacdc/promptable_models/', 'cellacdc/docs/',
                   'cellacdc/_deprecated/', 'cellacdc/scripts/',
                   'cellacdc/_profile/', 'cellacdc/resources/',
                   'cellacdc/precompiled/')


def compute_qt_names():
    """dir(QMainWindow) with the installed Qt binding (offscreen)."""
    code = ('import os, json; os.environ.setdefault("QT_QPA_PLATFORM", '
            '"offscreen"); from qtpy.QtWidgets import QMainWindow; '
            'print(json.dumps(sorted(set(dir(QMainWindow)))))')
    proc = subprocess.run([sys.executable, '-c', code], capture_output=True,
                          text=True, timeout=120)
    if proc.returncode != 0:
        raise RefactorError(f'Cannot list Qt names: {proc.stderr[-500:]}')
    return json.loads(proc.stdout)


def qt_names():
    if os.path.exists(QT_NAMES_FILE):
        with open(QT_NAMES_FILE, encoding='utf-8') as file:
            data = json.load(file)
        names = set()
        for value in data.values():
            if isinstance(value, list):
                names.update(value)
        return names
    names = set(compute_qt_names()) | set(EXTRA_QT_NAMES)
    return names


class Checker:
    def __init__(self, plan, commit=FINAL, deep=True):
        self.plan = plan
        self.commit = commit
        self.deep = deep
        self.issues = list(plan.issues)

    def error(self, check, message, where=None):
        self.issues.append(Issue('error', check, message, where))

    def warning(self, check, message, where=None):
        self.issues.append(Issue('warning', check, message, where))

    def info(self, check, message, where=None):
        self.issues.append(Issue('info', check, message, where))

    # ------------------------------------------------------------- run
    def run(self):
        steps = [self.check_keys, self.check_coverage, self.check_dests,
                 self.check_paths, self.check_aliases, self.check_mixins,
                 self.check_globals, self.check_plugin_api,
                 self.check_names, self.check_sizes]
        if self.deep:
            steps.append(self.check_cycles)
        for step in steps:
            try:
                step()
            except RefactorError as err:
                self.error(step.__name__, str(err))
        self.apply_allow()
        return self.issues

    def apply_allow(self):
        for issue in self.issues:
            if issue.level != 'error':
                continue
            for entry in self.plan.allow:
                if entry.get('check') not in (None, issue.check):
                    continue
                match = entry.get('match', '')
                if match and match not in issue.message:
                    continue
                issue.level = 'warning'
                issue.message += f' (allowed: {entry.get("reason", "")})'
                break

    # ----------------------------------------------------------- keys
    def check_keys(self):
        plan = self.plan
        for (old_file, key), move in sorted(plan.items.items()):
            if not plan.base.exists(old_file):
                self.error('keys', f'{old_file} does not exist at the base',
                           move.shard)
                continue
            bmod = plan.base_module(old_file)
            if bmod.item(key) is None:
                near = [i.key for i in bmod.items
                        if i.key.split(' #')[0].lower() ==
                        key.split(' #')[0].lower()][:3]
                report = (self.error if move.commit in MOVE_COMMITS
                          else self.warning)
                report('keys', f'{old_file}::{key} not found'
                       + (f' (did you mean {near}?)' if near else '')
                       + ('' if move.commit in MOVE_COMMITS else
                          f' (commit {move.commit}: done by a fixup)'),
                       move.shard)
        for (old_file, cls), split in sorted(plan.classes.items()):
            if not plan.base.exists(old_file):
                self.error('keys', f'{old_file} does not exist', split.shard)
                continue
            bmod = plan.base_module(old_file)
            if bmod.class_node(cls) is None:
                self.error('keys', f'No class {cls} in {old_file}',
                           split.shard)
                continue
            members = {m.key for m in bmod.class_items(cls)
                       if m.kind == 'method'}
            for meth in split.methods:
                if meth not in members:
                    self.error('keys', f'{old_file}::{cls}.{meth} is not a '
                               'method of the class', split.shard)

    # ------------------------------------------------------- coverage
    def check_coverage(self):
        plan = self.plan
        for path in plan.fully_split_modules():
            bmod = plan.base_module(path)
            missing = [i.key for i in bmod.items
                       if not i.optional and (path, i.key) not in plan.items]
            if missing:
                self.error('coverage', f'{path}: {len(missing)} unmapped '
                           f'items: {missing[:15]}'
                           + (' ...' if len(missing) > 15 else ''))
            commits = {plan.items[(path, i.key)].commit for i in bmod.items
                       if (path, i.key) in plan.items}
            shim = plan.shims.get(path)
            if shim and commits and any(
                    not commit_le(c, shim['commit']) for c in commits):
                self.error('coverage', f'{path}: items move after its shim '
                           f'commit {shim["commit"]}: {sorted(commits)}')

    # ---------------------------------------------------------- dests
    def _base_package(self, folder):
        return self.plan.base.exists(folder + '/__init__.py')

    def check_dests(self):
        plan = self.plan
        shim_paths = set(plan.shims) | {m.old for m in plan.module_moves}
        new_paths = set()
        for move in plan.items.values():
            if move.dest is None:
                continue
            new_paths.add(move.dest)
            if move.dest in shim_paths:
                self.error('dests', f'{move.ref} -> {move.dest}, an old '
                           'path that becomes a shim', move.shard)
            self._check_packages(move.dest, move.commit, move.ref)
        for split in plan.classes.values():
            for (file, name), spec in split.mixins.items():
                new_paths.add(file)
                self._check_packages(file, spec.get('commit') or FINAL,
                                     f'{split.ref} mixin {name}')
        for move in plan.module_moves:
            new_paths.add(move.new)
            # a moved folder needs packages for its parents only (it may
            # hold data files)
            self._check_packages(move.new, move.commit,
                                 f'module move {move.old}')
        for path in plan.packages:
            new_paths.add(path + '/__init__.py')
        ignored = is_ignored(plan.repo, sorted(new_paths))
        for path in sorted(ignored):
            self.error('dests', f'{path} is ignored by .gitignore')

    def _check_packages(self, path, commit, ref):
        plan = self.plan
        parts = path.split('/')[:-1]
        for i in range(2, len(parts) + 1):
            folder = '/'.join(parts[:i])
            if self._base_package(folder):
                continue
            spec = plan.packages.get(folder)
            if spec is None:
                self.error('dests', f'{folder} has no package entry (needed '
                           f'by {ref})')
            elif not commit_le(spec['commit'], commit):
                self.error('dests', f'package {folder} is created in '
                           f'{spec["commit"]}, after {ref} ({commit})')

    # ---------------------------------------------------------- paths
    def final_modules(self):
        """Paths of the Python modules of the final layout that the plan
        creates or keeps at old paths (shims)."""
        plan = self.plan
        paths = set()
        for move in plan.items.values():
            if move.dest:
                paths.add(plan.module_path(move.dest, FINAL))
        for split in plan.classes.values():
            for (file, _) in split.mixins:
                paths.add(file)
        for move in plan.module_moves:
            paths.add(move.new if move.is_python_file else move.new + '/')
        for path in plan.packages:
            paths.add(path + '/__init__.py')
        for old, spec in plan.shims.items():
            if spec['kind'] in ('alias', 'split'):
                paths.add(old)
        for move in plan.module_moves:
            if move.is_python_file and move.shim in ('alias', 'split') and \
                    move.old not in plan.shims:
                paths.add(move.old)
        return paths

    def check_paths(self):
        plan = self.plan
        modules = self.final_modules()
        packages = {p for p in plan.packages}
        packages |= {m.new for m in plan.module_moves
                     if not m.is_python_file}
        for path in sorted(modules):
            if not path.endswith('.py'):
                continue
            folder = path[:-3]
            if folder in packages or self._base_package(folder):
                self.error('paths', f'{path} and the package {folder}/ have '
                           'the same dotted name')
        # names of package-replaced modules equal to submodules
        for path, spec in plan.packages.items():
            if not plan.base.exists(path + '.py'):
                continue
            subs = plan.package_submodules(path, FINAL)
            bmod = plan.base_module(path + '.py')
            clash = sorted(set(bmod.defined) & subs)
            clash += sorted(set(bmod.bindings) & subs - set(clash))
            for name in clash:
                self.info('paths', f'{dotted(path)}.{name} (old name) is now '
                          'the submodule of the same name; not forwarded')

    # -------------------------------------------------------- aliases
    def check_aliases(self):
        plan = self.plan
        by_alias = {}
        modules = set()
        for path in plan.modules:
            modules.add(dotted(path))
        for move in plan.items.values():
            if move.dest:
                modules.add(dotted(plan.module_path(move.dest, FINAL)))
        for move in plan.module_moves:
            if move.is_python_file:
                modules.add(dotted(move.new))
        for split in plan.classes.values():
            for (file, _) in split.mixins:
                modules.add(dotted(file))
        for module in sorted(modules):
            alias = plan.alias_of(module)
            by_alias.setdefault(alias, []).append(module)
            if not plan.has_alias(module) and module.count('.') >= 2:
                self.warning('aliases', f'{module} has no alias in the '
                             f'mapping (derived: {alias})')
        for alias, mods in sorted(by_alias.items()):
            if len(mods) > 1:
                self.error('aliases', f'alias {alias!r} is used by '
                           f'{", ".join(mods)}')
            if not alias.isidentifier():
                self.error('aliases', f'alias {alias!r} of {mods[0]} is not '
                           'an identifier')
        self._check_alias_collisions()

    def incoming(self):
        """{final module path: [(bmod, item)]} of the moved items, plus
        the module-moved files (all their items)."""
        plan = self.plan
        out = {}
        for (old_file, key), move in plan.items.items():
            if move.dest is None or not plan.base.exists(old_file):
                continue
            bmod = plan.base_module(old_file)
            item = bmod.item(key)
            if item is None:
                continue
            out.setdefault(plan.module_path(move.dest, FINAL), []).append(
                (bmod, item))
        return out

    def _needed_homes(self, bmod, item, dest):
        """{home dotted: [names]} that `item` (moving to `dest`) reaches
        through a module alias."""
        plan = self.plan
        homes = {}
        for name in bmod.free_names(item):
            def_item = bmod.defined.get(name)
            if def_item is None:
                continue
            loc = plan.item_location(bmod.path, def_item.key, FINAL)
            if loc is None or loc == dest:
                continue
            homes.setdefault(dotted(loc), []).append(name)
        return homes

    def _check_alias_collisions(self):
        """An alias collides when the file that imports it defines or
        imports the same name at module level, or when a qualified
        reference sits in a scope where the alias is a local name."""
        plan = self.plan
        reported = set()
        # (bmod, item or member, final module) of every piece of code
        units = {}
        for dest, pairs in self.incoming().items():
            for bmod, item in pairs:
                units.setdefault(dest, []).append((bmod, item))
        for split in plan.classes.values():
            if not plan.base.exists(split.old_file):
                continue
            bmod = plan.base_module(split.old_file)
            if bmod.class_node(split.cls) is None:
                continue
            for member in bmod.class_items(split.cls):
                ref = split.methods.get(member.key)
                if ref:
                    units.setdefault(ref[0], []).append((bmod, member))
        for dest, pairs in sorted(units.items()):
            top_names = set()
            for bmod, item in pairs:
                if item.kind not in ('method', 'classstmt'):
                    top_names.update(item.names)
            for bmod, item in pairs:
                for ref in bmod.analysis.refs_in(item.first, item.last):
                    if ref.ctx != 'load':
                        continue
                    def_item = bmod.defined.get(ref.name)
                    if def_item is None:
                        continue
                    loc = plan.item_location(bmod.path, def_item.key, FINAL)
                    if loc is None or loc == dest:
                        continue
                    alias = plan.alias_of(dotted(loc))
                    key = (dest, alias, ref.lineno, bmod.path)
                    if key in reported:
                        continue
                    if alias in top_names:
                        reported.add(key)
                        self.error('aliases', f'alias {alias!r} of '
                                   f'{dotted(loc)} is also a top-level name '
                                   f'of {dest}')
                    elif is_shadowed(ref, alias):
                        reported.add(key)
                        self.error('aliases', f'{bmod.path}:{ref.lineno}: '
                                   f'{ref.name} becomes {alias}.{ref.name} '
                                   f'in {dest}, but {alias!r} is a local '
                                   'name there')

    # --------------------------------------------------------- mixins
    def check_mixins(self):
        plan = self.plan
        qt = qt_names()
        mixin_hosts = {}
        for (old_file, cls), split in sorted(plan.classes.items()):
            if not plan.base.exists(old_file):
                continue
            bmod = plan.base_module(old_file)
            node = bmod.class_node(cls)
            if node is None:
                continue
            for mixin in split.mixins:
                prev = mixin_hosts.setdefault(mixin, split.host)
                if prev != split.host:
                    self.error('mixins', f'mixin {mixin[1]} gets methods of '
                               f'two hosts ({prev}, {split.host})')
            members = bmod.class_items(cls)
            moved = set(split.methods)
            analysis = bmod.analysis
            class_locals = set()
            for member in members:
                class_locals.update(member.names)
            for member in members:
                if member.kind != 'method':
                    # class-body statements must not use moved methods
                    used = {n.id for n in ast.walk(member.node)
                            if isinstance(n, ast.Name)}
                    for name in sorted(used & moved):
                        self.error('mixins', f'{cls}: class-body statement '
                                   f'{member.key!r} uses the moved method '
                                   f'{name}', old_file)
                    continue
                if member.key not in moved:
                    continue
                where = f'{old_file}::{cls}.{member.key}'
                if member.key in qt:
                    self.error('mixins', f'{member.key} is a QMainWindow '
                               'attribute: it must stay in the class body',
                               where)
                seen_hazards = set()
                for hazard in analysis.hazards_in(member.first, member.last):
                    if (hazard.kind, hazard.name) in seen_hazards:
                        continue
                    seen_hazards.add((hazard.kind, hazard.name))
                    if hazard.kind == 'super0':
                        self.error('mixins', 'uses zero-argument super()',
                                   where)
                    elif hazard.kind == '__class__':
                        self.error('mixins', 'uses __class__', where)
                    elif hazard.kind == 'mangled':
                        self.error('mixins', f'uses the mangled name '
                                   f'{hazard.name}', where)
                    elif hazard.kind == 'global':
                        self.error('mixins', f'uses global {hazard.name}',
                                   where)
                node_m = member.node
                exprs = list(node_m.decorator_list) + \
                    list(node_m.args.defaults) + \
                    [d for d in node_m.args.kw_defaults if d is not None]
                for expr in exprs:
                    for sub in ast.walk(expr):
                        # `@name.setter` uses the getter of the same name,
                        # which moves together with it
                        if isinstance(sub, ast.Name) and \
                                sub.id in class_locals and \
                                sub.id != member.key:
                            self.error('mixins', f'decorator/default uses '
                                       f'the class-scope name {sub.id}',
                                       where)
            self._check_bases(split, bmod, node)

    def _check_bases(self, split, bmod, node):
        if not split.bases:
            self.error('mixins', f'{split.ref}: no bases list')
            return
        listed = []
        for ref in split.bases:
            if '::' in ref:
                file, _, name = ref.partition('::')
                listed.append((file, name))
            else:
                listed.append((None, ref))
        for mixin in split.mixins:
            if mixin not in listed:
                self.error('mixins', f'{split.ref}: mixin {mixin[1]} is not '
                           'in the bases')
        existing = []
        for base in node.bases:
            loc = self._resolve_base(bmod, base)
            existing.append((loc, ast.get_source_segment(bmod.src.text,
                                                          base)))
        for loc, text in existing:
            if loc is None:
                if (None, text) not in listed:
                    self.error('mixins', f'{split.ref}: base {text} is not '
                               'in the new bases list')
                continue
            if loc not in listed and (None, text) not in listed:
                self.error('mixins', f'{split.ref}: base {text} '
                           f'({loc[0]}::{loc[1]}) is dropped')
        # overrides: a moved method that a later base also defines must sit
        # in a mixin listed before that base
        index = {ref: i for i, ref in enumerate(listed)}
        for meth, mixin in split.methods.items():
            mpos = index.get(mixin)
            if mpos is None:
                continue
            for loc, text in existing:
                if loc is None:
                    continue
                defined = self._class_methods(loc)
                if meth in defined and index.get(loc, -1) < mpos and \
                        index.get(loc) is not None:
                    self.error('mixins', f'{split.ref}.{meth} overrides '
                               f'{text}.{meth} but its mixin {mixin[1]} '
                               f'comes after {text} in the bases')

    def _resolve_base(self, bmod, base):
        plan = self.plan
        module = None
        name = None
        if isinstance(base, ast.Name):
            if base.id in bmod.defined:
                return (plan.item_location(
                    bmod.path, bmod.defined[base.id].key, FINAL), base.id)
            b = bmod.bindings.get(base.id)
            if b and b[-1].attr:
                module, name = b[-1].target, b[-1].attr
        elif isinstance(base, ast.Attribute) and isinstance(base.value,
                                                            ast.Name):
            b = bmod.bindings.get(base.value.id)
            if b and b[-1].is_module_value:
                bb = b[-1]
                module = f'{bb.target}.{bb.attr}' if bb.attr else bb.target
                name = base.attr
        if module is None or not module.startswith('cellacdc'):
            return None
        res = plan.resolve_name(module, name, FINAL)
        if res is None or res[1] is None:
            return None
        home = res[0].replace('.', '/') + '.py'
        return (home, res[1])

    def _class_methods(self, loc):
        """Method names of the class at final location loc=(path, name),
        read from the base text of the class."""
        plan = self.plan
        path, name = loc
        for old in plan.old_module_files() + [path]:
            if not plan.base.exists(old):
                continue
            bmod = plan.base_module(old)
            item = bmod.defined.get(name)
            if item is None or item.kind != 'class':
                continue
            if plan.item_location(old, item.key, FINAL) != path:
                continue
            return {m.key for m in bmod.class_items(name)
                    if m.kind == 'method'}
        return set()

    # -------------------------------------------------------- globals
    def check_globals(self):
        plan = self.plan
        for old_file in plan.old_module_files():
            bmod = plan.base_module(old_file)
            hazards = [h for h in bmod.analysis.hazards
                       if h.kind == 'global']
            if not hazards:
                continue
            written = {h.name for h in hazards}
            for name in sorted(written):
                def_item = bmod.defined.get(name)
                if def_item is None:
                    continue
                home = plan.item_location(old_file, def_item.key, FINAL)
                users = set()
                for ref in bmod.analysis.refs:
                    if ref.name != name:
                        continue
                    loc = self._location_of_line(bmod, ref.lineno)
                    if loc is not None:
                        users.add(loc)
                apart = sorted(u for u in users if u != home)
                if apart:
                    self.error('globals', f'{old_file}: global {name} lives '
                               f'in {home} but is used (and written with '
                               f'`global`) in {apart}')

    def _location_of_line(self, bmod, lineno):
        plan = self.plan
        for item in bmod.items:
            if item.first <= lineno <= item.last:
                if item.kind == 'class':
                    key = (bmod.path, item.key)
                    split = plan.classes.get(key)
                    if split is not None:
                        for member in bmod.class_items(item.key):
                            if member.first <= lineno <= member.last and \
                                    member.key in split.methods:
                                return split.methods[member.key][0]
                return plan.item_location(bmod.path, item.key, FINAL)
        return None

    # ----------------------------------------------------- plugin api
    def check_plugin_api(self):
        plan = self.plan
        for file, names in plan.plugin_api.items():
            if not plan.base.exists(file):
                continue
            bmod = plan.base_module(file)
            for name in names:
                if name not in bmod.defined and name not in bmod.bindings:
                    self.warning('plugin_api', f'{file}: plugin API name '
                                 f'{name} does not exist')

    # ---------------------------------------------------------- names
    def check_names(self):
        plan = self.plan
        table = plan.name_table(FINAL)
        final = {p[:-3].replace('/', '.') if p.endswith('.py') else
                 p.rstrip('/').replace('/', '.') for p in self.final_modules()}
        final = {m[:-len('.__init__')] if m.endswith('.__init__') else m
                 for m in final}
        for module, names in table['names'].items():
            for name, home in names.items():
                if not home.startswith('cellacdc.'):
                    continue
                if home in final or plan.base_is_module(home):
                    continue
                self.error('names', f'{module}.{name} -> {home}, which does '
                           'not exist in the final layout')

    # ---------------------------------------------------------- sizes
    def check_sizes(self):
        """Estimated sizes of the final modules that the plan creates or
        changes: the lines of their items (base text) plus about 25 lines
        of imports and docstring."""
        plan = self.plan
        sizes = {}
        changed = set()
        for old_file in plan.old_module_files():
            bmod = plan.base_module(old_file)
            for item in bmod.items:
                if item.kind in ('import', 'importblock'):
                    loc = plan.module_path(old_file, FINAL)
                else:
                    loc = plan.item_location(old_file, item.key, FINAL)
                if loc is None:
                    continue
                if loc != old_file:
                    changed.add(loc)
                sizes[loc] = sizes.get(loc, 0) + item.last - item.first + 1
        for split in plan.classes.values():
            if not plan.base.exists(split.old_file):
                continue
            bmod = plan.base_module(split.old_file)
            if bmod.class_node(split.cls) is None:
                continue
            host = plan.item_location(split.old_file, split.cls, FINAL)
            for member in bmod.class_items(split.cls):
                ref = split.methods.get(member.key)
                if ref:
                    n = member.last - member.first + 1
                    sizes[ref[0]] = sizes.get(ref[0], 0) + n
                    sizes[host] = sizes.get(host, 0) - n
                    changed.update((ref[0], host))
        for path in sorted(changed):
            if not path.endswith('.py'):
                continue
            estimate = sizes.get(path, 0) + 25
            if plan.base.exists(path) and path not in plan.old_module_files():
                estimate += len(plan.base_module(path).src.lines)
            if estimate < SIZE_MIN:
                self.warning('sizes', f'{path}: about {estimate} lines '
                             f'(< {SIZE_MIN})')
            elif estimate > SIZE_MAX:
                self.warning('sizes', f'{path}: about {estimate} lines '
                             f'(> {SIZE_MAX})')

    # --------------------------------------------------------- cycles
    def check_cycles(self):
        graph, eager = self.build_graph()
        sccs = strongly_connected(graph)
        comp = {}
        for i, scc in enumerate(sccs):
            for node in scc:
                comp[node] = i
        reported = set()
        base_files = set(self.plan.base.files())
        changed = {dotted(p) for p in self.final_modules()
                   if p.endswith('.py')}
        changed |= {dotted(p) for p in self.plan.old_module_files()}
        for (src, dst), uses in sorted(eager.items()):
            if src == dst or comp.get(src) != comp.get(dst):
                continue
            key = (src, dst)
            if key in reported:
                continue
            reported.add(key)
            if src not in changed and dst not in changed and \
                    src.replace('.', '/') + '.py' in base_files:
                continue    # a cycle of unchanged modules (pre-existing)
            back = shortest_path(graph, dst, src) or []
            path = ' -> '.join([src] + back)
            self.error('cycles', f'{src} uses {dst} at import time '
                       f'({", ".join(sorted(uses)[:5])}), but {dst} '
                       f'imports {src} back at module level: {path}')

    def build_graph(self):
        """Import graph of the final layout: {module: {modules}} and the
        eager edges {(src, dst): {names}}."""
        plan = self.plan
        graph = {}
        eager = {}
        units = {}      # final module path -> [(bmod, item)]
        for old_file in plan.old_module_files():
            bmod = plan.base_module(old_file)
            for item in bmod.items:
                if item.kind in ('import', 'docstring'):
                    continue
                loc = plan.item_location(old_file, item.key, FINAL)
                if loc is None:
                    continue
                units.setdefault(loc, []).append((bmod, item))
        touched = set(plan.old_module_files())
        for path in sorted(plan.base.files()):
            if not path.startswith('cellacdc/') or not path.endswith('.py'):
                continue
            if path in touched or path.startswith(SKIP_GRAPH_DIRS):
                continue
            try:
                bmod = plan.base_module(path)
            except (SyntaxError, RefactorError, UnicodeDecodeError):
                continue
            for item in bmod.items:
                if item.kind in ('import', 'docstring'):
                    continue
                units.setdefault(path, []).append((bmod, item))
        spans = []      # (bmod, first, last, final path)
        for path, pairs in units.items():
            for bmod, item in pairs:
                split = plan.classes.get((bmod.path, item.key)) \
                    if item.kind == 'class' else None
                if split is None:
                    spans.append((bmod, item.first, item.last, path, ()))
                    continue
                moved = []
                for member in bmod.class_items(item.key):
                    ref = split.methods.get(member.key)
                    if ref:
                        moved.append((member.first, member.last))
                        spans.append((bmod, member.first, member.last,
                                      ref[0], ()))
                spans.append((bmod, item.first, item.last, path,
                              tuple(moved)))
                # the host imports its mixins at import time (bases)
                for (file, name) in split.mixins:
                    graph.setdefault(dotted(path), set()).add(dotted(file))
                    eager.setdefault((dotted(path), dotted(file)),
                                     set()).add(name)
        for bmod, first, last, path, exclude in spans:
            src = dotted(path)
            graph.setdefault(src, set())
            for dst, name, is_eager in self._item_edges(
                    bmod, first, last, path, exclude):
                if dst == src or not dst.startswith('cellacdc'):
                    continue
                graph[src].add(dst)
                if is_eager:
                    eager.setdefault((src, dst), set()).add(name)
        return graph, eager

    def _item_edges(self, bmod, first, last, path, exclude=()):
        plan = self.plan
        attrs = _attribute_map(bmod)
        for ref in bmod.analysis.refs_in(first, last):
            if any(a <= ref.lineno <= b for a, b in exclude):
                continue
            if ref.ctx != 'load':
                continue
            name = ref.name
            def_item = bmod.defined.get(name)
            if def_item is not None:
                loc = plan.item_location(bmod.path, def_item.key, FINAL)
                if loc is not None and loc != path:
                    yield dotted(loc), name, ref.eager
                continue
            bindings = bmod.bindings.get(name)
            if not bindings:
                continue
            b = bindings[-1]
            if not b.target.startswith('cellacdc'):
                continue
            if b.is_module_value:
                module = f'{b.target}.{b.attr}' if b.attr else (
                    b.target if b.alias.asname else None)
                if module is None:
                    continue
                attr = attrs.get((ref.lineno, ref.col))
                if attr is None:
                    yield plan.remap_module(module, FINAL), name, ref.eager
                    continue
                res = plan.resolve_name(module, attr, FINAL)
                if res is not None:
                    yield res[0], f'{name}.{attr}', ref.eager
            else:
                res = plan.resolve_name(b.target, b.attr, FINAL)
                if res is not None:
                    # a from-import binds at import time: always eager
                    yield res[0], name, True


_ATTR_CACHE = {}


def _attribute_map(bmod):
    key = id(bmod)
    if key not in _ATTR_CACHE:
        out = {}
        for node in ast.walk(bmod.tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value,
                                                              ast.Name):
                out[(node.value.lineno, node.value.col_offset)] = node.attr
        _ATTR_CACHE[key] = out
    return _ATTR_CACHE[key]


def shortest_path(graph, start, goal):
    """Shortest path start -> ... -> goal (breadth-first), or None."""
    from collections import deque
    prev = {start: None}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node == goal:
            path = []
            while node is not None:
                path.append(node)
                node = prev[node]
            return path[::-1]
        for nxt in sorted(graph.get(node, ())):
            if nxt not in prev:
                prev[nxt] = node
                queue.append(nxt)
    return None


def strongly_connected(graph):
    """Tarjan's algorithm (iterative). Returns a list of sets."""
    index = {}
    low = {}
    on_stack = set()
    stack = []
    result = []
    counter = [0]
    nodes = set(graph)
    for targets in graph.values():
        nodes.update(targets)
    for start in sorted(nodes):
        if start in index:
            continue
        work = [(start, iter(sorted(graph.get(start, ()))))]
        index[start] = low[start] = counter[0]
        counter[0] += 1
        stack.append(start)
        on_stack.add(start)
        while work:
            node, it = work[-1]
            advanced = False
            for nxt in it:
                if nxt not in index:
                    index[nxt] = low[nxt] = counter[0]
                    counter[0] += 1
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, iter(sorted(graph.get(nxt, ())))))
                    advanced = True
                    break
                elif nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                scc = set()
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    scc.add(w)
                    if w == node:
                        break
                result.append(scc)
    return result


def format_report(issues, show_info=False):
    lines = []
    for level in ('error', 'warning', 'info'):
        if level == 'info' and not show_info:
            continue
        group = [i for i in issues if i.level == level]
        if not group:
            continue
        lines.append(f'== {level.upper()}S ({len(group)})')
        by_check = {}
        for issue in group:
            by_check.setdefault(issue.check, []).append(issue)
        for check in sorted(by_check):
            lines.append(f'-- {check} ({len(by_check[check])})')
            for issue in by_check[check]:
                loc = f'  [{issue.where}]' if issue.where else ''
                lines.append(f'   {issue.message}{loc}')
    n_err = sum(1 for i in issues if i.level == 'error')
    n_warn = sum(1 for i in issues if i.level == 'warning')
    lines.append(f'plan_check: {n_err} errors, {n_warn} warnings')
    return '\n'.join(lines)


def run(repo, commit=FINAL, deep=True, show_info=False, mapping_dir=None,
        out=sys.stdout):
    plan = Plan(repo, mapping_dir=mapping_dir)
    checker = Checker(plan, commit=commit, deep=deep)
    issues = checker.run()
    print(format_report(issues, show_info=show_info), file=out)
    return 1 if any(i.level == 'error' for i in issues) else 0


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--repo', default=None)
    parser.add_argument('--commit', default=FINAL)
    parser.add_argument('--quick', action='store_true',
                        help='skip the import-cycle analysis')
    parser.add_argument('--info', action='store_true')
    parser.add_argument('--update-qt-names', action='store_true',
                        help=f'recompute {QT_NAMES_FILE} with the installed '
                        'Qt binding (merged with the PyQt5/6 lists there)')
    args = parser.parse_args(argv)
    from . import DEFAULT_REPO_ROOT
    repo = args.repo or DEFAULT_REPO_ROOT
    if args.update_qt_names:
        update_qt_names()
    return run(repo, commit=args.commit, deep=not args.quick,
               show_info=args.info)


def update_qt_names():
    data = {}
    if os.path.exists(QT_NAMES_FILE):
        with open(QT_NAMES_FILE, encoding='utf-8') as file:
            data = json.load(file)
    import importlib
    try:
        qtpy = importlib.import_module('qtpy')
        api = qtpy.API_NAME
    except ImportError:
        api = 'current'
    data[api] = compute_qt_names()
    data['extra'] = EXTRA_QT_NAMES
    with open(QT_NAMES_FILE, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=1, sort_keys=True)


if __name__ == '__main__':
    sys.exit(main())
