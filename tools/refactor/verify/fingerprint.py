"""Static fingerprints of every class, function, method and module-level item.

Usage::

    python tools/refactor/verify/fingerprint.py snapshot --out <dir> [--repo <tree>]
    python tools/refactor/verify/fingerprint.py compare --before <dir> --after <dir>
        [--mapping-table <name-table.json>] [--mapping-dir <shards>]
        [--changes <changes.jsonl> ...] [--ignore-docstrings] [--report <json>]
    python tools/refactor/verify/fingerprint.py show --repo <tree> <module:Qual.name>

Items (keys ``'<module>:<Qual.name>'``) are, per module of ``cellacdc/``
(``qrc_resources*`` excluded): top-level classes and functions, methods and
nested classes (``Class.method``), module-level assignments (key = first
target name) and other module-level statements (``stmt:<first line>``, with
``' #2'``... for repeats, as in the mapping shards). Imports are not items.

Each item's fingerprint is computed statically from the AST: ``ast.dump`` of
its body (line/column attributes dropped), after this normalization:

- every reference to a module-level binding (bare name or ``module.attr``
  chain, also names bound by imports inside functions) becomes a ``Ref()``
  placeholder; the resolved target (the definition site found by following
  imports through the tree: ``D|module|name``, an in-tree module
  ``M|module``, an external object ``E|dotted`` or ``U|module|name`` when
  not found statically) is stored separately, in occurrence order;
- import statements inside bodies are dropped (relative import depth and
  alias names do not matter, only what the names resolve to);
- string constants that are dotted ``cellacdc.*`` paths become refs too.

Class items hold decorators and non-method statements of the class body; the
bases and class keywords are compared separately (``bases``).

``compare`` maps both snapshots to canonical keys and canonical refs with the
name table and the mapping shards (``common.Mapping``): an item is unchanged
when the normalized dump is equal and its canonical refs are equal, so
rewriting ``widgets.myMessageBox`` to ``widgets_message_box.myMessageBox`` or
a bare name to ``alias.Name`` is no change when both resolve to the same new
home. Methods moved into mixins are keyed by their host class. Every other
difference, every lost item and every new item is an error unless listed in
``changes.jsonl`` (``common.load_changes``: any string value naming the item
as ``<file>::<Qual>`` or ``<module>:<Qual>``). New items in compat modules
(shims, ``_compat*``, new package ``__init__``) and mixin class shells are
ignored; copies of an unchanged module-level statement (e.g. the idempotent
``pg.setConfigOption`` copied into every module using ``pg``) are reported as
``copied``. Planned base lists of split classes (shards ``classes.bases``) are
checked against the after tree.

Snapshots are only comparable when taken with the same Python minor version
(``ast.dump`` differs between versions); compare warns otherwise.
"""
import argparse
import ast
import hashlib
import os
import re
import sys
import warnings

try:
    from . import common
except ImportError:
    import common

DOTTED_CELLACDC = re.compile(r'cellacdc(\.[A-Za-z_][A-Za-z0-9_]*)+')

SNAPSHOT_FILE = 'fingerprint.json'


# --- Module bindings ----------------------------------------------------------

class ModuleInfo:
    def __init__(self, name, relpath, tree):
        self.name = name
        self.relpath = relpath
        self.is_pkg_init = relpath.endswith('/__init__.py')
        self.tree = tree
        self.package = name if self.is_pkg_init else name.rpartition('.')[0]
        self.bindings = {}
        self.stars = []
        if tree is not None:
            self._collect(tree.body, in_handler=False)

    def absolute(self, level, modname):
        """Absolute module name of an import of `modname` at `level`."""
        if not level:
            return modname or ''
        parts = self.package.split('.') if self.package else []
        up = level - 1
        if up > len(parts):
            return ('.' * level) + (modname or '')
        if up:
            parts = parts[:-up]
        base = '.'.join(parts)
        if modname:
            return f'{base}.{modname}' if base else modname
        return base

    def _bind(self, name, binding, in_handler):
        if in_handler and name in self.bindings:
            # except-branch fallbacks (`X = None`) keep the try's binding
            return
        self.bindings[name] = binding

    def _collect(self, body, in_handler):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                self._bind(node.name, ['def'], in_handler)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        self._bind(
                            alias.asname, ['import', alias.name], in_handler
                        )
                    else:
                        top = alias.name.split('.')[0]
                        self._bind(top, ['import', top], in_handler)
            elif isinstance(node, ast.ImportFrom):
                src = self.absolute(node.level, node.module)
                for alias in node.names:
                    if alias.name == '*':
                        self.stars.append(src)
                        continue
                    self._bind(
                        alias.asname or alias.name, ['from', src, alias.name],
                        in_handler
                    )
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign)
                    else [node.target]
                )
                for target in targets:
                    for name in _target_names(target):
                        self._bind(name, ['assign'], in_handler)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                for name in _target_names(node.target):
                    self._bind(name, ['assign'], in_handler)
                self._collect(node.body, in_handler)
                self._collect(node.orelse, in_handler)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if item.optional_vars is not None:
                        for name in _target_names(item.optional_vars):
                            self._bind(name, ['assign'], in_handler)
                self._collect(node.body, in_handler)
            elif isinstance(node, (ast.If, ast.While)):
                self._collect(node.body, in_handler)
                self._collect(node.orelse, in_handler)
            elif isinstance(node, ast.Try) or (
                hasattr(ast, 'TryStar') and isinstance(node, ast.TryStar)
            ):
                self._collect(node.body, in_handler)
                for handler in node.handlers:
                    if handler.name:
                        self._bind(handler.name, ['assign'], True)
                    self._collect(handler.body, True)
                self._collect(node.orelse, in_handler)
                self._collect(node.finalbody, in_handler)
            else:
                for sub in ast.walk(node):
                    if isinstance(sub, ast.NamedExpr):
                        for name in _target_names(sub.target):
                            self._bind(name, ['assign'], in_handler)


def _target_names(target):
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names = []
        for elt in target.elts:
            names.extend(_target_names(elt))
        return names
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    return []


class Resolver:
    """Follows module-level bindings to definition sites across the tree."""

    def __init__(self, modules):
        # name -> object with .bindings, .stars
        self.modules = modules
        self._cache = {}

    def resolve(self, mod, name, _seen=None):
        key = (mod, name)
        if key in self._cache:
            return self._cache[key]
        seen = _seen or set()
        if key in seen:
            return ['U', mod, name]
        seen.add(key)
        result = self._resolve(mod, name, seen)
        if _seen is None:
            self._cache[key] = result
        return result

    def _resolve(self, mod, name, seen):
        info = self.modules.get(mod)
        if info is None:
            if mod == common.PACKAGE or mod.startswith(common.PACKAGE + '.'):
                return ['U', mod, name]
            return ['E', f'{mod}.{name}']
        binding = info.bindings.get(name)
        sub = f'{mod}.{name}'
        if binding is None:
            if sub in self.modules:
                return ['M', sub]
            for star in info.stars:
                if star in self.modules:
                    result = self.resolve(star, name, seen)
                    if result[0] != 'U':
                        return result
            return ['U', mod, name]
        kind = binding[0]
        if kind in ('def', 'assign'):
            return ['D', mod, name]
        if kind == 'import':
            target = binding[1]
            return ['M', target] if target in self.modules else ['E', target]
        if kind == 'from':
            src, attr = binding[1], binding[2]
            if src in self.modules:
                subsrc = f'{src}.{attr}'
                if subsrc in self.modules:
                    sb = self.modules[src].bindings.get(attr)
                    if sb is None or sb[0] == 'import' or (
                        sb[0] == 'from' and f'{sb[1]}.{sb[2]}' == subsrc
                    ):
                        return ['M', subsrc]
                return self.resolve(src, attr, seen)
            return ['E', f'{src}.{attr}']
        return ['U', mod, name]

    def resolve_import(self, info, node):
        """{local name: ref} bound by an Import/ImportFrom node in a body."""
        bound = {}
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    target = alias.name
                    bound[alias.asname] = (
                        ['M', target] if target in self.modules
                        else ['E', target]
                    )
                else:
                    top = alias.name.split('.')[0]
                    bound[top] = (
                        ['M', top] if top in self.modules else ['E', top]
                    )
        else:
            src = info.absolute(node.level, node.module)
            for alias in node.names:
                if alias.name == '*':
                    continue
                local = alias.asname or alias.name
                sub = f'{src}.{alias.name}'
                if sub in self.modules:
                    bound[local] = ['M', sub]
                elif src in self.modules:
                    bound[local] = self.resolve(src, alias.name)
                else:
                    bound[local] = ['E', sub]
        return bound


# --- Normalization --------------------------------------------------------

class Ref(ast.expr):
    """Placeholder of a resolved reference (the target is in ``.ref``)."""
    _fields = ()


class Scope:
    def __init__(self, kind):
        self.kind = kind
        self.locals = set()
        self.imports = {}
        self.globals = set()
        self.nonlocals = set()


_NESTED_SCOPES = (
    ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
)


def _scan_bindings(nodes, scope, resolver, info):
    """Bindings of a function/class body, not descending into nested scopes."""
    stack = list(nodes)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            scope.locals.add(node.name)
            # decorators/defaults/bases run in this scope
            stack.extend(node.decorator_list)
            if isinstance(node, ast.ClassDef):
                stack.extend(node.bases)
                stack.extend(k.value for k in node.keywords)
            else:
                stack.extend(node.args.defaults)
                stack.extend(d for d in node.args.kw_defaults if d)
            continue
        if isinstance(node, ast.Lambda):
            stack.extend(node.args.defaults)
            stack.extend(d for d in node.args.kw_defaults if d)
            continue
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp,
                             ast.GeneratorExp)):
            # walrus targets inside comprehensions bind here
            for sub in ast.walk(node):
                if isinstance(sub, ast.NamedExpr):
                    scope.locals.update(_target_names(sub.target))
            if node.generators:
                stack.append(node.generators[0].iter)
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for name, ref in resolver.resolve_import(info, node).items():
                if name in scope.imports and scope.imports[name] != ref:
                    scope.locals.add(name)
                scope.imports[name] = ref
            continue
        if isinstance(node, ast.Global):
            scope.globals.update(node.names)
            continue
        if isinstance(node, ast.Nonlocal):
            scope.nonlocals.update(node.names)
            continue
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            scope.locals.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            scope.locals.add(node.name)
        elif isinstance(node, ast.alias):
            pass
        for name in ('name',):
            # match statement captures (MatchAs/MatchStar/MatchMapping rest)
            if type(node).__name__ in ('MatchAs', 'MatchStar') and getattr(
                node, name, None
            ):
                scope.locals.add(getattr(node, name))
        stack.extend(ast.iter_child_nodes(node))
    # import-bound names that are also assigned are plain locals
    for name in list(scope.imports):
        if name in scope.locals:
            del scope.imports[name]
    scope.locals -= scope.globals


def _args_names(args):
    names = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
    if args.vararg:
        names.append(args.vararg.arg)
    if args.kwarg:
        names.append(args.kwarg.arg)
    return names


class Normalizer(ast.NodeTransformer):
    def __init__(self, resolver, info, scopes):
        self.resolver = resolver
        self.info = info
        self.scopes = list(scopes)

    # name lookup -----------------------------------------------------------
    def lookup(self, name):
        innermost = True
        for scope in reversed(self.scopes):
            if scope.kind == 'module':
                break
            if scope.kind == 'class' and not innermost:
                continue
            innermost = False
            if name in scope.globals:
                break
            if name in scope.imports:
                return scope.imports[name]
            if name in scope.locals or name in scope.nonlocals:
                return None
        if name in self.info.bindings or name in _star_names(self, name):
            return self.resolver.resolve(self.info.name, name)
        return None

    def _ref(self, ref, node):
        new = Ref()
        new.ref = list(ref)
        return ast.copy_location(new, node)

    # visitors --------------------------------------------------------------
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            ref = self.lookup(node.id)
            if ref is not None:
                return self._ref(ref, node)
        return node

    def visit_Attribute(self, node):
        node.value = self.visit(node.value)
        value = node.value
        if (isinstance(value, Ref) and value.ref[0] == 'M'
                and isinstance(node.ctx, ast.Load)):
            return self._ref(
                self.resolver.resolve(value.ref[1], node.attr), node
            )
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str) and DOTTED_CELLACDC.fullmatch(
            node.value
        ):
            return self._ref(['S', node.value], node)
        return node

    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None

    def _visit_list(self, nodes):
        result = []
        for node in nodes:
            new = self.visit(node)
            if new is None:
                continue
            if isinstance(new, list):
                result.extend(new)
            else:
                result.append(new)
        return result

    def _visit_args_outer(self, args):
        args.defaults = self._visit_list(args.defaults)
        args.kw_defaults = [
            self.visit(d) if d is not None else None for d in args.kw_defaults
        ]
        for arg in args.posonlyargs + args.args + args.kwonlyargs + [
            args.vararg, args.kwarg
        ]:
            if arg is not None and arg.annotation is not None:
                arg.annotation = self.visit(arg.annotation)

    def function_scope(self, node):
        scope = Scope('function')
        scope.locals.update(_args_names(node.args))
        body = node.body if isinstance(node.body, list) else [node.body]
        _scan_bindings(body, scope, self.resolver, self.info)
        scope.locals.update(_args_names(node.args))
        return scope

    def visit_FunctionDef(self, node):
        node.decorator_list = self._visit_list(node.decorator_list)
        self._visit_args_outer(node.args)
        if node.returns is not None:
            node.returns = self.visit(node.returns)
        self.scopes.append(self.function_scope(node))
        try:
            node.body = self._visit_list(node.body)
        finally:
            self.scopes.pop()
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        self._visit_args_outer(node.args)
        self.scopes.append(self.function_scope(node))
        try:
            node.body = self.visit(node.body)
        finally:
            self.scopes.pop()
        return node

    def class_scope(self, node):
        scope = Scope('class')
        _scan_bindings(node.body, scope, self.resolver, self.info)
        return scope

    def visit_ClassDef(self, node):
        node.decorator_list = self._visit_list(node.decorator_list)
        node.bases = self._visit_list(node.bases)
        for keyword in node.keywords:
            keyword.value = self.visit(keyword.value)
        self.scopes.append(self.class_scope(node))
        try:
            node.body = self._visit_list(node.body)
        finally:
            self.scopes.pop()
        return node

    def _visit_comprehension(self, node, fields):
        generators = node.generators
        if generators:
            generators[0].iter = self.visit(generators[0].iter)
        scope = Scope('function')
        for gen in generators:
            scope.locals.update(_target_names(gen.target))
        self.scopes.append(scope)
        try:
            for i, gen in enumerate(generators):
                if i:
                    gen.iter = self.visit(gen.iter)
                gen.target = self.visit(gen.target)
                gen.ifs = self._visit_list(gen.ifs)
            for field in fields:
                setattr(node, field, self.visit(getattr(node, field)))
        finally:
            self.scopes.pop()
        return node

    def visit_ListComp(self, node):
        return self._visit_comprehension(node, ('elt',))

    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node):
        return self._visit_comprehension(node, ('key', 'value'))


def _star_names(normalizer, name):
    """Names reachable through star imports of the module (resolved lazily)."""
    info = normalizer.info
    if not info.stars:
        return ()
    ref = normalizer.resolver.resolve(info.name, name)
    return (name,) if ref[0] != 'U' else ()


def _strip_docstrings(node):
    for sub in ast.walk(node):
        body = getattr(sub, 'body', None)
        if not isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef,
                                ast.ClassDef, ast.Module)):
            continue
        if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            sub.body = body[1:]


def _collect_refs(node, out):
    """Ref targets in ``ast.dump`` order."""
    if isinstance(node, Ref):
        out.append(node.ref)
        return
    if isinstance(node, list):
        for item in node:
            _collect_refs(item, out)
        return
    if not isinstance(node, ast.AST):
        return
    for field in node._fields:
        _collect_refs(getattr(node, field, None), out)


def _digest(text):
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:20]


def _dump(nodes):
    if isinstance(nodes, list):
        return '[' + ', '.join(ast.dump(n) for n in nodes) + ']'
    return ast.dump(nodes)


# --- Items --------------------------------------------------------------------

def _is_import_only(node):
    """True for a module-level statement holding only imports (and pass)."""
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.Pass)):
        return True
    if isinstance(node, ast.If):
        return all(_is_import_only(n) for n in node.body + node.orelse)
    if isinstance(node, ast.Try):
        parts = node.body + node.orelse + node.finalbody
        for handler in node.handlers:
            parts = parts + handler.body
        return all(_is_import_only(n) for n in parts)
    return False


def _first_line(lines, node):
    return lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ''


def item_key(node, lines):
    """Key of a module-level statement as in the mapping shards (no repeats)."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        target = node.targets[0] if isinstance(node, ast.Assign) else (
            node.target
        )
        names = _target_names(target)
        if names:
            return names[0]
    return 'stmt:' + _first_line(lines, node)


class ItemBuilder:
    def __init__(self, resolver):
        self.resolver = resolver
        self.items = {}

    def _record(self, info, qual, kind, node, body_nodes, base_nodes=None):
        key = f'{info.name}:{qual}'
        n = 2
        base_key = key
        while key in self.items:
            key = f'{base_key} #{n}'
            n += 1
        dump = _dump(body_nodes)
        refs = []
        _collect_refs(body_nodes, refs)
        _strip_docstrings_list(body_nodes)
        dump_nodoc = _dump(body_nodes)
        record = {
            'k': kind, 'f': info.relpath, 'l': node.lineno,
            'e': getattr(node, 'end_lineno', node.lineno),
            'h': _digest(dump), 'hn': _digest(dump_nodoc), 'r': refs,
        }
        if base_nodes is not None:
            base_refs = []
            _collect_refs(base_nodes, base_refs)
            record['bh'] = _digest(_dump(base_nodes))
            record['br'] = base_refs
        self.items[key] = record
        return key

    def add_module(self, info, lines):
        module_scope = Scope('module')
        counts = {}
        for node in info.tree.body:
            if (isinstance(node, ast.Expr) and isinstance(node.value,
                                                           ast.Constant)
                    and isinstance(node.value.value, str)
                    and node is info.tree.body[0]):
                continue  # module docstring
            if _is_import_only(node):
                continue
            key = item_key(node, lines)
            counts[key] = counts.get(key, 0) + 1
            if counts[key] > 1:
                key = f'{key} #{counts[key]}'
            if isinstance(node, ast.ClassDef):
                self.add_class(info, key, node, [module_scope])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                normalizer = Normalizer(self.resolver, info, [module_scope])
                normalizer.visit(node)
                self._record(info, key, 'def', node, node)
            else:
                normalizer = Normalizer(self.resolver, info, [module_scope])
                new = normalizer.visit(node)
                kind = 'stmt' if key.startswith('stmt:') else 'assign'
                if new is None:
                    continue
                self._record(info, key, kind, node, new)

    def add_class(self, info, qual, node, scopes):
        outer = Normalizer(self.resolver, info, scopes)
        node.decorator_list = outer._visit_list(node.decorator_list)
        node.bases = outer._visit_list(node.bases)
        for keyword in node.keywords:
            keyword.value = outer.visit(keyword.value)
        class_scope = outer.class_scope(node)
        inner_scopes = list(scopes) + [class_scope]
        body = []
        method_counts = {}
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = child.name
                method_counts[name] = method_counts.get(name, 0) + 1
                if method_counts[name] > 1:
                    name = f'{name} #{method_counts[name]}'
                normalizer = Normalizer(self.resolver, info, inner_scopes)
                normalizer.visit(child)
                self._record(info, f'{qual}.{name}', 'method', child, child)
            elif isinstance(child, ast.ClassDef):
                self.add_class(info, f'{qual}.{child.name}', child,
                               inner_scopes)
            else:
                normalizer = Normalizer(self.resolver, info, inner_scopes)
                new = normalizer.visit(child)
                if new is not None:
                    body.append(new)
        head = ast.ClassDef(
            name=node.name, bases=[], keywords=[], body=body,
            decorator_list=node.decorator_list,
        )
        if hasattr(ast.ClassDef, '_field_types') or 'type_params' in (
            ast.ClassDef._fields
        ):
            head.type_params = []
        bases = node.bases + [ast.keyword(arg=k.arg, value=k.value)
                              for k in node.keywords]
        self._record(info, qual, 'class', node, head, bases)


def _strip_docstrings_list(nodes):
    for node in nodes if isinstance(nodes, list) else [nodes]:
        _strip_docstrings(node)


def parse_tree(repo):
    files = common.list_package_files(repo)
    infos = {}
    errors = {}
    sources = {}
    for relpath in files:
        if relpath.split('/')[-1].startswith('qrc_resources'):
            continue
        name = common.module_name(relpath)
        with open(os.path.join(repo, relpath), 'rb') as file:
            data = file.read()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                tree = ast.parse(data, filename=relpath)
        except SyntaxError as err:
            errors[relpath] = f'{type(err).__name__}: {err}'
            tree = None
        infos[name] = ModuleInfo(name, relpath, tree)
        sources[name] = data.decode('utf-8', errors='replace').splitlines()
    return infos, sources, errors


COMPAT_MODULES = ('_compat', '_compat_moves')
COMPAT_BLOCK_NAMES = ('__getattr__', '__dir__')


def _is_pure_compat(mod, minfo):
    """Shims (`__cellacdc_compat__ = True`) and the compat machinery, as
    opposed to real modules that only got a compat block."""
    if mod.split('.')[-1] in COMPAT_MODULES:
        return True
    bindings = minfo.get('b')
    if isinstance(bindings, dict):
        return '__cellacdc_compat__' in bindings
    return True


def uses_compat(modname, bindings):
    """True for the compat machinery and modules importing it (shims)."""
    if modname.split('.')[-1] in COMPAT_MODULES:
        return True
    for binding in bindings.values():
        names = [part for part in binding[1:] if isinstance(part, str)]
        for name in names:
            if name.split('.')[-1] in COMPAT_MODULES:
                return True
    return False


def snapshot_data(repo):
    infos, sources, errors = parse_tree(repo)
    resolver = Resolver(infos)
    builder = ItemBuilder(resolver)
    for name, info in infos.items():
        if info.tree is None:
            continue
        builder.add_module(info, sources[name])
    ref_index = {}
    ref_table = []

    def intern(ref):
        key = '|'.join(ref)
        idx = ref_index.get(key)
        if idx is None:
            idx = ref_index[key] = len(ref_table)
            ref_table.append(key)
        return idx

    items = {}
    for key, record in builder.items.items():
        record['r'] = [intern(ref) for ref in record['r']]
        if 'br' in record:
            record['br'] = [intern(ref) for ref in record['br']]
        items[key] = record
    modules = {
        name: {
            'f': info.relpath, 'pkg': info.is_pkg_init,
            'b': info.bindings, 's': info.stars,
            'compat': uses_compat(name, info.bindings),
        }
        for name, info in infos.items()
    }
    return {
        'meta': common.meta(repo),
        'python_minor': '.'.join(map(str, sys.version_info[:2])),
        'parse_errors': errors,
        'modules': modules,
        'refs': ref_table,
        'items': items,
    }


def snapshot(repo, out):
    data = snapshot_data(repo)
    path = os.path.join(out, SNAPSHOT_FILE)
    common.write_json(path, data, indent=None)
    kinds = {}
    for record in data['items'].values():
        kinds[record['k']] = kinds.get(record['k'], 0) + 1
    summary = {
        'items': len(data['items']), 'by_kind': kinds,
        'modules': len(data['modules']), 'refs': len(data['refs']),
        'parse_errors': len(data['parse_errors']),
        'unresolved_refs': sum(1 for r in data['refs'] if r.startswith('U|')),
    }
    print(f'[fingerprint] {path}: {summary}')
    return summary


# --- Compare --------------------------------------------------------------

class _StoredModule:
    def __init__(self, entry):
        self.bindings = entry['b']
        self.stars = entry.get('s', [])
        self.is_pkg_init = entry.get('pkg', False)
        self.relpath = entry.get('f', '')


class Canon:
    """Canonical refs of a snapshot (mapped through `mapping`)."""

    def __init__(self, snap, mapping, after_resolver):
        self.snap = snap
        self.mapping = mapping
        self.after_resolver = after_resolver
        self.pkg = {
            name: entry.get('pkg', False)
            for name, entry in snap['modules'].items()
        }
        self._cache = {}

    def ref(self, idx):
        value = self._cache.get(idx)
        if value is None:
            value = self._cache[idx] = self._canon(self.snap['refs'][idx])
        return value

    def _canon(self, text, depth=0):
        kind, _, rest = text.partition('|')
        m = self.mapping
        if kind == 'D':
            mod, _, name = rest.partition('|')
            newmod, newname = m.map_def(mod, name, self.pkg.get(mod, False))
            return f'D|{newmod}|{newname}'
        if kind == 'M':
            return 'M|' + m.map_module(rest, self.pkg.get(rest, False))
        if kind == 'S':
            return 'S|' + m.map_dotted(rest)
        if kind == 'U':
            # not found statically, e.g. a name looked up through a compat
            # shim: its home per the name table (or the module's new name),
            # resolved again in the after tree
            mod, _, name = rest.partition('|')
            home = m.names.get(mod, {}).get(name)
            mapped = m.map_module(mod, self.pkg.get(mod, False))
            target = home or mapped
            if self.after_resolver is not None and (home or mapped != mod):
                resolved = '|'.join(self.after_resolver.resolve(target, name))
                if not resolved.startswith('U|'):
                    return self._canon_after(resolved, depth + 1)
            if home:
                return f'D|{home}|{name}'
            return f'U|{mapped}|{name}'
        return text

    def _canon_after(self, text, depth):
        # resolved in the after tree: only name-level mapping applies
        kind, _, rest = text.partition('|')
        if kind == 'D':
            mod, _, name = rest.partition('|')
            home = self.mapping.names.get(mod, {}).get(name)
            return f'D|{home or mod}|{name}'
        return text

    def refs(self, indices):
        return [self.ref(i) for i in indices]


def _identifiers(snap, key, canonical):
    record = snap['items'][key]
    mod, _, qual = key.partition(':')
    qual_plain = qual
    ids = {key, canonical, f"{record['f']}::{qual_plain}"}
    base = qual_plain.split(' #')[0]
    ids.add(f"{record['f']}::{base}")
    ids.add(f'{mod}:{base}')
    return ids


def compare(before, after, mapping, changes=frozenset(),
            ignore_docstrings=False):
    errors = []
    warnings_ = []
    info = []
    if before.get('python_minor') != after.get('python_minor'):
        warnings_.append(
            'snapshots taken with different Python versions '
            f"({before.get('python_minor')} vs {after.get('python_minor')}): "
            'ast.dump may differ'
        )
    after_resolver = Resolver({
        name: _StoredModule(entry)
        for name, entry in after['modules'].items()
    })
    canon_before = Canon(before, mapping, after_resolver)
    canon_after = Canon(after, mapping, after_resolver)
    hash_field = 'hn' if ignore_docstrings else 'h'

    def keyed(snap, side):
        result = {}
        skipped = {'compat': 0, 'mixin_shells': 0}
        for key, record in snap['items'].items():
            mod, _, qual = key.partition(':')
            pkg = snap['modules'].get(mod, {}).get('pkg', False)
            if side == 'after':
                minfo = snap['modules'].get(mod, {})
                if minfo.get('compat') and (
                        _is_pure_compat(mod, minfo)
                        or qual in COMPAT_BLOCK_NAMES):
                    # shims and the compat machinery; in real modules
                    # only the compat block (__getattr__/__dir__)
                    skipped['compat'] += 1
                    continue
                if record['k'] == 'class' and mapping.is_mixin_shell(mod, qual):
                    skipped['mixin_shells'] += 1
                    continue
            canonical = mapping.canonical_key(mod, qual, pkg)
            result.setdefault(canonical, []).append(key)
        return result, skipped

    before_keys, _ = keyed(before, 'before')
    after_keys, skipped = keyed(after, 'after')
    info.append(f'after items skipped: {skipped}')

    def listed(snap, key, canonical):
        return bool(_identifiers(snap, key, canonical) & changes)

    def signature(snap, canon, key):
        record = snap['items'][key]
        return (record[hash_field], tuple(canon.refs(record['r'])))

    def pair(bkeys, akeys):
        """Pairs of a canonical key's items: equal ones first, then by order.

        Several items share a canonical key when statements repeat (e.g.
        ``try:`` blocks, numbered per module) or names are redefined.
        """
        if len(bkeys) <= 1 and len(akeys) <= 1:
            return list(zip(bkeys, akeys)), bkeys[len(akeys):], akeys[
                len(bkeys):
            ]
        pairs = []
        left_b, left_a = list(bkeys), list(akeys)
        for bkey in list(left_b):
            bsig = signature(before, canon_before, bkey)
            match = next((a for a in left_a
                          if signature(after, canon_after, a) == bsig), None)
            if match is not None:
                pairs.append((bkey, match))
                left_b.remove(bkey)
                left_a.remove(match)
        n = min(len(left_b), len(left_a))
        pairs.extend(zip(left_b[:n], left_a[:n]))
        return pairs, left_b[n:], left_a[n:]

    changed, lost, new, copied, excused, bases_report = [], [], [], [], [], []
    same = 0
    unpaired_after = {}
    for canonical in sorted(set(before_keys) | set(after_keys)):
        pairs, lost_keys, new_keys = pair(
            before_keys.get(canonical, []), after_keys.get(canonical, [])
        )
        if new_keys:
            unpaired_after[canonical] = new_keys
        after_sigs = {signature(after, canon_after, a)
                      for a in after_keys.get(canonical, [])}
        for bkey in lost_keys:
            brec = before['items'][bkey]
            bmod, _, bqual = bkey.partition(':')
            deleted = (bmod, bqual) in mapping.deleted
            entry = {
                'canonical': canonical, 'before': bkey,
                'file': f"{brec['f']}:{brec['l']}", 'kind': brec['k'],
            }
            if brec['k'] in ('assign', 'stmt') and signature(
                    before, canon_before, bkey) in after_sigs:
                # identical copies of a binding in two old modules mapped
                # to one home that keeps one (move.py merged-duplicate)
                excused.append(dict(entry, why='merged identical binding'))
            elif deleted or listed(before, bkey, canonical):
                excused.append(dict(entry, why='deleted' if deleted
                                    else 'changes.jsonl'))
            else:
                lost.append(entry)
                entry['_sig'] = signature(before, canon_before, bkey)
        for bkey, akey in pairs:
            brec = before['items'][bkey]
            arec = after['items'][akey]
            equal = (
                brec[hash_field] == arec[hash_field]
                and canon_before.refs(brec['r']) == canon_after.refs(arec['r'])
            )
            if 'bh' in brec or 'bh' in arec:
                _compare_bases(
                    bkey, akey, brec, arec, canon_before, canon_after,
                    mapping, bases_report
                )
            if equal:
                same += 1
                continue
            entry = {
                'canonical': canonical, 'before': bkey, 'after': akey,
                'before_file': f"{brec['f']}:{brec['l']}",
                'after_file': f"{arec['f']}:{arec['l']}", 'kind': brec['k'],
                'body_equal': brec[hash_field] == arec[hash_field],
            }
            if not entry['body_equal']:
                pass
            else:
                br = canon_before.refs(brec['r'])
                ar = canon_after.refs(arec['r'])
                entry['ref_diff'] = [
                    [b, a] for b, a in zip(br, ar) if b != a
                ][:10]
                if len(br) != len(ar):
                    entry['ref_count'] = [len(br), len(ar)]
            if listed(before, bkey, canonical) or listed(
                after, akey, canonical
            ):
                excused.append(dict(entry, why='changes.jsonl'))
            else:
                changed.append(entry)
    # A statement's key is its first source line, so rewriting a reference
    # in it (C12: `myutils.check_install_x()` -> `model_installs....`)
    # changes the key: pair a lost statement with an unpaired one of the
    # same module that has the same normalized dump and canonical refs.
    still_lost = []
    for entry in lost:
        sig = entry.pop('_sig', None)
        if entry['kind'] != 'stmt' or sig is None:
            still_lost.append(entry)
            continue
        module = entry['canonical'].partition(':')[0]
        match = None
        for canonical, akeys in unpaired_after.items():
            if canonical.partition(':')[0] != module:
                continue
            for akey in akeys:
                if after['items'][akey]['k'] == 'stmt' and signature(
                        after, canon_after, akey) == sig:
                    match = (canonical, akey)
                    break
            if match:
                break
        if match is None:
            still_lost.append(entry)
            continue
        unpaired_after[match[0]].remove(match[1])
        same += 1
    lost = still_lost
    before_stmts = {}
    for canonical, bkeys in before_keys.items():
        for bkey in bkeys:
            brec = before['items'][bkey]
            if brec['k'] == 'stmt':
                sig = (brec['h'], tuple(canon_before.refs(brec['r'])))
                before_stmts[sig] = bkey
    for canonical, akeys in unpaired_after.items():
        for akey in akeys:
            arec = after['items'][akey]
            entry = {
                'canonical': canonical, 'after': akey,
                'file': f"{arec['f']}:{arec['l']}", 'kind': arec['k'],
            }
            if arec['k'] == 'stmt':
                sig = (arec['h'], tuple(canon_after.refs(arec['r'])))
                if sig in before_stmts:
                    copied.append(dict(entry, of=before_stmts[sig]))
                    continue
            if listed(after, akey, canonical):
                excused.append(dict(entry, why='changes.jsonl'))
            else:
                new.append(entry)
    base_errors = [b for b in bases_report if b.get('error')]
    errors.extend(
        f"changed: {c['canonical']} ({c['before_file']} -> {c['after_file']})"
        for c in changed
    )
    errors.extend(f"lost: {c['canonical']} ({c['file']})" for c in lost)
    errors.extend(f"new: {c['canonical']} ({c['file']})" for c in new)
    errors.extend(f"bases: {b['class']}: {b['error']}" for b in base_errors)
    summary = {
        'before_items': len(before['items']), 'after_items': len(after['items']),
        'same': same, 'changed': len(changed), 'lost': len(lost),
        'new': len(new), 'copied_stmts': len(copied),
        'excused': len(excused), 'bases_checked': len(bases_report),
        'bases_errors': len(base_errors),
    }
    return {
        'ok': not errors, 'summary': summary, 'errors': errors,
        'warnings': warnings_, 'info': info, 'changed': changed,
        'lost': lost, 'new': new, 'copied_stmts': copied, 'excused': excused,
        'bases': bases_report,
    }


def _planned_base_refs(spec_bases, mapping):
    refs = []
    for base in spec_bases:
        if '::' in base:
            path, cls = base.split('::', 1)
            mod, name = mapping.map_def(common.path_to_module(path), cls)
            refs.append(f'D|{mod}|{name}')
        else:
            refs.append(base)
    return refs


def _compare_bases(bkey, akey, brec, arec, canon_before, canon_after, mapping,
                   report):
    bref = canon_before.refs(brec.get('br', []))
    aref = canon_after.refs(arec.get('br', []))
    if brec.get('bh') == arec.get('bh') and bref == aref:
        return
    bmod, _, bqual = bkey.partition(':')
    planned = mapping.class_bases.get((bmod, bqual))
    entry = {'class': bkey, 'after': akey, 'before_bases': bref,
             'after_bases': aref}
    if planned is None:
        entry['error'] = 'bases changed (not a planned split)'
    else:
        want = _planned_base_refs(planned, mapping)
        ok = len(want) == len(aref) and all(
            w == a or (not w.startswith('D|') and a.rsplit('.', 1)[-1]
                       .rsplit('|', 1)[-1] == w)
            for w, a in zip(want, aref)
        )
        entry['planned'] = want
        if not ok:
            entry['error'] = 'bases differ from the planned bases'
    report.append(entry)


# --- show ---------------------------------------------------------------------

def show(repo, key):
    infos, sources, _ = parse_tree(repo)
    resolver = Resolver(infos)
    mod, _, qual = key.partition(':')
    info = infos[mod]
    builder = ItemBuilder(resolver)
    builder.add_module(info, sources[mod])
    record = builder.items.get(key)
    if record is None:
        print(f'no item {key}; items of {mod}:')
        for k in builder.items:
            print('  ', k)
        return 1
    # recompute the dump of that node for display
    infos2, sources2, _ = parse_tree(repo)
    info2 = infos2[mod]
    target = _find_node(info2.tree, qual)
    print(f"{key}  {record['f']}:{record['l']}-{record['e']}  h={record['h']}")
    if target is not None:
        scopes = [Scope('module')]
        Normalizer(Resolver(infos2), info2, scopes).visit(target)
        print(ast.dump(target, indent=1))
    print('refs:')
    for ref in record['r']:
        print('  ', '|'.join(ref))
    return 0


def _find_node(tree, qual):
    parts = qual.split(' #')[0].split('.')
    body = tree.body
    node = None
    for part in parts:
        node = next(
            (n for n in body if isinstance(n, (
                ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef
            )) and n.name == part), None
        )
        if node is None:
            return None
        body = node.body
    return node


# --- CLI / run ------------------------------------------------------------

def load_snapshot(folder):
    return common.read_json(os.path.join(folder, SNAPSHOT_FILE))


def run(commit, repo=None, before='baseline', table=None, mapping_dir=None,
        changes=None, ignore_docstrings=False):
    """Snapshot the tree as `commit`; compare with `before` if it exists."""
    repo = repo or common.REPO_ROOT
    out = common.out_dir(commit)
    snapshot(repo, out)
    if not before or before == commit:
        return None
    before_dir = common.out_dir(before)
    if not os.path.exists(os.path.join(before_dir, SNAPSHOT_FILE)):
        return None
    mapping = common.Mapping.load(
        table or common.default_name_table(commit),
        mapping_dir or common.MAPPING_DIR
    )
    listed = common.load_changes(
        changes if changes is not None else common.changes_files_up_to(commit)
    )
    report = compare(
        load_snapshot(before_dir), load_snapshot(out), mapping, listed,
        ignore_docstrings
    )
    common.write_json(os.path.join(out, 'fingerprint_compare.json'), report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('snapshot')
    p.add_argument('--out', required=True)
    p.add_argument('--repo', default=common.REPO_ROOT)
    p = sub.add_parser('compare')
    p.add_argument('--before', required=True)
    p.add_argument('--after', required=True)
    p.add_argument('--mapping-table')
    p.add_argument('--mapping-dir', default=common.MAPPING_DIR)
    p.add_argument('--changes', action='append')
    p.add_argument('--ignore-docstrings', action='store_true')
    p.add_argument('--report')
    p.add_argument('--max-print', type=int, default=40)
    p = sub.add_parser('show')
    p.add_argument('--repo', default=common.REPO_ROOT)
    p.add_argument('key')
    args = parser.parse_args(argv)
    if args.cmd == 'snapshot':
        snapshot(args.repo, args.out)
        return 0
    if args.cmd == 'show':
        return show(args.repo, args.key)
    mapping = common.Mapping.load(args.mapping_table, args.mapping_dir)
    changes = args.changes
    if changes is None:
        changes = [os.path.join(args.after, 'changes.jsonl')]
    report = compare(
        load_snapshot(args.before), load_snapshot(args.after), mapping,
        common.load_changes(changes), args.ignore_docstrings
    )
    report_path = args.report or os.path.join(
        args.after, 'fingerprint_compare.json'
    )
    common.write_json(report_path, report)
    print(f"[fingerprint] {report['summary']}")
    for line in report['warnings']:
        print('WARNING:', line)
    for line in report['errors'][:args.max_print]:
        print('ERROR:', line)
    if len(report['errors']) > args.max_print:
        print(f"... {len(report['errors']) - args.max_print} more errors in "
              f'{report_path}')
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
