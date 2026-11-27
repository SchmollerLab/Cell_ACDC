"""Scope-aware references to module globals.

ModuleAnalysis walks a module's AST together with its symtable (the
compiler's own scope analysis) and records:

- refs: every ast.Name that resolves to a module global (load, store or
  del), with its position and whether it is evaluated at import time
  ("eager": module-level code, class bodies, decorators, default values,
  base classes and, without `from __future__ import annotations`,
  annotations of module- and class-level definitions);
- hazards: zero-argument super(), __class__, name mangling (`self.__x`)
  and `global` declarations/writes, with their line numbers.

Child symbol tables are matched to AST nodes by (type, name, line).
List/set/dict comprehensions are inlined by the compiler on Python >= 3.12
(no child table): their targets are then handled here.
"""
import ast
import re
import symtable
import warnings

from . import RefactorError
from .source import char_col, split_lines, strip_eol

_STRING_START = re.compile(r'''([rRuU]?)('{3}|"{3}|'|")''')

COMP_NAMES = {
    ast.ListComp: 'listcomp', ast.SetComp: 'setcomp',
    ast.DictComp: 'dictcomp', ast.GeneratorExp: 'genexpr',
}


class NameRef:
    __slots__ = ('name', 'lineno', 'col', 'end_col', 'ctx', 'eager',
                 'scope_kind', 'func', 'scope', 'string')

    def __init__(self, name, lineno, col, end_col, ctx, eager, scope_kind,
                 func, scope=None):
        self.name = name
        self.lineno = lineno
        self.col = col              # UTF-8 byte column (ast)
        self.end_col = end_col
        self.ctx = ctx              # 'load', 'store' or 'del'
        self.eager = eager
        self.scope_kind = scope_kind
        self.func = func            # innermost function/lambda node or None
        self.scope = scope          # the _Scope the name is in
        self.string = False         # in a string annotation (col None:
        #                             it could not be located)

    def __repr__(self):
        return (f'NameRef({self.name!r}, {self.lineno}:{self.col}, '
                f'{self.ctx}, eager={self.eager})')


class Hazard:
    __slots__ = ('kind', 'name', 'lineno')

    def __init__(self, kind, name, lineno):
        self.kind = kind    # 'super0', '__class__', 'mangled', 'global'
        self.name = name
        self.lineno = lineno

    def __repr__(self):
        return f'Hazard({self.kind}, {self.name!r}, line {self.lineno})'


def _table_type(table):
    t = table.get_type()
    return str(getattr(t, 'value', t)).lower()


class _Scope:
    def __init__(self, kind, table, parent, eager, func=None,
                 class_node=None):
        self.kind = kind
        self.table = table
        self.parent = parent
        self.eager = eager
        self.func = func
        self.class_node = class_node
        self.comp_locals = set()
        self.declared_global = set()
        self._index = None

    def _table_scope(self):
        scope = self
        while scope.table is None:
            scope = scope.parent
        return scope

    def take_child(self, type_, name, lineno):
        scope = self._table_scope()
        if scope._index is None:
            scope._index = {}
            for child in scope.table.get_children():
                key = (_table_type(child), child.get_name(),
                       child.get_lineno())
                scope._index.setdefault(key, []).append(child)
        lst = scope._index.get((type_, name, lineno))
        if lst:
            return lst.pop(0)
        return None


def is_shadowed(ref, name):
    """True if `name` used where `ref` is would not reach the module
    global `name` (a local, parameter, closure variable or class attribute
    of the scopes around `ref`)."""
    scope = ref.scope
    innermost = True
    while scope is not None and scope.kind != 'module':
        if scope.table is None:
            if name in scope.comp_locals:
                return True
        elif scope.kind == 'class' and not innermost:
            pass    # class scopes are invisible to nested scopes
        else:
            try:
                sym = scope.table.lookup(name)
            except KeyError:
                sym = None
            if sym is not None and not sym.is_global() and (
                    sym.is_local() or sym.is_parameter() or sym.is_free()
                    or sym.is_assigned() or sym.is_imported()):
                return True
        innermost = False
        scope = scope.parent
    return False


def has_future_annotations(tree):
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == '__future__':
            if any(a.name == 'annotations' for a in node.names):
                return True
    return False


class ModuleAnalysis:
    def __init__(self, text, filename='<module>', tree=None):
        self.text = text
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SyntaxWarning)
            self.tree = tree if tree is not None else ast.parse(text,
                                                                filename)
            try:
                self.table = symtable.symtable(text, filename, 'exec')
            except SyntaxError as err:
                raise RefactorError(f'{filename}: {err}')
        self.filename = filename
        self.future_annotations = has_future_annotations(self.tree)
        self.refs = []
        self.hazards = []
        module = _Scope('module', self.table, None, eager=True)
        for stmt in self.tree.body:
            self._visit(stmt, module)

    # ---------------------------------------------------------- queries
    def refs_in(self, first, last):
        return [r for r in self.refs if first <= r.lineno <= last]

    def hazards_in(self, first, last):
        return [h for h in self.hazards if first <= h.lineno <= last]

    def global_names_in(self, first, last, eager_only=False, ctx=None):
        names = []
        for ref in self.refs_in(first, last):
            if eager_only and not ref.eager:
                continue
            if ctx is not None and ref.ctx != ctx:
                continue
            if ref.name not in names:
                names.append(ref.name)
        return names

    # --------------------------------------------------------- walking
    def _is_global(self, scope, name):
        s = scope
        while s.table is None:
            if name in s.comp_locals:
                return False
            s = s.parent
        if s.kind == 'module':
            return True
        try:
            sym = s.table.lookup(name)
        except KeyError:
            return True
        return sym.is_global()

    def _record_name(self, node, scope):
        if isinstance(node.ctx, ast.Load):
            ctx = 'load'
        elif isinstance(node.ctx, ast.Store):
            ctx = 'store'
        else:
            ctx = 'del'
        name = node.id
        if scope.class_node is not None and name.startswith('__') \
                and not name.endswith('__'):
            self.hazards.append(Hazard('mangled', name, node.lineno))
        if name == '__class__' and scope.kind != 'module':
            self.hazards.append(Hazard('__class__', name, node.lineno))
        if not self._is_global(scope, name):
            return
        if ctx != 'load' and scope.kind not in ('module', 'class'):
            self.hazards.append(Hazard('global', name, node.lineno))
        self.refs.append(NameRef(
            name, node.lineno, node.col_offset, node.end_col_offset, ctx,
            scope.eager and not self._lazy, scope.kind, scope.func, scope
        ))

    def _visit_list(self, nodes, scope):
        for node in nodes:
            if node is not None:
                self._visit(node, scope)

    _lazy = False

    def _visit_annotation(self, expr, scope, evaluated=True):
        """Names of an annotation: evaluated ones are eager (unless future
        annotations); names inside string annotations (and in annotations
        never evaluated) are lazy refs, still needed by pyflakes/F821."""
        previous = self._lazy
        self._lazy = previous or self.future_annotations or not evaluated
        try:
            self._visit(expr, scope)
            self._visit_string_annotations(expr, scope)
        finally:
            self._lazy = previous

    def _visit_string_annotations(self, expr, scope):
        skip = set()
        for node in ast.walk(expr):
            if isinstance(node, ast.Subscript):
                value = node.value
                name = getattr(value, 'id', None) or getattr(value, 'attr',
                                                             None)
                if name == 'Literal':
                    skip.update(id(n) for n in ast.walk(node.slice))
        for node in ast.walk(expr):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and id(node) not in skip):
                self._record_string_annotation(node, scope)

    def _record_string_annotation(self, node, scope):
        text = node.value
        try:
            inner = ast.parse(text, mode='eval')
        except SyntaxError:
            return
        base_col = None
        if node.lineno == node.end_lineno:
            lines = self._lines()
            line = strip_eol(lines[node.lineno - 1])
            start = char_col(line, node.col_offset)
            end = char_col(line, node.end_col_offset)
            literal = line[start:end]
            match = _STRING_START.match(literal)
            if match and '\\' not in literal:
                quote = match.group(2)
                body = literal[len(match.group(0)):len(literal) - len(quote)]
                if body == text:
                    base_col = start + len(match.group(0))
        for sub in ast.walk(inner):
            if not isinstance(sub, ast.Name):
                continue
            if not self._is_global(scope, sub.id):
                continue
            col = end_col = None
            if base_col is not None:
                line = strip_eol(self._lines()[node.lineno - 1])
                c0 = base_col + sub.col_offset
                col = len(line[:c0].encode('utf-8'))
                end_col = col + len(sub.id.encode('utf-8'))
            ref = NameRef(sub.id, node.lineno, col, end_col, 'load', False,
                          scope.kind, scope.func, scope)
            ref.string = True
            self.refs.append(ref)

    def _lines(self):
        if not hasattr(self, '_line_cache'):
            self._line_cache = split_lines(self.text)
        return self._line_cache

    def _visit_arguments(self, args, scope, annotations):
        self._visit_list(args.defaults, scope)
        self._visit_list([d for d in args.kw_defaults if d is not None],
                         scope)
        if annotations:
            all_args = list(args.posonlyargs) + list(args.args) + \
                list(args.kwonlyargs)
            for extra in (args.vararg, args.kwarg):
                if extra is not None:
                    all_args.append(extra)
            for arg in all_args:
                if arg.annotation is not None:
                    self._visit_annotation(arg.annotation, scope)

    def _child_scope(self, node, scope, kind, type_, name, eager, func,
                     class_node):
        table = scope.take_child(type_, name, node.lineno)
        if table is None and kind != 'comp':
            raise RefactorError(
                f'{self.filename}:{node.lineno}: no symbol table for '
                f'{type_} {name!r}'
            )
        return _Scope(kind, table, scope, eager, func=func,
                      class_node=class_node)

    def _visit(self, node, scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._visit_list(node.decorator_list, scope)
            self._visit_arguments(node.args, scope, annotations=True)
            if node.returns is not None:
                self._visit_annotation(node.returns, scope)
            new = self._child_scope(
                node, scope, 'function', 'function', node.name, False,
                node, scope.class_node
            )
            self._visit_list(node.body, new)
            return
        if isinstance(node, ast.Lambda):
            self._visit_arguments(node.args, scope, annotations=False)
            new = self._child_scope(
                node, scope, 'function', 'function', 'lambda', False,
                node, scope.class_node
            )
            self._visit(node.body, new)
            return
        if isinstance(node, ast.ClassDef):
            self._visit_list(node.decorator_list, scope)
            self._visit_list(node.bases, scope)
            self._visit_list([k.value for k in node.keywords], scope)
            new = self._child_scope(
                node, scope, 'class', 'class', node.name, scope.eager,
                scope.func, node
            )
            self._visit_list(node.body, new)
            return
        comp_name = COMP_NAMES.get(type(node))
        if comp_name is not None:
            gens = node.generators
            self._visit(gens[0].iter, scope)
            new = self._child_scope(
                node, scope, 'comp', 'function', comp_name, scope.eager,
                scope.func, scope.class_node
            )
            if new.table is None:
                for gen in gens:
                    for sub in ast.walk(gen.target):
                        if isinstance(sub, ast.Name):
                            new.comp_locals.add(sub.id)
            for i, gen in enumerate(gens):
                self._visit(gen.target, new)
                if i > 0:
                    self._visit(gen.iter, new)
                self._visit_list(gen.ifs, new)
            if isinstance(node, ast.DictComp):
                self._visit(node.key, new)
                self._visit(node.value, new)
            else:
                self._visit(node.elt, new)
            return
        if isinstance(node, ast.Name):
            self._record_name(node, scope)
            return
        if isinstance(node, ast.Global):
            for name in node.names:
                self.hazards.append(Hazard('global', name, node.lineno))
            return
        if isinstance(node, ast.Call):
            func = node.func
            if (isinstance(func, ast.Name) and func.id == 'super'
                    and not node.args and not node.keywords):
                self.hazards.append(Hazard('super0', 'super', node.lineno))
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if scope.class_node is not None and attr.startswith('__') \
                    and not attr.endswith('__'):
                self.hazards.append(Hazard('mangled', attr, node.lineno))
        if isinstance(node, ast.AnnAssign):
            # Annotations of module/class variables are evaluated (unless
            # future annotations); in functions they never are
            self._visit(node.target, scope)
            if node.value is not None:
                self._visit(node.value, scope)
            self._visit_annotation(
                node.annotation, scope,
                evaluated=scope.kind in ('module', 'class'))
            return
        for child in ast.iter_child_nodes(node):
            self._visit(child, scope)
