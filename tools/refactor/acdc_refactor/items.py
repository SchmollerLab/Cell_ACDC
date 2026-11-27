"""Top-level items and class members with exact source spans and keys.

An item is one statement of a module body (or of a class body) together
with:

- its decorators;
- the comment lines in the gap before it, back to the previous statement
  (blank lines at the top of the gap stay outside: they are the separator);
- trailing comment lines indented deeper than the item (commented-out tail
  code), up to the first line that is indented as much or less.

Mapping keys (tools/refactor/README.md, "Mapping format"):

- def / class: its name;
- assignment / annotated assignment: the first target name (also inside a
  tuple), `X` of `X, Y = ...`;
- any other statement: 'stmt:' + its first source line stripped;
- the module docstring: '__doc__' (the 'stmt:' key also works);
- import statements: 'import:' + first line stripped (only used for
  imports with side effects that must move as text);
- repeats of a key get ' #2', ' #3', ... in source order.

Top-level if/try/with blocks that contain only imports (and `pass`) are
'importblock' items: optional (never required for coverage), their imports
are copied on demand like other imports.
"""
import ast
import io
import tokenize

from . import RefactorError
from .source import is_blank, is_comment, indent_of, strip_eol

SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
               ast.Lambda)


class Item:
    __slots__ = (
        'key', 'base_key', 'kind', 'names', 'node', 'first', 'code_first',
        'last', 'indent', 'optional', 'gap_before', 'index', 'alt_key'
    )

    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs.get(name))

    def __repr__(self):
        return (f'Item({self.key!r}, {self.kind}, '
                f'lines {self.first}-{self.last})')

    @property
    def name(self):
        return self.base_key


def first_line_key(src, node):
    return 'stmt:' + strip_eol(src.line(node.lineno)).strip()


def code_first_line(node):
    decorators = getattr(node, 'decorator_list', None) or []
    lines = [d.lineno for d in decorators] + [node.lineno]
    return min(lines)


def _target_names(target, out):
    if isinstance(target, ast.Name):
        out.append(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            _target_names(elt, out)
    elif isinstance(target, ast.Starred):
        _target_names(target.value, out)
    return out


def bound_names(node):
    """Names a module-level statement binds in the module namespace."""
    names = []

    def visit(n):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                          ast.ClassDef)):
            names.append(n.name)
            return
        if isinstance(n, ast.Lambda):
            return
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp,
                          ast.GeneratorExp)):
            # Only walrus targets leak out of comprehensions
            for sub in ast.walk(n):
                if isinstance(sub, ast.NamedExpr):
                    _target_names(sub.target, names)
            return
        if isinstance(n, ast.Import):
            for alias in n.names:
                names.append(alias.asname or alias.name.split('.')[0])
            return
        if isinstance(n, ast.ImportFrom):
            for alias in n.names:
                if alias.name != '*':
                    names.append(alias.asname or alias.name)
            return
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store,
                                                           ast.Del)):
            names.append(n.id)
            return
        if isinstance(n, ast.ExceptHandler) and n.name:
            names.append(n.name)
        match_as = getattr(ast, 'MatchAs', None)
        if match_as is not None and isinstance(n, match_as) and n.name:
            names.append(n.name)
        for child in ast.iter_child_nodes(n):
            visit(child)

    visit(node)
    seen = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return seen


def is_import_block(node):
    """An if/try/with block whose statements are only imports and pass."""
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.Pass)):
        return True
    if isinstance(node, ast.If):
        bodies = [node.body, node.orelse]
    elif isinstance(node, ast.Try) or (
            hasattr(ast, 'TryStar') and isinstance(node, ast.TryStar)):
        bodies = [node.body, node.orelse, node.finalbody]
        bodies.extend(h.body for h in node.handlers)
        if any(h.name for h in node.handlers):
            return False
    elif isinstance(node, ast.With):
        if any(item.optional_vars is not None for item in node.items):
            return False
        bodies = [node.body]
    else:
        return False
    has_import = False
    for body in bodies:
        for stmt in body:
            if not is_import_block(stmt):
                return False
            if not isinstance(stmt, ast.Pass):
                has_import = True
    return has_import


def _module_item_key(src, node, index):
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return 'import', 'import:' + strip_eol(src.line(node.lineno)).strip()
    if (index == 0 and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)):
        return 'docstring', '__doc__'
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return 'def', node.name
    if isinstance(node, ast.ClassDef):
        return 'class', node.name
    if isinstance(node, ast.Assign):
        names = _target_names(node.targets[0], [])
        if names:
            return 'assign', names[0]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return 'assign', node.target.id
    if is_import_block(node):
        return 'importblock', first_line_key(src, node)
    return 'stmt', first_line_key(src, node)


def _spans(src, body, region_first, region_last, items):
    """Fill first/last/gap_before of `items` (one per statement of
    `body`, in order) within the line region."""
    n = len(body)
    ends = []
    for i, node in enumerate(body):
        end = node.end_lineno
        limit = (code_first_line(body[i + 1]) - 1 if i + 1 < n
                 else region_last)
        indent = indent_of(src.line(node.lineno))
        j = end + 1
        last = end
        while j <= limit:
            line = src.line(j)
            if is_blank(line):
                j += 1
                continue
            if is_comment(line) and indent_of(line) > indent:
                last = j
                j += 1
                continue
            break
        ends.append(last)
    prev_end = region_first - 1
    for i, node in enumerate(body):
        code_first = code_first_line(node)
        if i > 0 and code_first <= ends[i - 1]:
            raise RefactorError(
                f'Statements share line {code_first} (semicolons?): '
                'cannot split them'
            )
        k = prev_end + 1
        gap = 0
        while k < code_first and is_blank(src.line(k)):
            k += 1
            gap += 1
        first = k if k < code_first else code_first
        item = items[i]
        item.first = first
        item.code_first = code_first
        item.last = ends[i]
        item.gap_before = gap
        item.indent = indent_of(src.line(node.lineno))
        prev_end = ends[i]


def _module_region_first(src):
    first = 1
    for lineno in (1, 2):
        if lineno > len(src.lines):
            break
        line = strip_eol(src.line(lineno)).strip()
        if line.startswith('#!') or (
                line.startswith('#') and 'coding' in line):
            first = lineno + 1
        else:
            break
    return first


def module_items(src, tree):
    """Items for every top-level statement (imports included)."""
    items = []
    counts = {}
    for index, node in enumerate(tree.body):
        kind, base_key = _module_item_key(src, node, index)
        counts[base_key] = counts.get(base_key, 0) + 1
        key = base_key
        if counts[base_key] > 1:
            key = f'{base_key} #{counts[base_key]}'
        if kind == 'docstring':
            names = []
        else:
            names = bound_names(node)
        items.append(Item(
            key=key, base_key=base_key, kind=kind, names=names, node=node,
            optional=kind in ('importblock', 'docstring', 'import'),
            index=index,
            alt_key=first_line_key(src, node) if kind == 'docstring' else None
        ))
    if tree.body:
        _spans(src, tree.body, _module_region_first(src), len(src.lines),
               items)
    return items


def class_header_end(src, node):
    """Line of the ':' that ends a class (or def) header."""
    first_body = code_first_line(node.body[0])
    text = ''.join(src.lines[node.lineno - 1:first_body])
    depth = 0
    seen_name = False
    tokens = tokenize.generate_tokens(io.StringIO(text).readline)
    try:
        for tok in tokens:
            if tok.type == tokenize.NAME and tok.string in ('class', 'def'):
                seen_name = True
            if tok.type == tokenize.OP:
                if tok.string in '([{':
                    depth += 1
                elif tok.string in ')]}':
                    depth -= 1
                elif tok.string == ':' and depth == 0 and seen_name:
                    return node.lineno + tok.start[0] - 1
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    raise RefactorError(f'Cannot find the end of the header at line '
                        f'{node.lineno}')


def class_items(src, class_node):
    """Items for the statements of a class body: methods are 'method'
    items keyed by name (two defs with one name share the key), other
    statements 'classstmt' items keyed 'stmt:<first line>'."""
    items = []
    counts = {}
    for index, node in enumerate(class_node.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind, key, base_key = 'method', node.name, node.name
        else:
            base_key = first_line_key(src, node)
            counts[base_key] = counts.get(base_key, 0) + 1
            key = base_key
            if counts[base_key] > 1:
                key = f'{base_key} #{counts[base_key]}'
            kind = 'classstmt'
        items.append(Item(
            key=key, base_key=base_key, kind=kind,
            names=bound_names(node), node=node, optional=False,
            index=index
        ))
    header_end = class_header_end(src, class_node)
    _spans(src, class_node.body, header_end + 1, class_node.end_lineno,
           items)
    # Trailing comments of the last member may not go past the class
    return items


def items_by_key(items):
    return {item.key: item for item in items}


def find_module_item(items, key):
    """The item with `key` ('name', 'name #2', 'stmt:...', '__doc__')."""
    for item in items:
        if item.key == key or (item.alt_key and item.alt_key == key):
            return item
    return None


def candidate_items(items, key):
    """Items whose key without the ' #n' suffix matches `key`'s."""
    base = _strip_repeat(key)
    return [item for item in items
            if item.base_key == base or item.alt_key == base]


def _strip_repeat(key):
    head, sep, tail = key.rpartition(' #')
    if sep and tail.isdigit():
        return head
    return key


def repeat_index(key):
    head, sep, tail = key.rpartition(' #')
    if sep and tail.isdigit():
        return int(tail)
    return 1


def base_key_of(key):
    return _strip_repeat(key)
