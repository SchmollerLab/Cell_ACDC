"""Import bindings, relative/absolute resolution and import emission.

Module contexts are (dotted name, is_package) pairs, e.g.
('cellacdc.gui.mixins.navigation', False) or ('cellacdc.gui', True).

Import style of the refactor (plan rule 6): relative imports only inside
the importing file's part package (the first package below cellacdc,
e.g. cellacdc.gui, cellacdc.shared; top-level modules of cellacdc form the
part '<root>'), absolute imports otherwise. A relative import in a file
that moves is recomputed from its absolute target, so `from . import
widgets` keeps meaning cellacdc.widgets after a move into cellacdc/gui/.
"""
import ast
import io
import tokenize

from . import RefactorError
from .source import strip_eol

ROOT = 'cellacdc'


class ModuleCtx:
    __slots__ = ('dotted', 'is_package')

    def __init__(self, dotted, is_package=False):
        self.dotted = dotted
        self.is_package = is_package

    @property
    def package(self):
        if self.is_package:
            return self.dotted
        return self.dotted.rpartition('.')[0]

    def __repr__(self):
        return f'ModuleCtx({self.dotted!r}, pkg={self.is_package})'


def path_to_ctx(path):
    """'cellacdc/x/y.py' -> ModuleCtx('cellacdc.x.y'); '__init__.py' ->
    the package; a folder -> the package."""
    path = path.replace('\\', '/')
    if path.endswith('/__init__.py'):
        return ModuleCtx(path[:-len('/__init__.py')].replace('/', '.'), True)
    if path.endswith('.py'):
        return ModuleCtx(path[:-3].replace('/', '.'), False)
    return ModuleCtx(path.rstrip('/').replace('/', '.'), True)


def dotted_to_paths(dotted):
    base = dotted.replace('.', '/')
    return [base + '.py', base + '/__init__.py']


def part_of(ctx):
    parts = ctx.dotted.split('.')
    if parts[0] != ROOT:
        return None
    if len(parts) >= 3 or (len(parts) == 2 and ctx.is_package):
        return parts[1]
    return '<root>'


def part_of_dotted(dotted, is_package=False):
    return part_of(ModuleCtx(dotted, is_package))


def resolve_relative(ctx, level, module):
    """Absolute dotted name of `from <level dots><module> import ...`."""
    if level == 0:
        return module
    parts = ctx.package.split('.')
    if level - 1 > len(parts) - 1:
        raise RefactorError(
            f'Relative import beyond the top package in {ctx.dotted}'
        )
    base = parts[:len(parts) - (level - 1)]
    if module:
        base = base + module.split('.')
    return '.'.join(base)


def relative_expr(ctx, target):
    """Relative module expression from module `ctx` to the absolute dotted
    `target` (a module or package): '.', '..x', '.y.z'."""
    pkg = ctx.package.split('.')
    tgt = target.split('.')
    common = 0
    while common < min(len(pkg), len(tgt)) and pkg[common] == tgt[common]:
        common += 1
    if common == 0:
        raise RefactorError(f'No common package: {ctx.dotted} {target}')
    level = len(pkg) - common + 1
    rest = '.'.join(tgt[common:])
    return '.' * level + rest


def from_expr(ctx, target, force_absolute=False):
    """Module expression for a from-import of `target` in `ctx`."""
    if not force_absolute and target.split('.')[0] == ROOT:
        tpart = part_of_target(target)
        if tpart is not None and tpart == part_of(ctx) and tpart != '<root>':
            return relative_expr(ctx, target)
        if tpart == '<root>' and part_of(ctx) == '<root>':
            return relative_expr(ctx, target)
    return target


def part_of_target(target):
    parts = target.split('.')
    if parts[0] != ROOT:
        return None
    if len(parts) == 1:
        return '<root>'
    if len(parts) == 2:
        # cellacdc.x: a top-level module or a part package; a from-import
        # from a part package itself (from cellacdc.gui import x) counts as
        # that part
        return _PART_PACKAGES_CACHE.get(parts[1], '<root>')
    return parts[1]


# Filled by the caller (move.py) with the folder names below cellacdc that
# are packages: {'gui': 'gui', ...}; anything else is a root module.
_PART_PACKAGES_CACHE = {}


def set_part_packages(names):
    _PART_PACKAGES_CACHE.clear()
    for name in names:
        _PART_PACKAGES_CACHE[name] = name


class ImportBinding:
    __slots__ = ('name', 'node', 'alias', 'guard', 'top', 'target', 'attr',
                 'is_module_value', 'order')

    def __init__(self, **kw):
        for slot in self.__slots__:
            setattr(self, slot, kw.get(slot))

    def __repr__(self):
        return (f'ImportBinding({self.name!r} <- {self.target}'
                f'{"." + self.attr if self.attr else ""})')


def _iter_block_imports(node):
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        yield node
        return
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.stmt) or isinstance(child,
                                                     ast.excepthandler):
            yield from _iter_block_imports(child)


def import_bindings(tree, ctx, is_module, import_block_ids=None):
    """Bindings created by top-level imports and by imports inside the
    top-level import blocks (`import_block_ids`: ids of the guard nodes).

    `is_module(dotted)` tells whether a dotted name is a module/package.
    Returns {name: [ImportBinding, ...]} in source order.
    """
    import_block_ids = import_block_ids or set()
    out = {}
    order = 0
    for top in tree.body:
        if isinstance(top, (ast.Import, ast.ImportFrom)):
            nodes = [top]
            guard = None
        elif id(top) in import_block_ids:
            nodes = list(_iter_block_imports(top))
            guard = top
        else:
            continue
        for node in nodes:
            for alias in node.names:
                order += 1
                if isinstance(node, ast.Import):
                    name = alias.asname or alias.name.split('.')[0]
                    b = ImportBinding(
                        name=name, node=node, alias=alias, guard=guard,
                        top=top, target=alias.name, attr=None,
                        is_module_value=True, order=order
                    )
                else:
                    if alias.name == '*':
                        continue
                    target = resolve_relative(ctx, node.level, node.module)
                    sub = f'{target}.{alias.name}'
                    b = ImportBinding(
                        name=alias.asname or alias.name, node=node,
                        alias=alias, guard=guard, top=top, target=target,
                        attr=alias.name, is_module_value=bool(is_module(sub)),
                        order=order
                    )
                out.setdefault(b.name, []).append(b)
    return out


# ------------------------------------------------------------- retargeting
def _module_expr_span(stmt_text):
    """(start, end) character offsets of the module expression between
    'from' and 'import' in an ImportFrom statement text."""
    tokens = tokenize.generate_tokens(io.StringIO(stmt_text).readline)
    start = end = None
    offsets = _line_offsets(stmt_text)
    for tok in tokens:
        if tok.type == tokenize.NAME and tok.string == 'from' and start is None:
            start = offsets[tok.end[0] - 1] + tok.end[1]
        elif tok.type == tokenize.NAME and tok.string == 'import' \
                and start is not None:
            end = offsets[tok.start[0] - 1] + tok.start[1]
            break
    if start is None or end is None:
        raise RefactorError(f'Not a from-import: {stmt_text!r}')
    return start, end


def _line_offsets(text):
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def original_expr(node):
    return '.' * node.level + (node.module or '')


def plan_importfrom(node, old_ctx, new_ctx, remap, is_module):
    """How to rewrite a from-import that moves from `old_ctx` to `new_ctx`.

    remap(dotted) -> dotted to use for a (possibly moved) module.
    Returns None (unchanged), ('prefix', expr) when only the module
    expression changes, or ('groups', [(expr, [(name, asname)])]).
    """
    if node.level == 0 and node.module == '__future__':
        return None
    target = resolve_relative(old_ctx, node.level, node.module)
    groups = []
    for alias in node.names:
        if alias.name == '*':
            new_target, name, asname = remap(target), '*', None
        else:
            sub = f'{target}.{alias.name}'
            new_sub = remap(sub) if is_module(sub) else None
            if new_sub is not None and new_sub != sub:
                new_target, _, name = new_sub.rpartition('.')
                asname = alias.asname or (
                    alias.name if name != alias.name else None)
            else:
                new_target = remap(target)
                name, asname = alias.name, alias.asname
        if node.level == 0 and new_target == target:
            expr = node.module
        elif node.level == 0:
            expr = new_target
        else:
            expr = from_expr(new_ctx, new_target)
        if groups and groups[-1][0] == expr:
            groups[-1][1].append((name, asname))
        else:
            groups.append((expr, [(name, asname)]))
    same_names = (
        len(groups) == 1
        and [(a.name, a.asname) for a in node.names] == groups[0][1]
    )
    if same_names:
        if groups[0][0] == original_expr(node):
            return None
        return ('prefix', groups[0][0])
    return ('groups', groups)


def format_from(expr, names, indent='', width=79):
    parts = [n if not a else f'{n} as {a}' for n, a in names]
    one = f'{indent}from {expr} import {", ".join(parts)}'
    if len(one) <= width or len(parts) == 1:
        return [one]
    lines = [f'{indent}from {expr} import (']
    current = indent + '    '
    for i, part in enumerate(parts):
        piece = part + (',' if i < len(parts) - 1 else '')
        if current.strip() and len(current) + 1 + len(piece) > width:
            lines.append(current.rstrip())
            current = indent + '    '
        current += (piece if not current.strip() else ' ' + piece)
    if current.strip():
        lines.append(current.rstrip())
    lines.append(f'{indent})')
    return lines


def format_import(names, indent=''):
    parts = [n if not a else f'{n} as {a}' for n, a in names]
    return [f'{indent}import {", ".join(parts)}']


def stmt_text(src, node):
    """Source text of a (possibly multi-line) statement, without the EOL
    of its last line; the first line starts at column 0."""
    lines = src.lines[node.lineno - 1:node.end_lineno]
    text = ''.join(lines)
    return strip_eol(text)


def rewrite_import_stmt(src, node, old_ctx, new_ctx, remap, is_module,
                        keep=None):
    """Text lines (without EOL, original indentation) of an import
    statement moved from old_ctx to new_ctx, keeping only the aliases
    named in `keep` (None = all). Returns (lines, changed)."""
    indent = strip_eol(src.line(node.lineno))[:node.col_offset]
    aliases = [a for a in node.names
               if keep is None or (a.asname or (
                   a.name.split('.')[0] if isinstance(node, ast.Import)
                   else a.name)) in keep]
    if not aliases:
        return [], True
    full = len(aliases) == len(node.names)
    text = stmt_text(src, node)
    if isinstance(node, ast.Import):
        if full:
            return [strip_eol(line) for line in text.splitlines()], False
        return format_import([(a.name, a.asname) for a in aliases],
                             indent), True
    if not full:
        sub = ast.ImportFrom(module=node.module, names=aliases,
                             level=node.level)
        sub.lineno = node.lineno
        plan = plan_importfrom(sub, old_ctx, new_ctx, remap, is_module)
        if plan is None:
            plan = ('groups', [(original_expr(node),
                                [(a.name, a.asname) for a in aliases])])
        elif plan[0] == 'prefix':
            plan = ('groups', [(plan[1],
                                [(a.name, a.asname) for a in aliases])])
        lines = []
        for expr, names in plan[1]:
            lines.extend(format_from(expr, names, indent))
        return lines, True
    plan = plan_importfrom(node, old_ctx, new_ctx, remap, is_module)
    if plan is None:
        return [strip_eol(line) for line in text.splitlines()], False
    if plan[0] == 'prefix':
        start, end = _module_expr_span(text)
        new = text[:start] + ' ' + plan[1] + ' ' + text[end:]
        # keep the original spacing style: single spaces around the expr
        new = text[:start] + _respace(text[start:end], plan[1]) + text[end:]
        return [strip_eol(line) for line in new.splitlines()], True
    lines = []
    for expr, names in plan[1]:
        lines.extend(format_from(expr, names, indent))
    return lines, True


def _respace(old_span, expr):
    lead = old_span[:len(old_span) - len(old_span.lstrip())] or ' '
    trail = old_span[len(old_span.rstrip()):] or ' '
    return lead + expr + trail


# --------------------------------------------------------- import blocks
def _branch_header_lines(src, first_stmt_line, header_first):
    """Header lines from `header_first` up to the line before the first
    statement of the branch."""
    return [strip_eol(src.line(i)) for i in range(header_first,
                                                  first_stmt_line)]


def _find_keyword_line(src, before_line, keyword, not_before):
    for lineno in range(before_line - 1, not_before - 1, -1):
        text = strip_eol(src.line(lineno)).strip()
        if text.startswith(keyword):
            return lineno
    raise RefactorError(f'Cannot find {keyword!r} before line {before_line}')


def render_import_block(src, block, keep_by_stmt, old_ctx, new_ctx, remap,
                        is_module):
    """Text lines of a top-level import block with only the needed imports.

    keep_by_stmt: {id(import node): set of bound names to keep}.
    Branches left empty get `pass`. Returns [] when nothing is needed.
    """
    if not _block_needed(block, keep_by_stmt):
        return []
    return _render_stmt(src, block, keep_by_stmt, old_ctx, new_ctx, remap,
                        is_module)


def _block_needed(node, keep):
    for sub in ast.walk(node):
        if isinstance(sub, (ast.Import, ast.ImportFrom)) and id(sub) in keep:
            return True
    return False


def _render_body(src, body, indent, keep, old_ctx, new_ctx, remap,
                 is_module):
    out = []
    for stmt in body:
        out.extend(_render_stmt(src, stmt, keep, old_ctx, new_ctx, remap,
                                is_module))
    if not out:
        out = [indent + 'pass']
    return out


def _body_indent(src, body):
    line = strip_eol(src.line(body[0].lineno))
    return line[:len(line) - len(line.lstrip())]


def _render_stmt(src, stmt, keep, old_ctx, new_ctx, remap, is_module):
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        names = keep.get(id(stmt))
        if not names:
            return []
        lines, _ = rewrite_import_stmt(src, stmt, old_ctx, new_ctx, remap,
                                       is_module, keep=names)
        return lines
    if isinstance(stmt, ast.Pass):
        return [strip_eol(src.line(stmt.lineno))]
    if isinstance(stmt, (ast.If, ast.Try, ast.With)) and not _block_needed(
            stmt, keep):
        return []
    if isinstance(stmt, ast.If):
        out = _branch_header_lines(src, stmt.body[0].lineno, stmt.lineno)
        out.extend(_render_body(src, stmt.body, _body_indent(src, stmt.body),
                                keep, old_ctx, new_ctx, remap, is_module))
        if stmt.orelse:
            first = stmt.orelse[0]
            text = strip_eol(src.line(first.lineno)).strip()
            if (len(stmt.orelse) == 1 and isinstance(first, ast.If)
                    and text.startswith('elif')):
                sub = _render_stmt(src, first, keep, old_ctx, new_ctx,
                                   remap, is_module)
                if not sub:
                    sub = [strip_eol(src.line(first.lineno))] + \
                        [_body_indent(src, first.body) + 'pass']
                out.extend(sub)
            else:
                else_line = _find_keyword_line(
                    src, first.lineno, 'else', stmt.body[-1].end_lineno + 1)
                out.extend(_branch_header_lines(src, first.lineno,
                                                else_line))
                out.extend(_render_body(
                    src, stmt.orelse, _body_indent(src, stmt.orelse), keep,
                    old_ctx, new_ctx, remap, is_module))
        return out
    if isinstance(stmt, ast.Try):
        out = _branch_header_lines(src, stmt.body[0].lineno, stmt.lineno)
        out.extend(_render_body(src, stmt.body, _body_indent(src, stmt.body),
                                keep, old_ctx, new_ctx, remap, is_module))
        for handler in stmt.handlers:
            out.extend(_branch_header_lines(src, handler.body[0].lineno,
                                            handler.lineno))
            out.extend(_render_body(
                src, handler.body, _body_indent(src, handler.body), keep,
                old_ctx, new_ctx, remap, is_module))
        prev_end = (stmt.handlers[-1].end_lineno if stmt.handlers
                    else stmt.body[-1].end_lineno)
        for branch, keyword in ((stmt.orelse, 'else'),
                                (stmt.finalbody, 'finally')):
            if not branch:
                continue
            kw_line = _find_keyword_line(src, branch[0].lineno, keyword,
                                         prev_end + 1)
            out.extend(_branch_header_lines(src, branch[0].lineno, kw_line))
            out.extend(_render_body(
                src, branch, _body_indent(src, branch), keep, old_ctx,
                new_ctx, remap, is_module))
            prev_end = branch[-1].end_lineno
        return out
    if isinstance(stmt, ast.With):
        out = _branch_header_lines(src, stmt.body[0].lineno, stmt.lineno)
        out.extend(_render_body(src, stmt.body, _body_indent(src, stmt.body),
                                keep, old_ctx, new_ctx, remap, is_module))
        return out
    raise RefactorError(
        f'Unsupported statement in an import block at line {stmt.lineno}'
    )


def nested_imports(node):
    """Import statements anywhere inside `node` (functions included)."""
    return [n for n in ast.walk(node)
            if isinstance(n, (ast.Import, ast.ImportFrom))]
