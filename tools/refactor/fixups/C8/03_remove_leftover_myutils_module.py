"""C8: delete the leftover cellacdc/myutils.py next to the new myutils/
package.

The codemod moves every item of myutils.py into the package but leaves
the file with its imports (imports are not items) and a compat block.
Python imports the package (a directory with __init__.py wins over a
module of the same name), so the file is dead code that the C12 rewrite
would still edit and that test_no_old_paths would report (moves shard
note: no shim file for cellacdc/myutils.py, the package __init__ serves
the old names). Only a file that holds nothing but imports, import-only
`if` blocks, the compat assignment, comments and a docstring is deleted;
anything else stops the pipeline."""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import FixupError, log_change, run  # noqa: E402

REL = 'cellacdc/myutils.py'
PACKAGE_INIT = 'cellacdc/myutils/__init__.py'
COMPAT = '__getattr__, __dir__ = _acdc_compat.moved_module(__name__)'


def _is_import_block(node):
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return True
    if isinstance(node, (ast.If, ast.Try)):
        bodies = [node.body, node.orelse]
        if isinstance(node, ast.Try):
            bodies += [node.finalbody] + [h.body for h in node.handlers]
        return all(_is_import_block(child)
                   for body in bodies for child in body)
    return False


def main(repo):
    path = os.path.join(repo, REL)
    if not os.path.exists(path):
        print(f'  {REL}: already done')
        return
    if not os.path.exists(os.path.join(repo, PACKAGE_INIT)):
        raise FixupError(f'{PACKAGE_INIT} missing: the myutils package was '
                         'not created')
    with open(path, encoding='utf-8') as file:
        src = file.read()
    tree = ast.parse(src)
    for index, node in enumerate(tree.body):
        if index == 0 and isinstance(node, ast.Expr) and isinstance(
                getattr(node, 'value', None), ast.Constant):
            continue
        if _is_import_block(node):
            continue
        if ast.get_source_segment(src, node).strip() == COMPAT:
            continue
        raise FixupError(f'{REL}: unexpected statement at line '
                         f'{node.lineno}: '
                         f'{ast.get_source_segment(src, node)[:80]!r}')
    os.remove(path)
    print(f'  {REL}: deleted (imports and compat block only)')
    log_change(REL, [REL], 'deleted the leftover module next to the '
               'myutils package (imports and compat block only)')


run(main)
