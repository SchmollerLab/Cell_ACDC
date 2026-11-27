"""C1: debugutils.check_unused_methods parsed the file 'cellacdc/gui.py'
(relative to the working directory) to find the class `node_name`. guiWin
is already composed of mixins (gui_navigation, gui_annotations, ...) and
its methods move into more mixins in C10, so it now reads the source of
the classes of the window's MRO that belong to cellacdc
(inspect.getsource), wherever they are defined: a method is unused when no
`self.<name>` or `<one of those classes>.<name>` reference in their methods
names it. The class is looked up in the module that defines guiWin (by
default guiWin itself).

Debug-only helper (never called by the package); the output now also
covers the mixins' methods.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, run  # noqa: E402

DEBUGUTILS = 'cellacdc/debugutils.py'

OLD = '''\
def check_unused_methods(node_name):
    import ast
    from pathlib import Path
    from collections import Counter

    file_path = Path("cellacdc/gui.py")
    src = file_path.read_text(encoding="utf-8")
    tree = ast.parse(src)

    gui_cls = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == node_name:
            gui_cls = node
            break

    if gui_cls is None:
        raise RuntimeError(f"{node_name} class not found")

    methods = []
    refs = Counter()

    for node in gui_cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.append(node.name)

            for n in ast.walk(node):
                # Count any self.method reference (calls, signal connections, passing callbacks, etc.)
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self":
                    refs[n.attr] += 1

                # Also count guiWin.method(...) direct class-qualified calls inside class
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == node_name:
                    refs[n.attr] += 1
'''

NEW = '''\
def check_unused_methods(node_name='guiWin'):
    """Print the methods of the main window class `node_name` (a class of
    the module that defines guiWin) and of its cellacdc base classes (the
    mixins, wherever they are defined) that no `self.<method>` or
    `<one of those classes>.<method>` reference in their methods uses."""
    import ast
    import textwrap
    from collections import Counter
    from .gui import guiWin

    gui_cls = vars(sys.modules[guiWin.__module__]).get(node_name)
    if not isinstance(gui_cls, type):
        raise RuntimeError(f"{node_name} class not found")

    classes = [
        cls for cls in gui_cls.__mro__
        if cls.__module__.split('.')[0] == 'cellacdc'
    ]
    class_names = {cls.__name__ for cls in classes}
    methods = []
    refs = Counter()

    for cls in classes:
        source = textwrap.dedent(inspect.getsource(cls))
        class_node = ast.parse(source).body[0]
        for node in class_node.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in methods:
                methods.append(node.name)

            for n in ast.walk(node):
                # Count any self.method reference (calls, signal connections, passing callbacks, etc.)
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self":
                    refs[n.attr] += 1

                # Also count class-qualified calls (guiWin.method(...), <Mixin>.method(...))
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in class_names:
                    refs[n.attr] += 1
'''


def main(repo):
    replace(repo, DEBUGUTILS, OLD, NEW,
            items=[DEBUGUTILS + '::check_unused_methods'])


run(main)
