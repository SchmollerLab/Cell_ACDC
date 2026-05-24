"""Smoke tests for component module imports."""

import importlib
import unittest


COMPONENT_MODULES = [
    "cellacdc.components.palette",
    "cellacdc.components.base",
    "cellacdc.components.inputs_basic",
]


class TestComponentImports(unittest.TestCase):
    def test_leaf_component_modules_import(self):
        for module_name in COMPONENT_MODULES:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)

    def test_widgets_module_compiles(self):
        import py_compile

        py_compile.compile("cellacdc/widgets/__init__.py", doraise=True)


if __name__ == "__main__":
    unittest.main()
