"""Smoke tests for split god-file packages."""

import py_compile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PACKAGES = {
    "cellacdc.tools": [
        "base",
        "concat",
        "align",
    ],
    "cellacdc.utils": [
        "logging",
        "paths",
        "install",
        "dataframe",
    ],
    "cellacdc.workers": [
        "_base",
        "segm",
        "tracking",
        "io",
    ],
    "cellacdc.widgets": [
        "canvas.histogram",
        "canvas.imshow",
        "controls.dialogs",
        "controls.inputs",
        "toolbars._base",
    ],
    "cellacdc.dialogs": [
        "_base",
        "general",
        "tracking",
        "measurements",
    ],
}

SHIMS = [
    "cellacdc/apps.py",
]


class TestSplitPackages(unittest.TestCase):
    def _module_path(self, module_name: str, leaf: str) -> Path:
        base = ROOT / module_name.replace(".", "/")
        return base / f"{leaf.replace('.', '/')}.py"

    def test_leaf_modules_compile(self):
        for module_name in PACKAGES:
            for leaf in PACKAGES[module_name]:
                path = self._module_path(module_name, leaf)
                with self.subTest(path=str(path)):
                    py_compile.compile(path, doraise=True)

    def test_package_init_modules_compile(self):
        checked = set()
        for module_name in PACKAGES:
            base = ROOT / module_name.replace(".", "/")
            paths = [base / "__init__.py"]
            for leaf in PACKAGES[module_name]:
                if "." in leaf:
                    subpkg = leaf.split(".", 1)[0]
                    paths.append(base / subpkg / "__init__.py")
            for path in paths:
                key = str(path)
                if key in checked:
                    continue
                checked.add(key)
                with self.subTest(path=key):
                    py_compile.compile(path, doraise=True)

    def test_shim_modules_compile(self):
        for rel_path in SHIMS:
            with self.subTest(path=rel_path):
                py_compile.compile(ROOT / rel_path, doraise=True)


if __name__ == "__main__":
    unittest.main()
