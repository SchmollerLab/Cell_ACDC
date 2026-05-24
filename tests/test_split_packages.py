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
        "controls",
        "canvas",
        "toolbars",
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
    def test_leaf_modules_compile(self):
        for module_name in PACKAGES:
            for leaf in PACKAGES[module_name]:
                path = ROOT / module_name.replace(".", "/") / f"{leaf}.py"
                with self.subTest(path=str(path)):
                    py_compile.compile(path, doraise=True)

    def test_package_init_modules_compile(self):
        for module_name in PACKAGES:
            path = ROOT / module_name.replace(".", "/") / "__init__.py"
            with self.subTest(path=str(path)):
                py_compile.compile(path, doraise=True)

    def test_shim_modules_compile(self):
        for rel_path in SHIMS:
            with self.subTest(path=rel_path):
                py_compile.compile(ROOT / rel_path, doraise=True)


if __name__ == "__main__":
    unittest.main()
