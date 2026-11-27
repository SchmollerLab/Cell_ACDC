"""C1 (synthesis, C1 row): `import cellacdc` imports the optional 3D volume
renderer (vispy) in a try block that ignores every ModuleNotFoundError. A
missing cellacdc module (e.g. an import path broken by a move) would then
silently drop VolumeRendererWindow instead of failing: such errors are
re-raised, a missing optional dependency is still ignored.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, run  # noqa: E402

INIT = 'cellacdc/__init__.py'


def main(repo):
    replace(repo, INIT,
            "if GUI_INSTALLED:\n"
            "    try:\n"
            "        from cellacdc.volume_renderer.canvas import VolumeRendererWindow\n"
            "    except ModuleNotFoundError:\n"
            "        pass",
            "if GUI_INSTALLED:\n"
            "    try:\n"
            "        from cellacdc.volume_renderer.canvas import VolumeRendererWindow\n"
            "    except ModuleNotFoundError as err:\n"
            "        # Optional dependencies (vispy) may be missing, cellacdc modules\n"
            "        # may not\n"
            "        if (err.name or '').split('.')[0] == 'cellacdc':\n"
            "            raise",
            items=[INIT + '::stmt:if GUI_INSTALLED:'])


run(main)
