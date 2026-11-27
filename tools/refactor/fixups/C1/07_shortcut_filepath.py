"""C1: `shortcut_filepath` is defined once, in cellacdc/__init__.py (next
to the other settings paths), instead of in gui.py and gui_viewer_tools.py
(synthesis R4). gui.py and gui_viewer_tools.py bind it from there (the
shortcuts mixin keeps reading it as a module global, which the tests'
settings isolation redirects); the launcher reads it, and the favourite
metrics path, from cellacdc instead of through the gui module (which
becomes a package in C3).

Same value everywhere: os.path.join(settings_folderpath, 'shortcuts.ini').
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import ensure_line_after, log_change, replace, run  # noqa: E402

INIT = 'cellacdc/__init__.py'
GUI = 'cellacdc/gui.py'
VIEWER_TOOLS = 'cellacdc/gui_viewer_tools.py'
LAUNCHER = 'cellacdc/_main.py'


def main(repo):
    # (the old text must not be part of the new one: idempotency)
    replace(repo, INIT,
            "    settings_folderpath, 'favourite_func_metrics.csv'\n"
            ")\n"
            "recentPaths_path = ",
            "    settings_folderpath, 'favourite_func_metrics.csv'\n"
            ")\n"
            "shortcut_filepath = os.path.join(settings_folderpath, 'shortcuts.ini')\n"
            "recentPaths_path = ",
            items=[INIT + '::shortcut_filepath'])

    # gui.py: the constant leaves, the name is imported (the launcher and
    # plugins may read cellacdc.gui.shortcut_filepath)
    replace(repo, GUI,
            "                             # local updating (since we actually need to call RP twice!)\n"
            "shortcut_filepath = os.path.join(settings_folderpath, 'shortcuts.ini')\n",
            "                             # local updating (since we actually need to call RP twice!)\n",
            items=[GUI + '::shortcut_filepath'])
    ensure_line_after(repo, GUI,
                      'from . import favourite_func_metrics_csv_path',
                      'from . import shortcut_filepath')

    changed = replace(
        repo, VIEWER_TOOLS,
        "from .cca_functions import _calc_rot_vol\n"
        "\n"
        "shortcut_filepath = os.path.join(settings_folderpath, 'shortcuts.ini')\n",
        "from .cca_functions import _calc_rot_vol\n"
        "from . import shortcut_filepath\n",
        items=[VIEWER_TOOLS + '::shortcut_filepath'])
    if changed:
        # Its readers now resolve the name to cellacdc/__init__.py
        log_change(VIEWER_TOOLS,
                   [VIEWER_TOOLS + '::ShortcutsGuiElements.initShortcuts',
                    VIEWER_TOOLS + '::ShortcutsGuiElements.setShortcuts'],
                   'shortcut_filepath is imported from cellacdc')

    replace(repo, LAUNCHER,
            "            gui.favourite_func_metrics_csv_path, \n"
            "            # gui.custom_annot_path, \n"
            "            gui.shortcut_filepath, \n",
            "            favourite_func_metrics_csv_path, \n"
            "            # gui.custom_annot_path, \n"
            "            shortcut_filepath, \n",
            items=[LAUNCHER + '::mainWin.checkConfigFiles'])
    ensure_line_after(repo, LAUNCHER,
                      'from . import cellacdc_path',
                      'from . import favourite_func_metrics_csv_path, shortcut_filepath')


run(main)
