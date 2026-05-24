"""Qt event loop helpers for script and notebook usage."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING
from warnings import warn

if TYPE_CHECKING:
    from qtpy.QtWidgets import QApplication

_APP_REF = None
_IPYTHON_WAS_HERE_FIRST = "IPython" in sys.modules


def _ipython_has_eventloop() -> bool:
    ipy_module = sys.modules.get("IPython")
    if not ipy_module:
        return False

    shell = ipy_module.get_ipython()  # type: ignore[attr-defined]
    if not shell:
        return False

    return shell.active_eventloop == "qt"


def _pycharm_has_eventloop(app: QApplication) -> bool:
    in_pycharm = "PYCHARM_HOSTED" in os.environ
    in_event_loop = getattr(app, "_in_event_loop", False)
    return in_pycharm and in_event_loop


def get_qapp(*, splashscreen: bool = False):
    """Get or create the Qt QApplication used by Cell-ACDC."""
    global _APP_REF

    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        from cellacdc._run import setup_gui_runtime

        app, _splash = setup_gui_runtime(splashscreen=splashscreen)
        _APP_REF = app
    elif _APP_REF is None:
        _APP_REF = app

    return app


def quit_app() -> None:
    """Close open viewers and quit if Cell-ACDC started the QApplication."""
    from qtpy.QtWidgets import QApplication

    from cellacdc.viewer import Viewer

    for viewer in list(Viewer._instances):
        viewer.close()

    QApplication.closeAllWindows()

    app = QApplication.instance()
    if app is None:
        return

    if (
        QApplication.applicationName() == "Cell-ACDC"
        and not _ipython_has_eventloop()
    ):
        QApplication.quit()


def run(*, force: bool = False, max_loop_level: int = 1, _func_name: str = "run"):
    """Start the Qt event loop."""
    if _ipython_has_eventloop():
        return

    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()

    if app is not None and _pycharm_has_eventloop(app):
        return

    if app is None:
        raise RuntimeError(
            "No Qt app has been created. Create one with "
            "`cellacdc.get_qapp()` or `cellacdc.Viewer()`."
        )

    if not app.topLevelWidgets() and not force:
        warn(
            f"Refusing to run a QApplication with no topLevelWidgets. "
            f"To run the app anyway, use `{_func_name}(force=True)`.",
            stacklevel=2,
        )
        return

    if app.thread().loopLevel() >= max_loop_level:
        loops = app.thread().loopLevel()
        warn(
            f"A QApplication is already running with {loops} event loop(s). "
            f"To enter another event loop, use "
            f"`{_func_name}(max_loop_level={loops + 1})`.",
            stacklevel=2,
        )
        return

    app.exec_()
