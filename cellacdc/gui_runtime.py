"""Shared Qt bootstrap for composable GUI variants."""

from __future__ import annotations

import os
import sys


def bootstrap_qt(*, splashscreen: bool = False):
    from cellacdc._run import _setup_app, _setup_gui_libraries, _setup_numpy

    requires_exit = _setup_gui_libraries(exit_at_end=False)
    _setup_numpy()
    if requires_exit:
        from cellacdc._run import _exit_on_setup

        _exit_on_setup()

    from qtpy import QtCore, QtWidgets

    try:
        QtWidgets.QApplication.setAttribute(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        pass

    import pyqtgraph as pg

    pg.setConfigOption("imageAxisOrder", "row-major")

    if os.name == "nt":
        try:
            import ctypes

            myappid = "schmollerlab.cellacdc.pyqt.v1"
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception:
            pass

    app, splash = _setup_app(splashscreen=splashscreen)
    return app, splash


def run_event_loop(app):
    sys.exit(app.exec_())
