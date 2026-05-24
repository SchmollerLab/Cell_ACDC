"""Minimal composable GUI — foundation for feature-specific variants."""

from __future__ import annotations

import os
import sys

from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget

from . import myutils
from .gui_bundles import BASIC_GUI_ROOTS
from .gui_runtime import bootstrap_qt, run_event_loop
from .mixins import AppShell
from .myutils import setupLogger


class BasicGuiWin(QMainWindow, AppShell):
    """Small GUI built from the basic mixin bundle."""

    def __init__(self, app, parent=None, version=None):
        super().__init__(parent)
        self.app = app
        self._version = version
        self._appName = "Cell-ACDC Basic"
        self.mainWin = None
        self.launcherSlot = None
        self.buttonToRestore = None
        self.closeGUI = False
        self._acdc_version = myutils.read_version()
        self.newWindows = []

        from .config import parser_args

        self.debug = parser_args["debug"]

    def run(self, module="acdc_gui_basic", logs_path=None):
        QMainWindow.setWindowIcon(self, QIcon(":icon.ico"))
        QMainWindow.setWindowTitle(self, "Cell-ACDC Basic")

        logger, logs_path, log_path, log_filename = setupLogger(
            module=module, logs_path=logs_path, caller="Cell-ACDC"
        )
        self.module = module
        self.logger = logger
        self.log_path = log_path
        self.log_filename = log_filename
        self.logs_path = logs_path
        self.logger.info("Initializing basic GUI")

        self.loadLastSessionSettings()
        self.is_error_state = False
        self.pos_i = 0
        self.isDataLoaded = False

        self._build_minimal_chrome()
        self.show()
        self.logger.info("Basic GUI ready")

    def _build_minimal_chrome(self):
        from qtpy.QtGui import QAction

        file_menu = self.menuBar().addMenu("&File")
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = self.menuBar().addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.showAbout)
        help_menu.addAction(about_action)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(
            QLabel(
                "Basic GUI\n\n"
                f"Mixin bundle: {', '.join(BASIC_GUI_ROOTS)}\n"
                "Next: add segmentation and data-loading mixins here."
            ),
            alignment=Qt.AlignCenter,
        )
        self.setCentralWidget(central)

        status = self.statusBar()
        status.showMessage("Ready")


def main():
    app, _splash = bootstrap_qt()
    version = myutils.read_version()
    win = BasicGuiWin(app, version=version)
    win.run()
    run_event_loop(app)


if __name__ == "__main__":
    main()
