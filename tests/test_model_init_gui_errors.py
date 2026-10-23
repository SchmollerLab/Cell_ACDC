import os
import subprocess
import os
import sys
import textwrap

import pytest


@pytest.mark.parametrize("entry_point", ["segmentation", "budding"])
def test_model_initialization_error_does_not_escape_qt_callback(entry_point):
    """An escaped Qt callback exception can abort PyQt; isolate the regression."""
    script = textwrap.dedent("""
        import sys
        from types import SimpleNamespace
        from unittest.mock import Mock

        from qtpy.QtCore import QTimer
        from qtpy.QtWidgets import QApplication, QDialog

        import cellacdc
        from cellacdc import gui, segm
        from cellacdc.service.client import LocalBackend
        from cellacdc.service.model_execution import run_in_background

        app = QApplication([])
        app.setQuitOnLastWindowClosed(False)
        # Do not let an escaped Qt exception pass as a successful test run.
        sys.excepthook = lambda *args: os._exit(97)
        errors = []
        def report_error(window, name):
            errors.append((name, sys.exc_info()[1]))
            dialog = QDialog()
            QTimer.singleShot(0, dialog.accept)
            dialog.exec_()

        cellacdc._critical_exception_gui = report_error
        cellacdc._exception_handler_clean_progress = lambda window: None

        backend = LocalBackend(
            "skip_segmentation", env_python=sys.executable,
            init_model_kwargs={"segm_data": None},
        )

        def initialize():
            return run_in_background(None, backend.start)

        if sys.argv[1] == "segmentation":
            selector = SimpleNamespace(exec_=initialize)
            segm.apps.SelectFoldersToAnalyse = Mock(return_value=selector)
            window = SimpleNamespace()
            callback = lambda: segm.segmWin.main(window)
            expected_name = "main"
        else:
            class Window:
                @property
                def is_win(self):
                    return initialize()

            window = Window()
            callback = lambda: gui.guiWin.autoAssignBud_YeastMate(window)
            expected_name = "autoAssignBud_YeastMate"

        def verify_alive():
            assert len(errors) == 1, errors
            assert errors[0][0] == expected_name, errors
            error = errors[0][1]
            assert isinstance(error, RuntimeError), error
            assert "Initializing the skip_segmentation model failed" in str(error)
            assert "Model server error 500" in str(error)
            assert not backend.is_running()
            assert run_in_background(None, lambda: 42) == 42
            print("GUI survived initialization failure", flush=True)
            app.quit()

        # Run verification after the callback's nested background event loop.
        def initialize_and_verify():
            callback()
            verify_alive()

        QTimer.singleShot(0, initialize_and_verify)
        app.exec_()
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, entry_point],
        capture_output=True, text=True, timeout=60,
        env=dict(os.environ, QT_QPA_PLATFORM="offscreen"),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "GUI survived initialization failure" in result.stdout
