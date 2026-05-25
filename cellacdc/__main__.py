#!/usr/bin/env python
import os
import logging

import os
import numpy as np

import site
sitepackages = site.getsitepackages()
site_packages = [p for p in sitepackages if p.endswith('site-packages')][0]

cellacdc_path = os.path.dirname(os.path.abspath(__file__))
cellacdc_installation_path = os.path.dirname(cellacdc_path)
if cellacdc_installation_path != site_packages:
    # Running developer version. Delete cellacdc folder from site_packages 
    # if present from a previous installation of cellacdc from PyPi
    cellacdc_path_pypi = os.path.join(site_packages, 'cellacdc')
    if os.path.exists(cellacdc_path_pypi):
        import shutil
        try:
            shutil.rmtree(cellacdc_path_pypi)
        except Exception as err:
            print(err)
            print(
                '[ERROR]: Previous Cell-ACDC installation detected. '
                f'Please, manually delete this folder and re-start the software '
                f'"{cellacdc_path_pypi}". '
                'Thank you for you patience!'
            )
            exit()
        print('*'*60)
        input(
            '[WARNING]: Cell-ACDC had to clean-up and older installation. '
            'Please, re-start the software. Thank you for your patience! '
            '(Press any key to exit). '
        )
        exit()

from cellacdc import _run


def _handle_parser_early_exit():
    from cellacdc.config import parser_args

    if parser_args['version'] or parser_args['info']:
        from cellacdc.myutils import get_info_version_text
        print(get_info_version_text())
        exit()

    if parser_args['reset']:
        from cellacdc.myutils import reset_settings
        print(reset_settings())
        exit()


def _bootstrap_gui_app():
    from ._run import (
        _setup_gui_libraries,
        _setup_symlink_app_name_macos,
        _setup_numpy,
        download_model_params,
        _exit_on_setup,
    )

    _setup_symlink_app_name_macos()

    requires_exit = _setup_gui_libraries(exit_at_end=False)

    _setup_numpy()

    download_model_params()

    if requires_exit:
        _exit_on_setup()

    from qtpy import QtWidgets, QtCore

    if os.name == 'nt':
        try:
            import ctypes
            myappid = 'schmollerlab.cellacdc.pyqt.v1'
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception:
            pass

    try:
        QtWidgets.QApplication.setAttribute(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        pass

    import pyqtgraph as pg
    pg.setConfigOption('imageAxisOrder', 'row-major')
    try:
        import numba  # noqa: F401
        pg.setConfigOption('useNumba', True)
    except Exception:
        pass

    try:
        import cupy as cp  # noqa: F401
        pg.setConfigOption('useCupy', True)
    except Exception:
        pass

    return _run._setup_app(splashscreen=True)


def _read_gui_version(logger_func=None):
    from cellacdc import myutils

    version, success = myutils.read_version(
        logger=logger_func, return_success=True
    )
    if success:
        return version

    error = myutils.check_install_package(
        'setuptools_scm', pypi_name='setuptools-scm'
    )
    if error and logger_func is not None:
        logger_func(error)
    return myutils.read_version(logger=logger_func)


def run():
    from cellacdc.config import parser_args

    _handle_parser_early_exit()

    PARAMS_PATH = parser_args['params']

    if PARAMS_PATH:
        _run.run_cli(PARAMS_PATH)
    else:
        run_gui()


def main():
    # Keep compatibility with users that installed older versions
    # where the entry point was main()
    run()


def run_gui():
    app, splashScreen = _bootstrap_gui_app()

    from cellacdc import myutils

    print('Launching application...')

    from cellacdc._main import mainWin

    if not splashScreen.isVisible():
        splashScreen.show()

    win = mainWin(app)

    try:
        myutils.check_matplotlib_version(qparent=win)
    except Exception:
        pass

    version = _read_gui_version(logger_func=win.logger.info)
    win.setVersion(version)
    win.launchWelcomeGuide()
    win.show()
    try:
        win.welcomeGuide.showPage(win.welcomeGuide.welcomeItem)
    except AttributeError:
        pass
    win.logger.info('**********************************************')
    win.logger.info(f'Welcome to Cell-ACDC v{version}')
    win.logger.info('**********************************************')
    win.logger.info('----------------------------------------------')
    win.logger.info(
        'NOTE: If application is not visible, it is probably minimized\n'
        'or behind some other open windows.'
    )
    win.logger.info('----------------------------------------------')
    splashScreen.close()
    app.exec_()


def run_gui_direct():
    """Launch the annotation GUI directly, skipping the module launcher."""
    _handle_parser_early_exit()

    app, splashScreen = _bootstrap_gui_app()

    from cellacdc import myutils
    from cellacdc.gui import guiWin

    print('Launching GUI...')

    if not splashScreen.isVisible():
        splashScreen.show()

    version = _read_gui_version()
    gui_windows = []

    def launch_gui_window(checked=False):
        win = guiWin(
            app,
            mainWin=None,
            version=version,
            launcherSlot=launch_gui_window,
        )
        gui_windows.append(win)
        win.sigClosed.connect(_gui_window_closed)
        win.run()
        return win

    def _gui_window_closed(closed_win):
        try:
            gui_windows.remove(closed_win)
        except ValueError:
            pass

    win = launch_gui_window()

    try:
        myutils.check_matplotlib_version(qparent=win)
    except Exception:
        pass

    win.logger.info('**********************************************')
    win.logger.info(f'Welcome to Cell-ACDC GUI v{version}')
    win.logger.info('**********************************************')
    win.logger.info('----------------------------------------------')
    win.logger.info(
        'NOTE: If application is not visible, it is probably minimized\n'
        'or behind some other open windows.'
    )
    win.logger.info('----------------------------------------------')
    splashScreen.close()
    app.exec_()
