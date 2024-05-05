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

def run():
    from cellacdc.config import parser_args

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
    from ._run import _setup_gui_libraries, _setup_symlink_app_name_macos
    
    _setup_symlink_app_name_macos()
    
    _setup_gui_libraries()
    
    from qtpy import QtGui, QtWidgets, QtCore
    # from . import qrc_resources

    if os.name == 'nt':
        try:
            # Set taskbar icon in windows
            import ctypes
            myappid = 'schmollerlab.cellacdc.pyqt.v1' # arbitrary string
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception as e:
            pass

    # Needed by pyqtgraph with display resolution scaling
    try:
        QtWidgets.QApplication.setAttribute(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception as e:
        pass

    import pyqtgraph as pg
    # Interpret image data as row-major instead of col-major
    pg.setConfigOption('imageAxisOrder', 'row-major')
    try:
        import numba
        pg.setConfigOption("useNumba", True)
    except Exception as e:
        pass

    try:
        import cupy as cp
        pg.setConfigOption("useCupy", True)
    except Exception as e:
        pass

    # Create the application
    app, splashScreen = _run._setup_app(splashscreen=True)

    from cellacdc import myutils, printl
    from cellacdc import qrc_resources

    print('Launching application...')

    from cellacdc._main import mainWin
    
    if not splashScreen.isVisible():
        splashScreen.show()
    
    win = mainWin(app)

    try:
        myutils.check_matplotlib_version(qparent=win)
    except Exception as e:
        pass
    version, success = myutils.read_version(
        logger=win.logger.info, return_success=True
    )
    if not success:
        error = myutils.check_install_package(
            'setuptools_scm', pypi_name='setuptools-scm'
        )
        if error:
            win.logger.info(error)
        else:
            version = myutils.read_version(logger=win.logger.info)
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
    win.logger.info('NOTE: If application is not visible, it is probably minimized\n'
        'or behind some other open windows.')
    win.logger.info('----------------------------------------------')
    splashScreen.close()
    # splashScreenApp.quit()
    # modernWin.show()
    app.exec_()