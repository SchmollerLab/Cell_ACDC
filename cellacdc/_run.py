import os
import shutil
import sys
from importlib import import_module

def _install_tables(parent_software='Cell-ACDC'):
    try:
        import tables
    except Exception as e:
        if parent_software == 'Cell-ACDC':
            issues_url = 'https://github.com/SchmollerLab/Cell_ACDC/issues'
            note_txt = (
                'If the installation fails, you can still use Cell-ACDC, but we '
                'highly recommend you report the issue (see link below) and we '
                'will be very happy to help. Thank you for your patience!'
                
            )
        else:
            issues_url = 'https://github.com/SchmollerLab/Cell_ACDC/issues'
            note_txt = (
                'If the installation fails, report the issue (see link below) and we '
                'will be very happy to help. Thank you for your patience!'
            )
        while True:
            txt = (
                f'{parent_software} needs to install a library called `tables`.\n\n'
                f'{note_txt}\n\n'
                f'Report issue here: {issues_url}\n'
            )
            print('-'*60)
            print(txt)
            answer = input('Do you want to install it now ([y]/n)? ')
            if answer.lower() == 'y' or not answer:
                try:
                    import subprocess
                    subprocess.check_call(
                        [sys.executable, '-m', 'pip', 'install', '-U', 'tables']
                    )
                except Exception as err:
                    print('-'*60)
                    print(
                        '[WARNING]: Installation of `tables` with pip failed. '
                        'Trying with conda...'
                    )
                    print('-'*60)
                try:
                    import subprocess
                    subprocess.check_call(
                        ['conda', 'install', '-y', 'pytables']
                    )
                except Exception as err:
                    import traceback
                    traceback.print_exc()
                    print('*'*60)
                    if parent_software == 'Cell-ACDC':
                        msg_type = '[WARNING]'
                        log_func = print
                    else:
                        msg_type = '[ERROR]'
                        log_func = exit
                    
                    log_func(
                        f'{msg_type}: Installation of `tables` failed. '
                        'Please report the issue here (**including the error message above**): '
                        f'{issues_url}'
                    )
                    print('^'*60)
                finally:
                    break
            elif answer.lower() == 'n':
                raise e
            else:
                print(
                    f'"{answer}" is not a valid answer. '
                    'Type "y" for "yes", or "n" for "no".'
                )

def _setup_gui_libraries():
    
    
    warn_restart = False
    
    # Force PyQt6 if available
    try:
        from PyQt6 import QtCore
        os.environ["QT_API"] = "pyqt6"
    except Exception as e:
        pass

    try:
        import qtpy
    except ModuleNotFoundError as e:
        txt = (
            'Since version 1.3.1 Cell-ACDC requires the package `qtpy`.\n\n'
            'You can let Cell-ACDC install it now, or you can abort '
            'and install it manually with the command `pip install qtpy`.'
        )
        print('-'*60)
        print(txt)
        while True:
            answer = input('Do you want to install it now ([y]/n)? ')
            if answer.lower() == 'y' or not answer:
                import subprocess
                subprocess.check_call(
                    [sys.executable, '-m', 'pip', 'install', '-U', 'qtpy']
                )
                break
            elif answer.lower() == 'n':
                raise e
            else:
                print(
                    f'"{answer}" is not a valid answer. '
                    'Type "y" for "yes", or "n" for "no".'
                )
    except ImportError as e:
        # Ignore that qtpy is installed but there is no PyQt bindings --> this 
        # is handled in the next block
        pass
    
    from . import is_mac_arm64
    default_qt = 'PyQt5' if is_mac_arm64 else 'PyQt6'
    
    try:
        from qtpy.QtCore import Qt
    except Exception as e:
        txt = (
            'Since version 1.3.1 Cell-ACDC does not install a GUI library by default.\n\n'
            f'You can let Cell-ACDC install it now (default library is `{default_qt}`), '
            'or you can abort (press "n")\n'
            'and install a compatible GUI library with one of '
            'the following commands:\n\n'
            '    * pip install PyQt6\n'
            '    * pip install PyQt5 (or `conda install pyqt`)\n'
            '    * pip install PySide2\n'
            '    * pip install PySide6\n\n'
            f'Note: if `{default_qt}` installation fails, you could try installing any '
            'of the other libraries.\n\n'
        )
        print('-'*60)
        print(txt)
        while True:
            answer = input(f'Do you want to install {default_qt} now ([y]/n)? ')
            if answer.lower() == 'y' or not answer:
                import subprocess
                if is_mac_arm64:
                    subprocess.check_call(
                        ['conda', 'install', '-y', 'pyqt']
                    )
                else:
                    subprocess.check_call(
                        [sys.executable, '-m', 'pip', 'install', '-U', 'PyQt6']
                    )
                warn_restart = True
                break
            elif answer.lower() == 'n':
                raise e
            else:
                print(
                    f'"{answer}" is not a valid answer. '
                    'Type "y" for "yes", or "n" for "no".'
                )
    
    try:
        import pyqtgraph
    except ModuleNotFoundError:
        import subprocess
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '-U', 'pyqtgraph']
        )
        warn_restart = True
    
    try:
        import seaborn
    except ModuleNotFoundError:
        import subprocess
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '-U', 'seaborn']
        )
        warn_restart = True
    
    if warn_restart:
        print('*'*60)
        input(
            '[WARNING]: Cell-ACDC had to install the required GUI libraries. '
            'Please, re-start the software. Thank you for your patience! '
            '(Press any key to exit). '
        )
        exit()

def _setup_app(splashscreen=False, icon_path=None, logo_path=None):
    from qtpy import QtCore
    if QtCore.QCoreApplication.instance() is not None:
        return QtCore.QCoreApplication.instance(), None
    
    from qtpy import QtWidgets
    # Handle high resolution displays:
    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    if hasattr(QtCore.Qt, 'AA_PluginApplication'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_PluginApplication, False)

    # Check OS dark or light mode
    from qtpy.QtWidgets import QApplication, QStyleFactory
    from qtpy.QtGui import QPalette, QIcon
    from . import settings_csv_path, resources_folderpath
    
    app = QApplication([])
    app.setStyle(QStyleFactory.create('Fusion'))
    is_OS_dark_mode = app.palette().color(QPalette.Window).getHsl()[2] < 100
    app.toggle_dark_mode = False
    if is_OS_dark_mode:
        # Switch to dark mode if scheme was never selected by user and OS is 
        # dark mode
        import pandas as pd
        df_settings = pd.read_csv(settings_csv_path, index_col='setting')
        if 'colorScheme' not in df_settings.index:
            app.toggle_dark_mode = True
    
    if icon_path is None:
        icon_path = os.path.join(resources_folderpath, 'icon.ico')
        app.setWindowIcon(QIcon(icon_path))
    
    if logo_path is None:
        logo_path = os.path.join(resources_folderpath, 'logo.png')
    
    from qtpy import QtWidgets, QtGui

    splashScreen = None
    if splashscreen:
        class SplashScreen(QtWidgets.QSplashScreen):
            def __init__(self, logo_path, icon_path):
                super().__init__()
                self.setPixmap(QtGui.QPixmap(logo_path))
                self.setWindowIcon(QIcon(icon_path))
                self.setWindowFlags(
                    QtCore.Qt.WindowStaysOnTopHint 
                    | QtCore.Qt.SplashScreen 
                    | QtCore.Qt.FramelessWindowHint
                )
            
            def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:
                pass
            
            def showEvent(self, event):
                self.raise_()
        
        # Launch splashscreen
        splashScreen = SplashScreen(logo_path, icon_path)
        splashScreen.show()  
    
    from ._palettes import getPaletteColorScheme, setToolTipStyleSheet
    from ._palettes import get_color_scheme
    from . import qrc_resources_path
    from .qrc_resources import qt_resource_data
    from . import printl
    
    # Check if there are new icons --> replace qrc_resources.py
    scheme = get_color_scheme()
    if scheme == 'light':
        from . import qrc_resources_light_path as qrc_resources_scheme_path
        qrc_resources_scheme = import_module('cellacdc.qrc_resources_light')
        qt_resource_data_scheme = qrc_resources_scheme.qt_resource_data
    else:
        from . import qrc_resources_dark_path as qrc_resources_scheme_path
        qrc_resources_scheme = import_module('cellacdc.qrc_resources_dark')
        qt_resource_data_scheme = qrc_resources_scheme.qt_resource_data
    
    if qt_resource_data_scheme != qt_resource_data:
        # When we add new icons the qrc_resources.py file needs to be replaced
        shutil.copyfile(qrc_resources_scheme_path, qrc_resources_path)
    
    from . import load
    scheme = get_color_scheme()
    palette = getPaletteColorScheme(app.palette(), scheme=scheme)
    app.setPalette(palette)     
    # load.rename_qrc_resources_file(scheme)
    # setToolTipStyleSheet(app, scheme=scheme)
    
    return app, splashScreen

def run_segm_workflow(workflow_params, logger, log_path):
    logger.info('Initializing segmentation and tracking kernel...')
    from cellacdc import core
    kernel = core.SegmKernel(logger, log_path, is_cli=True)
    kernel.init_args_from_params(workflow_params, logger.info)
    ch_filepaths = kernel.parse_paths(workflow_params)
    stop_frame_nums = kernel.parse_stop_frame_numbers(workflow_params)
    for ch_filepath, stop_frame_n in zip(ch_filepaths, stop_frame_nums):
        logger.info(f'Processing "{ch_filepath}"...')
        kernel.run(ch_filepath, stop_frame_n)

def run_cli(ini_filepath):
    from cellacdc import myutils
    logger, logs_path, log_path, log_filename = myutils.setupLogger(
        module='cli', logs_path=None
    )
    
    logger.info(f'Reading workflow file "{ini_filepath}"...')
    from cellacdc import load
    workflow_params = load.read_segm_workflow_from_config(ini_filepath)
    workflow_type = workflow_params['workflow']['type']
    
    if workflow_type == 'segmentation and/or tracking':
        run_segm_workflow(workflow_params, logger, log_path)
    
    
    