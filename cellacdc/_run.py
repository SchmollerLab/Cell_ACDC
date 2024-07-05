import os
import shutil
import sys
from importlib import import_module
import traceback
from tqdm import tqdm

def _install_tables(parent_software='Cell-ACDC'):
    from . import try_input_install_package, is_conda_env
    try:
        import tables
        return False
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
            if is_conda_env():
                command_txt = 'conda install pytables'
                alt_command_txt = 'pip install --upgrade tables'
                cmd_args = [command_txt]
                alt_cmd_args1 = command_txt.split(' ')
                alt_cmd_args2 = [sys.executable, '-m', *alt_command_txt.split(' ')]
                pkg_mng = 'conda'
                alt_pkg_mng = 'pip'
                shell = True
                alt_shell = False
            else:
                alt_command_txt = 'conda install pytables'
                command_txt = 'pip install --upgrade tables'
                cmd_args = [sys.executable, '-m', *command_txt.split(' ')]
                alt_cmd_args1 = alt_command_txt.split(' ')
                alt_cmd_args2 = [alt_command_txt]
                pkg_mng = 'pip'
                alt_pkg_mng = 'conda'
                shell = False
                alt_shell = True
                
            answer = try_input_install_package('tables', command_txt)
            
            if answer.lower() == 'y' or not answer:
                import subprocess, traceback
                try:
                    subprocess.check_call(cmd_args, shell=shell)
                    break
                except Exception as err:
                    traceback.print_exc()
                    print('-'*100)
                    print(
                        f'[WARNING]: Installation with command `{cmd_args}` '
                        f'failed. Trying with `{alt_cmd_args1}`...'
                    )
                    print('-'*100)
                
                try:
                    subprocess.check_call(alt_cmd_args1, shell=shell)
                    break
                except Exception as err:
                    traceback.print_exc()
                    print('-'*100)
                    print(
                        f'[WARNING]: Installation of `tables` with '
                        f'{pkg_mng} failed. Trying with {alt_pkg_mng}...'
                    )
                    print('-'*100)
                
                import pdb; pdb.set_trace()
                try:
                    subprocess.check_call(alt_cmd_args2, shell=alt_shell)
                    break
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
                        'Please report the issue here (**including the error '
                        f'message above**): {issues_url}'
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

        return True

def _setup_symlink_app_name_macos():
    """On Mac generate a symlink from the Python path defined in the shebang 
    of the `acdc` binary called Cell-ACDC and modify the shebang to run 
    the acdc binary from the symlink. This will correctly display Cell-ACDC 
    in the menubar instead of Python.
    """    
    from . import is_mac, printl
    if not is_mac:
        return
    
    import subprocess
    acdc_binary_path = os.path.dirname(sys.executable)
    symlink = os.path.join(acdc_binary_path, 'Cell-ACDC')
    if os.path.exists(symlink):
        return
    
    for acdc_exec_name in ('acdc', 'cellacdc'):
        acdc_exec_path = os.path.join(acdc_binary_path, acdc_exec_name)
        try:
            with open(acdc_exec_path, 'r') as bin:
                acdc_exec_text = bin.read()
                shebang = acdc_exec_text.split('\n')[0][2:]
            if not os.path.exists(symlink):
                command = f'ln -s {shebang} {symlink}'
                subprocess.check_call(command, shell=True)
            acdc_exec_text = acdc_exec_text.replace(shebang, symlink)
            with open(acdc_exec_path, 'w') as bin:
                bin.write(acdc_exec_text)
        except Exception as err:
            printl(traceback.format_exc())
            print('[WARNING]: Failed at creating Cell-ACDC symlink')

def _setup_gui_libraries(caller_name='Cell-ACDC'):
    from . import try_input_install_package, is_conda_env
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
            f'{caller_name} needs to install the package `qtpy`.\n\n'
            f'You can let {caller_name} install it now, or you can abort '
            'and install it manually with the command `pip install qtpy`.'
        )
        print('-'*60)
        print(txt)
        command_txt = 'pip install --upgrade qtpy'   
        while True:
            answer = try_input_install_package('qtpy', command_txt)
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
        traceback.print_exc()
        txt = (
            f'{caller_name} needs to install a GUI library (default library is '
            f'`{default_qt}`).\n\n'
            'You can install it now or you can close (press "n") and install\n'
            'a compatible GUI library with one of '
            'the following commands:\n\n'
            '    * pip install PyQt6==6.6.0 PyQt6-Qt6==6.6.0\n'
            '    * pip install PyQt5 (or `conda install pyqt`)\n'
            '    * pip install PySide2\n'
            '    * pip install PySide6\n\n'
            f'Note: if `{default_qt}` installation fails, you could try installing any '
            'of the other libraries.\n'
        )
        print('-'*60)
        print(txt)
        pip_command = 'pip install -U PyQt6==6.6.0 PyQt6-Qt6==6.6.0'
        if is_mac_arm64:
            commnad_txt = 'conda install -y pyqt'
            pkg_name = 'pyqt'
        else:
            commnad_txt = pip_command
            pkg_name = 'PyQt6'
        while True:
            answer = try_input_install_package(pkg_name, commnad_txt)
            if answer.lower() == 'y' or not answer:
                import subprocess
                if is_mac_arm64 and is_conda_env():
                    subprocess.check_call(
                        ['conda install -y pyqt'], shell=True
                    )
                else:
                    pip_args = pip_command.split()
                    subprocess.check_call([sys.executable, '-m', *pip_args])
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
        version = pyqtgraph.__version__.split('.')
        pg_major, pg_minor, pg_patch = [int(val) for val in version]
        # if pg_major < 1:
        #     raise ModuleNotFoundError('pyqtgraph must be upgraded')
        if pg_minor < 13:
            raise ModuleNotFoundError('pyqtgraph must be upgraded')
        if pg_patch < 7:
            raise ModuleNotFoundError('pyqtgraph must be upgraded')
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
            f'[NOTE]: {caller_name} had to install the required GUI libraries. '
            'Please, re-start the software. Thank you for your patience! '
            '(Press any key to exit). '
        )
        exit()

def _setup_app(splashscreen=False, icon_path=None, logo_path=None, scheme=None):
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
    if scheme is None:
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
    pbar = tqdm(total=len(ch_filepaths), ncols=100)
    for ch_filepath, stop_frame_n in zip(ch_filepaths, stop_frame_nums):
        logger.info(f'\nProcessing "{ch_filepath}"...')
        kernel.run(ch_filepath, stop_frame_n)
        pbar.update()
    pbar.close()

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
    
    logger.info('**********************************************')
    logger.info(f'Cell-ACDC command-line closed. {myutils.get_salute_string()}')
    logger.info('**********************************************')
    
    
    