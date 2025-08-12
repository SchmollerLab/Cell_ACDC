import os
import shutil
import sys
from importlib import import_module
import traceback
from tqdm import tqdm
from . import config, myutils

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
            conda_prefix, pip_prefix = myutils.get_pip_conda_prefix()
            conda_list, pip_list = myutils.get_pip_conda_prefix(list_return=True)

            conda_txt = f'{conda_prefix} pytables'
            pip_text = f'{pip_prefix} --upgrade tables'

            conda_list = conda_list + ['pytables']
            pip_list = pip_list + ['--upgrade', 'tables']
            if is_conda_env():
                command_txt = conda_txt
                alt_command_txt = pip_text
                cmd_args = [command_txt]
                alt_cmd_args1 = conda_list
                alt_cmd_args2 = pip_list
                pkg_mng = 'conda'
                alt_pkg_mng = 'pip'
                shell = True
                alt_shell = False
            else:
                alt_command_txt = conda_txt
                command_txt = pip_text
                cmd_args = pip_list
                alt_cmd_args1 = conda_list
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

def _setup_gui_libraries(caller_name='Cell-ACDC', exit_at_end=True):
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
        conda_prefix, pip_prefix = myutils.get_pip_conda_prefix()
        conda_list, pip_list = myutils.get_pip_conda_prefix(list_return=True)
        
        command_txt = f'{pip_prefix} --upgrade qtpy'
        
        txt = (
            f'{caller_name} needs to install the package `qtpy`.\n\n'
            f'You can let {caller_name} install it now, or you can abort '
            f'and install it manually with the command `{command_txt}`.'
        )
        print('-'*60)
        print(txt)

        while True:
            from .config import parser_args
            if parser_args['yes']:
                answer = 'y'
            else:
                answer = try_input_install_package('qtpy', command_txt)
            
            if answer.lower() == 'y' or not answer:
                import subprocess
                cmd = pip_list + ['-U', 'qtpy']
                subprocess.check_call(cmd)
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
    
    try: # no need to handle no_cli, acdc is run with -y flag
        from qtpy.QtCore import Qt
    except Exception as e:
        traceback.print_exc()
        txt = (
            f'{caller_name} needs to install a GUI library (default library is '
            f'`{default_qt}`).\n\n'
            'You can install it now or you can close (press "n") and install\n'
            'a compatible GUI library with one of '
            'the following commands:\n\n'
            f'    * {pip_prefix} PyQt6==6.6.0 PyQt6-Qt6==6.6.0\n'
            f'    * {pip_prefix} PyQt5 (or `conda install pyqt`)\n'
            f'    * {pip_prefix} PySide2\n'
            f'    * {pip_prefix} PySide6\n\n'
            f'Note: If `{default_qt}` installation fails, you could try installing any '
            'of the other libraries.\n'
        )
        print('-'*60)
        print(txt)
        pip_command = f'{pip_prefix} -U PyQt6==6.6.0 PyQt6-Qt6==6.6.0'
        if is_mac_arm64:
            commnad_txt = f'{conda_prefix} pyqt'
            pkg_name = 'pyqt'
        else:
            commnad_txt = pip_command
            pkg_name = 'PyQt6'
        while True:
            from .config import parser_args
            if parser_args['yes']:
                answer = 'y'
            else:
                answer = try_input_install_package(pkg_name, commnad_txt)
            if answer.lower() == 'y' or not answer:
                import subprocess
                if is_mac_arm64 and is_conda_env():
                    subprocess.check_call(
                        [f'{conda_prefix} pyqt'], shell=True
                    )
                else:
                    pip_args = pip_list + ['-U', 'PyQt6==6.6.0', 'PyQt6-Qt6==6.6.0']
                    subprocess.check_call(pip_args)
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
    
    if not warn_restart:
        return warn_restart
    
    if not exit_at_end:
        return warn_restart
    
    _exit_on_setup(caller_name=caller_name)
    
    return warn_restart

def _exit_on_setup(caller_name='Cell-ACDC'):
    print('*'*60)
    note_text = (
        f'[NOTE]: {caller_name} had to install the required libraries. '
    )
    note_text = (
        f'{note_text}'
        'Please, re-start the software. Thank you for your patience! '
    )
    
    from .config import parser_args
    if parser_args['yes']:
        print(note_text)
    else:
        note_text = (
            f'{note_text}'
            '(Press any key to exit). '
        )
        input(note_text)
        
    exit()
    
def download_model_params():
    print("Downloading specified models...")
    from .config import parser_args
    if parser_args['cpModelsDownload'] or parser_args['AllModelsDownload']:
        print('[INFO]: Downloading Cellpose models...')
        from cellpose import models
        model_names = ["cyto", "cyto2", "cyto3", "nuclei"]

        try:
            # download size model weights

            from cellpose.models import size_model_path, model_path
            for model_name in model_names:
                print(f'[INFO]: Downloading {model_name} model weights...')
                try:
                    size_model_path(model_name)
                    model_path(model_name)
                except Exception as e:
                    print(
                        f'[WARNING]: Failed to download {model_name} model weights. '
                    )
                    print(e)
                    pass
            
            from cellpose.denoise import MODEL_NAMES
            for model_name in MODEL_NAMES:
                print(f'[INFO]: Downloading {model_name} model weights...')
                try:
                    model_path(model_name)
                except Exception as e:
                    print(
                        f'[WARNING]: Failed to download {model_name} model weights. '
                    )
                    if model_name in ["oneclick_per_cyto2", 
                                      "oneclick_seg_cyto2", 
                                      "oneclick_rec_cyto2",
                                      "oneclick_per_nuclei",
                                      "oneclick_seg_nuclei",
                                      "oneclick_rec_nuclei"]:
                        print(f' This model is not available for download. ')
                    print(e)
                    pass
        except Exception as e:
            print(
                '[WARNING]: Failed to download Cellpose model weights. '
            )
            print(e)
            pass
    if parser_args['StarDistModelsDownload'] or parser_args['AllModelsDownload']:
        print('[INFO]: Downloading StarDist models...')
        try:
            from cellacdc.models import STARDIST_MODELS
            from stardist.models import StarDist2D, StarDist3D
            for model_type in [StarDist2D, StarDist3D]:
                for model_name in STARDIST_MODELS:
                    print(f'[INFO]: Downloading {model_name} model weights...')
                    try:
                        model_type.from_pretrained(model_name)
                    except Exception as e:
                        print(
                            f'[WARNING]: Failed to download {model_name} model weights. '
                        )
                        print(e)
                        pass
        except Exception as e:
            print(
                '[WARNING]: Failed to download StarDist model weights. '
            )
            print(e)
            pass
    if parser_args['YeaZModelsDownload'] or parser_args['AllModelsDownload']:
        print('[INFO]: Downloading YeaZ models...')
        from cellacdc.myutils import _download_yeaz_models
        try:
            _download_yeaz_models()
        except Exception as e:
            print(
                '[WARNING]: Failed to download YeaZ model weights. '
            )
            print(e)
            pass
    if parser_args['DeepSeaModelsDownload'] or parser_args['AllModelsDownload']:
        print('[INFO]: Downloading DeepSea models...')
        from cellacdc.myutils import _download_deepsea_models
        try:
            _download_deepsea_models()
        except Exception as e:
            print(
                '[WARNING]: Failed to download DeepSea model weights. '
            )
            print(e)
            pass

    if parser_args['TrackastraModelsDownload'] or parser_args['AllModelsDownload']:
        print('[INFO]: Downloading TrackAstra models...')
        # from cellacdc.myutils import _download_trackastra_models
        from trackastra.model import Trackastra
        try:
            from cellacdc.trackers.Trackastra import get_pretrained_model_names
            model_names = get_pretrained_model_names()
            for model_name in model_names:
                print(f'[INFO]: Downloading {model_name} model weights...')
                try:
                    Trackastra.from_pretrained(model_name)
                except Exception as e:
                    print(
                        f'[WARNING]: Failed to download {model_name} model weights. '
                    )
                    print(e)
                    pass
        except Exception as e:
            print(
                '[WARNING]: Failed to download TrackAstra model weights. '
            )
            print(e)
            pass
                
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
    
    app = QApplication(['Cell-ACDC'])
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
    from . import acdc_qrc_resources
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
    
    if qt_resource_data_scheme != acdc_qrc_resources.qt_resource_data:
        from . import _copy_qrc_resources_file
        proceed = _copy_qrc_resources_file(qrc_resources_scheme_path)
        if not proceed:
            print('-'*100)
            print(
                'Cell-ACDC had to reset the GUI icons. '
                'Please re-start the application. Thank you for your patience!'
            )
            print('-'*100)
            exit()
            
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
    
    download_model_params()
    
    logger.info(f'Reading workflow file "{ini_filepath}"...')
    from cellacdc import load
    workflow_params = load.read_segm_workflow_from_config(ini_filepath)
    workflow_type = workflow_params['workflow']['type']
    
    if workflow_type == 'segmentation and/or tracking':
        run_segm_workflow(workflow_params, logger, log_path)
    
    logger.info('**********************************************')
    logger.info(f'Cell-ACDC command-line closed. {myutils.get_salute_string()}')
    logger.info('**********************************************')
    
    
def _setup_numpy(caller_name='Cell-ACDC'):
    import urllib.request
    import json
    import re
    
    from . import try_input_install_package
    
    numpy_versions = []
    url = "https://pypi.org/pypi/numba/json"
    try:
        with urllib.request.urlopen(url) as response:
            data = json.load(response)
            requires_dist = data["info"].get("requires_dist", [])
            numpy_versions = [
                req for req in requires_dist if "numpy" in req.lower()
            ]
    except urllib.error.URLError as e:
        print(f"Could not update np: {e}")
        return
    
    if not numpy_versions:
        print(
            f'[WARNING]: Could not find NumPy version requirements for Numba. '
            'Please, install the latest version of NumPy manually.'
        )
        return
    
    numpy_versions_txt = numpy_versions[0]
    
    max_version = re.findall(r'<=?(\d+\.\d+)', numpy_versions_txt)
    min_version = re.findall(r'>=?(\d+\.\d+)', numpy_versions_txt)
    if max_version:
        max_version = max_version[0]
    else:
        max_version = ''
        
    if min_version:
        min_version = min_version[0]
    else:
        min_version = ''
    
    import numpy
    installed_numpy_version = numpy.__version__
    is_numpy_version_within_range = myutils.is_pkg_version_within_range(
        installed_numpy_version,
        min_version=min_version,
        max_version=max_version
    )
    
    if is_numpy_version_within_range:
        return
    
    conda_prefix, pip_prefix = myutils.get_pip_conda_prefix()
    conda_list, pip_list = myutils.get_pip_conda_prefix(list_return=True)

    command_txt = f'{pip_prefix} --upgrade "{numpy_versions_txt}"'
    
    txt = (
        f'{caller_name} needs to upgrade the package `numpy`.\n\n'
        f'The current version is {installed_numpy_version}, but it needs to be '
        f'between {min_version} and {max_version}.\n\n'
        f'You can let {caller_name} install it now, or you can abort '
        f'and install it manually with the command `{command_txt}`.'
    )
    print('-'*60)
    print(txt)
    
    while True:
        from .config import parser_args
        if parser_args['yes']:
            answer = 'y'
        else:
            answer = try_input_install_package('qtpy', command_txt)
        
        if answer.lower() == 'y' or not answer:
            import subprocess
            cmd = pip_list + ['-U', numpy_versions_txt]
            subprocess.check_call(cmd)
            break
        elif answer.lower() == 'n':
            raise ModuleNotFoundError(f'Numba requires {numpy_versions_txt} ')
        else:
            print(
                f'"{answer}" is not a valid answer. '
                'Type "y" for "yes", or "n" for "no".'
            )