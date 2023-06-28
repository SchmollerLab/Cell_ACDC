import os
import shutil
import sys

def _setup_gui():
    from . import qrc_resources_path, qrc_resources_light_path
    
    # Force PyQt6 if available
    try:
        from PyQt6 import QtCore
        os.environ["QT_API"] = "pyqt6"
    except Exception as e:
        pass

    # Set default qrc resources
    if not os.path.exists(qrc_resources_path):
        # Load default light mode
        shutil.copyfile(qrc_resources_light_path, qrc_resources_path)

    # Replace 'from PyQt5' with 'from qtpy' in qrc_resources.py file
    try:
        with open(qrc_resources_path, 'r') as qrc_py:
            text = qrc_py.read()
            text = text.replace('from PyQt5', 'from qtpy')
        with open(qrc_resources_path, 'w') as qrc_py:
            qrc_py.write(text)
    except Exception as err:
        raise err

    try:
        import qtpy
    except ModuleNotFoundError as e:
        while True:
            txt = (
                'Since version 1.3.1 Cell-ACDC requires the package `qtpy`.\n\n'
                'You can let Cell-ACDC install it now, or you can abort '
                'and install it manually with the command `pip install qtpy`.'
            )
            print('-'*60)
            print(txt)
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

    try:
        from qtpy.QtCore import Qt
    except Exception as e:
        while True:
            txt = (
                'Since version 1.3.1 Cell-ACDC does not install a GUI library by default.\n\n'
                'You can let Cell-ACDC install it now (default library is `PyQt6`), '
                'or you can abort (press "n")\n'
                'and install a compatible GUI library with one of '
                'the following commands:\n\n'
                '    * pip install PyQt6\n'
                '    * pip install PyQt5\n'
                '    * pip install PySide2\n'
                '    * pip install PySide6\n\n'
                'Note: if `PyQt6` installation fails, you could try installing any '
                'of the other libraries.\n\n'
            )
            print('-'*60)
            print(txt)
            answer = input('Do you want to install PyQt6 now ([y]/n)? ')
            if answer.lower() == 'y' or not answer:
                import subprocess
                subprocess.check_call(
                    [sys.executable, '-m', 'pip', 'install', '-U', 'PyQt6']
                )
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
    
    try:
        import seaborn
    except ModuleNotFoundError:
        import subprocess
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '-U', 'seaborn']
        )

def _setup_app(splashscreen=False):
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
    if is_OS_dark_mode:
        import pandas as pd
        df_settings = pd.read_csv(settings_csv_path, index_col='setting')
        if 'colorScheme' not in df_settings.index:
            df_settings.at['colorScheme', 'value'] = 'dark'
            df_settings.to_csv(settings_csv_path)
    
    icon_path = os.path.join(resources_folderpath, 'icon.ico')
    app.setWindowIcon(QIcon(icon_path))
    
    from qtpy import QtWidgets, QtGui

    splashScreen = None
    if splashscreen:
        class AcdcSPlashScreen(QtWidgets.QSplashScreen):
            def __init__(self):
                super().__init__()
                logo_path = os.path.join(resources_folderpath, 'logo.png')
                self.setPixmap(QtGui.QPixmap(logo_path))
            
            def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:
                pass
        
        # Launch splashscreen
        splashScreen = AcdcSPlashScreen()
        splashScreen.setWindowIcon(QIcon(icon_path))
        splashScreen.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint 
            | QtCore.Qt.SplashScreen 
            | QtCore.Qt.FramelessWindowHint
        )
        splashScreen.show()
        splashScreen.raise_()        
    
    from ._palettes import getPaletteColorScheme, setToolTipStyleSheet
    from ._palettes import get_color_scheme
    from . import load
    scheme = get_color_scheme()
    palette = getPaletteColorScheme(app.palette(), scheme=scheme)
    app.setPalette(palette)     
    load.rename_qrc_resources_file(scheme)
    setToolTipStyleSheet(app, scheme=scheme)
    
    return app, splashScreen