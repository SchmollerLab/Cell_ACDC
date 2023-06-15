import os
import pandas as pd

from qtpy import QtGui, QtWidgets, QtCore

from cellacdc import settings_csv_path

def _light_colors():
    colors = {
        'Window': (239, 239, 239, 255),
        'WindowText': (0, 0, 0, 255),
        'Base': (255, 255, 255, 255),
        'AlternateBase': (247, 247, 247, 255),
        'ToolTipBase': (255, 255, 220, 255),
        'ToolTipText': (0, 0, 0, 255),
        'Text': (0, 0, 0, 255),
        'Button': (239, 239, 239, 255),
        'ButtonText': (0, 0, 0, 255),
        'BrightText': (255, 255, 255, 255),
        'Link': (0, 0, 255, 255),
        'Highlight': (48, 140, 198, 255),
        'HighlightedText': (255, 255, 255, 255)
    }
    return colors

def _light_disabled_colors():
    disabled_colors = {
        'ButtonText': (128, 128, 128, 255), 
        'WindowText': (128, 128, 128, 255), 
        'Text': (128, 128, 128, 255), 
        'Light': (53, 53, 53, 255)
    }
    return disabled_colors

def _dark_disabled_colors():
    disabled_colors = {
        'ButtonText': (128, 128, 128, 255), 
        'WindowText': (128, 128, 128, 255), 
        'Text': (128, 128, 128, 255), 
        'Light': (53, 53, 53, 255)
    }
    return disabled_colors

def _dark_colors():
    colors = {
        'Window': (50, 50, 50, 255),
        'WindowText': (240, 240, 240, 255),
        'Base': (36, 36, 36, 255),
        'AlternateBase': (43, 43, 43, 255),
        'ToolTipBase': (255, 255, 220, 255),
        'ToolTipText': (0, 0, 0, 255),
        'Text': (240, 240, 240, 255),
        'Button': (50, 50, 50, 255),
        'ButtonText': (240, 240, 240, 255),
        'BrightText': (75, 75, 75, 255),
        'Link': (48, 140, 198, 255),
        'Highlight': (48, 140, 198, 255),
        'HighlightedText': (240, 240, 240, 255)
    }
    return colors

def getPaletteColorScheme(palette: QtGui.QPalette, scheme='light'):
    if scheme == 'light':
        colors = _light_colors()
        disabled_colors = _light_disabled_colors()
    else:
        colors = _dark_colors()
        disabled_colors = _dark_disabled_colors()
    for role, rgba in colors.items():
        colorRole = getattr(QtGui.QPalette, role)
        palette.setColor(colorRole, QtGui.QColor(*rgba))
    ColorGroup = QtGui.QPalette.Disabled
    for role, rgba in disabled_colors.items():
        colorRole = getattr(QtGui.QPalette, role)
        palette.setColor(ColorGroup, colorRole, QtGui.QColor(*rgba))
    return palette

def get_color_scheme():
    if not os.path.exists(settings_csv_path):
        return 'light'
    df_settings = pd.read_csv(settings_csv_path, index_col='setting')
    if 'colorScheme' not in df_settings.index:
        return 'light'
    else:
        return df_settings.at['colorScheme', 'value']
    
def lineedit_background_hex():
    scheme = get_color_scheme()
    if scheme == 'light':
        return r'{background:#ffffff;}'
    else:
        return r'{background:#242424;}'   

def lineedit_invalid_entry_stylesheet():
    return (
        # 'background: #FEF9C3;'
        'border-radius: 4px;'
        'border: 1.5px solid red;'
        'padding: 1px 0px 1px 0px'
    )

def setToolTipStyleSheet(app, scheme='light'):
    if scheme == 'dark':
        app.setStyleSheet(r"QToolTip {"
            "color: #e6e6e6; background-color: #3c3c3c; border: 1px solid white;"
        "}"
        )
    else:
        app.setStyleSheet(r"QToolTip {"
            "color: #141414; background-color: #ffffff; border: 1px solid black;"
        "}"
        )
        