import os
import pandas as pd
import re

from qtpy import QtGui, QtWidgets, QtCore

from cellacdc import settings_csv_path

def base_color():
    scheme = get_color_scheme()
    if scheme == 'light':
        return '#4d4d4d'
    else:
        return '#d9d9d9'

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
        'Highlight': (108, 209, 77, 255),
        'HighlightedText': (255, 255, 255, 255)
    }
    return colors

def _light_disabled_colors():
    disabled_colors = {
        'ButtonText': (150, 150, 150, 255), 
        'WindowText': (128, 128, 128, 255), 
        'Text': (150, 150, 150, 255), 
        'Light': (255, 255, 255, 255),
        'Button': (230, 230, 230, 255),
        # 'Window': (200, 200, 200, 255),
        # 'Highlight': (0, 0, 0, 255),
        # 'HighlightedText': (0, 0, 0, 255),
        
    }
    return disabled_colors

def _dark_disabled_colors():
    disabled_colors = {
        'ButtonText': (150, 150, 150, 255), 
        'WindowText': (128, 128, 128, 255), 
        'Text': (128, 128, 128, 255), 
        'Light': (53, 53, 53, 255),
        'Button': (70, 70, 70, 255),
        # 'Window': (0, 0, 0, 255),
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
        'Highlight': (49, 97, 35, 255),
        'HighlightedText': (240, 240, 240, 255)
    }
    return colors

def getPainterColor():
    scheme = get_color_scheme()
    if scheme == 'light':
        return _light_colors()['Text']
    else:
        return _dark_colors()['Text']

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

def green():
    scheme = get_color_scheme()
    if scheme == 'light':
        return '#CFEB9B'
    else:
        return '#607a2f'

def TreeWidgetStyleSheet():
    scheme = get_color_scheme()
    if scheme == 'light':
        styleSheet = ("""
            QTreeWidget::item:hover {background-color:#E6E6E6;}
            QTreeWidget::item:hover {color:black;}
            QTreeWidget::item:selected {background-color:#CFEB9B;}
            QTreeWidget::item:selected {color:black;}
            QTreeView {
                selection-background-color: #CFEB9B;
                show-decoration-selected: 1;
            }
        """)
    else:
        styleSheet = ("""
            QTreeWidget::item:hover {background-color:#4d4d4d;}
            QTreeWidget::item:hover {color:white;}
            QTreeWidget::item:selected {background-color:#8dc427;}
            QTreeWidget::item:selected {color:white;}
            QTreeView {
                selection-background-color: #8dc427;
                show-decoration-selected: 1;
            }
        """)
    return styleSheet

def ListWidgetStyleSheet():
    styleSheet = TreeWidgetStyleSheet()
    styleSheet = styleSheet.replace('QTreeWidget', 'QListWidget')
    styleSheet = styleSheet.replace('QTreeView', 'QListView')
    return styleSheet

def QProgressBarColor():
    styleSheet = TreeWidgetStyleSheet()
    hex = re.findall(r'selection-background-color: (#[A-Za-z0-9]+)', styleSheet)[0]
    return QtGui.QColor(hex)    

def QProgressBarHighlightedTextColor():
    scheme = get_color_scheme()
    if scheme == 'light':
        return QtGui.QColor(0, 0, 0, 255)
    else:
        return QtGui.QColor(255, 255, 255, 255)

def moduleLaunchedButtonRgb(self):
    scheme = get_color_scheme()
    if scheme == 'light':
        return (241,221,0)
    else:
        return (241,221,0)