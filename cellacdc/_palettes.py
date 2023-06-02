from qtpy import QtGui, QtWidgets, QtCore

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
    else:
        colors = _dark_colors()
    for role, rgba in colors.items():
        colorRole = getattr(QtGui.QPalette, role)
        palette.setColor(colorRole, QtGui.QColor(*rgba))
    return palette
        
    