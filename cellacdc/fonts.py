from . import GUI_INSTALLED

if GUI_INSTALLED:
    from qtpy.QtGui import QFont

    font = QFont()
    font.setPixelSize(12)
    italicFont = QFont()
    italicFont.setPixelSize(12)
    italicFont.setItalic(True)
    
else:
    font = None
    italicFont = None