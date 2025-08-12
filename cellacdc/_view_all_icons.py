import sys
import os
import shutil

SCHEME = 'dark' 
FLAT = True

from qtpy.QtGui import QIcon
from qtpy.QtCore import Qt, QSize
from qtpy.QtWidgets import (
    QApplication, QPushButton, QStyleFactory, QWidget, QGridLayout, 
    QCheckBox
)

from cellacdc import _run
from cellacdc.load import get_all_svg_icons_aliases
from cellacdc._palettes import getPaletteColorScheme, setToolTipStyleSheet

app, splashScreen = _run._setup_app(splashscreen=True, scheme=SCHEME)

svg_aliases = get_all_svg_icons_aliases(sort=True)

# Distribute icons over a 16:9 grid
nicons = len(svg_aliases)
ncols = round((nicons / 16*9)**(1/2))
nrows = nicons // ncols
left_nicons =  nicons % ncols
if left_nicons > 0:
    nrows += 1

if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

win = QWidget()
layout = QGridLayout()
win.setLayout(layout)

idx = 0
buttons = []
for i in range(nrows):
    for j in range(ncols):
        if idx == nicons:
            break
        alias = svg_aliases[idx]
        icon = QIcon(f':{alias}')
        button = QPushButton(alias)
        button.setIcon(icon)
        button.setIconSize(QSize(32,32))
        button.setCheckable(True)
        if FLAT:
            button.setFlat(True)
        layout.addWidget(button, i, j)
        buttons.append(button)
        idx += 1

def setDisabled(checked):
    for button in buttons:
        button.setDisabled(checked)
        
checkbox = QCheckBox('Disable buttons')
checkbox.toggled.connect(setDisabled)
layout.addWidget(checkbox, i, j+1)

splashScreen.close()
win.showMaximized()
app.exec_()