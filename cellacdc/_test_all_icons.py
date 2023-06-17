import sys
import os

from qtpy.QtGui import QIcon
from qtpy.QtCore import Qt, QSize
from qtpy.QtWidgets import (
    QApplication, QPushButton, QStyleFactory, QWidget, QGridLayout
)

from cellacdc.load import get_all_svg_icons_aliases
from cellacdc._palettes import getPaletteColorScheme, setToolTipStyleSheet
import qrc_resources

SCHEME = 'dark' # 'light'

svg_aliases = get_all_svg_icons_aliases(sort=True)

# Distribute icons over a 16:9 grid
nicons = len(svg_aliases)
ncols = round((nicons / 16*9)**(1/2))
nrows = nicons // ncols
left_nicons =  nicons % ncols
if left_nicons > 0:
    nrows += 1

app = QApplication(sys.argv)
app.setStyle(QStyleFactory.create('Fusion'))
app.setPalette(app.style().standardPalette())
palette = getPaletteColorScheme(app.palette(), scheme=SCHEME)
app.setPalette(palette)     
setToolTipStyleSheet(app, scheme=SCHEME)

if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

win = QWidget()
layout = QGridLayout()
win.setLayout(layout)

idx = 0
for i in range(nrows):
    for j in range(ncols):
        if idx == nicons:
            break
        alias = svg_aliases[idx]
        icon = QIcon(f':{alias}')
        button = QPushButton(alias)
        button.setIcon(icon)
        button.setIconSize(QSize(32,32))
        layout.addWidget(button, i, j)
        idx += 1

win.showMaximized()
app.exec_()