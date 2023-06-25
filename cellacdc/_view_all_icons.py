import sys
import os
import shutil

SCHEME = 'dark' 
FLAT = True

cellacdc_path = os.path.dirname(os.path.abspath(__file__))
qrc_resources_light_path = os.path.join(cellacdc_path, 'qrc_resources_light.py')
qrc_resources_dark_path = os.path.join(cellacdc_path, 'qrc_resources_dark.py')
qrc_resources_path = os.path.join(cellacdc_path, 'qrc_resources.py')
os.remove(qrc_resources_path)

if SCHEME == 'dark' and os.path.exists(qrc_resources_dark_path):
    shutil.copyfile(qrc_resources_dark_path, qrc_resources_path)
elif SCHEME == 'light' and os.path.exists(qrc_resources_light_path):
    shutil.copyfile(qrc_resources_light_path, qrc_resources_path)

from qtpy.QtGui import QIcon
from qtpy.QtCore import Qt, QSize
from qtpy.QtWidgets import (
    QApplication, QPushButton, QStyleFactory, QWidget, QGridLayout, 
    QCheckBox
)

from cellacdc.load import get_all_svg_icons_aliases
from cellacdc._palettes import getPaletteColorScheme, setToolTipStyleSheet

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

win.showMaximized()
app.exec_()