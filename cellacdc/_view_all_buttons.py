import sys

SCHEME = 'dark' 
FLAT = False

from qtpy.QtGui import QIcon
from qtpy.QtCore import Qt, QSize
from qtpy.QtWidgets import (
    QApplication, QPushButton, QStyleFactory, QWidget, QGridLayout, 
    QCheckBox
)

from cellacdc import widgets, _run

app, splashScreen = _run._setup_app(splashscreen=True, scheme=SCHEME)

from cellacdc.load import get_all_buttons_names
from cellacdc._palettes import getPaletteColorScheme, setToolTipStyleSheet

buttons_names = get_all_buttons_names(sort=True)

# Distribute icons over a 16:9 grid
nicons = len(buttons_names)
ncols = round((nicons / 16*9)**(1/2))
nrows = nicons // ncols
left_nicons =  nicons % ncols
if left_nicons > 0:
    nrows += 1

win = QWidget()
layout = QGridLayout()
win.setLayout(layout)

idx = 0
buttons = []
for i in range(nrows):
    for j in range(ncols):
        if idx == nicons:
            break
        button_name = buttons_names[idx]
        button_class = getattr(widgets, button_name)
        button = button_class(button_name)
        button.setToolTip(button_name)
        # button.setIconSize(QSize(32,32))
        if not button.isFlat():
            button.setCheckable(True)
        if FLAT:
            button.setFlat(True)
        layout.addWidget(button, i, j)
        buttons.append(button)
        idx += 1

max_height = max([button.sizeHint().height() for button in buttons])
for button in buttons:
    button.setMinimumHeight(max_height*2)

def setDisabled(checked):
    for button in buttons:
        button.setDisabled(checked)
        
checkbox = QCheckBox('Disable buttons')
checkbox.toggled.connect(setDisabled)
layout.addWidget(checkbox, i, j+1)

layout.setRowStretch(i+1, 1)
layout.setColumnStretch(j+2, 1)

splashScreen.close()
win.show()
app.exec_()