from qtpy import QtGui, QtWidgets, QtCore

print(f'Using Qt version {QtCore.__version__}')

from pprint import pprint

app = QtWidgets.QApplication([])
app.setStyle(QtWidgets.QStyleFactory.create('Fusion'))
app.setPalette(app.style().standardPalette())

roles = (
    'Window', 'WindowText', 'Base', 'AlternateBase', 'ToolTipBase', 
    'ToolTipText', 'Text', 'Button', 'ButtonText', 'BrightText', 
    'Link', 'Highlight', 'HighlightedText'
)

colors = {}
palette = app.palette()
for role in roles:
    colorRole = getattr(QtGui.QPalette, role)
    rgba = app.palette().color(colorRole).getRgb()
    colors[role] = rgba

pprint(colors, sort_dicts=False)