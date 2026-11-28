"""Dialogs opened only from the GUI mixins in gui/mixins/, which the
measurements viewer also composes (from apps.py): changes here affect
both the main GUI and the viewer.

customAnnotationDialog (imports the measurements viewer's predictor
group box only locally, when the viewer passes predictorOptions),
AnnotateObjTrackSettingsDialog, FindIDDialog, ViewCcaTableWindow (with
DataFrameModel and pdDataFrameWidget), ShortcutEditorDialog,
ScaleBarPropertiesDialog, TimestampPropertiesDialog, ObjectCountDialog
and SetCustomLevelsLut. Never import gui/main_window/, gui/dialogs/,
gui/widgets/, gui/workers/ or gui/helpers.py from here.
"""
import re
from typing import Literal, List
import numpy as np
from natsort import natsorted
from functools import partial
import pandas as pd
import html
from qtpy import QtCore
from qtpy.QtGui import QIcon, QKeySequence, QTextDocument
from qtpy.QtCore import Qt, QSize, Signal, QEventLoop, QTimer
from qtpy.QtWidgets import (
    QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout, QPushButton,
    QHBoxLayout, QDialog, QAbstractItemView, QCheckBox, QComboBox, QGridLayout,
    QTableView, QGroupBox, QMessageBox, QPlainTextEdit, QSpacerItem
)
from cellacdc import _window_stacking
from cellacdc import load, measurements, html_utils
from cellacdc import printl
from cellacdc import qutils
from cellacdc.shared.widgets import (
    basic_widgets as widgets_basic_widgets,
    combo_boxes as widgets_combo_boxes,
    item_views as widgets_item_views,
    line_edits as widgets_line_edits,
    message_box as widgets_message_box,
    numeric_inputs as widgets_numeric_inputs,
    push_buttons as widgets_push_buttons,
)
from . import widgets as gui_mixin_widgets
from cellacdc import _core
from cellacdc import fonts
from cellacdc import QtScoped
from cellacdc.shared.dialogs import generic as dialogs_generic
from cellacdc.shared.dialogs import base as dialogs_base
from cellacdc.shared.widgets import style as widgets_style


class customAnnotationDialog(QDialog):
    sigDeleteSelecAnnot = Signal(object)

    def __init__(
            self, savedCustomAnnot, parent=None, state=None, 
            predictorOptions=None, reservedShortcuts=None, existingNames=(),
            allowDeleteSaved=True, columnConflict=None
        ):
        """Dialog to set the parameters of a custom annotation.

        Parameters
        ----------
        savedCustomAnnot : dict
            Previously saved annotations {name: state} that can be loaded.
        parent : QWidget, optional
        state : dict, optional
            State of the annotation to modify.
        predictorOptions : dict, optional
            If not None, add the predictor settings (measurements viewer).
            See `viewer_dialogs_predictor_editors._CustomAnnotationPredictorGroupbox`.
            The viewer also gets the "Overrides phases" option (state key
            'overridesPhases', always written as a bool).
        reservedShortcuts : dict or callable, optional
            {key sequence text: used by} of shortcuts that cannot be used,
            or a callable returning it. A callable is called on OK, so that
            keys taken while the dialog is open (the window stays usable)
            are refused too.
        existingNames : iterable of str, optional
            Names of other annotations that cannot be used.
        allowDeleteSaved : bool, optional
            If True (default), the dialog listing the saved annotations has
            a button to delete them (`sigDeleteSelecAnnot`). False when the
            host cannot persist the deletion (measurements viewer).
        columnConflict : callable, optional
            `columnConflict(name)` returns why the table column `name`
            cannot store the annotation ('' if it can), checked on OK so
            that the dialog stays open with the entered values (measurements
            viewer: `customAnnotColumnConflict`).
        """
        self.cancel = True
        self.loop = None
        self.clickedButton = None
        self.savedCustomAnnot = savedCustomAnnot
        self.allowDeleteSaved = allowDeleteSaved
        self._reservedShortcuts = reservedShortcuts
        self.existingNames = set(existingNames)
        self.columnConflict = columnConflict
        self.hasPredictor = predictorOptions is not None

        self.internalNames = measurements.get_all_acdc_df_colnames(
            include_custom=False
        )

        super().__init__(parent)

        self.setWindowTitle('Custom annotation')
        _window_stacking.set_window_flags_on_top(self, Qt.Window)

        layout = widgets_line_edits.FormLayout()

        row = 0
        nameInfoTxt = ("""
        <b>Name of the column</b> that will be saved in the <code>acdc_output.csv</code>
        file.<br><br>
        Valid charachters are letters and numbers separate by underscore
        or dash only.<br><br>
        Additionally, some names are <b>reserved</b> because they are used
        by Cell-ACDC for standard measurements.<br><br>
        Internally reserved names:
        """)
        self.nameInfoTxt = (f'{html_utils.paragraph(nameInfoTxt)}')
        self.nameWidget = widgets_line_edits.formWidget(
            widgets_line_edits.alphaNumericLineEdit(), addInfoButton=True,
            labelTextLeft='Name: ', parent=self, infoTxt=self.nameInfoTxt
        )
        self.nameWidget.infoButton.disconnect()
        self.nameWidget.infoButton.clicked.connect(self.showNameInfo)
        if state is not None:
            self.nameWidget.widget.setText(state['name'])
        self.nameWidget.widget.textChanged.connect(self.checkName)
        layout.addFormWidget(self.nameWidget, row=row)

        row += 1
        self.nameInfoLabel = QLabel()
        layout.addWidget(
            self.nameInfoLabel, row, 0, 1, 2, alignment=Qt.AlignCenter
        )

        row += 1
        spacing = QSpacerItem(10, 10)
        layout.addItem(spacing, row, 0)

        row += 1
        symbolInfoTxt = ("""
        <b>Symbol</b> that will be drawn on the annotated cell at
        the requested time frame.
        """)
        symbolInfoTxt = (f'{html_utils.paragraph(symbolInfoTxt)}')
        self.symbolWidget = widgets_line_edits.formWidget(
            widgets_combo_boxes.pgScatterSymbolsCombobox(), addInfoButton=True,
            labelTextLeft='Symbol: ', parent=self, infoTxt=symbolInfoTxt
        )
        if state is not None:
            self.symbolWidget.widget.setCurrentText(state['symbol'])
        layout.addFormWidget(self.symbolWidget, row=row)

        row += 1
        if self.hasPredictor:
            shortcutInfoTxt = ("""
            <b>Shortcut</b> that you can use to <b>annotate/un-annotate</b> 
            the displayed cell across its entire video history.<br><br>
            <b>Shift+Shortcut</b> re-evaluates the predictor on the displayed 
            cell.<br><br>Leave empty if you don't need a shortcut.
            """)
        else:
            shortcutInfoTxt = ("""
            <b>Shortcut</b> that you can use to <b>activate/deactivate</b> annotation
            of this event.<br><br> Leave empty if you don't need a shortcut.
            """)
        shortcutInfoTxt = (f'{html_utils.paragraph(shortcutInfoTxt)}')
        self.shortcutWidget = widgets_line_edits.formWidget(
            widgets_line_edits.ShortcutLineEdit(), addInfoButton=True,
            labelTextLeft='Shortcut: ', parent=self, infoTxt=shortcutInfoTxt
        )
        if state is not None:
            self.shortcutWidget.widget.setText(state['shortcut'])
        layout.addFormWidget(self.shortcutWidget, row=row)

        row += 1
        descInfoTxt = ("""
        <b>Description</b> will be used as the <b>tool tip</b> that will be
        displayed when you hover with th mouse cursor on the toolbar button
        specific for this annotation
        """)
        descInfoTxt = (f'{html_utils.paragraph(descInfoTxt)}')
        self.descWidget = widgets_line_edits.formWidget(
            QPlainTextEdit(), addInfoButton=True,
            labelTextLeft='Description: ', parent=self, infoTxt=descInfoTxt
        )
        if state is not None:
            self.descWidget.widget.setPlainText(state['description'])
        layout.addFormWidget(self.descWidget, row=row)

        row += 1
        optionsGroupBox = QGroupBox('Additional options')
        optionsLayout = QGridLayout()
        toggle = widgets_push_buttons.Toggle()
        toggle.setChecked(True)
        self.keepActiveToggle = toggle
        toggleLabel = QLabel('Keep tool active after using it: ')
        colorButtonLabel = QLabel('Symbol color: ')
        self.hideAnnotTooggle = widgets_push_buttons.Toggle()
        self.hideAnnotTooggle.setChecked(True)
        hideAnnotTooggleLabel = QLabel(
            'Hide annotation when button is not active: '
        )
        self.colorButton = widgets_push_buttons.myColorButton(color=(255, 0, 0))
        self.colorButton.clicked.disconnect()
        self.colorButton.clicked.connect(self.selectColor)

        optionsLayout.setColumnStretch(0, 1)
        optRow = 0
        optionsLayout.addWidget(toggleLabel, optRow, 1)
        optionsLayout.addWidget(toggle, optRow, 2)
        optRow += 1
        optionsLayout.addWidget(hideAnnotTooggleLabel, optRow, 1)
        optionsLayout.addWidget(self.hideAnnotTooggle, optRow, 2)
        optionsLayout.setColumnStretch(3, 1)
        optRow += 1
        optionsLayout.addWidget(colorButtonLabel, optRow, 1)
        optionsLayout.addWidget(self.colorButton, optRow, 2)

        # Measurements viewer only: the annotation replaces the phases
        # of the annotated cells (state key 'overridesPhases')
        self.overridesPhasesToggle = None
        if self.hasPredictor:
            overridesPhasesTooltip = (
                'Annotated cells get the annotation '
                'name as their phase on every frame,\n'
                'regardless of the phase translator '
                'rules. If a cell has several annotations\n'
                'with this option, the first one in the list (toolbar order) wins.'
            )
            self.overridesPhasesToggle = widgets_push_buttons.Toggle()
            self.overridesPhasesToggle.setToolTip(overridesPhasesTooltip)
            overridesPhasesLabel = QLabel('Overrides phases: ')
            overridesPhasesLabel.setToolTip(overridesPhasesTooltip)
            optRow += 1
            optionsLayout.addWidget(overridesPhasesLabel, optRow, 1)
            optionsLayout.addWidget(self.overridesPhasesToggle, optRow, 2)

        if state is not None:
            self.keepActiveToggle.setChecked(state.get('keepActive', True))
            self.hideAnnotTooggle.setChecked(state.get('isHideChecked', True))
            if state.get('symbolColor') is not None:
                self.colorButton.setColor(state['symbolColor'])
            if self.overridesPhasesToggle is not None:
                self.overridesPhasesToggle.setChecked(
                    bool(state.get('overridesPhases', False))
                )

        optionsGroupBox.setLayout(optionsLayout)
        layout.addWidget(optionsGroupBox, row, 1, alignment=Qt.AlignCenter)
        optionsInfoButton = QPushButton(self)
        optionsInfoButton.setCursor(Qt.WhatsThisCursor)
        optionsInfoButton.setIcon(QIcon(":info.svg"))
        optionsInfoButton.clicked.connect(self.showOptionsInfo)
        layout.addWidget(optionsInfoButton, row, 3, alignment=Qt.AlignRight)

        row += 1
        layout.addItem(QSpacerItem(5, 5), row, 0)

        self.predictorGroupbox = None
        if self.hasPredictor:
            row += 1
            predictorOptions = {
                'nameGetter': self.nameWidget.widget.text, **predictorOptions
            }
            # Measurements viewer only; imported here: gui/mixins/ imports
            # the viewer only lazily (see gui/mixins/__init__.py)
            from cellacdc.measurements_viewer.dialogs.predictor_editors import _CustomAnnotationPredictorGroupbox
            self.predictorGroupbox = _CustomAnnotationPredictorGroupbox(
                predictorOptions, 
                predictor=None if state is None else state.get('predictor'),
                autoApply=(
                    False if state is None else state.get('autoApply', False)
                ),
                parent=self
            )
            layout.addWidget(self.predictorGroupbox, row, 1, 1, 3)

        row += 1
        noteText = (
            '<i>NOTE: you can change these options later with<br>'
            '<b>RIGHT-click</b> on the associated left-side <b>toolbar button<b>.</i>'
        )
        noteLabel = QLabel(html_utils.paragraph(noteText, font_size='11px'))
        layout.addWidget(noteLabel, row, 1, 1, 3)

        buttonsLayout = QHBoxLayout()

        self.loadSavedAnnotButton = widgets_push_buttons.OpenFilePushButton(
            '  Load annotation...  '
        )
        if not savedCustomAnnot:
            self.loadSavedAnnotButton.setDisabled(True)
        self.okButton = widgets_push_buttons.okPushButton('  Ok  ')
        cancelButton = widgets_push_buttons.cancelPushButton('Cancel')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(self.loadSavedAnnotButton)
        buttonsLayout.addWidget(self.okButton)

        cancelButton.clicked.connect(self.cancelCallBack)
        self.cancelButton = cancelButton
        self.loadSavedAnnotButton.clicked.connect(self.loadSavedAnnot)
        self.okButton.clicked.connect(self.ok_cb)
        self.okButton.setFocus()

        mainLayout = QVBoxLayout()

        noteTxt = ("""
        Custom annotations will be <b>saved in the <code>acdc_output.csv</code></b><br>
        file as a column with the name you write in the field <b>Name</b><br>
        """)
        noteTxt = (f'{html_utils.paragraph(noteTxt, font_size="15px")}')
        noteLabel = QLabel(noteTxt)
        noteLabel.setAlignment(Qt.AlignCenter)
        mainLayout.addWidget(noteLabel)

        mainLayout.addLayout(layout)
        mainLayout.addStretch(1)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

    def checkName(self, text):
        if not text:
            txt = 'Name cannot be empty'
            self.nameInfoLabel.setText(
                html_utils.paragraph(
                    txt, font_size='11px', font_color='red'
                )
            )
            return
        for name in self.internalNames:
            if name.find(text) != -1:
                txt = (
                    f'"{text}" cannot be part of the name, '
                    'because <b>reserved<b>.'
                )
                self.nameInfoLabel.setText(
                    html_utils.paragraph(
                        txt, font_size='11px', font_color='red'
                    )
                )
                break
        else:
            self.nameInfoLabel.setText('')

    def loadSavedAnnot(self):
        items = list(self.savedCustomAnnot.keys())
        additionalButtons = (
            ('Delete selected annnotations', ) if self.allowDeleteSaved
            else ()
        )
        self.selectAnnotWin = widgets_item_views.QDialogListbox(
            'Load annotation parameters',
            'Select annotation to load:', items,
            additionalButtons=additionalButtons,
            parent=self, multiSelection=False
        )
        for button in getattr(self.selectAnnotWin, '_additionalButtons', ()):
            button.disconnect()
            button.clicked.connect(self.deleteSelectedAnnot)
        self.selectAnnotWin.exec_()
        if self.selectAnnotWin.cancel:
            return
        if self.selectAnnotWin.listBox.count() == 0:
            return
        if not self.selectAnnotWin.selectedItemsText:
            self.warnNoItemsSelected()
            return
        selectedName = self.selectAnnotWin.selectedItemsText[-1]
        selectedAnnot = self.savedCustomAnnot[selectedName]
        self.nameWidget.widget.setText(selectedAnnot['name'])
        self.symbolWidget.widget.setCurrentText(selectedAnnot['symbol'])
        self.shortcutWidget.widget.setText(selectedAnnot['shortcut'])
        self.descWidget.widget.setPlainText(selectedAnnot['description'])
        self.colorButton.setColor(selectedAnnot['symbolColor'])
        keySequence = widgets_line_edits.macShortcutToWindows(selectedAnnot['shortcut'])
        if keySequence:
            self.shortcutWidget.widget.keySequence = widgets_line_edits.KeySequenceFromText(keySequence)
        # Same options as restored from `state` in __init__
        self.keepActiveToggle.setChecked(selectedAnnot.get('keepActive', True))
        self.hideAnnotTooggle.setChecked(
            selectedAnnot.get('isHideChecked', True)
        )
        if self.predictorGroupbox is not None:
            self.predictorGroupbox.setPredictor(
                selectedAnnot.get('predictor'),
                selectedAnnot.get('autoApply', False)
            )
        if self.overridesPhasesToggle is not None:
            self.overridesPhasesToggle.setChecked(
                bool(selectedAnnot.get('overridesPhases', False))
            )

    def warnNoItemsSelected(self):
        msg = widgets_message_box.myMessageBox(parent=self)
        msg.setIcon(iconName='SP_MessageBoxWarning')
        msg.setWindowTitle('Delete annotation?')
        msg.addText('You didn\'t select any annotation!')
        msg.addButton('  Ok  ')
        msg.exec_()

    def deleteSelectedAnnot(self):
        msg = widgets_message_box.myMessageBox(parent=self)
        msg.setIcon(iconName='SP_MessageBoxWarning')
        msg.setWindowTitle('Delete annotation?')
        msg.addText('Are you sure you want to delete the selected annotations?')
        msg.addButton('Yes')
        cancelButton = msg.addButton(' Cancel ')
        msg.exec_()
        if msg.clickedButton == cancelButton:
            return
        for item in self.selectAnnotWin.listBox.selectedItems():
            name = item.text()
            self.savedCustomAnnot.pop(name)
        self.sigDeleteSelecAnnot.emit(self.selectAnnotWin.listBox.selectedItems())
        items = list(self.savedCustomAnnot.keys())
        self.selectAnnotWin.listBox.clear()
        self.selectAnnotWin.listBox.addItems(items)

    def selectColor(self):
        color = self.colorButton.color()
        self.colorButton.origColor = color
        self.colorButton.colorDialog.setCurrentColor(color)
        _window_stacking.set_window_flags_on_top(
            self.colorButton.colorDialog, Qt.Window, parent=self
        )
        self.colorButton.colorDialog.open()
        w = self.width()
        left = self.pos().x()
        colorDialogTop = self.colorButton.colorDialog.pos().y()
        self.colorButton.colorDialog.move(w+left+10, colorDialogTop)

    def showOptionsInfo(self):
        info = ("""
        <b>Keep tool active after using it</b>: Choose whether the tool
        should stay active or not after annotating.<br><br>
        <b>Hide annotation when button is not active</b>: Choose whether
        annotation on the cell/object should be visible only if the
        button is active or also when it is not active.<br>
        <i>NOTE: annotations are <b>always stored</b> no matter whether
        they are visible or not.</i><br><br>
        <b>Symbol color</b>: Choose color of the symbol that will be used
        to label annotated cell/object.
        """)
        if self.overridesPhasesToggle is not None:
            info = (f"""{info}<br><br>
            <b>Overrides phases</b>: Annotated cells get the annotation
            name as their <b>phase on every frame</b>, regardless of the
            time point to phase translator rules (which apply only to the
            other cells). If a cell has several annotations with this
            option, the first one in the list (toolbar order) wins.
            """)
        info = (f'{html_utils.paragraph(info)}')
        msg = widgets_message_box.myMessageBox()
        msg.setIcon()
        msg.setWindowTitle(f'Additional options info')
        msg.addText(info)
        msg.addButton('   Ok   ')
        msg.exec_()

    def ok_cb(self, checked=True):
        self.cancel = False
        self.clickedButton = self.okButton
        self.close()

    def cancelCallBack(self, checked=True):
        self.cancel = True
        self.clickedButton = self.cancelButton
        self.close()

    def showNameInfo(self):
        msg = widgets_message_box.myMessageBox()
        listView = widgets_item_views.readOnlyQList(msg)
        listView.addItems(self.internalNames)
        # listView.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        msg.information(
            self, 'Annotation Name info', self.nameInfoTxt,
            widgets=listView
        )

    def reservedShortcuts(self):
        """Current {key sequence text: used by} (see `__init__`)."""
        reserved = self._reservedShortcuts
        return dict((reserved() if callable(reserved) else reserved) or {})

    def closeEvent(self, event):
        if self.clickedButton is None or self.clickedButton==self.cancelButton:
            # cancel button or closed with 'x' button
            self.cancel = True
            # Return from `exec_` so that the caller sees the cancel
            if self.loop is not None:
                self.loop.exit()
            return

        if self.clickedButton==self.okButton and not self.nameWidget.widget.text():
            msg = QMessageBox()
            msg.critical(
                self, 'Empty name', 'The name cannot be empty!', msg.Ok
            )
            event.ignore()
            self.cancel = True
            return

        if self.clickedButton==self.okButton and self.nameInfoLabel.text():
            msg = widgets_message_box.myMessageBox()
            listView = widgets_item_views.listWidget(msg)
            listView.addItems(self.internalNames)
            listView.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
            name = self.nameWidget.widget.text()
            txt = (
                f'"{name}" cannot be part of the name, '
                'because it is <b>reserved</b> for standard measurements '
                'saved by Cell-ACDC.<br><br>'
                'Internally reserved names:'
            )
            msg.critical(
                self, 'Not a valid name', html_utils.paragraph(txt),
                widgets=listView
            )
            event.ignore()
            self.cancel = True
            return

        name = self.nameWidget.widget.text()
        if self.clickedButton==self.okButton and name in self.existingNames:
            msg = widgets_message_box.myMessageBox(wrapText=False)
            msg.warning(
                self, 'Name already used', html_utils.paragraph(
                    f'A custom annotation called "{name}" already exists.'
                )
            )
            event.ignore()
            self.cancel = True
            return

        conflict = (
            self.columnConflict(name) if self.columnConflict is not None
            else ''
        )
        if self.clickedButton==self.okButton and conflict:
            msg = widgets_message_box.myMessageBox()
            msg.warning(
                self, 'Name cannot be used', html_utils.paragraph(
                    f'<code>{html.escape(name)}</code> cannot be used as '
                    f'annotation column: {html.escape(conflict)}<br><br>'
                    'Choose a different name.'
                )
            )
            event.ignore()
            self.cancel = True
            return

        shortcut = self.shortcutWidget.widget.text().strip()
        usedBy = None
        if self.clickedButton==self.okButton and shortcut:
            usedBy = self.reservedShortcuts().get(
                QKeySequence(shortcut).toString()
            )
        if usedBy is not None:
            msg = widgets_message_box.myMessageBox(wrapText=False)
            msg.warning(
                self, 'Shortcut already used', html_utils.paragraph(
                    f'The shortcut "{html.escape(shortcut)}" is already used '
                    f'by "{html.escape(usedBy)}". Choose a different shortcut.'
                )
            )
            event.ignore()
            self.cancel = True
            return
        
        predictor = None
        if self.predictorGroupbox is not None:
            try:
                predictor = self.predictorGroupbox.validatePredictor()
            except Exception as err:
                msg = widgets_message_box.myMessageBox(wrapText=False)
                msg.warning(
                    self, 'Invalid predictor', 
                    html_utils.paragraph(html.escape(str(err)))
                )
                event.ignore()
                self.cancel = True
                return
        
        if self.hasPredictor:
            usage = (
                'select a cell and press the shortcut '
                '(or activate the button and RIGHT-CLICK\n'
                'on the cell) to annotate it at the all frames of its video history'
            )
        else:
            usage = 'activate the button and RIGHT-CLICK on cell to annotate'
        self.toolTip = qutils.wrap_tooltip(
            f'Name: {self.nameWidget.widget.text()}\n\n'
            f'Usage: {usage}\n\n'
            f'Description: {self.descWidget.widget.toPlainText()}\n\n'
            f'SHORTCUT: "{self.shortcutWidget.widget.text()}"'
        )

        symbol = self.symbolWidget.widget.currentText()
        self.symbol = re.findall(r"\'(.+)\'", symbol)[0]

        self.state = {
            'type': 'Single time-point',
            'name': self.nameWidget.widget.text(),
            'symbol':  self.symbolWidget.widget.currentText(),
            'shortcut': self.shortcutWidget.widget.text(),
            'description': self.descWidget.widget.toPlainText(),
            'keepActive': self.keepActiveToggle.isChecked(),
            'isHideChecked': self.hideAnnotTooggle.isChecked(),
            'symbolColor': self.colorButton.color()
        }
        if self.predictorGroupbox is not None:
            self.state['predictor'] = predictor
            self.state['autoApply'] = (
                self.predictorGroupbox.autoApplyCheckBox.isChecked()
            )
        if self.overridesPhasesToggle is not None:
            # Always written: modifying an annotation replaces its state
            self.state['overridesPhases'] = bool(
                self.overridesPhasesToggle.isChecked()
            )

        if self.loop is not None:
            self.loop.exit()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        super().show()
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()


class FindIDDialog(dialogs_generic.QLineEditDialog):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.okButton.setIcon(QIcon(':magnGlass.svg'))
        self.okButton.setText(' Find ')


class DataFrameModel(QtCore.QAbstractTableModel):
    # https://stackoverflow.com/questions/44603119/how-to-display-a-pandas-data-frame-with-pyqt5-pyside2
    DtypeRole = QtCore.Qt.UserRole + 1000
    ValueRole = QtCore.Qt.UserRole + 1001

    def __init__(self, df=pd.DataFrame(), parent=None):
        super(DataFrameModel, self).__init__(parent)
        self._dataframe = df

    def setDataFrame(self, dataframe):
        self.beginResetModel()
        self._dataframe = dataframe.copy()
        self.endResetModel()

    def dataFrame(self):
        return self._dataframe

    dataFrame = QtCore.Property(pd.DataFrame, fget=dataFrame,
                                    fset=setDataFrame)

    @QtCore.Slot(int, QtCore.Qt.Orientation, result=str)
    def headerData(self, section: int,
                   orientation: QtCore.Qt.Orientation,
                   role: int = QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self._dataframe.columns[section]
            else:
                return str(self._dataframe.index[section])
        return QtCore.QVariant()

    def rowCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._dataframe.index)

    def columnCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return self._dataframe.columns.size

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < self.rowCount() \
            and 0 <= index.column() < self.columnCount()):
            return QtCore.QVariant()
        row = self._dataframe.index[index.row()]
        col = self._dataframe.columns[index.column()]
        dt = self._dataframe[col].dtype

        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter

        val = self._dataframe.iloc[row][col]
        if role == QtCore.Qt.DisplayRole:
            return str(val)
        elif role == DataFrameModel.ValueRole:
            return val
        if role == DataFrameModel.DtypeRole:
            return dt
        return QtCore.QVariant()

    def roleNames(self):
        roles = {
            QtCore.Qt.DisplayRole: b'display',
            DataFrameModel.DtypeRole: b'dtype',
            DataFrameModel.ValueRole: b'value'
        }
        return roles

class pdDataFrameWidget(QMainWindow):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle('Cell cycle annotations')

        _window_stacking.set_window_flags_on_top(self, Qt.Window)

        mainContainer = QWidget()
        self.setCentralWidget(mainContainer)

        layout = QVBoxLayout()
        self._layout = layout

        self.tableView = QTableView(self)
        layout.addWidget(self.tableView)
        model = DataFrameModel(df)
        self.tableView.setModel(model)
        for i in range(len(df.columns)):
            self.tableView.resizeColumnToContents(i)
        # layout.addWidget(QPushButton('Ok', self))
        mainContainer.setLayout(layout)

    def updateTable(self, df, IDs=None):
        if df is None:
            df = self.parent.getBaseCca_df()
        
        if IDs is not None:
            df = df.loc[IDs]
            
        df = df.reset_index()
        model = DataFrameModel(df)
        self.tableView.setModel(model)
        for i in range(len(df.columns)):
            self.tableView.resizeColumnToContents(i)

    def setGeometryWindow(self, maxWidth=1024):
        width = self.tableView.verticalHeader().width() + 4
        for j in range(self.tableView.model().columnCount()):
            width += self.tableView.columnWidth(j) + 4
        height = self.tableView.horizontalHeader().height() + 4
        h = height + (self.tableView.rowHeight(0) + 4)*10
        w = width if width<maxWidth else maxWidth
        self.setGeometry(100, 100, w, h)

        # Center window
        parent = self.parent
        if parent is not None:
            # Center the window on main window
            mainWinGeometry = parent.geometry()
            mainWinLeft = mainWinGeometry.left()
            mainWinTop = mainWinGeometry.top()
            mainWinWidth = mainWinGeometry.width()
            mainWinHeight = mainWinGeometry.height()
            mainWinCenterX = int(mainWinLeft + mainWinWidth/2)
            mainWinCenterY = int(mainWinTop + mainWinHeight/2)
            winGeometry = self.geometry()
            winWidth = winGeometry.width()
            winHeight = winGeometry.height()
            winLeft = int(mainWinCenterX - winWidth/2)
            winRight = int(mainWinCenterY - winHeight/2)
            self.move(winLeft, winRight)

    def closeEvent(self, event):
        self.parent.ccaTableWin = None


class ShortcutEditorDialog(dialogs_base.QBaseDialog):
    def __init__(
            self, 
            widgetsWithShortcut: dict, 
            delObjectKey='',
            delObjectButton: Literal['Middle click', 'Left click']='Middle click',
            zoomOutKeyValue: int=None,
            parent=None,
            mouseBindings: dict=None,
            hard_shortcuts: dict=None,
            highlighted_shortcut: str=None,
            show_delete_object_shortcut: bool=True
        ):
        # When show_delete_object_shortcut is False the "Delete object"
        # row is hidden and `delObjAction` stays None after Ok (callers
        # keep their own delete-object action)
        self.show_delete_object_shortcut = show_delete_object_shortcut
        self.delObjAction = None
        self.cancel = True
        super().__init__(parent)

        if hard_shortcuts is None:
            hard_shortcuts = {}
            
        self.new_hard_shortcuts = {}
        for shortcut, name in hard_shortcuts.items():
            shortcut = widgets_line_edits.KeySequenceFromText(shortcut)
            shortcut_str = shortcut.toString()
            self.new_hard_shortcuts[name] = shortcut_str

        self.setWindowTitle('Customize keyboard shortcuts')

        mainLayout = QVBoxLayout()

        self.customShortcuts = {}
        self.shortcutLineEdits = {}

        scrollArea = widgets_basic_widgets.ScrollArea(self)
        scrollArea.setWidgetResizable(True)
        self.scrollArea = scrollArea
        scrollAreaWidget = QWidget()
        entriesLayout = QGridLayout()
        
        conflict_prefix = 'Conflict with '
        all_names = (
            list(widgetsWithShortcut.keys()) 
            + list(self.new_hard_shortcuts.keys())
        )
        longest_name = max(all_names, key=len)
        self.conflict_text_formatter = (
            lambda name: (
                '<font color="red">'
                f'{conflict_prefix}'
                f'{name if not isinstance(name, tuple) else name[0]}'
                '</font>'
        ))
        longest_conflict_text = self.conflict_text_formatter(longest_name)

        doc = QTextDocument()
        warnConflictLabel = QLabel(longest_conflict_text)
        doc.setDefaultFont(warnConflictLabel.font())
        doc.setHtml(longest_conflict_text)
        max_warn_width = int(doc.idealWidth() + 10)
        
        row = 0
        
        if self.show_delete_object_shortcut:
            name = 'Delete object'
            button = widgets_push_buttons.PushButton(self, flat=True)
            button.setIcon(QIcon(":del_obj_click.svg"))
            self.delObjShortcutLineEdit = widgets_line_edits.ShortcutLineEdit(
                allowModifiers=True, notAllowedModifier=Qt.AltModifier,
                allowMouseButtons=True
            )
            if delObjectKey is not None:
                self.delObjShortcutLineEdit.setText(delObjectKey)
            self.delObjButtonCombobox = QComboBox()
            self.delObjButtonCombobox.addItems(['Middle click', 'Left click'])
            self.delObjButtonCombobox.setCurrentText(delObjectButton)
            entriesLayout.addWidget(button, row, 0)
            entriesLayout.addWidget(QLabel('Delete object:'), row, 1)
            entriesLayout.addWidget(self.delObjShortcutLineEdit, row, 2)
            entriesLayout.addWidget(
                self.delObjButtonCombobox, row, 3, alignment=Qt.AlignLeft
            )
            warnConflictLabel = QLabel('')
            self.delObjShortcutLineEdit.warnConflictLabel = warnConflictLabel
            entriesLayout.addWidget(warnConflictLabel, row, 4)
            warnConflictLabel.setMinimumWidth(max_warn_width)
            self.delObjShortcutLineEdit.name = name
            self.shortcutLineEdits[name] = self.delObjShortcutLineEdit
            self.delObjShortcutLineEdit.textChanged.connect(self.shortcutChanged)
            self.delObjShortcutLineEdit.clicked.connect(self.setShortcutLineEditEventFilter)
            self.delObjShortcutLineEdit.editingFinished.connect(self.releaseShortcutLineEditEventFilter) 
            row += 1

        name = 'Zoom out'
        button = widgets_push_buttons.PushButton(self, flat=True)
        label = QLabel('Zoom out:')
        self.zoomShortcutLineEdit = widgets_line_edits.ShortcutLineEdit(allowMouseButtons=True)
        if zoomOutKeyValue is not None:
            zoomOutKeySequence = widgets_line_edits.KeySequenceFromText(zoomOutKeyValue)
            self.zoomShortcutLineEdit.setText(zoomOutKeySequence.toString())
            self.zoomShortcutLineEdit.key = zoomOutKeyValue
        self.zoomShortcutLineEdit.textChanged.connect(self.shortcutChanged)
        self.zoomShortcutLineEdit.clicked.connect(
            self.setShortcutLineEditEventFilter)
        self.zoomShortcutLineEdit.editingFinished.connect(
            self.releaseShortcutLineEditEventFilter)
        entriesLayout.addWidget(button, row, 0)
        entriesLayout.addWidget(label, row, 1)
        entriesLayout.addWidget(self.zoomShortcutLineEdit, row, 2)
        self.shortcutLineEdits[name] = self.zoomShortcutLineEdit
        self.zoomShortcutLineEdit.name = name
        warnConflictLabel = QLabel('')
        self.zoomShortcutLineEdit.warnConflictLabel = warnConflictLabel
        entriesLayout.addWidget(warnConflictLabel, row, 4)
        warnConflictLabel.setMinimumWidth(max_warn_width)
        
        row += 1
        
        # sort dicitonaries
        keep_at_beginning = ['Next', 'Previous']
        if highlighted_shortcut is not None:
            keep_at_beginning.insert(0, highlighted_shortcut)

        # used for organizing the shortcuts into groups based on their names
        grouped_keys = {
            'keep_at_beginning': [],
            'other': [],
            '(lineage tree)': [],
            '(cell cycle analysis)': [],
        }

        widgetsWithShortcut = self._groupAndSortShortcuts(
            widgetsWithShortcut, keep_at_beginning, grouped_keys
        )
        
        # based on name, only check uniquness within these groups
        # and the forbidden ones.
        exclusivity_groups = {
            '(lineage tree)': [],
            '(cell cycle analysis)': [],
            'other': [],
        }
        
        exclusivity_groups['other'] = grouped_keys['other'] + grouped_keys['keep_at_beginning']
        for key, value in grouped_keys.items():
            if key not in ['keep_at_beginning', 'other']:
                exclusivity_groups[key] = value
                
        self.exclusivity_groups_reverse = {
            name: group for group, names in exclusivity_groups.items() for name in names
        }

        grouped_keys = {
            'keep_at_beginning': [],
            'other': [],
        }
        self.new_hard_shortcuts = self._groupAndSortShortcuts(
            self.new_hard_shortcuts, keep_at_beginning, grouped_keys
        )
            
        for group, shortcuts in widgetsWithShortcut.items():
            if not shortcuts:
                continue
            title = group.strip('()').replace('_', ' ').capitalize()
            entriesLayout.addWidget(QLabel(f'<b>{title}</b>'), row, 0, 1, 5)
            row += 1
            for name, widget in shortcuts.items():
                button = widgets_push_buttons.PushButton(self, flat=True)
                try:
                    button.setIcon(widget.icon())
                except:
                    pass
                label = QLabel(f'{name}:')
                shortcutLineEdit = widgets_line_edits.ShortcutLineEdit(
                    allowMouseButtons=True
                )
                if mouseBindings is not None and name in mouseBindings:
                    mouse_button = mouseBindings[name]
                    btn_name = QtScoped.mouse_button_name(mouse_button)
                    shortcutLineEdit.setText(f'Mouse {btn_name}')
                    isShortcutKeyPress = False
                    isShortcutMouseButton = True
                elif hasattr(widget, 'keyPressShortcut'):
                    shortcutLineEdit.key = widget.keyPressShortcut
                    shortcut = widgets_line_edits.KeySequenceFromText(
                        widget.keyPressShortcut
                    )
                    isShortcutKeyPress = True
                    isShortcutMouseButton = False
                else:
                    shortcut = widget.shortcut()
                    isShortcutKeyPress = False
                    isShortcutMouseButton = False

                if isShortcutMouseButton:
                    mouse_button = mouseBindings[name]
                    btn_name = QtScoped.mouse_button_name(mouse_button)
                    shortcutLineEdit.setText(f'Mouse {btn_name}')
                else:
                    shortcutLineEdit.setText(shortcut.toString())

                shortcutLineEdit.textChanged.connect(self.shortcutChanged)
                shortcutLineEdit.isShortcutKeyPress = isShortcutKeyPress
                shortcutLineEdit.isShortcutMouseButton = isShortcutMouseButton
                shortcutLineEdit.clicked.connect(
                    self.setShortcutLineEditEventFilter
                )
                shortcutLineEdit.editingFinished.connect(
                    self.releaseShortcutLineEditEventFilter
                )

                entriesLayout.addWidget(button, row, 0)
                entriesLayout.addWidget(label, row, 1)
                entriesLayout.addWidget(shortcutLineEdit, row, 2)
                shortcutLineEdit.name = name
                self.shortcutLineEdits[name] = shortcutLineEdit

                warnConflictLabel = QLabel('')
                shortcutLineEdit.warnConflictLabel = warnConflictLabel
                entriesLayout.addWidget(warnConflictLabel, row, 4)
                warnConflictLabel.setMinimumWidth(max_warn_width)

                if (
                        highlighted_shortcut is not None
                        and name == highlighted_shortcut
                    ):
                    shortcutLineEdit.setStyleSheet(
                        widgets_style.LINEEDIT_WARNING_STYLESHEET
                    )
                row += 1

        for group, shortcuts in self.new_hard_shortcuts.items():
            if not shortcuts:
                continue
            title = group.strip('()').replace('_', ' ').capitalize()
            title = title + ' (not customizable)'
            entriesLayout.addWidget(QLabel(f'<b>{title}</b>'), row, 0, 1, 5)
            row += 1
            for what, shortcut in shortcuts.items():
                button = widgets_push_buttons.PushButton(self, flat=True)
                if isinstance(what, str):
                    label = QLabel(f'{what}:')
                else:
                    (name, widget) = what
                    try:
                        button.setIcon(widget.icon())
                    except:
                        pass
                    label = QLabel(f'{name}:')
                lineEditTxt = QLabel(shortcut)
                entriesLayout.addWidget(button, row, 0)
                entriesLayout.addWidget(label, row, 1)
                entriesLayout.addWidget(lineEditTxt, row, 2)
                row += 1
        
        entriesLayout.setColumnStretch(0, 0)
        entriesLayout.setColumnStretch(1, 0)
        entriesLayout.setColumnStretch(2, 1)
        entriesLayout.setColumnStretch(3, 0)
        entriesLayout.setColumnStretch(4, 0)

        scrollAreaWidget.setLayout(entriesLayout)
        scrollArea.setWidget(scrollAreaWidget)
        buttonsLayout = widgets_push_buttons.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addWidget(scrollArea)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setFont(fonts.font)
        self.setLayout(mainLayout)

    def _groupAndSortShortcuts(
            self, 
            shortcuts_dict, 
            keep_at_beginning, 
            grouped_keys
        ):
        for key in shortcuts_dict.keys():
            actual_key = key
            if isinstance(key, tuple):
                key = key[0]
            found = False
            if key in keep_at_beginning:
                grouped_keys['keep_at_beginning'].append(actual_key)
            else:
                for group_key in grouped_keys.keys():
                    if key.endswith(group_key) and group_key not in ['keep_at_beginning', 'other']:
                        grouped_keys[group_key].append(actual_key)
                        found = True
                        break
                if not found:
                    grouped_keys['other'].append(actual_key)

        grouped_shortcuts = {}
        for group, group_list in grouped_keys.items():
            sorted_keys = natsorted(
                group_list,
                key=lambda x: x[0] if isinstance(x, tuple) else x
            )
            grouped_shortcuts[group] = {
                k: shortcuts_dict[k]
                for k in sorted_keys
            }
        return grouped_shortcuts
        
    def setShortcutLineEditEventFilter(self):
        sender = self.sender()
        previous = getattr(self, '_activeShortcutLineEdit', None)
        if previous is not None and previous is not sender:
            QApplication.instance().removeEventFilter(previous)
            previous.isRecording = False

        QApplication.instance().installEventFilter(sender)
        sender.isRecording = True
        self._activeShortcutLineEdit = sender

    def releaseShortcutLineEditEventFilter(self):
        sender = self.sender()
        QApplication.instance().removeEventFilter(sender)
        sender.isRecording = False
        if getattr(self, '_activeShortcutLineEdit', None) is sender:
            self._activeShortcutLineEdit = None
        
    def shortcutChanged(self, text):
        sender = self.sender()
        self.checkDuplicateShortcuts(text, sender=sender)
        self.updateMouseBindings(text, sender=sender)
    
    def updateMouseBindings(self, text, sender=None):
        if sender is None:
            sender = self.sender()
        if text.startswith('Mouse '):
            sender.isShortcutMouseButton = True
        else:
            sender.isShortcutMouseButton = False
    
    def checkDuplicateShortcuts(self, text, sender=None):
        if sender is None:
            sender = self.sender()
            
        warnConflictLabel = getattr(sender, 'warnConflictLabel', None)
        if warnConflictLabel is not None:
            warnConflictLabel.setText('')
            
            
        if hasattr(sender, 'conflictWith'):
            conflictWith = getattr(sender, 'conflictWith')
            if conflictWith is not None:
                for name_other, shortcutLineEdit in self.shortcutLineEdits.items():
                    if name_other == conflictWith:
                        warnConflictLabel_other = getattr(shortcutLineEdit, 'warnConflictLabel', None)
                        if warnConflictLabel_other is not None:
                            warnConflictLabel_other.setText('')
                        shortcutLineEdit.conflictWith = None
                
        sender.conflictWith = None
            
        if text == '':
            return

        # `text` (textChanged) is the displayed text, which uses macOS key
        # names on macOS ('Command+O'), while ShortcutLineEdit.text() gives
        # the portable form ('Ctrl+O'): compare portable forms
        text = widgets_line_edits.macShortcutToWindows(text)
        name = sender.name
        group_updated = self.exclusivity_groups_reverse.get(name, 'other')
        for name_other, shortcutLineEdit in self.shortcutLineEdits.items():
            group = self.exclusivity_groups_reverse.get(name_other, 'other')
            if group != group_updated:
                continue    
            if shortcutLineEdit == sender:
                continue
            if shortcutLineEdit.text() != text:
                continue
            # shortcutLineEdit.setText('')
            warnConflictLabel = getattr(shortcutLineEdit, 'warnConflictLabel', None)
            if warnConflictLabel is not None:
                warnConflictLabel.setText(
                    self.conflict_text_formatter(name)
                )
            warnConflictLabel_sender = getattr(sender, 'warnConflictLabel', None)
            if warnConflictLabel_sender is not None:
                warnConflictLabel_sender.setText(
                    self.conflict_text_formatter(name_other)
                )
            sender.conflictWith = name_other
            break
        for shortcuts in self.new_hard_shortcuts.values():
            for name_other, shortcut_txt in shortcuts.items():
                if widgets_line_edits.macShortcutToWindows(shortcut_txt) == text:
                    warnConflictLabel = getattr(
                        sender, 'warnConflictLabel', None
                    )
                    if warnConflictLabel is not None:
                        warnConflictLabel.setText(
                            self.conflict_text_formatter(name_other)
                        )
                    return
    
    def warnInvalidKeySequenceDelObjWithLeftClick(self):
        txt = html_utils.paragraph(
            'The selected key sequence to delete objects with "Left click" '
            'is invalid.<br><br>'
            'Only "Middle click" can be used without pressing keys.<br><br>'
            'Thank you for your patience!'
        )
        msg = widgets_message_box.myMessageBox()
        msg.warning(self, 'Invalid key sequence to delete objects', txt)
    
    def ok_cb(self):
        if self.show_delete_object_shortcut:
            delObjButtonText = self.delObjButtonCombobox.currentText()
            delObjKeySequence = self.delObjShortcutLineEdit.keySequence
            if delObjButtonText == 'Left click' and delObjKeySequence is None:
                self.warnInvalidKeySequenceDelObjWithLeftClick()
                return
        
        self.shortcutLineEdits.pop('Zoom out', None)
        # 'Delete object' row is missing when show_delete_object_shortcut
        # is False
        self.shortcutLineEdits.pop('Delete object', None)
        self.cancel = False
        self.mouseBindings = dict()
        for name, shortcutLineEdit in self.shortcutLineEdits.items():
            text = shortcutLineEdit.text()
            if shortcutLineEdit.isShortcutMouseButton:
                button_name = text.split('Mouse ')[-1]
                button = getattr(Qt.MouseButton, button_name, None)
                if button is None:
                    printl(f'Warning: could not find mouse button for {button_name}')
                    continue
                self.mouseBindings[name] = button
                self.customShortcuts[name] = (text, button)
            elif shortcutLineEdit.isShortcutKeyPress:
                self.customShortcuts[name] = (text, shortcutLineEdit.key)
            else:
                self.customShortcuts[name] = (
                    text, shortcutLineEdit.keySequence
                )
        
        if self.show_delete_object_shortcut:
            delObjQtButton = (
                Qt.MouseButton.LeftButton if delObjButtonText == 'Left click'
                else Qt.MouseButton.MiddleButton
            )
            self.delObjAction = delObjKeySequence, delObjQtButton
        self.zoomOutKeyValue = self.zoomShortcutLineEdit.key
        
        self.close()
        
    def closeEvent(self, event):
        active = getattr(self, '_activeShortcutLineEdit', None)
        if active is not None:
            QApplication.instance().removeEventFilter(active)
            active.isRecording = False
            self._activeShortcutLineEdit = None
        super().closeEvent(event)
    
    def showEvent(self, event) -> None:
        minimumWidth = self.scrollArea.minimumWidthNoScrollbar()
        self.resize(int(minimumWidth * 1.1), self.height())
        self.move(self.x(), 100)


class ScaleBarPropertiesDialog(dialogs_base.QBaseDialog):
    sigValueChanged = Signal(object)
    
    def __init__(
            self, maxLength, maxThickness, PhysicalSizeX, parent=None, 
            **properties
        ):
        super().__init__(parent=parent)
        
        self.cancel = True
        self.setWindowTitle('Scale bar properties')
        
        self.PhysicalSizeX = PhysicalSizeX
        
        mainLayout = QVBoxLayout()
        
        formLayout = widgets_line_edits.FormLayout()
        formLayout.setVerticalSpacing(10)
        formLayout.setHorizontalSpacing(50)
        
        row = 0
        unitCombobox = QComboBox()
        unitFormWidget = widgets_line_edits.formWidget(
            unitCombobox, labelTextLeft='Physical unit'
        )
        unitCombobox.addItems(
            ['nm', 'μm', 'mm', 'cm']
        )
        if properties.get('unit') is None:
            unitCombobox.setCurrentIndex(1)
        else:
            unitCombobox.setCurrentText(properties.get('unit'))
        formLayout.addFormWidget(
            unitFormWidget, row=row, 
            leftLabelAlignment=Qt.AlignLeft
        )       
        self.unitCombobox = unitCombobox
        
        row += 1
        lengthDoubleSpinbox = widgets_numeric_inputs.DoubleSpinBox()
        lengthDoubleSpinbox.setMaximum(maxLength)
        lengthDoubleSpinbox.setMinimum(PhysicalSizeX)
        lengthDoubleSpinbox.setDecimals(1)
        if properties.get('length_unit') is not None:
            lengthDoubleSpinbox.setValue(properties.get('length_unit'))
        else:
            deafultLength = np.ceil(PhysicalSizeX*15)
            lengthDoubleSpinbox.setValue(round(deafultLength))
        lengthFormWidget = widgets_line_edits.formWidget(
            lengthDoubleSpinbox, labelTextLeft='Length (μm)'
        )
        self.lengthFormWidget = lengthFormWidget
        self.lengthDoubleSpinbox = lengthDoubleSpinbox
        formLayout.addFormWidget(
            lengthFormWidget, row=row, 
            leftLabelAlignment=Qt.AlignLeft
        )       
        
        row += 1
        thicknessSpinbox = widgets_numeric_inputs.DoubleSpinBox()
        thicknessSpinbox.setMaximum(maxThickness)
        thicknessSpinbox.setMinimum(1)
        if properties.get('thickness') is not None:
            thicknessSpinbox.setValue(properties.get('thickness'))
        else:
            thicknessSpinbox.setValue(round(4, 1))
        thicknessSpinbox.setDecimals(1)
        thicknessFormWidget = widgets_line_edits.formWidget(
            thicknessSpinbox, labelTextLeft='Thickness (pixel)'
        )
        formLayout.addFormWidget(
            thicknessFormWidget, row=row, 
            leftLabelAlignment=Qt.AlignLeft
        )       
        self.thicknessSpinbox = thicknessSpinbox
        
        row += 1
        locCombobox = QComboBox()
        locFormWidget = widgets_line_edits.formWidget(
            locCombobox, labelTextLeft='Location'
        )
        locCombobox.addItems(
            ['Bottom-right', 'Bottom-left', 'Top-left', 'Top-right', 'Custom']
        )
        loc = properties.get('loc')
        if isinstance(loc, str):
            locCombobox.setCurrentText(loc.capitalize())
        formLayout.addFormWidget(
            locFormWidget, row=row, 
            leftLabelAlignment=Qt.AlignLeft
        )       
        self.locCombobox = locCombobox
        
        row += 1
        self.colorButton = widgets_push_buttons.myColorButton(color=(255, 255, 255))
        if properties.get('color') is not None:
            self.colorButton.setColor(properties.get('color'))
        colorFormWidget = widgets_line_edits.formWidget(
            self.colorButton, labelTextLeft='Color',
            widgetAlignment=Qt.AlignCenter, stretchWidget=False
        )
        formLayout.addFormWidget(
            colorFormWidget, row=row, 
            leftLabelAlignment=Qt.AlignLeft
        )  
        
        row += 1
        displayTextToggle = widgets_push_buttons.Toggle()
        if properties.get('is_text_visible') is not None:
            displayTextToggle.setChecked(properties.get('is_text_visible'))
        else:
            displayTextToggle.setChecked(True)
        displayTextFormWidget = widgets_line_edits.formWidget(
            displayTextToggle, labelTextLeft='Display text',
            widgetAlignment=Qt.AlignCenter, stretchWidget=False
        )
        formLayout.addFormWidget(
            displayTextFormWidget, row=row, 
            leftLabelAlignment=Qt.AlignLeft
        )       
        self.displayTextToggle = displayTextToggle
        
        row += 1
        fontSizeSpinbox = widgets_numeric_inputs.SpinBox()
        if properties.get('font_size') is not None:
            fontSizeSpinbox.setValue(int(properties.get('font_size')))
        else:
            fontSizeSpinbox.setValue(12)
        fontSizeFormWidget = widgets_line_edits.formWidget(
            fontSizeSpinbox, labelTextLeft='Font size (px)'
        )
        self.fontSizeSpinbox = fontSizeSpinbox
        formLayout.addFormWidget(
            fontSizeFormWidget, row=row, 
            leftLabelAlignment=Qt.AlignLeft
        ) 
        
        row += 1
        decimalsSpinbox = widgets_numeric_inputs.SpinBox()
        decimalsSpinbox.setMaximum(6)
        decimalsSpinbox.setMinimum(0)
        if properties.get('num_decimals') is not None:
            decimalsSpinbox.setValue(properties.get('num_decimals'))
        else:
            decimalsSpinbox.setValue(0)
        decimalsFormWidget = widgets_line_edits.formWidget(
            decimalsSpinbox, labelTextLeft='Number of decimals'
        )
        formLayout.addFormWidget(
            decimalsFormWidget, row=row, 
            leftLabelAlignment=Qt.AlignLeft
        )       
        self.decimalsSpinbox = decimalsSpinbox
        
        row += 1
        moveWithZoomToggle = widgets_push_buttons.Toggle()
        moveWithZoomWidget = widgets_line_edits.formWidget(
            moveWithZoomToggle, labelTextLeft='Move scale bar with zoom',
            widgetAlignment=Qt.AlignCenter, stretchWidget=False
        )
        formLayout.addFormWidget(
            moveWithZoomWidget, row=row, 
            leftLabelAlignment=Qt.AlignLeft
        )
        self.moveWithZoomToggle = moveWithZoomToggle
        
        mainLayout.addLayout(formLayout)
        
        buttonsLayout = widgets_push_buttons.CancelOkButtonsLayout()
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        mainLayout.addStretch()
        
        self.setLayout(mainLayout)
        self.setFont(fonts.font)
        
        self.unitCombobox.currentTextChanged.connect(self.updateLengthUnit)
        self.colorButton.clicked.disconnect()
        self.colorButton.clicked.connect(self.selectColor)
        
        self.colorButton.sigColorChanging.connect(self.onValueChanged)
        self.lengthDoubleSpinbox.valueChanged.connect(self.onValueChanged)
        self.thicknessSpinbox.valueChanged.connect(self.onValueChanged)
        self.locCombobox.currentTextChanged.connect(self.onValueChanged)
        self.displayTextToggle.toggled.connect(self.onValueChanged)
        self.fontSizeSpinbox.valueChanged.connect(self.onValueChanged)
        self.decimalsSpinbox.valueChanged.connect(self.onValueChanged)
        self.moveWithZoomToggle.toggled.connect(self.onValueChanged)
    
    def onValueChanged(self, *args, **kwargs):
        self.sigValueChanged.emit(self.kwargs())
    
    def selectColor(self):
        color = self.colorButton.color()
        self.colorButton.origColor = color
        self.colorButton.colorDialog.setCurrentColor(color)
        _window_stacking.set_window_flags_on_top(
            self.colorButton.colorDialog, Qt.Window, parent=self
        )
        self.colorButton.colorDialog.open()
        w = self.width()
        left = self.pos().x()
        colorDialogTop = self.colorButton.colorDialog.pos().y()
        self.colorButton.colorDialog.move(w+left+10, colorDialogTop)
    
    def updateLengthUnit(self, unit):
        newText = re.sub(
            r'\(.*\)', f'({unit})', 
            self.lengthFormWidget.labelLeft.text()
        )
        self.lengthFormWidget.labelLeft.setText(newText)
        self.onValueChanged(self)
    
    def kwargs(self):
        unit = self.unitCombobox.currentText()
        length_unit = self.lengthDoubleSpinbox.value()
        length_um = _core.convert_length(length_unit, unit, 'μm')
        length_pixel = length_um/self.PhysicalSizeX
        kwargs = {
            'thickness': self.thicknessSpinbox.value(),
            'length_pixel': length_pixel,
            'length_unit': length_unit,
            'is_text_visible': self.displayTextToggle.isChecked(),
            'color': self.colorButton.color(),
            'loc': self.locCombobox.currentText().lower(),
            'font_size': self.fontSizeSpinbox.value(),
            'unit': unit,
            'num_decimals': self.decimalsSpinbox.value(),
            'move_with_zoom': self.moveWithZoomToggle.isChecked()
        }
        return kwargs
    
    def ok_cb(self):
        self.cancel = False
        self.close()


class TimestampPropertiesDialog(dialogs_base.QBaseDialog):
    sigValueChanged = Signal(object)
    
    def __init__(self, parent=None, **properties):
        super().__init__(parent=parent)
        
        self.cancel = True
        self.setWindowTitle('Timestamp preferences')
        
        mainLayout = QVBoxLayout()
        
        formLayout = widgets_line_edits.FormLayout()
        formLayout.setVerticalSpacing(10)
        formLayout.setHorizontalSpacing(50)
        
        row = 0
        self.startTimeWidget = gui_mixin_widgets.TimeWidget()
        if properties.get('start_timedelta') is not None:
            self.startTimeWidget.setValuesFromTimedelta(
                properties.get('start_timedelta')
            )
        startTimeFormWidget = widgets_line_edits.formWidget(
            self.startTimeWidget, labelTextLeft='Start time',
        )
        formLayout.addFormWidget(
            startTimeFormWidget, row=row, 
            leftLabelAlignment=Qt.AlignLeft
        )
        
        row += 1
        self.colorButton = widgets_push_buttons.myColorButton(color=(255, 255, 255))
        if properties.get('color') is not None:
            self.colorButton.setColor(properties.get('color'))
        colorFormWidget = widgets_line_edits.formWidget(
            self.colorButton, labelTextLeft='Color',
            widgetAlignment=Qt.AlignCenter, stretchWidget=False
        )
        formLayout.addFormWidget(
            colorFormWidget, row=row, 
            leftLabelAlignment=Qt.AlignLeft
        ) 
        
        row += 1
        fontSizeWidget = gui_mixin_widgets.FontSizeWidget()
        if properties.get('font_size') is not None:
            fontSizeWidget.setValue(properties.get('font_size'))
        else:
            fontSizeWidget.setValue(12)
        fontSizeFormWidget = widgets_line_edits.formWidget(
            fontSizeWidget, labelTextLeft='Font size (px)'
        )
        self.fontSizeWidget = fontSizeWidget
        formLayout.addFormWidget(
            fontSizeFormWidget, row=row, 
            leftLabelAlignment=Qt.AlignLeft
        )
        
        row += 1
        locCombobox = QComboBox()
        locFormWidget = widgets_line_edits.formWidget(
            locCombobox, labelTextLeft='Location'
        )
        locCombobox.addItems(
            ['Top-left', 'Top-right', 'Bottom-left', 'Bottom-right', 'Custom']
        )
        loc = properties.get('loc')
        if isinstance(loc, str):
            locCombobox.setCurrentText(loc.capitalize())
        formLayout.addFormWidget(
            locFormWidget, row=row, 
            leftLabelAlignment=Qt.AlignLeft
        )       
        self.locCombobox = locCombobox
        
        row += 1
        moveWithZoomToggle = widgets_push_buttons.Toggle()
        moveWithZoomWidget = widgets_line_edits.formWidget(
            moveWithZoomToggle, labelTextLeft='Move timestamp with zoom',
            widgetAlignment=Qt.AlignCenter, stretchWidget=False
        )
        formLayout.addFormWidget(
            moveWithZoomWidget, row=row, 
            leftLabelAlignment=Qt.AlignLeft
        )
        self.moveWithZoomToggle = moveWithZoomToggle
        
        mainLayout.addLayout(formLayout)
        
        buttonsLayout = widgets_push_buttons.CancelOkButtonsLayout()
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        mainLayout.addStretch()
        
        self.setLayout(mainLayout)
        self.setFont(fonts.font)
        
        self.colorButton.clicked.disconnect()
        self.colorButton.clicked.connect(self.selectColor)
        
        self.startTimeWidget.sigValueChanged.connect(self.onValueChanged)
        
        self.locCombobox.currentTextChanged.connect(self.onValueChanged)
        self.fontSizeWidget.sigTextChanged.connect(self.onValueChanged)
        self.moveWithZoomToggle.toggled.connect(self.onValueChanged)
    
    def onValueChanged(self, *args, **kwargs):
        self.sigValueChanged.emit(self.kwargs())
    
    def selectColor(self):
        color = self.colorButton.color()
        self.colorButton.origColor = color
        self.colorButton.colorDialog.setCurrentColor(color)
        _window_stacking.set_window_flags_on_top(
            self.colorButton.colorDialog, Qt.Window, parent=self
        )
        self.colorButton.colorDialog.open()
        w = self.width()
        left = self.pos().x()
        colorDialogTop = self.colorButton.colorDialog.pos().y()
        self.colorButton.colorDialog.move(w+left+10, colorDialogTop)
    
    def kwargs(self):
        kwargs = {
            'color': self.colorButton.color(),
            'start_timedelta': self.startTimeWidget.timedelta(),
            'loc': self.locCombobox.currentText().lower(),
            'font_size': self.fontSizeWidget.text(),
            'move_with_zoom': self.moveWithZoomToggle.isChecked()
        }
        return kwargs
    
    def ok_cb(self):
        self.cancel = False
        self.close()


class SetCustomLevelsLut(dialogs_base.QBaseDialog):
    sigLevelsChanged = Signal(object)
    
    def __init__(
            self, 
            init_min_value=None, 
            init_max_value=None, 
            minimum_min_value=0,
            maximum_max_value=None, 
            parent=None
        ):
        super().__init__(parent=parent)
        
        self.cancel = True
        
        self.setWindowTitle('Custom LUT levels')
        
        layout = QVBoxLayout()
        
        self.minLevelSlider = widgets_numeric_inputs.sliderWithSpinBox(
            title='Minimum', 
            title_loc='top', 
        )
        self.minLevelSlider.setMinimum(minimum_min_value)
            
        if init_min_value is not None:
            self.minLevelSlider.setValue(init_min_value)
        
        layout.addWidget(self.minLevelSlider)
        
        self.maxLevelSlider = widgets_numeric_inputs.sliderWithSpinBox(
            title='Maximum', 
            title_loc='top', 
        )
        self.maxLevelSlider.setMinimum(minimum_min_value)
        if init_max_value is not None:
            self.maxLevelSlider.setValue(init_max_value)
        
        if maximum_max_value is not None:
            self.maxLevelSlider.setMaximum(maximum_max_value)
            self.minLevelSlider.setMaximum(maximum_max_value)
            
        layout.addWidget(self.maxLevelSlider)
        
        self.minLevelSlider.sigValueChange.connect(self.emitLevelsChanged)
        self.maxLevelSlider.sigValueChange.connect(self.emitLevelsChanged)
        
        buttonsLayout = widgets_push_buttons.CancelOkButtonsLayout()
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        layout.addSpacing(20)
        layout.addLayout(buttonsLayout)
        
        self.setLayout(layout)
    
    def sizeHint(self):
        heightHint = super().sizeHint().height()
        widthHint = super().sizeHint().width()*2
        return QSize(widthHint, heightHint)
        
    
    def levels(self):
        levels = (self.minLevelSlider.value(), self.maxLevelSlider.value())
        return levels
    
    def emitLevelsChanged(self, value):
        self.sigLevelsChanged.emit(self.levels())
    
    def ok_cb(self):
        self.cancel = False
        self.selectedLevels = self.levels()
        self.close()


class ViewCcaTableWindow(pdDataFrameWidget):
    sigUpdateCcaTable = Signal(object)
    
    def __init__(self, df, parent=None):
        super().__init__(df, parent=parent)
        
        updateTableButton = widgets_push_buttons.reloadPushButton(
            'Update table with visible IDs...'
        )
        buttonsLayout = QHBoxLayout()
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(updateTableButton)
        
        self._layout.insertLayout(0, buttonsLayout)
        
        updateTableButton.clicked.connect(self.emitUpdateCcaTable)
    
    def emitUpdateCcaTable(self):
        self.sigUpdateCcaTable.emit(self)


class ObjectCountDialog(dialogs_base.QBaseDialog):
    sigShowEvent = Signal()
    sigUpdateCounts = Signal()
    sigClose = Signal(object)
    
    def __init__(
            self, 
            categoryCountMapper: dict,
            parent=None,
            data: list['load.loadData'] | None=None
        ):
        super().__init__(parent=parent)
        self.setWindowTitle('Object count')
        
        self.cancel = False
        mainLayout = QVBoxLayout()

        cancelOkLayout = widgets_push_buttons.CancelOkButtonsLayout()
        cancelOkLayout.okButton.clicked.connect(self.ok_cb)
        cancelOkLayout.cancelButton.clicked.connect(self.close)
        
        self.data = data
        if data is not None:
            saveCountsButton = widgets_push_buttons.savePushButton(
                'Export counts to CSV table'
            )
            saveCountsButton.clicked.connect(self.saveCounts)
            cancelOkLayout.insertWidget(3, saveCountsButton)
        
        updateCountsButton = widgets_push_buttons.reloadPushButton('Update counts')
        cancelOkLayout.insertWidget(3, updateCountsButton)
        updateCountsButton.clicked.connect(self.emitUpdateCounts)
        
        mainLayout.addWidget(
            QLabel(html_utils.paragraph('Object count<br>', font_size='18px')), 
            alignment=Qt.AlignLeft
        )
        self.showHideButtons = []
        self.categoryLabelMapper = {}
        for category, count in categoryCountMapper.items():
            categoryLayout = QHBoxLayout()
            categoryLayout.addSpacing(10)
            catText = html_utils.paragraph(
                f'<br>{category}<br>', font_size='13px'
            )
            catLabel = QLabel(catText)
            categoryLayout.addWidget(catLabel)
            categoryLayout.addStretch(1)
            
            countText = html_utils.paragraph(
                f'<br>{count}<br>', font_size='13px'
            )
            countLabel = QLabel(countText)
            categoryLayout.addWidget(countLabel)
            
            self.categoryLabelMapper[category] = countLabel
            
            showHideButton = widgets_push_buttons.showDetailsButton(txt='')
            showHideButton.setChecked(True)
            showHideButton.sigToggled.connect(
                partial(self.showHideCount, labels=(catLabel, countLabel))
            )
            showHideButton.setToolTip(f'Show/hide "{category}" count')
            categoryLayout.addSpacing(10)
            categoryLayout.addWidget(showHideButton)
            showHideButton.category = category
            
            self.showHideButtons.append(showHideButton)
            
            categoryLayout.setStretch(0, 0)
            categoryLayout.setStretch(1, 0)
            categoryLayout.setStretch(3, 0)
            
            mainLayout.addLayout(categoryLayout)
            mainLayout.addWidget(widgets_basic_widgets.QHLine())
        
        mainLayout.addSpacing(10)
        
        infoLayout = QHBoxLayout()
        self.livePreviewCheckbox = QCheckBox('Live preview')
        self.livePreviewCheckbox.setChecked(True)
        infoLayout.addWidget(self.livePreviewCheckbox)
        infoLayout.addStretch(1)
        self.warnLabel = QLabel('')
        infoLayout.addWidget(self.warnLabel)
        self.livePreviewCheckbox.toggled.connect(self.updateWarnLabel)
        mainLayout.addLayout(infoLayout)
        
        mainLayout.addSpacing(30)
        mainLayout.addStretch(1)
        mainLayout.addLayout(cancelOkLayout)
        
        self.setLayout(mainLayout)
    
    def saveCounts(self, checked=False):
        categories = self.activeCategories()
        for posData in self.data:
            countMapper, numObjsPerFrame = (
                posData.countObjectsInSegm(categories)
            )
            countMapper.pop('In current frame', None)
            df_count_endname = posData.saveObjCounts(
                countMapper, numObjsPerFrame
            )
        
        txt = html_utils.paragraph(f"""
            Done!<br><br>
            Objects count table saved in every loaded Position folder<br> 
            as a <b>CSV file ending with</b> <code>{df_count_endname}</code>
        """)
        msg = widgets_message_box.myMessageBox(wrapText=False)
        msg.information(self, 'Objects count saved', txt)
    
    def updateWarnLabel(self, checked):
        if not checked:
            self.warnLabel.setText(
                html_utils.paragraph(
                    'WARNING: without live preview, counts are not updated', 
                    font_color='red'
                )
            )
        else:
            self.warnLabel.setText('')
    
    def emitUpdateCounts(self):
        self.sigUpdateCounts.emit()
    
    def activeCategories(self) -> List[str]:
        activeCategories = []
        for showHideButton in self.showHideButtons:
            if not showHideButton.isChecked():
                continue
            activeCategories.append(showHideButton.category)
        
        return activeCategories
    
    def showHideCount(self, checked, labels):
        for label in labels:
            label.setVisible(checked)
        
        QTimer.singleShot(100, self.resizeToHeightHint)

    def updateCounts(self, categoryCountMapper):
        for category, count in categoryCountMapper.items():
            countLabel = self.categoryLabelMapper[category]
            countText = html_utils.paragraph(
                f'<br>{count}<br>', font_size='13px'
            )
            countLabel.setText(countText)
                
    
    def resizeToHeightHint(self):
        heightHint = self.sizeHint().height()
        self.resize(self.width(), heightHint)
        
    def showEvent(self, event):
        widthHint = self.sizeHint().width()
        self.resize(int(widthHint*1.5), self.height())
        self.sigShowEvent.emit()
    
    def ok_cb(self):
        self.cancel = False
        self.close()
    
    def closeEvent(self, event):
        self.sigClose.emit(event)


class AnnotateObjTrackSettingsDialog(dialogs_base.QBaseDialog):
    sigValuesChanged = Signal(dict)
    
    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.cancel = True

        self.setWindowTitle('Annotate object track settings')

        mainLayout = QVBoxLayout()
        formLayout = widgets_line_edits.FormLayout()

        row = 0
        lengthInfoTxt = html_utils.paragraph(
            'Number of past frames to include when drawing each cell track. '
            'Higher values show longer history and can add visual clutter.'
        )
        self.lengthSpinBox = widgets_numeric_inputs.SpinBox()
        self.lengthSpinBox.setMinimum(1)
        self.lengthSpinBox.setMaximum(999)
        self.lengthWidget = widgets_line_edits.formWidget(
            self.lengthSpinBox,
            labelTextLeft='Track length (frames): ',
            parent=self,
            stretchWidget=False,
            addInfoButton=True,
            infoTxt=lengthInfoTxt,
        )
        formLayout.addFormWidget(self.lengthWidget, row=row)

        row += 1
        colorInfoTxt = html_utils.paragraph(
            'RGB color used for cell movement tracks. '
            'This affects both line color and fade rendering.'
        )
        self.colorButton = widgets_push_buttons.myColorButton(color=tuple(settings['color']))
        try:
            self.colorButton.clicked.disconnect()
        except TypeError:
            pass
        self.colorButton.clicked.connect(self.selectColor)
        self.colorWidget = widgets_line_edits.formWidget(
            self.colorButton,
            labelTextLeft='Track color: ',
            parent=self,
            stretchWidget=False,
            addInfoButton=True,
            infoTxt=colorInfoTxt,
        )
        formLayout.addFormWidget(self.colorWidget, row=row)

        row += 1
        fadeInfoTxt = html_utils.paragraph(
            'Number of track segments over which fading out is applied. '
            'Lower values create a longer "full boldness" trail.'
        )
        self.nFadeSpinBox = widgets_numeric_inputs.SpinBox()
        self.nFadeSpinBox.setMinimum(1)
        self.nFadeSpinBox.setMaximum(999)
        self.nFadeWidget = widgets_line_edits.formWidget(
            self.nFadeSpinBox,
            labelTextLeft='Full boldness trail (segments): ',
            parent=self,
            stretchWidget=False,
            addInfoButton=True,
            infoTxt=fadeInfoTxt,
        )
        formLayout.addFormWidget(self.nFadeWidget, row=row)

        row += 1
        maxWidthInfoTxt = html_utils.paragraph(
            'Maximum line width for the newest part of each track. '
            'Must be greater than or equal to "Min line width".'
        )
        self.maxWidthSpinBox = widgets_numeric_inputs.SpinBox()
        self.maxWidthSpinBox.setMinimum(1)
        self.maxWidthSpinBox.setMaximum(100)
        self.maxWidthWidget = widgets_line_edits.formWidget(
            self.maxWidthSpinBox,
            labelTextLeft='Max line width: ',
            parent=self,
            stretchWidget=False,
            addInfoButton=True,
            infoTxt=maxWidthInfoTxt,
        )
        formLayout.addFormWidget(self.maxWidthWidget, row=row)

        row += 1
        minWidthInfoTxt = html_utils.paragraph(
            'Minimum line width for the oldest visible track segments. '
            'Automatically constrained to stay below or equal to "Max line width".'
        )
        self.minWidthSpinBox = widgets_numeric_inputs.SpinBox()
        self.minWidthSpinBox.setMinimum(1)
        self.minWidthSpinBox.setMaximum(100)
        self.minWidthWidget = widgets_line_edits.formWidget(
            self.minWidthSpinBox,
            labelTextLeft='Min line width: ',
            parent=self,
            stretchWidget=False,
            addInfoButton=True,
            infoTxt=minWidthInfoTxt,
        )
        formLayout.addFormWidget(self.minWidthWidget, row=row)

        row += 1
        maxAlphaInfoTxt = html_utils.paragraph(
            'Opacity of the newest track segments, in percent (0-100). '
            '100 is fully opaque, 0 is fully transparent.'
        )
        self.maxAlphaSpinBox = widgets_numeric_inputs.SpinBox()
        self.maxAlphaSpinBox.setRange(0, 100)
        self.maxAlphaWidget = widgets_line_edits.formWidget(
            self.maxAlphaSpinBox,
            labelTextLeft='Max alpha (%): ',
            parent=self,
            stretchWidget=False,
            addInfoButton=True,
            infoTxt=maxAlphaInfoTxt,
        )
        formLayout.addFormWidget(self.maxAlphaWidget, row=row)

        row += 1
        minAlphaInfoTxt = html_utils.paragraph(
            'Opacity of the oldest visible track segments, in percent (0-100). '
            'Automatically constrained to stay below or equal to "Max alpha".'
        )
        self.minAlphaSpinBox = widgets_numeric_inputs.SpinBox()
        self.minAlphaSpinBox.setRange(0, 100)
        self.minAlphaWidget = widgets_line_edits.formWidget(
            self.minAlphaSpinBox,
            labelTextLeft='Min alpha (%): ',
            parent=self,
            stretchWidget=False,
            addInfoButton=True,
            infoTxt=minAlphaInfoTxt,
        )
        formLayout.addFormWidget(self.minAlphaWidget, row=row)
        row += 1
        showOnlyVisible3DInfoTxt = html_utils.paragraph(
            'Show only visible objects in 3D view.'
        )
        self.showOnlyVisible3DCheckBox = widgets_push_buttons.Toggle()
        self.showOnlyVisible3DCheckBox.setChecked(
            settings['3D_show_only_visible']
        )
        self.showOnlyVisible3DWidget = widgets_line_edits.formWidget(
            self.showOnlyVisible3DCheckBox,
            labelTextLeft='3D show only visible: ',
            parent=self,
            stretchWidget=False,
            addInfoButton=True,
            infoTxt=showOnlyVisible3DInfoTxt,
        )
        formLayout.addFormWidget(self.showOnlyVisible3DWidget, row=row)
        
        row += 1
        againstPrevInfoTxt = html_utils.paragraph(
            'When enabled, each object is linked only to its position in the '
            'immediately previous frame. '
            'When disabled, the full track history up to "Track length" is drawn. '
            'Is mutually exclusive with all other options!'
        )
        self.againstPrevToggle = widgets_push_buttons.Toggle()
        self.againstPrevWidget = widgets_line_edits.formWidget(
            self.againstPrevToggle,
            labelTextLeft='Only against previous frame: ',
            parent=self,
            stretchWidget=False,
            valueGetterName='isChecked',
            addInfoButton=True,
            infoTxt=againstPrevInfoTxt,
        )
        formLayout.addFormWidget(self.againstPrevWidget, row=row)

        row += 1
        againstPrevLineColorInfoTxt = html_utils.paragraph(
            'RGB color of the line connecting each object to its previous-frame '
            'position when "Only against previous frame" is enabled.'
        )
        self.againstPrevLineColorButton = widgets_push_buttons.myColorButton(
            color=tuple(settings['against_prev_line_color'])
        )
        try:
            self.againstPrevLineColorButton.clicked.disconnect()
        except TypeError:
            pass
        self.againstPrevLineColorButton.clicked.connect(
            lambda: self.selectColor(self.againstPrevLineColorButton)
        )
        self.againstPrevLineColorWidget = widgets_line_edits.formWidget(
            self.againstPrevLineColorButton,
            labelTextLeft='Against-prev line color: ',
            parent=self,
            stretchWidget=False,
            addInfoButton=True,
            infoTxt=againstPrevLineColorInfoTxt,
        )
        formLayout.addFormWidget(self.againstPrevLineColorWidget, row=row)

        row += 1
        againstPrevLineWidthInfoTxt = html_utils.paragraph(
            'Thickness of the line connecting each object to its previous-frame '
            'position when "Only against previous frame" is enabled.'
        )
        self.againstPrevLineWidthSpinBox = widgets_numeric_inputs.SpinBox()
        self.againstPrevLineWidthSpinBox.setMinimum(1)
        self.againstPrevLineWidthSpinBox.setMaximum(100)
        self.againstPrevLineWidthWidget = widgets_line_edits.formWidget(
            self.againstPrevLineWidthSpinBox,
            labelTextLeft='Against-prev line thickness: ',
            parent=self,
            stretchWidget=False,
            addInfoButton=True,
            infoTxt=againstPrevLineWidthInfoTxt,
        )
        formLayout.addFormWidget(self.againstPrevLineWidthWidget, row=row)

        row += 1
        againstPrevContourColorInfoTxt = html_utils.paragraph(
            'RGB color of previous-frame contours when "Only against previous '
            'frame" is enabled.'
        )
        self.againstPrevContourColorButton = widgets_push_buttons.myColorButton(
            color=tuple(settings['against_prev_contour_color'])
        )
        try:
            self.againstPrevContourColorButton.clicked.disconnect()
        except TypeError:
            pass
        self.againstPrevContourColorButton.clicked.connect(
            lambda: self.selectColor(self.againstPrevContourColorButton)
        )
        self.againstPrevContourColorWidget = widgets_line_edits.formWidget(
            self.againstPrevContourColorButton,
            labelTextLeft='Against-prev contour color: ',
            parent=self,
            stretchWidget=False,
            addInfoButton=True,
            infoTxt=againstPrevContourColorInfoTxt,
        )
        formLayout.addFormWidget(self.againstPrevContourColorWidget, row=row)

        row += 1
        againstPrevContourWidthInfoTxt = html_utils.paragraph(
            'Thickness of previous-frame contours when "Only against previous '
            'frame" is enabled.'
        )
        self.againstPrevContourWidthSpinBox = widgets_numeric_inputs.SpinBox()
        self.againstPrevContourWidthSpinBox.setMinimum(1)
        self.againstPrevContourWidthSpinBox.setMaximum(100)
        self.againstPrevContourWidthWidget = widgets_line_edits.formWidget(
            self.againstPrevContourWidthSpinBox,
            labelTextLeft='Against-prev contour thickness: ',
            parent=self,
            stretchWidget=False,
            addInfoButton=True,
            infoTxt=againstPrevContourWidthInfoTxt,
        )
        formLayout.addFormWidget(self.againstPrevContourWidthWidget, row=row)
        row += 1

        buttonsLayout = widgets_push_buttons.CancelOkButtonsLayout()
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addLayout(formLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        self.setLayout(mainLayout)

        self.lengthSpinBox.valueChanged.connect(self.emitValuesChanged)
        self.againstPrevToggle.toggled.connect(self._againstPrevToggled)
        self.nFadeSpinBox.valueChanged.connect(self.emitValuesChanged)
        self.maxWidthSpinBox.valueChanged.connect(self._maxWidthChanged)
        self.minWidthSpinBox.valueChanged.connect(self._minWidthChanged)
        self.maxAlphaSpinBox.valueChanged.connect(self._maxAlphaChanged)
        self.minAlphaSpinBox.valueChanged.connect(self._minAlphaChanged)
        self.colorButton.sigColorChanging.connect(self.emitValuesChanged)
        self.againstPrevLineColorButton.sigColorChanging.connect(self.emitValuesChanged)
        self.againstPrevLineWidthSpinBox.valueChanged.connect(self.emitValuesChanged)
        self.againstPrevContourColorButton.sigColorChanging.connect(self.emitValuesChanged)
        self.againstPrevContourWidthSpinBox.valueChanged.connect(self.emitValuesChanged)
        self.showOnlyVisible3DCheckBox.toggled.connect(self.emitValuesChanged)

        self._historyTrackWidgets = (
            self.lengthWidget,
            self.colorWidget,
            self.nFadeWidget,
            self.maxWidthWidget,
            self.minWidthWidget,
            self.maxAlphaWidget,
            self.minAlphaWidget,
            self.showOnlyVisible3DWidget,
        )
        self._againstPrevOnlyWidgets = (
            self.againstPrevLineColorWidget,
            self.againstPrevLineWidthWidget,
            self.againstPrevContourColorWidget,
            self.againstPrevContourWidthWidget,
        )

        self._isSyncing = False
        self.setValues(settings)

    def setValues(self, settings: dict):
        settings = dict(settings)

        self.lengthSpinBox.setValue(int(settings['length']))
        self.againstPrevToggle.setChecked(bool(settings['against_prev']))
        self.colorButton.setColor(tuple(settings['color']))
        self.nFadeSpinBox.setValue(int(settings['n_fade']))
        self.maxWidthSpinBox.setValue(int(settings['max_width']))
        self.minWidthSpinBox.setValue(int(settings['min_width']))
        self.maxAlphaSpinBox.setValue(int(settings['max_alpha_perc']))
        self.minAlphaSpinBox.setValue(int(settings['min_alpha_perc']))
        self.againstPrevLineColorButton.setColor(
            tuple(settings['against_prev_line_color'])
        )
        self.againstPrevLineWidthSpinBox.setValue(
            int(settings['against_prev_line_width'])
        )
        self.againstPrevContourColorButton.setColor(
            tuple(settings['against_prev_contour_color'])
        )
        self.againstPrevContourWidthSpinBox.setValue(
            int(settings['against_prev_contour_width'])
        )
        self.showOnlyVisible3DCheckBox.setChecked(
            bool(settings.get('3D_show_only_visible', False))
        )
        self._syncRanges()
        self._updateRelevantOptionsState()

    def values(self):
        r, g, b, _ = self.colorButton.color().getRgb()
        line_r, line_g, line_b, _ = self.againstPrevLineColorButton.color().getRgb()
        cont_r, cont_g, cont_b, _ = (
            self.againstPrevContourColorButton.color().getRgb()
        )
        return {
            'length': int(self.lengthSpinBox.value()),
            'against_prev': bool(self.againstPrevToggle.isChecked()),
            'color': (int(r), int(g), int(b)),
            'n_fade': int(self.nFadeSpinBox.value()),
            'max_width': int(self.maxWidthSpinBox.value()),
            'min_width': int(self.minWidthSpinBox.value()),
            'max_alpha_perc': int(self.maxAlphaSpinBox.value()),
            'min_alpha_perc': int(self.minAlphaSpinBox.value()),
            'against_prev_line_color': (
                int(line_r), int(line_g), int(line_b)
            ),
            'against_prev_line_width': int(self.againstPrevLineWidthSpinBox.value()),
            'against_prev_contour_color': (
                int(cont_r), int(cont_g), int(cont_b)
            ),
            'against_prev_contour_width': (
                int(self.againstPrevContourWidthSpinBox.value())
            ),
            '3D_show_only_visible': bool(self.showOnlyVisible3DCheckBox.isChecked()),
        }

    def _againstPrevToggled(self, checked):
        self._updateRelevantOptionsState()
        self.emitValuesChanged()

    def _updateRelevantOptionsState(self):
        is_against_prev = self.againstPrevToggle.isChecked()

        def _set_row_disabled(form_widget, disabled):
            form_widget.setDisabled(disabled)
            # `formWidget.setDisabled` may not disable the inner control when the
            # control is wrapped in a layout (stretchWidget=False).
            try:
                form_widget.widget.setDisabled(disabled)
            except Exception:
                pass

        for widget in self._historyTrackWidgets:
            _set_row_disabled(widget, is_against_prev)
        for widget in self._againstPrevOnlyWidgets:
            _set_row_disabled(widget, not is_against_prev)

    def _syncRanges(self):
        if self._isSyncing:
            return
        self._isSyncing = True

        if self.minWidthSpinBox.value() > self.maxWidthSpinBox.value():
            self.minWidthSpinBox.setValue(self.maxWidthSpinBox.value())

        if self.minAlphaSpinBox.value() > self.maxAlphaSpinBox.value():
            self.minAlphaSpinBox.setValue(self.maxAlphaSpinBox.value())

        self._isSyncing = False

    def _maxWidthChanged(self, value):
        self._syncRanges()
        self.emitValuesChanged()

    def _minWidthChanged(self, value):
        self._syncRanges()
        self.emitValuesChanged()

    def _maxAlphaChanged(self, value):
        self._syncRanges()
        self.emitValuesChanged()

    def _minAlphaChanged(self, value):
        self._syncRanges()
        self.emitValuesChanged()

    def emitValuesChanged(self, *args):
        self.sigValuesChanged.emit(self.values())

    def selectColor(self, colorButton=None):
        colorButton = self.colorButton if colorButton is None else colorButton
        color = colorButton.color()
        colorButton.origColor = color
        colorButton.colorDialog.setCurrentColor(color)
        _window_stacking.set_window_flags_on_top(
            colorButton.colorDialog, Qt.Window, parent=self
        )
        colorButton.colorDialog.open()
        w = self.width()
        left = self.pos().x()
        colorDialogTop = colorButton.colorDialog.pos().y()
        colorButton.colorDialog.move(w+left+10, colorDialogTop)

    def ok_cb(self):
        self.cancel = False
        self.settings = self.values()
        self.sigValuesChanged.emit(self.settings)
        self.close()
