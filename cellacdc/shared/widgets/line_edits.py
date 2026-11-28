"""Text inputs.

- Validated line edits: ValidLineEdit, ElidingLineEdit,
  alphaNumericLineEdit, LineEdit and the FloatLineEdit / IntLineEdit /
  VectorLineEdit family (shared invalid-entry stylesheet from style.py).
- File and folder path controls with recent paths: filePathControl,
  FolderPathControl, CsvFilePathControl.
- Labelled form rows: formWidget, FormLayout.
- Keyboard shortcuts: ShortcutLineEdit, KeySequenceFromText,
  QKeyEventToString, macShortcutToWindows, windowsShortcutToMac,
  modifierKeyToText. Tests simulate macOS by patching is_mac in this
  module.
"""
import os
import re
import numpy as np
from qtpy.QtCore import Signal, Qt, QRegularExpression, QEvent, QObject, PYQT6
from qtpy.QtGui import (
    QFont, QKeyEvent, QRegularExpressionValidator, QIcon, QKeySequence,
    QShowEvent, QFontMetrics
)
from qtpy.QtWidgets import (
    QHBoxLayout, QCheckBox, QWidget, QLineEdit, QGridLayout, QPushButton,
    QFrame
)
from cellacdc import myutils, is_mac, html_utils
from cellacdc import apps
from cellacdc import QtScoped
from cellacdc.acdc_regex import float_regex
from cellacdc.config import STANDARD_MOUSE_BUTTONS
from . import style as widgets_style
from . import push_buttons as widgets_push_buttons
from . import message_box as widgets_message_box
from . import basic_widgets as widgets_basic_widgets


class ValidLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)

    def setInvalidStyleSheet(self):
        if not hasattr(self, '_validStyleSheet'):
            self._validStyleSheet = self.styleSheet()
            
        self.setStyleSheet(widgets_style.LINEEDIT_INVALID_ENTRY_STYLESHEET)
    
    def setValidStyleSheet(self):
        if not hasattr(self, '_validStyleSheet'):
            self._validStyleSheet = self.styleSheet()

        self.setStyleSheet(self._validStyleSheet)


class ElidingLineEdit(ValidLineEdit):    
    def __init__(self, parent=None, minWidth=None):
        super().__init__(parent)
        self._text = ''
        self._minWidth = minWidth
        if minWidth is not None:
            self.setMinimumWidth(minWidth)
        
        self.textEdited.connect(self.setText)
        self.installEventFilter(self)
        self._elide = True
    
    def setText(self, text: str, width=None, elide=True) -> None:
        if width is None:
            width = self._minWidth
        
        if width is None:
            try:
                textToPrevRatio = len(text)/len(self.text())
                width = round(self.width()*textToPrevRatio)
            except ZeroDivisionError:
                width = self.width()

        if width > self.width():
            width = self.width()
            
        self._text = text
        if not elide or not self._elide:
            super().setText(text)
            return
        
        fm = QFontMetrics(self.font())
        elidedText = fm.elidedText(text, Qt.ElideLeft, width)
        
        super().setText(elidedText)
        self.setToolTip(text)
    
    def text(self):
        return self._text

    def resizeEvent(self, event):
        newWidth = event.size().width()
        self.setText(self._text, width=newWidth)
        event.accept()
    
    def eventFilter(self, a0: 'QObject', a1: 'QEvent') -> bool:
        isFocusIn = a1.type() == QtScoped.QEventTypeAttribute('FocusIn')
        if isFocusIn and (self.isReadOnly() or not self.isEnabled()):
            self.clearFocus()
            return True
        return super().eventFilter(a0, a1)
    
    def focusInEvent(self, event):
        super().focusInEvent(event)
        self._elide = False
        self.setText(self._text, elide=False)
        self.setCursorPosition(len(self.text()))

    def focusOutEvent(self, event):
        self._elide = True
        super().focusOutEvent(event)
        self.setText(self._text)


class filePathControl(QFrame):
    sigValueChanged = Signal(str)
    
    def __init__(
            self, parent=None, browseFolder=False, 
            fileManagerTitle='Select file', 
            validExtensions=None, 
            startFolder='', 
            elide=False,
            add_browse_recent=False,
            add_open_in_file_manager=False
        ):
        super().__init__(parent)

        layout = QHBoxLayout()
        if elide:
            self.le = ElidingLineEdit()
        else:
            self.le = ValidLineEdit()
            
        self.browseButton = widgets_push_buttons.browseFileButton(
            openFolder=browseFolder, 
            title=fileManagerTitle, 
            ext=validExtensions, 
            start_dir=startFolder,
            get_start_dir_func=self.le.text
        )

        layout.addWidget(self.le)
        layout.addWidget(self.browseButton)

        if add_browse_recent and browseFolder:
            recentPathsButton = widgets_push_buttons.PushButton()
            recentPathsButton.setIcon(QIcon(':open_file.svg'))
            recentPathsButton.setToolTip(
                'Select one existing folder from recent paths.'
            )
            layout.addWidget(recentPathsButton)
            self.recentPathsButton = recentPathsButton
            self.recentPathsButton.clicked.connect(self.browseRecentPaths)

        if add_open_in_file_manager:
            openInFileManagerButton = widgets_push_buttons.showInFileManagerButton()
            openInFileManagerButton.setToolTip(
                'Open the current path in file manager...'
            )
            layout.addWidget(openInFileManagerButton)
            self.openInFileManagerButton = openInFileManagerButton
            self.openInFileManagerButton.clicked.connect(self.showInFileManager)

        self.setLayout(layout)

        self.le.editingFinished.connect(self.setTextTooltip)
        self.browseButton.sigPathSelected.connect(self.setText)
    
        self.setFrameStyle(QFrame.Shape.StyledPanel)
    
    def showInFileManager(self):
        path = self.path()
        if not path or not os.path.exists(path):
            print(f'WARNING: {path} is not a valid path')
            msg = widgets_message_box.myMessageBox(wrapText=False, showCentered=False)
            txt = html_utils.paragraph(
                f'{path} is not a valid path :(.'
            )
            msg.warning(self, 'Not a valid path', txt)
            return

        if not os.path.isdir(path):
            path = os.path.dirname(path)
        
        myutils.showInExplorer(path)      

    def browseRecentPaths(self):
        recent_paths, dates = myutils.get_recent_paths(self)
        if not recent_paths:
            msg = widgets_message_box.myMessageBox(wrapText=False, showCentered=False)
            txt = html_utils.paragraph(
                'There are no existing recent paths :(.<br><br>'
            )
            msg.warning(self, 'No existing recent paths', txt)
            return
        
        try:
            self.recentPathsDialog.cancel = True
            self.recentPathsDialog.close()
        except Exception as err:
            pass

        self.recentPathsDialog = apps.SearchableListboxDialog(
            recent_paths,
            title='Select recent paths',
            searchLineEditText='Search recent paths...',
            parent=self,
            multiSelection=False,
            dates=dates,
        )
        self.recentPathsDialog.sigOk.connect(
            self.setFromRecentPath
        )
        self.recentPathsDialog.hide()
        self.recentPathsDialog.show()
        self.recentPathsDialog.raise_()
        self.recentPathsDialog.activateWindow()
    
    def setFromRecentPath(self, recent_path: os.PathLike):
        self.setText(recent_path)

    def setText(self, text):
        self.le.setText(text)
        self.le.setToolTip(text)
        self.sigValueChanged.emit(self.le.text())

    def setTextTooltip(self):
        self.le.setToolTip(self.le.text())
        self.sigValueChanged.emit(self.le.text())
    
    def path(self):
        return self.le.text()
    
    def showEvent(self, a0: QShowEvent) -> None:
        self.le.setFixedHeight(self.browseButton.height())
        return super().showEvent(a0)

    def setInvalid(self, invalid: bool):
        if invalid:
            self.le.setInvalidStyleSheet()
        else:
            self.le.setValidStyleSheet()

class FolderPathControl(filePathControl):
    def __init__(self, **kwargs):
        super().__init__(
            browseFolder=True, 
            **kwargs
        )

class CsvFilePathControl(filePathControl):
    def __init__(self, **kwargs):
        super().__init__(
            browseFolder=False,
            fileManagerTitle='Select a CSV file',
            validExtensions={'CSV files': ['.csv', '.CSV']},
            **kwargs
        )

    # Custom-widget protocol used by the automatic model parameters GUI,
    # where getter/setter may be called with the widget as first argument.
    def setValue(self, *args):
        self.setText(str(args[-1]))

    def value(self, *args):
        return self.path()

    @staticmethod
    def cast_dtype(value) -> str:
        return str(value)


class alphaNumericLineEdit(QLineEdit):
    sigInvalidCharacterPressed = Signal(str)
    sigInvalidCharactersEntered = Signal(object)
    
    def __init__(self, parent=None, additionalChars='', onlyWarn=False, 
                 formatter: callable = None, ignore_file_ext=False):
        super().__init__(parent)
        self.validPattern = fr'^[a-zA-Z0-9{additionalChars}_\-]+$'
        self.invalidPattern = fr'[^a-zA-Z0-9{additionalChars}_\-]'
        # Typed spaces become underscores only when spaces are not allowed
        self._replaceTypedSpace = (
            re.search(self.invalidPattern, ' ') is not None
        )
        self.formatter = formatter
        self.ignore_file_ext = ignore_file_ext

        if not onlyWarn:
            regExp = QRegularExpression(self.validPattern)
            self.setValidator(QRegularExpressionValidator(regExp))
        else:
            self.textChanged.connect(self.emitInvalidCharactersEntered)

    def emitInvalidCharactersEntered(self, text):
        invalidCharacters = self.invalidCharacters()
        if not invalidCharacters:
            return
        
        self.sigInvalidCharactersEntered.emit(set(invalidCharacters))
    
    def invalidCharacters(self):
        text = self.text()
        if self.formatter is not None:
            text = self.formatter(text)
        ext = self.ignore_file_ext
        if isinstance(ext, str) and ext and text.endswith(ext):
            text = text[:-len(ext)]
        return re.findall(fr'{self.invalidPattern}', text)
    
    def keyPressEvent(self, event: QKeyEvent):
        text = event.text()

        if not text:
            return super().keyPressEvent(event)

        if event.modifiers() & (
            Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.AltModifier
            | Qt.KeyboardModifier.MetaModifier
        ):
            return super().keyPressEvent(event)

        if not text.isprintable():
            return super().keyPressEvent(event)

        if text == ' ' and self._replaceTypedSpace:
            # replace the invalid space with an underscore before QLineEdit
            # processes it (spaces allowed by `additionalChars` are kept)
            event = QKeyEvent(
                event.type(),
                Qt.Key.Key_Underscore,
                event.modifiers(),
                '_',
            )

        super().keyPressEvent(event)

        if event.text() in self.text():
            return

        self.sigInvalidCharacterPressed.emit(event.text())


class FormLayout(QGridLayout):
    def __init__(self):
        QGridLayout.__init__(self)

    def addFormWidget(
            self, formWidget, 
            leftLabelAlignment=Qt.AlignRight, 
            align=None, 
            row=0
        ):
        for col, item in enumerate(formWidget.items):
            if col==0:
                alignment = leftLabelAlignment
            elif col==2:
                alignment = Qt.AlignLeft
            else:
                alignment = align
            try:
                if alignment is None:
                    self.addWidget(item, row, col)
                else:
                    self.addWidget(item, row, col, alignment=alignment)
            except TypeError:
                self.addLayout(item, row, col)
        
        self.setColumnStretch(0, 0)
        self.setColumnStretch(1, 1)

def macShortcutToWindows(shortcut: str):
    if shortcut is None:
        return
    
    s = (shortcut
        .replace('Control', 'Meta')
        .replace('Option', 'Alt')
        .replace('Command', 'Ctrl')
    )
    return s

def windowsShortcutToMac(shortcut: str):
    if shortcut is None:
        return
    
    if not is_mac:
        return shortcut
    
    s = (shortcut
        .replace('Meta', 'Control')
        .replace('Alt', 'Option')
        .replace('Ctrl', 'Command')
    )
    return s


def QKeyEventToString(event: QKeyEvent, notAllowedModifier=None):
    isAltKey = event.key()==Qt.Key_Alt
    isCtrlKey = event.key()==Qt.Key_Control
    isShiftKey = event.key()==Qt.Key_Shift
    isModifierKey = isAltKey or isCtrlKey or isShiftKey
    
    modifiers = event.modifiers()
    isNotAllowedMod = (
        notAllowedModifier is not None and modifiers == notAllowedModifier
    )
    if isNotAllowedMod:
        return 
    
    modifers_value = modifiers.value if PYQT6 else modifiers
    if isModifierKey:
        keySequenceText = KeySequenceFromText(modifers_value).toString()
    else:
        keySequenceText = QKeySequence(modifers_value | event.key()).toString()
    
    keySequenceText = keySequenceText.encode('ascii', 'ignore').decode('utf-8')
    
    return keySequenceText

class ShortcutLineEdit(QLineEdit):
    clicked = Signal()
    def __init__(
            self, parent=None, allowModifiers=False, notAllowedModifier=None,
            allowMouseButtons=False
        ):
        self.keySequence = None
        super().__init__(parent)
        self._allowModifiers = allowModifiers
        self._notAllowedModifier = notAllowedModifier
        self.setAlignment(Qt.AlignCenter)
        self._allowMouseButtons = allowMouseButtons
        
    def eventFilter(self, obj, event):
        if event.type() == QtScoped.QEventTypeAttribute('MouseButtonPress'):
            button = event.button()
            if (
                    self._allowMouseButtons 
                    and button not in STANDARD_MOUSE_BUTTONS
                ):
                btn_name = QtScoped.mouse_button_name(button)
                self.setText(f'Mouse {btn_name}')
                return True  
            
        return super().eventFilter(obj, event) 
    
    def text(self):
        text = macShortcutToWindows(super().text())
        
        return text
    
    def setText(self, text):
        text = windowsShortcutToMac(text)
        
        super().setText(text)
        if not text:
            self.keySequence = None
            return
        if 'Mouse ' in text:
            self.keySequence = None
            return
        try:
            self.keySequence = KeySequenceFromText(self.text())
        except Exception as e:
            pass

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Backspace or event.key() == Qt.Key_Delete:
            self.setText('')
            return

        keySequenceText = QKeyEventToString(
            event, notAllowedModifier=self._notAllowedModifier
        )
        self.setText(keySequenceText)
        self.key = event.key()
    
    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if self.text().endswith('+'):
            if not self._allowModifiers:
                self.setText('')
            else:
                self.setText(self.text().rstrip('+').strip())
                
    # catch mouse press events
    def mousePressEvent(self, event):
        button = event.button()

        if button in STANDARD_MOUSE_BUTTONS or not self._allowMouseButtons:
            super().mousePressEvent(event)
        else:
            # covers XButton1, XButton2, and any further extra buttons
            btn_name = QtScoped.mouse_button_name(button)
            self.setText(f'Mouse {btn_name}')
            
    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.clicked.emit()  # or call your filter-install logic directly

    # def focusOutEvent(self, event):
    #     self.editingFinished.emit()  # or call your filter-release logic directly
    #     super().focusOutEvent(event)
        
    def sizeHint(self):
        size_hint = super().sizeHint()
        width = int(size_hint.width() * 1.5)
        size_hint.setWidth(width)
        return size_hint


class formWidget(QWidget):
    sigApplyButtonClicked = Signal(object)
    sigComputeButtonClicked = Signal(object)
    sigCogButtonClicked = Signal(object)

    def __init__(
            self, widget,
            initialVal=None,
            stretchWidget=True,
            widgetAlignment=None,
            labelTextLeft='',
            labelTextRight='',
            font=None,
            addInfoButton=False,
            addApplyButton=False,
            addComputeButton=False,
            addActivateCheckbox=False,
            addCogButton=False,
            key='',
            infoTxt='',
            valueGetterName='value',
            toolTip='',
            wrapInfoTxt=False,
            labelLeftSuffix='',
            parent=None
        ):
        QWidget.__init__(self, parent)
        self.widget = widget
        self.key = key
        self.infoTxt = infoTxt
        self.wrapInfoTxt = wrapInfoTxt
        self.widgetAlignment = widgetAlignment
        self.valueGetterName = valueGetterName

        widget.setParent(self)

        if isinstance(initialVal, bool):
            widget.setChecked(initialVal)
        elif isinstance(initialVal, str):
            widget.setCurrentText(initialVal)
        elif isinstance(initialVal, float) or isinstance(initialVal, int):
            widget.setValue(initialVal)

        self.items = []

        if font is None:
            font = QFont()
            font.setPixelSize(13)

        self.labelTextLeft = labelTextLeft
        self.labelLeft = widgets_basic_widgets.QClickableLabel(widget)
        self.labelLeft.setText(f'{labelTextLeft}{labelLeftSuffix}')
        self.labelLeft.setFont(font)
        self.items.append(self.labelLeft)

        if not stretchWidget:
            widgetLayout = QHBoxLayout()
            if widgetAlignment != 'left':
                widgetLayout.addStretch(1)
            widgetLayout.addWidget(widget)
            if widgetAlignment != 'right':
                widgetLayout.addStretch(1)
            self.items.append(widgetLayout)
        else:
            self.items.append(widget)

        self.labelRight = widgets_basic_widgets.QClickableLabel(widget)
        self.labelRight.setText(labelTextRight)
        self.labelRight.setFont(font)
        self.items.append(self.labelRight)
        
        if toolTip:
            self.labelLeft.setToolTip(toolTip)
            self.widget.setToolTip(toolTip)
            self.labelRight.setToolTip(toolTip)

        if addInfoButton:
            infoButton = QPushButton(self)
            infoButton.setCursor(Qt.WhatsThisCursor)
            infoButton.setIcon(QIcon(":info.svg"))
            if labelTextLeft:
                infoButton.setToolTip(
                    f'Info about "{self.labelLeft.text()}" parameter'
                )
            else:
                infoButton.setToolTip(
                    f'Info about "{self.labelRight.text()}" measurement'
                )
            infoButton.clicked.connect(self.showInfo)
            self.infoButton = infoButton
            self.items.append(infoButton)

        if addApplyButton:
            applyButton = QPushButton(self)
            applyButton.setCursor(Qt.PointingHandCursor)
            applyButton.setCheckable(True)
            applyButton.setIcon(QIcon(":apply.svg"))
            applyButton.setToolTip(f'Apply this step and visualize results')
            applyButton.clicked.connect(self.applyButtonClicked)
            self.items.append(applyButton)

        if addComputeButton:
            computeButton = QPushButton(self)
            computeButton.setCursor(Qt.BusyCursor)
            computeButton.setIcon(QIcon(":compute.svg"))
            computeButton.setToolTip(f'Compute this step and visualize results')
            computeButton.clicked.connect(self.computeButtonClicked)
            self.items.append(computeButton)
        
        if addCogButton:
            cogButton = widgets_push_buttons.setPushButton(self)
            cogButton.clicked.connect(self.cogButtonClicked)
            self.cogButton = cogButton
            self.items.append(cogButton)

        self.activateCheckbox = None
        if addActivateCheckbox:
            self.activateCheckbox = QCheckBox('Activate')
            self.activateCheckbox.setChecked(False)
            self.widget.setDisabled(True)
            self.activateCheckbox.toggled.connect(self.setWidgetEnabled)
            self.items.append(self.activateCheckbox)

        self.labelLeft.clicked.connect(self.tryChecking)
        self.labelRight.clicked.connect(self.tryChecking)
    
    def setWidgetEnabled(self, checked):
        self.widget.setDisabled(not checked)
    
    def value(self):
        if self.activateCheckbox is None:
            return getattr(self.widget, self.valueGetterName)()
        
        if not self.activateCheckbox.isChecked():
            return
        
        return getattr(self.widget, self.valueGetterName)()

    def tryChecking(self, label):
        try:
            self.widget.setChecked(not self.widget.isChecked())
        except AttributeError as e:
            pass

    def applyButtonClicked(self):
        self.sigApplyButtonClicked.emit(self)

    def computeButtonClicked(self):
        self.sigComputeButtonClicked.emit(self)
    
    def cogButtonClicked(self):
        self.sigCogButtonClicked.emit(self)

    def showInfo(self):
        msg = widgets_message_box.myMessageBox(wrapText=self.wrapInfoTxt)
        msg.setIcon()
        msg.setWindowTitle(f'{self.labelLeft.text()} info')
        msg.addText(self.infoTxt)
        msg.addButton('   Ok   ')
        msg.exec_()
    
    def setDisabled(self, disabled: bool) -> None:
        for item in self.items:
            if isinstance(item, QHBoxLayout):
                for i in range(item.count()):
                    widget = item.itemAt(i).widget()
                    try:
                        widget.setDisabled(disabled)
                    except Exception as err:
                        pass
            else:
                try:
                    item.setDisabled(disabled)
                except Exception as err:
                    pass


class FloatLineEdit(QLineEdit):
    valueChanged = Signal(float)

    def __init__(
            self, *args, notAllowed=None, allowNegative=True, initial=None,
            readOnly=False, decimals=6, warningValues=None
        ):
        QLineEdit.__init__(self, *args)
        if readOnly:
            self.setReadOnly(readOnly)
        self.notAllowed = notAllowed
        self.warningValues = warningValues
        self._maximum = np.inf
        self._minimum = -np.inf
        self._decimals = decimals

        self.isNumericRegExp = rf'^{float_regex(allow_negative=allowNegative)}$'
        regExp = QRegularExpression(self.isNumericRegExp)
        self.setValidator(QRegularExpressionValidator(regExp))
        self.setAlignment(Qt.AlignCenter)

        font = QFont()
        font.setPixelSize(11)
        self.setFont(font)

        self.textChanged.connect(self.emitValueChanged)
        
        if initial is not None:
            self.setValue(initial)
        else:
            self.setValue(0)  
    
    def setDecimals(self, decimals):
        self._decimals = decimals

    def castMinMax(self, value: int):
        if value > self._maximum:
            value = self._maximum
        if value < self._minimum:
            value = self._minimum
        return value
    
    def setValue(self, value: float):
        value = self.castMinMax(value)
        self.setText(str(round(value, self._decimals)))

    def value(self):
        m = re.match(self.isNumericRegExp, self.text())
        if m is not None:
            text = m.group(0)
            try:
                val = float(text)
            except ValueError:
                val = 0.0
        else:
            val = 0.0
        
        return self.castMinMax(val)
    
    def setMaximum(self, maximum):
        self._maximum = maximum
        self.setValue(self.value())
    
    def setMinimum(self, minimum):
        self._minimum = minimum
        self.setValue(self.value())

    def emitValueChanged(self, text):
        val = self.value()
        reset_stylesheet = True
        if self.warningValues is not None and val in self.warningValues:
            self.setStyleSheet(widgets_style.LINEEDIT_WARNING_STYLESHEET)
            reset_stylesheet = False
        
        if self.notAllowed is not None and val in self.notAllowed:
            self.setStyleSheet(widgets_style.LINEEDIT_INVALID_ENTRY_STYLESHEET)
            reset_stylesheet = False
        else:
            self.valueChanged.emit(self.value())
        
        if reset_stylesheet:
            self.setStyleSheet('')

class IntLineEdit(QLineEdit):
    valueChanged = Signal(float)

    def __init__(
            self, *args, notAllowed=None, allowNegative=True, initial=None,
            readOnly=False, maximum=None, minimum=None
        ):
        QLineEdit.__init__(self, *args)
        self.notAllowed = notAllowed
        if readOnly:
            self.setReadOnly(readOnly)

        maximum = maximum if maximum is not None else np.inf
        minimum = minimum if minimum is not None else -np.inf
        self._maximum = maximum
        self._minimum = minimum
        
        self._regExp = r'\d+'
        if allowNegative:
            self._regExp = r'-?\d+'

        regExp = QRegularExpression(self._regExp)
        self.setValidator(QRegularExpressionValidator(regExp))
        self.setAlignment(Qt.AlignCenter)

        font = QFont()
        font.setPixelSize(11)
        self.setFont(font)

        self.textChanged.connect(self.emitValueChanged)
        
        if initial is not None:
            self.setValue(initial)
        else:
            self.setValue(0)    
    
    def setMaximum(self, maximum):
        self._maximum = maximum
        self.setValue(self.value())
    
    def setMinimum(self, minimum):
        self._minimum = minimum
        self.setValue(self.value())

    def castMinMax(self, value: int):
        if value > self._maximum:
            value = self._maximum
        if value < self._minimum:
            value = self._minimum
        return value
    
    def setValue(self, value: int):
        value = self.castMinMax(value)
        self.setText(str(value))

    def value(self):
        m = re.match(self._regExp, self.text())
        if m is not None:
            text = m.group(0)
            try:
                val = int(text)
            except ValueError:
                val = 0
        else:
            val = 0
        
        return self.castMinMax(val)

    def emitValueChanged(self, text):
        if not text:
            return
        
        val = self.value()
        self.setValue(val)
        if self.notAllowed is not None and val in self.notAllowed:
            self.setStyleSheet(widgets_style.LINEEDIT_INVALID_ENTRY_STYLESHEET)
        else:
            self.setStyleSheet('')
            self.valueChanged.emit(self.value())


class VectorLineEdit(QLineEdit):
    valueChanged = Signal(object)
    valueChangeFinished = Signal(object)
    
    def __init__(self, parent=None, initial=None):
        super().__init__(parent)
        
        self._minimum = -np.inf
        
        float_re = float_regex()
        vector_regex = fr'\(?\[?{float_re}(,\s?{float_re})+\)?\]?'
        regex = fr'^{vector_regex}$|^{float_re}$'
        self.validRegex = regex
        
        regExp = QRegularExpression(regex)
        self.setValidator(QRegularExpressionValidator(regExp))
        self.setAlignment(Qt.AlignCenter)
        
        self.textChanged.connect(self.emitValueChanged)
        self.editingFinished.connect(self.emitValueChangeFinished)
        if initial is None:
            self.setText('0.0')
        
        font = QFont()
        font.setPixelSize(11)
        self.setFont(font)
    
    def emitValueChangeFinished(self):
        value = self.value()
        self.textChanged.disconnect()
        self.editingFinished.disconnect()
        self.setValue(value)
        self.textChanged.connect(self.emitValueChanged)
        self.editingFinished.connect(self.emitValueChangeFinished)
        
        self.emitValueChanged(self.text(), signal=self.valueChangeFinished)
        
    def emitValueChanged(self, text, signal=None):
        m = re.match(self.validRegex, text)
        if m is None:
            self.setStyleSheet(widgets_style.LINEEDIT_INVALID_ENTRY_STYLESHEET)
            return

        if signal is None:
            signal = self.valueChanged
        
        self.setStyleSheet('')
        signal.emit(self.value())
    
    def increaseValue(self, step):
        value = self.value()
        if isinstance(value, (float, int)):
            value += step
        else:
            value = [val+step for val in value]
            value = str(value).lstrip('[').rstrip(']')
        self.setValue(value)
        self.emitValueChangeFinished()
    
    def decreaseValue(self, step):
        value = self.value()
        if isinstance(value, (float, int)):
            value -= step
        else:
            value = [val-step for val in value]
            value = str(value).lstrip('[').rstrip(']')
        self.setText(value)
        self.emitValueChangeFinished()
    
    def setValue(self, value):
        if isinstance(value, (float, int)):
            if value < self._minimum:
                value = self._minimum
        else:
            clipped = []
            for val in value:
                if val < self._minimum:
                    val = self._minimum
                clipped.append(val)
            value = str(clipped).lstrip('[').rstrip(']')
        self.setText(value)
    
    def setText(self, text):
        super().setText(str(text))
    
    def clipValue(self, val: float):
        if val < self._minimum:
            val = self._minimum
        return val
    
    def value(self):
        m = re.match(self.validRegex, self.text())
        if m is None:
            return 0.0
        
        try: 
            value = self.clipValue(float(self.text()))
            return value
        except Exception as e:
            text = self.text()
            text = text.replace('(', '')
            text = text.replace(')', '')
            text = text.replace('[', '')
            text = text.replace(']', '')
            values = text.split(',')
            return [self.clipValue(float(value)) for value in values]
    
    def setMinimum(self, minimum):
        self._minimum = float(minimum)


class LineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
    
    def value(self):
        return self.text()

    def setValue(self, value):
        self.setText(str(value))


class KeySequenceFromText(QKeySequence):
    def __init__(self, text: str):
        if isinstance(text, str):
            text = macShortcutToWindows(text)
        super().__init__(text)
        self._text = text
    
    def toString(self):
        if isinstance(self._text, str):
            return windowsShortcutToMac(self._text)
        else:
            return windowsShortcutToMac(super().toString())

def modifierKeyToText(modifierKey: int):
    if modifierKey == Qt.ControlModifier:
        return 'Ctrl'
    elif modifierKey == Qt.AltModifier:
        return 'Alt'
    elif modifierKey == Qt.ShiftModifier:
        return 'Shift'
    elif modifierKey == Qt.MetaModifier:
        return 'Meta'
    else:
        return ''
