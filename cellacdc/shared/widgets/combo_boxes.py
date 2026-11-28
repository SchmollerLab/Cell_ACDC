"""Combo boxes: ComboBox (sigTextChanged and previousText(), used by the
mode selector of the GUI), QCenteredComboBox, AlphaNumericComboBox,
TypableComboBox, font size selection (FontSizeCombobox, FontSizeQMenu),
scatter symbol pickers (pgScatterSymbolsCombobox,
VisPyMarkersSymbolsCombobox), PreProcessingSelector (choice and
parameters of preprocessing recipe steps) and
YeazV2SelectModelNameCombobox.
"""
from typing import Dict, get_args
from qtpy.QtCore import Signal, Qt
from qtpy.QtGui import QIntValidator
from qtpy.QtWidgets import QApplication, QComboBox, QWidgetAction, QMenu
import qtpy.compat
from cellacdc import myutils, html_utils
from cellacdc import apps
from cellacdc import plot
from cellacdc import urls
from cellacdc import QtScoped
from cellacdc.config import PREPROCESS_MAPPER


class QCenteredComboBox(QComboBox):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setAlignment(Qt.AlignCenter)
        self.lineEdit().installEventFilter(self)

        self.currentIndexChanged.connect(self.centerItems)

        self._isPopupVisibile = False
    
    def centerItems(self, idx):
        for i in range(self.count()):
            self.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)
    
    def eventFilter(self, lineEdit, event):
        # Reimplement show popup on click
        if event.type() == QtScoped.QEventTypeAttribute('MouseButtonPress') and self.isEnabled():
            if self._isPopupVisibile:
                self.hidePopup()
                self._isPopupVisibile = False
            else:
                self.showPopup()
                self._isPopupVisibile = True
            return True
        return False

class AlphaNumericComboBox(QCenteredComboBox):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
    
    def addItems(self, items):
        self._dtype = type(items[0])
        super().addItems([str(item) for item in items])
    
    def setCurrentValue(self, value):
        super().setCurrentText(str(value))
    
    def currentValue(self):
        return self._dtype(super().currentText())


class pgScatterSymbolsCombobox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)

        symbols = [
            "'o' circle (default)",
            "'s' square",
            "'t' triangle",
            "'d' diamond",
            "'+' plus",
            "'t1' triangle pointing upwards",
            "'t2' triangle pointing right side",
            "'t3' triangle pointing left side",
            "'p' pentagon",
            "'h' hexagon",
            "'star'",
            "'x' cross",
            "'arrow_up'",
            "'arrow_right'",
            "'arrow_down'",
            "'arrow_left'",
            "'crosshair'"
        ]
        self.addItems(symbols)

class VisPyMarkersSymbolsCombobox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)

        symbols = get_args(plot.VisPyMarkerSymbols)
        self.addItems(symbols)


class ComboBox(QComboBox):
    sigTextChanged = Signal(str)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._previousText = None
        self._valueChanged = False
        self.currentTextChanged.connect(self.emitTextChanged)
        self.installEventFilter(self)
    
    def eventFilter(self, object, event) -> bool:
        if object == self and event.type() == QtScoped.QEventTypeAttribute('Wheel'):
            # Forward event to parent so QScrollArea can scroll
            QApplication.sendEvent(self.parent(), event)
            return True  # Consume for the combo itself
        
        return super().eventFilter(object, event)
    
    def text(self):
        return self.currentText()
    
    def emitTextChanged(self, text):
        self._valueChanged = True
        self.sigTextChanged.emit(text)
    
    def mousePressEvent(self, event):
        self._previousText = self.currentText()
        super().mousePressEvent(event)
    
    def previousText(self):
        return self._previousText

    def addItems(self, items):
        super().addItems(items)
        self._previousText = items[0]
    
    def itemsText(self):
        return [self.itemText(i) for i in range(self.count())]
    
    def setCurrentIndex(self, idx):
        itemsText = self.itemsText()
        currentText = itemsText[idx]
        self._valueChanged = currentText != self._previousText
        self._previousText = self.currentText()
        super().setCurrentIndex(idx)
    
    def setCurrentText(self, text):
        currentText = text
        self._valueChanged = currentText != self._previousText
        self._previousText = self.currentText()
        super().setCurrentText(text)


class PreProcessingSelector(QComboBox):
    sigValuesChanged = Signal(dict, int)
    preprocessMapper = PREPROCESS_MAPPER
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent = parent
        
        self.addItems(self.preprocessMapper.keys())
        self.methodToDefaultValuesMapper = {}
        self.step_n = -1
        self.setParamsWindow = None

    def htmlInfo(self):
        href = html_utils.href_tag('GitHub page', urls.issues_url)
        docstring = self.preprocessMapper[self.currentText()]['docstring']
        if docstring is None:
            text = 'This function is not documented, yet. Sorry :('
        else:
            text = html_utils.rst_docstring_to_html(docstring)
        text = (
            f'{text}<br><br>'
            f'Feel free to submit an issue on our {href} if you '
            'need help with this filter.'
        )
        return text
        
    def setParams(self, method: str, kwargToValueMapper: Dict[str, str]):
        self.methodToDefaultValuesMapper[method] = kwargToValueMapper
    
    def askSetParams(self, df_metadata=None, addApplyButton=False):
        method = self.currentText()
        function = self.preprocessMapper[method]['function']
        params_argspecs = myutils.get_function_argspec(
            function, 
            args_to_skip={
                'logger_func', 
                'apply_to_all_zslices',
                'apply_to_all_frames'
            }
        )
        default_values = self.methodToDefaultValuesMapper.get(method, {})
        for kwarg, value in default_values.items():
            for p, param_argspec in enumerate(params_argspecs):
                if param_argspec.name != kwarg:
                    continue
                
                if hasattr(param_argspec.type, 'cast_dtype'):
                    cls = param_argspec.type
                    value = cls.cast_dtype(value)
                else:
                    value = param_argspec.type(value)
                
                if value == param_argspec.default:
                    continue
                param_argspec = param_argspec._replace(default=value)
                params_argspecs[p] = param_argspec
        
        if self.setParamsWindow is not None:
            self.setParamsWindow.raise_()
            self.setParamsWindow.activateWindow()
            return
        
        self.setParamsWindow = apps.FunctionParamsDialog(
            params_argspecs, 
            df_metadata=df_metadata,
            function_name=method,
            addApplyButton=addApplyButton,
            parent=self._parent
        )
        self.setParamsWindow.sigValuesChanged.connect(self.emitValuesChanged)
        self.setParamsWindow.emitValuesChanged()
        self.setParamsWindow.exec_()
        if self.setParamsWindow.cancel:
            # Every edit was already emitted live (step kwargs and preview):
            # Cancel (or closing the window) restores the opening values.
            if self.currentText() == method:
                self.emitValuesChanged(
                    self.setParamsWindow.original_function_kwargs
                )
            self.setParamsWindow = None
            return
        
        self.setParams(method, self.setParamsWindow.function_kwargs)
        
        function_kwargs = self.setParamsWindow.function_kwargs
        self.setParamsWindow = None
        
        return function_kwargs

    def emitValuesChanged(self, functionKwargs: dict):
        self.sigValuesChanged.emit(functionKwargs, self.step_n)


class YeazV2SelectModelNameCombobox(ComboBox):
    sigValueChanged = Signal(str)
    
    def __init__(
            self, *args, 
            custom_select_item_text='Select custom weights file...', 
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self._csi_text = custom_select_item_text
        self.sigTextChanged.connect(self.onTextChanged)
        self.initItems()
    
    def initItems(self):
        from cellacdc.models._yeaz_v2_weights import load_models_filepath
        models_name, models_name_filepath_mapper = load_models_filepath()
        self.addItems(models_name)
    
    def onTextChanged(self, text):
        if text != self._csi_text:
            return
        
        start_dir = myutils.getMostRecentPath()
        model_filepath = qtpy.compat.getopenfilename(
            parent=self, 
            caption='Select YeaZ weights file', 
            filters='All Files (*)',
            basedir=start_dir
        )[0]
        if not model_filepath:
            self.setCurrentIndex(0)
            return
        
        msg = html_utils.paragraph(f"""
        Insert a <b>name</b> for the following YeaZ model:<br><br>
        <code>{model_filepath}</code><br>
        """)
        modelNameWindow = apps.QLineEditDialog(
            title='Insert a name for the model',
            msg=msg,
            allowEmpty=False,
            parent=self
        )
        modelNameWindow.exec_()
        if modelNameWindow.cancel:
            self.setCurrentIndex(0)
            return

        model_name = modelNameWindow.enteredValue
        
        from cellacdc.models._yeaz_v2_weights import add_model_filepath
        add_model_filepath(model_name, model_filepath)
        
        self.addItem(model_name)
        self.setCurrentText(model_name)
        
        print(
            'YeaZ_v2 model added!\n\n'
            f'  * Name: {model_name}\n'
            f'  * File path: {model_filepath}\n'
        )
    
    def addItem(self, item):
        idx = self.count() - 1
        self.insertItem(idx, item)
    
    def addItems(self, items):
        super().clear()
        super().addItems(items)
        super().addItem(self._csi_text)
        idx = len(items)
        font = self.font()
        font.setItalic(True)
        self.setItemData(idx, font, Qt.FontRole)
    
    def setValue(self, value: str):
        self.setCurrentText(value)
    
    def value(self, *args):
        return self.currentText()


class TypableComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.NoInsert)
        self.lineEdit().editingFinished.connect(self.commit_text)

    def commit_text(self):
        text = self.currentText()
        index = self.findText(text, Qt.MatchExactly)

        if index < 0:
            self.addItem(text)
            index = self.count() - 1

        self.setCurrentIndex(index)

class FontSizeCombobox(TypableComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.addItems([
            *[str(i) for i in range(8, 13)],
            *[str(i*2) for i in range(6, 16)],
            *['36', '48', '72']
        ])

        self.lineEdit().setValidator(QIntValidator(1, 9999, self.lineEdit()))

class FontSizeQMenu(QMenu):
    sigFontSizeChanged = Signal(int)

    def __init__(self, prefix='', parent=None):
        super().__init__(parent)

        title = f'{prefix}font size'.capitalize()
        self.setTitle(title)

        self.fontSizeCombobox = FontSizeCombobox(self)
        self.fontSizeAction = QWidgetAction(self)
        self.fontSizeAction.setDefaultWidget(self.fontSizeCombobox)
        self.addAction(self.fontSizeAction)

        self.fontSizeCombobox.currentTextChanged.connect(
            self.emitSigFontSizeChanged
        )
    
    def emitSigFontSizeChanged(self, font_size_str):
        try:
            font_size = int(font_size_str)
        except (TypeError, ValueError):
            return

        self.sigFontSizeChanged.emit(font_size)

    def setFontSize(self, font_size: int):
        text = str(int(font_size))
        combo = self.fontSizeCombobox

        index = combo.findText(text)
        if index < 0:
            combo.addItem(text)
            index = combo.count() - 1

        combo.setCurrentIndex(index)
