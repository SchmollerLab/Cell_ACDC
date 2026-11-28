"""Preprocessing and function-parameter dialogs (from apps.py).

ArgWidget, FunctionParamsDialog (emits every edit live), the recipe
widgets and dialogs PreProcessParamsWidget (also the base of the
viewer's measurement processing parameters), PreProcessRecipeDialog
(main GUI) and PreProcessRecipeDialogUtil (utilities, viewer), and the
combine-channels setup base CombineChannelsSetupDialog with
CombineChannelsWidget (subclassed in gui/dialogs/segmentation.py and
utils/combineChannels.py).
"""
import os
import re
from typing import Dict, Iterable
import datetime
import numpy as np
import traceback
from functools import partial
import json
from qtpy.QtGui import QColor
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QMenu, QLabel, QWidget, QVBoxLayout, QLineEdit, QPushButton, QHBoxLayout,
    QButtonGroup, QCheckBox, QComboBox, QGridLayout, QSpinBox, QDoubleSpinBox,
    QGroupBox
)
from cellacdc import html_utils
from cellacdc import config
from cellacdc import preproc_recipes_path, combine_channels_recipes_path
from cellacdc import printl
from cellacdc.myutils import filesystem
from cellacdc import qutils
from ..widgets import (
    basic_widgets as widgets_basic_widgets,
    combo_boxes as widgets_combo_boxes,
    item_views as widgets_item_views,
    line_edits as widgets_line_edits,
    message_box as widgets_message_box,
    numeric_inputs as widgets_numeric_inputs,
    push_buttons as widgets_push_buttons,
)
from cellacdc import _types
from . import base as dialogs_base
from . import file_selection as dialogs_file_selection
from . import generic as dialogs_generic
from . import combine_measurements as dialogs_combine_measurements


class ArgWidget:
    def __init__(self, name, type, widget, defaultVal, valueSetter, valueGetter, changeSig=None):
        self.name = name
        self.type = type
        self.widget = widget
        self.defaultVal = defaultVal
        self.valueSetter = valueSetter
        self.valueGetter = valueGetter
        if changeSig is not None:
            self.changeSig = changeSig


class FunctionParamsDialog(dialogs_base.QBaseDialog):
    sigValuesChanged = Signal(dict)
    
    def __init__(
            self, params_argspecs, 
            function_name='Function', 
            df_metadata=None,
            parent=None,
            addApplyButton=False
        ):
        self.cancel = True
        self.df_metadata = df_metadata
        
        super().__init__(parent)
        
        self.setWindowTitle(f'{function_name} parameters')
        
        self.mainLayout = QVBoxLayout()
        
        widgetsLayout, self.argsWidgets = self.getWidgetsLayout(params_argspecs)
        
        buttonsLayout = widgets_push_buttons.CancelOkButtonsLayout()

        self.buttonsLayout = buttonsLayout
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.cancel_cb)
        
        if addApplyButton:
            applyButton = widgets_push_buttons.viewPushButton('Apply')
            applyButton.clicked.connect(self.emitValuesChanged)
            buttonsLayout.insertWidget(3, applyButton)
            self.applyButton = applyButton
        
        self.mainLayout.addLayout(widgetsLayout)
        self.mainLayout.addSpacing(20)
        self.mainLayout.addLayout(buttonsLayout)

        self.setLayout(self.mainLayout)
        
        # original value for cancel
        self.original_function_kwargs = self.functionKwargs()

    def cancel_cb(self):
        # Edits are emitted live (`sigValuesChanged`): the caller restores
        # `original_function_kwargs` on cancel (see
        # `widgets_combo_boxes.PreProcessingSelector.askSetParams`)
        self.cancel = True
        self.function_kwargs = self.original_function_kwargs
        self.close()
    
    def emitValuesChanged(self, *args, **kwargs):
        self.sigValuesChanged.emit(self.functionKwargs())
    
    def functionKwargs(self):        
        function_kwargs = {
            argWidget.name:argWidget.valueGetter(argWidget.widget)
            for argWidget in self.argsWidgets
        }
        return function_kwargs
    
    def kwargWidgetMapper(self) -> Dict[str, tuple]:
        kwarg_widget_mapper = {
            argWidget.name:(argWidget.widget, argWidget.valueSetter)
            for argWidget in self.argsWidgets
        }
        return kwarg_widget_mapper
    
    def ok_cb(self):
        self.cancel = False
        
        self.function_kwargs = self.functionKwargs()
        
        self.close()
    
    def getValueFromMetadata(self, name):
        try:
            value = self.df_metadata.at[name, 'values']
        except Exception as e:
            # traceback.print_exc()
            value = None
        return value
    
    def getWidgetsLayout(self, params_argspecs):
        widgetsLayout = QGridLayout()
        ArgsWidgets_list = []
        
        for row, ArgSpec in enumerate(params_argspecs):
            if _types.is_widget_not_required(ArgSpec):
                continue
            
            arg_name = ArgSpec.name         
            var_name = arg_name.replace('_', ' ')
            var_name = f'{var_name[0].upper()}{var_name[1:]}'
            label = QLabel(f'{var_name}:  ')
            metadata_val = self.getValueFromMetadata(ArgSpec.name)
            widgetsLayout.addWidget(label, row, 0, alignment=Qt.AlignLeft)
            try:
                values = ArgSpec.type().values
                isCustomListType = True
            except Exception as err:
                isCustomListType = False
            
            isVectorEntry = False
            try:
                if isinstance(ArgSpec.type(), _types.Vector):
                    isVectorEntry = True
            except Exception as err:
                pass
            
            isFolderPath = False
            try:
                if isinstance(ArgSpec.type(), _types.FolderPath):
                    isFolderPath = True
            except Exception as err:
                pass
            
            isCustomWidget = hasattr(ArgSpec.type, 'isWidget')
            
            if isCustomWidget:
                widget = ArgSpec.type().widget
                self.checkIfTypeCLassHasCastDtype(widget)
                defaultVal = ArgSpec.default
                valueSetter = widget.setValue
                valueGetter = widget.value
                widgetsLayout.addWidget(widget, row, 1, 1, 2)
                try:
                    widget.sigValueChanged.connect(self.emitValuesChanged)
                except Exception as err:
                    pass
            elif isVectorEntry:
                vectorLineEdit = widgets_line_edits.VectorLineEdit()
                self.checkIfTypeCLassHasCastDtype(ArgSpec.type)
                vectorLineEdit.setValue(ArgSpec.default)
                defaultVal = ArgSpec.default
                valueSetter = widgets_line_edits.VectorLineEdit.setValue
                valueGetter = widgets_line_edits.VectorLineEdit.value
                widget = vectorLineEdit
                widgetsLayout.addWidget(vectorLineEdit, row, 1, 1, 2)
                widget.valueChangeFinished.connect(self.emitValuesChanged)
            elif isFolderPath:
                folderPathControl = widgets_line_edits.FolderPathControl()
                self.checkIfTypeCLassHasCastDtype(ArgSpec.type)
                folderPathControl.setText(str(ArgSpec.default))
                widget = folderPathControl
                defaultVal = str(ArgSpec.default)
                valueSetter = widgets_line_edits.FolderPathControl.setText
                valueGetter = widgets_line_edits.FolderPathControl.path
                widgetsLayout.addWidget(folderPathControl, row, 1, 1, 2)
                widget.sigValueChanged.connect(self.emitValuesChanged)
            elif ArgSpec.type == bool:
                booleanGroup = QButtonGroup()
                booleanGroup.setExclusive(True)
                checkBox = widgets_push_buttons.Toggle()
                checkBox.setChecked(ArgSpec.default)
                defaultVal = ArgSpec.default
                valueSetter = widgets_push_buttons.Toggle.setChecked
                valueGetter = widgets_push_buttons.Toggle.isChecked
                widget = checkBox
                widgetsLayout.addWidget(
                    checkBox, row, 1, 1, 2, alignment=Qt.AlignCenter
                )
                widget.toggled.connect(self.emitValuesChanged)
            elif ArgSpec.type == int:
                spinBox = widgets_numeric_inputs.SpinBox()
                if metadata_val is None:
                    spinBox.setValue(ArgSpec.default)
                else:
                    spinBox.setValue(int(metadata_val))
                    spinBox.isMetadataValue = True
                defaultVal = ArgSpec.default
                valueSetter = QSpinBox.setValue
                valueGetter = QSpinBox.value
                widget = spinBox
                widgetsLayout.addWidget(spinBox, row, 1, 1, 2)
                widget.sigValueChanged.connect(self.emitValuesChanged)
            elif ArgSpec.type == float:
                doubleSpinBox = widgets_line_edits.FloatLineEdit()
                if metadata_val is None:
                    doubleSpinBox.setValue(ArgSpec.default)
                else:
                    doubleSpinBox.setValue(float(metadata_val))
                    doubleSpinBox.isMetadataValue = True
                widget = doubleSpinBox
                defaultVal = ArgSpec.default
                valueSetter = widgets_line_edits.FloatLineEdit.setValue
                valueGetter = widgets_line_edits.FloatLineEdit.value
                widgetsLayout.addWidget(doubleSpinBox, row, 1, 1, 2)
                widget.valueChanged.connect(self.emitValuesChanged)
            elif ArgSpec.type == os.PathLike:
                filePathControl = widgets_line_edits.filePathControl()
                filePathControl.setText(str(ArgSpec.default))
                widget = filePathControl
                defaultVal = str(ArgSpec.default)
                valueSetter = widgets_line_edits.filePathControl.setText
                valueGetter = widgets_line_edits.filePathControl.path
                widgetsLayout.addWidget(filePathControl, row, 1, 1, 2)
                widget.sigValueChanged.connect(self.emitValuesChanged)
            elif isCustomListType:
                items = ArgSpec.type().values
                ArgSpec.type.cast_dtype = _types.to_str
                defaultVal = str(ArgSpec.default)
                combobox = widgets_combo_boxes.AlphaNumericComboBox()
                combobox.addItems(items)
                combobox.setCurrentValue(defaultVal)
                valueSetter = widgets_combo_boxes.AlphaNumericComboBox.setCurrentValue
                valueGetter = widgets_combo_boxes.AlphaNumericComboBox.currentValue
                widget = combobox
                widgetsLayout.addWidget(combobox, row, 1, 1, 2)
                widget.currentTextChanged.connect(self.emitValuesChanged)
            else:
                lineEdit = QLineEdit()
                lineEdit.setText(str(ArgSpec.default))
                lineEdit.setAlignment(Qt.AlignCenter)
                widget = lineEdit
                defaultVal = str(ArgSpec.default)
                valueSetter = QLineEdit.setText
                valueGetter = QLineEdit.text
                widgetsLayout.addWidget(lineEdit, row, 1, 1, 2)
                widget.editingFinished.connect(self.emitValuesChanged)
            
            if ArgSpec.desc:
                infoButton = self.getInfoButton(ArgSpec.name, ArgSpec.desc)
                widgetsLayout.addWidget(infoButton, row, 3)
            
            argsInfo = ArgWidget(
                name=ArgSpec.name,
                type=ArgSpec.type,
                widget=widget,
                defaultVal=defaultVal,
                valueSetter=valueSetter,
                valueGetter=valueGetter
            )
            ArgsWidgets_list.append(argsInfo)
        
        widgetsLayout.setColumnStretch(0, 0)
        widgetsLayout.setColumnStretch(1, 1)
        widgetsLayout.setColumnStretch(3, 0)

        return widgetsLayout, ArgsWidgets_list

    def checkIfTypeCLassHasCastDtype(self, cls):
        cast_dtype = getattr(cls, 'cast_dtype', None)
        if callable(cast_dtype):
            return
        
        raise AttributeError(
            'The custom type or widget does not have the `cast_dtype` method. '
            'Please, implement it. The method should cast the value to the '
            'correct type.'
        )
    
    def getInfoButton(self, param_name, infoText):
        infoButton = widgets_push_buttons.infoPushButton()
        infoButton.param_name = param_name
        infoButton.setToolTip(
            f'Click to get more info about `{param_name}` parameter...'
        )
        infoButton.infoText = infoText
        infoButton.clicked.connect(self.showInfoParam)
        return infoButton
    
    def showInfoParam(self):
        text = self.sender().infoText
        text = html_utils.rst_urls_to_html(text)
        text = html_utils.rst_to_html(text)
        text = html_utils.paragraph(text)
        param_name = self.sender().param_name
        msg = widgets_message_box.myMessageBox(wrapText=False)
        msg.information(self, f'Info about `{param_name}` parameter', text)


class PreProcessParamsWidget(QWidget):
    sigLoadRecipe = Signal()
    sigLoadSavedRecipe = Signal()
    sigValuesChanged = Signal(list)
    recipesPath = preproc_recipes_path
    recipePrefix = 'preprocessing_recipe'
    
    def __init__(self, df_metadata=None, addApplyButton=False, parent=None):
        super().__init__(parent)
        
        mainLayout = QVBoxLayout()
        
        self.df_metadata = df_metadata
        self.addApplyButton = addApplyButton
        
        groupbox = QGroupBox()
        self.groupbox = groupbox
        
        groupbox.setTitle('Pre-processing')
        groupbox.setCheckable(True)
        
        self.gridLayout = QGridLayout()        
        self.row = -1
        self.stepsWidgets = {}
        
        self.gridLayout.setColumnStretch(0, 0)
        self.gridLayout.setColumnStretch(1, 1)
        self.gridLayout.setColumnStretch(2, 0)
        self.gridLayout.setColumnStretch(3, 0)
        self.gridLayout.setColumnStretch(4, 0)
        groupbox.setLayout(self.gridLayout)
        
        buttonsLayout = QGridLayout()
        row = 0
        col = 0
        buttonsLayout.setColumnStretch(col, 1)
        
        loadRecipeButton = widgets_push_buttons.OpenFilePushButton('Load saved recipe...')
        self.loadRecipeButton = loadRecipeButton
        buttonsLayout.addWidget(loadRecipeButton, row, col+2)
        
        saveRecipeButton = widgets_push_buttons.savePushButton('Save current recipe...')
        self.saveRecipeButton = saveRecipeButton
        buttonsLayout.addWidget(saveRecipeButton, row+1, col+2)
        
        loadLastRecipeButton = widgets_push_buttons.reloadPushButton('Load last parameters')
        self.loadLastRecipeButton = loadLastRecipeButton
        buttonsLayout.addWidget(loadLastRecipeButton, row, col+1)
        
        self.buttonsLayout = buttonsLayout
        
        loadLastRecipeButton.clicked.connect(self.emitLoadRecipe)
        saveRecipeButton.clicked.connect(self.saveRecipe)
        loadRecipeButton.clicked.connect(self.selectAndLoadRecipe)
        
        mainLayout.addWidget(groupbox)
        mainLayout.addSpacing(10)
        mainLayout.addLayout(buttonsLayout)
        
        self.addStep(is_first=True)
        
        mainLayout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(mainLayout)
    
    def stepSizeHeightHint(self):
        stepWidgets = self.stepsWidgets[1]
        height = (
            stepWidgets['stepLabel'].minimumSizeHint().height()
            + stepWidgets['selector'].minimumSizeHint().height()
        )
        return height
    
    def setChecked(self, checked):
        self.groupbox.setChecked(checked)
    
    def emitLoadRecipe(self):
        self.sigLoadRecipe.emit()
    
    def loadRecipe(self, configPars: dict):
        for stepWidgets in list(self.stepsWidgets.values()):
            try:
                stepWidgets['delButton'].click()
            except Exception as err:
                pass
        
        configPars = self.sortStepsConfigPars(configPars)
        for s in range(1, len(configPars)):
            self.stepsWidgets[1]['addButton'].click()
        
        for i, (section, section_items) in enumerate(configPars.items()):
            step_n = i+1
            selector = self.stepsWidgets[step_n]['selector']
            kwarg_to_value_mapper = {}
            for option, value in section_items.items():
                if option == 'method':
                    selector.setCurrentText(value)
                    method = value
                else:
                    kwarg_to_value_mapper[option] = value
            selector.setParams(method, kwarg_to_value_mapper)
        
        self.setChecked(True)
    
    def sortStepsConfigPars(self, configPars: dict):
        sortedConfigPars = {}
        sortedKeys = sorted(
            configPars.keys(), 
            key=lambda key: int(re.findall(r'step(\d+)', key)[0])
        )
        for key in sortedKeys:
            sortedConfigPars[key] = configPars[key]
        return sortedConfigPars
    
    def saveRecipeUI(self, folder_path, ext, title, basename, hintText, 
                     default_text):# -> tuple[Literal[False], Literal['']] | tuple[Literal[True], Any]:        
        win = dialogs_file_selection.filenameDialog(
            title=title,
            basename=basename,
            ext=ext,
            hintText=hintText,
            allowEmpty=False,
            defaultEntry=default_text,
            parent=self,
        )
        win.exec_()
        if win.cancel:
            return False, ''
        
        self.cancel = False
        filepath = win.filename
        os.makedirs(folder_path, exist_ok=True)
        filepath = os.path.join(folder_path, filepath)
        
        if os.path.exists(filepath):
            proceed = self.warnExistingRecipeFile(filepath)
            if not proceed:
                return False, ''
        
        return True, filepath
    
    def saveRecipe(self):
        recipe = self.recipe()
        if recipe is None:
            return
        
        default_text = ''
        for step in recipe[:2]:
            method = step['method']
            selector = next(iter(self.stepsWidgets.values()))['selector']
            func_name = selector.preprocessMapper[method]['function_name']
            default_text = f'{default_text}-{func_name}'
        default_text = default_text.lstrip('-')
        
        proceed, ini_filepath = self.saveRecipeUI(self.recipesPath, '.ini',
                          'Filename for pre-processing recipe',
                          self.recipePrefix,
                          'Insert a <b>filename</b> for the pre-processing recipe:',
                          default_text
                          )
        if not proceed:
            return
        
        cp = self.recipeConfigPars('acdc')
        with open(ini_filepath, 'w') as configfile:
            cp.write(configfile)
        
        self.communicateSavingRecipeFinished(ini_filepath)
    
    def warnExistingRecipeFile(self, ini_filename):
        msg = widgets_message_box.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(
            'A file with the following name<br><br>'
            f'<code>{ini_filename}</code><br><br>'
            '<b>already exists</b>.<br><br>'
            'Do you want to <b>overwrite</b> the existing file?'
        )
        noButton, yesButton = msg.warning(
            self, 'File name existing', txt, 
            buttonsTexts=(
                'No, stop saving process', 
                'Yes, overwrite existing file'
            )
        )
        return msg.clickedButton == yesButton

    def warnNoAvailableRecipesToLoad(self):
        text = html_utils.paragraph(
            'There are no recipes saved. Sorry about that :('
        )
        msg = widgets_message_box.myMessageBox(wrapText=False)
        msg.warning(self, 'No recipes saved', text)

    def selectRecipeFilepath(self, recipes_path, recipe_prefix, ext_label, ext):
        availableRecipes = []
        displayNameToFilenameMapper = {}
        ext = ext.lstrip('.')
        if os.path.exists(recipes_path):
            for file in filesystem.listdir(recipes_path):
                if not file.startswith(recipe_prefix):
                    continue

                if not file.lower().endswith(f'.{ext.lower()}'):
                    continue

                parts = file.split(f'{recipe_prefix}_', 1)
                if len(parts) != 2:
                    continue

                endname = parts[1]
                filepath = os.path.join(recipes_path, file)
                modified_on = datetime.datetime.fromtimestamp(
                    os.path.getmtime(filepath)
                ).strftime(r'%Y/%m/%d %H:%M')
                availableRecipes.append((endname, modified_on))
                displayNameToFilenameMapper[endname] = file
        
        if not availableRecipes:
            import qtpy.compat
            filepath = qtpy.compat.getopenfilename(
                parent=self,
                caption=f'Select {ext_label} file to load recipe',
                filters=f'{ext_label} (*.{ext});;All Files (*)'
            )[0]
            return filepath or None
        
        browseButton = widgets_push_buttons.browseFileButton(
            f'Select {ext_label} file...',
            title=f'Select {ext_label} file to load recipe',
            openFolder=False,
            start_dir=filesystem.getMostRecentPath(),
            ext={ext_label: f'.{ext}'}
        )
        selectRecipeWin = dialogs_generic.QTreeDialog(
            availableRecipes,
            headerLabels=['Name', 'Date Modified'],
            title='Select recipe',
            infoText='Select recipe to load:<br>',
            parent=self,
            path_to_browse=recipes_path,
            additional_buttons=(browseButton,)
        )
        browseButton.sigPathSelected.connect(
            partial(
                self.recipeIniFileSelected,
                selectRecipeWin=selectRecipeWin,
                sender=browseButton
            )
        )
        selectRecipeWin.exec_()
        if selectRecipeWin.cancel:
            return None

        if selectRecipeWin.clickedButton == browseButton:
            return selectRecipeWin.selectedIniFilepath

        selected_endname = selectRecipeWin.selectedText
        filename = displayNameToFilenameMapper.get(selected_endname)
        if filename is None:
            return None

        return os.path.join(recipes_path, filename)
    
    def selectAndLoadRecipe(self):
        filepath = self.selectRecipeFilepath(
            self.recipesPath, self.recipePrefix, 'INI', 'ini'
        )
        if filepath is None:
            return
        cp = config.ConfigParser()
        cp.read(filepath)
        preprocConfigPars = {
            s: cp[s] for s in cp.sections() 
            if s.startswith('acdc.preprocess')
        }
        if not preprocConfigPars:
            return
        self.loadRecipe(preprocConfigPars)
    
    def recipeIniFileSelected(
            self, ini_filepath, selectRecipeWin=None, sender=None
        ):
        selectRecipeWin.clickedButton = sender
        selectRecipeWin.selectedIniFilepath = ini_filepath
        selectRecipeWin.cancel = False
        selectRecipeWin.close()

    def communicateSavingRecipeFinished(self, ini_filepath):
        text = html_utils.paragraph(
            'Done!<br><br>'
            'Pre-processing recipe saved to:'
        )
        msg = widgets_message_box.myMessageBox(wrapText=False)
        msg.information(
            self, 'Pre-processing recipe saved!', text, 
            commands=(ini_filepath,),
            path_to_browse=os.path.dirname(ini_filepath)
        )
        
    def createSelector(self):
        return widgets_combo_boxes.PreProcessingSelector()

    def addStep(self, is_first=False):
        stepWidgets = {}
        
        self.row += 1
        
        step_n = len(self.stepsWidgets)+1
        label = QLabel(f'Step {step_n}: ')
        self.gridLayout.addWidget(label, self.row, 0)
        stepWidgets['stepLabel'] = label
        
        selector = self.createSelector()
        self.gridLayout.addWidget(selector, self.row, 1)
        stepWidgets['selector'] = selector
        
        setParamsButton = widgets_push_buttons.setPushButton()
        setParamsButton.setToolTip(
            'Set step parameters'
        )
        self.gridLayout.addWidget(setParamsButton, self.row, 2)
        setParamsButton.clicked.connect(
            partial(self.setParamsStep, selector=selector)
        )
        stepWidgets['setParamsButton'] = setParamsButton
        
        infoButton = widgets_push_buttons.infoPushButton()
        self.gridLayout.addWidget(infoButton, self.row, 3)
        infoButton.clicked.connect(partial(self.showInfo, selector=selector))
        stepWidgets['infoButton'] = infoButton
        
        if is_first:
            addButton = widgets_push_buttons.addPushButton()
            self.gridLayout.addWidget(addButton, self.row, 4)
            addButton.clicked.connect(self.addStep)
            stepWidgets['addButton'] = addButton
        else:
            delButton = widgets_push_buttons.delPushButton()
            self.gridLayout.addWidget(delButton, self.row, 4)
            delButton.clicked.connect(self.removeStep)
            delButton.step_n = step_n
            stepWidgets['delButton'] = delButton
        
        self.row += 1
        selector.row = self.row
        selector.step_n = step_n

        hline = widgets_basic_widgets.QHLine()
        self.gridLayout.addWidget(hline, self.row, 0, 1, 6)
        stepWidgets['hline'] = hline
        self.row += 1
              
        self.stepsWidgets[step_n] = stepWidgets
        
        selector.sigValuesChanged.connect(self.emitValuesChanged)
        selector.currentTextChanged.connect(
            partial(self.clearInitKwargs, step_n=step_n)
        )
        
        self.resetStretch()
    
    def emitValuesChanged(self, functionKwargs, step_n):
        self.stepsWidgets[step_n]['step_kwargs'] = functionKwargs
        
        recipe = self.recipe(warn=False)
        if recipe is None:
            return
        
        self.sigValuesChanged.emit(recipe)
    
    def clearInitKwargs(self, selected_method, step_n=0):
        stepWidgets = self.stepsWidgets[step_n]
        stepWidgets.pop('step_kwargs', None)
    
    def resetStretch(self):
        for row in range(self.gridLayout.rowCount()):
            self.gridLayout.setRowStretch(row, 0)

        self.gridLayout.setRowStretch(self.gridLayout.rowCount(), 1)
        self.row = self.gridLayout.rowCount() - 1
        
    def showInfo(self, checked=False, selector=None):
        if selector is None:
            return
        
        htmlText = selector.htmlInfo()
        htmlText = html_utils.paragraph(htmlText)
        
        method = selector.currentText()
        msg = widgets_message_box.myMessageBox(wrapText=False)
        msg.information(self, f'Info about `{method}`', htmlText)
    
    def setParamsStep(
            self, checked=False, 
            selector: 'widgets_combo_boxes.PreProcessingSelector'=None
        ):
        step_n = selector.step_n
        stepFunctionKwargs = selector.askSetParams(
            df_metadata=self.df_metadata, 
            addApplyButton=self.addApplyButton
        )
        if stepFunctionKwargs is None:
            return
        
        self.stepsWidgets[step_n]['step_kwargs'] = stepFunctionKwargs
        
    def removeStep(self, checked=False, step_n=None):        
        if step_n is None:
            step_n = self.sender().step_n
        
        stepWidgets = self.stepsWidgets[step_n]
        
        stepWidgets['stepLabel'].hide()
        self.gridLayout.removeWidget(stepWidgets['stepLabel'])
        
        stepWidgets['selector'].hide()
        self.gridLayout.removeWidget(stepWidgets['selector'])
        
        stepWidgets['infoButton'].hide()
        self.gridLayout.removeWidget(stepWidgets['infoButton'])
        
        # stepWidgets['addButton'].hide()
        # self.gridLayout.removeWidget(stepWidgets['addButton'])
        
        stepWidgets['setParamsButton'].hide()
        self.gridLayout.removeWidget(stepWidgets['setParamsButton'])
        
        stepWidgets['delButton'].hide()
        self.gridLayout.removeWidget(stepWidgets['delButton'])
        self.row -= 1
        
        stepWidgets['hline'].hide()
        self.gridLayout.removeWidget(stepWidgets['hline'])
        self.row -= 1

        self.stepsWidgets.pop(step_n)
        
        stepsWidgetsMapper = {1: self.stepsWidgets[1]}
        for i, stepWidgets in enumerate(self.stepsWidgets.values()):
            if i == 0:
                continue
            step_n = i + 1
            label = stepWidgets['stepLabel']
            label.setText(f'Step {step_n}: ')
            stepWidgets['delButton'].step_n = step_n
            stepWidgets['selector'].step_n = step_n
            stepsWidgetsMapper[step_n] = stepWidgets
        
        self.stepsWidgets = stepsWidgetsMapper
        
        self.resetStretch()

    def isChecked(self):
        return self.groupbox.isChecked()
    
    def warnStepNotInit(self, method):
        msg = widgets_message_box.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(
            f'The parameters for the preprocessing step <b>{method}</b> '
            'were not initialized.<br><br>'
            'Please, click on the corresponding <code>Set step parameters</code> '
            'button to initialize this step (cog icon).<br><br>'
            'Thank you for your patience!'
        )
        msg.warning(self, 'Params not initialized!', txt)

    def recipe(self, warn=True):
        recipe = []
        if not self.groupbox.isChecked() and self.groupbox.isCheckable():
            return recipe
        
        for stepWidgets in self.stepsWidgets.values():
            method = stepWidgets['selector'].currentText()
            step_kwargs = stepWidgets.get('step_kwargs')
            if step_kwargs is None:
                if warn:
                    self.warnStepNotInit(method)
                return

            try:
                init_func = config.PREPROCESS_INIT_MAPPER[method]['function']
                init_func(**step_kwargs)
            except Exception as err:
                pass
            
            recipe.append({
                'method': method, 'kwargs': step_kwargs
            })
        
        return recipe
    
    def recipeConfigPars(self, model_name):
        cp = config.ConfigParser()
        if not self.groupbox.isChecked() and self.groupbox.isCheckable():
            return cp
        
        for s, step in enumerate(self.recipe()):
            section = f'{model_name}.preprocess.step{s+1}'
            cp[section] = {}
            cp[section]['method'] = step['method']
            for option, value in step['kwargs'].items():
                cp[section][option] = str(value)
        return cp

# class QComboBoxChangeColor(QComboBox):
#     def __init__(self, forbidden_items=None, parent=None):
#         super().__init__(parent)
#         self.forbiddenItems = forbidden_items or set()
#         self._defaultStyleSheet = self.styleSheet()
#         self.currentTextChanged.connect(self._updateColor)
    
#     def _updateColor(self, text=None):
#         if not hasattr(self, '_defaultStyleSheet'):
#             self._defaultStyleSheet = self.styleSheet()
#         if self.currentText() in self.forbiddenItems:
#             self.setStyleSheet(
#                 self._defaultStyleSheet + """
#                 /* Closed state */
#                 QComboBox {
#                     color: red;
#                 }

#                 /* Open state (popup visible) */
#                 QComboBox:on {
#                     color: white;
#                 }
#                 """
#             )
#         else:
#             self.setStyleSheet(self._defaultStyleSheet)

        
        
class CombineChannelsWidget(PreProcessParamsWidget):
    sigValuesChangedCombineChannels = Signal()
    
    def __init__(self, channel_names:Iterable[str], parent=None):
        self.channel_names = channel_names

        super().__init__(parent)

        self.parent = parent
        qutils.delete_widget(self.loadLastRecipeButton)
        qutils.delete_widget(self.saveRecipeButton)
        qutils.delete_widget(self.loadRecipeButton)

    def addStep(self, is_first=False):
        stepWidgets = {}
        
        self.row += 1
        if is_first:
            self.row += 1
        
        step_n = len(self.stepsWidgets)+1
        tooltip = (
            'Use this text in the formula'
        )
        if is_first:
            label = QLabel('Formula var')
            label.setToolTip(
                tooltip
            )
            self.gridLayout.addWidget(label, self.row-1, 1)
        name_edit = QLineEdit(text=f'img{step_n}')
        name_edit.setToolTip(
            tooltip
        )
        self.gridLayout.addWidget(name_edit, self.row, 1)
        stepWidgets['name_edit'] = name_edit
        name_edit.textChanged.connect(self.emitValuesChanged)

        tooltip = (
            'Select a channel or a segmentation mask'
        )
        if is_first:
            label = QLabel('Channel')
            label.setToolTip(
                tooltip
            )
            self.gridLayout.addWidget(label, self.row-1, 2)
        ch_selector = QComboBox()
        ch_selector.setToolTip(
            tooltip
        )
        ch_selector.addItems(self.channel_names)
        self.gridLayout.addWidget(ch_selector, self.row, 2)
        stepWidgets['selector'] = ch_selector
        ch_selector.currentTextChanged.connect(self.setBinarizeCheckableAndNorm)

        # add binarisaion spinbox
        tooltip = (
            'If binarize is selected, the channel will be binarized first, before applying offset and multiplier.\n'
            'If inverse binarize is selected, the channel will be binerized and '
            'then the logical NOT will be applied.'
        )
        if is_first:
            label = QLabel('Binarize')
            label.setToolTip(
                tooltip
            )
            self.gridLayout.addWidget(label, self.row-1, 5)
        options = ['No', 'binarize', 'inverse binarize']
        self.binarizeCombobox = QComboBox()
        self.binarizeCombobox.addItems(options)
        self.binarizeCombobox.setCurrentIndex(0)
        self.binarizeCombobox.setEnabled(False)
        self.binarizeCombobox.setToolTip(
            tooltip
        )
        self.binarizeCombobox.currentIndexChanged.connect(self.emitValuesChanged)
        self.gridLayout.addWidget(self.binarizeCombobox, self.row, 5)
        stepWidgets['binarize'] = self.binarizeCombobox
        
        tooltip = (
            'Min value of the channel to be normalized to.'
        )
        if is_first:
            label = QLabel('Min val')
            label.setToolTip(
                tooltip
            )
            self.gridLayout.addWidget(label, self.row-1, 6)
        self.minValueSpinbox = QDoubleSpinBox()
        self.minValueSpinbox.setRange(-np.inf, np.inf)
        self.minValueSpinbox.setSingleStep(0.1)
        self.minValueSpinbox.setValue(0)
        self.minValueSpinbox.setToolTip(
            tooltip
        )
        
        self.minValueSpinbox.valueChanged.connect(self.emitValuesChanged)
        self.gridLayout.addWidget(self.minValueSpinbox, self.row, 6)
        stepWidgets['minValueSpinbox'] = self.minValueSpinbox
        
        tooltip = (
            'Max value of the channel to be normalized to.'
        )
        if is_first:
            label = QLabel('Max val')
            label.setToolTip(
                tooltip
            )
            self.gridLayout.addWidget(label, self.row-1, 7)
        self.maxValueSpinbox = QDoubleSpinBox()
        self.maxValueSpinbox.setRange(-np.inf, np.inf)
        self.maxValueSpinbox.setSingleStep(0.1)
        self.maxValueSpinbox.setValue(1)
        self.maxValueSpinbox.setToolTip(
            tooltip
        )
        
        self.maxValueSpinbox.valueChanged.connect(self.emitValuesChanged)
        self.gridLayout.addWidget(self.maxValueSpinbox, self.row, 7)
        stepWidgets['maxValueSpinbox'] = self.maxValueSpinbox

        if is_first:
            addButton = widgets_push_buttons.addPushButton()
            self.gridLayout.addWidget(addButton, self.row, 8)
            addButton.clicked.connect(self.addStep)
            stepWidgets['addButton'] = addButton
            
        else:
            delButton = widgets_push_buttons.delPushButton()
            self.gridLayout.addWidget(delButton, self.row, 8)
            delButton.clicked.connect(self.removeStep)
            delButton.step_n = step_n
            stepWidgets['delButton'] = delButton
        
        self.row += 1
        ch_selector.row = self.row
        ch_selector.step_n = step_n

        hline = widgets_basic_widgets.QHLine()
        self.gridLayout.addWidget(hline, self.row, 0, 1, 8)
        stepWidgets['hline'] = hline
        self.row += 1
              
        self.stepsWidgets[step_n] = stepWidgets
                
        self.resetStretch()
        self.sigValuesChangedCombineChannels.emit()
        self.setBinarizeCheckableAndNorm()
    
    def emitValuesChanged(self, *args):
        self.sigValuesChangedCombineChannels.emit()
        
    def setBinarizeCheckableAndNorm(self):
        for step_n, stepWidgets in self.stepsWidgets.items():
            binarizeSelector = stepWidgets['binarize']
            channel = stepWidgets['selector'].currentText()
            if "segm" in channel:
                binarizeSelector.setEnabled(True)
                # set min and max to 0 and 1 and disable
                stepWidgets['minValueSpinbox'].setValue(0)
                stepWidgets['maxValueSpinbox'].setValue(1)
                stepWidgets['minValueSpinbox'].setEnabled(False)
                stepWidgets['maxValueSpinbox'].setEnabled(False)
            else:
                binarizeSelector.setEnabled(False)
                binarizeSelector.setCurrentIndex(0)
                # set min and max to 0 and 1 and enable
                stepWidgets['minValueSpinbox'].setEnabled(True)
                stepWidgets['maxValueSpinbox'].setEnabled(True)
        
        self.emitValuesChanged()

    def removeStep(self, checked=False, step_n=None):        
        if step_n is None:
            step_n = self.sender().step_n
        
        stepWidgets = self.stepsWidgets[step_n]
        
        stepWidgets['name_edit'].hide()
        self.gridLayout.removeWidget(stepWidgets['name_edit'])
        
        stepWidgets['selector'].hide()
        self.gridLayout.removeWidget(stepWidgets['selector'])
        
        stepWidgets['binarize'].hide()
        self.gridLayout.removeWidget(stepWidgets['binarize'])
        
        stepWidgets['minValueSpinbox'].hide()
        self.gridLayout.removeWidget(stepWidgets['minValueSpinbox'])
        
        stepWidgets['maxValueSpinbox'].hide()
        self.gridLayout.removeWidget(stepWidgets['maxValueSpinbox'])
        
        stepWidgets['delButton'].hide()
        self.gridLayout.removeWidget(stepWidgets['delButton'])
        
        self.row -= 1
        
        stepWidgets['hline'].hide()
        self.gridLayout.removeWidget(stepWidgets['hline'])
        self.row -= 1

        self.stepsWidgets.pop(step_n)
        
        stepsWidgetsMapper = {1: self.stepsWidgets[1]}
        for i, stepWidgets in enumerate(self.stepsWidgets.values()):
            if i == 0:
                continue
            step_n = i + 1
            stepWidgets['delButton'].step_n = step_n
            stepWidgets['selector'].step_n = step_n
            stepsWidgetsMapper[step_n] = stepWidgets
        
        self.stepsWidgets = stepsWidgetsMapper
        
        self.resetStretch()
        self.sigValuesChangedCombineChannels.emit()

    def steps(self):
        steps = {}
        if not self.groupbox.isChecked() and self.groupbox.isCheckable():
            return steps
        
        for step_number, stepWidgets in self.stepsWidgets.items():
            name = stepWidgets['name_edit'].text()
            channel = stepWidgets['selector'].currentText()
            binarize = stepWidgets['binarize'].currentText()
            min_val = stepWidgets['minValueSpinbox'].value()
            max_val = stepWidgets['maxValueSpinbox'].value()
            steps[step_number] = {
                'name': name,
                'channel': channel,
                'binarize': binarize,
                'min_val': min_val,
                'max_val': max_val,
            }

        steps = dict(sorted(steps.items()))
        return steps


class PreProcessRecipeDialog(dialogs_base.QBaseDialog):
    sigApplyImage = Signal(object)
    sigApplyZstack = Signal(object)
    sigApplyAllFrames = Signal(object)
    sigApplyAllPos = Signal(object)
    sigPreviewToggled = Signal(bool)
    sigValuesChanged = Signal(list)
    sigSavePreprocData = Signal(object)
    sigClose = Signal(object)
    paramsWidgetClass = PreProcessParamsWidget
    
    def __init__(
            self, 
            isTimelapse=False, 
            isZstack=False, 
            isMultiPos=False, 
            df_metadata=None,
            addApplyButton=False,
            parent=None,
            hideOnClosing=False,
        ):
        super().__init__(parent=parent)
        
        self.setWindowTitle('Pre-processing recipe')
        
        self.cancel = True
        self.hideOnClosing = hideOnClosing
        
        mainLayout = QVBoxLayout()
        
        keepInputDataTypeLayout = QHBoxLayout()
        self.keepInputDataTypeToggle = widgets_push_buttons.Toggle()
        self.keepInputDataTypeToggle.setChecked(True)
        self.keepInputDataTypeToggle.toggled.connect(self.emitValuesChanged)

        keepInputDataTypeLayout.addStretch(1)
        keepInputDataTypeLayout.addWidget(QLabel('Keep input data type: '))
        keepInputDataTypeLayout.addWidget(self.keepInputDataTypeToggle)
        keepInputDataTypeInfoButton = widgets_push_buttons.infoPushButton()
        keepInputDataTypeLayout.addWidget(keepInputDataTypeInfoButton)
        keepInputDataTypeInfoButton.clicked.connect(
            self.showInfoKeepInputDataType
        )
        self.keepInputDataTypeLayout = keepInputDataTypeLayout
        
        self.preProcessParamsWidget = self.paramsWidgetClass(
            df_metadata=df_metadata, 
            addApplyButton=addApplyButton, 
            parent=self
        )
        self.preProcessParamsWidget.groupbox.setCheckable(False)
        
        buttonsLayout = QGridLayout() # self.preProcessParamsWidget.buttonsLayout
        self.buttonsLayout = buttonsLayout
        self.previewCheckbox = QCheckBox('Preview')
        buttonsLayout.addWidget(self.previewCheckbox, 0, 0)
        
        # Relocate buttons of PreProcessParamsWidget to this dialog
        pPPWBL = self.preProcessParamsWidget.buttonsLayout
        loadRecipeButtIdx = pPPWBL.indexOf(
            self.preProcessParamsWidget.loadRecipeButton
        )
        self.loadRecipeButton = pPPWBL.takeAt(loadRecipeButtIdx).widget()
        buttonsLayout.addWidget(self.loadRecipeButton, 0, 1)
        
        saveRecipeButtIdx = pPPWBL.indexOf(
            self.preProcessParamsWidget.saveRecipeButton
        )
        self.saveRecipeButton = pPPWBL.takeAt(saveRecipeButtIdx).widget()
        buttonsLayout.addWidget(self.saveRecipeButton, 1, 1)
        
        loadLastRecipeButtIdx = pPPWBL.indexOf(
            self.preProcessParamsWidget.loadLastRecipeButton
        )
        self.loadLastRecipeButton = pPPWBL.takeAt(loadLastRecipeButtIdx).widget()
        buttonsLayout.addWidget(self.loadLastRecipeButton, 1, 0)
        
        self.loadLastRecipeButton.hide()
        
        # self.cancelButton = widgets_push_buttons.cancelPushButton('Cancel')
        # buttonsLayout.insertWidget(2, self.cancelButton)
        # buttonsLayout.insertSpacing(3, 20)
        
        self.allButtons = [
            self.previewCheckbox,
            self.loadRecipeButton,
            self.saveRecipeButton,
        ]
        col = 3
        row = 0
        self.applyCurrentFrameButton = widgets_push_buttons.okPushButton(
            'Apply to displayed image'
        )
        buttonsLayout.addWidget(self.applyCurrentFrameButton, row, col)
        self.applyCurrentFrameButton.clicked.connect(
            partial(self.apply, signal=self.sigApplyImage)
        )
        self.allButtons.append(self.applyCurrentFrameButton)
        
        infoLayout = QHBoxLayout()
        buttonsHeight = self.applyCurrentFrameButton.sizeHint().height()
        self.loadingCircle = widgets_basic_widgets.LoadingCircleAnimation(
            size=buttonsHeight
        )
        sp = self.loadingCircle.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        self.loadingCircle.setSizePolicy(sp)
        self.loadingCircle.setVisible(False)
        infoLayout.addWidget(self.loadingCircle)
        
        self.infoLabel = QLabel(
            "<i>(Feel free to use Cell-ACDC while waiting)</i>"
        )
        sp = self.infoLabel.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        self.infoLabel.setSizePolicy(sp)
        self.infoLabel.hide()
        infoLayout.addWidget(self.infoLabel)
        
        buttonsLayout.addLayout(
            infoLayout, row+1, 0, 3, 2, 
            alignment=Qt.AlignBottom | Qt.AlignLeft
        )

        if isZstack:
            row += 1
            self.applyAllZslicesButton = widgets_push_buttons.threeDPushButton(
                'Apply to all z-slices of current image'
            )
            buttonsLayout.addWidget(self.applyAllZslicesButton, row, col)
            self.applyAllZslicesButton.clicked.connect(self.applyAllZslices)
            self.allButtons.append(self.applyAllZslicesButton)
        if isTimelapse:
            row += 1
            self.applyAllFramesButton = widgets_push_buttons.futurePushButton(
                'Apply to all frames'
            )
            buttonsLayout.addWidget(self.applyAllFramesButton, row, col)
            self.applyAllFramesButton.clicked.connect(self.applyAllFrames)
            self.allButtons.append(self.applyAllFramesButton)
        if isMultiPos:
            row += 1
            self.applyAllPosButton = widgets_push_buttons.futurePushButton(
                'Apply to all Positions'
            )
            buttonsLayout.addWidget(self.applyAllPosButton, row, col)
            self.applyAllPosButton.clicked.connect(
                partial(self.apply, signal=self.sigApplyAllPos)
            )
            self.allButtons.append(self.applyAllPosButton)

        row += 1
        self.savePreprocButton = widgets_push_buttons.savePushButton(
            'Save pre-processed data...'
        )
        buttonsLayout.addWidget(self.savePreprocButton, row, col)

        self.allButtons.append(self.savePreprocButton)
        self.savePreprocButton.clicked.connect(self.emitSignalSavePreprocData)
        
        self.previewCheckbox.toggled.connect(self.emitSigPreviewToggled)
        self.preProcessParamsWidget.sigValuesChanged.connect(
            self.emitValuesChanged
        )
        
        # self.cancelButton.clicked.connect(self.close)
        
        mainLayout.addLayout(keepInputDataTypeLayout)
        mainLayout.addSpacing(20)
        mainLayout.addWidget(self.preProcessParamsWidget)
        mainLayout.addLayout(buttonsLayout)
        self.mainLayout = mainLayout
        
        self.setLayout(mainLayout)

    def applyAllZslices(self, checked=False):
        # Preview needs to be turned off because we are computing on every 
        # z-slice 
        self.previewCheckbox.setChecked(False)
        self.apply(signal=self.sigApplyZstack)
    
    def applyAllFrames(self, checked=False):
        # Preview needs to be turned off because we are computing on all frames
        self.previewCheckbox.setChecked(False)
        self.apply(signal=self.sigApplyAllFrames)
    
    def emitSigPreviewToggled(self):
        self.sigPreviewToggled.emit(self.previewCheckbox.isChecked())

    def showInfoKeepInputDataType(self):
        txt = html_utils.paragraph("""
            If checked, the data type of the pre-processed data will be 
            the same as the input data type.<br><br>
            This is useful to avoid saving the pre-processed data as 
            floating-point numbers (e.g., 32-bit float) which might 
            increase the file size.<br><br>
            We <b>recommend keeping this option checked</b>.
        """)
        msg = widgets_message_box.myMessageBox(wrapText=False)
        msg.information(self, 'Keep input data type', txt)

    def emitSignalSavePreprocData(self):
        self.sigSavePreprocData.emit(self)
           
    def emitValuesChanged(self):
        recipe = self.recipe(warn=False)
        if recipe is None:
            return
        
        self.sigValuesChanged.emit(recipe)
    
    def setDisabled(self, disabled: bool):
        self.preProcessParamsWidget.setDisabled(disabled)
        self.loadingCircle.setVisible(disabled)
        self.infoLabel.setVisible(disabled)
        for button in self.allButtons:
            try:
                button.setDisabled(disabled)
            except RuntimeError as e:
                printl(traceback.format_exc())
                printl(f'Error: {e}')
                printl(f'Button: {button}')
    
    def apply(self, checked=False, signal: Signal=None):
        recipe = self.recipe()
        if recipe is None:
            return
        
        if signal is not None:
            signal.emit(recipe)
            
        if self.hideOnClosing:
            self.setDisabled(True)
            self.infoLabel.setText(
                f"{self.sender().text().replace('Apply', 'Applying')}...<br>"
                "<i>(Feel free to use Cell-ACDC while waiting)</i>"
            )
        else:
            self.ok_cb()
    
    def appliedFinished(self):
        self.setDisabled(False)
    
    def recipe(self, warn=True):
        recipe = self.preProcessParamsWidget.recipe(warn=warn)
        if recipe is None:
            return
        
        for step in recipe:
            step['keep_input_data_type'] = (
                self.keepInputDataTypeToggle.isChecked()
            )
        return recipe
    
    def recipeConfigPars(self):
        return self.preProcessParamsWidget.recipeConfigPars('acdc')

    def ok_cb(self):
        if self.hideOnClosing:
            self.hide()
            return

        self.cancel = False
        self.close()
    
    def close(self):
        super().close()
        self.sigClose.emit(self)

class PreProcessRecipeDialogUtil(PreProcessRecipeDialog):
    selectionLabel = 'Select channel(s) to pre-process:'

    def __init__(
            self,
            channel_names: Iterable[str],
            df_metadata=None,
            parent=None,
        ):
        self.cancel = True
        
        super().__init__(
            isTimelapse=False, 
            isZstack=False, 
            isMultiPos=False, 
            addApplyButton=False,
            df_metadata=df_metadata,
            parent=parent,
            hideOnClosing=False
        )
                
        self.listSelector = self.createListSelector(channel_names)
        
        self.mainLayout.insertWidget(0, self.listSelector)
        self.mainLayout.insertWidget(
            0, QLabel(self.selectionLabel)
        )
        self.mainLayout.insertSpacing(2, 10)
        self.mainLayout.insertWidget(2, widgets_basic_widgets.QHLine())
        
        self.savePreprocButton.hide()
        self.previewCheckbox.hide()
        self.applyCurrentFrameButton.setText('Ok')
        
        buttonsLayout = self.preProcessParamsWidget.buttonsLayout
        
        saveRecipeButtonIndex = buttonsLayout.indexOf(
            self.preProcessParamsWidget.saveRecipeButton
        )        
        
        if saveRecipeButtonIndex == -1:
            return
        
        saveRecipeButtonItem = buttonsLayout.takeAt(saveRecipeButtonIndex)
        
        buttonsLayout.addItem(saveRecipeButtonItem, 0, 2)
    
    def createListSelector(self, names):
        selector = widgets_item_views.listWidget(
            isMultipleSelection=True, minimizeHeight=True
        )
        selector.addItems(names)
        selector.setCurrentRow(0)
        return selector

    def warnChannelSelectionEmpty(self):
        txt = html_utils.paragraph("""
            You did <b>not select any channel</b>.<br><br>
            <b>Channel selection cannot be empty</b>.<br><br>
            Thank you for your patience!
        """)
        msg = widgets_message_box.myMessageBox(wrapText=False)
        msg.warning(self, 'No channel selected', txt)
    
    def ok_cb(self):
        selectedChannelItems = self.listSelector.selectedIndexes()
        if not selectedChannelItems:
            self.warnChannelSelectionEmpty()
            return
        
        recipe = self.recipe()
        if recipe is None:
            return
        
        self.selectedRecipe = recipe
        self.selectedChannels = [item.data() for item in selectedChannelItems]
        
        self.cancel = False
        self.close()






# class ComboDelegate(QStyledItemDelegate):
#     def __init__(self, bad_values, parent=None):
#         super().__init__(parent)
#         self.bad_values = bad_values

#     def paint(self, painter, option, index):
#         text = index.data()
#         if text in self.bad_values:
#             option.palette.setColor(option.palette.Text, QColor("red"))
#         super().paint(painter, option, index)
        
class CombineChannelsSetupDialog(PreProcessRecipeDialog):
    sigApplyImage = Signal(dict, bool, str)
    sigApplyZstack = Signal(dict, bool, str)
    sigApplyAllFrames = Signal(dict, bool, str)
    sigApplyAllPos = Signal(dict, bool, str)
    sigValuesChanged = Signal()
    sigSaveAsSegmCheckboxToggled = Signal(bool)

    
    # sigApplyAllZslices = Signal(dict, bool, str)
    # sigApplyAllFramesZslices = Signal(dict, bool, str)

    def __init__(
            self,
            channel_names,
            df_metadata=None,
            parent=None,
            hideOnClosing=False,
            isTimelapse=False,
            isZstack=False,
            isMultiPos=False,
            ):
        
        self.combineChannelsWidget = CombineChannelsWidget(channel_names, parent=self)
        self.warnExistingRecipeFile = self.combineChannelsWidget.warnExistingRecipeFile
        self.communicateSavingRecipeFinished = self.combineChannelsWidget.communicateSavingRecipeFinished
        self.saveRecipeUI = self.combineChannelsWidget.saveRecipeUI
        self.selectRecipeFilepath = self.combineChannelsWidget.selectRecipeFilepath
        
        super().__init__(
            isTimelapse=isTimelapse,
            isZstack=isZstack,
            isMultiPos=isMultiPos,
            df_metadata=df_metadata,
            parent=parent,
            hideOnClosing=hideOnClosing,
        )

        self.combineChannelsWidget.sigValuesChangedCombineChannels.connect(
            self.emitValuesChangedSteps
        )
        

        self.segm_blinked = False
        self.validFormula = True # allow empty formula
        self.forbiddenChannels = set() # channels that cannot be combined

        self.mainLayout.setSpacing(4)
    
        self.mainLayout.insertWidget(2, self.combineChannelsWidget)
        self.combineChannelsWidget.groupbox.setCheckable(False)
        self.combineChannelsWidget.groupbox.setTitle('Combine and manipulate channels and/or segmentation files')
        
        self.formulaEditWidget = dialogs_combine_measurements.FormulaEditWidget(
            parent=self, validator=self._validateFormula
        )
        self._updateFormulaVariableNames()
        self.formulaEditWidget.sigFormulaChanged.connect(self.formulaChanged)
        self.formulaEditWidget.setToolTip(
            'Enter a formula to combine the channels. For example '
            '"img1 + img2 * 0.5"'
        )
        self.mainLayout.insertWidget(3, self.formulaEditWidget)
        
        buttonsLayoutSaveGroup = QGridLayout()
        
        row = 0
        col = 0
        loadRecipeButton = widgets_push_buttons.OpenFilePushButton('Load saved recipe')
        self.loadRecipeButtonComb = loadRecipeButton
        buttonsLayoutSaveGroup.addWidget(loadRecipeButton, row, col)
        self.loadRecipeButtonComb.clicked.connect(self.selectAndLoadRecipe)
        
        col += 1
        saveRecipeButton = widgets_push_buttons.savePushButton('Save current recipe')
        self.saveRecipeButtonComb = saveRecipeButton
        buttonsLayoutSaveGroup.addWidget(saveRecipeButton, row, col)
        saveRecipeButton.clicked.connect(self.saveRecipe)
        saveRecipeButton.setToolTip(
            'Save the current recipe to a file\n'
            f'Location: <b>{combine_channels_recipes_path}</b>'
        )
        
        col += 1
        loadLastRecipeButton = widgets_push_buttons.reloadPushButton('Load last recipe')
        self.loadLastRecipeButtonComb = loadLastRecipeButton
        buttonsLayoutSaveGroup.addWidget(loadLastRecipeButton, row, col)        
        self.mainLayout.addLayout(buttonsLayoutSaveGroup)
        loadLastRecipeButton.clicked.connect(self.loadLastRecipe)
        self.setLoadLastRecipe()
        
        loadLastRecipeButton.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        loadLastRecipeButton.customContextMenuRequested.connect(
            self._showLoadRecipeContextMenu
        )

        self.cancel = True

        self.setWindowTitle('Combine and manipulate channels and/or segmentation files')
        self.preProcessParamsWidget.hide()
        self.mainLayout.removeWidget(self.preProcessParamsWidget)

        self.savePreprocButton.setText('Save combined data...')
        
        
        tooltip = (
            'Save as a segmentation file, for example '
            'when combining a binary mask with a segmentation mask.'
            )
        label = QLabel('Save as segmentation:')
        self.saveAsSegmlabel = label
        label.setToolTip(tooltip)
        self.saveAsSegmCheckbox = widgets_push_buttons.Toggle()
        self.saveAsSegmCheckbox.setToolTip(tooltip)
        self.saveAsSegmCheckbox.setChecked(False)
        self.saveAsSegmCheckbox.setEnabled(False)
        self.saveAsSegmCheckbox.toggled.connect(self.emitSaveAsSegmCheckboxToggled)
        
        self.keepInputDataTypeLayout.insertWidget(0, label)
        self.keepInputDataTypeLayout.insertWidget(1, self.saveAsSegmCheckbox)

    def _validateFormula(self, formula):
        channels = {
            step['selector'].currentText()
            for step in self.combineChannelsWidget.stepsWidgets.values()
        }
        if self.forbiddenChannels.intersection(channels):
            return False, (
                'Channels that are forbidden are not allowed to be used!:\n'
                f'{self.forbiddenChannels}'
            )
        return self.formulaEditWidget.checkSymbolicValidity()

    def setLoadLastRecipe(self):
        filepath = self._lastRecipePath()
        if not os.path.exists(filepath):
            self.loadLastRecipeButtonComb.setEnabled(False)

    def returLoadSecondLastRecipe(self):
        filepath = self._secondLastRecipePath()
        if not os.path.exists(filepath):
            return False
        return True

    def _showLoadRecipeContextMenu(self, pos):
        menu = QMenu(self)
        action = menu.addAction('Load recipe from before the last one')
        action.triggered.connect(self.loadPreviousRecipe)
        action.setEnabled(self.returLoadSecondLastRecipe())
        menu.exec(self.loadLastRecipeButtonComb.mapToGlobal(pos))

    def loadPreviousRecipe(self):
        filepath = self._secondLastRecipePath()
        if not os.path.exists(filepath):
            return

        self.loadRecipe(filepath)

    def loadLastRecipe(self):
        filepath = self._lastRecipePath()
        if not os.path.exists(filepath):
            return
            
        self.loadRecipe(filepath)
        
    def saveLastRecipe(self):
        os.makedirs(combine_channels_recipes_path, exist_ok=True)
        filepath = self._lastRecipePath()

        same = False
        if os.path.exists(filepath):
            steps_curr = self._getSaveRecipyDict()
            with open(filepath, 'r') as f:
                steps_prev = json.load(f)
            same = self._recipesMatch(steps_curr, steps_prev)

        if same:
            return

        if os.path.exists(filepath):
            new_filename = self._secondLastRecipePath()
            if os.path.exists(new_filename):
                os.remove(new_filename)
            os.rename(filepath, new_filename)
        self.saveRecipe(filepath=filepath)


    def _recipesMatch(self, steps_curr, steps_prev):
        # Normalize current dict to strings for comparison with JSON-loaded dict
        def normalize(d):
            return {str(k): str(v) for k, v in d.items()}

        for raw_key in steps_curr:
            key = str(raw_key)
            if key not in steps_prev:
                return False
            if key in ('formula', 'keep_input_data_type', 'save_as_segm'):
                if str(steps_curr[raw_key]) != str(steps_prev[key]):
                    return False
            else:
                step_dict = normalize(steps_curr[raw_key])
                step_dict_prev = steps_prev[key]
                for key2, val2 in step_dict.items():
                    if key2 not in step_dict_prev:
                        return False
                    if val2 != str(step_dict_prev[key2]):
                        return False
        return True
        
    def _lastRecipePath(self):
        return os.path.join(combine_channels_recipes_path, '.last_combine_channels_recipe.json')
    
    def _secondLastRecipePath(self):
        return os.path.join(combine_channels_recipes_path, '.previous_combine_channels_recipe.json')
    
    def _getSaveRecipyDict(self):
        steps = self.combineChannelsWidget.steps() # already returns a copy
        formula = self.formulaEditWidget.text()
        steps['formula'] = formula
        steps['keep_input_data_type'] = self.keepInputDataTypeToggle.isChecked()
        steps['save_as_segm'] = self.saveAsSegmCheckbox.isChecked()
        return steps
        
    def saveRecipe(self, dummy=None, filepath=None):
        os.makedirs(combine_channels_recipes_path, exist_ok=True)
        
        filepath_provided = filepath is not None
        if not filepath_provided:
            folder_content = filesystem.listdir(combine_channels_recipes_path)
            num_recipes = len(folder_content)
            default_text = f'{num_recipes + 1}'
            proceed, filepath = self.saveRecipeUI(
                combine_channels_recipes_path, 
                '.json', 
                'Save recipe', 
                'combine_channels_recipe',
                'Insert a <b>filename</b> for the recipe:',
                default_text
            )
        
            if not proceed:
                return
            
        steps = self._getSaveRecipyDict()
        
        with open(filepath, 'w') as f:
            json.dump(steps, f, indent=2)
        
        if not filepath_provided:
            self.communicateSavingRecipeFinished(filepath)
    
    def selectAndLoadRecipe(self):
        filepath = self.selectRecipeFilepath(
            combine_channels_recipes_path, 
            'combine_channels_recipe', 'JSON', 'json'
        )
        if filepath is None:
            return

        self.loadRecipe(filepath)
        
    def loadRecipe(self, filepath):
        with open(filepath, 'r') as f:
            recipe = json.load(f)
            
        recipe = dict(sorted(recipe.items()))
        keys_used = set()
        for key, value in recipe.items():
            if key == 'formula':
                formula = value
                continue
            if key == 'keep_input_data_type':
                self.keepInputDataTypeToggle.setChecked(value)
                continue
            if key == 'save_as_segm':
                self.saveAsSegmCheckbox.setChecked(value)
                continue
            
            name = value['name']
            channel = value['channel']
            binarize = value['binarize']
            min_val = float(value['min_val'])
            max_val = float(value['max_val'])
            key = int(key)
            stepWidgetsNum = len(self.combineChannelsWidget.stepsWidgets)
            if key > stepWidgetsNum:
                self.combineChannelsWidget.addStep()
            
            stepWidgets = self.combineChannelsWidget.stepsWidgets[key]
            idx = stepWidgets['selector'].findText(channel)
            if idx == -1:
                stepWidgets['selector'].addItem(channel)
                # stepWidgets['selector'].forbiddenItems.add(channel)
                blinker = qutils.QControlBlink(
                    stepWidgets['selector'], 
                    qparent=self
                )
                blinker.start()
                stepWidgets['selector'].blinker = blinker
                self.forbiddenChannels.add(channel)
            
            stepWidgets['selector'].setCurrentText(channel)
            stepWidgets['name_edit'].setText(name)
            stepWidgets['binarize'].setCurrentText(binarize)
            stepWidgets['minValueSpinbox'].setValue(min_val)
            stepWidgets['maxValueSpinbox'].setValue(max_val)
            
            keys_used.add(key)
        
        # remove extra steps
        keys_present = set(range(1, len(self.combineChannelsWidget.stepsWidgets)+1))
        extra_keys = keys_present - keys_used
        extra_keys = list(extra_keys)
        extra_keys.sort(reverse=True)
        for key in extra_keys:
            self.combineChannelsWidget.removeStep(step_n = key) 
            # updates key dynamically so I have to rely that missing indx are always last steps

        # update formula
        self.formulaEditWidget.setText(formula)
        
        for stepWidgets in self.combineChannelsWidget.stepsWidgets.values():
            combo = stepWidgets['selector']
            # set forbidden channels red in all steps
            for i in range(combo.count()):
                item = combo.itemText(i)
                if item in self.forbiddenChannels:
                    combo.setItemData(i, QColor('red'), Qt.ForegroundRole)
        
    def _updateFormulaVariableNames(self):
        names = [
            stepWidgets['name_edit'].text()
            for stepWidgets in self.combineChannelsWidget.stepsWidgets.values()
        ]
        self.formulaEditWidget.setVariableNames(names)

    def formulaChanged(self, formula_str, is_valid):
        self.setButtonsEnabled(is_valid)
        self.validFormula = is_valid
        if is_valid:
            self.sigValuesChanged.emit()
    
    def setButtonsEnabled(self, enabled):
        for i in range(self.buttonsLayout.count()):
            item = self.buttonsLayout.itemAt(i)
            widget = item.widget()
            if widget is None:
                continue
            if isinstance(widget, QPushButton):
                label = widget.text().lower().rstrip().lstrip()
                if 'apply' in label or 'save' in label or 'ok' in label:
                    if enabled:
                        try:
                            widget.setEnabled(True)
                        except:
                            pass
                    else:
                        try:
                            widget.setDisabled(True)
                        except:
                            pass
                    
    
    def saveAsSegm(self):
        return self.saveAsSegmCheckbox.isChecked()
    
    def emitSaveAsSegmCheckboxToggled(self):
        if self.validFormula:
            self.sigSaveAsSegmCheckboxToggled.emit(self.saveAsSegm())
    
    def autoCheckSaveAsSegmCheckbox(self):
        any_not_seg = False
        for step in self.combineChannelsWidget.steps().values():
            channel = step['channel']
            if 'segm' not in channel:
                any_not_seg = True
                break
                
        if any_not_seg:
            self.saveAsSegmCheckbox.setChecked(False)
            self.saveAsSegmCheckbox.setEnabled(False)
        else:
            if not self.segm_blinked:
                self.saveAsSegmCheckbox.setEnabled(True)
                self.blinker = qutils.QControlBlink(
                    self.saveAsSegmCheckbox, 
                    qparent=self
                )
                self.blinker.start()
                self.segm_blinked = True

    def apply(self, checked=False, signal: Signal=None):
        steps = self.combineChannelsWidget.steps()
        formula = self.formulaEditWidget.text()
        keep_input_dtype = self.keepInputDataTypeToggle.isChecked()
        if not steps or not self.validFormula:
            return
        
        if signal is not None:
            try:
                signal.emit(steps, formula)
            except TypeError as err:
                signal.emit(steps, keep_input_dtype, formula)
            

        self.saveLastRecipe()
        if self.hideOnClosing:
            self.setDisabled(True)
            self.infoLabel.setText(
                f"{self.sender().text().replace('Apply', 'Applying')}...<br>"
                "<i>(Feel free to use Cell-ACDC while waiting)</i>"
            )
        else:
            self.ok_cb(saveLastRecipe=False)
    # Not needed anymore since now we funnel all changes to the formulaEditWidget, which then verifies the formula and
    # emits a signal via  formulaChangeda
    # def emitValuesChanged(self):
    #     if not self.validFormula:
    #         return
    #     self.sigValuesChanged.emit()
        
    def emitValuesChangedSteps(self):
        self.autoCheckSaveAsSegmCheckbox()
        self._updateFormulaVariableNames()

    def ok_cb(self, dummy=None, saveLastRecipe=True):
        if not self.validFormula:
            return
        
        if saveLastRecipe:
            self.saveLastRecipe()
            
        self.keepInputDataType = self.keepInputDataTypeToggle.isChecked()
        self.selectedSteps = self.combineChannelsWidget.steps()
        self.formula = self.formulaEditWidget.text()
        self.cancel = False
        self.close()
