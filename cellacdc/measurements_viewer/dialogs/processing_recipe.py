"""Recipe dialog, parameter widget and function selector of measurement
processing in the measurements viewer (plugins:
measurement_processing/).
"""
from cellacdc.shared.widgets import combo_boxes as widgets_combo_boxes
import configparser
import html
import os
import pandas as pd
from qtpy.QtWidgets import QAbstractItemView, QLabel, QLineEdit
from cellacdc import html_utils
from cellacdc.myutils import segm_models
from cellacdc import settings_folderpath
from cellacdc.shared.widgets import (
    line_edits as widgets_line_edits,
    message_box as widgets_message_box,
    search_widgets as widgets_search_widgets,
)
from qtpy.QtWidgets import QTreeView
from .. import widgets as viewer_widgets
from cellacdc.shared.dialogs.preprocessing import (
    PreProcessParamsWidget,
    PreProcessRecipeDialogUtil,
)


class SignalProcessingSelector(widgets_combo_boxes.PreProcessingSelector):
    from cellacdc.measurement_processing import MEASUREMENT_PROCESSING_MAPPER as preprocessMapper


class MeasurementProcessingParamsWidget(PreProcessParamsWidget):
    recipesPath = os.path.join(settings_folderpath, 'measurement_processing_recipes')
    recipePrefix = 'measurement_processing_recipe'

    def createSelector(self):
        return SignalProcessingSelector()

    @staticmethod
    def castRecipeValue(param, value):
        """Convert a recipe value (e.g. a string read from a saved INI
        recipe) to the type of the ``run()`` parameter ``param``.

        Booleans are parsed with ``configparser.ConfigParser.BOOLEAN_STATES``
        because ``bool('False')`` is True. Raises ``ValueError`` for values
        that cannot be converted.
        """
        if not isinstance(value, str):
            return param.type(value)
        if param.type is bool:
            try:
                return configparser.ConfigParser.BOOLEAN_STATES[
                    value.strip().lower()
                ]
            except KeyError:
                raise ValueError(
                    f'Invalid boolean value {value!r} for "{param.name}".'
                ) from None
        if hasattr(param.type, 'cast_dtype'):
            return param.type.cast_dtype(value)
        return param.type(value)

    def recipe(self, warn=True):
        recipe = []
        for step in self.stepsWidgets.values():
            selector = step['selector']
            method = selector.currentText()
            function = selector.preprocessMapper[method]['function']
            saved = selector.methodToDefaultValuesMapper.get(method, {})
            kwargs = {
                param.name: self.castRecipeValue(
                    param, saved.get(param.name, param.default)
                )
                for param in segm_models.get_function_argspec(function)
            }
            kwargs.update(step.get('step_kwargs', {}))
            recipe.append({'method': method, 'kwargs': kwargs})
        return recipe

    def loadRecipe(self, configPars):
        from cellacdc.measurement_processing import MEASUREMENT_PROCESSING_MAPPER

        methods = [section.get('method') for section in configPars.values()]
        if any(method not in MEASUREMENT_PROCESSING_MAPPER for method in methods):
            msg = widgets_message_box.myMessageBox(wrapText=False)
            msg.warning(
                self, 'Invalid measurement recipe',
                'This recipe contains unsupported measurement-processing steps.'
            )
            return
        # Saved INI recipes store every value as a string: convert them to
        # the run() parameter types so that both recipe() and the parameters
        # dialog (askSetParams) get typed values (e.g. 'False' -> False).
        typedConfigPars = {}
        try:
            for section, section_items in configPars.items():
                method = section_items.get('method')
                function = MEASUREMENT_PROCESSING_MAPPER[method]['function']
                params = {
                    param.name: param
                    for param in segm_models.get_function_argspec(function)
                }
                typedItems = {}
                for option, value in section_items.items():
                    param = params.get(option)
                    if option != 'method' and param is not None:
                        value = self.castRecipeValue(param, value)
                    typedItems[option] = value
                typedConfigPars[section] = typedItems
        except (TypeError, ValueError) as err:
            msg = widgets_message_box.myMessageBox(wrapText=False)
            msg.warning(
                self, 'Invalid measurement recipe',
                'This recipe contains invalid parameter values:<br><br>'
                f'{html.escape(str(err))}'
            )
            return
        super().loadRecipe(typedConfigPars)
        for step in self.stepsWidgets.values():
            step.pop('step_kwargs', None)


class MeasurementProcessingRecipeDialog(PreProcessRecipeDialogUtil):
    """Select measurement leaves by channel/type, with recursive fuzzy search."""
    paramsWidgetClass = MeasurementProcessingParamsWidget
    selectionLabel = 'Select measurement(s) to process:'
    emptySuffixMessage = 'Enter a suffix for the output column(s).'

    def __init__(
            self, measurement_names, parent=None,
            acdc_df=None, allChNames=()
        ):
        self._measurementDataframe = acdc_df
        self._measurementChannels = allChNames
        super().__init__(measurement_names, parent=parent)
        self.measurement_names = measurement_names
        self.measurementsSearchLineEdit = QLineEdit()
        self.measurementsSearchLineEdit.setPlaceholderText('Search measurement...')
        self.measurementsSearchLineEdit.setClearButtonEnabled(True)
        self.measurementsSearchLineEdit.textChanged.connect(self.filterMeasurements)
        self.mainLayout.insertWidget(
            self.mainLayout.indexOf(self.listSelector),
            self.measurementsSearchLineEdit
        )
        self.setWindowTitle('Measurement-processing recipe')
        self.preProcessParamsWidget.groupbox.setTitle('Measurement processing')
        self.keepInputDataTypeToggle.setChecked(False)

        self.append_new_colname = widgets_line_edits.alphaNumericLineEdit(parent=self)
        self.append_new_colname.setPlaceholderText('Append to column for saving...')
        self.mainLayout.insertWidget(
            self.mainLayout.indexOf(self.keepInputDataTypeLayout),
            self.append_new_colname
        )
        self.append_new_colname_tip = QLabel('')
        self.append_new_colname.textChanged.connect(
            self.append_new_colname_changed
        )
        self.listSelector.selectionModel().selectionChanged.connect(
            lambda: self.append_new_colname_changed(self.append_new_colname.text())
        )
        self.append_new_colname_changed(self.append_new_colname.text())
        self.mainLayout.insertWidget(
            self.mainLayout.indexOf(self.append_new_colname) + 1,
            self.append_new_colname_tip
        )

        for i in range(self.keepInputDataTypeLayout.count()):
            widget = self.keepInputDataTypeLayout.itemAt(i).widget()
            if widget is not None:
                widget.hide()
                
    def get_new_colnames(self):
        """Return ``(warn, display, full_list)`` for the output column names.

        ``warn`` is True when the names cannot be used: no suffix was
        entered for the selected measurement(s) (outputs would overwrite
        the inputs) or an output column already exists. ``display`` is
        then the message to show to the user.
        """
        warn, display = False, ''
        text = self.append_new_colname.text()
        
        selected_channels = [index.data() for index in self.listSelector.selectedIndexes()]
        if text and selected_channels:
            full_list = [f"{selected_channel}_{text}" for selected_channel in selected_channels]
            if len(selected_channels) > 1:
                diplay_channel_part = ', '.join(selected_channels)
                display = f'"{diplay_channel_part}"_{text}'
            else:
                display = f'{selected_channels[0]}_{text}'
        elif selected_channels:
            full_list = selected_channels
            return True, self.emptySuffixMessage, full_list
        elif text:
            full_list = [text]
            display = text
        else:
            full_list = []
            display = ""
                    
        forbidden_names = [file_name for file_name in full_list if file_name in self.measurement_names]
        if forbidden_names:
            display = (
                'Output column(s) already exist: '
                + ', '.join(forbidden_names) + '. Choose another suffix.'
            )
            warn = True
            return warn, display, full_list

        warn = False
        return warn, display, full_list

    def append_new_colname_changed(self, text):
        warn, tip, _ = self.get_new_colnames()
        self.append_new_colname_tip.setText(tip)
        # A missing suffix is a neutral hint, not an error
        if warn and tip != self.emptySuffixMessage:
            self.append_new_colname_tip.setStyleSheet("color: red;")
        else:
            self.append_new_colname_tip.setStyleSheet("")

    def createListSelector(self, names):
        names = list(names)
        dataframe = self._measurementDataframe
        if dataframe is None:
            dataframe = pd.DataFrame(columns=names, dtype=float)
        self.measurementsModel = viewer_widgets.MeasurementTreeModel(self)
        self.measurementsModel.populate(dataframe[names], self._measurementChannels)
        self.measurementsSearchProxy = widgets_search_widgets.MeasurementTreeSearchProxy(self)
        self.measurementsSearchProxy.setSourceModel(self.measurementsModel)
        selector = QTreeView()
        selector.setHeaderHidden(True)
        selector.setUniformRowHeights(True)
        selector.setModel(self.measurementsSearchProxy)
        selector.setEditTriggers(QAbstractItemView.NoEditTriggers)
        selector.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.measurementsHighlightDelegate = widgets_search_widgets._ButtonSearchHighlightDelegate(
            selector
        )
        selector.setItemDelegate(self.measurementsHighlightDelegate)
        selector.expandAll()
        first = next(self.measurementsModel.measurementItems(), None)
        if first is not None:
            selector.setCurrentIndex(
                self.measurementsSearchProxy.mapFromSource(first.index())
            )
        return selector

    def filterMeasurements(self, text):
        """Filter and highlight measurements with the viewer's fuzzy search."""
        self.measurementsSearchProxy.setFilterText(text)
        self.measurementsHighlightDelegate.setFilterText(text)
        if text.strip():
            self.listSelector.expandAll()
        self.listSelector.viewport().update()

    def warnChannelSelectionEmpty(self):
        msg = widgets_message_box.myMessageBox(wrapText=False)
        msg.warning(
            self, 'No measurement selected', html_utils.paragraph(
                'You did <b>not select any measurement</b>.<br><br>'
                'Select at least one measurement to process.'
            )
        )

    def ok_cb(self):
        warn, display, full_list = self.get_new_colnames()
        if warn:
            msg = widgets_message_box.myMessageBox(wrapText=False)
            msg.warning(self, 'Invalid output column name', display)
            return
        self.saveNames = full_list
        super().ok_cb()
